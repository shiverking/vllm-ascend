# SPDX-License-Identifier: Apache-2.0
"""Text-only requests; fixed-prefix transport is used only for the main step."""

import asyncio
import base64
import copy
import hashlib
import io
import json
from pathlib import Path

import httpx
from PIL import Image, ImageOps

from .parser import parse_direct_guided_text
from .resize import smart_resize

PROMPT_NAMES = (
    "caption_tags_direct_guided_v25",
    "caption_tags_direct_guided_v25_recovery",
    *(f"caption_tags_direct_guided_v25_part{i}" for i in range(4)),
)
PREPROCESS = {
    "max_edge": 4096,
    "normalize_quality": 92,
    "factor": 32,
    "min_tokens": 4,
    "max_tokens": 2048,
    "jpeg_quality": 95,
}


class ContractError(ValueError):
    """An attempt failed; bounded recovery may use a different prompt."""


def validate_content(value):
    if not isinstance(value, dict) or set(value) != {"summary", "tags", "detailed"}:
        raise ContractError("Invalid direct_guided fields")
    for field in ("summary", "detailed"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ContractError("Empty direct_guided " + field)
    tags = value["tags"]
    if (
        not isinstance(tags, list)
        or not 3 <= len(tags) <= 8
        or any(not isinstance(tag, str) or not tag.strip() for tag in tags)
    ):
        raise ContractError("Invalid direct_guided tags")
    return value


def prepare_image(path):
    """Same EXIF/thumbnail/JPEG92 then smart-resize/JPEG95 sequence as V25."""
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.thumbnail((4096, 4096))
        normalized = io.BytesIO()
        image.save(normalized, format="JPEG", quality=92)
    normalized.seek(0)
    with Image.open(normalized) as opened:
        image = opened.convert("RGB")
        height, width = smart_resize(
            image.height, image.width, factor=32, min_pixels=4 * 32**2, max_pixels=2048 * 32**2
        )
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BICUBIC)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
    url = "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()
    return hashlib.sha256(raw).hexdigest(), url


def template_fingerprint(request):
    value = copy.deepcopy(request)
    for key in ("cache_salt", "max_tokens", "temperature", "seed"):
        value.pop(key, None)
    for message in value["messages"]:
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if part.get("type") == "image_url":
                    part["image_url"] = {"url": "<fixed-prefix-image>"}
                    part.pop("uuid", None)
                    part.pop("image_ref", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class DirectGuidedClient:
    def __init__(self, base_url, model, concurrency):
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.model = model
        self.http = httpx.AsyncClient(
            timeout=600, limits=httpx.Limits(max_connections=concurrency + 8, max_keepalive_connections=concurrency + 8)
        )
        root = Path(__file__).with_name("prompts")
        self.prompts = {name: json.loads((root / f"{name}.json").read_text()) for name in PROMPT_NAMES}
        self.prompt_fingerprint = hashlib.sha256(
            json.dumps(self.prompts, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        self.prefix_lock = asyncio.Lock()
        self.prefix_id = None
        self.template_sha256 = None

    def request(self, prompt_name, image_url):
        prompt = self.prompts[prompt_name]
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt["instruction"] + "\n输入证据（分段文本）：\n"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "max_tokens": prompt["max_tokens"],
        }

    async def post(self, path, body):
        response = await self.http.post(self.base_url + path, json=body)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as error:
            raise ContractError("Invalid JSON HTTP envelope") from error

    async def register(self, image_url):
        request = self.request(PROMPT_NAMES[0], image_url)
        registration = await self.post("/v1/fixed-prefixes", request)
        if registration.get("state") != "ready":
            raise ContractError("Prefix did not become ready")
        self.prefix_id = registration["prefix_id"]
        self.template_sha256 = template_fingerprint(request)
        return registration

    async def generate(self, prompt_name, image_url):
        request = self.request(prompt_name, image_url)
        if prompt_name == PROMPT_NAMES[0]:
            if not self.prefix_id or template_fingerprint(request) != self.template_sha256:
                raise ContractError("Register the current template before generation")
            prefix_id = self.prefix_id
            payload = {
                "prefix_id": prefix_id,
                "image_url": {"url": image_url},
                "max_tokens": request["max_tokens"],
                "temperature": 0,
            }
            try:
                response = await self.post("/v1/fixed-prefix/completions", payload)
            except httpx.HTTPStatusError as error:
                if error.response.status_code != 404:
                    raise
                # A missing prefix means no generation took place. Register once
                # after a restart; concurrent callers reuse that new registration.
                async with self.prefix_lock:
                    if self.prefix_id == prefix_id:
                        await self.register(image_url)
                payload["prefix_id"] = self.prefix_id
                response = await self.post("/v1/fixed-prefix/completions", payload)
        else:
            # A recovery is an ordinary text request, without grammar or JSON repair.
            budget = await self.post(
                "/tokenize",
                {
                    "model": self.model,
                    "messages": request["messages"],
                    "chat_template_kwargs": {"enable_thinking": False},
                    "add_generation_prompt": True,
                },
            )
            if budget["count"] + request["max_tokens"] > budget["max_model_len"]:
                raise ContractError("direct_guided context budget exceeded")
            response = await self.post("/v1/chat/completions", request)
        try:
            choice = response["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ContractError("Incomplete output: " + str(choice["finish_reason"]))
            return validate_content(parse_direct_guided_text(choice["message"]["content"]))
        except (KeyError, IndexError, TypeError) as error:
            raise ContractError("Invalid completion response") from error

    async def close(self):
        try:
            if self.prefix_id:
                response = await self.http.delete(self.base_url + "/v1/fixed-prefixes/" + self.prefix_id)
                if response.status_code != 404:
                    response.raise_for_status()
        finally:
            await self.http.aclose()
