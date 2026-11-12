# SPDX-License-Identifier: Apache-2.0
"""Parse three text sections once after generation, without token grammar."""

from __future__ import annotations

import json
from typing import Any

import regex as re

_CANONICAL = ("SUMMARY", "DETAILED", "TAGS")
_ALIASES = {
    "SUMMARY": ("SUMMARY", "摘要", "概述", "简述", "简要描述"),
    "DETAILED": ("DETAILED", "DETAIL", "DETAILS", "详细", "详细描述", "完整描述", "细节描述"),
    "TAGS": ("TAGS", "TAG", "标签", "主体标签", "主题标签"),
}


def strip_code_fence(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|text|markdown|md)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _normalize_text(text: str) -> str:
    text = strip_code_fence(text)
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    text = text.replace("：", ":")
    return text.strip()


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def unwrap_provider_payload(payload: Any, _depth: int = 0) -> Any:
    if _depth > 8:
        return payload

    if isinstance(payload, dict):
        if any(field_from_mapping(payload, name) is not None for name in _CANONICAL):
            return payload

        for key in ("raw_text", "text", "output_text", "content"):
            if key in payload and payload.get(key) not in (None, ""):
                return unwrap_provider_payload(payload.get(key), _depth + 1)

        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and message.get("content") is not None:
                    return unwrap_provider_payload(message.get("content"), _depth + 1)
                if first.get("text") is not None:
                    return unwrap_provider_payload(first.get("text"), _depth + 1)
        return payload

    if isinstance(payload, list):
        if len(payload) == 1:
            return unwrap_provider_payload(payload[0], _depth + 1)
        return payload

    if not isinstance(payload, str):
        return payload

    raw = strip_code_fence(payload)
    parsed = _try_json(raw)
    if isinstance(parsed, (dict, list, str)):
        return unwrap_provider_payload(parsed, _depth + 1)
    return _normalize_text(raw)


def _norm_key(key: Any) -> str:
    text = str(key or "").strip().replace("：", ":").strip("*#` ")
    return re.sub(r"[\s_:\-]+", "", text).casefold()


def field_from_mapping(obj: dict[str, Any], name: str) -> Any:
    aliases = _ALIASES.get(name, (name,))
    alias_keys = {_norm_key(x) for x in aliases}
    for key, value in obj.items():
        if _norm_key(key) in alias_keys:
            return value
    return None


def _label_pattern(name: str) -> str:
    aliases = _ALIASES.get(name, (name,))
    parts = [re.escape(x) for x in aliases]
    return "(?:" + "|".join(parts) + ")"


def extract_labeled_field(text: str, name: str, next_names: tuple[str, ...]) -> str:
    text = _normalize_text(text)
    current = _label_pattern(name)
    next_patterns = "|".join(_label_pattern(x) for x in next_names)

    lead = r"(?:^|[\r\n]|(?<=[。！？.!?；;])|(?<=\s))\s*(?:[-*#>]\s*)*(?:\*\*|__)?\s*"
    suffix = r"\s*(?:\*\*|__)?\s*:\s*"

    if next_patterns:
        next_label = lead + rf"(?:{next_patterns})" + suffix
        pattern = rf"(?ims){lead}{current}{suffix}(.*?)(?={next_label}|\Z)"
    else:
        pattern = rf"(?ims){lead}{current}{suffix}(.*?)\Z"

    match = re.search(pattern, text)
    return " ".join(match.group(1).split()).strip() if match else ""


def tags_from_value(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        text = _normalize_text(value)
        if not text:
            return []

        parsed = _try_json(text)
        if isinstance(parsed, (list, dict)):
            value = parsed
        else:
            parts = re.split(r"\s*[|｜]\s*", text)
            if len(parts) == 1 and any(x in text for x in (",", "，", ";", "；")):
                parts = re.split(r"\s*[,，;；]\s*", text)
            value = parts

    if isinstance(value, dict):
        if any(k in value for k in ("tag", "name", "label")):
            value = [value]
        else:
            value = list(value.values())

    if not isinstance(value, (list, tuple)):
        value = [value]

    output, seen = [], set()
    for item in value:
        if isinstance(item, dict):
            tag = str(item.get("tag") or item.get("name") or item.get("label") or item.get("标签") or "").strip()
        else:
            tag = str(item or "").strip()

        tag = " ".join(tag.split()).strip(" |｜,，;；")
        key = tag.casefold()
        if not tag or key in seen:
            continue

        seen.add(key)
        output.append(tag)
        if len(output) >= 8:
            break

    return output


def parse_direct_guided_text(raw):
    value = unwrap_provider_payload(raw)
    if isinstance(value, dict):
        return {
            "summary": field_from_mapping(value, "SUMMARY") or "",
            "tags": tags_from_value(field_from_mapping(value, "TAGS")),
            "detailed": field_from_mapping(value, "DETAILED") or "",
        }
    text = str(value or "")
    fields = {}
    for name in ("SUMMARY", "TAGS", "DETAILED"):
        following = tuple(other for other in ("SUMMARY", "TAGS", "DETAILED") if other != name)
        fields[name.lower()] = extract_labeled_field(text, name, following)
    fields["tags"] = tags_from_value(fields["tags"])
    return fields
