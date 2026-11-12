# SPDX-License-Identifier: Apache-2.0
"""Bounded attempts with durable task checkpoints and exclusive task leases."""

import asyncio
import fcntl
import hashlib
import json
import os
import time

import httpx

from .client import PREPROCESS, PROMPT_NAMES, ContractError, validate_content


def atomic_save(path, task):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        json.dump(task, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class Recovery:
    def __init__(self, client, state_dir, concurrency, recovery_concurrency):
        self.client = client
        self.root = state_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.primary = asyncio.Semaphore(concurrency)
        self.recovery = asyncio.Semaphore(recovery_concurrency)
        self.total = asyncio.Semaphore(concurrency + recovery_concurrency)

    async def generate(self, image_path, image_hash, image_url):
        identity = [
            image_hash,
            self.client.model,
            self.client.base_url,
            self.client.prompt_fingerprint,
            "temperature=0;thinking=false;context=16384",
            PREPROCESS,
        ]
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.root / (key + ".json")
        with (self.root / (key + ".lock")).open("a") as lease:
            while True:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.1)
            try:
                return await self._generate(path, key, identity, image_path, image_url)
            finally:
                fcntl.flock(lease, fcntl.LOCK_UN)

    async def _generate(self, path, key, identity, image_path, image_url):
        task = (
            json.loads(path.read_text())
            if path.exists()
            else {
                "key": key,
                "identity": identity,
                "image_path": str(image_path),
                "created_at": time.time(),
                "attempts": [],
                "steps": {},
                "state": "pending",
            }
        )

        def save(state):
            task.update(state=state, updated_at=time.time())
            atomic_save(path, task)

        async def call(prompt, budget=1):
            if prompt in task["steps"]:
                return validate_content(task["steps"][prompt])
            failed = sum(attempt["prompt"] == prompt and attempt["status"] == "failed" for attempt in task["attempts"])
            if failed >= budget:
                raise ContractError("Recovery budget exhausted: " + prompt)
            for _ in range(failed, budget):
                attempt = {"prompt": prompt, "status": "queued"}
                task["attempts"].append(attempt)
                save("running" if prompt == PROMPT_NAMES[0] else "recovery_pending")
                gate = self.primary if prompt == PROMPT_NAMES[0] else self.recovery
                try:
                    async with gate, self.total:
                        attempt["status"] = "running"
                        save(task["state"])
                        value = validate_content(await self.client.generate(prompt, image_url))
                        attempt["status"] = "succeeded"
                        task["steps"][prompt] = value
                        save(task["state"])
                        return value
                except (ContractError, httpx.HTTPError) as error:
                    attempt.update(status="failed", error=str(error))
                    save("recovery_pending")
            raise ContractError("Recovery step failed: " + prompt)

        if task["state"] == "succeeded":
            return validate_content(task["result"]), path
        if task["state"] == "needs_review":
            raise ContractError("Task needs review; inspect " + str(path))
        for attempt in task["attempts"]:
            if attempt["status"] in ("queued", "running"):
                attempt["status"] = "interrupted"
        try:
            try:
                result = await call(PROMPT_NAMES[0])
            except ContractError:
                save("recovery_pending")
                try:
                    result = await call(PROMPT_NAMES[1])
                except ContractError:
                    parts = [await call(prompt, budget=2) for prompt in PROMPT_NAMES[2:]]
                    result = {
                        "summary": parts[0]["summary"],
                        "tags": parts[0]["tags"],
                        "detailed": "\n".join(dict.fromkeys(part["detailed"] for part in parts)),
                    }
            task["result"] = validate_content(result)
            save("succeeded")
            return result, path
        except asyncio.CancelledError:
            save("recovery_pending")
            raise
        except Exception:
            save("needs_review")
            raise
