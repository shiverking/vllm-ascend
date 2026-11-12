# SPDX-License-Identifier: Apache-2.0
"""Process a single image or directory without importing cloud-drive-search."""

import argparse
import asyncio
import json
from pathlib import Path

from .client import DirectGuidedClient, prepare_image
from .recovery import Recovery


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:1025")
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--concurrency", type=int, default=320)
    parser.add_argument("--recovery-concurrency", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("direct-guided-output/results.jsonl"))
    parser.add_argument("--state-dir", type=Path, default=Path("direct-guided-output/state"))
    args = parser.parse_args()
    if args.concurrency < 1 or args.recovery_concurrency < 1:
        parser.error("Concurrency must be positive")
    return args


async def run(args):
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
    images = (
        [args.images]
        if args.images.is_file()
        else sorted(path for path in args.images.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
    )
    if not images:
        raise ValueError("No supported images found: " + str(args.images))
    # Some uploaded HEIF files carry a .JPG suffix; register by content support.
    try:
        from pillow_heif import register_heif_opener
    except ImportError:
        if any(path.suffix.lower() in {".heic", ".heif"} for path in images):
            raise ValueError("Install pillow-heif to process HEIC/HEIF input") from None
    else:
        register_heif_opener()
    client = DirectGuidedClient(args.base_url, args.model, args.concurrency + args.recovery_concurrency)
    recovery = Recovery(client, args.state_dir, args.concurrency, args.recovery_concurrency)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    queue = asyncio.Queue()
    for path in images:
        queue.put_nowait(path)
    failures = 0
    try:
        _, image_url = await asyncio.to_thread(prepare_image, images[0])
        await client.register(image_url)
        with args.output.open("a") as output:

            async def worker():
                nonlocal failures
                while not queue.empty():
                    path = queue.get_nowait()
                    try:
                        image_hash, image_url = await asyncio.to_thread(prepare_image, path)
                        result, checkpoint = await recovery.generate(path, image_hash, image_url)
                        row = {
                            "image": str(path),
                            "status": "succeeded",
                            "result": result,
                            "checkpoint": str(checkpoint),
                        }
                    except Exception as error:
                        failures += 1
                        row = {"image": str(path), "status": "needs_review", "error": str(error)}
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    queue.task_done()

            tasks = [
                asyncio.create_task(worker())
                for _ in range(min(len(images), args.concurrency + args.recovery_concurrency))
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await client.close()
    print(f"完成 {len(images)} 张；成功 {len(images) - failures}；需复核 {failures}；结果 {args.output}")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(arguments())))
