#!/usr/bin/env python3
"""Build a deterministic vLLM custom_audio JSONL dataset from local files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUPPORTED_SUFFIXES = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all files")
    args = parser.parse_args()

    audio_dir = args.audio_dir.expanduser().resolve()
    if not audio_dir.is_dir():
        parser.error(f"audio directory does not exist: {audio_dir}")
    if args.limit < 0:
        parser.error("--limit must be >= 0")

    audio_files = sorted(
        path.resolve()
        for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if args.limit:
        audio_files = audio_files[: args.limit]
    if not audio_files:
        parser.error(f"no supported audio files found below: {audio_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for path in audio_files:
            output.write(
                json.dumps(
                    {"prompt": "Transcribe the audio.", "audio": str(path)},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote {len(audio_files)} audio samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
