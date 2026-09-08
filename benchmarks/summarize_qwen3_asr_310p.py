#!/usr/bin/env python3
"""Summarize repeated Qwen3-ASR serving runs by their median result."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


FILE_RE = re.compile(r"^(?P<config>.+)-c(?P<concurrency>\d+)-r(?P<run>\d+)\.json$")
METRICS = (
    "request_throughput",
    "output_throughput",
    "rtfx",
    "median_e2el_ms",
    "p90_e2el_ms",
    "p99_e2el_ms",
    "median_ttft_ms",
    "p90_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "p90_tpot_ms",
    "p99_tpot_ms",
)


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--expected-runs", type=int, default=3)
    args = parser.parse_args()

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(args.result_dir.glob("*-c*-r*.json")):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        with path.open(encoding="utf-8") as source:
            result = json.load(source)
        grouped[(match["config"], int(match["concurrency"]))].append(result)

    if not grouped:
        parser.error(f"no benchmark result JSON files found in {args.result_dir}")

    rows: list[dict[str, Any]] = []
    incomplete = False
    for (config, concurrency), results in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        row: dict[str, Any] = {
            "configuration": config,
            "client_concurrency": concurrency,
            "runs": len(results),
            "completed_median": statistics.median(
                float(result.get("completed", 0)) for result in results
            ),
            "failed_total": sum(int(result.get("failed", 0)) for result in results),
        }
        if len(results) != args.expected_runs:
            incomplete = True
        for metric in METRICS:
            values = [
                value
                for result in results
                if (value := numeric(result.get(metric))) is not None
            ]
            row[metric] = statistics.median(values) if values else ""
        rows.append(row)

    output = args.result_dir / "summary.csv"
    with output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    headers = (
        "configuration",
        "client_concurrency",
        "runs",
        "failed_total",
        "request_throughput",
        "rtfx",
        "p50_e2el_ms",
        "p99_e2el_ms",
    )
    print("\t".join(headers))
    for row in rows:
        display = dict(row)
        display["p50_e2el_ms"] = display["median_e2el_ms"]
        print("\t".join(str(display[name]) for name in headers))
    print(f"Full median summary: {output}")
    if incomplete:
        print(f"WARNING: at least one group does not contain {args.expected_runs} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
