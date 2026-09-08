"""Compare P2 detailed benchmark JSON against historical throughput/texts."""
import argparse
import json
from pathlib import Path


def compare(current, baseline, concurrency):
    historic = {1: (0.28, 83.66), 20: (2.83, 145.31)}[concurrency]
    throughput = current["request_throughput"]
    result = {
        "concurrency": concurrency,
        "completed": current.get("completed"),
        "failed": current.get("failed"),
        "total_output_tokens": current.get("total_output_tokens"),
        "historical_request_throughput": historic[0],
        "request_throughput": throughput,
        "throughput_change_percent": (throughput / historic[0] - 1) * 100,
        "historical_mean_tpot_ms": historic[1],
        "mean_tpot_ms": current.get("mean_tpot_ms"),
        "performance_target_met": throughput >= (0.266 if concurrency == 1 else 3.40),
        "transcription_equivalence": "unverified: no baseline detailed texts",
        "wer": "not measured",
    }
    if baseline is not None:
        old, new = baseline.get("generated_texts"), current.get("generated_texts")
        if old and new and len(old) == len(new):
            mismatches = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
            result["transcription_equivalence"] = "same-order comparison; verify identical input manifest"
            result["text_mismatch_indices"] = mismatches
            result["all_texts_equal"] = not mismatches
        else:
            result["transcription_equivalence"] = "unverified: missing texts or different request counts"
    # Performance success alone must never be reported as P2 acceptance.
    result["promotion_accepted"] = False
    result["remaining_evidence"] = "verify input identity, runtime hit/gather counters, and MatMul policy"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--concurrency", type=int, choices=(1, 20), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    current = json.loads(args.result.read_text())
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    result = compare(current, baseline, args.concurrency)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
