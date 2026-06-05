"""Analyze reasoning steps from LLM-generated traces."""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze reasoning steps in LLM-generated traces.")
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL with LLM traces.")
    parser.add_argument("--output_report", type=Path, help="Save analysis report to JSON.")
    args = parser.parse_args()

    samples = load_jsonl(args.input)

    if not samples:
        print("No samples found.")
        return

    # Extract reasoning step counts
    step_counts = []
    benign_steps = []
    malicious_steps = []

    for sample in samples:
        num_steps = sample.get("num_reasoning_steps", 0)
        step_counts.append(num_steps)

        label = sample.get("label")
        if label == 0:
            benign_steps.append(num_steps)
        elif label == 1:
            malicious_steps.append(num_steps)

    # Compute statistics
    report = {
        "total_samples": len(samples),
        "reasoning_steps": {
            "mean": float(np.mean(step_counts)) if step_counts else 0,
            "median": float(np.median(step_counts)) if step_counts else 0,
            "std": float(np.std(step_counts)) if step_counts else 0,
            "min": int(np.min(step_counts)) if step_counts else 0,
            "max": int(np.max(step_counts)) if step_counts else 0,
        },
        "by_label": {
            "benign": {
                "count": len(benign_steps),
                "mean_steps": float(np.mean(benign_steps)) if benign_steps else 0,
                "median_steps": float(np.median(benign_steps)) if benign_steps else 0,
            },
            "malicious": {
                "count": len(malicious_steps),
                "mean_steps": float(np.mean(malicious_steps)) if malicious_steps else 0,
                "median_steps": float(np.median(malicious_steps)) if malicious_steps else 0,
            },
        },
    }

    print("=== Reasoning Steps Analysis ===")
    print(f"Total samples: {report['total_samples']}")
    print(f"Mean reasoning steps: {report['reasoning_steps']['mean']:.2f}")
    print(f"Median reasoning steps: {report['reasoning_steps']['median']:.1f}")
    print(f"Std dev: {report['reasoning_steps']['std']:.2f}")
    print(f"Range: {report['reasoning_steps']['min']} - {report['reasoning_steps']['max']}")
    print()
    print("Benign samples:")
    print(f"  Count: {report['by_label']['benign']['count']}")
    print(f"  Mean steps: {report['by_label']['benign']['mean_steps']:.2f}")
    print(f"  Median steps: {report['by_label']['benign']['median_steps']:.1f}")
    print()
    print("Malicious samples:")
    print(f"  Count: {report['by_label']['malicious']['count']}")
    print(f"  Mean steps: {report['by_label']['malicious']['mean_steps']:.2f}")
    print(f"  Median steps: {report['by_label']['malicious']['median_steps']:.1f}")

    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        with args.output_report.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {args.output_report}")


if __name__ == "__main__":
    main()
