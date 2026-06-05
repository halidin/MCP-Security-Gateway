from __future__ import annotations

import argparse
import json
from pathlib import Path

from interceptor.features import parse_trace_steps
from interceptor.io import load_jsonl
from interceptor.model import DriftClassifier


def detection_horizon_for_sample(clf: DriftClassifier, row: dict, threshold: float) -> int | None:
    label = int(row.get("label", 0))
    if label == 0:
        return None

    steps = row.get("trace_steps")
    if not isinstance(steps, list) or not steps:
        steps = parse_trace_steps(row.get("agent_trace", ""))

    if not steps:
        return None

    tool_call_step = row.get("tool_call_step")
    if tool_call_step is None:
        tool_call_step = len(steps)

    tool_call_step = int(tool_call_step)

    first_alert_step = None
    for i in range(1, len(steps) + 1):
        prefix = " ".join(steps[:i])
        score = clf.predict_proba([row["user_goal"]], [prefix])[0]
        if score >= threshold:
            first_alert_step = i
            break

    if first_alert_step is None:
        return None

    return max(0, tool_call_step - first_alert_step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate detection horizon on test data.")
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    clf = DriftClassifier.load(str(args.model_path))
    rows = load_jsonl(args.test)

    malicious = [r for r in rows if int(r.get("label", 0)) == 1]
    horizons = []
    for row in malicious:
        h = detection_horizon_for_sample(clf, row, args.threshold)
        if h is not None:
            horizons.append(h)

    report = {
        "samples_total": len(rows),
        "samples_malicious": len(malicious),
        "detected_malicious": len(horizons),
        "avg_detection_horizon_steps": (sum(horizons) / len(horizons)) if horizons else 0.0,
        "threshold": args.threshold,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
