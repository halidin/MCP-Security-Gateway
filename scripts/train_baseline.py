from __future__ import annotations

import argparse
import json
from pathlib import Path

from interceptor.io import load_jsonl
from interceptor.model import DriftClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline IPI detector.")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--model", choices=["random_forest", "logistic"], default="random_forest")
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    train_rows = load_jsonl(args.train)
    test_rows = load_jsonl(args.test)

    clf = DriftClassifier(model_type=args.model)
    clf.fit(
        goals=[r["user_goal"] for r in train_rows],
        traces=[r["agent_trace"] for r in train_rows],
        labels=[int(r["label"]) for r in train_rows],
    )

    result = clf.evaluate(
        goals=[r["user_goal"] for r in test_rows],
        traces=[r["agent_trace"] for r in test_rows],
        labels=[int(r["label"]) for r in test_rows],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / f"{args.model}.joblib"
    clf.save(str(model_path))

    metrics = {
        "model": args.model,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "test_samples": len(test_rows),
    }
    metrics_path = args.out_dir / f"{args.model}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
