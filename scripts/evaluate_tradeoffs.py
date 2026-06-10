"""Evaluate model performance tradeoffs and detection horizon distributions across thresholds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from interceptor.features import parse_trace_steps
from interceptor.io import load_jsonl
from interceptor.model import DriftClassifier


def calculate_metrics(y_true: list[int], y_scores: list[float], threshold: float) -> dict[str, float]:
    y_pred = [1 if score >= threshold else 0 for score in y_scores]
    
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    # Calculate False Positive Rate (FPR)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detection tradeoffs and horizon distributions.")
    parser.add_argument("--model_path", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    # Load model and dataset
    clf = DriftClassifier.load(str(args.model_path))
    rows = load_jsonl(args.test)

    if not rows:
        print("Empty test dataset.")
        return

    y_true = [int(r.get("label", 0)) for r in rows]
    goals = [r.get("user_goal", "") for r in rows]
    traces = [r.get("agent_trace", "") for r in rows]

    print("Pre-computing complete trace scores...")
    # Pre-compute scores for overall classification metrics
    y_scores = clf.predict_proba(goals, traces)

    print("Pre-computing prefix scores for malicious traces...")
    # Filter malicious rows and gather all their step prefixes for batch evaluation
    malicious_indices = [idx for idx, label in enumerate(y_true) if label == 1]
    
    prefixes_to_evaluate: list[str] = []
    prefix_mapping: list[tuple[int, int]] = [] # list of (malicious_row_index, step_idx_in_row)
    sample_steps: dict[int, list[str]] = {}
    sample_tool_call: dict[int, int] = {}

    for idx in malicious_indices:
        row = rows[idx]
        steps = row.get("trace_steps")
        if not isinstance(steps, list) or not steps:
            steps = parse_trace_steps(row.get("agent_trace", ""))
        
        sample_steps[idx] = steps
        
        tool_call = row.get("tool_call_step")
        if tool_call is None:
            tool_call = len(steps)
        sample_tool_call[idx] = int(tool_call)

        for step_i in range(1, len(steps) + 1):
            prefix = " ".join(steps[:step_i])
            prefixes_to_evaluate.append(prefix)
            prefix_mapping.append((idx, step_i - 1))

    # Batch evaluate all step prefixes
    prefix_goals = [rows[idx]["user_goal"] for idx, _ in prefix_mapping]
    prefix_scores = clf.predict_proba(prefix_goals, prefixes_to_evaluate)

    # Reconstruct prefix scores mapped by malicious row index
    sample_prefix_scores: dict[int, list[float]] = {idx: [] for idx in malicious_indices}
    for (idx, _), score in zip(prefix_mapping, prefix_scores):
        sample_prefix_scores[idx].append(score)

    # Threshold grid search from 0.1 to 0.9
    thresholds = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]
    results: list[dict[str, Any]] = []

    for t in thresholds:
        metrics = calculate_metrics(y_true, y_scores, t)
        
        # Calculate horizons for this threshold
        horizons: list[int] = []
        for idx in malicious_indices:
            scores = sample_prefix_scores[idx]
            tool_call_step = sample_tool_call[idx]
            
            first_alert_step = None
            for step_i, score in enumerate(scores, start=1):
                if score >= t:
                    first_alert_step = step_i
                    break
            
            if first_alert_step is not None:
                horizons.append(max(0, tool_call_step - first_alert_step))

        # Compute horizon statistics
        if horizons:
            horizon_mean = float(np.mean(horizons))
            horizon_median = float(np.median(horizons))
            horizon_std = float(np.std(horizons))
            horizon_min = int(np.min(horizons))
            horizon_max = int(np.max(horizons))
            
            # Horizon value distribution histogram/counts
            unique_h, counts_h = np.unique(horizons, return_counts=True)
            horizon_dist = {int(k): int(v) for k, v in zip(unique_h, counts_h)}
        else:
            horizon_mean = 0.0
            horizon_median = 0.0
            horizon_std = 0.0
            horizon_min = 0
            horizon_max = 0
            horizon_dist = {}

        results.append({
            "threshold": t,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "fpr": metrics["fpr"],
            "detected_malicious": len(horizons),
            "total_malicious": len(malicious_indices),
            "horizon_mean": horizon_mean,
            "horizon_median": horizon_median,
            "horizon_std": horizon_std,
            "horizon_min": horizon_min,
            "horizon_max": horizon_max,
            "horizon_distribution": horizon_dist,
        })

    # Save results to JSON
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_name = f"tradeoffs_{args.model_path.stem}.json"
    report_path = args.out_dir / report_name
    
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved trade-off metrics to {report_path}")
    print("\n" + "="*80)
    print(f" PERFORMANCE & HORIZON TRADEOFFS: {args.model_path.stem.upper()}")
    print("="*80)
    print(f"{'Threshold':<10} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'FPR':<6} | {'Horizon (Mean/Med)':<20} | {'Detected':<8}")
    print("-"*80)
    
    # Print all evaluated thresholds for summary
    for res in results:
        horizon_str = f"{res['horizon_mean']:.2f} / {res['horizon_median']:.1f}"
        detected_str = f"{res['detected_malicious']}/{res['total_malicious']}"
        print(f"{res['threshold']:<10.2f} | {res['precision']:<6.3f} | {res['recall']:<6.3f} | {res['f1']:<6.3f} | {res['fpr']:<6.3f} | {horizon_str:<20} | {detected_str:<8}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
