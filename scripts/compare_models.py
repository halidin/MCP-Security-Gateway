from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Avoid transformers probing its TensorFlow/Keras backend (not needed; we use PyTorch).
os.environ.setdefault("USE_TF", "0")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from interceptor.model import DriftClassifier
from interceptor.features import combine_goal_and_trace

# ── Settings ──────────────────────────────────────────────────────────────────
TRAIN_TRACES   = Path("data/traces/train.jsonl")
TEST_TRACES    = Path("data/traces/test.jsonl")
OUT_DIR        = Path("models")
TFIDF_MODELS   = ["logistic", "svm", "random_forest", "gradient_boosting"]
SENTENCE_MODEL = "all-MiniLM-L6-v2"
# ─────────────────────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_sentence_transformer(train, test, train_labels, test_labels) -> tuple[float, float, float, float]:
    from sentence_transformers import SentenceTransformer

    print(f"  {'sentence-transformer':<22}", end=" ", flush=True)
    t0 = time.time()

    model = SentenceTransformer(SENTENCE_MODEL)
    train_texts = [combine_goal_and_trace(r["user_goal"], r["agent_trace"]) for r in train]
    test_texts  = [combine_goal_and_trace(r["user_goal"], r["agent_trace"]) for r in test]

    train_emb = model.encode(train_texts, batch_size=64, show_progress_bar=False)
    test_emb  = model.encode(test_texts,  batch_size=64, show_progress_bar=False)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_emb, train_labels)
    preds = clf.predict(test_emb)

    elapsed   = time.time() - t0
    precision = float(precision_score(test_labels, preds, zero_division=0))
    recall    = float(recall_score(test_labels, preds, zero_division=0))
    f1        = float(f1_score(test_labels, preds, zero_division=0))
    return precision, recall, f1, elapsed


def main() -> None:
    train = load_jsonl(TRAIN_TRACES)
    test  = load_jsonl(TEST_TRACES)

    train_goals  = [r["user_goal"]   for r in train]
    train_traces = [r["agent_trace"] for r in train]
    train_labels = [int(r["label"])  for r in train]
    test_goals   = [r["user_goal"]   for r in test]
    test_traces  = [r["agent_trace"] for r in test]
    test_labels  = [int(r["label"])  for r in test]

    print(f"\n{'Model':<24} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time':>8}")
    print("-" * 68)

    results = []

    # ── TF-IDF based models ───────────────────────────────────────────────────
    for model_type in TFIDF_MODELS:
        clf = DriftClassifier(model_type=model_type)
        t0 = time.time()
        clf.fit(train_goals, train_traces, train_labels)
        elapsed = time.time() - t0
        result = clf.evaluate(test_goals, test_traces, test_labels)
        clf.save(str(OUT_DIR / f"{model_type}.joblib"))
        label = f"tfidf + {model_type}"
        print(f"  {label:<22} {result.precision:>9.1%} {result.recall:>9.1%} {result.f1:>9.1%} {elapsed:>6.1f}s")
        results.append((label, result.precision, result.recall, result.f1))

    # ── Sentence Transformer ──────────────────────────────────────────────────
    precision, recall, f1, elapsed = run_sentence_transformer(
        train, test, train_labels, test_labels
    )
    label = f"sentence-transformer"
    print(f"{precision:>9.1%} {recall:>9.1%} {f1:>9.1%} {elapsed:>6.1f}s")
    results.append((label, precision, recall, f1))

    # ── Summary ───────────────────────────────────────────────────────────────
    best = max(results, key=lambda x: x[3])
    print(f"\n  Best F1: {best[0]}  ({best[3]:.1%})\n")


if __name__ == "__main__":
    main()
