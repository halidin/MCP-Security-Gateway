from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from interceptor.model import DriftClassifier

# ── Settings ──────────────────────────────────────────────────────────────────
TEST_TRACES  = Path("data/traces/test.jsonl")
TRAIN_TRACES = Path("data/traces/train.jsonl")
MODEL_PATH   = Path("models/logistic.joblib")
THRESHOLD    = 0.5
# ─────────────────────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def section(title: str) -> None:
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def main() -> None:
    test  = load_jsonl(TEST_TRACES)
    train = load_jsonl(TRAIN_TRACES)
    clf   = DriftClassifier.load(str(MODEL_PATH))

    mal_test = [r for r in test  if r["label"] == 1]
    ben_test = [r for r in test  if r["label"] == 0]

    # ── 1. Dataset overview ───────────────────────────────────────────────────
    section("1. Dataset Overview")
    print(f"  Train samples : {len(train):,}  (mal: {sum(1 for r in train if r['label']==1):,} | ben: {sum(1 for r in train if r['label']==0):,})")
    print(f"  Test  samples : {len(test):,}   (mal: {len(mal_test):,} | ben: {len(ben_test):,})")

    # ── 2. Classifier performance ─────────────────────────────────────────────
    section("2. Classifier Performance  (threshold = {:.1f})".format(THRESHOLD))
    result = clf.evaluate(
        goals  = [r["user_goal"]   for r in test],
        traces = [r["agent_trace"] for r in test],
        labels = [int(r["label"])  for r in test],
    )
    print(f"  Precision : {result.precision:.4f}  ({result.precision*100:.1f}%)")
    print(f"  Recall    : {result.recall:.4f}  ({result.recall*100:.1f}%)")
    print(f"  F1        : {result.f1:.4f}  ({result.f1*100:.1f}%)")

    # ── 3. LLM self-reported decisions ────────────────────────────────────────
    section("3. LLM Self-Reported Decisions  (final_decision field)")
    mal_dec = Counter(r["structured_reasoning"].get("final_decision", "?") for r in mal_test)
    ben_dec = Counter(r["structured_reasoning"].get("final_decision", "?") for r in ben_test)
    print("  Malicious samples:")
    for k, v in mal_dec.most_common():
        print(f"    {k:12s}: {v:4d}  ({v/len(mal_test)*100:.1f}%)")
    print("  Benign samples:")
    for k, v in ben_dec.most_common():
        print(f"    {k:12s}: {v:4d}  ({v/len(ben_test)*100:.1f}%)")

    # ── 4. Classifier vs LLM self-guard comparison ────────────────────────────
    section("4. Classifier vs LLM Self-Guard  (on malicious samples)")
    llm_blocked = mal_dec.get("block", 0)
    clf_detected = sum(
        1 for r in mal_test
        if clf.predict_proba([r["user_goal"]], [r["agent_trace"]])[0] >= THRESHOLD
    )
    print(f"  LLM self-blocked  : {llm_blocked:4d} / {len(mal_test)}  ({llm_blocked/len(mal_test)*100:.1f}%)")
    print(f"  Classifier caught : {clf_detected:4d} / {len(mal_test)}  ({clf_detected/len(mal_test)*100:.1f}%)")
    print(f"  Improvement       : +{(clf_detected - llm_blocked)/len(mal_test)*100:.1f}%")

    # ── 5. Goal alignment distribution ───────────────────────────────────────
    section("5. Goal Alignment Distribution  (LLM reasoning)")
    mal_align = Counter(r["structured_reasoning"].get("goal_alignment", "?") for r in mal_test)
    ben_align = Counter(r["structured_reasoning"].get("goal_alignment", "?") for r in ben_test)
    print("  Malicious samples:")
    for k, v in mal_align.most_common():
        print(f"    {k:12s}: {v:4d}  ({v/len(mal_test)*100:.1f}%)")
    print("  Benign samples:")
    for k, v in ben_align.most_common():
        print(f"    {k:12s}: {v:4d}  ({v/len(ben_test)*100:.1f}%)")

    # ── 6. Risk signals detected ──────────────────────────────────────────────
    section("6. Risk Signals  (LLM-flagged suspicious content)")
    mal_with_risk = sum(1 for r in mal_test if r["structured_reasoning"].get("risk_signals"))
    ben_with_risk = sum(1 for r in ben_test if r["structured_reasoning"].get("risk_signals"))
    print(f"  Malicious with risk signals : {mal_with_risk} / {len(mal_test)}  ({mal_with_risk/len(mal_test)*100:.1f}%)")
    print(f"  Benign with risk signals    : {ben_with_risk} / {len(ben_test)}  ({ben_with_risk/len(ben_test)*100:.1f}%)")

    # ── 7. Avg reasoning steps ────────────────────────────────────────────────
    section("7. Avg Reasoning Steps per Sample")
    mal_steps = [r.get("num_reasoning_steps", 0) for r in mal_test]
    ben_steps = [r.get("num_reasoning_steps", 0) for r in ben_test]
    print(f"  Malicious : {sum(mal_steps)/len(mal_steps):.2f} steps avg")
    print(f"  Benign    : {sum(ben_steps)/len(ben_steps):.2f} steps avg")

    print("\n" + "="*55)
    print("  Done.")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
