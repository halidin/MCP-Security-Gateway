"""Generate all poster charts — metrics computed live from data."""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from interceptor.model import DriftClassifier
from interceptor.features import combine_goal_and_trace

# ── Settings ──────────────────────────────────────────────────────────────────
TRAIN_TRACES   = Path("data/traces/train.jsonl")
TEST_TRACES    = Path("data/traces/test.jsonl")
OUT            = Path("reports/plots")
THRESHOLD      = 0.5
TFIDF_MODELS   = ["logistic", "svm", "random_forest", "gradient_boosting"]
SENTENCE_MODEL = "all-MiniLM-L6-v2"
# ─────────────────────────────────────────────────────────────────────────────

BG    = "#0f1629"
BLUE  = "#64b5f6"
RED   = "#ef5350"
GRN   = "#66bb6a"
ORG   = "#ffa726"
GREY  = "#aaaaaa"
WHITE = "white"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def save_fig(stem: str) -> None:
    for ext in ("png", "svg"):
        out = OUT / f"{stem}.{ext}"
        kw = dict(bbox_inches="tight", facecolor=BG)
        if ext == "png":
            kw["dpi"] = 160
        plt.savefig(out, **kw)
        print(f"  Saved -> {out}")


def compute_metrics(train: list[dict], test: list[dict]) -> dict:
    """Train all TF-IDF models and return metrics dict."""
    train_goals  = [r["user_goal"]   for r in train]
    train_traces = [r["agent_trace"] for r in train]
    train_labels = [int(r["label"])  for r in train]
    test_goals   = [r["user_goal"]   for r in test]
    test_traces  = [r["agent_trace"] for r in test]
    test_labels  = [int(r["label"])  for r in test]

    results = {}
    for model_type in TFIDF_MODELS:
        print(f"  Training {model_type}...", end=" ", flush=True)
        clf = DriftClassifier(model_type=model_type)
        t0 = time.time()
        clf.fit(train_goals, train_traces, train_labels)
        elapsed = time.time() - t0
        ev = clf.evaluate(test_goals, test_traces, test_labels)
        results[model_type] = {
            "precision": ev.precision,
            "recall":    ev.recall,
            "f1":        ev.f1,
            "time":      elapsed,
            "clf":       clf,
        }
        print(f"F1={ev.f1:.3f}  ({elapsed:.1f}s)")

    return results, test_goals, test_traces, test_labels


def compute_sentence_transformer_metrics(train: list[dict], test: list[dict]) -> dict:
    """Train a Logistic Regression head on sentence-transformer embeddings."""
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score

    train_texts = [combine_goal_and_trace(r["user_goal"], r["agent_trace"]) for r in train]
    test_texts  = [combine_goal_and_trace(r["user_goal"], r["agent_trace"]) for r in test]
    train_labels = [int(r["label"]) for r in train]
    test_labels  = [int(r["label"]) for r in test]

    print(f"  Training sentence-transformer ({SENTENCE_MODEL})...", end=" ", flush=True)
    t0 = time.time()
    model = SentenceTransformer(SENTENCE_MODEL)
    train_emb = model.encode(train_texts, batch_size=64, show_progress_bar=False)
    test_emb  = model.encode(test_texts,  batch_size=64, show_progress_bar=False)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_emb, train_labels)
    preds = clf.predict(test_emb)
    elapsed = time.time() - t0

    result = {
        "precision": float(precision_score(test_labels, preds, zero_division=0)),
        "recall":    float(recall_score(test_labels, preds, zero_division=0)),
        "f1":        float(f1_score(test_labels, preds, zero_division=0)),
        "time":      elapsed,
        "clf":       None,
    }
    print(f"F1={result['f1']:.3f}  ({elapsed:.1f}s)")
    return result


def llm_vs_classifier_stats(test: list[dict], scores: list[float]) -> tuple[float, float]:
    mal = [(r, s) for r, s in zip(test, scores) if r["label"] == 1]
    llm_blocked = sum(
        1 for r, _ in mal
        if r["structured_reasoning"].get("final_decision", "") == "block"
    )
    clf_caught = sum(1 for _, s in mal if s >= THRESHOLD)
    return llm_blocked / len(mal) * 100, clf_caught / len(mal) * 100


# ── Chart 1: LLM Self-Guard vs Classifier ────────────────────────────────────
def plot_llm_vs_classifier(llm_pct: float, clf_pct: float) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    categories = ["LLM Self-Guard", "MCP Interceptor"]
    values = [llm_pct, clf_pct]
    colors = [RED, GRN]

    bars = ax.bar(categories, values, color=colors, width=0.45, zorder=3,
                  edgecolor="#222", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", color=WHITE,
                fontsize=18, fontweight="bold")

    improvement = clf_pct - llm_pct
    ax.text(0.5, 0.55, f"+{improvement:.1f} pp\nimprovement",
            transform=ax.transAxes, ha="center", va="center",
            color=GRN, fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0a2010", edgecolor=GRN, alpha=0.85))

    ax.set_ylim(0, 100)
    ax.set_ylabel("Attack Detection Rate (%)", color=GREY, fontsize=11)
    ax.set_title("LLM Self-Guard vs External Classifier\non Malicious Test Samples",
                 color=WHITE, fontsize=10, fontweight="bold", pad=12)
    ax.axhline(y=llm_pct, color=RED, linewidth=0.8, linestyle="--", alpha=0.4, zorder=2)
    ax.axhline(y=clf_pct, color=GRN, linewidth=0.8, linestyle="--", alpha=0.4, zorder=2)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], color=GREY)
    ax.tick_params(axis="x", colors=WHITE, labelsize=10)
    ax.grid(axis="y", color="#1e2a3a", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#333")

    plt.tight_layout()
    save_fig("chart_llm_vs_classifier")
    plt.close()


# ── Chart 2: Model Comparison — F1 horizontal bar ────────────────────────────
def plot_model_comparison(metrics: dict) -> None:
    labels     = ["TF-IDF + Logistic", "TF-IDF + Random Forest", "TF-IDF + Grad. Boost", "TF-IDF + SVM"]
    keys       = ["logistic", "random_forest", "gradient_boosting", "svm"]
    f1_vals    = [metrics[k]["f1"] * 100 for k in keys]
    times      = [metrics[k]["time"] for k in keys]
    bar_colors = [GRN if i == 0 else BLUE for i in range(len(keys))]

    # sort by F1 descending
    order = sorted(range(len(keys)), key=lambda i: -f1_vals[i])
    labels     = [labels[i]     for i in order]
    f1_vals    = [f1_vals[i]    for i in order]
    times      = [times[i]      for i in order]
    bar_colors = [GRN if i == 0 else BLUE for i in range(len(order))]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bars = ax.barh(y, f1_vals, color=bar_colors, height=0.55, zorder=3,
                   edgecolor="#222", linewidth=0.8)

    for bar, val, t in zip(bars, f1_vals, times):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%  ({t:.1f}s)", va="center", color=WHITE, fontsize=9.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=WHITE, fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_xlabel("F1 Score (%)", color=GREY, fontsize=10)
    ax.set_title("Model Comparison — F1 Score & Training Time",
                 color=WHITE, fontsize=12, fontweight="bold", pad=12)
    ax.axvline(x=f1_vals[0], color=GRN, linewidth=1, linestyle="--", alpha=0.5, zorder=2)
    ax.set_xticks(range(0, 101, 10))
    ax.set_xticklabels([f"{v}%" for v in range(0, 101, 10)], color=GREY, fontsize=8)
    ax.grid(axis="x", color="#1e2a3a", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#333")

    best_patch  = mpatches.Patch(color=GRN,  label="Best model")
    other_patch = mpatches.Patch(color=BLUE, label="Other TF-IDF variants")
    ax.legend(handles=[best_patch, other_patch],
              facecolor="#1a1f3a", edgecolor="#333", labelcolor=WHITE, fontsize=8.5, loc="lower right")

    plt.tight_layout()
    save_fig("chart_model_comparison")
    plt.close()


# ── Chart 3: Precision / Recall / F1 grouped bar (top 3 by F1) ───────────────
def plot_prf_breakdown(metrics: dict) -> None:
    sorted_keys = sorted(TFIDF_MODELS, key=lambda k: -metrics[k]["f1"])
    top_keys    = sorted_keys[:3]
    top_names   = {"logistic": "Logistic", "random_forest": "Random Forest",
                   "gradient_boosting": "Grad. Boost", "svm": "SVM"}
    names = [top_names[k] for k in top_keys]
    p_vals = [metrics[k]["precision"] * 100 for k in top_keys]
    r_vals = [metrics[k]["recall"]    * 100 for k in top_keys]
    f_vals = [metrics[k]["f1"]        * 100 for k in top_keys]

    x = np.arange(len(names))
    w = 0.22

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    b1 = ax.bar(x - w, p_vals, w, label="Precision", color="#3c1ad4", zorder=3, edgecolor="#222")
    b2 = ax.bar(x,     r_vals, w, label="Recall",    color="#629bdb",  zorder=3, edgecolor="#222")
    b3 = ax.bar(x + w, f_vals, w, label="F1",        color="#99a3bf",  zorder=3, edgecolor="#222")

    for group in [b1, b2, b3]:
        for bar in group:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", va="bottom",
                    color=WHITE, fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, color=WHITE, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)", color=GREY, fontsize=10)
    ax.set_title("Precision / Recall / F1 — Top 3 TF-IDF Models",
                 color=WHITE, fontsize=12, fontweight="bold", pad=12)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], color=GREY)
    ax.tick_params(axis="x", colors=WHITE)
    ax.grid(axis="y", color="#1e2a3a", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.legend(facecolor="#1a1f3a", edgecolor="#333", labelcolor=WHITE, fontsize=9)

    plt.tight_layout()
    save_fig("chart_prf_breakdown")
    plt.close()


# ── Chart 4: All models — PRF portrait horizontal bars ───────────────────────
def plot_all_models_prf(metrics: dict) -> None:
    tfidf_keys = [k for k in TFIDF_MODELS if k in metrics]
    keys       = sorted(tfidf_keys, key=lambda k: -metrics[k]["f1"])
    if "sentence_transformer" in metrics:
        keys.append("sentence_transformer")
    display    = {"logistic": "Logistic", "random_forest": "Random Forest",
                  "gradient_boosting": "Grad. Boost", "svm": "SVM",
                  "sentence_transformer": "Sentence Transformer - LR"}
    bar_labels = ["Precision", "Recall", "F1"]
    bar_colors = ["#3c1ad4", "#629bdb", "#FFFFFF"]

    n_models  = len(keys)
    n_metrics = 3
    group_h   = 0.22          # height of one bar
    gap       = 0.18          # gap between models
    total_h   = n_models * (n_metrics * group_h + gap)

    fig, ax = plt.subplots(figsize=(7, max(6, total_h + 1.2)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    yticks, yticklabels = [], []

    for i, key in enumerate(keys):
        base_y = (n_models - 1 - i) * (n_metrics * group_h + gap)
        vals = [
            metrics[key]["precision"] * 100,
            metrics[key]["recall"]    * 100,
            metrics[key]["f1"]        * 100,
        ]
        for j, (val, color, lbl) in enumerate(zip(vals, bar_colors, bar_labels)):
            y = base_y + j * group_h
            ax.barh(y, val, height=group_h * 0.82, color=color,
                    zorder=3, edgecolor="#222", linewidth=0.6,
                    label=lbl if i == 0 else None)
            ax.text(val + 0.5, y, f"{val:.1f}%", va="center",
                    color=WHITE, fontsize=8.5)

        mid_y = base_y + group_h
        yticks.append(mid_y)
        if key == "sentence_transformer":
            yticklabels.append(f"Sentence\nTransformer\n- LR\n({SENTENCE_MODEL})")
        else:
            yticklabels.append(display[key])

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, color=WHITE, fontsize=10.5)
    ax.set_xlim(0, 140)
    ax.set_xlabel("Score (%)", color=GREY, fontsize=10)
    ax.set_title("All Models — Precision / Recall / F1",
                 color=WHITE, fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels([f"{v}%" for v in range(0, 101, 20)], color=GREY, fontsize=8)
    ax.grid(axis="x", color="#1e2a3a", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.tick_params(axis="y", colors=WHITE)

    p_patch = mpatches.Patch(color="#3c1ad4", label="Precision")
    r_patch = mpatches.Patch(color="#629bdb",  label="Recall")
    f_patch = mpatches.Patch(color="#FFFFFF",  label="F1")
    ax.legend(handles=[p_patch, r_patch, f_patch],
              facecolor="#1a1f3a", edgecolor="#333", labelcolor=WHITE,
              fontsize=9, loc="upper right")

    plt.tight_layout()
    save_fig("chart_all_models_prf")
    plt.close()


# ── Chart 5: Threshold trade-off ─────────────────────────────────────────────
def plot_threshold_tradeoff(test: list[dict], scores: list[float]) -> None:
    from sklearn.metrics import f1_score, recall_score

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]

    y_true = [int(r["label"]) for r in test]

    tpr, fpr, f1 = [], [], []
    for t in thresholds:
        y_pred = [1 if s >= t else 0 for s in scores]
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        tpr.append(recall_score(y_true, y_pred, zero_division=0) * 100)
        fpr.append((fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0)
        f1.append(f1_score(y_true, y_pred, zero_division=0) * 100)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot(thresholds, tpr, color=GRN,  linewidth=2.2, marker="o", markersize=6, label="TP - Attacks Caught")
    ax.plot(thresholds, fpr, color=RED,  linewidth=2.2, marker="o", markersize=6, label="FP - False Alarms")
    ax.plot(thresholds, f1,  color=BLUE, linewidth=2.2, marker="o", markersize=6, label="F1 Score", linestyle="--")

    # mark chosen threshold
    chosen = 0.50
    ax.axvline(x=chosen, color="#ff6b6b", linewidth=1.6, linestyle="--", alpha=0.8, zorder=4)
    # ax.text(chosen + 0.005, 96, "Chosen\nthreshold\n(0.5)", color="#ff6b6b",
    #         fontsize=8, va="top")

    # annotate chosen point values
    idx = thresholds.index(chosen)
    for val, color, dy in [(tpr[idx], GRN, 4), (fpr[idx], RED, -8), (f1[idx], BLUE, -16)]:
        ax.annotate(f"{val:.1f}%", xy=(chosen, val), xytext=(chosen - 0.07, val + dy),
                    color=color, fontsize=8, fontweight="bold")

    ax.set_xlim(0.25, 0.95)
    ax.set_ylim(0, 108)
    ax.set_xticks(thresholds)
    ax.set_xticklabels([str(t) for t in thresholds], color=GREY, fontsize=8.5)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)], color=GREY)
    ax.set_xlabel("Decision Threshold", color=GREY, fontsize=10)
    ax.set_ylabel("Rate (%)", color=GREY, fontsize=10)
    ax.set_title("Threshold Trade-off - TP vs FP vs F1",
                 color=WHITE, fontsize=12, fontweight="bold", pad=12)
    ax.grid(color="#1e2a3a", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.legend(facecolor="#1a1f3a", edgecolor="#333", labelcolor=WHITE, fontsize=9)

    plt.tight_layout()
    save_fig("chart_threshold_tradeoff")
    plt.close()


# ── Chart 6: Score distribution histogram ────────────────────────────────────
def plot_score_distribution(test: list[dict], scores: list[float]) -> None:
    mal_scores = [s for r, s in zip(test, scores) if r["label"] == 1]
    ben_scores = [s for r, s in zip(test, scores) if r["label"] == 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bins = np.linspace(0, 1, 40)
    ax.hist(ben_scores, bins=bins, color=BLUE, alpha=0.75, label="Benign", zorder=3)
    ax.hist(mal_scores, bins=bins, color=RED,  alpha=0.75, label="Malicious", zorder=3)

    ax.axvline(x=THRESHOLD, color="#ff6b6b", linewidth=1.8, linestyle="--", zorder=4)
    ax.text(THRESHOLD + 0.01, ax.get_ylim()[1] * 0.92,
            f"Threshold\n({THRESHOLD})", color="#ff6b6b", fontsize=8.5, va="top")

    ax.set_xlabel("Malicious Probability Score", color=GREY, fontsize=10)
    ax.set_ylabel("Number of Samples", color=GREY, fontsize=10)
    ax.set_title("Score Distribution — Benign vs Malicious Test Samples",
                 color=WHITE, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels([f"{v:.0%}" for v in np.arange(0, 1.1, 0.1)], color=GREY, fontsize=8)
    ax.tick_params(axis="y", colors=GREY)
    ax.grid(axis="y", color="#1e2a3a", linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#333")

    ax.legend(facecolor="#1a1f3a", edgecolor="#333", labelcolor=WHITE, fontsize=10)

    plt.tight_layout()
    save_fig("chart_score_distribution")
    plt.close()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    print("Loading data...")
    train = load_jsonl(TRAIN_TRACES)
    test  = load_jsonl(TEST_TRACES)

    print("Training TF-IDF models...")
    metrics, *_ = compute_metrics(train, test)

    metrics["sentence_transformer"] = compute_sentence_transformer_metrics(train, test)

    print("Scoring test samples with the logistic model...")
    clf = metrics["logistic"]["clf"]
    test_scores = clf.predict_proba(
        [r["user_goal"] for r in test],
        [r["agent_trace"] for r in test],
    )

    print("\nGenerating charts...")
    llm_pct, clf_pct = llm_vs_classifier_stats(test, test_scores)
    plot_llm_vs_classifier(llm_pct, clf_pct)
    plot_model_comparison(metrics)
    plot_prf_breakdown(metrics)
    plot_all_models_prf(metrics)
    plot_threshold_tradeoff(test, test_scores)
    plot_score_distribution(test, test_scores)
    print("\nDone.")


if __name__ == "__main__":
    main()
