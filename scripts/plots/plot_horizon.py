"""Detection-horizon charts: how early (in reasoning steps) MCPGuard would
flag a malicious trace, relative to the step where the tool call happens.

This is a retrospective replay (see evaluate_horizon.py): for each malicious
sample, the trace is split into steps and the classifier scores the
incrementally-growing prefix after each step. The "horizon" is
num_steps - first_alert_step (how many steps of lead time the guard has).
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from interceptor.features import combine_goal_and_trace, parse_trace_steps
from interceptor.model import DriftClassifier

# ── Settings ──────────────────────────────────────────────────────────────────
TRAIN_TRACES = Path("data/traces/train.jsonl")
TEST_TRACES  = Path("data/traces/test.jsonl")
MODEL_PATH   = Path("models/logistic.joblib")
MODELS_DIR   = Path("models")
OUT          = Path("reports/plots")
THRESHOLD    = 0.5
MAX_STEPS_SHOWN = 8
SENTENCE_MODEL = "all-MiniLM-L6-v2"

MODEL_ORDER  = ["logistic", "random_forest", "gradient_boosting", "svm"]
MODEL_LABELS = {
    "logistic":             "Logistic Regression",
    "random_forest":        "Random Forest",
    "gradient_boosting":    "Gradient Boosting",
    "svm":                  "SVM",
    "sentence_transformer": "Sentence Transformer - LR",
}
MODEL_COLORS = {
    "logistic":             "#ffffff",
    "random_forest":        "#22d3ee",
    "gradient_boosting":    "#3b82f6",
    "svm":                  "#818cf8",
    "sentence_transformer": "#f472b6",
}
# ─────────────────────────────────────────────────────────────────────────────

BG    = "#0f1629"
BLUE  = "#64b5f6"
ORG   = "#ffa726"
GRN   = "#66bb6a"
RED   = "#ef5350"
GREY  = "#aaaaaa"
WHITE = "white"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def style_ax(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=GREY)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(True, color="#2a2f4a", linewidth=0.6, zorder=0)


def save_fig(stem: str) -> None:
    for ext in ("png", "svg"):
        out = OUT / f"{stem}.{ext}"
        kw = dict(bbox_inches="tight", facecolor=BG)
        if ext == "png":
            kw["dpi"] = 160
        plt.savefig(out, **kw)
        print(f"  Saved -> {out}")


def compute_score_sequences(clf: DriftClassifier, malicious: list[dict]) -> list[tuple[list[float], int]]:
    """For each malicious sample, return (scores per step prefix, num_steps)."""
    jobs: list[tuple[int, str, str]] = []  # (sample_idx, goal, prefix)
    sample_steps: list[list[str]] = []

    for idx, row in enumerate(malicious):
        steps = row.get("trace_steps")
        if not isinstance(steps, list) or not steps:
            steps = parse_trace_steps(row.get("agent_trace", ""))
        sample_steps.append(steps)
        for i in range(1, len(steps) + 1):
            prefix = " ".join(steps[:i])
            jobs.append((idx, row["user_goal"], prefix))

    if not jobs:
        return []

    goals   = [g for _, g, _ in jobs]
    prefixes = [p for _, _, p in jobs]
    scores  = clf.predict_proba(goals, prefixes)

    per_sample_scores: list[list[float]] = [[] for _ in malicious]
    for (idx, _, _), score in zip(jobs, scores):
        per_sample_scores[idx].append(score)

    results = []
    for idx, row in enumerate(malicious):
        steps = sample_steps[idx]
        results.append((per_sample_scores[idx], len(steps)))
    return results


def compute_score_sequences_sentence_transformer(
    malicious: list[dict], train: list[dict]
) -> list[tuple[list[float], int]]:
    """Same as compute_score_sequences, but using a SentenceTransformer
    embedding + LogisticRegression head instead of the TF-IDF pipeline."""
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    print(f"  Training sentence-transformer ({SENTENCE_MODEL}) + LR...")
    train_texts = [combine_goal_and_trace(r["user_goal"], r["agent_trace"]) for r in train]
    train_labels = [int(r["label"]) for r in train]

    st_model = SentenceTransformer(SENTENCE_MODEL)
    train_emb = st_model.encode(train_texts, batch_size=64, show_progress_bar=False)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(train_emb, train_labels)

    jobs: list[tuple[int, str]] = []  # (sample_idx, combined text)
    sample_steps: list[list[str]] = []

    for idx, row in enumerate(malicious):
        steps = row.get("trace_steps")
        if not isinstance(steps, list) or not steps:
            steps = parse_trace_steps(row.get("agent_trace", ""))
        sample_steps.append(steps)
        for i in range(1, len(steps) + 1):
            prefix = " ".join(steps[:i])
            jobs.append((idx, combine_goal_and_trace(row["user_goal"], prefix)))

    if not jobs:
        return []

    texts = [t for _, t in jobs]
    print(f"  Encoding {len(texts)} step-prefixes...")
    embs = st_model.encode(texts, batch_size=64, show_progress_bar=False)
    scores = lr.predict_proba(embs)[:, 1]

    per_sample_scores: list[list[float]] = [[] for _ in malicious]
    for (idx, _), score in zip(jobs, scores):
        per_sample_scores[idx].append(float(score))

    results = []
    for idx, row in enumerate(malicious):
        steps = sample_steps[idx]
        results.append((per_sample_scores[idx], len(steps)))
    return results


def first_alert_step(scores: list[float], threshold: float) -> int | None:
    for i, s in enumerate(scores, start=1):
        if s >= threshold:
            return i
    return None


def plot_horizon_histogram(seqs: list[tuple[list[float], int]]) -> None:
    horizons = []
    for scores, num_steps in seqs:
        fa = first_alert_step(scores, THRESHOLD)
        if fa is not None:
            horizons.append(max(0, num_steps - fa))

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BG)
    style_ax(ax)

    max_h = max(horizons) if horizons else 0
    bins = np.arange(-0.5, max_h + 1.5, 1)
    ax.hist(horizons, bins=bins, color=BLUE, edgecolor="#222", zorder=3)

    mean_h = np.mean(horizons) if horizons else 0.0
    ax.axvline(mean_h, color=ORG, linestyle="--", linewidth=2, zorder=4,
               label=f"Mean = {mean_h:.2f} steps")

    ax.set_xlabel("Detection horizon (steps before tool call)", color=GREY, fontsize=10)
    ax.set_ylabel("Number of malicious samples", color=GREY, fontsize=10)
    ax.set_title(f"Detection Horizon Distribution (threshold = {THRESHOLD})",
                  color=WHITE, fontsize=12, fontweight="bold")
    legend = ax.legend(facecolor=BG, edgecolor="#444", fontsize=9)
    for text in legend.get_texts():
        text.set_color(WHITE)

    plt.tight_layout()
    save_fig("chart_horizon_histogram")
    plt.close()


def plot_cumulative_detection(seqs: list[tuple[list[float], int]]) -> None:
    n = len(seqs)
    counts = np.zeros(MAX_STEPS_SHOWN, dtype=int)
    undetected = 0

    for scores, _ in seqs:
        fa = first_alert_step(scores, THRESHOLD)
        if fa is None:
            undetected += 1
            continue
        step = min(fa, MAX_STEPS_SHOWN)
        counts[step - 1] += 1

    cumulative = np.cumsum(counts) / n * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BG)
    style_ax(ax)

    x = np.arange(1, MAX_STEPS_SHOWN + 1)
    ax.bar(x, cumulative, color=GRN, edgecolor="#222", zorder=3, width=0.6)

    for xi, yi in zip(x, cumulative):
        ax.text(xi, yi + 1.5, f"{yi:.1f}%", ha="center", va="bottom",
                color=WHITE, fontsize=8.5, fontweight="bold")

    ax.set_xlabel("Reasoning step", color=GREY, fontsize=10)
    ax.set_ylabel("Cumulative % of malicious samples flagged", color=GREY, fontsize=10)
    ax.set_title(f"Cumulative Detection Rate by Step (threshold = {THRESHOLD})",
                  color=WHITE, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    save_fig("chart_horizon_cumulative")
    plt.close()
    print(f"  Undetected (never crossed threshold): {undetected}/{n} ({undetected/n*100:.1f}%)")


def plot_horizon_vs_threshold(seqs: list[tuple[list[float], int]]) -> None:
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    avg_horizons = []
    detection_rates = []

    for t in thresholds:
        horizons = []
        detected = 0
        for scores, num_steps in seqs:
            fa = first_alert_step(scores, t)
            if fa is not None:
                detected += 1
                horizons.append(max(0, num_steps - fa))
        avg_horizons.append(np.mean(horizons) if horizons else 0.0)
        detection_rates.append(detected / len(seqs) * 100)

    fig, ax1 = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BG)
    style_ax(ax1)

    ax1.plot(thresholds, avg_horizons, "o-", color=BLUE, linewidth=2, label="Avg. detection horizon (steps)")
    ax1.set_xlabel("Decision threshold", color=GREY, fontsize=10)
    ax1.set_ylabel("Avg. detection horizon (steps)", color=BLUE, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=BLUE)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, detection_rates, "o-", color=ORG, linewidth=2, label="Detection rate (%)")
    ax2.set_ylabel("Detection rate (%)", color=ORG, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=ORG)
    ax2.set_facecolor("none")
    for spine in ax2.spines.values():
        spine.set_color("#444")

    ax1.axvline(0.5, color=GREY, linestyle="--", linewidth=1.2, zorder=1)
    ax1.text(0.5, ax1.get_ylim()[1] * 0.95, " chosen = 0.5", color=GREY, fontsize=8.5, va="top")

    ax1.set_title("Detection Horizon vs. Threshold", color=WHITE, fontsize=12, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines1 + lines2, labels1 + labels2, facecolor=BG, edgecolor="#444", fontsize=9, loc="lower left")
    for text in legend.get_texts():
        text.set_color(WHITE)

    plt.tight_layout()
    save_fig("chart_horizon_vs_threshold")
    plt.close()


def plot_model_horizon_comparison(malicious: list[dict], train: list[dict] | None = None) -> None:
    """Average detection horizon for every trained model at THRESHOLD,
    with a side panel listing how many malicious samples each model caught."""
    total = len(malicious)
    labels, values, colors, detected_counts = [], [], [], []
    for name in MODEL_ORDER:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            print(f"  (skipping {name}: {path} not found)")
            continue
        clf = DriftClassifier.load(str(path))
        seqs = compute_score_sequences(clf, malicious)
        horizons = []
        detected = 0
        for scores, num_steps in seqs:
            fa = first_alert_step(scores, THRESHOLD)
            if fa is not None:
                detected += 1
                horizons.append(max(0, num_steps - fa))
        mean_h = float(np.mean(horizons)) if horizons else 0.0
        labels.append(MODEL_LABELS[name])
        values.append(mean_h)
        colors.append(MODEL_COLORS[name])
        detected_counts.append(detected)
        print(f"  {name:20s} mean horizon = {mean_h:.2f} steps  detected={detected}/{total}")

    if train is not None:
        seqs = compute_score_sequences_sentence_transformer(malicious, train)
        horizons = []
        detected = 0
        for scores, num_steps in seqs:
            fa = first_alert_step(scores, THRESHOLD)
            if fa is not None:
                detected += 1
                horizons.append(max(0, num_steps - fa))
        mean_h = float(np.mean(horizons)) if horizons else 0.0
        labels.append(MODEL_LABELS["sentence_transformer"])
        values.append(mean_h)
        colors.append(MODEL_COLORS["sentence_transformer"])
        detected_counts.append(detected)
        print(f"  {'sentence_transformer':20s} mean horizon = {mean_h:.2f} steps  detected={detected}/{total}")

    fig, (ax, ax_side) = plt.subplots(
        1, 2, figsize=(11.5, 5.5), gridspec_kw={"width_ratios": [3, 1.1], "wspace": 0.05}
    )
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("#1e293b")

    bars = ax.bar(labels, values, color=colors, width=0.45, edgecolor="none", zorder=3)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.12, f"{h:.2f}",
                ha="center", va="bottom", color="#f1f5f9", fontsize=10, fontweight="bold")

    ax.set_title(f"Average Detection Horizon Comparison (Threshold = {THRESHOLD:.2f})",
                 color="#f8fafc", fontsize=13, fontweight="bold", pad=20)
    ax.set_ylabel("Mean Horizon Steps (earlier is better)", color="#cbd5e1", fontsize=11)
    ax.set_ylim(0, max(values) + 1.0 if values else 5.5)
    ax.grid(True, axis="y", linestyle=":", alpha=0.3, color="#64748b", zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#cbd5e1", labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=5, ha="center")

    # Annotate the sentence-transformer bar with its underlying model name.
    if MODEL_LABELS["sentence_transformer"] in labels:
        i = labels.index(MODEL_LABELS["sentence_transformer"])
        ax.text(i, -0.45, f"({SENTENCE_MODEL})", ha="center", va="top",
                color="#64748b", fontsize=8, fontstyle="italic")

    # ── Side panel: detected counts ────────────────────────────────────────
    ax_side.set_facecolor("#1e293b")
    ax_side.set_xlim(0, 1)
    ax_side.set_ylim(0, 1)
    ax_side.axis("off")

    ax_side.text(0.5, 0.96, "Detected /\n1000 malicious", ha="center", va="top",
                  color="#f8fafc", fontsize=11, fontweight="bold")

    n = len(labels)
    row_h = 0.72 / n
    for i, (name, det, color) in enumerate(zip(labels, detected_counts, colors)):
        y = 0.74 - i * row_h
        ax_side.add_patch(plt.Rectangle((0.04, y - 0.025), 0.05, 0.05,
                                         transform=ax_side.transAxes,
                                         facecolor=color, edgecolor="none"))
        ax_side.text(0.14, y, name, ha="left", va="center", color="#cbd5e1", fontsize=9.5,
                      transform=ax_side.transAxes)
        pct = det / total * 100
        ax_side.text(0.14, y - row_h * 0.45, f"{det}/{total}  ({pct:.1f}%)", ha="left", va="center",
                      color="#f1f5f9", fontsize=10, fontweight="bold",
                      transform=ax_side.transAxes)

    for spine in ax_side.spines.values():
        spine.set_visible(False)

    save_fig("chart_horizon_model_comparison")
    plt.close()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    test = load_jsonl(TEST_TRACES)
    train = load_jsonl(TRAIN_TRACES)
    malicious = [r for r in test if int(r["label"]) == 1]
    clf = DriftClassifier.load(str(MODEL_PATH))

    print(f"Computing score sequences for {len(malicious)} malicious samples...")
    seqs = compute_score_sequences(clf, malicious)

    plot_horizon_histogram(seqs)
    plot_cumulative_detection(seqs)
    plot_horizon_vs_threshold(seqs)

    print("Computing per-model horizon comparison...")
    plot_model_horizon_comparison(malicious, train)


if __name__ == "__main__":
    main()
