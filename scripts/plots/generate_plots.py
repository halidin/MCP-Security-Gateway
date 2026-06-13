"""Generate tradeoffs, grouped threshold, and accuracy comparison curve plots (F1 curves & Precision-Recall curves) in dark mode."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_report(path: Path) -> list[dict] | None:
    if not path.exists():
        print(f"WARNING: Report {path} not found. Skipping.")
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    reports_dir = Path("reports")
    plots_dir = Path("reports/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    models = ["logistic", "random_forest", "gradient_boosting", "svm"]
    model_labels = {
        "logistic": "Logistic Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "svm": "SVM (Calibrated LinearSVC)"
    }

    # Custom premium color scheme: white, cyan, blue, indigo
    colors = {
        "logistic": "#ffffff",          # White
        "random_forest": "#22d3ee",     # Cyan
        "gradient_boosting": "#3b82f6",  # Blue
        "svm": "#818cf8"                # Indigo/Slate Blue
    }

    # Load all reports
    data = {}
    for model in models:
        report_data = load_report(reports_dir / f"tradeoffs_{model}.json")
        if report_data:
            data[model] = report_data

    if not data:
        print("No reports found to plot.")
        return

    # Set matplotlib dark background style globally
    plt.style.use('dark_background')

    # =========================================================================
    # PLOT 1: Single Threshold 0.60 Horizon Comparison
    # =========================================================================
    print("Generating dark-mode average horizon steps comparison at threshold = 0.60...")
    comp_labels = []
    comp_horizon = []
    comp_colors = []

    for model in models:
        if model not in data:
            continue
        target_rec = next((r for r in data[model] if abs(r["threshold"] - 0.6) < 0.01), None)
        if target_rec:
            comp_labels.append(model_labels[model])
            comp_horizon.append(target_rec["horizon_mean"])
            comp_colors.append(colors[model])

    if comp_labels:
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
        fig.patch.set_facecolor('#0f172a')  # Figure background
        ax.set_facecolor('#1e293b')         # Axes background

        # Colored bars according to the classifier color scheme
        bars = ax.bar(comp_labels, comp_horizon, color=comp_colors, width=0.45, edgecolor="none", zorder=3)
        
        ax.set_title('Average Detection Horizon Comparison (Threshold = 0.60)', fontsize=13, fontweight='bold', pad=20, color='#f8fafc')
        ax.set_ylabel('Mean Horizon Steps (earlier is better)', fontsize=11, color='#cbd5e1')
        ax.set_ylim(0, 5.5)

        ax.grid(True, axis='y', linestyle=":", alpha=0.3, color='#64748b', zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        ax.tick_params(colors='#cbd5e1', labelsize=10)
        plt.xticks(rotation=5, ha="center")

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2., 
                height + 0.12, 
                f"{height:.2f}", 
                ha='center', 
                va='bottom', 
                fontweight="bold", 
                color='#f1f5f9',
                fontsize=10
            )

        plt.tight_layout()
        plot_path = plots_dir / "horizon_comparison_0.60.png"
        plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches="tight")
        plt.close()
        print(f"  Saved {plot_path}")

    # =========================================================================
    # PLOT 2: Grouped Bar Chart of Malicious Detected Across All Thresholds
    # =========================================================================
    print("Generating grouped malicious detected counts across all thresholds...")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    series = {}
    for model in models:
        if model in data:
            series[model] = [
                next((r["detected_malicious"] for r in data[model] if abs(r["threshold"] - t) < 0.01), 0)
                for t in thresholds
            ]

    if len(series) == len(models):
        fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')

        x = np.arange(len(thresholds))
        width = 0.18  # Width of each individual bar

        # Plot 4 bars side-by-side at each threshold index
        rects1 = ax.bar(x - 1.5*width, series["logistic"], width, label=model_labels["logistic"], color=colors["logistic"], edgecolor="none", zorder=3)
        rects2 = ax.bar(x - 0.5*width, series["random_forest"], width, label=model_labels["random_forest"], color=colors["random_forest"], edgecolor="none", zorder=3)
        rects3 = ax.bar(x + 0.5*width, series["gradient_boosting"], width, label=model_labels["gradient_boosting"], color=colors["gradient_boosting"], edgecolor="none", zorder=3)
        rects4 = ax.bar(x + 1.5*width, series["svm"], width, label=model_labels["svm"], color=colors["svm"], edgecolor="none", zorder=3)

        ax.set_title('Malicious Traces Detected Comparison across All Thresholds', fontsize=14, fontweight='bold', pad=20, color='#f8fafc')
        ax.set_xlabel('Decision Threshold', fontsize=11, color='#cbd5e1', labelpad=10)
        ax.set_ylabel('Malicious Traces Detected (out of 1000)', fontsize=11, color='#cbd5e1')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t:.1f}" for t in thresholds], color='#cbd5e1')
        ax.set_ylim(0, 1100)

        ax.grid(True, axis='y', linestyle=":", alpha=0.3, color='#64748b', zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        ax.tick_params(colors='#cbd5e1', labelsize=10)
        
        # Legend with dark theme styling
        ax.legend(
            frameon=True, 
            facecolor='#1e293b', 
            edgecolor='#475569', 
            shadow=False, 
            labelcolor='#f1f5f9',
            loc="upper right",
            fontsize=10
        )

        plt.tight_layout()
        plot_path = plots_dir / "horizon_comparison_all_thresholds.png"
        plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches="tight")
        plt.close()
        print(f"  Saved {plot_path}")

    # =========================================================================
    # PLOT 3: F1-Score Curves Comparison
    # =========================================================================
    print("Generating F1-Score curves comparison across all models...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')

    markers = {"logistic": "o", "random_forest": "s", "gradient_boosting": "^", "svm": "x"}

    for model in models:
        if model not in data:
            continue
        records = data[model]
        t_list = [r["threshold"] for r in records]
        f1_list = [r["f1"] for r in records]
        
        ax.plot(
            t_list, 
            f1_list, 
            marker=markers[model], 
            label=model_labels[model], 
            color=colors[model], 
            linewidth=2,
            markersize=6
        )

    ax.set_title('F1-Score Accuracy Curve Comparison', fontsize=14, fontweight='bold', pad=20, color='#f8fafc')
    ax.set_xlabel('Decision Threshold', fontsize=11, color='#cbd5e1')
    ax.set_ylabel('F1-Score', fontsize=11, color='#cbd5e1')
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.3, color='#64748b')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors='#cbd5e1', labelsize=10)
    ax.legend(frameon=True, facecolor='#1e293b', edgecolor='#475569', labelcolor='#f1f5f9', fontsize=10)

    plt.tight_layout()
    plot_path = plots_dir / "f1_curves_comparison.png"
    plt.savefig(plot_path, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches="tight")
    plt.close()
    print("\nAll plots generated successfully. Others were removed.")


if __name__ == "__main__":
    main()
