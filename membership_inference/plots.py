"""
Plotting helpers for the membership-inference attacks.

Saved under stats/membership_inference/<attack>/<model>/<date>/.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from membership_inference.regression_signals import LOW_FPR_THRESHOLDS, tpr_at_fpr

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Fixed categorical identity: never reassigned or cycled between plots.
COLOR_MEMBER = "#2a78d6"       # slot 1: blue
COLOR_NONMEMBER = "#eb6834"    # slot 2: orange
COLOR_POPULATION = "#1baf7a"   # slot 3: aqua
COLOR_CHANCE = BASELINE

COLOR_METRIC_AUC = "#2a78d6"
COLOR_METRIC_ADVANTAGE = "#eb6834"
COLOR_METRIC_TPR_LOW_FPR = "#1baf7a"

FONT_STACK = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_STACK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK_PRIMARY,
})


def _style_axes(ax, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
        ax.spines[spine].set_linewidth(1)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.set_axisbelow(True)
    ax.grid(True, axis="both", color=GRIDLINE, linewidth=0.8)
    ax.title.set_color(INK_PRIMARY)
    ax.title.set_fontsize(11)
    ax.title.set_fontweight("bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)


def _style_legend(ax, **kwargs):
    return ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9, **kwargs)


def _style_suptitle(fig, title):
    fig.suptitle(title, color=INK_PRIMARY, fontsize=13, fontweight="bold")


def plot_roc_curve(member_scores, nonmember_scores, out_path, title):
    """Linear + log-log ROC (member vs. non-member). Log-log matters because MIA
    attacks are often only meaningfully above chance in the low-FPR corner, which a
    linear ROC hides."""
    member_scores = np.asarray(member_scores, dtype=float)
    nonmember_scores = np.asarray(nonmember_scores, dtype=float)
    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])

    fpr, tpr, _ = roc_curve(labels, scores)
    auc = float(roc_auc_score(labels, scores))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].plot(fpr, tpr, color=COLOR_MEMBER, linewidth=2, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color=COLOR_CHANCE, linestyle="--", linewidth=1.5, label="chance")
    _style_axes(axes[0], xlabel="False Positive Rate", ylabel="True Positive Rate")
    axes[0].set_title("ROC — linear")
    _style_legend(axes[0], loc="lower right")

    eps = 1e-4
    axes[1].plot(np.clip(fpr, eps, 1), np.clip(tpr, eps, 1), color=COLOR_MEMBER, linewidth=2, label=f"AUC = {auc:.3f}")
    axes[1].plot([eps, 1], [eps, 1], color=COLOR_CHANCE, linestyle="--", linewidth=1.5, label="chance")

    # TPR@low-FPR checkpoints
    for target_fpr in LOW_FPR_THRESHOLDS:
        point_tpr = tpr_at_fpr(fpr, tpr, target_fpr)
        axes[1].scatter([target_fpr], [max(point_tpr, eps)], s=45, color=COLOR_MEMBER,
                         edgecolor=SURFACE, linewidth=1.2, zorder=5)
        axes[1].annotate(f"{point_tpr:.3f}", (target_fpr, max(point_tpr, eps)),
                          textcoords="offset points", xytext=(6, 4), fontsize=8, color=INK_SECONDARY)

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(eps, 1)
    axes[1].set_ylim(eps, 1)
    _style_axes(axes[1], xlabel="False Positive Rate", ylabel="True Positive Rate")
    axes[1].set_title("ROC — log-log (labels: TPR @ FPR)")
    _style_legend(axes[1], loc="lower right")

    _style_suptitle(fig, title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tpr_at_low_fpr(tpr_at_fpr_dict, out_path, title):
    """Dedicated TPR@low-FPR view: one bar per checkpoint (0.1%/1%/10% FPR)."""
    thresholds = sorted(tpr_at_fpr_dict.keys())
    values = [tpr_at_fpr_dict[t] for t in thresholds]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(thresholds))
    ax.bar(x, values, width=0.5, color=COLOR_MEMBER, alpha=0.85, zorder=3)
    for xi, t, v in zip(x, thresholds, values):
        ax.plot([xi - 0.25, xi + 0.25], [t, t], color=INK_MUTED, linestyle="--", linewidth=1.5, zorder=4)
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 5),
                    ha="center", fontsize=9, color=INK_SECONDARY)

    ax.set_xticks(x)
    ax.set_xticklabels([f"FPR = {t*100:g}%" for t in thresholds])
    _style_axes(ax, ylabel="True Positive Rate")
    ax.set_title(title)
    from matplotlib.lines import Line2D
    chance_handle = Line2D([0], [0], color=INK_MUTED, linestyle="--", linewidth=1.5, label="chance (TPR = FPR)")
    ax.legend(handles=[chance_handle], frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_loss_distribution(member_loss, nonmember_loss, out_path, title, population_loss=None):
    """Overlaid loss histograms — the signal the attack's AUC is summarizing.

    Squared-error loss is heavily right-skewed (member losses cluster near zero,
    non-member losses have a long tail), so a linear x-axis crushes almost the
    entire distribution into the first bin. Log-x bins show the actual separation
    across the full range instead of clipping the tail away."""
    member_loss = np.asarray(member_loss, dtype=float)
    nonmember_loss = np.asarray(nonmember_loss, dtype=float)
    series = [("member", member_loss, COLOR_MEMBER), ("non-member", nonmember_loss, COLOR_NONMEMBER)]
    if population_loss is not None:
        series.append(("population", np.asarray(population_loss, dtype=float), COLOR_POPULATION))

    all_vals = np.concatenate([vals for _, vals, _ in series])
    all_vals = all_vals[all_vals > 0]
    lo = max(np.percentile(all_vals, 5), 1e-6) if len(all_vals) else 1e-6
    hi = np.percentile(all_vals, 99) if len(all_vals) else 1.0
    bins = np.logspace(np.log10(lo), np.log10(max(hi, lo * 10)), 40)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, vals, color in series:
        # Values outside [lo, hi] simply fall outside every bin (excluded from
        # the plot, not piled onto the edges) — only the informative middle
        # range is shown, since the extreme tails of a squared-error loss
        # aren't where the member/non-member separation lives.
        ax.hist(vals, bins=bins, density=True, histtype="stepfilled",
                 facecolor=color, edgecolor=color, alpha=0.35, linewidth=1.5,
                 label=f"{label} (n={len(vals)})")
    ax.set_xscale("log")
    _style_axes(ax, xlabel="loss (squared error, log scale)", ylabel="density")
    ax.set_title(title)
    _style_legend(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_percentile_distribution(member_percentile, nonmember_percentile, out_path, title):
    """Population-attack-only: member vs. non-member rank within the population loss
    distribution — checks whether members really do cluster at low percentiles."""
    member_percentile = np.asarray(member_percentile, dtype=float)
    nonmember_percentile = np.asarray(nonmember_percentile, dtype=float)
    bins = np.linspace(0, 1, 41)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(member_percentile, bins=bins, density=True, histtype="stepfilled",
             facecolor=COLOR_MEMBER, edgecolor=COLOR_MEMBER, alpha=0.35, linewidth=1.5,
             label=f"member (n={len(member_percentile)})")
    ax.hist(nonmember_percentile, bins=bins, density=True, histtype="stepfilled",
             facecolor=COLOR_NONMEMBER, edgecolor=COLOR_NONMEMBER, alpha=0.35, linewidth=1.5,
             label=f"non-member (n={len(nonmember_percentile)})")
    _style_axes(ax, xlabel="percentile rank within population loss distribution", ylabel="density")
    ax.set_title(title)
    _style_legend(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_across_dates(dates, auc_values, advantage_values, tpr_at_1pct_fpr_values, out_path, title):
    """Trend of attack strength across recommendation-date snapshots."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(dates))
    ax.plot(x, auc_values, color=COLOR_METRIC_AUC, marker="o", markersize=8,
             linewidth=2, label="ROC-AUC")
    ax.plot(x, advantage_values, color=COLOR_METRIC_ADVANTAGE, marker="s", markersize=8,
             linewidth=2, label="attack advantage")
    ax.plot(x, tpr_at_1pct_fpr_values, color=COLOR_METRIC_TPR_LOW_FPR, marker="^", markersize=8,
             linewidth=2, label="TPR @ 1% FPR")
    ax.axhline(0.5, color=INK_MUTED, linestyle="--", linewidth=1.2, label="chance (AUC = 0.5)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(dates, rotation=45, ha="right")
    _style_axes(ax, ylabel="metric value")
    ax.set_title(title)
    _style_legend(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
