"""
Plotting helpers for the membership-inference attacks.

Saved under stats/plots/membership_inference/<attack>/<model>/<date>/ (PLOT_ROOT).
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from trustworthiness_eval.privacy.membership_inference.regression_signals import LOW_FPR_THRESHOLDS, tpr_at_fpr

PLOT_ROOT = Path("stats") / "plots" / "membership_inference"

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


def plot_percentile_distribution(member_percentile, nonmember_percentile, out_path, title,
                                 xlabel="population score (fraction of population with loss ≥ sample's)"):
    """Member vs. non-member score in [0, 1] — the population score (fraction of the
    population with loss >= the sample's), or with another `xlabel` the RMIA score —
    checks whether members really do cluster near 1."""
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
    _style_axes(ax, xlabel=xlabel, ylabel="density")
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


def plot_evaluation_levels(levels, out_path, title):
    """Attack strength at each evaluation level (single rows, then asset-period blocks),
    one panel per metric, with the asset-bootstrap 95% interval as a whisker.

    `levels`: list of {"label", "roc_auc", "attack_advantage", "tpr_at_fpr_0.01"}, each
    metric a {"value", "ci_low", "ci_high"} dict."""
    panels = [("roc_auc", "ROC-AUC", 0.5), ("attack_advantage", "attack advantage", 0.0),
              ("tpr_at_fpr_0.01", "TPR @ 1% FPR", 0.01)]
    labels = [level["label"] for level in levels]
    y = np.arange(len(levels))[::-1]

    fig, axes = plt.subplots(1, len(panels), figsize=(12, 1.6 + 0.6 * len(levels)), sharey=True)
    for ax, (key, name, chance) in zip(axes, panels):
        values = np.array([level[key]["value"] for level in levels])
        low = np.array([level[key]["ci_low"] for level in levels])
        high = np.array([level[key]["ci_high"] for level in levels])
        ax.axvline(chance, color=COLOR_CHANCE, linestyle="--", linewidth=1.5)
        ax.hlines(y, low, high, color=COLOR_MEMBER, linewidth=2, zorder=3)
        ax.scatter(values, y, s=55, color=COLOR_MEMBER, edgecolor=SURFACE, linewidth=1.2, zorder=4)
        for yi, v, hi in zip(y, values, high):
            ax.annotate(f"{v:.3f}", (hi, yi), textcoords="offset points", xytext=(6, -3),
                        fontsize=9, color=INK_SECONDARY)
        _style_axes(ax, xlabel=f"{name} (dashed = chance)")
        ax.grid(False, axis="y")
        ax.set_xlim(min(chance, low.min()) - 0.03, min(1.0, high.max()) + 0.12)
        ax.set_ylim(-0.6, len(levels) - 0.4)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    _style_suptitle(fig, title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_temporal_analysis(bins, out_path, title, threshold_label):
    """How the attack's verdict depends on a sample's distance from the training cutoff.

    `bins`: DataFrame with one row per (is_member, distance bin): days_mid, flag_rate,
    auc_vs_opposite (each with _low/_high asset-bootstrap bounds) and median_loss.
    Three stacked panels on one time axis — one measure each, never a shared y-axis."""
    panels = [
        ("flag_rate", f"share flagged as member\n({threshold_label})", None),
        ("auc_vs_opposite", "ROC-AUC against all samples\nof the other class", 0.5),
        ("median_loss", "median loss (log scale)", None),
    ]
    series = [(True, "member (training period)", COLOR_MEMBER), (False, "non-member (test period)", COLOR_NONMEMBER)]

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 10), sharex=True)
    for ax, (key, ylabel, chance) in zip(axes, panels):
        for is_member, label, color in series:
            part = bins[bins["is_member"] == is_member].sort_values("days_mid")
            if f"{key}_low" in part.columns:
                ax.fill_between(part["days_mid"], part[f"{key}_low"], part[f"{key}_high"],
                                 color=color, alpha=0.18, linewidth=0)
            ax.plot(part["days_mid"], part[key], color=color, linewidth=2, marker="o", markersize=5,
                     markeredgecolor=SURFACE, markeredgewidth=1, label=label)
        ax.axvline(0, color=INK_MUTED, linestyle="--", linewidth=1.2)
        if chance is not None:
            ax.axhline(chance, color=COLOR_CHANCE, linestyle=":", linewidth=1.5)
        if key == "median_loss":
            ax.set_yscale("log")
        else:
            ax.set_ylim(-0.03, 1.03)
        _style_axes(ax, ylabel=ylabel)
    axes[0].annotate("training cutoff", (0, 1.0), xycoords=("data", "axes fraction"), textcoords="offset points",
                     xytext=(-6, 4), ha="right", fontsize=9, color=INK_SECONDARY)
    _style_legend(axes[0], loc="lower left")
    axes[-1].set_xlabel("trading days from the training cutoff (shaded = 95% interval, assets resampled)",
                         color=INK_SECONDARY, fontsize=10)
    _style_suptitle(fig, title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
