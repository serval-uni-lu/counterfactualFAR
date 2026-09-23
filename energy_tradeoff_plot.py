#!/usr/bin/python
"""Compare every model run that has energy tracking against the default model
(rfr_100, i.e. plain RFR with no tuning) on energy consumption (total, CPU, GPU,
RAM) and on monthly_prof@10 / ndcg@10 — all expressed as % change vs. the
default, so cost and quality sit on the same indexed scale and can be read
side by side. Row 1 is energy (cost), row 2 is performance (quality), each
row with its own color pair so the two kinds of metric are never confused.

Output: stats/plots/model/energy_vs_performance_tradeoff.png
"""
import os
import re

import pandas as pd
import matplotlib.pyplot as plt

STATS_MODEL_DIR = os.path.join("stats", "model")
OUT_DIR = os.path.join("stats", "plots", "model")
OUT_FILE = os.path.join(OUT_DIR, "energy_vs_performance_tradeoff.png")

REFERENCE_RUN = "rfr_100_full_short_internal_kpis"

ENERGY_METRICS = ["energy_consumed_kwh", "cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh"]
PERFORMANCE_METRICS = ["monthly_prof@10", "ndcg@10"]
METRICS = ENERGY_METRICS + PERFORMANCE_METRICS

METRIC_TITLES = {
    "energy_consumed_kwh": "Total energy (higher = worse)",
    "cpu_energy_kwh": "CPU energy (higher = worse)",
    "gpu_energy_kwh": "GPU energy (higher = worse)",
    "ram_energy_kwh": "RAM energy (higher = worse)",
    "monthly_prof@10": "Monthly profitability@10 (higher = better)",
    "ndcg@10": "NDCG@10 (higher = better)",
}

# Energy row: increase/decrease in cost. Performance row: its own color pair
# so the two rows are never read as the same kind of quantity.
ENERGY_UP, ENERGY_DOWN = "#e34948", "#2a78d6"      # red / blue
PERF_UP, PERF_DOWN = "#0ca30c", "#eb6834"          # green / orange
GRAY = "#898781"


def _display_name(run_prefix):
    m = re.search(r"tabpfn_sample([\d.]+)$", run_prefix)
    if m:
        return f"tabpfn (sample {m.group(1)})"
    if run_prefix.startswith("rfr_tuned"):
        return "rfr (tuned)"
    if run_prefix.startswith("lgbm_tuned"):
        return "lgbm (tuned)"
    return run_prefix


def _sort_key(run_prefix):
    m = re.search(r"tabpfn_sample([\d.]+)$", run_prefix)
    if m:
        return (1, float(m.group(1)))
    if run_prefix.startswith("rfr_tuned"):
        return (0, 0)
    if run_prefix.startswith("lgbm_tuned"):
        return (0, 1)
    return (0, 2)


def load_means():
    means = {}
    for fname in sorted(os.listdir(STATS_MODEL_DIR)):
        if not fname.endswith(".csv") or "_exp1_" in fname or "_exp2_" in fname:
            continue
        run_prefix = fname[: -len(".csv")]
        df = pd.read_csv(os.path.join(STATS_MODEL_DIR, fname))
        if "energy_consumed_kwh" not in df["metric"].values:
            continue
        row_means = df.set_index("metric")["mean"]
        if not all(m in row_means.index for m in METRICS):
            continue
        means[run_prefix] = row_means[METRICS]
    return means


def _draw_panel(ax, values, up_color, down_color, title, symlog, show_ylabels):
    colors = [up_color if v > 0 else down_color for v in values]
    ax.barh(values.index, values, color=colors)
    ax.axvline(0, color=GRAY, linewidth=1.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("% change vs. default (rfr, untuned)")
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9, labelleft=show_ylabels)

    if symlog:
        ax.set_xscale("symlog", linthresh=10)

    for y, v in enumerate(values):
        ax.annotate(
            f"{v:+.0f}%",
            xy=(v, y),
            xytext=(6 if v >= 0 else -6, 0),
            textcoords="offset points",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8,
            color="#0b0b0b",
        )


def main():
    means = load_means()
    if REFERENCE_RUN not in means:
        raise ValueError(f"Reference run '{REFERENCE_RUN}' not found among energy-tracked runs: {sorted(means)}")

    ref = means.pop(REFERENCE_RUN)
    runs = sorted(means.keys(), key=_sort_key)

    pct_change = pd.DataFrame(
        {metric: [(means[r][metric] - ref[metric]) / abs(ref[metric]) * 100 for r in runs] for metric in METRICS},
        index=[_display_name(r) for r in runs],
    )

    n_cols = len(ENERGY_METRICS)
    fig = plt.figure(figsize=(6 * n_cols, 2 * max(4, 0.5 * len(runs))))
    gs = fig.add_gridspec(2, n_cols, hspace=0.35, wspace=0.3)

    first_ax = None
    for i, metric in enumerate(ENERGY_METRICS):
        ax = fig.add_subplot(gs[0, i], sharey=first_ax)
        first_ax = first_ax or ax
        # energy spans two orders of magnitude (rfr/lgbm near 0%, tabpfn up to
        # +2000%+); symlog keeps the small changes readable next to the large ones.
        _draw_panel(ax, pct_change[metric], ENERGY_UP, ENERGY_DOWN, METRIC_TITLES[metric],
                    symlog=True, show_ylabels=(i == 0))

    perf_span = n_cols // len(PERFORMANCE_METRICS)
    for i, metric in enumerate(PERFORMANCE_METRICS):
        ax = fig.add_subplot(gs[1, i * perf_span:(i + 1) * perf_span], sharey=first_ax)
        _draw_panel(ax, pct_change[metric], PERF_UP, PERF_DOWN, METRIC_TITLES[metric],
                    symlog=False, show_ylabels=(i == 0))

    first_ax.invert_yaxis()
    fig.suptitle("Energy vs. performance trade-off, relative to the default model (rfr_100)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=170)
    plt.close(fig)
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    main()
