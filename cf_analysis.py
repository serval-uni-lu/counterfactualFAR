"""cf_analysis.py — Counterfactual Analysis
==========================================
Answers 7 diagnostic questions from CF output files:

  1. Feasibility   — Which assets get CFs? Distribution of CF count per asset.
  2. Effort        — How much do prices need to change (distance / sparsity)?
  3. Score lift    — How much does the CF improve the profitability score?
  4. Price pattern — Which window positions drive the change? What direction?
  5. Temporal      — How do experiments (2020 vs 2021) compare?
  6. Segmentation  — Asset clusters by effort/lift profile and actionability.
  7. Actionability — Are CF changes realistic (< 5% relative price change)?
  8. Data utility  — Do CF price windows look statistically like real price windows?

Usage:
    python cf_analysis.py                        # all sections, default dirs
    python cf_analysis.py --sections feasibility lift
    python cf_analysis.py --cf-dir counterfactuals/rfr_n-100_kpi-full_short_internal_kpis
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

CF_ROOT = "counterfactuals"
_MODEL_TAG = "rfr_n-100_kpi-full_short_internal_kpis"
OUT_ROOT = os.path.join("stats", "cf_analysis")

# Thresholds for actionability classification
ACTIONABLE_REL_DELTA_THRESHOLD = 0.05   # mean relative price change < 5%
ACTIONABLE_SPARSITY_THRESHOLD = 0.50    # changes on < 50% of window days


# ─────────────────────────────────────────────────────────────────────────────
# File discovery and loading
# ─────────────────────────────────────────────────────────────────────────────

def _discover_cf_files(cf_dir):
    """Find all cf_details_*.csv files and their paired summary files.

    Returns list of dicts with keys: details_path, summary_path, exp_date, method, label.
    Files are expected to follow the naming convention:
        cf_details_{model_tag}_{YYYY-MM-DD}_{method}.csv
    """
    details_files = sorted(glob.glob(os.path.join(cf_dir, "cf_details_*.csv")))
    results = []
    for details_path in details_files:
        name = os.path.basename(details_path)
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", name)
        method_match = re.search(r"(\d{4}-\d{2}-\d{2})_([^.]+)\.csv$", name)
        exp_date = date_match.group(1) if date_match else "unknown"
        method = method_match.group(2) if method_match else "unknown"

        summary_path = os.path.join(cf_dir, name.replace("cf_details_", "summary_"))
        if not os.path.exists(summary_path):
            summary_path = None

        results.append({
            "details_path": details_path,
            "summary_path": summary_path,
            "exp_date": exp_date,
            "method": method,
            "label": f"{exp_date} ({method})",
        })
    return results


def _safe_label(label):
    return re.sub(r"[^A-Za-z0-9_-]", "_", label)


def _load_details(path):
    df = pd.read_csv(path)
    df["col_timestamp"] = pd.to_datetime(df["col_timestamp"], format="mixed")
    # cf_details is documented to hold only successful queries (see section_feasibility's
    # note); process_results.py's fill_no_cf_rows() appends NaN placeholder rows for no_cf/
    # skipped queries for other bookkeeping, so drop them back out here.
    if "cf_prediction" in df.columns:
        df = df[df["cf_prediction"].notna()].reset_index(drop=True)
    return df


def _load_summary(path):
    df = pd.read_csv(path)
    df["col_timestamp"] = pd.to_datetime(df["col_timestamp"], format="mixed")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Feasibility
# ─────────────────────────────────────────────────────────────────────────────

def section_feasibility(cf_files, out_dir):
    """How many assets/queries got a CF? Distribution of CF count per asset."""
    print("\n" + "=" * 60)
    print("SECTION 1: FEASIBILITY")
    print("=" * 60)
    print(
        "NOTE: cf_details only contains successful queries (no CF = not written).\n"
        "      Coverage % relative to total queries requires generator logs.\n"
        "      Here we report distributions of CFs found.\n"
    )

    summary_rows = []
    all_asset_counts = {}

    for f in cf_files:
        df = _load_details(f["details_path"])
        label = f["label"]

        n_queries = df["query_index"].nunique()
        n_assets = df["col_item"].nunique()
        mean_lift = df["lift"].mean() if "lift" in df.columns else float("nan")

        asset_counts = df.groupby("col_item")["query_index"].nunique()
        all_asset_counts[label] = asset_counts

        print(f"[{label}]")
        print(f"  Queries with CF  : {n_queries}")
        print(f"  Assets with CF   : {n_assets}")
        print(f"  Mean lift        : {mean_lift:.4f}")
        print(f"  CFs per asset    : mean={asset_counts.mean():.1f}  "
              f"median={asset_counts.median():.1f}  max={asset_counts.max()}")
        print(f"  Top 5 assets by CF count:")
        for asset, cnt in asset_counts.nlargest(5).items():
            print(f"    {asset}: {cnt}")

        summary_rows.append({
            "label": label,
            "n_queries": n_queries,
            "n_assets": n_assets,
            "mean_lift": mean_lift,
            "mean_cfs_per_asset": asset_counts.mean(),
        })

    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "feasibility_summary.csv"), index=False)

    # Plot: CF count distribution per asset for each experiment
    n = len(cf_files)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    for i, (f, (label, asset_counts)) in enumerate(zip(cf_files, all_asset_counts.items())):
        ax = axes[0, i]
        ax.hist(asset_counts.values, bins=30, color="#0181cb", alpha=0.8)
        ax.axvline(asset_counts.median(), color="red", linestyle="--", linewidth=1,
                   label=f"median={asset_counts.median():.0f}")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("CF count per asset")
        ax.set_ylabel("Number of assets")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.suptitle("Distribution of CF count per asset", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feasibility_cf_count_per_asset.png"), dpi=150)
    plt.close(fig)
    print("\nSaved feasibility plots and CSV.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Effort to improve
# ─────────────────────────────────────────────────────────────────────────────

def section_effort(cf_files, out_dir):
    """Distribution of effort metrics: how much price change is required."""
    print("\n" + "=" * 60)
    print("SECTION 2: EFFORT TO IMPROVE")
    print("=" * 60)

    effort_metrics = ["l1_dist", "l2_dist", "mean_abs_delta", "mean_rel_delta", "n_changed", "sparsity"]

    for f in cf_files:
        df = _load_details(f["details_path"])
        label = f["label"]

        print(f"\n[{label}]")
        available = [m for m in effort_metrics if m in df.columns]
        for m in available:
            vals = df[m].dropna()
            print(f"  {m:20s}: mean={vals.mean():.4f}  median={vals.median():.4f}  "
                  f"p25={vals.quantile(0.25):.4f}  p75={vals.quantile(0.75):.4f}")

        n_cols = 3
        n_rows = -(-len(available) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = axes.flatten()

        for i, metric in enumerate(available):
            ax = axes[i]
            vals = df[metric].dropna()
            ax.hist(vals, bins=30, color="#0181cb", alpha=0.8)
            ax.axvline(vals.median(), color="red", linestyle="--", linewidth=1,
                       label=f"median={vals.median():.3f}")
            ax.set_title(metric)
            ax.legend(fontsize=7)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
        for j in range(len(available), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Effort metrics — {label}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"effort_{_safe_label(label)}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved effort plot.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Score lift
# ─────────────────────────────────────────────────────────────────────────────

def section_lift(cf_files, out_dir):
    """Lift distribution and factual vs CF prediction scatter."""
    print("\n" + "=" * 60)
    print("SECTION 3: SCORE LIFT")
    print("=" * 60)

    for f in cf_files:
        df = _load_details(f["details_path"])
        label = f["label"]

        print(f"\n[{label}]")
        if "lift" in df.columns:
            lift = df["lift"].dropna()
            print(f"  lift: mean={lift.mean():.4f}  median={lift.median():.4f}  "
                  f"min={lift.min():.4f}  max={lift.max():.4f}")
            print(f"  lift > 0 (valid improvement): {(lift > 0).mean() * 100:.1f}%")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        if "lift" in df.columns:
            axes[0].hist(df["lift"].dropna(), bins=30, color="#ffbc42", alpha=0.8)
            axes[0].axvline(df["lift"].median(), color="red", linestyle="--", linewidth=1,
                            label=f"median={df['lift'].median():.3f}")
            axes[0].set_title("Lift (cf_pred - factual_pred)")
            axes[0].set_xlabel("Lift")
            axes[0].legend(fontsize=8)
            axes[0].grid(axis="y", linestyle="--", alpha=0.3)

        if "factual_prediction" in df.columns and "cf_prediction" in df.columns:
            axes[1].scatter(df["factual_prediction"], df["cf_prediction"],
                            alpha=0.3, s=8, color="#0181cb")
            lims = [df[["factual_prediction", "cf_prediction"]].min().min(),
                    df[["factual_prediction", "cf_prediction"]].max().max()]
            axes[1].plot(lims, lims, "k--", linewidth=1, label="y = x")
            axes[1].set_xlabel("Factual prediction")
            axes[1].set_ylabel("CF prediction")
            axes[1].set_title("Factual vs CF prediction")
            axes[1].legend(fontsize=8)
            axes[1].grid(linestyle="--", alpha=0.3)

        fig.suptitle(f"Score lift — {label}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"lift_{_safe_label(label)}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved lift plot.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Price pattern (delta per window position)
# ─────────────────────────────────────────────────────────────────────────────

def section_price_pattern(cf_files, out_dir):
    """Average price delta per window position — reveals what the model rewards."""
    print("\n" + "=" * 60)
    print("SECTION 4: PRICE PATTERN (what the model wants)")
    print("=" * 60)

    for f in cf_files:
        if f["summary_path"] is None:
            print(f"[{f['label']}] No summary file — skipping price pattern analysis.")
            continue

        label = f["label"]
        summary = _load_summary(f["summary_path"])

        factual = summary[summary["row_type"] == "factual"].copy().reset_index(drop=True)
        cf = summary[summary["row_type"] == "counterfactual"].copy().reset_index(drop=True)

        def parse_windows(df):
            parsed = df["window_line"].apply(json.loads)
            return pd.DataFrame(parsed.tolist(), index=df.index)

        try:
            factual_wins = parse_windows(factual)
            cf_wins = parse_windows(cf)
        except Exception as e:
            print(f"[{label}] Could not parse window_line JSON: {e}")
            continue

        # Align factual and CF windows by (query_index, cf_index)
        key_cols = ["query_index", "cf_index", "col_item"]
        f_keyed = pd.concat([factual[key_cols].reset_index(drop=True),
                              factual_wins.reset_index(drop=True)], axis=1)
        c_keyed = pd.concat([cf[key_cols].reset_index(drop=True),
                              cf_wins.reset_index(drop=True)], axis=1)
        merged = f_keyed.merge(c_keyed, on=key_cols, suffixes=("_f", "_c"))

        window_cols = sorted(factual_wins.columns.tolist(),
                             key=lambda x: int(x.split("_")[1]) if x.startswith("w_") else 0)

        deltas = pd.DataFrame()
        for col in window_cols:
            fc, cc = col + "_f", col + "_c"
            if fc in merged.columns and cc in merged.columns:
                deltas[col] = merged[cc] - merged[fc]

        if deltas.empty:
            print(f"[{label}] Could not compute deltas.")
            continue

        mean_delta = deltas.mean()
        std_delta = deltas.std()
        positions = range(len(mean_delta))

        print(f"\n[{label}]")
        print(f"  Position most increased : {mean_delta.idxmax()} (+{mean_delta.max():.4f})")
        print(f"  Position most decreased : {mean_delta.idxmin()} ({mean_delta.min():.4f})")
        print(f"  Positions with delta > 0: {(mean_delta > 0).sum()}/{len(mean_delta)}")
        print(f"  Overall direction       : {'upward' if mean_delta.mean() > 0 else 'downward'} "
              f"(mean={mean_delta.mean():.4f})")

        mean_delta.to_csv(
            os.path.join(out_dir, f"price_pattern_{_safe_label(label)}.csv"),
            header=["mean_delta"]
        )

        fig, axes = plt.subplots(2, 1, figsize=(12, 7))

        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in mean_delta]
        axes[0].bar(positions, mean_delta.values, color=colors, alpha=0.8)
        axes[0].fill_between(
            positions,
            mean_delta.values - std_delta.values,
            mean_delta.values + std_delta.values,
            alpha=0.15, color="gray", label="±1 std"
        )
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_xticks(list(positions))
        axes[0].set_xticklabels(window_cols, rotation=45, ha="right", fontsize=7)
        axes[0].set_ylabel("Mean price delta (CF − factual)")
        axes[0].set_title("Average price change per window position\n(green=increase, red=decrease)")
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="y", linestyle="--", alpha=0.3)

        pct_increase = (deltas > 0).mean() * 100
        pct_decrease = (deltas < 0).mean() * 100
        axes[1].bar(positions, pct_increase.values, label="% increased", color="#2ecc71", alpha=0.75)
        axes[1].bar(positions, -pct_decrease.values, label="% decreased", color="#e74c3c", alpha=0.75)
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_xticks(list(positions))
        axes[1].set_xticklabels(window_cols, rotation=45, ha="right", fontsize=7)
        axes[1].set_ylabel("% of queries")
        axes[1].set_title("Direction of change per window position")
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="y", linestyle="--", alpha=0.3)

        fig.suptitle(f"Price pattern — {label}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"price_pattern_{_safe_label(label)}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved price pattern plot and CSV.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Temporal comparison
# ─────────────────────────────────────────────────────────────────────────────

def _classify_lag_copies(merged, C, artifacts_dir, exp_date, model_tag, tol=1e-6, max_lag=100):
    """Flag which counterfactual windows are exact copies of the *same asset's own* real
    window ending 1..max_lag observations before the query — i.e. retrieved history rather
    than a genuinely searched/mutated candidate. Returns a bool array aligned with `merged`,
    or None if the training/testing price files for this window aren't available.
    """
    train_path = os.path.join(artifacts_dir, f"training_data_{exp_date}_00-00-00_{model_tag}.csv")
    test_path = os.path.join(artifacts_dir, f"testing_data_{exp_date}_00-00-00_{model_tag}.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        return None

    px = pd.concat([pd.read_csv(train_path), pd.read_csv(test_path)])
    px = px.drop_duplicates(["col_item", "col_timestamp", "col_rating"])
    px["col_timestamp"] = pd.to_datetime(px["col_timestamp"])
    px = px.sort_values(["col_item", "col_timestamp"])

    window_size = C.shape[1]
    ts = pd.to_datetime(merged["col_timestamp"])
    is_copy = np.zeros(len(merged), dtype=bool)

    for asset, s in px.groupby("col_item"):
        rows = merged.index[merged["col_item"] == asset]
        if len(rows) == 0:
            continue
        dates, vals = s["col_timestamp"].values, s["col_rating"].values.astype(float)
        if len(vals) < window_size + 1:
            continue
        windows = sliding_window_view(vals, window_size)
        pos = pd.Series(np.arange(len(dates)), index=dates)
        pos = pos[~pos.index.duplicated()]
        for r in rows:
            q = pos.get(ts.loc[r])
            if q is None:
                continue
            ks = np.arange(1, max_lag + 1)
            j = q - ks - (window_size - 1)
            ok = j >= 0
            if not ok.any():
                continue
            cand = windows[j[ok]]
            c = C[merged.index.get_loc(r)]
            rel = np.abs(cand - c).max(axis=1) / (np.abs(c).max() + 1e-12)
            if (rel < tol).any():
                is_copy[merged.index.get_loc(r)] = True
    return is_copy


def section_data_utility(cf_files, out_dir, artifacts_dir=None, model_tag=None):
    """Compare factual vs CF windows for distributional realism ("data utility"):
    does a counterfactual price path look like a real one, not just hit the target score?

    Stratified by whether the CF is an exact copy of the asset's own earlier window
    (retrieved real history — realism is guaranteed by construction) versus not
    (genuinely searched/mutated by DiCE — the only subset this actually tests).
    """
    print("\n" + "=" * 60)
    print("SECTION 8: DATA UTILITY (does a CF look like real price data?)")
    print("=" * 60)

    for f in cf_files:
        if f["summary_path"] is None:
            print(f"[{f['label']}] No summary file — skipping data utility analysis.")
            continue

        label = f["label"]
        summary = _load_summary(f["summary_path"])
        factual = summary[summary["row_type"] == "factual"].copy().reset_index(drop=True)
        cf = summary[summary["row_type"] == "counterfactual"].copy().reset_index(drop=True)

        def parse_windows(df):
            parsed = df["window_line"].apply(json.loads)
            return pd.DataFrame(parsed.tolist(), index=df.index)

        try:
            factual_wins = parse_windows(factual)
            cf_wins = parse_windows(cf)
        except Exception as e:
            print(f"[{label}] Could not parse window_line JSON: {e}")
            continue

        key_cols = ["query_index", "cf_index", "col_item", "col_timestamp"]
        f_keyed = pd.concat([factual[key_cols].reset_index(drop=True),
                              factual_wins.reset_index(drop=True)], axis=1)
        c_keyed = pd.concat([cf[key_cols].reset_index(drop=True),
                              cf_wins.reset_index(drop=True)], axis=1)
        merged = f_keyed.merge(c_keyed, on=key_cols, suffixes=("_f", "_c")).reset_index(drop=True)

        window_cols = sorted(factual_wins.columns.tolist(),
                             key=lambda x: int(x.split("_")[1]) if x.startswith("w_") else 0)
        if len(window_cols) < 2:
            print(f"[{label}] Window too short for returns — skipping.")
            continue

        F = merged[[c + "_f" for c in window_cols]].to_numpy(dtype=float)
        C = merged[[c + "_c" for c in window_cols]].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ret_f_all = np.diff(F, axis=1) / F[:, :-1]
            ret_c_all = np.diff(C, axis=1) / C[:, :-1]
        finite = np.isfinite(ret_f_all).all(axis=1) & np.isfinite(ret_c_all).all(axis=1)
        merged, F, C = merged[finite].reset_index(drop=True), F[finite], C[finite]
        ret_f_all, ret_c_all = ret_f_all[finite], ret_c_all[finite]
        if len(ret_f_all) == 0:
            print(f"[{label}] No valid windows after filtering — skipping.")
            continue

        # --- Terminal (query-date) price: real vs CF, pooled over ALL found CFs -----------
        # Not split by lag-copy: that distinction matters for window *shape* (below), but a
        # copy's terminal price is just as real a number as its window is, so splitting here
        # would only add a panel without adding information.
        price_f_all, price_c_all = F[:, -1], C[:, -1]
        price_rel_all = price_c_all / price_f_all - 1
        print(f"\n[{label}] Terminal price, all {len(price_f_all)} found CFs (real vs CF, pooled):")
        print(f"  median relative change {100 * np.median(price_rel_all):+.3f}%  |  "
              f"p10={100 * np.percentile(price_rel_all, 10):+.2f}%  "
              f"p90={100 * np.percentile(price_rel_all, 90):+.2f}%  |  "
              f"CF price higher than real: {100 * (price_rel_all > 0).mean():.1f}%")

        fig_p, axes_p = plt.subplots(1, 2, figsize=(13, 5))

        lo, hi = np.percentile(price_rel_all * 100, [0.5, 99.5])
        axes_p[0].hist(price_rel_all * 100, bins=np.linspace(lo, hi, 80), color="#16a085", alpha=0.85)
        axes_p[0].axvline(0, color="black", linewidth=1, linestyle="--", label="no change")
        axes_p[0].axvline(100 * np.median(price_rel_all), color="red", linewidth=1, linestyle="--",
                          label=f"median={100 * np.median(price_rel_all):+.2f}%")
        axes_p[0].set_xlabel("Relative price change, CF vs real (%)")
        axes_p[0].set_ylabel("Number of queries")
        axes_p[0].set_title("Terminal price change")
        axes_p[0].legend(fontsize=8)
        axes_p[0].grid(axis="y", linestyle="--", alpha=0.3)

        pmin, pmax = min(price_f_all.min(), price_c_all.min()), max(price_f_all.max(), price_c_all.max())
        axes_p[1].scatter(price_f_all, price_c_all, s=5, alpha=0.08, color="#16a085", linewidths=0)
        axes_p[1].plot([pmin, pmax], [pmin, pmax], color="black", linewidth=1, linestyle="--",
                      label="y = x (CF price = real price)")
        axes_p[1].set_xscale("log")
        axes_p[1].set_yscale("log")
        axes_p[1].set_xlim(pmin, pmax)
        axes_p[1].set_ylim(pmin, pmax)
        axes_p[1].set_xlabel("Real price on query date (log scale)")
        axes_p[1].set_ylabel("CF's implied price (log scale)")
        axes_p[1].set_title("Real vs CF price (log scale — see left panel for the real signal)")
        axes_p[1].legend(fontsize=8, loc="upper left")
        axes_p[1].grid(linestyle="--", alpha=0.3, which="both")
        axes_p[1].set_aspect("equal")

        fig_p.suptitle(f"Terminal price, before vs after — {label}", fontsize=13)
        fig_p.tight_layout()
        price_out_path = os.path.join(out_dir, f"data_utility_price_{_safe_label(label)}.png")
        fig_p.savefig(price_out_path, dpi=150)
        plt.close(fig_p)
        print(f"[{label}] Saved: {price_out_path}")

        # --- Window shape (returns / volatility): split by lag-copy, since that's where -----
        # retrieval vs. genuine search actually changes the answer.
        is_copy = None
        if artifacts_dir and model_tag:
            print(f"[{label}] Classifying lag copies against {artifacts_dir} (this can take a few minutes)...")
            is_copy = _classify_lag_copies(merged, C, artifacts_dir, f["exp_date"], model_tag)
            if is_copy is None:
                print(f"[{label}] Training/testing price files not found under {artifacts_dir} — "
                      "showing the unstratified (pooled) comparison instead.")

        groups = [("All found CFs", np.ones(len(merged), dtype=bool))] if is_copy is None else [
            ("Lag copies (retrieved history)", is_copy),
            ("Not a lag copy (searched/mutated)", ~is_copy),
        ]

        fig, axes = plt.subplots(2, len(groups), figsize=(6.5 * len(groups), 9), squeeze=False)

        for col, (gname, gmask) in enumerate(groups):
            ret_f, ret_c = ret_f_all[gmask], ret_c_all[gmask]
            n = gmask.sum()
            print(f"\n[{label}] {gname}: n = {n} ({100 * n / len(gmask):.1f}% of found CFs)")
            if n == 0:
                for row in (0, 1):
                    axes[row, col].set_title(f"{gname}\n(no rows)")
                continue

            pooled_f, pooled_c = ret_f.ravel(), ret_c.ravel()
            vol_f, vol_c = ret_f.std(axis=1), ret_c.std(axis=1)
            vol_diff = vol_c - vol_f  # paired: this query's CF minus this exact query's own factual
            print("  return quantiles (%), factual vs CF:")
            for q in (10, 25, 50, 75, 90):
                print(f"    p{q:<3d}: {100 * np.percentile(pooled_f, q):+.3f}  vs  {100 * np.percentile(pooled_c, q):+.3f}")
            print(f"  window volatility (std of within-window returns), median: "
                  f"factual={100 * np.median(vol_f):.3f}%  CF={100 * np.median(vol_c):.3f}%")
            print(f"  paired per-query difference (CF − its own factual), median: "
                  f"{100 * np.median(vol_diff):+.3f} pp  |  CF more volatile than its own query: "
                  f"{100 * (vol_diff > 0).mean():.1f}%")

            lo, hi = np.percentile(np.concatenate([pooled_f, pooled_c]), [0.5, 99.5])
            bins = np.linspace(lo, hi, 80)

            ax = axes[0, col]
            ax.hist(pooled_f, bins=bins, density=True, histtype="step", linewidth=1.8,
                    color="#0181cb", label="Factual")
            ax.hist(pooled_c, bins=bins, density=True, histtype="step", linewidth=1.8,
                    color="#ffbc42", label="Counterfactual")
            ax.set_xlabel("Day-over-day return within window")
            ax.set_ylabel("Density")
            ax.set_title(f"{gname} (n={n})")
            ax.legend(fontsize=8)
            ax.grid(linestyle="--", alpha=0.3)

            ax = axes[1, col]
            vmax = np.percentile(np.concatenate([vol_f, vol_c]), 99.5)
            ax.scatter(vol_f, vol_c, s=6, alpha=0.15, color="#8e44ad", linewidths=0)
            ax.plot([0, vmax], [0, vmax], color="black", linewidth=1, linestyle="--",
                    label="y = x (CF as volatile as its own query)")
            ax.set_xlim(0, vmax)
            ax.set_ylim(0, vmax)
            ax.set_xlabel("Factual volatility (this query)")
            ax.set_ylabel("Counterfactual volatility (this query's CF)")
            above = 100 * (vol_diff > 0).mean()
            ax.set_title(f"Window-level volatility, paired by query\n"
                         f"CF more volatile in {above:.1f}% of queries")
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(linestyle="--", alpha=0.3)
            ax.set_aspect("equal")

        fig.suptitle(f"Data utility — {label}", fontsize=13)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"data_utility_{_safe_label(label)}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\n[{label}] Saved: {out_path}")


def section_temporal(cf_files, out_dir):
    """Compare key metrics across experiment dates, grouped by method."""
    print("\n" + "=" * 60)
    print("SECTION 5: TEMPORAL COMPARISON")
    print("=" * 60)

    compare_metrics = ["lift", "l1_dist", "mean_rel_delta", "sparsity", "n_changed"]

    by_method = {}
    for f in cf_files:
        by_method.setdefault(f["method"], []).append(f)

    for method, files in by_method.items():
        if len(files) < 2:
            print(f"[{method}] Only one experiment date — skipping temporal comparison.")
            continue

        print(f"\n[method={method}]")
        rows = []
        for f in files:
            df = _load_details(f["details_path"])
            row = {"exp_date": f["exp_date"], "n_queries": df["query_index"].nunique(),
                   "n_assets": df["col_item"].nunique()}
            for m in compare_metrics:
                if m in df.columns:
                    row[f"{m}_mean"] = df[m].mean()
                    row[f"{m}_median"] = df[m].median()
            rows.append(row)
            print(f"  {f['exp_date']}: queries={row['n_queries']}, assets={row['n_assets']}, "
                  f"mean_lift={row.get('lift_mean', float('nan')):.4f}, "
                  f"mean_sparsity={row.get('sparsity_mean', float('nan')):.4f}")

        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(os.path.join(out_dir, f"temporal_{method}.csv"), index=False)

        available = [m for m in compare_metrics if f"{m}_mean" in summary_df.columns]
        n_cols = min(3, len(available))
        n_rows = -(-len(available) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        palette = ["#0181cb", "#ffbc42", "#2ecc71", "#e74c3c"]
        dates = summary_df["exp_date"].tolist()
        x = range(len(dates))

        for i, metric in enumerate(available):
            ax = axes[i]
            vals = summary_df[f"{metric}_mean"].values
            bars = ax.bar(x, vals, color=palette[:len(dates)], alpha=0.85)
            ax.set_xticks(list(x))
            ax.set_xticklabels(dates, rotation=20, ha="right")
            ax.set_title(f"{metric} (mean)")
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)
        for j in range(len(available), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Temporal comparison — {method}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"temporal_{method}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved temporal comparison plot and CSV.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Asset segmentation
# ─────────────────────────────────────────────────────────────────────────────

def section_segmentation(cf_files, out_dir):
    """Cluster assets by lift/effort profile and flag actionable vs structural."""
    print("\n" + "=" * 60)
    print("SECTION 6: ASSET SEGMENTATION")
    print("=" * 60)
    print(f"  Actionable = mean_rel_delta < {ACTIONABLE_REL_DELTA_THRESHOLD*100:.0f}% "
          f"AND sparsity < {ACTIONABLE_SPARSITY_THRESHOLD:.0%}\n")

    for f in cf_files:
        df = _load_details(f["details_path"])
        label = f["label"]

        needed = {"lift", "l1_dist", "sparsity", "mean_rel_delta"}
        if not needed.issubset(df.columns):
            print(f"[{label}] Missing columns for segmentation — skipping.")
            continue

        asset_stats = df.groupby("col_item").agg(
            lift=("lift", "mean"),
            l1_dist=("l1_dist", "mean"),
            sparsity=("sparsity", "mean"),
            mean_rel_delta=("mean_rel_delta", "mean"),
            max_rel_delta=("max_rel_delta", "mean"),
            n_queries=("query_index", "nunique"),
        ).reset_index()

        actionable = (
            (asset_stats["mean_rel_delta"] < ACTIONABLE_REL_DELTA_THRESHOLD) &
            (asset_stats["sparsity"] < ACTIONABLE_SPARSITY_THRESHOLD)
        )
        asset_stats["segment"] = "Structural shift"
        asset_stats.loc[actionable, "segment"] = "Actionable"

        n_actionable = actionable.sum()
        n_total = len(asset_stats)
        print(f"[{label}]")
        print(f"  Actionable : {n_actionable}/{n_total} ({100*n_actionable/n_total:.1f}%)")
        print(f"  Structural : {n_total - n_actionable}/{n_total}")
        print(f"  Top 5 actionable assets (lowest mean_rel_delta):")
        top5 = asset_stats[actionable].nsmallest(5, "mean_rel_delta")[
            ["col_item", "mean_rel_delta", "sparsity", "lift"]
        ]
        print(top5.to_string(index=False))

        out_csv = os.path.join(out_dir, f"segmentation_{_safe_label(label)}.csv")
        asset_stats.to_csv(out_csv, index=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Scatter: l1_dist vs lift, colored by sparsity
        sc = axes[0].scatter(
            asset_stats["l1_dist"], asset_stats["lift"],
            c=asset_stats["sparsity"], cmap="RdYlGn_r",
            alpha=0.7, s=40, vmin=0, vmax=1
        )
        plt.colorbar(sc, ax=axes[0], label="sparsity")
        axes[0].set_xlabel("Mean l1_dist (total price movement required)")
        axes[0].set_ylabel("Mean lift (score improvement)")
        axes[0].set_title("Effort vs lift (color = sparsity)")
        axes[0].grid(linestyle="--", alpha=0.3)

        # Scatter: sparsity vs mean_rel_delta with segment color
        seg_colors = {"Actionable": "#2ecc71", "Structural shift": "#e74c3c"}
        for seg, grp in asset_stats.groupby("segment"):
            axes[1].scatter(grp["sparsity"], grp["mean_rel_delta"],
                            label=f"{seg} (n={len(grp)})",
                            color=seg_colors[seg], alpha=0.7, s=40)
        axes[1].axvline(ACTIONABLE_SPARSITY_THRESHOLD, color="gray",
                        linestyle="--", linewidth=1, label=f"sparsity={ACTIONABLE_SPARSITY_THRESHOLD}")
        axes[1].axhline(ACTIONABLE_REL_DELTA_THRESHOLD, color="gray",
                        linestyle="--", linewidth=1,
                        label=f"rel_delta={ACTIONABLE_REL_DELTA_THRESHOLD*100:.0f}%")
        axes[1].set_xlabel("Sparsity (fraction of days changed)")
        axes[1].set_ylabel("Mean relative price change")
        axes[1].set_title("Actionability map")
        axes[1].legend(fontsize=8)
        axes[1].grid(linestyle="--", alpha=0.3)

        fig.suptitle(f"Asset segmentation — {label}", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"segmentation_{_safe_label(label)}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved segmentation plot and CSV.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Actionability
# ─────────────────────────────────────────────────────────────────────────────

def section_actionability(cf_files, out_dir):
    """CDF of mean_rel_delta and sparsity — how realistic are the CFs?"""
    print("\n" + "=" * 60)
    print("SECTION 7: ACTIONABILITY")
    print("=" * 60)

    rel_thresholds = [0.02, 0.05, 0.10, 0.20]
    sparsity_thresholds = [0.25, 0.50, 0.75, 1.00]

    all_rows = []
    for f in cf_files:
        df = _load_details(f["details_path"])
        label = f["label"]

        if "mean_rel_delta" not in df.columns or "sparsity" not in df.columns:
            print(f"[{label}] Missing columns — skipping.")
            continue

        print(f"\n[{label}]")
        print("  % CFs with mean_rel_delta below threshold:")
        for t in rel_thresholds:
            pct = (df["mean_rel_delta"] < t).mean() * 100
            print(f"    < {t*100:4.0f}%: {pct:.1f}%")
        print("  % CFs with sparsity at or below threshold:")
        for t in sparsity_thresholds:
            pct = (df["sparsity"] <= t).mean() * 100
            print(f"    <= {t:.2f}: {pct:.1f}%")

        all_rows.append((label, df))

    if not all_rows:
        return

    palette = ["#0181cb", "#ffbc42", "#2ecc71", "#e74c3c"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (label, df) in enumerate(all_rows):
        color = palette[i % len(palette)]

        vals = df["mean_rel_delta"].dropna().sort_values().values
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        axes[0].plot(vals, cdf, label=label, color=color, linewidth=1.8)

        vals_s = df["sparsity"].dropna().sort_values().values
        cdf_s = np.arange(1, len(vals_s) + 1) / len(vals_s)
        axes[1].plot(vals_s, cdf_s, label=label, color=color, linewidth=1.8)

    axes[0].axvline(ACTIONABLE_REL_DELTA_THRESHOLD, color="red", linestyle="--", linewidth=1,
                    label=f"actionable threshold ({ACTIONABLE_REL_DELTA_THRESHOLD*100:.0f}%)")
    axes[0].set_xlabel("Mean relative price change")
    axes[0].set_ylabel("Cumulative fraction of CFs")
    axes[0].set_title("CDF of mean relative delta")
    axes[0].legend(fontsize=8)
    axes[0].grid(linestyle="--", alpha=0.3)

    axes[1].axvline(ACTIONABLE_SPARSITY_THRESHOLD, color="red", linestyle="--", linewidth=1,
                    label=f"actionable threshold ({ACTIONABLE_SPARSITY_THRESHOLD:.0%})")
    axes[1].set_xlabel("Sparsity (fraction of window days changed)")
    axes[1].set_ylabel("Cumulative fraction of CFs")
    axes[1].set_title("CDF of sparsity")
    axes[1].legend(fontsize=8)
    axes[1].grid(linestyle="--", alpha=0.3)

    fig.suptitle("Actionability of CFs", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "actionability.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved actionability plot.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Counterfactual analysis — 7 diagnostic sections from CF output files."
    )
    parser.add_argument(
        "--cf-dir",
        default=os.path.join(CF_ROOT, _MODEL_TAG),
        help="Directory containing cf_details_*.csv and summary_*.csv files",
    )
    parser.add_argument(
        "--out-dir",
        default=OUT_ROOT,
        help="Output directory for analysis plots and CSVs",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=os.path.join("artifacts_for_counterfactuals", _MODEL_TAG),
        help="Directory with training_data_*.csv / testing_data_*.csv, used by the "
             "data-utility section to classify CFs that are exact copies of the asset's "
             "own earlier window (pass '' to skip and get the unstratified comparison)",
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        default=["all"],
        choices=["all", "feasibility", "effort", "lift", "pattern", "temporal", "segmentation", "actionability", "data-utility"],
        help="Which sections to run (default: all)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cf_files = _discover_cf_files(args.cf_dir)
    if not cf_files:
        raise ValueError(f"No cf_details_*.csv files found in {args.cf_dir}")

    print(f"Found {len(cf_files)} CF result file(s):")
    for f in cf_files:
        has_summary = "yes" if f["summary_path"] else "no"
        print(f"  [{f['label']}]  summary={has_summary}  {f['details_path']}")

    run_all = "all" in args.sections

    if run_all or "feasibility" in args.sections:
        section_feasibility(cf_files, args.out_dir)
    if run_all or "effort" in args.sections:
        section_effort(cf_files, args.out_dir)
    if run_all or "lift" in args.sections:
        section_lift(cf_files, args.out_dir)
    if run_all or "pattern" in args.sections:
        section_price_pattern(cf_files, args.out_dir)
    if run_all or "temporal" in args.sections:
        section_temporal(cf_files, args.out_dir)
    if run_all or "segmentation" in args.sections:
        section_segmentation(cf_files, args.out_dir)
    if run_all or "actionability" in args.sections:
        section_actionability(cf_files, args.out_dir)
    if run_all or "data-utility" in args.sections:
        section_data_utility(cf_files, args.out_dir, artifacts_dir=args.artifacts_dir or None, model_tag=_MODEL_TAG)

    print(f"\nAll outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
