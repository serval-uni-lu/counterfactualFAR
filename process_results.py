import argparse
import glob
import json
import os
import re
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_ROOT = "results"
STATS_DIR = "stats"

# These experiment ranges are defined in run_recommendation.py -> dates = [...]
# We classify windows by split date (the date in each metrics filename).
EXPERIMENT_SPLIT_RANGES = [
    ("exp1_2019-08-01_to_2021-02-26", datetime(2019, 8, 1), datetime(2020, 8, 28)),
    ("exp2_2020-09-14_to_2022-05-23", datetime(2020, 9, 14), datetime(2021, 11, 23)),
]


def extract_split_date_from_filename(file_path):
    name = os.path.basename(file_path)
    m = re.search(r"(\d{4}-\d{2}-\d{2})_metrics\.csv$", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d")
    except ValueError:
        return None


def assign_experiment_label(split_date):
    if split_date is None:
        return "unclassified"
    for label, min_d, max_d in EXPERIMENT_SPLIT_RANGES:
        if min_d <= split_date <= max_d:
            return label
    return "unclassified"


def load_window_metrics(results_folder, file_prefix):
    pattern = os.path.join(results_folder, f"{file_prefix}_*_metrics.csv")
    all_candidates = sorted(glob.glob(pattern))
    exact_name_re = re.compile(rf"^{re.escape(file_prefix)}_\d{{4}}-\d{{2}}-\d{{2}}_metrics\.csv$")
    metric_files = [
        path for path in all_candidates
        if exact_name_re.match(os.path.basename(path))
    ]

    dfs = []
    for path in metric_files:
        split_date = extract_split_date_from_filename(path)
        exp_label = assign_experiment_label(split_date)
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["metric", "value"]
        )
        df["window_file"] = os.path.basename(path)
        df["split_date"] = split_date
        df["experiment"] = exp_label
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No per-window metrics CSV files found in {results_folder} for prefix {file_prefix}")

    all_metrics = pd.concat(dfs, ignore_index=True)
    all_metrics["value"] = pd.to_numeric(all_metrics["value"], errors="coerce")
    all_metrics = all_metrics.dropna(subset=["value"])
    return all_metrics, metric_files


def discover_metric_prefixes(results_folder):
    metric_files = sorted(glob.glob(os.path.join(results_folder, "*_metrics.csv")))
    prefixes = set()

    for path in metric_files:
        name = os.path.basename(path)
        m = re.match(r"(.+)_\d{4}-\d{2}-\d{2}_metrics\.csv$", name)
        if m:
            prefixes.add(m.group(1))

    return sorted(prefixes)


def compute_full_stats_per_metric(all_metrics):
    return (
        all_metrics
        .groupby("metric", as_index=False)
        .agg(
            mean=("value", "mean"),
            median=("value", "median"),
            std=("value", "std"),
            min=("value", "min"),
            max=("value", "max"),
        )
        .sort_values("metric")
    )


def _sanitize_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")



def save_all_models_plots(all_run_stats_tables):
    if not all_run_stats_tables:
        return

    plot_root = os.path.join(STATS_DIR, "plots", "model")
    os.makedirs(plot_root, exist_ok=True)

    long_rows = []
    for run_prefix, stats_df in all_run_stats_tables:
        for _, row in stats_df.iterrows():
            long_rows.append(
                {
                    "run": run_prefix,
                    "metric": str(row["metric"]),
                    "mean": float(row["mean"]),
                    "median": float(row["median"]),
                    "std": float(row["std"]),
                    "min": float(row["min"]),
                    "max": float(row["max"]),
                }
            )

    all_df = pd.DataFrame(long_rows)
    if all_df.empty:
        return

    # For each metric, compare ALL model runs in one bar chart using mean values.
    metrics = sorted(all_df["metric"].unique().tolist())
    for metric_name in metrics:
        metric_df = all_df[all_df["metric"] == metric_name].copy().sort_values("run")
        os.makedirs(plot_root, exist_ok=True)
        out_file = os.path.join(plot_root, f"{_sanitize_filename(metric_name)}.png")

        fig, ax = plt.subplots(figsize=(12, 7.2))
        ax.bar(metric_df["run"], metric_df["mean"])
        ax.set_title(f"all_models | {metric_name} | mean", fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=65, labelsize=11)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_file, dpi=170)
        plt.close(fig)


CF_ROOT = "counterfactuals"
_MODEL_TAG = "rfr_n-100_kpi-full_short_internal_kpis"


def _plot_single_asset_timeseries(df, asset_id, summary_path, query_index, out_dir):
    """Plot factual vs CF window for each query of one asset."""
    if query_index is not None:
        df = df[df["query_index"] == query_index]
        if df.empty:
            print(f"query_index={query_index} not found for asset {asset_id} in {os.path.basename(summary_path)}, skipping")
            return

    exp_tag = re.search(r"\d{4}-\d{2}-\d{2}", os.path.basename(summary_path))
    exp_tag = exp_tag.group(0) if exp_tag else "unknown"

    for q_idx, group in df.groupby("query_index"):
        factual_row = group[group["row_type"] == "factual"]
        cf_rows = group[group["row_type"] == "counterfactual"]
        if factual_row.empty or cf_rows.empty:
            continue

        query_date = pd.to_datetime(factual_row.iloc[0]["col_timestamp"])
        factual_prices = list(json.loads(factual_row.iloc[0]["window_line"]).values())
        window_size = len(factual_prices)
        dates = [query_date - timedelta(days=(window_size - 1 - i)) for i in range(window_size)]

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(dates, factual_prices, label="Factual", color="#0181cb", linewidth=2, marker="o", markersize=3, alpha=0.9)
        for cf_i, (_, cf_row) in enumerate(cf_rows.iterrows()):
            cf_prices = list(json.loads(cf_row["window_line"]).values())
            ax.plot(dates, cf_prices, label=f"CF", color="#ffbc42", linewidth=1.5, marker="o", markersize=3, alpha=0.5)

        ax.set_title(f"Asset {asset_id} | query_index={q_idx} | query_date={query_date.date()}", fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("Price", fontsize=8)
        ax.legend(fontsize=7)
        ax.set_xticks(dates)
        ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates], rotation=45, ha="right", fontsize=6)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"cf_ts_{asset_id}_q{q_idx}_{exp_tag}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")


def _plot_timeseries_all_assets(details_files, out_dir, artifacts_dir=None):
    """For each experiment, plot full factual price series with CF points overlaid."""
    for path in details_files:
        exp_tag = re.search(r"\d{4}-\d{2}-\d{2}", os.path.basename(path))
        exp_tag = exp_tag.group(0) if exp_tag else "unknown"
        cf_df = pd.read_csv(path)
        cf_df["col_timestamp"] = pd.to_datetime(cf_df["col_timestamp"], format="mixed")

        # Load full testing time series for all factual points
        testing_df = None
        if artifacts_dir:
            testing_pattern = os.path.join(
                artifacts_dir, f"testing_data_{exp_tag}_00-00-00_*.csv"
            )
            testing_files = glob.glob(testing_pattern)
            if testing_files:
                testing_df = pd.read_csv(testing_files[0])
                testing_df["col_timestamp"] = pd.to_datetime(testing_df["col_timestamp"])
                testing_df = testing_df.sort_values(["col_item", "col_timestamp"])

        assets = list(cf_df["col_item"].unique())
        assets_per_page = 5
        asset_out_dir = os.path.join(out_dir, f"cf_timeseries_{exp_tag}")
        os.makedirs(asset_out_dir, exist_ok=True)

        n_saved = 0
        for page_i, page_start in enumerate(range(0, len(assets), assets_per_page)):
            out_path = os.path.join(asset_out_dir, f"page_{page_i + 1:03d}.png")
            if os.path.exists(out_path):
                continue

            page_assets = assets[page_start: page_start + assets_per_page]
            fig, axes = plt.subplots(len(page_assets), 1, figsize=(14, 4 * len(page_assets)), squeeze=False)

            for ax, asset in zip(axes[:, 0], page_assets):
                if testing_df is not None:
                    full_series = testing_df[testing_df["col_item"] == asset]
                    ax.plot(full_series["col_timestamp"], full_series["col_rating"],
                            label="Factual", color="#0181cb", linewidth=1.5)

                asset_cf = cf_df[cf_df["col_item"] == asset].sort_values("col_timestamp").reset_index(drop=True)
                if testing_df is not None:
                    factual_dates = set(testing_df[testing_df["col_item"] == asset]["col_timestamp"])
                else:
                    factual_dates = set()

                cf_ts_plot = []
                cf_val_plot = []
                for i, row in asset_cf.iterrows():
                    cf_ts_plot.append(row["col_timestamp"])
                    cf_val_plot.append(row["cf_rating"])
                    if i < len(asset_cf) - 1:
                        next_ts = asset_cf.loc[i + 1, "col_timestamp"]
                        between = [d for d in factual_dates
                                   if row["col_timestamp"] < d < next_ts]
                        if between:
                            cf_ts_plot.append(next_ts)
                            cf_val_plot.append(float("nan"))

                ax.plot(cf_ts_plot, cf_val_plot,
                        label="CF", color="#ffbc42", linewidth=1.5, marker="o", markersize=3)

                if testing_df is None:
                    ax.plot(asset_cf["col_timestamp"], asset_cf["factual_rating"],
                            label="Factual", color="#0181cb", linewidth=1.5, marker="o", markersize=3)

                ax.set_title(f"Asset {asset}")
                ax.set_ylabel("Price")
                ax.legend()
                ax.tick_params(axis="x", rotation=30)
                ax.grid(axis="y", linestyle="--", alpha=0.3)

            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            n_saved += 1

        n_pages = -(-len(assets) // assets_per_page)
        print(f"Saved {n_saved}/{n_pages} page(s) ({len(assets)} assets) to: {asset_out_dir}")


def _plot_price_timeseries_all_assets(details_files, out_dir, artifacts_dir=None):
    """For each experiment, plot full train+test price series per asset, 5 per page.

    Asset order matches the CF timeseries plots (order of first appearance in cf_details).
    Train and test segments are shown in different colours with a vertical separator.
    Output folder: out_dir/../prices/price_timeseries_{exp_tag}/
    """
    prices_root = os.path.join(os.path.dirname(out_dir), "prices")

    for path in details_files:
        exp_tag = re.search(r"\d{4}-\d{2}-\d{2}", os.path.basename(path))
        exp_tag = exp_tag.group(0) if exp_tag else "unknown"

        cf_df = pd.read_csv(path)
        assets = list(cf_df["col_item"].unique())  # same order as CF plots

        if not artifacts_dir:
            print(f"Skipping price plots for {exp_tag}: no artifacts_dir provided")
            continue

        train_files = glob.glob(os.path.join(artifacts_dir, f"training_data_{exp_tag}_*.csv"))
        test_files  = glob.glob(os.path.join(artifacts_dir, f"testing_data_{exp_tag}_*.csv"))
        if not train_files or not test_files:
            print(f"Skipping price plots for {exp_tag}: training/testing CSVs not found")
            continue

        train_df = pd.read_csv(train_files[0])
        test_df  = pd.read_csv(test_files[0])
        train_df["col_timestamp"] = pd.to_datetime(train_df["col_timestamp"])
        test_df["col_timestamp"]  = pd.to_datetime(test_df["col_timestamp"])

        asset_out_dir = os.path.join(prices_root, f"price_timeseries_{exp_tag}")
        os.makedirs(asset_out_dir, exist_ok=True)

        assets_per_page = 5
        n_saved = 0
        for page_i, page_start in enumerate(range(0, len(assets), assets_per_page)):
            out_path = os.path.join(asset_out_dir, f"page_{page_i + 1:03d}.png")
            if os.path.exists(out_path):
                continue

            page_assets = assets[page_start: page_start + assets_per_page]
            fig, axes = plt.subplots(len(page_assets), 1, figsize=(14, 4 * len(page_assets)), squeeze=False)

            for ax, asset in zip(axes[:, 0], page_assets):
                tr = train_df[train_df["col_item"] == asset].sort_values("col_timestamp")
                te = test_df[test_df["col_item"] == asset].sort_values("col_timestamp")

                if not tr.empty:
                    ax.plot(tr["col_timestamp"], tr["col_rating"],
                            color="#0181cb", linewidth=1.2, label="Train")
                if not te.empty:
                    ax.plot(te["col_timestamp"], te["col_rating"],
                            color="#ff6b35", linewidth=1.2, label="Test")

                if not tr.empty and not te.empty:
                    split_date = tr["col_timestamp"].max()
                    ax.axvline(split_date, color="grey", linewidth=0.8, linestyle="--")

                ax.set_title(f"{asset} | {exp_tag}", fontsize=8)
                ax.set_ylabel("Price", fontsize=7)
                ax.legend(fontsize=7)
                ax.tick_params(axis="x", rotation=30, labelsize=6)
                ax.tick_params(axis="y", labelsize=7)
                ax.grid(axis="y", linestyle="--", alpha=0.3)

            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            n_saved += 1

        n_pages = -(-len(assets) // assets_per_page)
        print(f"Saved {n_saved}/{n_pages} price page(s) ({len(assets)} assets) to: {asset_out_dir}")


def _plot_all_assets_summary(details_files, out_dir):
    """Aggregate CF metrics per experiment and plot distributions."""
    if not details_files:
        return

    proximity_metrics = ["l1_dist", "l2_dist", "mean_abs_delta", "mean_rel_delta"]
    realism_metrics = ["max_abs_delta", "max_rel_delta"]

    for path in details_files:
        exp_tag = re.search(r"\d{4}-\d{2}-\d{2}", os.path.basename(path))
        exp_tag = exp_tag.group(0) if exp_tag else "unknown"
        df = pd.read_csv(path)

        # Metric distributions — 3x3 grid
        all_metrics = proximity_metrics + realism_metrics
        n_cols = 3
        n_rows = -(-len(all_metrics) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

        for i, metric in enumerate(all_metrics):
            ax = axes[i // n_cols, i % n_cols]
            if metric in df.columns:
                ax.hist(df[metric].dropna(), bins=30)
                ax.set_title(metric)
                ax.grid(axis="y", linestyle="--", alpha=0.3)
        for i in range(len(all_metrics), n_rows * n_cols):
            axes[i // n_cols, i % n_cols].set_visible(False)

        summary_path = os.path.join(out_dir, f"cf_summary_{exp_tag}.png")
        if os.path.exists(summary_path):
            print(f"Skipping (exists): {summary_path}")
        else:
            fig.suptitle(f"CF metrics — all assets | {exp_tag}", fontsize=13)
            fig.tight_layout()
            fig.savefig(summary_path, dpi=150)
            print(f"Saved: {summary_path}")
        plt.close(fig)

        # Scatter: factual vs CF prediction
        scatter_path = os.path.join(out_dir, f"cf_scatter_{exp_tag}.png")
        if os.path.exists(scatter_path):
            print(f"Skipping (exists): {scatter_path}")
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(df["factual_prediction"], df["cf_prediction"], alpha=0.4, s=10)
            lims = [df[["factual_prediction", "cf_prediction"]].min().min(),
                    df[["factual_prediction", "cf_prediction"]].max().max()]
            ax.plot(lims, lims, "k--", linewidth=1)
            ax.set_xlabel("Factual prediction")
            ax.set_ylabel("CF prediction")
            ax.set_title(f"Factual vs CF prediction | {exp_tag}")
            ax.grid(linestyle="--", alpha=0.3)
            fig.tight_layout()
            fig.savefig(scatter_path, dpi=150)
            plt.close(fig)
            print(f"Saved: {scatter_path}")


def fill_no_cf_rows(cf_dir: str, artifacts_dir: str | None = None) -> None:
    """Add no_cf placeholder rows from summary into cf_details and cf_timeseries.

    Rows are matched by query_index — already-present indices are not duplicated.
    factual_rating / col_rating are populated from the testing data when artifacts_dir
    is provided; all CF metric columns are left as NaN.
    """
    for summary_path in sorted(glob.glob(os.path.join(cf_dir, "summary_*.csv"))):
        exp_tag = re.search(r"\d{4}-\d{2}-\d{2}", os.path.basename(summary_path))
        if not exp_tag:
            continue
        exp_tag = exp_tag.group(0)

        summary = pd.read_csv(summary_path)
        no_cf = summary[summary["row_type"] == "no_cf"][
            ["query_index", "col_item", "col_timestamp"]
        ].drop_duplicates("query_index").copy()
        if no_cf.empty:
            continue

        # Look up factual rating from testing data if available
        factual_rating: dict[int, float] = {}
        if artifacts_dir:
            test_files = glob.glob(
                os.path.join(artifacts_dir, f"testing_data_{exp_tag}_*.csv")
            )
            if test_files:
                testing = pd.read_csv(test_files[0])
                testing["col_timestamp"] = pd.to_datetime(testing["col_timestamp"])
                no_cf["col_timestamp"] = pd.to_datetime(no_cf["col_timestamp"])
                for _, row in no_cf.iterrows():
                    mask = (
                        (testing["col_item"] == row["col_item"]) &
                        (testing["col_timestamp"] == row["col_timestamp"])
                    )
                    matches = testing.loc[mask, "col_rating"]
                    if not matches.empty:
                        factual_rating[int(row["query_index"])] = float(matches.iloc[0])

        for details_path in sorted(glob.glob(os.path.join(cf_dir, f"cf_details_*{exp_tag}*.csv"))):
            df = pd.read_csv(details_path)
            new = no_cf[~no_cf["query_index"].isin(df["query_index"])].copy()
            if new.empty:
                continue
            for col in df.columns:
                if col not in new.columns:
                    new[col] = float("nan")
            if "factual_rating" in new.columns:
                new["factual_rating"] = new["query_index"].map(factual_rating)
            df = pd.concat([df, new[df.columns]], ignore_index=True)
            df = df.sort_values("query_index").reset_index(drop=True)
            df.to_csv(details_path, index=False)
            print(f"fill_no_cf: added {len(new)} rows to {os.path.basename(details_path)}")

        for ts_path in sorted(glob.glob(os.path.join(cf_dir, f"cf_timeseries_*{exp_tag}*.csv"))):
            df = pd.read_csv(ts_path)
            new = no_cf[~no_cf["query_index"].isin(df["query_index"])].copy()
            if new.empty:
                continue
            for col in df.columns:
                if col not in new.columns:
                    new[col] = float("nan")
            if "col_rating" in new.columns:
                new["col_rating"] = new["query_index"].map(factual_rating)
            df = pd.concat([df, new[df.columns]], ignore_index=True)
            df = df.sort_values("query_index").reset_index(drop=True)
            df.to_csv(ts_path, index=False)
            print(f"fill_no_cf: added {len(new)} rows to {os.path.basename(ts_path)}")


def sort_cf_outputs(cf_dir: str) -> None:
    """Sort all CF output CSVs by query_index in place."""
    patterns = ["cf_details_*.csv", "summary_*.csv", "cf_timeseries_*.csv"]
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(cf_dir, pattern))):
            df = pd.read_csv(path)
            if "query_index" not in df.columns:
                print(f"Skipping {os.path.basename(path)}: no query_index column")
                continue
            df = df.sort_values("query_index").reset_index(drop=True)
            df.to_csv(path, index=False)
            print(f"Sorted and saved: {path} ({len(df)} rows)")


def plot_cf_analysis(cf_dir: str, asset_id: str | None, query_index: int | None, out_dir: str, args=None) -> None:
    fill_no_cf_rows(cf_dir, artifacts_dir=args.artifacts_dir if args else None)
    details_files = sorted(glob.glob(os.path.join(cf_dir, "cf_details_*.csv")))
    summary_files = sorted(glob.glob(os.path.join(cf_dir, "summary_*.csv")))
    os.makedirs(out_dir, exist_ok=True)

    if asset_id is None:
        # Default: aggregate comparison across all assets
        if not details_files:
            raise ValueError(f"No cf_details_*.csv files found in {cf_dir}")
        for details_path in details_files:
            exp_tag = re.search(r"\d{4}-\d{2}-\d{2}", os.path.basename(details_path))
            exp_tag = exp_tag.group(0) if exp_tag else "unknown"
            df = pd.read_csv(details_path)
            print(f"[{exp_tag}] Found {df['query_index'].nunique()} queries across {df['col_item'].nunique()} assets")
        _plot_all_assets_summary(details_files, out_dir)
        _plot_timeseries_all_assets(details_files, out_dir, artifacts_dir=args.artifacts_dir)
        _plot_price_timeseries_all_assets(details_files, out_dir, artifacts_dir=args.artifacts_dir)
    else:
        # Per-asset window plot — asset-id and query-index must be specified
        if query_index is None:
            raise ValueError("--query-index is required when --asset-id is specified")
        if not summary_files:
            raise ValueError(f"No summary_*.csv files found in {cf_dir}")
        for summary_path in summary_files:
            df = pd.read_csv(summary_path)
            df = df[df["col_item"].astype(str) == str(asset_id)]
            if df.empty:
                print(f"Asset {asset_id} not found in {os.path.basename(summary_path)}, skipping")
                continue
            _plot_single_asset_timeseries(df, asset_id, summary_path, query_index, out_dir)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser("model", help="Aggregate model metrics and plots (default analysis)")

    cf_parser = subparsers.add_parser("cf", help="Counterfactual analysis plots")
    cf_parser.add_argument("--asset-id", default=None, help="Asset ID to plot (default: all assets)")
    cf_parser.add_argument("--query-index", type=int, default=None, help="Specific query index (default: all)")
    cf_parser.add_argument("--cf-dir", default=os.path.join(CF_ROOT, _MODEL_TAG), help="Directory with CF output files")
    cf_parser.add_argument("--out-dir", default=os.path.join(STATS_DIR, "plots", "cf"), help="Output directory for plots")
    cf_parser.add_argument("--artifacts-dir", default=os.path.join("artifacts_for_counterfactuals", _MODEL_TAG), help="Artifacts directory containing testing_data_*.csv files")
    cf_parser.add_argument("--sort", action="store_true", help="Sort CF output CSVs by query_index and save in place (no plots)")

    args = parser.parse_args()

    if args.mode == "cf":
        if args.sort:
            sort_cf_outputs(args.cf_dir)
            return
        plot_cf_analysis(args.cf_dir, args.asset_id, args.query_index, args.out_dir, args=args)
        return

    model_stats_dir = os.path.join(STATS_DIR, "model")
    os.makedirs(model_stats_dir, exist_ok=True)
    all_run_stats_tables = []

    model_folders = [
        name for name in sorted(os.listdir(RESULTS_ROOT))
        if os.path.isdir(os.path.join(RESULTS_ROOT, name)) and name != "stats"
    ]

    if not model_folders:
        raise ValueError(f"No model folders found in {RESULTS_ROOT}")

    for model_name in model_folders:
        model_folder = os.path.join(RESULTS_ROOT, model_name)
        run_prefixes = discover_metric_prefixes(model_folder)
        run_stats_tables = []

        if not run_prefixes:
            print(f"Skipping {model_name}: no matching per-window metrics files")
            continue

        for run_prefix in run_prefixes:
            try:
                all_metrics, metric_files = load_window_metrics(model_folder, run_prefix)
            except ValueError:
                print(f"Skipping {run_prefix}: no matching per-window metrics files")
                continue

            full_stats = compute_full_stats_per_metric(all_metrics)
            output_file = os.path.join(model_stats_dir, f"{run_prefix}.csv")
            full_stats.to_csv(output_file, index=False)
            run_stats_tables.append((run_prefix, full_stats))
            all_run_stats_tables.append((run_prefix, full_stats))

            # Per-experiment stats (including unclassified if present)
            for exp_label, exp_df in all_metrics.groupby("experiment"):
                exp_stats = compute_full_stats_per_metric(exp_df)
                exp_output_file = os.path.join(model_stats_dir, f"{run_prefix}_{exp_label}.csv")
                exp_stats.to_csv(exp_output_file, index=False)

            print(f"Model run: {run_prefix}")
            print(f"  Windows used: {len(metric_files)}")
            print(f"  Saved stats: {output_file}")
            print(f"  Saved per-experiment stats in: {model_stats_dir}")

        print(f"  Saved stats for model group: {model_name}")

    save_all_models_plots(all_run_stats_tables)
    print(f"Saved all-model plots in: {os.path.join(STATS_DIR, 'plots', 'model')}")


if __name__ == "__main__":
    main()
