import glob
import os
import re
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_ROOT = "results"
STATS_DIR = os.path.join(RESULTS_ROOT, "stats")
SUMMARY_COLUMNS = ["mean", "median", "std", "min", "max"]

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


def _run_label(model_name, run_prefix):
    prefix = f"{model_name}_"
    if run_prefix.startswith(prefix):
        return run_prefix[len(prefix):]
    return run_prefix


def save_model_comparison_plots(model_name, run_stats_tables):
    if not run_stats_tables:
        return

    plot_root = os.path.join(STATS_DIR, "plots", model_name)
    metrics = sorted(
        {
            metric
            for _, stats_df in run_stats_tables
            for metric in stats_df["metric"].astype(str).tolist()
        }
    )

    for metric_name in metrics:
        rows = []
        for run_prefix, stats_df in run_stats_tables:
            row = stats_df[stats_df["metric"] == metric_name]
            if row.empty:
                continue
            row = row.iloc[0]
            rows.append(
                {
                    "label": _run_label(model_name, run_prefix),
                    "mean": float(row["mean"]),
                    "median": float(row["median"]),
                    "std": float(row["std"]),
                    "min": float(row["min"]),
                    "max": float(row["max"]),
                }
            )

        if not rows:
            continue

        metric_df = pd.DataFrame(rows).sort_values("label")

        for summary_col in SUMMARY_COLUMNS:
            out_dir = os.path.join(plot_root, summary_col)
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"{_sanitize_filename(metric_name)}.png")

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(metric_df["label"], metric_df[summary_col])
            ax.set_title(f"{model_name} | {metric_name} | {summary_col}")
            ax.set_xlabel("Model parameters")
            ax.set_ylabel(summary_col)
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_file, dpi=160)
            plt.close(fig)


def save_all_models_plots(all_run_stats_tables):
    if not all_run_stats_tables:
        return

    plot_root = os.path.join(STATS_DIR, "plots")
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

    # Compute common y-scale per metric family (e.g., volatility@1..@1000 share one scale).
    all_df["metric_family"] = all_df["metric"].astype(str).str.split("@").str[0]
    family_y_limits = {}
    for family, family_df in all_df.groupby("metric_family"):
        y_min = float(family_df["mean"].min())
        y_max = float(family_df["mean"].max())
        if y_min == y_max:
            pad = max(0.05, abs(y_min) * 0.05)
        else:
            pad = 0.08 * (y_max - y_min)
        family_y_limits[family] = (y_min - pad, y_max + pad)

    # For each metric, compare ALL model runs in one bar chart using mean values.
    metrics = sorted(all_df["metric"].unique().tolist())
    for metric_name in metrics:
        metric_df = all_df[all_df["metric"] == metric_name].copy().sort_values("run")
        os.makedirs(plot_root, exist_ok=True)
        out_file = os.path.join(plot_root, f"{_sanitize_filename(metric_name)}.png")
        metric_family = str(metric_name).split("@")[0]

        fig, ax = plt.subplots(figsize=(12, 7.2))
        ax.bar(metric_df["run"], metric_df["mean"])
        ax.set_title(f"all_models | {metric_name} | mean", fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if metric_family in family_y_limits:
            ax.set_ylim(*family_y_limits[metric_family])
        ax.tick_params(axis="x", rotation=65, labelsize=11)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_file, dpi=170)
        plt.close(fig)


def main():
    os.makedirs(STATS_DIR, exist_ok=True)
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
            output_file = os.path.join(STATS_DIR, f"{run_prefix}.csv")
            full_stats.to_csv(output_file, index=False)
            run_stats_tables.append((run_prefix, full_stats))
            all_run_stats_tables.append((run_prefix, full_stats))

            # Per-experiment stats (including unclassified if present)
            for exp_label, exp_df in all_metrics.groupby("experiment"):
                exp_stats = compute_full_stats_per_metric(exp_df)
                exp_output_file = os.path.join(STATS_DIR, f"{run_prefix}_{exp_label}.csv")
                exp_stats.to_csv(exp_output_file, index=False)

            print(f"Model run: {run_prefix}")
            print(f"  Windows used: {len(metric_files)}")
            print(f"  Saved stats: {output_file}")
            print(f"  Saved per-experiment stats in: {STATS_DIR}")

        print(f"  Saved stats for model group: {model_name}")

    save_all_models_plots(all_run_stats_tables)
    print(f"Saved all-model plots in: {os.path.join(STATS_DIR, 'plots')}")


if __name__ == "__main__":
    main()
