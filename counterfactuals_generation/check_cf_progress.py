"""Report counterfactual-generation progress across all (model tag, window) combinations,
without re-running anything.

generate_counterfactuals.py --resume already tracks completion per window: every query
outcome (found / no_cf / skipped) writes a sentinel row to that window's
counterfactuals_results/{tag}/summary_{tag}_{date}_{method}.csv, keyed by query_index. This script
just reads those sentinel counts back and compares them against the number of queries in
each window's testing CSV, so you get one table for the whole sweep instead of eyeballing
each window's log/output individually.

Usage:
    python check_cf_progress.py                       # every model tag, every window
    python check_cf_progress.py --model-tag rfr_n-100_kpi-full_short_internal_kpis
    python check_cf_progress.py --method genetic       # default; matches the slurm script
    python check_cf_progress.py --incomplete-only      # only show windows not yet 100%
"""

import argparse

import pandas as pd

from generate_counterfactuals import (
    _ARTIFACTS_DIR,
    _PKL_PREFIX,
    DEFAULT_DICE_METHOD,
    _derive_data_paths,
    _derive_output_paths,
    _discover_model_tags,
    _discover_window_dates,
)


def _count_csv_rows(path) -> int:
    """Fast row count (excluding header) for the simple, unquoted testing CSVs."""
    with open(path, "rb") as fh:
        return sum(1 for _ in fh) - 1


def _count_done_queries(summary_path) -> int:
    """Number of distinct query_index values already written to a window's summary CSV."""
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        return 0
    try:
        df = pd.read_csv(summary_path, usecols=["query_index"])
    except (ValueError, pd.errors.EmptyDataError):
        return 0
    return int(df["query_index"].dropna().nunique())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-tag", type=str, nargs="+", default=None, metavar="TAG",
                         help="Restrict to these model tags. Defaults to every tag under artifacts_for_counterfactuals/.")
    parser.add_argument("--method", type=str, default=DEFAULT_DICE_METHOD, choices=["genetic", "random", "kdtree"],
                         help="CF method used when the outputs were generated (default: %(default)s).")
    parser.add_argument("--incomplete-only", action="store_true", help="Only print windows that are not yet 100%% done.")
    args = parser.parse_args()

    model_tags = args.model_tag if args.model_tag else _discover_model_tags()

    rows = []
    for tag in model_tags:
        tag_dir = _ARTIFACTS_DIR / tag
        for window_date in _discover_window_dates(tag):
            pkl_path = tag_dir / f"{_PKL_PREFIX}{window_date}_00-00-00_{tag}.pkl"
            if not pkl_path.exists():
                continue
            _, testing_path = _derive_data_paths(pkl_path)
            _, out_summary, _ = _derive_output_paths(pkl_path, args.method)

            if not testing_path.exists():
                rows.append({"model_tag": tag, "window": window_date, "done": 0, "total": None, "status": "MISSING testing CSV"})
                continue

            total = _count_csv_rows(testing_path)
            done = _count_done_queries(out_summary)
            pct = 100.0 * done / total if total else 0.0
            status = "DONE" if done >= total else ("IN PROGRESS" if done > 0 else "NOT STARTED")
            rows.append({"model_tag": tag, "window": window_date, "done": done, "total": total, "pct": pct, "status": status})

    result = pd.DataFrame(rows)
    if result.empty:
        print("No model tags / windows discovered.")
        return

    if args.incomplete_only:
        result = result[result["status"] != "DONE"]

    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(result.to_string(index=False, formatters={"pct": "{:.1f}%".format}))

    print()
    for tag, group in result.groupby("model_tag"):
        n_done = (group["status"] == "DONE").sum()
        print(f"{tag}: {n_done}/{len(group)} windows fully done")

    overall_done = (result["status"] == "DONE").sum()
    print(f"\nOverall: {overall_done}/{len(result)} windows fully done across {len(model_tags)} model tag(s).")


if __name__ == "__main__":
    main()
