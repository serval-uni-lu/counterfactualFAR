"""
Adds SR@k, SVR@k and the suitability-filtered metrics to existing results, from the saved rankings
(<prefix>_<date>_recs.txt). Nothing is retrained or re-scored.

For every ranking file it:
  - rebuilds that window's split exactly as recommendation.py does,
  - evaluates the metrics from metrics/risk_metrics.py at the usual cutoffs,
  - appends them to <prefix>_<date>_metrics.csv (a copy of the original is kept once as
    _metrics.csv.bak, and every run rebuilds the file from it, so re-running never duplicates lines),
  - writes the customer-asset pairs that break suitability to <prefix>_<date>_svr_violations.csv
    and the per-group fairness values to <prefix>_<date>_fairness_groups.csv.

Usage (from the repo root):
    python compute_risk_metrics.py                      # every model folder under results/
    python compute_risk_metrics.py --models rfr --prefix rfr_tuned_full_short_internal_kpis
    python compute_risk_metrics.py --dry-run            # list what would be processed
"""

import argparse
import datetime as dt
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

pd.set_option("future.infer_string", False)  # the data loaders call .flatten() on string columns

from data.filter.asset.asset_with_test_price import AssetWithTestPrice
from data.filter.customer.customer_in_train import CustomerInTrain
from data.filter.data_filter import DataFilter
from data.filter.rating.ratings_not_in_train import RatingsNotInTrain
from data.filter.timeseries.no_filter import NoFilter
from data.financial_asset_time_series import FinancialAssetTimeSeries
from data.financial_data_continuous import FinancialContinuousData
from data.financial_interaction_data import FinancialInteractionData
from metrics.risk_metrics import risk_metrics
from utils.constants import DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_USER_COL

ROOT = Path(__file__).resolve().parent
# Same experiment definitions as run_recommendation.py: (min_date, max_date, num_splits, num_future).
EXPERIMENTS = {
    "exp1": ("2019-08-01", "2021-02-26", 28, 13),
    "exp2": ("2020-09-14", "2022-05-23", 31, 13),
}
CUTOFFS = [1, 5, 10, 20, 50, 100, 1000]
RECS_RE = re.compile(r"^(?P<prefix>.+)_(?P<date>\d{4}-\d{2}-\d{2})_recs\.txt$")


def schedule(data):
    """Recommendation date -> evaluation date, over both experiments."""
    out = {}
    for lo, hi, n_splits, n_future in EXPERIMENTS.values():
        dates, futures = data.get_dates(pd.Timestamp(lo), pd.Timestamp(hi), n_splits, n_future)
        for d, f in zip(dates, futures):
            d, f = pd.Timestamp(d), pd.Timestamp(f)
            if out.get(d, f) != f:
                raise ValueError(f"{d.date()} has two evaluation dates: {out[d].date()} and {f.date()}")
            out[d] = f
    return out


def profitability(time_series, rec, fut):
    """Raw return per asset between the two dates (compute_profitability in recommendation.py)."""
    p0 = time_series[time_series[DEFAULT_TIMESTAMP_COL] == rec].set_index(DEFAULT_ITEM_COL)[DEFAULT_RATING_COL]
    p1 = time_series[time_series[DEFAULT_TIMESTAMP_COL] == fut].set_index(DEFAULT_ITEM_COL)[DEFAULT_RATING_COL]
    both = p0.index.intersection(p1.index)
    return ((p1[both] - p0[both]) / p0[both]).to_dict()


def update_metrics_file(path, values):
    """Rewrites a tab-separated metrics file as its original lines (kept once in <file>.bak) plus
    `values`, so re-running never leaves lines from an earlier version of these metrics."""
    backup = path.with_name(path.name + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    keep = [line for line in backup.read_text().splitlines()
            if "\t" in line and line.split("\t", 1)[0] not in values]
    keep += [f"{k}\t{v}" for k, v in values.items()]
    path.write_text("\n".join(keep) + "\n")


def evaluate(recs, metrics, customers):
    values = {}
    for name, metric in metrics:
        res = metric.evaluate_cutoffs(recs, list(CUTOFFS), customers, True)
        for cutoff in CUTOFFS:
            if cutoff in res:
                values[f"{name}@{cutoff}"] = res[cutoff][1]
        values.update(getattr(metric, "reference_values", dict)())
    return dict(sorted(values.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--models", default=None, help="comma-separated model folders (default: all)")
    parser.add_argument("--prefix", default=None, help="only ranking files whose prefix equals this")
    parser.add_argument("--data-dir", default="FAR-Trans-Data")
    parser.add_argument("--risk-free", type=float, default=0.0, help="annual risk-free rate for SR@k")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results = ROOT / args.results_dir
    folders = [results / m for m in args.models.split(",")] if args.models else \
        [p for p in sorted(results.iterdir()) if p.is_dir()]
    by_date = defaultdict(list)
    for folder in folders:
        for path in sorted(folder.glob("*_recs.txt")):
            m = RECS_RE.match(path.name)
            if not m or (args.prefix and m["prefix"] != args.prefix):
                continue
            metrics_file = path.with_name(f"{m['prefix']}_{m['date']}_metrics.csv")
            if metrics_file.exists():
                by_date[pd.Timestamp(m["date"])].append((path, metrics_file))
    n_files = sum(len(v) for v in by_date.values())
    print(f"{n_files} ranking files over {len(by_date)} dates")
    if args.dry_run or not n_files:
        for d in sorted(by_date):
            for path, _ in by_date[d]:
                print(" ", path.parent.name + "/" + path.name)
        return

    data_dir = ROOT / args.data_dir
    data = FinancialContinuousData(FinancialInteractionData(str(data_dir / "transactions.csv")),
                                   FinancialAssetTimeSeries(str(data_dir / "close_prices.csv")))
    data.load()
    sched = schedule(data)
    customer_info = data_dir / "customer_information.csv"

    done = 0
    for rec in sorted(by_date):
        if rec not in sched:
            print(f"skip {rec.date()}: not a recommendation date of exp1/exp2")
            continue
        fut = sched[rec]
        split = data.split(rec - dt.timedelta(days=36525), rec, fut,
                           DataFilter(CustomerInTrain(), AssetWithTestPrice(), RatingsNotInTrain(), NoFilter(),
                                      False, True, False))
        prof = profitability(split.time_series, rec, fut)
        earlier = [d for d in sched if d < rec]
        # Built once per date: the metric objects only cache date-level data (risk classes, prices).
        metrics = risk_metrics(split, rec, fut, dict(prof), customer_info, args.risk_free,
                               prev_date=max(earlier) if earlier else None)
        for path, metrics_file in by_date[rec]:
            recs = pd.read_csv(path)
            # Saved sorted by customer, then score (best first): keep that order per customer.
            recs = recs.sort_values([DEFAULT_USER_COL, DEFAULT_RATING_COL], ascending=[False, False], kind="mergesort")
            values = evaluate(recs, metrics, split.users)
            update_metrics_file(metrics_file, values)
            for _, metric in metrics:
                for suffix, table in getattr(metric, "extra_tables", dict)().items():
                    table.to_csv(metrics_file.with_name(metrics_file.name.replace("_metrics.csv", f"_{suffix}.csv")),
                                 index=False)
            done += 1
            nan = float("nan")
            print(f"[{done}/{n_files}] {path.parent.name}/{path.name}: SR@10={values.get('sharpe@10', nan):.3f} "
                  f"(matched market {values.get('market_matched_sharpe', nan):.3f}), "
                  f"SVR@10={values.get('svr@10', nan):.3f}, filtered={values.get('svr_filtered@10', nan):.3f}, "
                  f"nDCG@10={values.get('ndcg_filtered@10', nan):.4f} filtered", flush=True)


if __name__ == "__main__":
    main()
