"""
Holdings inference from the recommendation lists of this project's models.

Every customer is recommended the same asset ranking minus the assets they
already bought, so the gaps in a customer's list give those purchases away. The
observer modelled here sees a customer's top-k list and knows the global ranking
(any customer without purchases receives it unfiltered): every asset ranked above
the last one shown, but absent from the list, is inferred to be held.

Scope: this analysis only applies if the per-customer lists (the _recs.txt files,
or part of them) are shared with a party that does not already have the
customers' transactions. If the lists never leave the pipeline there is nothing
to observe and no exposure. The trained model does not create the risk either:
with the model and price data one can rebuild the global ranking, but not the
per-customer lists, since removing the held assets needs the transactions.

Takes the same --model tag and --dates as the membership inference attacks and
reads the lists recommendation.py wrote for that window
(<results>/<family>/<run>_<date>_recs.txt). Nothing is trained or re-scored: the
leak comes from the filtering step, not from the fitted model. The inferred
holdings are checked against the purchases in transactions.csv before the
recommendation date.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from trustworthiness_eval.privacy.membership_inference.regression_signals import available_dates
from utils.constants import DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_USER_COL

FULL_LIST = "full"


def results_prefix(model: str, results_dir: str = "results") -> str:
    """Run prefix recommendation.py used for this artifact model tag, e.g.
    rfr_n-100_kpi-full_short_internal_kpis -> results/rfr/rfr_100_full_short_internal_kpis."""
    match = re.fullmatch(r"([a-z]+)_n-([^_]+)(_.+)?_kpi-(.+)", model)
    if match is None:
        raise ValueError(f"Cannot derive the results prefix from model tag {model}; pass --results-prefix")
    family, n_estimators, tuned_params, kpi_tag = match.groups()
    # Only a tuned model carries extra hyperparameters in its artifact tag.
    name_n = "tuned" if tuned_params else n_estimators
    return str(Path(results_dir) / family / f"{family}_{name_n}_{kpi_tag}")


def load_lists(recs_path: Path):
    """Presence matrix (customers x assets, assets in global rank order) of the recommendation lists."""
    return presence_matrix(pd.read_csv(recs_path))


def presence_matrix(recs: pd.DataFrame):
    """Presence matrix of recommendation lists given as rows (customer, asset, score)."""
    # Ties are broken by asset id so the global order and every customer's order agree.
    recs = recs.sort_values([DEFAULT_RATING_COL, DEFAULT_ITEM_COL], ascending=[False, True])
    assets = recs[DEFAULT_ITEM_COL].drop_duplicates().tolist()
    customers = sorted(recs[DEFAULT_USER_COL].unique())

    present = np.zeros((len(customers), len(assets)), dtype=bool)
    rows = pd.Index(customers).get_indexer(recs[DEFAULT_USER_COL])
    cols = pd.Index(assets).get_indexer(recs[DEFAULT_ITEM_COL])
    present[rows, cols] = True
    return present, customers, assets


def load_holdings(data_dir: Path, date: str, customers: list, assets: list) -> np.ndarray:
    """Ground truth: which ranked assets each customer transacted before the recommendation date."""
    transactions = pd.read_csv(data_dir / "transactions.csv", usecols=["customerID", "ISIN", "timestamp"])
    transactions = transactions[transactions["timestamp"] < date]
    rows = pd.Index(customers).get_indexer(transactions["customerID"])
    cols = pd.Index(assets).get_indexer(transactions["ISIN"])
    known = (rows >= 0) & (cols >= 0)

    held = np.zeros((len(customers), len(assets)), dtype=bool)
    held[rows[known], cols[known]] = True
    return held


def infer_holdings(present: np.ndarray, k) -> np.ndarray:
    """Assets inferred as held from each customer's top-k list: the gaps above the last asset shown."""
    if k == FULL_LIST:
        return ~present
    # Global rank of the k-th asset shown to each customer (their last one when the list is shorter).
    shown = np.cumsum(present, axis=1)
    last_shown = np.argmax(shown >= np.minimum(k, shown[:, -1:]), axis=1)
    above_last = np.arange(present.shape[1])[None, :] < last_shown[:, None]
    return ~present & above_last


def inference_metrics(inferred: np.ndarray, held: np.ndarray) -> dict:
    """How much of the customers' holdings the inferred sets recover, and how reliably."""
    correct = inferred & held
    with_holdings = held.any(axis=1)
    return {
        "n_customers": int(len(held)),
        "n_customers_with_holdings": int(with_holdings.sum()),
        # Share of all (customer, held asset) pairs that are recovered.
        "holdings_recall": float(correct.sum() / held.sum()) if held.sum() else np.nan,
        # Share of inferred pairs that are real holdings.
        "holdings_precision": float(correct.sum() / inferred.sum()) if inferred.sum() else np.nan,
        "customers_exposed": float(correct.any(axis=1).mean()),
        "customers_fully_exposed": float((correct.sum(axis=1) == held.sum(axis=1))[with_holdings].mean())
        if with_holdings.any() else np.nan,
        "mean_holdings_revealed": float(correct.sum(axis=1).mean()),
        "mean_holdings": float(held.sum(axis=1).mean()),
    }


def run_for_date(prefix: str, date: str, cutoffs: list, data_dir: Path) -> list:
    present, customers, assets = load_lists(Path(f"{prefix}_{date}_recs.txt"))
    held = load_holdings(data_dir, date, customers, assets)

    rows = []
    for k in cutoffs:
        metrics = inference_metrics(infer_holdings(present, k), held)
        rows.append({"date": date, "k": k, **metrics})
        print(f"[{date}] k={k}: recall={metrics['holdings_recall']:.4f} precision={metrics['holdings_precision']:.4f} "
              f"customers exposed={metrics['customers_exposed']:.4f}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Infer customers' holdings from the recommendation lists of a fitted model.")
    parser.add_argument("--model", required=True, help="Model tag directory name under artifacts_for_counterfactuals/")
    parser.add_argument("--dates", type=str, default=None,
                        help="Comma-separated recommendation dates (YYYY-MM-DD). Defaults to every date with a fitted pkl for this model.")
    parser.add_argument("--results-prefix", type=str, default=None,
                        help="Run prefix of the recommendation.py outputs, without the date. Derived from --model by default.")
    parser.add_argument("--cutoffs", type=str, default=f"10,20,50,100,{FULL_LIST}",
                        help=f"Comma-separated list lengths the observer sees ('{FULL_LIST}' for the whole list).")
    parser.add_argument("--data-dir", type=str, default="FAR-Trans-Data")
    parser.add_argument("--output-dir", type=str, default="trustworthiness_eval/outputs")
    args = parser.parse_args()

    dates = args.dates.split(",") if args.dates else available_dates(args.model)
    if not dates:
        raise FileNotFoundError(f"No fitted pkl files found for model={args.model} under artifacts_for_counterfactuals/")
    prefix = args.results_prefix or results_prefix(args.model)
    cutoffs = [k if k == FULL_LIST else int(k) for k in args.cutoffs.split(",")]

    rows = []
    for date in dates:
        rows.extend(run_for_date(prefix, date, cutoffs, Path(args.data_dir)))
    per_date = pd.DataFrame(rows)

    # Each date weighted equally, as in the other evaluations.
    pooled = per_date.drop(columns="date").groupby("k", sort=False).mean().reset_index()
    pooled.insert(0, "date", "ALL")

    output_dir = Path(args.output_dir) / args.model / "all_dates"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([per_date, pooled], ignore_index=True).to_csv(output_dir / "holdings_inference.csv", index=False)

    print(f"\nMean across {len(dates)} dates:")
    print(pooled.drop(columns="date").to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved to {output_dir}/")


if __name__ == "__main__":
    main()
