"""
Market-shock robustness ("Finance metrics", Robustness) on the saved model of each window. Nothing
is retrained: the saved pipeline re-scores every asset on modified price histories.

Scenarios (reference assumptions of the protocol, applied to the 20 trading days up to the
recommendation date; the asset type comes from the FAR-Trans category and sub-category):
    equity    stocks and equity funds fall 30% over 20 trading days, daily volatility doubled
    rates     bonds and bond funds fall 10% (+200 bp x reference duration of 5 years), spread over
              the same 20 days
    combined  both

Per window and scenario:
    J@10        Jaccard similarity of each customer's top-10 before and after the shock (distribution
                over customers), for the suitability-filtered and the unfiltered lists
    SVR@10      under the shock, with risk classes recomputed on the shocked prices:
                  shocked_filtered     the shocked list after the filter (must remain 0)
                  shocked_unfiltered   the shocked ranking before the filter
                  stale_filtered       the pre-shock filtered list judged by the shocked classes,
                                       i.e. what a system that does not re-classify would recommend
    reclassified_up  share of assets moved to a higher risk class by the shock

Usage (from the repo root):
    python trustworthiness_eval/robustness.py                         # last window of exp1 and exp2
    python trustworthiness_eval/robustness.py                         # every saved window
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("future.infer_string", False)

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "counterfactuals_generation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from compute_risk_metrics import schedule  # noqa: E402
from data.filter.asset.asset_with_test_price import AssetWithTestPrice  # noqa: E402
from data.filter.customer.customer_in_train import CustomerInTrain  # noqa: E402
from data.filter.data_filter import DataFilter  # noqa: E402
from data.filter.rating.ratings_not_in_train import RatingsNotInTrain  # noqa: E402
from data.filter.timeseries.no_filter import NoFilter  # noqa: E402
from data.financial_asset_time_series import FinancialAssetTimeSeries  # noqa: E402
from data.financial_data_continuous import FinancialContinuousData  # noqa: E402
from data.financial_interaction_data import FinancialInteractionData  # noqa: E402
from generate_counterfactuals import _load_model  # noqa: E402
from metrics.suitability import Suitability  # noqa: E402
from utils.constants import DEFAULT_ITEM_COL as ITEM  # noqa: E402
from utils.constants import DEFAULT_RATING_COL as PRICE  # noqa: E402
from utils.constants import DEFAULT_TIMESTAMP_COL as TS  # noqa: E402
from utils.constants import DEFAULT_USER_COL as USER  # noqa: E402

K = 10
SHOCK_DAYS = 20
EQUITY_FALL, EQUITY_VOL_MULT = 0.30, 2.0
BOND_FALL = 0.02 * 5  # duration x 2%, reference duration 5 years
LAST_WINDOWS = ["2020-08-28", "2021-11-23"]


def asset_types(data_dir):
    info = pd.read_csv(data_dir / "asset_information.csv").sort_values("timestamp").drop_duplicates("ISIN", keep="last")
    cat, sub = info["assetCategory"], info["assetSubCategory"].fillna("")
    equity = set(info.loc[(cat == "Stock") | ((cat == "MTF") & (sub == "Equity")), "ISIN"])
    bonds = set(info.loc[(cat == "Bond") | ((cat == "MTF") & sub.isin(["Bond", "Bonds"])), "ISIN"])
    return equity, bonds


def shock(history, rec, assets, fall, vol_mult):
    """Rewrites the last SHOCK_DAYS daily log returns (up to rec) of `assets`: deviations from their
    mean are multiplied by vol_mult and the total move is set to a fall of `fall`."""
    out = history.copy()
    drift = np.log(1.0 - fall) / SHOCK_DAYS
    for a, idx in out[out[ITEM].isin(assets) & (out[TS] <= rec)].groupby(ITEM).groups.items():
        idx = list(idx)[-(SHOCK_DAYS + 1):]
        p = out.loc[idx, PRICE].to_numpy(dtype=float)
        if len(p) < SHOCK_DAYS + 1 or (p <= 0).any():
            continue
        lr = np.diff(np.log(p))
        new = vol_mult * (lr - lr.mean()) + drift
        out.loc[idx[1:], PRICE] = p[0] * np.exp(np.cumsum(new))
    return out


def scores(model, history, rec):
    preds = np.asarray(model.predict(history)).reshape(-1)
    kpis = getattr(getattr(model, "transformer", None), "last_kpis_df_", None)
    if kpis is None:
        kpis = model._generate_kpis_df(history)
    kpis = kpis[[ITEM, TS]].copy()
    kpis["score"] = preds
    at = kpis[pd.to_datetime(kpis[TS]) == rec]
    return at.drop_duplicates(ITEM).set_index(ITEM)["score"]


def top_lists(score, held, users, suit=None, k=K):
    order = list(score.sort_values(ascending=False, kind="mergesort").index)
    out = {}
    for u in users:
        h = held.get(u, ())
        cap = suit.max_class.get(u) if suit is not None else None
        lst = []
        for a in order:
            if a in h:
                continue
            if suit is not None and (cap is None or suit.asset_class.get(a, 99) > cap):
                continue
            lst.append(a)
            if len(lst) == k:
                break
        out[u] = lst
    return out


def svr(lists, suit):
    vals = [np.mean([suit.asset_class.get(a, 0) > suit.max_class[u] for a in items])
            for u, items in lists.items() if u in suit.max_class and items]
    return float(np.mean(vals)) if vals else np.nan


def jaccard(a, b):
    vals = []
    for u in a:
        s, t = set(a[u]), set(b.get(u, []))
        if s or t:
            vals.append(len(s & t) / len(s | t))
    return np.array(vals)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="rfr_n-193_leaf-96_depth-30_kpi-full_short_internal_kpis")
    parser.add_argument("--dates", default="all", help="comma-separated dates, or 'all'")
    parser.add_argument("--data-dir", default="FAR-Trans-Data")
    parser.add_argument("--output-dir", default="trustworthiness_eval/outputs")
    args = parser.parse_args()

    artifacts = ROOT / "artifacts_for_counterfactuals" / args.model
    data_dir = ROOT / args.data_dir
    data = FinancialContinuousData(FinancialInteractionData(str(data_dir / "transactions.csv")),
                                   FinancialAssetTimeSeries(str(data_dir / "close_prices.csv")))
    data.load()
    sched = schedule(data)
    saved = sorted(p.name.split("_")[3] for p in artifacts.glob("profitability_recommendation_pipeline_*.pkl"))
    dates = saved if args.dates == "all" else args.dates.split(",")
    equity, bonds = asset_types(data_dir)
    customer_info = data_dir / "customer_information.csv"

    rows = []
    for date in dates:
        rec = pd.Timestamp(date)
        fut = sched[rec]
        split = data.split(rec - dt.timedelta(days=36525), rec, fut,
                           DataFilter(CustomerInTrain(), AssetWithTestPrice(), RatingsNotInTrain(), NoFilter(),
                                      False, True, False))
        users = split.users & set(split.test[USER].unique())
        held = split.train.groupby(USER)[ITEM].agg(set).to_dict()
        stem = f"{date}_00-00-00_{args.model}"
        history = pd.concat([pd.read_csv(artifacts / f"training_data_{stem}.csv"),
                             pd.read_csv(artifacts / f"testing_data_{stem}.csv")], ignore_index=True)
        history[TS] = pd.to_datetime(history[TS])
        history = history.drop_duplicates([ITEM, TS], keep="last")
        history = history[history[TS] <= rec].sort_values([ITEM, TS]).reset_index(drop=True)
        model = _load_model(artifacts / f"profitability_recommendation_pipeline_{stem}.pkl")

        base_score = scores(model, history, rec)
        base_score = base_score[base_score.index.isin(list(split.assets))]
        saved_pred = pd.read_csv(artifacts / f"predictions_{stem}.csv").set_index(ITEM)["prediction"]
        check = float((base_score - saved_pred.reindex(base_score.index)).abs().max())
        suit = Suitability(history, rec, customer_info)
        base_f = top_lists(base_score, held, users, suit)
        base_u = top_lists(base_score, held, users)
        print(f"[{date}] baseline reproduces saved scores (max diff {check:.1e}); "
              f"SVR@10 filtered {svr(base_f, suit):.3f}, unfiltered {svr(base_u, suit):.3f}", flush=True)

        for name, parts in [("equity", [(equity, EQUITY_FALL, EQUITY_VOL_MULT)]),
                            ("rates", [(bonds, BOND_FALL, 1.0)]),
                            ("combined", [(equity, EQUITY_FALL, EQUITY_VOL_MULT), (bonds, BOND_FALL, 1.0)])]:
            shocked = history
            for assets, fall, mult in parts:
                shocked = shock(shocked, rec, assets, fall, mult)
            sc = scores(model, shocked, rec)
            sc = sc[sc.index.isin(list(split.assets))]
            s_suit = Suitability(shocked, rec, None, profiles=suit.profile)
            new_f = top_lists(sc, held, users, s_suit)
            new_u = top_lists(sc, held, users)
            jf, ju = jaccard(base_f, new_f), jaccard(base_u, new_u)
            common = set(suit.asset_class) & set(s_suit.asset_class)
            rows.append({
                "date": date, "scenario": name,
                "assets_shocked": len([a for a in base_score.index if a in set().union(*[p[0] for p in parts])]),
                "j@10_filtered_mean": jf.mean(), "j@10_filtered_median": np.median(jf),
                "j@10_filtered_p10": np.percentile(jf, 10), "j@10_filtered_share_unchanged": (jf == 1).mean(),
                "j@10_unfiltered_mean": ju.mean(), "j@10_unfiltered_median": np.median(ju),
                "svr@10_shocked_filtered": svr(new_f, s_suit), "svr@10_shocked_unfiltered": svr(new_u, s_suit),
                "svr@10_stale_filtered": svr(base_f, s_suit),
                "reclassified_up_share": float(np.mean([s_suit.asset_class[a] > suit.asset_class[a] for a in common])),
                "baseline_max_abs_score_diff": check,
            })
            r = rows[-1]
            print(f"  {name:8s} J@10 filtered {r['j@10_filtered_mean']:.3f} (unfiltered {r['j@10_unfiltered_mean']:.3f}), "
                  f"SVR@10 shocked filtered {r['svr@10_shocked_filtered']:.3f}, stale list {r['svr@10_stale_filtered']:.3f}, "
                  f"reclassified up {r['reclassified_up_share']:.1%}", flush=True)

    out_dir = ROOT / args.output_dir / args.model / "all_dates"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "robustness_market_shocks.csv", index=False)
    print(f"saved {out_dir / 'robustness_market_shocks.csv'}")


if __name__ == "__main__":
    main()
