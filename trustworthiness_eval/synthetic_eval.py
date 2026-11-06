"""
The evaluation on synthetic data: the counterfactual (CF) price series in place of the real prices.

For each saved recommendation date, the generator wrote one CF price per asset and day of the window's
test period (6 months before to 6 months after the recommendation date; cf_timeseries_*.csv). Those
prices replace the real close prices on the same asset-days; asset-days without a CF, and every day
outside that period, keep the real price. Customers and transactions are the real ones (the CFs do not
generate them). The saved model of each date is used as it is; nothing is retrained.

On that synthetic data, every metric of the protocol that does not need the 21-day CF windows themselves:
    accuracy      nDCG@k (with random / popularity baselines), monthly ROI@k, %prof@k, Vol@k, SR@k, market
                  references; R², RMSE and MAE of the 6-month return prediction on the test rows
    suitability   SVR@k filtered / unfiltered and sensitivity checks, risk-class stability
    fairness      group gaps, within-profile ROI gap, exposure (Gini, coverage), in-house ratio
    privacy       holdings inference from the recommendation lists
    robustness    the same market shocks as robustness.py (equity, rates, combined); calm vs stress
                  comes from the 61 time points of the metrics above
    monitoring    PSI of the model score and the features against the model's training data

--real runs the same code on the real prices (check that it reproduces the published run).

Usage (from the repo root):
    python trustworthiness_eval/synthetic_eval.py --dates all --jobs 8
    python trustworthiness_eval/synthetic_eval.py --dates 2020-08-28 --real
"""

import argparse
import copy
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "trustworthiness_eval", ROOT / "counterfactuals_generation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

pd.set_option("future.infer_string", False)

from compute_risk_metrics import profitability, schedule  # noqa: E402
from data.filter.asset.asset_with_test_price import AssetWithTestPrice  # noqa: E402
from data.filter.customer.customer_in_train import CustomerInTrain  # noqa: E402
from data.filter.data_filter import DataFilter  # noqa: E402
from data.filter.rating.ratings_not_in_train import RatingsNotInTrain  # noqa: E402
from data.filter.timeseries.no_filter import NoFilter  # noqa: E402
from data.financial_asset_time_series import FinancialAssetTimeSeries  # noqa: E402
from data.financial_data_continuous import FinancialContinuousData  # noqa: E402
from data.financial_interaction_data import FinancialInteractionData  # noqa: E402
from common import _load_model, model_test_accuracy  # noqa: E402
from metrics.suitability import Suitability  # noqa: E402
from monitoring import psi  # noqa: E402
from recommendation import compute_volatility  # noqa: E402
from reported_metrics import build_recs, holdings_values, window_values  # noqa: E402
from robustness import (BOND_FALL, EQUITY_FALL, EQUITY_VOL_MULT, asset_types, jaccard, scores, shock,  # noqa: E402
                        svr, top_lists)
from utils.constants import DEFAULT_ITEM_COL as ITEM  # noqa: E402
from utils.constants import DEFAULT_RATING_COL as PRICE  # noqa: E402
from utils.constants import DEFAULT_TIMESTAMP_COL as TS  # noqa: E402
from utils.constants import DEFAULT_USER_COL as USER  # noqa: E402


def cf_prices(cf_dir, tag, date):
    """One CF price per asset and day (the last step of that day's CF window). Non-positive CF prices
    (176 of 6.7 million) are not prices; those asset-days keep the real price."""
    c = pd.read_csv(cf_dir / f"cf_timeseries_{tag}_{date}_genetic.csv", on_bad_lines="skip")
    c[TS] = pd.to_datetime(c[TS], errors="coerce")
    c[PRICE] = pd.to_numeric(c[PRICE], errors="coerce")
    c = c.dropna(subset=[ITEM, TS, PRICE])
    c = c[c[PRICE] > 0]
    c[ITEM] = c[ITEM].astype(str)
    return c.drop_duplicates([ITEM, TS], keep="first").set_index([ITEM, TS])[PRICE]


def replace_prices(frame, cf):
    """Close prices with the CF price wherever one exists for the same asset and day."""
    out = frame.copy()
    key = pd.MultiIndex.from_arrays([out[ITEM].astype(str), pd.to_datetime(out[TS])])
    new = cf.reindex(key).to_numpy()
    hit = ~np.isnan(new)
    out.loc[hit, PRICE] = new[hit]
    return out, float(hit.sum())


def one_date(date, args, out_file):
    tag = args.model
    artifacts = ROOT / "artifacts_for_counterfactuals" / tag
    cf_dir = ROOT / "counterfactuals_results" / tag
    data_dir = ROOT / args.data_dir
    customer_info = data_dir / "customer_information.csv"
    rec = pd.Timestamp(date)

    data = FinancialContinuousData(FinancialInteractionData(str(data_dir / "transactions.csv")),
                                   FinancialAssetTimeSeries(str(data_dir / "close_prices.csv")))
    data.load()
    sched = schedule(data)
    fut = sched[rec]
    prev = max((d for d in sched if d < rec), default=None)

    stem = f"{date}_00-00-00_{tag}"
    history = pd.concat([pd.read_csv(artifacts / f"training_data_{stem}.csv"),
                         pd.read_csv(artifacts / f"testing_data_{stem}.csv")], ignore_index=True)
    history[TS] = pd.to_datetime(history[TS])
    history = history.drop_duplicates([ITEM, TS], keep="last").sort_values([ITEM, TS]).reset_index(drop=True)

    info = {"date": date, "prices": "real" if args.real else "synthetic"}
    if not args.real:
        cf = cf_prices(cf_dir, tag, date)
        syn = copy.copy(data.time_series)
        syn.data, n_close = replace_prices(data.time_series.data, cf)
        data.time_series = syn
        history, n_hist = replace_prices(history, cf)
        period = history[(history[TS] > rec - dt.timedelta(days=183)) & (history[TS] <= fut)]
        info.update({"cf_prices": len(cf), "close_prices_replaced": n_close,
                     "test_period_share_replaced": n_hist / max(len(period), 1)})

    split = data.split(rec - dt.timedelta(days=36525), rec, fut,
                       DataFilter(CustomerInTrain(), AssetWithTestPrice(), RatingsNotInTrain(), NoFilter(),
                                  False, True, False))
    model = _load_model(artifacts / f"profitability_recommendation_pipeline_{stem}.pkl")
    past = history[history[TS] <= rec].reset_index(drop=True)
    score = scores(model, past, rec)
    score = score[score.index.isin(list(split.assets))]

    # Accuracy, suitability, fairness, exposure, privacy -------------------------------------- #
    prof = profitability(split.time_series, rec, fut)
    vol = compute_volatility(split.time_series, rec, fut)
    recs = build_recs(score, split)
    vals = window_values(recs, split, rec, fut, prof, vol, prev, customer_info)
    vals.update(holdings_values(recs, data_dir, date))
    _, acc = model_test_accuracy(model, history[history[TS] <= fut].reset_index(drop=True), rec)
    vals.update(acc)
    metrics = [{**info, "metric": k, "value": float(v)} for k, v in vals.items()
               if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)]

    # Monitoring: PSI against the training data ----------------------------------------------- #
    feats = list(model.kpi_features)
    dev = pd.read_csv(artifacts / f"kpis_train_{stem}.csv")
    kpis = model._generate_kpis_df(past)
    cur = kpis[pd.to_datetime(kpis[TS]) == rec]
    importance = dict(zip(feats, getattr(model.model, "feature_importances_", np.zeros(len(feats)))))
    f_psi = {f: psi(dev[f], cur[f]) for f in feats if f in cur}
    top = sorted(feats, key=lambda f: -importance.get(f, 0))[:5]
    s_psi = psi(np.asarray(model.model.predict(dev[feats])).reshape(-1), score.values)
    top_max = max(f_psi[f] for f in top if f in f_psi)
    monitoring = [{**info, "psi_score": s_psi, "psi_feature_max": max(f_psi.values()),
                   "feature_max": max(f_psi, key=f_psi.get), "psi_top5_importance_max": top_max,
                   "review_triggered": bool(s_psi > 0.25 or top_max > 0.25)}]

    # Robustness: market shocks --------------------------------------------------------------- #
    users = split.users & set(split.test[USER].unique())
    held = split.train.groupby(USER)[ITEM].agg(set).to_dict()
    suit = Suitability(past, rec, customer_info)
    base_f, base_u = top_lists(score, held, users, suit), top_lists(score, held, users)
    equity, bonds = asset_types(data_dir)
    shocks = []
    for name, parts in [("equity", [(equity, EQUITY_FALL, EQUITY_VOL_MULT)]), ("rates", [(bonds, BOND_FALL, 1.0)]),
                        ("combined", [(equity, EQUITY_FALL, EQUITY_VOL_MULT), (bonds, BOND_FALL, 1.0)])]:
        shocked = past
        for assets, fall, mult in parts:
            shocked = shock(shocked, rec, assets, fall, mult)
        sc = scores(model, shocked, rec)
        sc = sc[sc.index.isin(list(split.assets))]
        s_suit = Suitability(shocked, rec, None, profiles=suit.profile)
        new_f, new_u = top_lists(sc, held, users, s_suit), top_lists(sc, held, users)
        jf, ju = jaccard(base_f, new_f), jaccard(base_u, new_u)
        common = set(suit.asset_class) & set(s_suit.asset_class)
        shocks.append({**info, "scenario": name, "j@10_filtered_mean": jf.mean(),
                       "j@10_filtered_share_unchanged": (jf == 1).mean(), "j@10_unfiltered_mean": ju.mean(),
                       "svr@10_shocked_filtered": svr(new_f, s_suit), "svr@10_stale_filtered": svr(base_f, s_suit),
                       "reclassified_up_share": float(np.mean([s_suit.asset_class[a] > suit.asset_class[a]
                                                               for a in common]))})

    pd.to_pickle({"metrics": metrics, "monitoring": monitoring, "shocks": shocks}, out_file)
    v = {m["metric"]: m["value"] for m in metrics}
    print(f"[{date} {info['prices']}] nDCG@10 {v.get('ndcg@10', np.nan):.4f}, monthly ROI@10 "
          f"{v.get('monthly_roi@10', np.nan):.4f}, SR@10 {v.get('sharpe@10', np.nan):.3f}, SVR filtered "
          f"{v.get('svr_filtered@10', np.nan):.3f}, R² {v.get('test_r2', np.nan):.3f}, PSI score {s_psi:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="rfr_n-193_leaf-96_depth-30_kpi-full_short_internal_kpis")
    parser.add_argument("--dates", default="all", help="comma-separated dates, or 'all'")
    parser.add_argument("--data-dir", default="FAR-Trans-Data")
    parser.add_argument("--real", action="store_true", help="real prices instead of the CF series (check)")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--output-dir", default="trustworthiness_eval/outputs")
    args = parser.parse_args()

    artifacts = ROOT / "artifacts_for_counterfactuals" / args.model
    cf_dir = ROOT / "counterfactuals_results" / args.model
    saved = sorted(p.name.split("_")[3] for p in artifacts.glob("profitability_recommendation_pipeline_*.pkl"))
    saved = [d for d in saved if (cf_dir / f"cf_timeseries_{args.model}_{d}_genetic.csv").exists()]
    dates = saved if args.dates == "all" else args.dates.split(",")

    out_dir = ROOT / args.output_dir / args.model / "synthetic"
    by_date = out_dir / ("by_date_real" if args.real else "by_date")
    by_date.mkdir(parents=True, exist_ok=True)
    todo = [d for d in dates if not (by_date / f"{d}.pkl").exists()]
    print(f"{len(dates)} dates, {len(dates) - len(todo)} already done, {len(todo)} to run", flush=True)
    Parallel(n_jobs=args.jobs, backend="loky")(delayed(one_date)(d, args, by_date / f"{d}.pkl") for d in todo)

    parts = [pd.read_pickle(by_date / f"{d}.pkl") for d in dates if (by_date / f"{d}.pkl").exists()]
    suffix = "_real" if args.real else ""
    for key in ("metrics", "monitoring", "shocks"):
        df = pd.DataFrame([r for p in parts for r in p[key]])
        df.to_csv(out_dir / f"synthetic_{key}{suffix}.csv", index=False)
        print(f"saved synthetic_{key}{suffix}.csv ({df['date'].nunique()} dates)")


if __name__ == "__main__":
    main()
