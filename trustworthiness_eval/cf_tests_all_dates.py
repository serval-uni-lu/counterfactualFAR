"""
Tests of the model that use the counterfactuals (CFs), at every saved recommendation date.

Only the CFs at the recommendation date that stay valid when re-scored end to end through the saved model
are used (one per asset: the 21-day price window that raises the asset's predicted return), with their
re-scored value. Four tests:

    stability    the top-10 lists rebuilt with each asset's CF score in place of its real score
                 (filtered by suitability, as customers receive them, and unfiltered): Jaccard J@10
                 against the real lists, share of customers whose list is unchanged, Kendall tau of the
                 two score vectors, and the share of single CFs that would bring their asset into the top-10.
    fairness     quality of the CFs that explain each customer's filtered top-10 (share of the list
                 explained, proximity, sparsity), compared across segment, capacity band and risk profile:
                 gap between the best and worst group, with a permutation test.
    privacy      whether a CF reveals a real past price window of the same asset (the generator seeds
                 its search from those windows): distance of each CF to the nearest past window, share
                 that copy one exactly or almost exactly, against the same distance for the real window.
                 The CFs contain no customer data.
    suitability  the asset's PRIIPs risk class recomputed with the CF prices in place of the real
                 last 21 days: share of CFs that move it to a higher / lower class, and share of the
                 explained assets in customers' filtered top-10 whose CF class is above the customer's limit.

Usage (from the repo root):
    python trustworthiness_eval/cf_tests_all_dates.py --dates all --jobs 8
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "trustworthiness_eval", ROOT / "counterfactuals_generation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

pd.set_option("future.infer_string", False)

from compute_risk_metrics import schedule  # noqa: E402
from data.filter.asset.asset_with_test_price import AssetWithTestPrice  # noqa: E402
from data.filter.customer.customer_in_train import CustomerInTrain  # noqa: E402
from data.filter.data_filter import DataFilter  # noqa: E402
from data.filter.rating.ratings_not_in_train import RatingsNotInTrain  # noqa: E402
from data.filter.timeseries.no_filter import NoFilter  # noqa: E402
from data.financial_asset_time_series import FinancialAssetTimeSeries  # noqa: E402
from data.financial_data_continuous import FinancialContinuousData  # noqa: E402
from data.financial_interaction_data import FinancialInteractionData  # noqa: E402
from common import (ITEM, PRICE, TS, USER, W, _load_model, asset_reference, customer_attributes,  # noqa: E402
                      load_cfs, price_history, rechecked_at_rec, window_stats)
from metrics.suitability import (MIN_RETURNS, TRADING_DAYS, VOL_WINDOW_YEARS, Suitability,  # noqa: E402
                                 risk_class)
from robustness import jaccard, top_lists  # noqa: E402

K = 10
ATTRIBUTES = {"customerType": "segment", "investmentCapacity": "capacity band", "riskLevel": "risk profile"}
MIN_GROUP = 10
N_PERM = 200
EXACT, NEAR = 1e-9, 0.001  # mean relative difference to a past window: exact copy / within 0.1%


def load_history(artifacts, tag, date, rec):
    stem = f"{date}_00-00-00_{tag}"
    h = pd.concat([pd.read_csv(artifacts / f"training_data_{stem}.csv"),
                   pd.read_csv(artifacts / f"testing_data_{stem}.csv")], ignore_index=True)
    h[TS] = pd.to_datetime(h[TS])
    h = h.drop_duplicates([ITEM, TS], keep="last")
    return h[h[TS] <= rec].sort_values([ITEM, TS]).reset_index(drop=True)


# --- stability ------------------------------------------------------------------------------- #

def stability(factual, cf_scores, has_cf, held, users, suit):
    rows = []
    for name, s in (("filtered", suit), ("unfiltered", None)):
        a, b = top_lists(factual, held, users, s), top_lists(cf_scores, held, users, s)
        j = jaccard(a, b)
        rows.append({"list": name, "j@10_mean": j.mean(), "j@10_median": np.median(j),
                     "share_customers_unchanged": (j == 1).mean()})
    rank = factual.rank(ascending=False, method="first")
    entering = [(rank[x] > K) and (1 + int((factual.drop(x) > cf_scores[x]).sum()) <= K) for x in has_cf]
    common = {"kendall_tau_scores": kendalltau(factual, cf_scores).correlation,
              "assets_scored": len(factual), "assets_with_valid_cf": len(has_cf),
              "coverage_assets": len(has_cf) / len(factual),
              "single_cf_share_entering_top10": float(np.mean(entering)) if entering else np.nan}
    return [{**r, **common} for r in rows]


# --- fairness of the explanations ------------------------------------------------------------ #

def perm_gap(values, groups, rng):
    """Gap between the highest and lowest group mean (groups of at least MIN_GROUP), and its
    permutation p-value."""
    sizes = pd.Series(groups).value_counts()
    keep = np.isin(groups, sizes[sizes >= MIN_GROUP].index)
    v, g = values[keep], groups[keep]
    if len(np.unique(g)) < 2:
        return None
    codes, uniq = pd.factorize(g)
    counts = np.bincount(codes)

    def gap(c):
        m = np.bincount(c, weights=v, minlength=len(uniq)) / counts
        return m.max() - m.min(), m

    obs, means = gap(codes)
    null = np.array([gap(rng.permutation(codes))[0] for _ in range(N_PERM)])
    return {"gap": obs, "perm_p": (1 + (null >= obs - 1e-12).sum()) / (N_PERM + 1), "n_groups": len(uniq),
            "min_group": uniq[means.argmin()], "max_group": uniq[means.argmax()]}


def explanation_fairness(lists, valid, stats_rec, attrs, rng):
    prox = stats_rec.drop_duplicates(ITEM).set_index(ITEM)["mean_rel_change"]
    nchg = stats_rec.drop_duplicates(ITEM).set_index(ITEM)["n_changed"]
    rows = []
    for u, items in lists.items():
        if not items:
            continue
        exp = [a for a in items if a in valid]
        rows.append({USER: u, "explained_share": len(exp) / len(items),
                     "proximity": np.nanmean([prox[a] for a in exp if a in prox.index]) if exp else np.nan,
                     "sparsity": np.mean([nchg[a] / W for a in exp if a in nchg.index]) if exp else np.nan})
    cust = pd.DataFrame(rows).set_index(USER).join(attrs, how="left")
    out = []
    for attr in ATTRIBUTES:
        for metric in ("explained_share", "proximity", "sparsity"):
            d = cust[cust[attr].notna() & (cust[attr] != "Not_Available") & cust[metric].notna()]
            r = perm_gap(d[metric].to_numpy(float), d[attr].to_numpy(), rng)
            if r:
                out.append({"attribute": attr, "metric": metric, "overall_mean": d[metric].mean(), **r})
    return out


# --- privacy --------------------------------------------------------------------------------- #

def nearest_past_window(history, rec, cfs):
    """Mean relative difference of each CF (and of its real window) to the closest real 21-day window
    of the same asset that ends before the recommendation date."""
    by_asset = {a: g[PRICE].to_numpy(float) for a, g in history[history[TS] < rec].groupby(ITEM)}
    rows = []
    for _, r in cfs.iterrows():
        p = by_asset.get(r[ITEM])
        if p is None or len(p) < W:
            continue
        past = np.lib.stride_tricks.sliding_window_view(p, W)
        with np.errstate(divide="ignore", invalid="ignore"):
            d_cf = np.nanmean(np.abs(past - r["cf_win"]) / np.abs(past), axis=1)
            d_f = np.nanmean(np.abs(past - r["fact_win"]) / np.abs(past), axis=1)
        rows.append({ITEM: r[ITEM], "cf_to_past": np.nanmin(d_cf), "fact_to_past": np.nanmin(d_f),
                     "cf_to_fact": r["mean_rel_change"]})
    d = pd.DataFrame(rows)
    return {"n_cfs": len(d),
            "cf_exact_copy_of_past_window": (d["cf_to_past"] < EXACT).mean(),
            "cf_within_0.1pct_of_past_window": (d["cf_to_past"] < NEAR).mean(),
            "real_within_0.1pct_of_past_window": (d["fact_to_past"] < NEAR).mean(),
            "cf_closer_to_past_window_than_to_real": (d["cf_to_past"] < d["cf_to_fact"]).mean(),
            "cf_to_past_median": d["cf_to_past"].median(), "real_to_past_median": d["fact_to_past"].median()}


# --- suitability ----------------------------------------------------------------------------- #

def cf_classes(history, rec, cfs):
    """PRIIPs class of each asset with its CF prices in place of the real last 21 days (same
    volatility rule as metrics.suitability)."""
    start = rec - pd.DateOffset(years=VOL_WINDOW_YEARS)
    ts = history[(history[TS] <= rec) & (history[TS] > start)]
    px = ts.pivot_table(index=TS, columns=ITEM, values=PRICE, aggfunc="last").sort_index()

    def vol(col):
        r = col.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        return r.std() * np.sqrt(TRADING_DAYS) if r.notna().sum() >= MIN_RETURNS else np.nan

    out = {}
    for _, r in cfs.iterrows():
        a = r[ITEM]
        if a not in px:
            continue
        col = px[a].copy()
        idx = col.dropna().index[-W:]
        if len(idx) < W:
            continue
        col.loc[idx] = r["cf_win"]
        v = vol(col)
        if not np.isnan(v):
            out[a] = int(risk_class(pd.Series([v])).iloc[0])
    return out


def suitability_under_cf(suit, cls_cf, lists):
    common = [a for a in cls_cf if a in suit.asset_class]
    up = [cls_cf[a] > suit.asset_class[a] for a in common]
    down = [cls_cf[a] < suit.asset_class[a] for a in common]
    pairs = [cls_cf[a] > suit.max_class[u] for u, items in lists.items() if u in suit.max_class
             for a in items if a in cls_cf]
    return {"n_cfs_classified": len(common), "cf_class_up_share": float(np.mean(up)) if up else np.nan,
            "cf_class_down_share": float(np.mean(down)) if down else np.nan,
            "explained_pairs": len(pairs),
            "explained_pairs_above_customer_limit": float(np.mean(pairs)) if pairs else np.nan}


# --- one date -------------------------------------------------------------------------------- #

def one_date(date, args, out_file):
    tag = args.model
    artifacts = ROOT / "artifacts_for_counterfactuals" / tag
    cf_dir = ROOT / "counterfactuals_results" / tag
    data_dir = ROOT / args.data_dir
    rec = pd.Timestamp(date)

    data = FinancialContinuousData(FinancialInteractionData(str(data_dir / "transactions.csv")),
                                   FinancialAssetTimeSeries(str(data_dir / "close_prices.csv")))
    data.load()
    fut = schedule(data)[rec]
    split = data.split(rec - dt.timedelta(days=36525), rec, fut,
                       DataFilter(CustomerInTrain(), AssetWithTestPrice(), RatingsNotInTrain(), NoFilter(),
                                  False, True, False))
    users = split.users & set(split.test[USER].unique())
    held = split.train.groupby(USER)[ITEM].agg(set).to_dict()

    history = load_history(artifacts, tag, date, rec)
    suit = Suitability(history, rec, data_dir / "customer_information.csv")
    attrs = customer_attributes(data_dir, rec)

    factual = pd.read_csv(artifacts / f"predictions_{date}_00-00-00_{tag}.csv").set_index(ITEM)["prediction"]
    factual = factual[factual.index.isin(list(split.assets))]

    q, _ = load_cfs(cf_dir, tag, date, on_bad_lines="skip")
    # Only CFs that stay valid when re-scored end to end through the saved model, with the re-scored value.
    model = _load_model(artifacts / f"profitability_recommendation_pipeline_{date}_00-00-00_{tag}.pkl")
    _, at_rec = rechecked_at_rec(model, price_history(artifacts, tag, date), q[q[TS] == rec])
    stats_rec = window_stats(at_rec, asset_reference(history, rec))
    valid = at_rec.drop_duplicates(ITEM).set_index(ITEM)
    has_cf = factual.index.intersection(valid.index)
    cf_scores = factual.copy()
    cf_scores.loc[has_cf] = valid.loc[has_cf, "cf_score"]
    cfs = stats_rec.drop_duplicates(ITEM)
    cfs = cfs[cfs[ITEM].isin(has_cf)]

    lists = top_lists(factual, held, users, suit)
    rng = np.random.default_rng(0)
    res = {
        "stability": [{"date": date, **r} for r in stability(factual, cf_scores, has_cf, held, users, suit)],
        "fairness": [{"date": date, **r} for r in explanation_fairness(lists, set(has_cf), cfs, attrs, rng)],
        "privacy": [{"date": date, **nearest_past_window(history, rec, cfs)}],
        "suitability": [{"date": date, **suitability_under_cf(suit, cf_classes(history, rec, cfs), lists)}],
    }
    pd.to_pickle(res, out_file)
    s, pv, su = res["stability"][0], res["privacy"][0], res["suitability"][0]
    print(f"[{date}] J@10 filtered {s['j@10_mean']:.3f}, CFs copying a past window {pv['cf_within_0.1pct_of_past_window']:.1%}, "
          f"class up {su['cf_class_up_share']:.1%}, explained pairs above limit "
          f"{su['explained_pairs_above_customer_limit']:.1%}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="rfr_n-193_leaf-96_depth-30_kpi-full_short_internal_kpis")
    parser.add_argument("--dates", default="all", help="comma-separated dates, or 'all'")
    parser.add_argument("--data-dir", default="FAR-Trans-Data")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--output-dir", default="trustworthiness_eval/outputs")
    args = parser.parse_args()

    artifacts = ROOT / "artifacts_for_counterfactuals" / args.model
    cf_dir = ROOT / "counterfactuals_results" / args.model
    saved = sorted(p.name.split("_")[3] for p in artifacts.glob("profitability_recommendation_pipeline_*.pkl"))
    saved = [d for d in saved if (cf_dir / f"summary_{args.model}_{d}_genetic.csv").exists()]
    dates = saved if args.dates == "all" else args.dates.split(",")

    out_dir = ROOT / args.output_dir / args.model / "all_dates"
    by_date = out_dir / "cf_tests_by_date"
    by_date.mkdir(parents=True, exist_ok=True)
    todo = [d for d in dates if not (by_date / f"{d}.pkl").exists()]
    print(f"{len(dates)} dates, {len(dates) - len(todo)} already done, {len(todo)} to run", flush=True)
    Parallel(n_jobs=args.jobs, backend="loky")(delayed(one_date)(d, args, by_date / f"{d}.pkl") for d in todo)

    parts = [pd.read_pickle(by_date / f"{d}.pkl") for d in dates if (by_date / f"{d}.pkl").exists()]
    for key in ("stability", "fairness", "privacy", "suitability"):
        df = pd.DataFrame([r for p in parts for r in p[key]])
        df.to_csv(out_dir / f"cf_test_{key}.csv", index=False)
        print(f"saved cf_test_{key}.csv ({df['date'].nunique()} dates)")


if __name__ == "__main__":
    main()
