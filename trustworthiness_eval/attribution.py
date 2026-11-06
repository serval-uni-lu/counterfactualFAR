"""
Feature attribution ("Finance metrics", Transparency and explainability) on the saved model of each
window. Nothing is retrained.

The model has no customer input: its score for an asset depends only on the asset's 30 price KPIs at
the recommendation date. Attributions are therefore per asset and date, and every customer who is
recommended an asset gets the same explanation for it. "recommended" rows below are the assets in at
least one customer's suitability-filtered top-10, weighted by the number of customers who receive them
(the doc's customer-asset pairs); "all" rows are every scored asset, unweighted.

    SHAP          exact TreeSHAP (tree-path-dependent) of the saved RandomForestRegressor on the KPI
                  rows scored at the recommendation date. base value + sum of SHAP = score (checked).
    stability     per asset: Spearman correlation of its 30 SHAP values, and Jaccard overlap of its
                  top-5 features, before vs after small perturbations of the price history (the model's
                  only input; perturbing customer data cannot change any score):
                      noise    i.i.d. multiplicative noise on every daily close (sigma 0.1%, 0.25%)
                      missing  2% of trading days dropped at random (the recommendation date is kept)
                  3 seeds each. The market shocks of robustness.py are run as context, not as a
                  stability test: a 30% fall should change the explanation.
    faithfulness  score change |f(x) - f(x')| when the n features with the largest |SHAP| are set to
                  their training median, against n random features (20 draws), n = 1, 3, 5, 10.
                  Faithful attributions give a top/random ratio well above 1.

Usage (from the repo root):
    python trustworthiness_eval/attribution.py                    # every saved window (~5 min each), into all_dates/
    python trustworthiness_eval/attribution.py --synthetic        # the same on the counterfactual series, into synthetic/

A run over some dates replaces those dates in the output files and keeps the others, so the dates can be
split over several runs, one after the other (parallel runs would overwrite each other's rows).
"""

import argparse
import copy
import datetime as dt
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import rankdata, spearmanr

pd.set_option("future.infer_string", False)

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "counterfactuals_generation", ROOT / "trustworthiness_eval"):
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
from robustness import BOND_FALL, EQUITY_FALL, EQUITY_VOL_MULT, asset_types, shock, top_lists  # noqa: E402
from utils.constants import DEFAULT_ITEM_COL as ITEM  # noqa: E402
from utils.constants import DEFAULT_RATING_COL as PRICE  # noqa: E402
from utils.constants import DEFAULT_TIMESTAMP_COL as TS  # noqa: E402
from utils.constants import DEFAULT_USER_COL as USER  # noqa: E402

K = 10
TOP_FEATURES = 5
NOISE_SIGMAS = (0.001, 0.0025)
MISSING_SHARE = 0.02
FAITH_NS = (1, 3, 5, 10)
FAITH_DRAWS = 20
WINDOWS = ("21d", "63d", "126d")
FEATURE_TYPES = {"past_profitability": "return", "roc": "return", "m": "return",
                 "volatility": "risk", "sharpe": "risk",
                 "avg_price": "price level", "exp_mean": "price level", "min": "price level", "max": "price level"}


def feature_group(feature):
    """(look-back window, feature type) of a KPI name such as 'volatility_63d'."""
    window = next((w for w in WINDOWS if feature.endswith("_" + w)), None)
    if window is None:  # MACD, rsi_14, dco_22
        return "oscillator", "oscillator"
    return window, FEATURE_TYPES.get(feature[: -len(window) - 1], "other")


def kpis_at(model, history, rec, assets):
    """KPI rows at the recommendation date for `assets`, indexed by asset, in the model's feature order."""
    kpis = model._generate_kpis_df(history)
    kpis = kpis[pd.to_datetime(kpis[TS]) == rec].drop_duplicates(ITEM).set_index(ITEM)
    return kpis.loc[kpis.index.isin(assets), list(model.kpi_features)].astype(float).dropna()


def add_noise(history, sigma, rng):
    out = history.copy()
    out[PRICE] = out[PRICE].to_numpy(dtype=float) * np.exp(rng.normal(0.0, sigma, len(out)))
    return out


def drop_days(history, rec, share, rng):
    keep = (rng.random(len(history)) >= share) | (history[TS] == rec)
    return history[keep].reset_index(drop=True)


def row_spearman(a, b):
    """Spearman correlation of each row of a with the same row of b."""
    ra, rb = rankdata(a, axis=1), rankdata(b, axis=1)
    ra, rb = ra - ra.mean(1, keepdims=True), rb - rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (ra * rb).sum(1) / den, np.nan)


def row_top_jaccard(a, b, n=TOP_FEATURES):
    ta, tb = np.argsort(-np.abs(a), axis=1)[:, :n], np.argsort(-np.abs(b), axis=1)[:, :n]
    return np.array([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(ta, tb)])


def summarise(values, weights):
    """Mean, median and 10th percentile over all assets (unweighted) and over recommended assets
    (weighted by number of customers)."""
    out = {}
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    for subset, w in (("all", np.ones(len(v), int)), ("recommended", weights)):
        rep = np.repeat(v[ok], w[ok])
        out.update({f"{subset}_mean": rep.mean() if len(rep) else np.nan,
                    f"{subset}_median": np.median(rep) if len(rep) else np.nan,
                    f"{subset}_p10": np.percentile(rep, 10) if len(rep) else np.nan})
    return out


def faithfulness(rf, X, sv, reference, weights, rng):
    base = rf.predict(X)
    order = np.argsort(-np.abs(sv), axis=1)
    rows_idx = np.arange(len(X))[:, None]
    out = []
    for n in FAITH_NS:
        top = X.copy()
        top[rows_idx, order[:, :n]] = reference[order[:, :n]]
        d_top = np.abs(rf.predict(top) - base)
        d_rand = np.zeros(len(X))
        for _ in range(FAITH_DRAWS):
            pick = np.argsort(rng.random(X.shape), axis=1)[:, :n]
            rand = X.copy()
            rand[rows_idx, pick] = reference[pick]
            d_rand += np.abs(rf.predict(rand) - base)
        d_rand /= FAITH_DRAWS
        for subset, w in (("all", np.ones(len(X))), ("recommended", weights.astype(float))):
            if w.sum() == 0:
                continue
            m_top, m_rand = np.average(d_top, weights=w), np.average(d_rand, weights=w)
            out.append({"n_features": n, "subset": subset,
                        "score_change_top_mean": m_top, "score_change_random_mean": m_rand,
                        "ratio_top_to_random": m_top / m_rand if m_rand > 0 else np.nan,
                        "share_assets_top_gt_random": np.average(d_top > d_rand, weights=w)})
    return out


def save_merged(df, path):
    """Writes df to path, keeping the rows of the dates df does not cover from the existing file."""
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old[~old["date"].astype(str).isin(df["date"].astype(str))], df], ignore_index=True)
    df.sort_values("date", kind="mergesort").to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="rfr_n-193_leaf-96_depth-30_kpi-full_short_internal_kpis")
    parser.add_argument("--dates", default="all", help="comma-separated dates, or 'all'")
    parser.add_argument("--seeds", type=int, default=3, help="repeats of each random perturbation")
    parser.add_argument("--data-dir", default="FAR-Trans-Data")
    parser.add_argument("--output-dir", default="trustworthiness_eval/outputs")
    parser.add_argument("--synthetic", action="store_true",
                        help="counterfactual price series in place of the real prices (see synthetic_eval.py)")
    args = parser.parse_args()

    artifacts = ROOT / "artifacts_for_counterfactuals" / args.model
    data_dir = ROOT / args.data_dir
    out_dir = ROOT / args.output_dir / args.model / ("synthetic" if args.synthetic else "all_dates")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = FinancialContinuousData(FinancialInteractionData(str(data_dir / "transactions.csv")),
                                   FinancialAssetTimeSeries(str(data_dir / "close_prices.csv")))
    data.load()
    sched = schedule(data)
    saved = sorted(p.name.split("_")[3] for p in artifacts.glob("profitability_recommendation_pipeline_*.pkl"))
    dates = saved if args.dates == "all" else args.dates.split(",")
    equity, bonds = asset_types(data_dir)
    customer_info = data_dir / "customer_information.csv"

    stability_rows, faith_rows = [], []
    real_prices = data.time_series
    for date in dates:
        rec = pd.Timestamp(date)
        if args.synthetic:
            from synthetic_eval import cf_prices, replace_prices
            cf = cf_prices(ROOT / "counterfactuals_results" / args.model, args.model, date)
            data.time_series = copy.copy(real_prices)
            data.time_series.data, _ = replace_prices(real_prices.data, cf)
        split = data.split(rec - dt.timedelta(days=36525), rec, sched[rec],
                           DataFilter(CustomerInTrain(), AssetWithTestPrice(), RatingsNotInTrain(), NoFilter(),
                                      False, True, False))
        users = split.users & set(split.test[USER].unique())
        held = split.train.groupby(USER)[ITEM].agg(set).to_dict()
        assets = set(split.assets)
        stem = f"{date}_00-00-00_{args.model}"
        history = pd.concat([pd.read_csv(artifacts / f"training_data_{stem}.csv"),
                             pd.read_csv(artifacts / f"testing_data_{stem}.csv")], ignore_index=True)
        history[TS] = pd.to_datetime(history[TS])
        history = history.drop_duplicates([ITEM, TS], keep="last")
        history = history[history[TS] <= rec].sort_values([ITEM, TS]).reset_index(drop=True)
        if args.synthetic:
            history, _ = replace_prices(history, cf)
        model = _load_model(artifacts / f"profitability_recommendation_pipeline_{stem}.pkl")
        rf, features = model.model, list(model.kpi_features)
        explainer = shap.TreeExplainer(rf)

        # Baseline: KPIs at the recommendation date, scores, SHAP, and who receives each asset.
        base = kpis_at(model, history, rec, assets)
        X = base.to_numpy()
        score = pd.Series(rf.predict(X), index=base.index)
        saved_pred = pd.read_csv(artifacts / f"predictions_{stem}.csv").set_index(ITEM)["prediction"]
        score_check = float((score - saved_pred.reindex(score.index)).abs().max())
        sv = explainer.shap_values(X)
        base_value = float(np.ravel(explainer.expected_value)[0])
        additivity = float(np.abs(sv.sum(1) + base_value - score.to_numpy()).max())
        suit = Suitability(history, rec, customer_info)
        lists = top_lists(score, held, users, suit)
        receivers = Counter(a for items in lists.values() for a in items)
        weights = np.array([receivers.get(a, 0) for a in base.index])
        print(f"[{date}] {len(base)} assets, {int((weights > 0).sum())} recommended to {len(lists)} customers; "
              f"scores match saved predictions (max diff {score_check:.1e}); "
              f"SHAP additivity error {additivity:.1e}", flush=True)

        # Global view, logged only: mean |SHAP| per feature vs the forest's impurity importance.
        mean_abs_all = np.abs(sv).mean(0)
        mean_abs_rec = np.average(np.abs(sv), axis=0, weights=weights) if weights.sum() else np.full(len(features), np.nan)
        impurity = rf.feature_importances_
        rho_imp = spearmanr(mean_abs_all, impurity).statistic
        top_shap = [features[i] for i in np.argsort(-mean_abs_rec)[:3]]
        print(f"  top features by mean |SHAP| (recommended): {', '.join(top_shap)}; "
              f"rank corr. with impurity importance {rho_imp:.2f}", flush=True)

        # Stability under small price perturbations, plus the market shocks as context.
        rng = np.random.default_rng(0)
        perturbations = [(f"noise_{s:g}", seed, lambda h, s=s: add_noise(h, s, rng))
                         for s in NOISE_SIGMAS for seed in range(args.seeds)]
        perturbations += [(f"missing_{MISSING_SHARE:g}", seed, lambda h: drop_days(h, rec, MISSING_SHARE, rng))
                          for seed in range(args.seeds)]
        perturbations += [
            ("shock_equity", 0, lambda h: shock(h, rec, equity, EQUITY_FALL, EQUITY_VOL_MULT)),
            ("shock_rates", 0, lambda h: shock(h, rec, bonds, BOND_FALL, 1.0)),
            ("shock_combined", 0, lambda h: shock(shock(h, rec, equity, EQUITY_FALL, EQUITY_VOL_MULT),
                                                    rec, bonds, BOND_FALL, 1.0)),
        ]
        base_top = set(score.nlargest(K).index)
        win21 = np.array([feature_group(f)[0] == "21d" for f in features])
        for name, seed, perturb in perturbations:
            new = kpis_at(model, perturb(history), rec, assets)
            common = base.index.intersection(new.index)
            pos = base.index.get_indexer(common)
            sv_new = explainer.shap_values(new.loc[common].to_numpy())
            new_score = pd.Series(rf.predict(new.loc[common].to_numpy()), index=common)
            w = weights[pos]
            row = {"date": date, "perturbation": name, "seed": seed, "assets": len(common),
                   "score_spearman": spearmanr(score.loc[common], new_score).statistic,
                   "j@10_unfiltered": len(base_top & set(new_score.nlargest(K).index)) / len(base_top | set(new_score.nlargest(K).index)),
                   "share_abs_shap_21d_before": np.abs(sv[pos][:, win21]).sum() / np.abs(sv[pos]).sum(),
                   "share_abs_shap_21d_after": np.abs(sv_new[:, win21]).sum() / np.abs(sv_new).sum()}
            row.update({f"spearman_{k}": v for k, v in summarise(row_spearman(sv[pos], sv_new), w).items()})
            row.update({f"top{TOP_FEATURES}_jaccard_{k}": v
                        for k, v in summarise(row_top_jaccard(sv[pos], sv_new), w).items()})
            stability_rows.append(row)
            print(f"  {name:15s} seed {seed}: SHAP Spearman recommended {row['spearman_recommended_mean']:.3f} "
                  f"(all {row['spearman_all_mean']:.3f}), top-{TOP_FEATURES} overlap "
                  f"{row[f'top{TOP_FEATURES}_jaccard_recommended_mean']:.3f}, score Spearman {row['score_spearman']:.3f}",
                  flush=True)

        # Faithfulness: replace the most-attributed features with their training median.
        reference = pd.read_csv(artifacts / f"kpis_train_{stem}.csv", usecols=features)[features].median().to_numpy()
        for r in faithfulness(rf, X, sv, reference, weights, np.random.default_rng(0)):
            faith_rows.append({"date": date, **r})
            if r["subset"] == "recommended":
                print(f"  faithfulness n={r['n_features']:2d}: top/random score change "
                      f"{r['ratio_top_to_random']:.2f}x, top > random for {r['share_assets_top_gt_random']:.0%}",
                      flush=True)

        # Write after every window so a long --dates all run keeps what it has done.
        for name, rows in (("stability", stability_rows), ("faithfulness", faith_rows)):
            save_merged(pd.DataFrame(rows), out_dir / f"attribution_{name}.csv")
    print(f"saved attribution_*.csv in {out_dir}")


if __name__ == "__main__":
    main()
