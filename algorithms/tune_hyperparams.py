#!/usr/bin/python
"""Tune RFR/LGBM hyperparameters with Optuna against several expanding-window
calibration folds, then save the config that is stable across those folds (not
just the one that wins on a single window) for `recommendation.py`'s "tuned"
mode to pick up.

Usage (run from anywhere; repo root is added to sys.path below):
    python3 algorithms/tune_hyperparams.py <dataset_path> rfr [--n-trials 20] [--num-folds 4] [--robustness-lambda 0.5] [--calibration-months 3] [--min-train-days 75]
    python3 algorithms/tune_hyperparams.py <dataset_path> lgbm [--n-trials 20] [--num-folds 4] [--robustness-lambda 0.5] [--calibration-months 3] [--min-train-days 75]

Output:
    results/hyperparam_selection/{model}_full_short_optuna_results.csv        (every trial, incl. per-fold scores)
    results/hyperparam_selection/{model}_full_short_optuna_results_best.json  (best trial's params + fold metadata)
"""

import argparse
import datetime as dt
import json
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import numpy as np
import optuna
import pandas as pd

from algorithms.lgbm_kpi_model import LGBMKPIModel
from algorithms.rfr_kpi_model import RFRKPIModel
from algorithms.profitability_prediction import ProfitabilityPrediction
from data.filter.asset.asset_with_test_price import AssetWithTestPrice
from data.filter.customer.customer_in_train import CustomerInTrain
from data.filter.data_filter import DataFilter
from data.filter.rating.ratings_not_in_train import RatingsNotInTrain
from data.filter.timeseries.no_filter import NoFilter
from data.financial_asset_time_series import FinancialAssetTimeSeries
from data.financial_data_continuous import FinancialContinuousData
from data.financial_interaction_data import FinancialInteractionData
from metrics.kpi_monthly_evaluation_metric import MonthlyKPIEvaluationMetric
from recommendation import compute_profitability, full_short_kpis
from utils.constants import DEFAULT_TIMESTAMP_COL

RFR = "rfr"
LGBM = "lgbm"
KPI_TYPE = "full_short"

DEPLOYMENT_MONTHS = 6

# Earliest recommendation date ever reported by run_recommendation.py, across both
# experiments (exp1 min_date=2019-08-01, exp2 min_date=2020-09-14 — exp1's is
# earlier). No calibration fold's validation period may reach this date: doing so
# would mean a fold used to pick hyperparameters overlaps a window whose result
# later gets reported, leaking the tuning process into what should be a held-out test.
CALIBRATION_CUTOFF = "2019-08-01"

OUTPUT_DIR = os.path.join(_REPO_ROOT, "results", "hyperparam_selection")


def _load_data(dataset_path):
    """Load interactions/time-series only — no precomputed KPIs. Tuning uses
    RFRKPIModel/LGBMKPIModel (the "internal" KPI path)."""
    interactions_file = os.path.join(dataset_path, "transactions.csv")
    time_series_file = os.path.join(dataset_path, "close_prices.csv")

    interaction_data = FinancialInteractionData(interactions_file)
    time_series_data = FinancialAssetTimeSeries(time_series_file)

    data = FinancialContinuousData(interaction_data, time_series_data)
    data.load()

    return data


def _kpi_available_from(data):
    """Earliest timestamp for which the internal KPI model can generate features"""
    probe = RFRKPIModel(k=5, kpi_type=KPI_TYPE, kpi_features=full_short_kpis, n_estimators=1)
    kpis = probe._generate_kpis_df(data.time_series.data)
    return pd.Timestamp(kpis[DEFAULT_TIMESTAMP_COL].min())


def _snap_to_trading_day(date, trading_dates):
    """Snap a datetime to the nearest earlier trading day in the dataset"""
    idx = np.searchsorted(trading_dates, np.datetime64(date), side="right") - 1
    idx = max(idx, 0)
    return pd.Timestamp(trading_dates[idx])


def _fold_origins(data, kpi_available_from, cutoff_date, months, num_folds, min_train_days):
    """Compute the recommendation dates for each expanding-window calibration fold."""
    earliest_available = kpi_available_from
    exclusion_buffer_days = months * 30
    earliest_origin = earliest_available + dt.timedelta(days=exclusion_buffer_days + min_train_days)
    latest_origin = pd.Timestamp(cutoff_date) - pd.DateOffset(months=months)

    if earliest_origin >= latest_origin:
        raise ValueError(
            f"Not enough history for any calibration fold: earliest usable origin "
            f"({earliest_origin.date()}) is not before the latest allowed origin "
            f"({latest_origin.date()} = {cutoff_date.date()} - {months} months). "
            f"This already accounts for ProfitabilityPrediction.train()'s own "
            f"{exclusion_buffer_days}-day exclusion buffer plus --min-train-days "
            f"({min_train_days}) of real training history on top of it. "
            "Lower --min-train-days/--num-folds, or CALIBRATION_CUTOFF is too early relative to the data."
        )

    trading_dates = np.sort(pd.to_datetime(data.time_series.data[DEFAULT_TIMESTAMP_COL]).unique())

    if num_folds == 1:
        return [_snap_to_trading_day(latest_origin, trading_dates)]

    span_days = (latest_origin - earliest_origin).days
    min_days_between_folds = 14
    if span_days < min_days_between_folds * (num_folds - 1):
        raise ValueError(
            f"Calibration folds would be crammed into only {span_days} days "
            f"({earliest_origin.date()} to {latest_origin.date()}) for {num_folds} folds — "
            f"consecutive origins would be ~{span_days / (num_folds - 1):.1f} days apart, "
            f"nearly the same training data repeated {num_folds} times. Fix by lowering "
            f"--calibration-months (shrinks the exclusion buffer on both ends) while raising "
            f"--min-train-days by the same number of days you lowered the buffer, to keep "
            f"earliest_origin anchored at the same (already data-validated) date — see "
            f"--calibration-months/--min-train-days help text for the current recommended pair."
        )

    step = (latest_origin - earliest_origin) / (num_folds - 1)
    raw_origins = [(earliest_origin + i * step).normalize() for i in range(num_folds)]
    return [_snap_to_trading_day(o, trading_dates) for o in raw_origins]


def _build_fold(data, rec_date, future_date):
    """(splitted_data, rec_date, monthly_metric) for one expanding-window fold —
    train is all available data before rec_date, exactly like the real outer
    deployment protocol (see the module docstring for why expanding, not sliding)."""
    min_split_date = rec_date - dt.timedelta(days=36525)
    splitted_data = data.split(
        min_split_date, rec_date, future_date,
        DataFilter(CustomerInTrain(), AssetWithTestPrice(), RatingsNotInTrain(), NoFilter(), False, True, False),
    )
    profitability = compute_profitability(splitted_data.time_series, rec_date, future_date, None)
    monthly_metric = MonthlyKPIEvaluationMetric(splitted_data, profitability, (future_date - rec_date).days)
    return splitted_data, rec_date, monthly_metric


def _make_objective(model_id, folds, robustness_lambda, calibration_months):
    def objective(trial):
        # each fold call regenerates technical indicators from raw price windows itself.
        if model_id == RFR:
            n_estimators = trial.suggest_int("n_estimators", 10, 500)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 100)
            max_depth = trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20, 30, 50])

            def make_model():
                return RFRKPIModel(
                    k=5, kpi_type=KPI_TYPE, kpi_features=full_short_kpis,
                    n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_depth=max_depth,
                    random_state=42, n_jobs=-1,
                )
        else:
            n_estimators = trial.suggest_int("n_estimators", 10, 500)
            num_leaves = trial.suggest_int("num_leaves", 2, 100)
            min_child_samples = trial.suggest_int("min_child_samples", 5, 100)

            def make_model():
                return LGBMKPIModel(
                    k=5, kpi_type=KPI_TYPE, kpi_features=full_short_kpis,
                    n_estimators=n_estimators, num_leaves=num_leaves, min_child_samples=min_child_samples,
                    random_state=42, n_jobs=-1,
                )

        fold_scores = []
        for splitted_data, rec_date, monthly_metric in folds:
            algorithm = ProfitabilityPrediction(
                make_model(), splitted_data, calibration_months, full_short_kpis, -1, save_for_testing=False,
            )
            algorithm.train(rec_date)
            if not algorithm.is_fitted:
                raise RuntimeError(
                    f"Fold with rec_date={rec_date.date()} produced zero usable training rows "
                    f"(ProfitabilityPrediction never fit a model for it) — increase --min-train-days "
                    f"or reduce --num-folds so every calibration fold has real training data."
                )
            recs = algorithm.recommend(rec_date, splitted_data.users, False, True)

            cutoff_results = monthly_metric.evaluate_cutoffs(recs, [10], splitted_data.users, True)
            _, monthly_prof_10 = cutoff_results[10]
            fold_scores.append(float(monthly_prof_10))

        fold_scores = np.array(fold_scores, dtype=float)
        mean_score = float(fold_scores.mean())
        std_score = float(fold_scores.std())
        trial.set_user_attr("fold_scores", fold_scores.tolist())
        trial.set_user_attr("fold_mean", mean_score)
        trial.set_user_attr("fold_std", std_score)
        return mean_score - robustness_lambda * std_score

    return objective


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_path", help="Path to the FAR-Trans dataset directory")
    parser.add_argument("model", choices=[RFR, LGBM], help="Model to tune")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--num-folds", type=int, default=4,
                         help="Number of expanding-window calibration folds to average/penalize across.")
    parser.add_argument("--robustness-lambda", type=float, default=0.5,
                         help="Weight on across-fold std dev in the objective (mean - lambda*std): "
                              "higher favors configs that are stable across folds over ones that are "
                              "merely best on average.")
    parser.add_argument("--min-train-days", type=int, default=75,
                         help="Minimum REAL training history required for the earliest calibration fold, on top of the "
                              "months*30-day exclusion buffer.")
    parser.add_argument("--calibration-months", type=int, default=3,
                         help="Horizon used for calibration folds only. Ideally this would equal the real "
                              "deployment horizon (DEPLOYMENT_MONTHS=6). But this dataset's pre-cutoff "
                              "runway (~308 days) is too short for calibration_months=6: the exclusion buffer "
                              "(months*30 days, counted at both ends) alone exceeds it.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = _load_data(args.dataset_path)
    kpi_available_from = _kpi_available_from(data)
    cutoff_date = dt.datetime.strptime(CALIBRATION_CUTOFF, "%Y-%m-%d")
    origins = _fold_origins(data, kpi_available_from, cutoff_date, args.calibration_months, args.num_folds, args.min_train_days)

    print(f"KPI rows available from {kpi_available_from.date()} onward.")
    print(f"Calibration folds (all strictly before {CALIBRATION_CUTOFF}, the earliest reported window; "
          f"calibration_months={args.calibration_months}, deployment_months={DEPLOYMENT_MONTHS}):")
    folds = []
    for origin in origins:
        future_date = origin + pd.DateOffset(months=args.calibration_months)
        print(f"  train through {origin.date()} -> validate through {future_date.date()}")
        folds.append(_build_fold(data, origin, future_date))

    objective = _make_objective(args.model, folds, args.robustness_lambda, args.calibration_months)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.n_trials)

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
    trials_csv = os.path.join(OUTPUT_DIR, f"{args.model}_{KPI_TYPE}_optuna_results.csv")
    trials_df.to_csv(trials_csv, index=False)
    print(f"Wrote {len(trials_df)} trials to {trials_csv}")

    best_params = {
        "model": args.model,
        "kpi_type": KPI_TYPE,
        **study.best_trial.params,
        "tuning_method": "multi_fold_robust",
        "num_folds": args.num_folds,
        "robustness_lambda": args.robustness_lambda,
        "calibration_months": args.calibration_months,
        "fold_origins": [str(o.date()) for o in origins],
        "fold_scores": study.best_trial.user_attrs.get("fold_scores"),
    }
    best_json = os.path.join(OUTPUT_DIR, f"{args.model}_{KPI_TYPE}_optuna_results_best.json")
    with open(best_json, "w") as handle:
        json.dump(best_params, handle, indent=2)

    print(f"Best objective (mean - {args.robustness_lambda}*std) = {study.best_value:.6f}, "
          f"fold scores = {study.best_trial.user_attrs.get('fold_scores')}, params saved to {best_json}")


if __name__ == "__main__":
    main()
