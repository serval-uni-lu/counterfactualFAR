#!/usr/bin/python
"""Tune RFR/LGBM hyperparameters once with Optuna, against a single fixed window,
then save the best params for `recommendation.py`'s "tuned" mode to pick up.

The tuning window is the first split of experiment 1 (2019-08-01) — the same
(rec_date, future_date) pair already evaluated for every baseline config in
results/rfr/*_2019-08-01_metrics.csv and results/lgbm/*_2019-08-01_metrics.csv.
This keeps the tuned config comparable to the untuned ones on at least that one
window, while the objective itself (monthly_prof@10, i.e. ROI) is evaluated fresh
per trial.


Usage (run from anywhere; repo root is added to sys.path below):
    python3 algorithms/tune_hyperparams.py <dataset_path> rfr [--n-trials 20]
    python3 algorithms/tune_hyperparams.py <dataset_path> lgbm [--n-trials 20]

Output:
    results/hyperparam_selection/{model}_full_short_optuna_results.csv   (every trial)
    results/hyperparam_selection/{model}_full_short_optuna_results_best.json (best trial's params)
"""

import argparse
import datetime as dt
import json
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

from algorithms.kpi_gen.ma_kpi_generator import MAKPIGenerator
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

RFR = "rfr"
LGBM = "lgbm"
KPI_TYPE = "full_short"
NUM_MONTHS = 6

# Exp1's own config from run_recommendation.py: min_date, max_date, num_splits, num_future.
# Index [0] of the returned (dates, future_dates) is the same window already evaluated
# in results/rfr/*_2019-08-01_metrics.csv and results/lgbm/*_2019-08-01_metrics.csv.
EXP1_MIN_DATE = "2019-08-01"
EXP1_MAX_DATE = "2021-02-26"
EXP1_NUM_SPLITS = 28
EXP1_NUM_FUTURE = 13

OUTPUT_DIR = os.path.join(_REPO_ROOT, "results", "hyperparam_selection")


def _load_split(dataset_path):
    interactions_file = os.path.join(dataset_path, "transactions.csv")
    time_series_file = os.path.join(dataset_path, "close_prices.csv")

    interaction_data = FinancialInteractionData(interactions_file)
    time_series_data = FinancialAssetTimeSeries(time_series_file)

    data = FinancialContinuousData(interaction_data, time_series_data)
    data.load()

    kpi_gen = MAKPIGenerator(data.time_series.data, 5, KPI_TYPE)
    kpi_gen.compute()
    data.add_kpis(kpi_gen.get_kpis())

    min_date = dt.datetime.strptime(EXP1_MIN_DATE, "%Y-%m-%d")
    max_date = dt.datetime.strptime(EXP1_MAX_DATE, "%Y-%m-%d")
    dates, future_dates = data.get_dates(min_date, max_date, EXP1_NUM_SPLITS, EXP1_NUM_FUTURE)
    rec_date, future_date = dates[0], future_dates[0]
    print(f"Tuning window: {rec_date} -> {future_date}")

    min_split_date = rec_date - dt.timedelta(days=36525)
    splitted_data = data.split(
        min_split_date, rec_date, future_date,
        DataFilter(CustomerInTrain(), AssetWithTestPrice(), RatingsNotInTrain(), NoFilter(), False, True, False),
    )

    profitability = compute_profitability(splitted_data.time_series, rec_date, future_date, None)
    monthly_metric = MonthlyKPIEvaluationMetric(splitted_data, profitability, (future_date - rec_date).days)

    return splitted_data, rec_date, monthly_metric


def _make_objective(model_id, splitted_data, rec_date, monthly_metric):
    def objective(trial):
        # Plain sklearn/lightgbm regressors (not RFRKPIModel/LGBMKPIModel): with
        # splitted_data.kpis already populated, ProfitabilityPrediction takes the
        # "external" (precomputed-KPI) branch and never regenerates KPIs from raw
        # prices, so each trial only pays for the actual model fit.
        if model_id == RFR:
            n_estimators = trial.suggest_int("n_estimators", 10, 500)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 100)
            max_depth = trial.suggest_categorical("max_depth", [None, 5, 10, 15, 20, 30, 50])
            alg_model = RandomForestRegressor(
                n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_depth=max_depth,
                random_state=42, n_jobs=-1,
            )
        else:
            n_estimators = trial.suggest_int("n_estimators", 10, 500)
            num_leaves = trial.suggest_int("num_leaves", 1, 100)
            min_child_samples = trial.suggest_int("min_child_samples", 5, 100)
            alg_model = LGBMRegressor(
                n_estimators=n_estimators, num_leaves=num_leaves, min_child_samples=min_child_samples,
                random_state=42, n_jobs=-1, verbose=-1,
            )

        algorithm = ProfitabilityPrediction(
            alg_model, splitted_data, NUM_MONTHS, full_short_kpis, -1, save_for_testing=False,
        )
        algorithm.train(rec_date)
        recs = algorithm.recommend(rec_date, splitted_data.users, False, True)

        cutoff_results = monthly_metric.evaluate_cutoffs(recs, [10], splitted_data.users, True)
        _, monthly_prof_10 = cutoff_results[10]
        return float(monthly_prof_10)

    return objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path", help="Path to the FAR-Trans dataset directory")
    parser.add_argument("model", choices=[RFR, LGBM], help="Model to tune")
    parser.add_argument("--n-trials", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    splitted_data, rec_date, monthly_metric = _load_split(args.dataset_path)
    objective = _make_objective(args.model, splitted_data, rec_date, monthly_metric)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.n_trials)

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_csv = os.path.join(OUTPUT_DIR, f"{args.model}_{KPI_TYPE}_optuna_results.csv")
    trials_df.to_csv(trials_csv, index=False)
    print(f"Wrote {len(trials_df)} trials to {trials_csv}")

    best_params = {"model": args.model, "kpi_type": KPI_TYPE, **study.best_trial.params}
    best_json = os.path.join(OUTPUT_DIR, f"{args.model}_{KPI_TYPE}_optuna_results_best.json")
    with open(best_json, "w") as handle:
        json.dump(best_params, handle, indent=2)
    print(f"Best monthly_prof@10 = {study.best_value:.6f}, params saved to {best_json}")


if __name__ == "__main__":
    main()
