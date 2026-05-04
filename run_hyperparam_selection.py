"""
Hyperparameter selection for profitability prediction models using Optuna (TPE).

Runs Optuna trials on three fixed dates and picks the configuration that
maximises mean profitability@10 (ROI at 6 months).

Selected dates (split -> evaluation):
  2019-04-01 -> 2019-10-01
  2019-10-01 -> 2020-04-01
  2020-01-31 -> 2020-07-31

Usage:
  python3 run_hyperparam_selection.py FAR-Trans-Data [--model rfr|lgbm] [--kpi-type full_short|full|basic|basic_short] [--n-trials 50]
"""

import argparse
import datetime as dt
import json
import os

import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

from algorithms.profitability_prediction import ProfitabilityPrediction
from algorithms.rfr_kpi_model import RFRKPIModel
from algorithms.lgbm_kpi_model import LGBMKPIModel
from data.filter.asset.asset_with_test_price import AssetWithTestPrice
from data.filter.customer.customer_in_train import CustomerInTrain
from data.filter.data_filter import DataFilter
from data.filter.rating.ratings_not_in_train import RatingsNotInTrain
from data.filter.timeseries.no_filter import NoFilter
from data.financial_asset_time_series import FinancialAssetTimeSeries
from data.financial_data_continuous import FinancialContinuousData
from data.financial_interaction_data import FinancialInteractionData
from metrics.kpi_evaluation_metric import KPIEvaluationMetric
from utils.constants import DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_USER_COL

try:
    pd.set_option("future.infer_string", False)
except Exception:
    pass

pd.options.mode.chained_assignment = None

CUTOFF = 10
MONTHS = 6

SELECTION_DATES = [
    ("2019-04-01", "2019-10-01"),
    ("2019-10-01", "2020-04-01"),
    ("2020-01-31", "2020-07-31"),
]

KPI_FEATURES = {
    "full_short": [
        "past_profitability_21d", "past_profitability_63d", "past_profitability_126d",
        "volatility_21d", "volatility_63d", "volatility_126d",
        "avg_price_21d", "avg_price_63d", "avg_price_126d",
        "sharpe_21d", "sharpe_63d", "sharpe_126d",
        "m_21d", "m_63d", "m_126d",
        "roc_21d", "roc_63d", "roc_126d",
        "MACD", "rsi_14", "dco_22",
        "min_21d", "min_63d", "min_126d",
        "max_21d", "max_63d", "max_126d",
        "exp_mean_21d", "exp_mean_63d", "exp_mean_126d",
    ],
    "full": [
        "past_profitability_63d", "past_profitability_126d", "past_profitability_189d",
        "volatility_63d", "volatility_126d", "volatility_189d",
        "avg_price_63d", "avg_price_126d", "avg_price_189d",
        "sharpe_63d", "sharpe_126d", "sharpe_189d",
        "m_63d", "m_126d", "m_189d",
        "roc_63d", "roc_126d", "roc_189d",
        "MACD", "rsi_14", "dco_22",
        "min_63d", "min_126d", "min_189d",
        "max_63d", "max_126d", "max_189d",
        "exp_mean_63d", "exp_mean_126d", "exp_mean_189d",
    ],
    "basic": [
        "past_profitability_63d", "past_profitability_126d", "past_profitability_189d",
        "volatility_63d", "volatility_126d", "volatility_189d",
        "avg_price_63d", "avg_price_126d", "avg_price_189d",
    ],
    "basic_short": [
        "past_profitability_21d", "past_profitability_63d", "past_profitability_126d",
        "volatility_21d", "volatility_63d", "volatility_126d",
        "avg_price_21d", "avg_price_63d", "avg_price_126d",
    ],
}


def compute_profitability(time_series, rec_date, future_date):
    rec_s = time_series[time_series[DEFAULT_TIMESTAMP_COL] == rec_date]
    fut_s = time_series[time_series[DEFAULT_TIMESTAMP_COL] == future_date]
    merged = rec_s.merge(fut_s, on=DEFAULT_ITEM_COL, suffixes=("_present", "_future"))
    merged["profitability"] = (
        (merged[DEFAULT_RATING_COL + "_future"] - merged[DEFAULT_RATING_COL + "_present"])
        / merged[DEFAULT_RATING_COL + "_present"]
    )
    return dict(zip(merged[DEFAULT_ITEM_COL], merged["profitability"]))


def evaluate_profitability_at_k(split, rec_date, future_date, model, feats, cutoff):
    prof_dict = compute_profitability(split.time_series, rec_date, future_date)
    for asset in split.assets:
        if asset not in prof_dict:
            prof_dict[asset] = 0.0

    algorithm = ProfitabilityPrediction(model, split, MONTHS, feats, -1)
    algorithm.train(rec_date)

    recs = algorithm.recommend(rec_date, split.users, False, True)
    recs = recs.sort_values(
        by=[DEFAULT_USER_COL, DEFAULT_RATING_COL], ascending=[False, False]
    )

    metric = KPIEvaluationMetric(split, prof_dict)
    customers = split.users & set(split.test[DEFAULT_USER_COL].unique())
    result = metric.evaluate(recs.groupby(DEFAULT_USER_COL).head(cutoff), cutoff, customers, True)
    return float(result[1])


def run_selection(data, model_name, kpi_type, feats, output_path, n_trials=50):
    delta = dt.timedelta(days=36525)
    data_filter = DataFilter(
        CustomerInTrain(), AssetWithTestPrice(), RatingsNotInTrain(), NoFilter(), False, True, False
    )

    splits = []
    for split_str, future_str in SELECTION_DATES:
        rec_date = pd.to_datetime(split_str)
        future_date = pd.to_datetime(future_str)
        split = data.split(rec_date - delta, rec_date, future_date, data_filter)
        splits.append((rec_date, future_date, split))

    def objective(trial):
        if model_name == "rfr":
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 50, 500, step=50),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100),
                "max_depth":        trial.suggest_categorical("max_depth", [10, 20, 30, None]),
            }
            model_fn = lambda: RFRKPIModel(
                n_estimators=params["n_estimators"],
                k=5,
                kpi_type=kpi_type,
                kpi_features=feats,
                random_state=42,
                max_features="sqrt",
                min_samples_leaf=params["min_samples_leaf"],
                max_depth=params["max_depth"],
                max_samples=0.8,
                n_jobs=-1,
            )
        else:
            params = {
                "n_estimators":      trial.suggest_int("n_estimators", 50, 500, step=50),
                "num_leaves":        trial.suggest_int("num_leaves", 8, 128),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            }
            model_fn = lambda: LGBMKPIModel(
                n_estimators=params["n_estimators"],
                k=5,
                kpi_type=kpi_type,
                kpi_features=feats,
                random_state=42,
                max_depth=5,
                num_leaves=params["num_leaves"],
                min_child_samples=params["min_child_samples"],
                subsample=0.6,
                colsample_bytree=0.6,
                learning_rate=0.03,
                reg_lambda=2.0,
                reg_alpha=0.0,
                n_jobs=-1,
            )

        scores = []
        for rec_date, future_date, split in splits:
            score = evaluate_profitability_at_k(split, rec_date, future_date, model_fn(), feats, CUTOFF)
            scores.append(score)
        mean_score = sum(scores) / len(scores)
        print(f"  trial {trial.number}: {params} -> mean profitability@{CUTOFF} = {mean_score:.4f}", flush=True)
        return mean_score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_score = study.best_value
    print(f"\nBest: {best_params}  (mean profitability@{CUTOFF} = {best_score:.4f})")

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df = trials_df.sort_values("value", ascending=False)
    trials_df.to_csv(output_path, index=False)
    print(f"Trial results saved to {output_path}")

    best_out = output_path.replace(".csv", "_best.json")
    with open(best_out, "w") as f:
        json.dump({"model": model_name, "kpi_type": kpi_type, **best_params}, f, indent=2)
    print(f"Best hyperparameters saved to {best_out}")

    return best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter selection via Optuna on three fixed dates")
    parser.add_argument("dataset_path", help="Path to FAR-Trans dataset directory")
    parser.add_argument("--model", choices=["rfr", "lgbm"], default="rfr")
    parser.add_argument("--kpi-type", choices=list(KPI_FEATURES.keys()), default="full_short")
    parser.add_argument("--output-dir", default="results/hyperparam_selection")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    interactions_file = os.path.join(args.dataset_path, "transactions.csv")
    time_series_file = os.path.join(args.dataset_path, "close_prices.csv")

    interaction_data = FinancialInteractionData(interactions_file)
    time_series_data = FinancialAssetTimeSeries(time_series_file)
    data = FinancialContinuousData(interaction_data, time_series_data)
    data.load()

    feats = KPI_FEATURES[args.kpi_type]
    output_csv = os.path.join(args.output_dir, f"{args.model}_{args.kpi_type}_optuna_results.csv")

    run_selection(data, args.model, args.kpi_type, feats, output_csv, n_trials=args.n_trials)
