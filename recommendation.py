#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.


import datetime as dt
import os
import sys
from multiprocessing import Process
from multiprocessing import Semaphore

import argparse

import numpy as np
import pandas as pd
from utils.constants import DEFAULT_TIMESTAMP_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_USER_COL
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

try:
    pd.set_option("future.infer_string", False)
except Exception:
    pass

from algorithms.kpi_gen.load_kpi_generator import LoadKPIGenerator
from algorithms.kpi_gen.ma_kpi_generator import MAKPIGenerator
from algorithms.lgbm_kpi_model import LGBMKPIModel
from algorithms.mlp_kpi_model import MLPKPIModel
from algorithms.rfr_kpi_model import RFRKPIModel
from algorithms.tabnet_kpi_model import TabNetKPIModel
from algorithms.profitability_prediction import ProfitabilityPrediction
from data.filter.asset.asset_with_test_price import AssetWithTestPrice
from data.filter.customer.customer_in_train import CustomerInTrain
from data.filter.data_filter import DataFilter
from data.filter.rating.ratings_not_in_train import RatingsNotInTrain
from data.filter.timeseries.no_filter import NoFilter
from data.financial_asset_time_series import FinancialAssetTimeSeries
from data.financial_data_continuous import FinancialContinuousData
from data.financial_interaction_data import FinancialInteractionData
from metrics.kpi_ann_evaluation_metric import AnnualizedKPIEvaluationMetric
from metrics.kpi_evaluation_metric import KPIEvaluationMetric
from metrics.kpi_monthly_evaluation_metric import MonthlyKPIEvaluationMetric
from metrics.pure_ndcg import PureNDCG

pd.options.mode.chained_assignment = None  # default='warn'

timea = dt.datetime.now()

class Object(object):
    pass


basic_kpis = ["past_profitability_63d", "past_profitability_126d", "past_profitability_189d",
            "volatility_63d", "volatility_126d", "volatility_189d",
            "avg_price_63d", "avg_price_126d", "avg_price_189d"]
full_kpis = ["past_profitability_63d", "past_profitability_126d", "past_profitability_189d",
            "volatility_63d", "volatility_126d", "volatility_189d",
            "avg_price_63d", "avg_price_126d", "avg_price_189d",
            "sharpe_63d", "sharpe_126d", "sharpe_189d",
            "m_63d", "m_126d", "m_189d",
            "roc_63d", "roc_126d", "roc_189d",
            "MACD", "rsi_14", "dco_22",
            "min_63d", "min_126d", "min_189d",
            "max_63d", "max_126d", "max_189d",
            "exp_mean_63d", "exp_mean_126d", "exp_mean_189d"]
basic_short_kpis = ["past_profitability_21d", "past_profitability_63d", "past_profitability_126d",
                    "volatility_21d", "volatility_63d", "volatility_126d",
                    "avg_price_21d", "avg_price_63d", "avg_price_126d"]
full_short_kpis = ["past_profitability_21d", "past_profitability_63d", "past_profitability_126d",
                "volatility_21d", "volatility_63d", "volatility_126d",
                "avg_price_21d", "avg_price_63d", "avg_price_126d",
                "sharpe_21d", "sharpe_63d", "sharpe_126d",
                "m_21d", "m_63d", "m_126d",
                "roc_21d", "roc_63d", "roc_126d",
                "MACD", "rsi_14", "dco_22",
                "min_21d", "min_63d", "min_126d",
                "max_21d", "max_63d", "max_126d",
                "exp_mean_21d", "exp_mean_63d", "exp_mean_126d"]


# Regression
RFR = "rfr"
LGBM = "lgbm"
MLP = "mlp"
TABNET = "tabnet"
MLP_KPI_TYPES = {"full", "basic", "basic_short", "full_short"}
TABNET_ALLOWED_KEYS = {"kpi", "kpi_type", "n_d", "n_a", "n_steps"}

def _parse_rfr_params(params):
    n = 100
    kpi_type = "full_short"
    use_internal = True
    min_samples_leaf = 5
    max_depth = 20

    for raw in params or []:
        token = str(raw).strip()
        if token == "":
            continue

        if "=" in token:
            key, val = token.split("=", 1)
            if key == "min_samples_leaf":
                min_samples_leaf = int(val)
            elif key == "max_depth":
                max_depth = None if val in ("None", "null") else int(val)
            continue

        token_lower = token.lower()
        if token_lower in MLP_KPI_TYPES:
            kpi_type = token_lower
            continue

        if token_lower in {"legacy", "external"}:
            use_internal = False
            continue

        if token.lstrip("+-").isdigit():
            n = int(token)

    return n, kpi_type, use_internal, min_samples_leaf, max_depth


def _parse_lgbm_params(params):
    n = 100
    kpi_type = "full_short"
    use_internal = True
    num_leaves = 12
    min_child_samples = 80

    for raw in params or []:
        token = str(raw).strip()
        if token == "":
            continue

        if "=" in token:
            key, val = token.split("=", 1)
            if key == "num_leaves":
                num_leaves = int(val)
            elif key == "min_child_samples":
                min_child_samples = int(val)
            continue

        token_lower = token.lower()
        if token_lower in MLP_KPI_TYPES:
            kpi_type = token_lower
            continue

        if token_lower in {"legacy", "external"}:
            use_internal = False
            continue

        if token.lstrip("+-").isdigit():
            n = int(token)

    return n, kpi_type, use_internal, num_leaves, min_child_samples


def _parse_mlp_params(params):
    hidden_sizes = [64, 32, 16]
    kpi_type = "full_short"

    if not params:
        return hidden_sizes, kpi_type

    hidden_tokens = []

    for raw in params:
        token = str(raw).strip()
        if token == "":
            continue

        token_lower = token.lower()
        if token_lower in MLP_KPI_TYPES:
            kpi_type = token_lower
            continue

        if token_lower.startswith("kpi="):
            value = token_lower.split("=", 1)[1].strip()
            if value in MLP_KPI_TYPES:
                kpi_type = value
            continue

        cleaned = token.replace("[", "").replace("]", "")
        pieces = [p.strip() for p in cleaned.split(",") if p.strip() != ""]
        if not pieces:
            continue

        if all(piece.lstrip("+-").isdigit() for piece in pieces):
            hidden_tokens.extend(pieces)

    if hidden_tokens:
        hidden_sizes = [int(x) for x in hidden_tokens]

    return hidden_sizes, kpi_type


def _parse_tabnet_params(params):
    config = {
        "kpi_type": "full_short",
        "n_d": 32,
        "n_a": 32,
        "n_steps": 3,
    }

    if not params:
        return config

    positional = []
    for raw in params:
        token = str(raw).strip()
        if token == "":
            continue

        token_lower = token.lower()
        if token_lower in MLP_KPI_TYPES:
            config["kpi_type"] = token_lower
            continue

        if "=" in token:
            key, value = token.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key not in TABNET_ALLOWED_KEYS:
                continue
            if key in {"kpi", "kpi_type"} and value.lower() in MLP_KPI_TYPES:
                config["kpi_type"] = value.lower()
            elif key in {"n_d", "n_a", "n_steps"}:
                if value.lstrip("+-").isdigit():
                    config[key] = int(value)
            continue

        positional.append(token)

    numeric = []
    for token in positional:
        try:
            numeric.append(float(token))
        except ValueError:
            pass

    if len(numeric) >= 1:
        config["n_d"] = int(numeric[0])
    if len(numeric) >= 2:
        config["n_a"] = int(numeric[1])
    if len(numeric) >= 3:
        config["n_steps"] = int(numeric[2])

    return config


def test(algorithm, eval_metrics, file, recomm_date, customers):
    """
    Function that (a) trains an algorithm, (b) generates recommendations and (c) evaluates an algorithm.
    Recommendations and evaluations are written into text files.
    :param algorithm: the recommendation algorithm to apply.
    :param eval_metrics: the evaluation metrics to apply.
    :param file: the name of the file in which to store the recommendation.
    :param recomm_date: the date of the recommendation.
    :param customers: the set of customers to use.
    """
    if os.path.exists(file + "_metrics.csv"):
        return

    timeaa = dt.datetime.now()
    print("Started " + file)

    # 1. Train the algorithm:
    algorithm.train(recomm_date)
    time_elapsed = dt.datetime.now() - timeaa
    print("Algorithm " + file + " trained (" + '{}'.format(time_elapsed) + ")")

    # 2. Generate the recommendations:
    recs = algorithm.recommend(recomm_date, customers, False, True)
    recs = recs.sort_values(by=[DEFAULT_USER_COL, DEFAULT_RATING_COL], ascending=[False, False])
    recs.to_csv(file + "_recs.txt", index=False)
    time_elapsed = dt.datetime.now() - timea

    print("Generated recommendations for algorithm " + file + " (" + '{}'.format(time_elapsed) + ")")
    # 3. Compute the metrics:
    cutoffs = [1, 5, 10, 20, 50, 100, 1000]
    metric_res = dict()
    for metric in eval_metrics:
        print("Started metric " + metric[0] + " for " + file)

        metric_dict = metric[1].evaluate_cutoffs(recs, cutoffs, customers, True)
        for cutoff in cutoffs:
            metric_name = metric[0] + "@" + str(cutoff)
            metric_res[metric_name] = metric_dict[cutoff]
        time_elapsed = dt.datetime.now() - timeaa
        print("Computed metric " + metric[0] + " for algorithm " + file + " (" + '{}'.format(time_elapsed) + ")")

    time_elapsed = dt.datetime.now() - timeaa
    print("Metrics computed for algorithm " + file + " (" + '{}'.format(time_elapsed) + ")")

    # Output the metrics:
    f = open(file + "_metrics.csv", "w")
    for key, val in metric_res.items():
        f.write(key + "\t" + str(val[1]) + "\n")
    gen_metrics = getattr(algorithm, "generalization_metrics_", {})
    for key, val in gen_metrics.items():
        f.write(key + "\t" + str(val) + "\n")
    f.close()


    cust_metric_df = None
    # Output the metrics by customer
    for key, val in metric_res.items():
        if cust_metric_df is None:
            cust_metric_df = val[0].rename(columns={"metric" : key})
        else:
            aux_df = val[0].rename(columns={"metric": key})
            cust_metric_df = cust_metric_df.merge(aux_df, on=DEFAULT_USER_COL)
    cust_metric_df.to_csv(file + "_customers.csv", index=False)

    time_elapsed = dt.datetime.now() - timea
    print("Algorithm " + file + " finished (" + '{}'.format(time_elapsed) + ")")


def regressor(model_id, param, financial_data, recommendation_date, eval_metrics, output_dir, file, num_months,
              save_for_testing=False):
    """
    Configures and runs regression models (predict future profitability of stocks, and rank them according to that
    prediction).
    :param param: the parameters of the regression model.
    :param financial_data: the split financial data to use.
    :param recommendation_date: the recommendation date.
    :param eval_metrics: the metrics to apply in the evaluation.
    :param output_dir: the output directory.
    :param file: the name of the file.
    :param num_months: the number of months to look into the future.
    """
    alg_model = None
    full = False
    
    # Parse parameters
    kpi_type = "full_short"
    hidden_sizes = None
    tabnet_cfg = None
    use_internal_rfr = True
    use_internal_lgbm = True
    n = 20

    if model_id == RFR:
        n, kpi_type, use_internal_rfr, min_samples_leaf, max_depth = _parse_rfr_params(param)
    elif model_id == LGBM:
        n, kpi_type, use_internal_lgbm, num_leaves, min_child_samples = _parse_lgbm_params(param)
    elif model_id == MLP:
        hidden_sizes, kpi_type = _parse_mlp_params(param)
    elif model_id == TABNET:
        tabnet_cfg = _parse_tabnet_params(param)
        kpi_type = tabnet_cfg["kpi_type"]
    
    # Determine features based on kpi_type
    if kpi_type == "full":
        feats = full_kpis
    elif kpi_type == "basic":
        feats = basic_kpis
    elif kpi_type == "basic_short":
        feats = basic_short_kpis
    else:
        # if kpi_type == "full_short":
        feats = full_short_kpis
    
    if model_id == RFR:
        if use_internal_rfr:
            alg_model = RFRKPIModel(
                n_estimators=n,
                k=5,
                kpi_type=kpi_type,
                kpi_features=feats,
                random_state=42,
                max_features="sqrt",
                min_samples_leaf=min_samples_leaf,
                max_depth=max_depth,
                max_samples=0.8,
                n_jobs=-1,
                param_grid=None,
            )
        else:
            alg_model = RandomForestRegressor(n_estimators=n)
    elif model_id == LGBM:
        if use_internal_lgbm:
            alg_model = LGBMKPIModel(
                n_estimators=n,
                k=5,
                kpi_type=kpi_type,
                kpi_features=feats,
                random_state=42,
                max_depth=5,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                subsample=0.6,
                colsample_bytree=0.6,
                learning_rate=0.03,
                reg_lambda=2.0,
                reg_alpha=0.0,
                n_jobs=-1,
            )
        else:
            alg_model = LGBMRegressor()
    elif model_id == MLP:
        # Create MLP model (KPI generation happens inside the model)
        alg_model = MLPKPIModel(
            hidden_sizes=hidden_sizes,
            k=5,
            kpi_type=kpi_type,
            kpi_features=feats,
            seed=42,
        )
    else:
        if tabnet_cfg is None:
            tabnet_cfg = _parse_tabnet_params(param)
        alg_model = TabNetKPIModel(
            k=5,
            kpi_type=tabnet_cfg["kpi_type"],
            kpi_features=feats,
            n_d=tabnet_cfg["n_d"],
            n_a=tabnet_cfg["n_a"],
            n_steps=tabnet_cfg["n_steps"],
            gamma=1.3,
            lambda_sparse=1e-4,
        )

    training_sizes_path = os.path.join(os.path.dirname(output_dir), "training_sizes.csv")
    algorithm = ProfitabilityPrediction(alg_model, financial_data, num_months, feats, -1,
                                        save_for_testing=save_for_testing,
                                        training_sizes_path=training_sizes_path)
    file_name = os.path.join(output_dir, file)
    test(algorithm, eval_metrics, file_name, recommendation_date, financial_data.users)




def get_name(rec_model, param):
    """
    Given a model, its parameters and a date, obtains the name of the file
    where the results shall be stored.
    :param rec_model: the name of the model.
    :param param: the parameters of the model.
    :return: the name of the model if everything goes right, None otherwise.
    """
    print("model:" + rec_model)

    algorithm_name = None
    
    if rec_model == MLP:
        hidden_sizes, kpi_type = _parse_mlp_params(param)
        hidden_sizes_str = ",".join(str(x) for x in hidden_sizes)
        algorithm_name = MLP + "_" + hidden_sizes_str + "_" + kpi_type
    elif rec_model == TABNET:
        tabnet_cfg = _parse_tabnet_params(param)
        algorithm_name = (
            TABNET
            + "_"
            + tabnet_cfg["kpi_type"]
            + "_nd"
            + str(tabnet_cfg["n_d"])
            + "_na"
            + str(tabnet_cfg["n_a"])
            + "_ns"
            + str(tabnet_cfg["n_steps"])
        )
    elif rec_model == LGBM:
        n, kpi_type, use_internal_lgbm, _, _ = _parse_lgbm_params(param)
        algorithm_name = LGBM + "_" + str(n) + "_" + kpi_type
        if use_internal_lgbm:
            algorithm_name += "_internal_kpis"
    else:
        # RFR (internal or external)
        n, kpi_type, use_internal_rfr, _, _ = _parse_rfr_params(param)
        algorithm_name = RFR + "_" + str(n) + "_" + kpi_type
        if use_internal_rfr:
            algorithm_name += "_internal_kpis"

    return algorithm_name



def compute_profitability(time_series, recommendation_date, evaluation_date, min_values):
    """
    Computes the profitability of assets.
    :param time_series: the time series containing the asset prices.
    :param recommendation_date: the recommendation date (starting date)
    :param evaluation_date: the future date (end date)
    :param min_values: if available, a file containing min values of prices.
    :return: a dataframe containing the (raw) profitability of assets between rec_date and future_date.
    """
    # In this case, it is impossible (as of now) that there is an asset without future date pricing:
    rec_series = time_series[time_series[DEFAULT_TIMESTAMP_COL] == recommendation_date]
    future_series = time_series[time_series[DEFAULT_TIMESTAMP_COL] == evaluation_date]
    # ndays = (future_date - rec_date).days

    aux_series = rec_series.merge(future_series, on=DEFAULT_ITEM_COL, suffixes=("_present", "_future"))
    aux_series["profitability"] = (aux_series[DEFAULT_RATING_COL + "_future"] - aux_series[
        DEFAULT_RATING_COL + "_present"]) / aux_series[DEFAULT_RATING_COL + "_present"]
    prof_dict = dict()
    for index, row in aux_series.iterrows():
        prof_dict[row[DEFAULT_ITEM_COL]] = row["profitability"]

    if min_values is not None:
        max_series = rec_series.merge(min_values, on=DEFAULT_ITEM_COL)
        max_series["profitability"] = (max_series["max_price"] - max_series[DEFAULT_RATING_COL]) / max_series[
            DEFAULT_RATING_COL]
        for index, row in max_series.iterrows():
            if row[DEFAULT_ITEM_COL] not in prof_dict:
                prof_dict[row[DEFAULT_ITEM_COL]] = row["profitability"]
    return prof_dict


def compute_volatility(time_series, recommendation_date, evaluation_date):
    """
    Computes the volatility of assets.
    :param time_series: the time series containing the asset prices.
    :param recommendation_date: the recommendation date (starting date)
    :param evaluation_date: the future date (end date)
    :return: a dataframe containing the (raw) profitability of assets between rec_date and future_date.
    """
    series = time_series[time_series[DEFAULT_TIMESTAMP_COL].between(recommendation_date, evaluation_date)]

    series_asset = dict()
    for asset in series[DEFAULT_ITEM_COL].unique().flatten():
        aux_series = series[series[DEFAULT_ITEM_COL] == asset]
        aux_series["profit"] = (aux_series[DEFAULT_RATING_COL] - aux_series[DEFAULT_RATING_COL].shift(1)) / aux_series[
            DEFAULT_RATING_COL].shift(1)
        aux_series = aux_series.dropna()

        series_asset[asset] = aux_series["profit"].std() * np.sqrt(252)

    return series_asset


def print_error_message():
    """
    Prints an error message in case there is an error with the program execution
    :return: the error message.
    """
    text = "ERROR: Invalid arguments:"
    text += "\n\tInteraction data file: file containing the interaction data."
    text += "\n\tTime series data file: file containing the time series."
    text += "\n\tDate format: the format to read the dates. Two valid options:"
    text += "\n\t\trange: to specify a range of dates. In this case, the following arguments are:"
    text += "\n\t\t\tMin. date: The minimum recommendation date to consider."
    text += "\n\t\t\tMax. date: The maximum recommendation date to consider."
    text += "\n\t\t\tNum. splits: the number of recommendation dates to consider (equally separated)."
    text += "\n\t\t\tNum. future: Number of steps in the future to consider."
    text += "\n\t\tfixed_dates: to specify a list of dates, the following arguments are:"
    text += "\n\t\t\trec_dates: a comma separated list of dates in %Y-%m-%d format."
    text += "\n\t\t\tfuture_dates: a comma separated list of evaluation dates in %Y-%m-%d format."
    text += "\n\tDirectory: the directory in which to store all the data"
    text += "\n\tDelta: how many days to consider before the recommendation date as training data."
    text += "\n\tModel: the recommendation model to consider"
    text += "\n\t"
    return text


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="financial_asset_recommendation",
        description="Runs financial asset recommendations and evaluates them.",
        epilog="Developed by University of Glasgow"
    )

    parser.add_argument("interactions", help="Customer-asset transaction data file.")
    parser.add_argument("time_series", help="Asset pricing data file.")
    subparsers = parser.add_subparsers(title='date_format', help='Data choice format.', dest='date_format')

    parser_range = subparsers.add_parser('range', help='Range of dates to use. This mode divides the dataset as '
                                                    'follows:\n'
                                                    '- First, divide the period between min_date and max_date into'
                                                    'num_splits + num_future dates\n'
                                                    '- Second, the first num_split dates are considered the split '
                                                    'dates (everything before them is the training set).\n'
                                                    '- The test set contains the data between the split date and '
                                                    '  num_future dates in the list afterwards.')
    parser_range.add_argument("min_date", help='Date of the first split. Format: %Y-%m-%d')
    parser_range.add_argument("max_date", help='End date of the last test set. Format: %Y-%m-%d')
    parser_range.add_argument("num_splits", help='Number of splits to consider.', type=int)
    parser_range.add_argument("num_future", help='Number of dates to look formward', type=int)
    parser_range.add_argument("output_dir", help="directory on which to store the outputs.")
    parser_range.add_argument("months", help="number of months to look into the future.")
    parser_range.add_argument("model", help="model identifier", choices=[RFR, LGBM, MLP, TABNET])
    parser_range.add_argument("params", help="model parameters", action="store", nargs="*")

    parser_fixed = subparsers.add_parser('fixed_dates', help='List of fixed dates to use. This mode provides fixed '
                                                            'lists of dates for split and test.')
    parser_fixed.add_argument('split_dates', help='Comma separated list of split dates. Date format: %Y-%m-%d')
    parser_fixed.add_argument('future_dates', help='Comma separated list of test end dates. Date format: %Y-%m-%d')
    parser_fixed.add_argument("output_dir", help="directory on which to store the outputs.")
    parser_fixed.add_argument("months", help="number of months to look into the future.")
    parser_fixed.add_argument("model", help="model identifier", choices=[RFR, LGBM, MLP, TABNET])
    parser_fixed.add_argument("params", help="model parameters", action="store", nargs="*")

    args = parser.parse_args()

    # First, we read the parameters:
    interaction_data_file = args.interactions
    time_series_data_file = args.time_series
    date_format = args.date_format

    p = 0
    dates_args = []
    future_dates_args = []
    num_splits = 0
    num_future = 0
    min_date: dt.datetime
    max_date: dt.datetime

    if date_format == "range":
        min_date = dt.datetime.strptime(args.min_date, "%Y-%m-%d")
        max_date = dt.datetime.strptime(args.max_date, "%Y-%m-%d")
        num_splits = args.num_splits
        num_future = args.num_future

    elif date_format == "fixed_dates":
        dates_args = args.split_dates.split(",")
        future_dates_args = args.future_dates.split(",")
        num_splits = len(dates_args)
        num_future = len(future_dates_args)
        min_date = min(dates_args)
        max_date = max(future_dates_args)
    else:
        sys.stderr.write(print_error_message())
        exit(-1)

    directory = args.output_dir
    months_term = args.months
    model = args.model
    params = args.params

    selected_kpi_type = "full_short"
    use_internal_rfr = True
    use_internal_lgbm = True
    if model == RFR:
        _, selected_kpi_type, use_internal_rfr, _, _ = _parse_rfr_params(params)
    elif model == LGBM:
        _, selected_kpi_type, use_internal_lgbm, _, _ = _parse_lgbm_params(params)
    elif model == MLP:
        _, selected_kpi_type = _parse_mlp_params(params)
    elif model == TABNET:
        selected_kpi_type = _parse_tabnet_params(params)["kpi_type"]

    # If the number of days is 0 for the delta, we choose as minimum date one in the distant past
    # (36525 days is exactly 100 years before the established date)
    delta = dt.timedelta(days=36525)
    # Now, we load the data:
    interaction_data = FinancialInteractionData(interaction_data_file)
    time_series_data = FinancialAssetTimeSeries(time_series_data_file)

    # First, load the data.
    data = FinancialContinuousData(interaction_data, time_series_data)
    data.load()
    timeb = dt.datetime.now() - timea
    print("Dataset loaded (" + '{}'.format(timeb) + ")")

    # Compute the technical indicators (required for Random Forest)
    if (model == RFR and not use_internal_rfr) or (model == LGBM and not use_internal_lgbm):
        kpi_file = os.path.join(directory, "kpis.csv")
        kpi_type = selected_kpi_type

        if os.path.exists(kpi_file):
            kpi_gen = LoadKPIGenerator(kpi_file)
        else:
            kpi_gen = MAKPIGenerator(data.time_series.data, 5, kpi_type)

        kpi_gen.compute()
        kpis = kpi_gen.get_kpis()

        if not os.path.exists(kpi_file):
            kpi_gen.print_kpis(kpi_file)

        data.add_kpis(kpis)

        timeb = dt.datetime.now() - timea
        print("Technical indicators computed (" + '{}'.format(timeb) + ")")

    dates = []
    future_dates = []
    # Now, we select the possible dates:
    if date_format == "range":
        print("Num splits:" + str(num_splits) + " Num future: " + str(num_future))
        dates, future_dates = data.get_dates(min_date, max_date, num_splits, num_future)  # Split schedule: recommendation dates (train cutoff) + future evaluation dates (test horizon end)
    else:
        print("Num splits:" + str(num_splits))
        for date in dates_args:
            dates.append(pd.to_datetime(date))
        for date in future_dates_args:
            future_dates.append(pd.to_datetime(date))

    print("Selected dates:")
    for i in range(0, len(dates)):
        print("\t" + str(i) + "Training date: " + str(dates[i]) + "\tFuture date: " + str(future_dates[i]))

    procs = []

    def_dates = []
    def_future_dates = []
    def_name = []

    # GPU models (mlp/tabnet) run multiple windows in parallel across GPUs.
    # RFR runs sequentially in the main process so n_jobs=-1 gets all CPUs without
    # joblib's nested-parallelism cap that kicks in inside a subprocess.
    import torch as _torch
    _n_gpus = _torch.cuda.device_count() if _torch.cuda.is_available() else 0
    _gpu_model = model in (MLP, TABNET)
    semaphore = Semaphore(max(1, _n_gpus)) if _gpu_model else None

    # We first check the selected model is good.
    f_name = get_name(model, params)
    if f_name is None:
        print("ERROR: Invalid parameters")
        exit(-1)

    # Then, we generate the dates for this.
    for i in range(0, len(dates)):
        if not os.path.exists(os.path.join(directory, f_name)):
            def_dates.append(dates[i])
            def_future_dates.append(future_dates[i])
            def_name.append(f_name)

    print(len(def_dates))
    for i in range(0, len(def_dates)):
        rec_date = def_dates[i]
        future_date = def_future_dates[i]
        min_split_date = rec_date - delta
        save_for_testing = True

        alg_name = def_name[i] + "_" + rec_date.strftime("%Y-%m-%d")
        # We only generate recommendations for those dates on which we have not previously generated
        # the recommendations.
        if os.path.exists(os.path.join(directory, alg_name + "_metrics.csv")):
            print("Skipped " + alg_name + " as it already exists")
            continue

        # Get the corresponding file names:
        splitted_data = data.split(min_split_date, rec_date, future_date,  # Core temporal split: train in [min_split_date, rec_date), test in [rec_date, future_date]
                                DataFilter(CustomerInTrain(), AssetWithTestPrice(), RatingsNotInTrain(),
                                            NoFilter(), False, True, False))

        timeb = dt.datetime.now() - timea
        print("Dataset splitted (" + '{}'.format(timeb) + ")")

        # We compute the profitability and volatility.
        profitability_df = compute_profitability(splitted_data.time_series, rec_date, future_date, None)
        volatility_df = compute_volatility(splitted_data.time_series, rec_date, future_date)

        # Define the metrics
        metrics = [
            ("profitability", KPIEvaluationMetric(splitted_data, profitability_df)),
            ("annualized_prof", AnnualizedKPIEvaluationMetric(splitted_data, profitability_df,
                                                            (future_date - rec_date).days)),
            ("monthly_prof", MonthlyKPIEvaluationMetric(splitted_data, profitability_df,
                                                        (future_date - rec_date).days)),
            ("volatility", KPIEvaluationMetric(splitted_data, volatility_df)),
            ("ndcg", PureNDCG(splitted_data))]

        # Now, we choose metrics:
        print("Executing algorithm: " + model + " Start date: " + str(rec_date) + " End date: " + str(future_date))
        # Next: we get the algorithm and the parameters:

        if _gpu_model:
            # MLP/TabNet: run windows in parallel subprocesses, one per GPU.
            def _run(sem, *args):
                try:
                    regressor(*args)
                finally:
                    sem.release()

            semaphore.acquire()
            proc = Process(target=_run, args=(semaphore, model, params, splitted_data, rec_date, metrics, directory,
                                              alg_name, months_term, save_for_testing))
            procs.append(proc)
            proc.start()
        else:
            # RFR: run directly in the main process so sklearn's n_jobs=-1 gets all
            # CPUs without joblib's nested-parallelism cap inside a subprocess.
            regressor(model, params, splitted_data, rec_date, metrics, directory, alg_name, months_term,
                      save_for_testing)

    if len(procs) > 0:
        for proc in procs:
            proc.join()
