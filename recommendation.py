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
import json
import os
import platform
import socket
import sys

import argparse

import numpy as np
import pandas as pd
import wandb
from utils.constants import DEFAULT_TIMESTAMP_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_USER_COL
from codecarbon import EmissionsTracker
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

try:
    pd.set_option("future.infer_string", False)
except Exception:
    pass

from algorithms.kpi_gen.load_kpi_generator import LoadKPIGenerator
from algorithms.kpi_gen.ma_kpi_generator import MAKPIGenerator
from algorithms.lgbm_kpi_model import LGBMKPIModel
from algorithms.lr_kpi_model import LRKPIModel
from algorithms.rfr_kpi_model import RFRKPIModel
from algorithms.tabfm_kpi_model import TabFMKPIModel
from algorithms.tabicl_kpi_model import TabICLKPIModel
from algorithms.tabpfn_kpi_model import TabPFNKPIModel
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
TABPFN = "tabpfn"
TABICL = "tabicl"
TABFM = "tabfm"
LR = "lr"
KPI_TYPES = {"full", "basic", "basic_short", "full_short"}

def _parse_rfr_params(params):
    n = 100
    kpi_type = "full_short"
    use_internal = True
    tuned = False

    for raw in params or []:
        token = str(raw).strip()
        if token == "":
            continue

        token_lower = token.lower()
        if token_lower in KPI_TYPES:
            kpi_type = token_lower
            continue

        if token_lower == "external":
            use_internal = False
            continue

        if token_lower == "tuned":
            tuned = True
            continue

        if token.lstrip("+-").isdigit():
            n = int(token)

    return n, kpi_type, use_internal, tuned


def _parse_lgbm_params(params):
    # n stays None unless explicitly given: an untuned run with no number should
    # use LGBMRegressor's own library default, not a value pinned in this code.
    n = None
    kpi_type = "full_short"
    use_internal = True
    tuned = False

    for raw in params or []:
        token = str(raw).strip()
        if token == "":
            continue

        token_lower = token.lower()
        if token_lower in KPI_TYPES:
            kpi_type = token_lower
            continue

        if token_lower == "external":
            use_internal = False
            continue

        if token_lower == "tuned":
            tuned = True
            continue

        if token.lstrip("+-").isdigit():
            n = int(token)

    return n, kpi_type, use_internal, tuned


def _parse_lr_params(params):
    kpi_type = "full_short"
    use_internal = True
    tuned = False

    for raw in params or []:
        token = str(raw).strip()
        if token == "":
            continue

        token_lower = token.lower()
        if token_lower in KPI_TYPES:
            kpi_type = token_lower
            continue

        if token_lower == "external":
            use_internal = False
            continue

        if token_lower == "tuned":
            tuned = True
            continue

    return kpi_type, use_internal, tuned


def _parse_foundation_model_params(params, label):
    """Shared parser for the pretrained in-context tabular foundation models
    (tabpfn, tabicl, tabfm): a kpi_type and, optionally, a training/
    generalization-metrics sample fraction (a bare number in (0, 1], e.g.
    "0.25") — applied to both the actual fit (bounds the GPU context the
    model trains/predicts on) and the generalization-metrics diagnostic,
    per-asset stratified.
    """
    kpi_type = "full_short"
    sample_pct = None

    for raw in params or []:
        token = str(raw).strip()
        if token == "":
            continue

        token_lower = token.lower()
        if token_lower in KPI_TYPES:
            kpi_type = token_lower
            continue

        try:
            value = float(token)
        except ValueError:
            value = None

        if value is not None and 0 < value <= 1:
            sample_pct = value
            continue

        raise ValueError(
            f"Unsupported {label} parameter: '{token}'. Only a kpi_type "
            f"({sorted(KPI_TYPES)}) or a sample fraction in (0, 1] is "
            f"supported for {label}."
        )

    return kpi_type, sample_pct


def _parse_tabpfn_params(params):
    return _parse_foundation_model_params(params, "tabpfn")


def _parse_tabicl_params(params):
    return _parse_foundation_model_params(params, "tabicl")


def _parse_tabfm_params(params):
    return _parse_foundation_model_params(params, "tabfm")


def _load_tuned_params(model_id, kpi_type):
    """Load Optuna-selected hyperparams saved by tune_hyperparams.py.

    Returns the parsed JSON dict, e.g. {"n_estimators": 50, "min_samples_leaf": 87,
    "max_depth": None} for rfr, or raises if the file is missing (a "tuned" run
    with no saved params to apply is a configuration error, not something to
    silently fall back on).
    """
    path = os.path.join("results", "hyperparam_selection", f"{model_id}_{kpi_type}_optuna_results_best.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved tuned params at {path}. Run tune_hyperparams.py for '{model_id}' first."
        )
    with open(path, "r") as handle:
        return json.load(handle)


def _release_gpu_memory():
    """Reclaim GPU memory between windows.

    Each window builds a brand-new model (TabPFN in particular loads a fresh
    pretrained transformer onto the GPU every window). Once that window's
    objects go out of scope in the caller, plain refcounting won't always free
    them immediately — sklearn Pipeline/torch module objects can hold internal
    reference cycles that need a gc pass — and even once freed, PyTorch's CUDA
    caching allocator keeps that memory reserved for reuse rather than handing
    it back to the driver. Across ~61 windows of varying (growing, since the
    training window expands over time) tensor shapes, that reserved memory
    fragments and keeps climbing, which shows up as "accumulating" GPU memory
    in nvidia-smi even though nothing is actually leaked at the Python level.
    Safe/near-free no-op for CPU-only models (rfr/lgbm/lr).
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        print(f"WARNING: Could not release GPU memory: {exc}", flush=True)


_machine_info_cache = None


def _get_machine_info():
    """Best-effort machine fingerprint, so results can later be traced back to
    the hardware they ran on (useful since this project runs across several
    machines). Computed once per process and reused for every window."""
    global _machine_info_cache
    if _machine_info_cache is not None:
        return _machine_info_cache

    info = {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }

    try:
        import cpuinfo
        info["cpu_model"] = cpuinfo.get_cpu_info().get("brand_raw", "unknown")
    except Exception as exc:
        info["cpu_model"] = f"unknown ({exc})"

    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception as exc:
        info["ram_total_gb"] = f"unknown ({exc})"

    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_model"] = torch.cuda.get_device_name(0)
        else:
            info["gpu_count"] = 0
            info["gpu_model"] = "none"
    except Exception as exc:
        info["gpu_count"] = "unknown"
        info["gpu_model"] = f"unknown ({exc})"

    _machine_info_cache = info
    return info


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

    # Energy/CO2 tracking spans train+recommend (best-effort: a cluster node without
    # RAPL/GPU access or internet for the geolocation lookup shouldn't fail the whole
    # window, just skip the energy columns for it).
    energy_metrics = {}
    emissions_path = f"{file}_emissions.csv"
    tracker = None
    try:
        tracker = EmissionsTracker(
            output_dir=os.path.dirname(emissions_path) or ".",
            output_file=os.path.basename(emissions_path),
            measure_power_secs=1,
            log_level="error",
            # "machine" (the default) attributes the whole machine's power draw
            # (idle GPU, other processes) to this run; "process" estimates only
            # this process's own share, so CPU-only models like RFR stop
            # reporting non-trivial GPU energy from background/idle draw.
            tracking_mode="process",
        )
        tracker.start()
    except Exception as exc:
        print(f"WARNING: Could not start energy tracker for {file}: {exc}", flush=True)
        tracker = None

    # 1. Train the algorithm:
    train_start = dt.datetime.now()
    algorithm.train(recomm_date)
    train_seconds = (dt.datetime.now() - train_start).total_seconds()
    time_elapsed = dt.datetime.now() - timeaa
    print("Algorithm " + file + " trained (" + '{}'.format(time_elapsed) + ")")

    # 2. Generate the recommendations:
    predict_start = dt.datetime.now()
    recs = algorithm.recommend(recomm_date, customers, False, True)
    predict_seconds = (dt.datetime.now() - predict_start).total_seconds()

    if tracker is not None:
        try:
            tracker.stop()
            emissions_df = pd.read_csv(emissions_path)
            last_run = emissions_df.iloc[-1]
            energy_metrics = {
                "energy_consumed_kwh": float(last_run["energy_consumed"]),
                "cpu_energy_kwh": float(last_run["cpu_energy"]),
                "gpu_energy_kwh": float(last_run["gpu_energy"]),
                "ram_energy_kwh": float(last_run["ram_energy"]),
            }
        except Exception as exc:
            print(f"WARNING: Could not read energy tracking results for {file}: {exc}", flush=True)

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
    # Computational cost
    timing_metrics = {
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "total_seconds": train_seconds + predict_seconds,
    }
    for key, val in timing_metrics.items():
        f.write(key + "\t" + str(val) + "\n")
    for key, val in energy_metrics.items():
        f.write(key + "\t" + str(val) + "\n")
    gen_metrics = getattr(algorithm, "generalization_metrics_", {})
    for key, val in gen_metrics.items():
        f.write(key + "\t" + str(val) + "\n")
    f.close()

    # Machine fingerprint, kept out of _metrics.csv (numeric fields like cpu_count
    # would otherwise get picked up as "metrics" by process_results.py's per-metric
    # aggregation/plots) so it stays pure provenance, checked separately.
    machine_info = _get_machine_info()
    with open(file + "_machine.json", "w") as mf:
        json.dump(machine_info, mf, indent=2)

    wandb.log({
        **{key: val[1] for key, val in metric_res.items()},
        **timing_metrics,
        **energy_metrics,
        **gen_metrics,
        **{"machine_" + key: val for key, val in machine_info.items()},
        "rec_date": str(recomm_date.date()),
    })


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
    use_internal_rfr = True
    use_internal_lgbm = True
    use_internal_lr = True
    tuned = False
    n = 20
    sample_pct = None

    if model_id == RFR:
        n, kpi_type, use_internal_rfr, tuned = _parse_rfr_params(param)
    elif model_id == LGBM:
        n, kpi_type, use_internal_lgbm, tuned = _parse_lgbm_params(param)
    elif model_id == LR:
        kpi_type, use_internal_lr, tuned = _parse_lr_params(param)
    elif model_id == TABPFN:
        kpi_type, sample_pct = _parse_tabpfn_params(param)
    elif model_id == TABICL:
        kpi_type, sample_pct = _parse_tabicl_params(param)
    elif model_id == TABFM:
        kpi_type, sample_pct = _parse_tabfm_params(param)

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
            rfr_kwargs = dict(k=5, kpi_type=kpi_type, kpi_features=feats, random_state=42, n_jobs=-1)
            if tuned:
                best = _load_tuned_params(RFR, kpi_type)
                rfr_kwargs.update(
                    n_estimators=best["n_estimators"],
                    min_samples_leaf=best["min_samples_leaf"],
                    max_depth=best["max_depth"],
                )
            else:
                rfr_kwargs["n_estimators"] = n
            alg_model = RFRKPIModel(**rfr_kwargs)
        else:
            alg_model = RandomForestRegressor(n_estimators=n)
    elif model_id == LGBM:
        if use_internal_lgbm:
            lgbm_kwargs = dict(k=5, kpi_type=kpi_type, kpi_features=feats, random_state=42, n_jobs=-1)
            if tuned:
                best = _load_tuned_params(LGBM, kpi_type)
                lgbm_kwargs.update(
                    n_estimators=best["n_estimators"],
                    num_leaves=best["num_leaves"],
                    min_child_samples=best["min_child_samples"],
                )
            elif n is not None:
                lgbm_kwargs["n_estimators"] = n
            alg_model = LGBMKPIModel(**lgbm_kwargs)
        else:
            alg_model = LGBMRegressor()
    elif model_id == LR:
        if use_internal_lr:
            lr_kwargs = dict(k=5, kpi_type=kpi_type, kpi_features=feats)
            if tuned:
                best = _load_tuned_params(LR, kpi_type)
                lr_kwargs["fit_intercept"] = best["fit_intercept"]
            alg_model = LRKPIModel(**lr_kwargs)
        else:
            alg_model = LinearRegression()
    elif model_id == TABPFN:
        alg_model = TabPFNKPIModel(
            k=5, kpi_type=kpi_type, kpi_features=feats, random_state=42,
            sample_pct=sample_pct,
        )
    elif model_id == TABICL:
        alg_model = TabICLKPIModel(
            k=5, kpi_type=kpi_type, kpi_features=feats, random_state=42,
            sample_pct=sample_pct,
        )
    elif model_id == TABFM:
        alg_model = TabFMKPIModel(
            k=5, kpi_type=kpi_type, kpi_features=feats, random_state=42,
            sample_pct=sample_pct,
        )
    else:
        raise ValueError(f"Unsupported model identifier: {model_id}")

    algorithm = ProfitabilityPrediction(alg_model, financial_data, num_months, feats, -1,
                                        save_for_testing=save_for_testing)
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

    if rec_model == LGBM:
        n, kpi_type, use_internal_lgbm, tuned = _parse_lgbm_params(param)
        name_n = "tuned" if tuned else (str(n) if n is not None else "default")
        algorithm_name = LGBM + "_" + name_n + "_" + kpi_type
        if use_internal_lgbm:
            algorithm_name += "_internal_kpis"
    elif rec_model == LR:
        kpi_type, use_internal_lr, tuned = _parse_lr_params(param)
        name_n = "tuned" if tuned else "default"
        algorithm_name = LR + "_" + name_n + "_" + kpi_type
        if use_internal_lr:
            algorithm_name += "_internal_kpis"
    elif rec_model == TABPFN:
        kpi_type, sample_pct = _parse_tabpfn_params(param)
        algorithm_name = TABPFN + "_" + kpi_type + "_internal_kpis"
        if sample_pct is not None:
            algorithm_name += "_tabpfn_sample" + str(sample_pct)
    elif rec_model == TABICL:
        kpi_type, sample_pct = _parse_tabicl_params(param)
        algorithm_name = TABICL + "_" + kpi_type + "_internal_kpis"
        if sample_pct is not None:
            algorithm_name += "_tabicl_sample" + str(sample_pct)
    elif rec_model == TABFM:
        kpi_type, sample_pct = _parse_tabfm_params(param)
        algorithm_name = TABFM + "_" + kpi_type + "_internal_kpis"
        if sample_pct is not None:
            algorithm_name += "_tabfm_sample" + str(sample_pct)
    else:
        # RFR (internal or external)
        n, kpi_type, use_internal_rfr, tuned = _parse_rfr_params(param)
        name_n = "tuned" if tuned else str(n)
        algorithm_name = RFR + "_" + name_n + "_" + kpi_type
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
    parser_range.add_argument("model", help="model identifier", choices=[RFR, LGBM, TABPFN, TABICL, TABFM, LR])
    parser_range.add_argument("params", help="model parameters", action="store", nargs="*")

    parser_fixed = subparsers.add_parser('fixed_dates', help='List of fixed dates to use. This mode provides fixed '
                                                            'lists of dates for split and test.')
    parser_fixed.add_argument('split_dates', help='Comma separated list of split dates. Date format: %Y-%m-%d')
    parser_fixed.add_argument('future_dates', help='Comma separated list of test end dates. Date format: %Y-%m-%d')
    parser_fixed.add_argument("output_dir", help="directory on which to store the outputs.")
    parser_fixed.add_argument("months", help="number of months to look into the future.")
    parser_fixed.add_argument("model", help="model identifier", choices=[RFR, LGBM, TABPFN, TABICL, TABFM, LR])
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
    use_internal_lr = True
    if model == RFR:
        _, selected_kpi_type, use_internal_rfr, _ = _parse_rfr_params(params)
    elif model == LGBM:
        _, selected_kpi_type, use_internal_lgbm, _ = _parse_lgbm_params(params)
    elif model == LR:
        selected_kpi_type, use_internal_lr, _ = _parse_lr_params(params)

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
    if (model == RFR and not use_internal_rfr) or (model == LGBM and not use_internal_lgbm) or (model == LR and not use_internal_lr):
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

    def_dates = []
    def_future_dates = []
    def_name = []

    # We first check the selected model is good.
    f_name = get_name(model, params)
    if f_name is None:
        print("ERROR: Invalid parameters")
        exit(-1)

    # One run per (experiment date-range, model config) invocation — matches how
    # run_recommendation.py/the slurm scripts shell out one recommendation.py process
    # per config, so progress across this process's windows can be watched live.
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "counterfactualFAR"),
        group=f_name,
        name=f"{f_name}_{min_date:%Y-%m-%d}_{max_date:%Y-%m-%d}",
        config={
            "model": model,
            "params": params,
            "kpi_type": selected_kpi_type,
            "min_date": min_date.isoformat(),
            "max_date": max_date.isoformat(),
            "num_splits": num_splits,
            "num_future": num_future,
            "months": months_term,
        },
    )

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

        # Run directly in the main process so sklearn/lightgbm's n_jobs=-1 gets all
        # CPUs without joblib's nested-parallelism cap inside a subprocess.
        regressor(model, params, splitted_data, rec_date, metrics, directory, alg_name, months_term,
                  save_for_testing)

        # Drop this window's data/algorithm objects before the next window builds
        # its own — see _release_gpu_memory() for why this matters for tabpfn.
        del splitted_data, profitability_df, volatility_df, metrics
        _release_gpu_memory()

    wandb.finish()
