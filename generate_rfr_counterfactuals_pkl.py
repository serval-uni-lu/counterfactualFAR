"""Generate time-series (price-window) counterfactuals with DiCE using internal RFR .pkl.

This script perturbs raw price-window values (`w_0 ... w_n`) and evaluates each candidate
through the saved internal RFR pipeline by regenerating KPIs from the synthetic window.
"""

import argparse
import json
import pickle
import time
import os
from pathlib import Path
from threading import local
from concurrent.futures import ThreadPoolExecutor

import dice_ml
import numpy as np
import pandas as pd
from dice_ml import Dice

from utils.constants import DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL


DEFAULT_MODEL_PKL = Path(
    "for_testing/profitability_recommendation_pipeline_2020-08-28_00-00-00_rfr_n-40_kpi-full_short_internal_kpis.pkl"
)
DEFAULT_TRAINING = Path("for_testing/training_data_2020-08-28_00-00-00_rfr_internal_kpis.csv")
DEFAULT_TESTING = Path("for_testing/testing_data_2020-08-28_00-00-00_rfr_internal_kpis.csv")
DEFAULT_CF_OUT = Path("results/counterfactuals/rfr_pkl_ts_counterfactuals.csv")
DEFAULT_SUMMARY_OUT = Path("results/counterfactuals/rfr_pkl_ts_counterfactuals_summary.csv")
FIXED_WINDOW_SIZE = 21
DEFAULT_MAX_REFERENCE_WINDOWS = 30
DEFAULT_DICE_METHOD = "random"
PREDICT_PROGRESS_EVERY = 25
MIN_POSITIVE_UPLIFT = 1e-6


def _load_model(model_path: Path):
    """Load an internal RFR pipeline object from pickle."""

    with open(model_path, "rb") as handle:
        model = pickle.load(handle)

    if not hasattr(model, "_generate_kpis_df"):
        raise ValueError("Loaded .pkl does not expose _generate_kpis_df (expected internal RFR model)")
    if not hasattr(model, "model"):
        raise ValueError("Loaded .pkl does not expose fitted RF model at .model")
    if not hasattr(model, "kpi_features") or not model.kpi_features:
        raise ValueError("Loaded .pkl has no kpi_features configured")
    return model


def _validate_ts(df: pd.DataFrame, name: str):
    """Validate required raw time-series columns."""

    # These are needed to reconstruct windows and regenerate KPIs.
    needed = {DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_RATING_COL}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{name} must contain {sorted(needed)}; missing={sorted(missing)}")


def validate_no_future_data(df, max_date, context_name):
    future_dates = df[df[DEFAULT_TIMESTAMP_COL] > max_date]
    if not future_dates.empty:
        print(f"WARNING: {context_name} contains {len(future_dates)} future dates!")
        print(f"First future date: {future_dates[DEFAULT_TIMESTAMP_COL].min()}")
        return False
    return True


def build_window_dataset(
    time_series_df: pd.DataFrame,
    window_size: int,
    query_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construct trailing fixed-length windows from raw time-series rows."""

    # Normalize and sort history for deterministic slicing.
    history_df = time_series_df.copy()
    history_df[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(history_df[DEFAULT_TIMESTAMP_COL])
    history_df = history_df.sort_values([DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]).reset_index(drop=True)

    # Query rows can differ from history (e.g., test queries with train+test history).
    if query_df is None:
        query_rows = history_df.copy()
        query_rows["query_index_original"] = query_rows.index.astype(int)
    else:
        query_rows = query_df.copy()
        query_rows["query_index_original"] = query_rows.index.astype(int)
        query_rows[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(query_rows[DEFAULT_TIMESTAMP_COL])
        query_rows = query_rows.sort_values([DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]).reset_index(drop=True)

    # Pre-split by asset once to avoid repeated filtering work in the loop.
    asset_to_series = {
        asset: group.sort_values(DEFAULT_TIMESTAMP_COL).reset_index(drop=True)
        for asset, group in history_df.groupby(DEFAULT_ITEM_COL, sort=False)
    }

    rows = []
    for _, row in query_rows.iterrows():
        item_id = row[DEFAULT_ITEM_COL]
        ts = row[DEFAULT_TIMESTAMP_COL]

        # Fetch one asset timeline.
        asset_series = asset_to_series.get(item_id)
        if asset_series is None or len(asset_series) == 0:
            continue

        asset_series = asset_series[asset_series[DEFAULT_TIMESTAMP_COL] <= ts]

        # Require enough history to fill the requested window.
        if len(asset_series) < window_size:
            continue

        # Flatten trailing price history into window features.
        win = asset_series.tail(window_size)
        feat = {f"w_{i}": float(v) for i, v in enumerate(win[DEFAULT_RATING_COL].values)}
        feat[DEFAULT_ITEM_COL] = item_id
        feat[DEFAULT_TIMESTAMP_COL] = ts
        feat["target"] = float(row.get("target", np.nan))
        feat["query_index_original"] = int(row["query_index_original"])
        rows.append(feat)

    if not rows:
        raise ValueError("No valid windows could be constructed. Increase data or reduce --window-size.")

    return pd.DataFrame(rows).sort_values([DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]).reset_index(drop=True)


class RFRPKLWindowWrapper:
    """DiCE-compatible predictor: price window -> internal KPI generation -> RF prediction."""

    def __init__(self, model, window_cols: list[str], full_time_series: pd.DataFrame, n_jobs: int = 1):
        self.model = model
        self.window_cols = list(window_cols)
        self.window_size = len(self.window_cols)
        self.n_jobs = max(1, int(n_jobs))
        self._rf_jobs_single = self.n_jobs
        self._rf_jobs_per_thread = 1
        kpi_type = str(getattr(self.model, "kpi_type", "full_short")).lower()
        k = int(getattr(self.model, "k", 5))
        periods = [21, 63, 126] if kpi_type == "short" else [21, 63, 126, 189]
        self.min_history_len = max(self.window_size, max(periods) + k)
        if hasattr(self.model, "model") and hasattr(self.model.model, "n_jobs"):
            self.model.model.n_jobs = self._rf_jobs_single
        self._model_pickle = pickle.dumps(self.model)
        self._thread_local = local()
        self.full_time_series = full_time_series.copy()
        self.full_time_series[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(self.full_time_series[DEFAULT_TIMESTAMP_COL])
        self.full_time_series = self.full_time_series.sort_values([DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]).reset_index(drop=True)
        self._context_item = None
        self._context_timestamp = None
        self._context_panel = None
        self._context_series = None

    def _get_thread_model(self):
        if not hasattr(self._thread_local, "model"):
            mdl = pickle.loads(self._model_pickle)
            if hasattr(mdl, "model") and hasattr(mdl.model, "n_jobs"):
                mdl.model.n_jobs = self._rf_jobs_per_thread
            self._thread_local.model = mdl
        return self._thread_local.model

    def set_query_context(self, item_id, timestamp):
        """Set the real asset/timestamp context for one query's CF search."""
        ts = pd.to_datetime(timestamp)
        panel_df = self.full_time_series[self.full_time_series[DEFAULT_TIMESTAMP_COL] <= ts].copy()
        panel_df = panel_df.sort_values([DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]).reset_index(drop=True)

        asset_series = panel_df[panel_df[DEFAULT_ITEM_COL] == item_id]
        asset_series = asset_series[asset_series[DEFAULT_TIMESTAMP_COL] <= ts].copy()
        asset_series = asset_series.sort_values(DEFAULT_TIMESTAMP_COL).reset_index(drop=True)

        if len(asset_series) < self.min_history_len:
            raise ValueError(
                f"Insufficient history for context asset={item_id} at {ts}: "
                f"need at least {self.min_history_len} rows for internal RFR prediction, got {len(asset_series)}"
            )

        self._context_item = item_id
        self._context_timestamp = ts
        self._context_panel = panel_df
        self._context_series = asset_series

    def _predict_from_ts_context(self, ts_df: pd.DataFrame, context_item, context_timestamp, model_obj=None) -> float:
        """Score one contextual raw time-series frame via the exact internal RFR path."""
        model_obj = self.model if model_obj is None else model_obj
        try:
            all_preds = np.asarray(model_obj.predict(ts_df)).reshape(-1)
        except ValueError as error:
            message = str(error)
            if "Found array with 0 sample(s)" in message:
                raise ValueError(
                    "Internal RFR produced no KPI rows for this context. "
                    "With test-only history, select a later --query-index or use a dataset with longer test history."
                ) from error
            raise

        kpis_df = getattr(getattr(model_obj, "transformer", None), "last_kpis_df_", None)
        if kpis_df is None or kpis_df.empty:
            raise ValueError("Internal RFR prediction did not expose KPI rows via transformer.last_kpis_df_")
        kpis_df = kpis_df.reset_index(drop=True)

        if context_timestamp is not None:
            selected = kpis_df[
                (kpis_df[DEFAULT_ITEM_COL] == context_item)
                & (kpis_df[DEFAULT_TIMESTAMP_COL] == context_timestamp)
            ]
            if selected.empty:
                selected = kpis_df[kpis_df[DEFAULT_ITEM_COL] == context_item].tail(1)
        else:
            selected = kpis_df[kpis_df[DEFAULT_ITEM_COL] == context_item].tail(1)

        if selected.empty:
            raise ValueError("Unable to locate KPI row for contextual internal RFR prediction")

        selected_idx = int(selected.index[-1])
        if selected_idx < 0 or selected_idx >= len(all_preds):
            raise ValueError(
                "Selected KPI row index is out of bounds for internal prediction output: "
                f"idx={selected_idx}, n_preds={len(all_preds)}"
            )
        return float(all_preds[selected_idx])

    def predict(self, X):
        # Accept both DataFrame and ndarray inputs because DiCE may use either.
        if isinstance(X, pd.DataFrame):
            arr = X[self.window_cols].astype(np.float64).values
        elif isinstance(X, np.ndarray):
            arr = np.asarray(X, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] != len(self.window_cols):
                raise ValueError(f"Expected {len(self.window_cols)} features, got {arr.shape[1]}")
        else:
            raise TypeError(f"Expected pandas DataFrame or numpy.ndarray, got {type(X).__name__}")

        # Evaluate each candidate window by mutating contextual raw time-series.
        # If a batch is provided, optionally parallelize row scoring with thread-local
        # model copies to avoid shared-state races in transformer.last_kpis_df_.
        use_parallel = arr.shape[0] > 1 and self.n_jobs > 1

        print(f"Entered wrapper.predict with batch={arr.shape[0]}", flush=True)

        def _score_one(task) -> float:
            idx, row_values = task
            total = arr.shape[0]
            if total <= 10 or (idx + 1) % PREDICT_PROGRESS_EVERY == 0 or idx == 0 or (idx + 1) == total:
                print(f"Processing candidate window {idx + 1}/{total}", flush=True)
            context_item = self._context_item
            context_timestamp = self._context_timestamp

            if self._context_panel is None or self._context_item is None or self._context_timestamp is None:
                raise ValueError(
                    "Query context is not set. Call set_query_context() before predict(); "
                    "synthetic fallback windows are disabled."
                )

            ts_df = self._context_panel.copy()
            item_mask = ts_df[DEFAULT_ITEM_COL] == context_item
            item_indices = ts_df.index[item_mask].to_numpy()
            if len(item_indices) < self.window_size:
                raise ValueError(
                    f"Insufficient context rows for CF window replacement on asset={context_item}: "
                    f"need {self.window_size}, got {len(item_indices)}"
                )
            target_indices = item_indices[-self.window_size :]
            ts_df.loc[target_indices, DEFAULT_RATING_COL] = row_values.astype(float)

            if use_parallel:
                model_obj = self._get_thread_model()
            else:
                model_obj = self.model
            return self._predict_from_ts_context(ts_df, context_item, context_timestamp, model_obj=model_obj)

        if not use_parallel:
            preds = []
            for i in range(arr.shape[0]):
                preds.append(_score_one((i, arr[i])))
            return np.asarray(preds, dtype=np.float64)

        with ThreadPoolExecutor(max_workers=min(self.n_jobs, arr.shape[0])) as executor:
            tasks = [(i, arr[i]) for i in range(arr.shape[0])]
            preds = list(executor.map(_score_one, tasks))
        return np.asarray(preds, dtype=np.float64)


def _build_kpi_predictions(model, series_df: pd.DataFrame, recommend_date: pd.Timestamp) -> pd.DataFrame:
    """Generate KPI rows and model predictions from all rows up to recommend_date."""

    history_cut = series_df[series_df[DEFAULT_TIMESTAMP_COL] <= recommend_date].copy()
    validate_no_future_data(history_cut, recommend_date, "history_cut_before_kpi_generation")
    if history_cut.empty:
        raise ValueError("No rows available on or before recommend-date")

    kpis_df = model._generate_kpis_df(history_cut)
    if kpis_df is None or kpis_df.empty:
        raise ValueError("Internal KPI generation returned no rows for the selected recommend-date")

    missing = [col for col in model.kpi_features if col not in kpis_df.columns]
    if missing:
        raise ValueError(f"Missing KPI feature columns for prediction: {missing}")

    preds = model.model.predict(kpis_df[model.kpi_features])
    scored = kpis_df[[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]].copy()
    scored["prediction"] = np.asarray(preds, dtype=np.float64)
    return scored


def main():
    parser = argparse.ArgumentParser(
        description="Generate RFR counterfactuals at a recommendation date (no target required)"
    )
    parser.add_argument("--model-pkl", type=Path, default=DEFAULT_MODEL_PKL)
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--testing", type=Path, default=DEFAULT_TESTING)
    parser.add_argument("--asset-id", type=str, default=None, help="Optional single asset to process")
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--method", type=str, default=DEFAULT_DICE_METHOD, choices=["genetic", "random", "kdtree"])
    parser.add_argument("--maxiterations", type=int, default=3)
    parser.add_argument("--total-cfs", type=int, default=1)
    parser.add_argument(
        "--max-reference-windows",
        type=int,
        default=DEFAULT_MAX_REFERENCE_WINDOWS,
        help="Maximum number of reference windows used by DiCE per query (<=0 disables cap)",
    )
    parser.add_argument(
        "--desired-max",
        type=float,
        default=None,
        help="Optional global upper bound for desired prediction range",
    )
    parser.add_argument("--out-cf", type=Path, default=DEFAULT_CF_OUT)
    parser.add_argument("--out-summary", type=Path, default=DEFAULT_SUMMARY_OUT)
    args = parser.parse_args()

    model = _load_model(args.model_pkl)

    training_ts = pd.read_csv(args.training)
    _validate_ts(training_ts, "training")
    training_ts[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(training_ts[DEFAULT_TIMESTAMP_COL])
    training_ts = training_ts.sort_values([DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]).reset_index(drop=True)

    testing_ts = pd.read_csv(args.testing)
    _validate_ts(testing_ts, "testing")
    testing_ts[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(testing_ts[DEFAULT_TIMESTAMP_COL])
    testing_ts = testing_ts.sort_values([DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]).reset_index(drop=True)

    if args.window_size is not None and int(args.window_size) != FIXED_WINDOW_SIZE:
        raise ValueError(f"This script is fixed to window-size={FIXED_WINDOW_SIZE}")

    window_size = FIXED_WINDOW_SIZE
    window_cols = [f"w_{i}" for i in range(window_size)]
    print(f"Using fixed window-size={window_size}")

    full_history = pd.concat([training_ts, testing_ts], ignore_index=True)
    full_history = full_history.drop_duplicates(subset=[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_RATING_COL])
    full_history = full_history.sort_values([DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]).reset_index(drop=True)
    max_test_ts = testing_ts[DEFAULT_TIMESTAMP_COL].max()
    validate_no_future_data(full_history, max_test_ts, "full_history")

    query_rows = testing_ts.copy()
    if query_rows.empty:
        raise ValueError("No rows in --testing; cannot build factual windows")

    if args.asset_id is not None:
        query_rows = query_rows[query_rows[DEFAULT_ITEM_COL] == args.asset_id].copy()
        if query_rows.empty:
            raise ValueError(f"No query rows found for asset-id={args.asset_id} in --testing")

    query_windows = build_window_dataset(full_history, window_size, query_df=query_rows)
    if query_windows.empty:
        raise ValueError("No valid query windows in --testing")

    n_jobs = os.cpu_count() if int(args.n_jobs) == -1 else int(args.n_jobs)
    wrapper = RFRPKLWindowWrapper(model, window_cols, full_time_series=full_history, n_jobs=max(1, n_jobs))
    dice_model = dice_ml.Model(model=wrapper, backend="sklearn", model_type="regressor")

    args.out_cf.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)

    prediction_columns = [
        "query_index",
        DEFAULT_ITEM_COL,
        DEFAULT_TIMESTAMP_COL,
        "cf_index",
        "factual_prediction",
        "cf_prediction",
        "desired_min",
        "desired_max",
    ]
    window_columns = [
        "query_index",
        DEFAULT_ITEM_COL,
        DEFAULT_TIMESTAMP_COL,
        "cf_index",
        "row_type",
        "window_line",
    ]

    pd.DataFrame(columns=prediction_columns).to_csv(args.out_cf, index=False)
    pd.DataFrame(columns=window_columns).to_csv(args.out_summary, index=False)

    print(f"Found {len(query_windows)} query windows from --testing")

    for _, query_row in query_windows.iterrows():
        query_row = query_row.to_frame().T.copy()
        q_idx = int(query_row.iloc[0]["query_index_original"])
        query_asset_id = query_row.iloc[0][DEFAULT_ITEM_COL]
        query_rec_time = pd.to_datetime(query_row.iloc[0][DEFAULT_TIMESTAMP_COL])
        query_x = query_row[window_cols]

        print(
            f"Starting query-index {q_idx} | asset={query_asset_id} | "
            f"timestamp={query_rec_time} | method={args.method}",
            flush=True,
        )

        wrapper.set_query_context(query_asset_id, query_rec_time)

        history_up_to_query = full_history[full_history[DEFAULT_TIMESTAMP_COL] <= query_rec_time].copy()
        validate_no_future_data(history_up_to_query, query_rec_time, "history_up_to_query")
        if history_up_to_query.empty:
            print(f"Skipping query-index {q_idx}: no history available up to query timestamp")
            continue

        kpi_scored = _build_kpi_predictions(model, history_up_to_query, query_rec_time)

        reference_rows = history_up_to_query[history_up_to_query[DEFAULT_TIMESTAMP_COL] < query_rec_time].copy()
        if reference_rows.empty:
            print(f"Skipping query-index {q_idx}: no historical rows before query timestamp")
            continue

        reference_windows = build_window_dataset(history_up_to_query, window_size, query_df=reference_rows)
        reference_windows = reference_windows.merge(
            kpi_scored,
            on=[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL],
            how="inner",
        )
        reference_windows = reference_windows.dropna(subset=["prediction"])
        if reference_windows.empty:
            print(f"Skipping query-index {q_idx}: reference windows could not be aligned with predictions")
            continue

        factual_pred = float(wrapper.predict(query_x)[0])
        desired_min = factual_pred + MIN_POSITIVE_UPLIFT

        if args.desired_max is None:
            asset_ref_max = reference_windows[
                reference_windows[DEFAULT_ITEM_COL] == query_asset_id
            ]["prediction"].max()
            if pd.isna(asset_ref_max):
                asset_ref_max = reference_windows["prediction"].max()
            desired_max = max(float(asset_ref_max), desired_min + MIN_POSITIVE_UPLIFT)
        else:
            desired_max = float(args.desired_max)
            if desired_max <= desired_min:
                desired_max = desired_min + MIN_POSITIVE_UPLIFT

        dice_train_query = reference_windows[
            (reference_windows[DEFAULT_ITEM_COL] == query_asset_id)
            & (reference_windows[DEFAULT_TIMESTAMP_COL] < query_rec_time)
        ].copy()
        if dice_train_query.empty:
            dice_train_query = reference_windows[
                reference_windows[DEFAULT_TIMESTAMP_COL] < query_rec_time
            ].copy()
        if dice_train_query.empty:
            print(f"Skipping query-index {q_idx}: no reference windows before recommendation time")
            continue

        max_reference_windows = int(args.max_reference_windows)
        if max_reference_windows > 0 and len(dice_train_query) > max_reference_windows:
            dice_train_query = dice_train_query.tail(max_reference_windows).copy()

        dice_data_query = dice_ml.Data(
            dataframe=dice_train_query[window_cols + ["prediction"]],
            continuous_features=window_cols,
            outcome_name="prediction",
        )
        exp_query = Dice(dice_data_query, dice_model, method=args.method)

        cf_kwargs = {
            "total_CFs": int(args.total_cfs),
            "desired_range": [desired_min, desired_max],
            "features_to_vary": list(window_cols),
        }
        if args.method == "genetic":
            cf_kwargs["maxiterations"] = int(args.maxiterations)

        t0 = time.time()
        dice_exp = exp_query.generate_counterfactuals(query_x, **cf_kwargs)
        elapsed = time.time() - t0
        print(f"Finished CF search for query-index {q_idx} in {elapsed:.1f}s")
        final_cfs = dice_exp.cf_examples_list[0].final_cfs_df.copy()
        if final_cfs is None or final_cfs.empty:
            continue

        for cf_idx, (_, cf_row) in enumerate(final_cfs.iterrows()):
            cf_x = pd.DataFrame([cf_row[window_cols].to_dict()])
            cf_pred = float(wrapper.predict(cf_x)[0])

            factual_window_map = {col: float(query_row.iloc[0][col]) for col in window_cols}
            cf_window_map = {col: float(cf_row[col]) for col in window_cols}

            prediction_row = {
                "query_index": q_idx,
                DEFAULT_ITEM_COL: query_asset_id,
                DEFAULT_TIMESTAMP_COL: query_rec_time,
                "cf_index": cf_idx,
                "factual_prediction": factual_pred,
                "cf_prediction": float(cf_pred),
                "desired_min": desired_min,
                "desired_max": desired_max,
            }

            factual_window_row = {
                "query_index": q_idx,
                DEFAULT_ITEM_COL: query_asset_id,
                DEFAULT_TIMESTAMP_COL: query_rec_time,
                "cf_index": cf_idx,
                "row_type": "factual",
                "window_line": json.dumps(factual_window_map),
            }
            counterfactual_window_row = {
                "query_index": q_idx,
                DEFAULT_ITEM_COL: query_asset_id,
                DEFAULT_TIMESTAMP_COL: query_rec_time,
                "cf_index": cf_idx,
                "row_type": "counterfactual",
                "window_line": json.dumps(cf_window_map),
            }

            pd.DataFrame([prediction_row], columns=prediction_columns).to_csv(
                args.out_cf,
                mode="a",
                header=False,
                index=False,
            )
            pd.DataFrame([factual_window_row, counterfactual_window_row], columns=window_columns).to_csv(
                args.out_summary,
                mode="a",
                header=False,
                index=False,
            )

    print(f"Saved counterfactuals: {args.out_cf}")
    print(f"Saved summary: {args.out_summary}")


if __name__ == "__main__":
    main()
