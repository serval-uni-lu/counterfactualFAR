"""Generate time-series (price-window) counterfactuals with DiCE using internal RFR .pkl.

This script perturbs raw price-window values (`w_0 ... w_n`) and evaluates each candidate
through the saved internal RFR pipeline by regenerating KPIs from the synthetic window.
"""

import argparse
import json
import pickle
import re
import time
import os
from pathlib import Path
from threading import local, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

import dice_ml
import numpy as np
import pandas as pd
from dice_ml import Dice
from raiutils.exceptions import UserConfigValidationException

from utils.constants import DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL


_ARTIFACTS_DIR = Path("artifacts_for_counterfactuals")
_MODEL_TAG = "rfr_n-100_kpi-full_short_internal_kpis"

# Last window of each experiment:
#   exp1: 2019-08-01 → 2020-08-28
#   exp2: 2020-09-14 → 2021-11-23
DEFAULT_EXPERIMENT_PKLS = [
    _ARTIFACTS_DIR / _MODEL_TAG / f"profitability_recommendation_pipeline_2020-08-28_00-00-00_{_MODEL_TAG}.pkl",
    _ARTIFACTS_DIR / _MODEL_TAG / f"profitability_recommendation_pipeline_2021-11-23_00-00-00_{_MODEL_TAG}.pkl",
]
FIXED_WINDOW_SIZE = 21
DEFAULT_MAX_REFERENCE_WINDOWS = 100
DEFAULT_DICE_METHOD = "genetic"
DEFAULT_MAXITERATIONS = 10
MIN_POSITIVE_UPLIFT = 1e-6
MAX_PREDICT_CALLS = 500  # hard limit on predict() calls per query to prevent runaway genetic search


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
    """DiCE-compatible predictor: price window -> internal KPI generation -> RF prediction.

    Query context (asset, timestamp, panel, series) is stored in thread-local storage so
    that multiple queries can be evaluated concurrently without state conflicts.
    """

    def __init__(self, model, window_cols: list[str], full_time_series: pd.DataFrame, n_jobs: int = 1):
        self.model = model
        self.window_cols = list(window_cols)
        self.window_size = len(self.window_cols)
        self.n_jobs = max(1, int(n_jobs))
        k = int(getattr(self.model, "k", 5))
        kpi_features = getattr(self.model, "kpi_features", None) or []
        period_values = [int(m) for f in kpi_features for m in re.findall(r"(\d+)d", f)]
        max_period = max(period_values) if period_values else 126
        self.min_history_len = max(self.window_size, max_period + k)
        if hasattr(self.model, "model") and hasattr(self.model.model, "n_jobs"):
            self.model.model.n_jobs = 1
        self._model_pickle = pickle.dumps(self.model)
        self._thread_local = local()
        self.full_time_series = full_time_series.copy()
        self.full_time_series[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(self.full_time_series[DEFAULT_TIMESTAMP_COL])
        self.full_time_series = self.full_time_series.sort_values([DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Thread-local query context properties
    # Context is keyed to the calling thread so concurrent queries are safe.
    # ------------------------------------------------------------------

    @property
    def _context_item(self):
        return getattr(self._thread_local, "ctx_item", None)

    @_context_item.setter
    def _context_item(self, v):
        self._thread_local.ctx_item = v

    @property
    def _context_timestamp(self):
        return getattr(self._thread_local, "ctx_ts", None)

    @_context_timestamp.setter
    def _context_timestamp(self, v):
        self._thread_local.ctx_ts = v

    @property
    def _context_panel(self):
        return getattr(self._thread_local, "ctx_panel", None)

    @_context_panel.setter
    def _context_panel(self, v):
        self._thread_local.ctx_panel = v

    @property
    def _context_series(self):
        return getattr(self._thread_local, "ctx_series", None)

    @_context_series.setter
    def _context_series(self, v):
        self._thread_local.ctx_series = v

    @property
    def _kpi_fallback_warned(self):
        return getattr(self._thread_local, "kpi_warned", False)

    @_kpi_fallback_warned.setter
    def _kpi_fallback_warned(self, v):
        self._thread_local.kpi_warned = v

    # ------------------------------------------------------------------

    def _get_thread_model(self):
        """Return a thread-local model copy (unpickled once per thread)."""
        if not hasattr(self._thread_local, "model"):
            mdl = pickle.loads(self._model_pickle)
            if hasattr(mdl, "model") and hasattr(mdl.model, "n_jobs"):
                mdl.model.n_jobs = 1
            self._thread_local.model = mdl
        return self._thread_local.model

    def set_query_context(self, item_id, timestamp):
        """Set the real asset/timestamp context for one query's CF search (thread-local)."""
        ts = pd.to_datetime(timestamp)
        panel_df = self.full_time_series[self.full_time_series[DEFAULT_TIMESTAMP_COL] <= ts].copy()
        panel_df = panel_df.sort_values([DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]).reset_index(drop=True)

        asset_series = panel_df[panel_df[DEFAULT_ITEM_COL] == item_id]
        asset_series = asset_series[asset_series[DEFAULT_TIMESTAMP_COL] <= ts].copy()
        asset_series = asset_series.sort_values(DEFAULT_TIMESTAMP_COL).reset_index(drop=True)

        if len(asset_series) < self.min_history_len:
            raise ValueError(
                f"Insufficient history for asset={item_id} at {ts}: "
                f"need {self.min_history_len} rows, got {len(asset_series)}"
            )

        self._context_item = item_id
        self._context_timestamp = ts
        self._context_panel = panel_df
        self._context_series = asset_series
        self._kpi_fallback_warned = False
        self._thread_local.predict_call_count = 0

    def _predict_from_ts_context(self, ts_df: pd.DataFrame, context_item, context_timestamp, model_obj=None) -> float:
        """Score one contextual raw time-series frame via the exact internal RFR path."""
        model_obj = self._get_thread_model() if model_obj is None else model_obj
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
                fallback = kpis_df[kpis_df[DEFAULT_ITEM_COL] == context_item].tail(1)
                if not fallback.empty and not self._kpi_fallback_warned:
                    fallback_ts = fallback.iloc[0][DEFAULT_TIMESTAMP_COL]
                    print(
                        f"WARNING: KPI row for ({context_item}, {context_timestamp}) not found. "
                        f"Scoring via most recent KPI row at {fallback_ts} for all candidates in this query.",
                        flush=True,
                    )
                    self._kpi_fallback_warned = True
                selected = fallback
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
        """Evaluate candidate windows against the thread-local query context."""
        self._thread_local.predict_call_count = getattr(self._thread_local, "predict_call_count", 0) + 1
        if self._thread_local.predict_call_count > MAX_PREDICT_CALLS:
            raise RuntimeError(
                f"predict() call limit ({MAX_PREDICT_CALLS}) exceeded for asset={self._context_item} — "
                "aborting runaway DiCE search."
            )

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

        context_item = self._context_item
        context_timestamp = self._context_timestamp
        context_panel = self._context_panel
        if context_panel is None or context_item is None or context_timestamp is None:
            raise ValueError(
                "Query context is not set. Call set_query_context() before predict(); "
                "synthetic fallback windows are disabled."
            )

        # Always use the thread-local model copy so concurrent queries do not share
        # transformer.last_kpis_df_ state and overwrite each other's KPI rows.
        model_obj = self._get_thread_model()
        total = arr.shape[0]
        preds = []
        for idx in range(total):
            print(
                f"Processing candidate window {idx + 1}/{total} "
                f"(candidate_index={idx + 1}, asset={context_item})",
                flush=True,
            )
            ts_df = context_panel[context_panel[DEFAULT_ITEM_COL] == context_item].copy()
            item_indices = ts_df.index.to_numpy()
            if len(item_indices) < self.window_size:
                raise ValueError(
                    f"Insufficient context rows for CF window replacement on asset={context_item}: "
                    f"need {self.window_size}, got {len(item_indices)}"
                )
            target_indices = item_indices[-self.window_size:]
            ts_df.loc[target_indices, DEFAULT_RATING_COL] = arr[idx].astype(float)
            preds.append(self._predict_from_ts_context(ts_df, context_item, context_timestamp, model_obj=model_obj))
        return np.asarray(preds, dtype=np.float64)


def _cf_utility_metrics(
    factual_win: np.ndarray,
    cf_win: np.ndarray,
    factual_pred: float,
    cf_pred: float,
    desired_min: float,
    desired_max: float,
) -> dict:
    """Compute standard counterfactual utility metrics for one (factual, CF) pair.

    Metrics
    -------
    validity        : 1 if cf_pred is within [desired_min, desired_max], else 0.
    lift            : cf_pred - factual_pred  (signed improvement in profitability score).
    l1_dist         : sum of absolute price changes across the window (total movement).
    l2_dist         : Euclidean distance between factual and CF price windows.
    mean_abs_delta  : l1_dist / window_size  (average per-step absolute price change).
    mean_rel_delta  : mean of |delta_i / factual_i| where factual_i != 0
                      (average relative price change; scale-free).
    n_changed       : number of window steps where the price actually changed (> 1e-8).
    sparsity        : n_changed / window_size  (0 = identical, 1 = all steps changed).
    max_abs_delta   : largest absolute price change at any single window step.
    max_rel_delta   : largest relative price change at any single window step.
    """
    delta = cf_win - factual_win
    abs_delta = np.abs(delta)
    changed_mask = abs_delta > 1e-8
    n = len(factual_win)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_delta = np.where(np.abs(factual_win) > 1e-8, abs_delta / np.abs(factual_win), 0.0)

    return {
        "validity": int(desired_min <= cf_pred <= desired_max),
        "lift": round(float(cf_pred - factual_pred), 8),
        "l1_dist": round(float(abs_delta.sum()), 8),
        "l2_dist": round(float(np.sqrt((delta ** 2).sum())), 8),
        "mean_abs_delta": round(float(abs_delta.mean()), 8),
        "mean_rel_delta": round(float(rel_delta.mean()), 8),
        "n_changed": int(changed_mask.sum()),
        "sparsity": round(float(changed_mask.sum() / n), 6),
        "max_abs_delta": round(float(abs_delta.max()), 8),
        "max_rel_delta": round(float(rel_delta.max()), 8),
    }


def _window_to_timeseries(
    prices: np.ndarray,
    timestamps,
    item_id,
    query_index: int,
) -> pd.DataFrame:
    """Return one row: the CF price at the query timestamp (last step of the window)."""
    return pd.DataFrame([{
        "query_index": query_index,
        DEFAULT_ITEM_COL: item_id,
        DEFAULT_TIMESTAMP_COL: timestamps[-1],
        DEFAULT_RATING_COL: float(prices[-1]),
    }])


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


def _derive_data_paths(pkl_path: Path) -> tuple[Path, Path]:
    """Derive training and testing CSV paths from the pkl path using the naming convention.

    Convention: profitability_recommendation_pipeline_{date}_{model}.pkl
             →  training_data_{date}_{model}.csv / testing_data_{date}_{model}.csv
    """
    suffix = pkl_path.stem.replace("profitability_recommendation_pipeline_", "")
    parent = pkl_path.parent
    return (
        parent / f"training_data_{suffix}.csv",
        parent / f"testing_data_{suffix}.csv",
    )


def _derive_output_paths(pkl_path: Path, method: str) -> tuple[Path, Path, Path]:
    """Derive the three output CSV paths from the pkl path, including the CF method."""
    pkl_name = pkl_path.stem
    model_dir = pkl_path.parent.name
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", pkl_name)
    date_tag = date_match.group(1) if date_match else "unknown_date"
    out_dir = Path("counterfactuals") / model_dir
    tag = f"{model_dir}_{date_tag}_{method}"
    return (
        out_dir / f"cf_details_{tag}.csv",
        out_dir / f"summary_{tag}.csv",
        out_dir / f"cf_timeseries_{tag}.csv",
    )


def _run_for_pkl(pkl_path: Path, training_path: Path, testing_path: Path,
                 out_cf: Path, out_summary: Path, out_timeseries: Path, args) -> None:
    """Run CF generation for one pkl / window."""

    print(f"\n{'='*70}", flush=True)
    print(f"PKL: {pkl_path}", flush=True)
    print(f"{'='*70}", flush=True)

    model = _load_model(pkl_path)

    training_ts = pd.read_csv(training_path)
    _validate_ts(training_ts, "training")
    training_ts[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(training_ts[DEFAULT_TIMESTAMP_COL])
    training_ts = training_ts.sort_values([DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]).reset_index(drop=True)

    testing_ts = pd.read_csv(testing_path)
    _validate_ts(testing_ts, "testing")
    testing_ts[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(testing_ts[DEFAULT_TIMESTAMP_COL])
    testing_ts = testing_ts.sort_values([DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]).reset_index(drop=True)

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
        raise ValueError("No rows in testing CSV; cannot build factual windows")

    if args.asset_id is not None:
        query_rows = query_rows[query_rows[DEFAULT_ITEM_COL] == args.asset_id].copy()
        if query_rows.empty:
            raise ValueError(f"No query rows found for asset-id={args.asset_id} in testing CSV")

    query_windows = build_window_dataset(full_history, window_size, query_df=query_rows)
    if query_windows.empty:
        raise ValueError("No valid query windows in testing CSV")

    n_workers = os.cpu_count() if int(args.n_jobs) == -1 else max(1, int(args.n_jobs))
    wrapper = RFRPKLWindowWrapper(model, window_cols, full_time_series=full_history, n_jobs=1)
    dice_model = dice_ml.Model(model=wrapper, backend="sklearn", model_type="regressor")

    out_cf.parent.mkdir(parents=True, exist_ok=True)

    # Resume: find query indices already written to any of the three output files.
    done_query_indices: set[int] = set()
    if getattr(args, "resume", False):
        for fpath in [out_cf, out_summary, out_timeseries]:
            if fpath.exists() and fpath.stat().st_size > 0:
                try:
                    existing = pd.read_csv(fpath, usecols=["query_index"])
                    indices = set(existing["query_index"].dropna().astype(int).tolist())
                    done_query_indices |= indices
                except Exception as e:
                    print(f"WARNING: Could not read {fpath.name} for resume ({e})", flush=True)
        if done_query_indices:
            print(
                f"Resuming: found {len(done_query_indices)} already-processed query indices "
                f"(from cf_details / summary / timeseries)",
                flush=True,
            )

    # If re-running a specific query, remove its existing rows from all output files first.
    if getattr(args, "query_index", None) is not None and getattr(args, "resume", False):
        rerun_idx = int(args.query_index)
        for fpath in [out_cf, out_summary, out_timeseries]:
            if fpath.exists() and fpath.stat().st_size > 0:
                try:
                    df = pd.read_csv(fpath)
                    df = df[df["query_index"] != rerun_idx]
                    df.to_csv(fpath, index=False)
                    print(f"Removed existing rows for query_index={rerun_idx} from {fpath.name}", flush=True)
                except Exception as e:
                    print(f"WARNING: Could not clean {fpath.name} for rerun ({e})", flush=True)

    prediction_columns = [
        "query_index",
        DEFAULT_ITEM_COL,
        DEFAULT_TIMESTAMP_COL,
        "cf_index",
        "factual_rating",     # last price in the factual window (at query timestamp)
        "cf_rating",          # last price in the CF window (at query timestamp)
        "factual_prediction",
        "cf_prediction",
        "desired_min",
        "desired_max",
        # Utility metrics
        "validity",       # 1 if cf_pred in [desired_min, desired_max]
        "lift",           # cf_pred - factual_pred
        "l1_dist",        # total absolute price movement across the window
        "l2_dist",        # Euclidean distance between factual and CF windows
        "mean_abs_delta", # l1_dist / window_size
        "mean_rel_delta", # mean relative price change (scale-free)
        "n_changed",      # number of window steps that changed
        "sparsity",       # n_changed / window_size (0=identical, 1=all changed)
        "max_abs_delta",  # largest price change at any single step
        "max_rel_delta",  # largest relative change at any single step
    ]
    window_columns = [
        "query_index",
        DEFAULT_ITEM_COL,
        DEFAULT_TIMESTAMP_COL,
        "cf_index",
        "row_type",
        "window_line",
    ]
    timeseries_columns = [
        "query_index",
        DEFAULT_ITEM_COL,
        DEFAULT_TIMESTAMP_COL,
        DEFAULT_RATING_COL,
    ]

    if not done_query_indices:
        pd.DataFrame(columns=prediction_columns).to_csv(out_cf, index=False)
        pd.DataFrame(columns=window_columns).to_csv(out_summary, index=False)
        pd.DataFrame(columns=timeseries_columns).to_csv(out_timeseries, index=False)

    print(f"Found {len(query_windows)} query windows from testing CSV")
    print(f"Running with {n_workers} parallel worker(s)")

    write_lock = Lock()

    def _write_results(pred_rows, summary_rows, ts_dfs):
        """Append one query's results to all three output files (call under write_lock)."""
        for row in pred_rows:
            pd.DataFrame([row], columns=prediction_columns).to_csv(
                out_cf, mode="a", header=False, index=False)
        for row in summary_rows:
            pd.DataFrame([row], columns=window_columns).to_csv(
                out_summary, mode="a", header=False, index=False)
        for ts_df in ts_dfs:
            ts_df[timeseries_columns].to_csv(
                out_timeseries, mode="a", header=False, index=False)

    def _run_query(q_idx, row_series):
        """Process one query. Returns (pred_rows, summary_rows, ts_dfs, was_skipped).

        was_skipped=True  → insufficient data / context error (counted in skipped tally).
        pred_rows=None    → CF generation ran but found nothing (not counted as skipped).
        """
        query_asset_id = row_series[DEFAULT_ITEM_COL]
        query_rec_time = pd.to_datetime(row_series[DEFAULT_TIMESTAMP_COL])
        query_row_df = row_series.to_frame().T.copy()
        query_x = query_row_df[window_cols]

        print(
            f"Starting query-index {q_idx} | asset={query_asset_id} | "
            f"timestamp={query_rec_time} | method={args.method}",
            flush=True,
        )

        try:
            wrapper.set_query_context(query_asset_id, query_rec_time)
        except ValueError as e:
            print(f"Skipping query-index {q_idx}: {e}", flush=True)
            return None, None, None, True

        window_timestamps = wrapper._context_series.tail(window_size)[DEFAULT_TIMESTAMP_COL].values

        history_up_to_query = full_history[
            (full_history[DEFAULT_TIMESTAMP_COL] <= query_rec_time) &
            (full_history[DEFAULT_ITEM_COL] == query_asset_id)
        ].copy()
        validate_no_future_data(history_up_to_query, query_rec_time, "history_up_to_query")
        if history_up_to_query.empty:
            print(f"Skipping query-index {q_idx}: no history available up to query timestamp", flush=True)
            return None, None, None, True

        # Use a thread-local model copy so concurrent _generate_kpis_df calls do not
        # overwrite each other's transformer.last_kpis_df_ state.
        thread_model = wrapper._get_thread_model()
        try:
            kpi_scored = _build_kpi_predictions(thread_model, history_up_to_query, query_rec_time)
        except Exception as e:
            print(f"Skipping query-index {q_idx}: KPI generation failed: {e}", flush=True)
            return None, None, None, True

        reference_rows = history_up_to_query[history_up_to_query[DEFAULT_TIMESTAMP_COL] < query_rec_time].copy()
        if reference_rows.empty:
            print(f"Skipping query-index {q_idx}: no historical rows before query timestamp", flush=True)
            return None, None, None, True

        reference_windows = build_window_dataset(history_up_to_query, window_size, query_df=reference_rows)
        reference_windows = reference_windows.merge(
            kpi_scored,
            on=[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL],
            how="inner",
        )
        reference_windows = reference_windows.dropna(subset=["prediction"])
        if reference_windows.empty:
            print(f"Skipping query-index {q_idx}: reference windows could not be aligned with predictions", flush=True)
            return None, None, None, True

        factual_pred = float(wrapper.predict(query_x)[0])
        # TODO: set desired_min = top-k recommendation threshold at query_rec_time instead of
        # factual_pred + ε. The meaningful CF for the recommendation goal is "what price pattern
        # would push this asset into the top-k ranked assets", not just "any improvement".
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
            print(
                f"Skipping query-index {q_idx} | asset={query_asset_id}: "
                f"factual score ({factual_pred:.6f}) already at or above reference max ({desired_max:.6f}); no room for improvement.",
                flush=True,
            )
            return None, None, None, True

        dice_train_query = reference_windows[
            (reference_windows[DEFAULT_ITEM_COL] == query_asset_id)
            & (reference_windows[DEFAULT_TIMESTAMP_COL] < query_rec_time)
        ].copy()
        if dice_train_query.empty:
            print(
                f"WARNING: No historical windows for asset={query_asset_id} before {query_rec_time}. "
                "Falling back to all-asset reference windows. CFs may reflect another asset's price scale.",
                flush=True,
            )
            dice_train_query = reference_windows[
                reference_windows[DEFAULT_TIMESTAMP_COL] < query_rec_time
            ].copy()
        if dice_train_query.empty:
            print(f"Skipping query-index {q_idx}: no reference windows before recommendation time", flush=True)
            return None, None, None, True

        max_reference_windows = int(args.max_reference_windows)
        if max_reference_windows > 0 and len(dice_train_query) > max_reference_windows:
            dice_train_query = dice_train_query.tail(max_reference_windows).copy()

        n_unique_windows = len(dice_train_query.drop_duplicates(subset=window_cols))
        population_size = int(args.total_cfs) + 17
        n_above_desired_min = (dice_train_query["prediction"] >= desired_min).sum()
        n_in_desired_range = (
            (dice_train_query["prediction"] >= desired_min) &
            (dice_train_query["prediction"] <= desired_max)
        ).sum()

        effective_method = args.method

        if effective_method == "kdtree":
            # For kdtree, pre-filter dice_train_query to only windows already within the
            # desired range. This guarantees every candidate in the KD-tree is valid,
            # preventing the verification-mismatch infinite loop where DiCE finds a
            # candidate via stored predictions but wrapper.predict() disagrees and keeps
            # searching with no remaining candidates.
            if n_in_desired_range == 0:
                print(
                    f"Skipping query-index {q_idx} | asset={query_asset_id}: "
                    f"no reference window has prediction in [desired_min={desired_min:.6f}, "
                    f"desired_max={desired_max:.6f}]; kdtree has no valid candidates.",
                    flush=True,
                )
                return None, None, None, False
            dice_train_query = dice_train_query[
                (dice_train_query["prediction"] >= desired_min) &
                (dice_train_query["prediction"] <= desired_max)
            ].copy()

        elif effective_method == "genetic":
            if n_above_desired_min == 0:
                print(
                    f"Skipping query-index {q_idx} | asset={query_asset_id}: "
                    f"no reference window has prediction >= desired_min ({desired_min:.6f}); "
                    f"DiCE cannot find a counterfactual.",
                    flush=True,
                )
                return None, None, None, False
            if n_unique_windows < population_size or n_above_desired_min < population_size:
                print(
                    f"Skipping query-index {q_idx} | asset={query_asset_id}: "
                    f"insufficient reference windows for genetic search "
                    f"(unique_windows={n_unique_windows}, n_above_desired_min={n_above_desired_min}, "
                    f"need {population_size}).",
                    flush=True,
                )
                return None, None, None, False

        else:
            # random / other methods: only need at least 1 window in desired range
            if n_in_desired_range == 0:
                print(
                    f"Skipping query-index {q_idx} | asset={query_asset_id}: "
                    f"no reference window in desired range; cannot seed search.",
                    flush=True,
                )
                return None, None, None, False

        dice_data_query = dice_ml.Data(
            dataframe=dice_train_query[window_cols + ["prediction"]],
            continuous_features=window_cols,
            outcome_name="prediction",
        )

        exp_query = Dice(dice_data_query, dice_model, method=effective_method)

        cf_kwargs = {
            "total_CFs": int(args.total_cfs),
            "desired_range": [desired_min, desired_max],
            "features_to_vary": list(window_cols),
        }
        if effective_method == "genetic":
            cf_kwargs["maxiterations"] = int(args.maxiterations)

        t0 = time.time()
        try:
            dice_exp = exp_query.generate_counterfactuals(query_x, **cf_kwargs)
        except ValueError as error:
            if "empty range for randrange" in str(error) and effective_method == "genetic":
                elapsed = time.time() - t0
                print(
                    f"Skipping query-index {q_idx} | asset={query_asset_id}: "
                    f"genetic algorithm failed (empty population) — too few valid candidates. "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
                return None, None, None, False
            else:
                raise
        except RuntimeError as error:
            if "predict() call limit" in str(error):
                elapsed = time.time() - t0
                print(
                    f"Skipping query-index {q_idx} | asset={query_asset_id}: "
                    f"predict() call limit ({MAX_PREDICT_CALLS}) exceeded — DiCE search aborted. "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
                return None, None, None, False
            raise
        except UserConfigValidationException as error:
            msg = str(error)
            if "No counterfactuals found" in msg:
                elapsed = time.time() - t0
                print(
                    f"No counterfactuals found for query-index {q_idx} | "
                    f"asset={query_asset_id} | timestamp={query_rec_time} | elapsed={elapsed:.1f}s",
                    flush=True,
                )
                return None, None, None, False
            raise

        elapsed = time.time() - t0
        final_cfs = dice_exp.cf_examples_list[0].final_cfs_df
        if final_cfs is None or final_cfs.empty:
            print(
                f"No valid CFs returned for query-index {q_idx} | "
                f"asset={query_asset_id} | timestamp={query_rec_time} | elapsed={elapsed:.1f}s",
                flush=True,
            )
            return None, None, None, False

        final_cfs = final_cfs.copy()
        print(
            f"Found {len(final_cfs)} CF(s) for query-index {q_idx} | "
            f"asset={query_asset_id} | timestamp={query_rec_time} | elapsed={elapsed:.1f}s",
            flush=True,
        )

        pred_rows = []
        summary_rows = []
        ts_dfs = []

        for cf_idx, (_, cf_row) in enumerate(final_cfs.iterrows()):
            if "prediction" in cf_row.index:
                cf_pred = float(cf_row["prediction"])
            else:
                cf_x = pd.DataFrame([cf_row[window_cols].to_dict()])
                cf_pred = float(wrapper.predict(cf_x)[0])

            factual_win = np.array([float(row_series[col]) for col in window_cols])
            cf_win = np.array([float(cf_row[col]) for col in window_cols])
            factual_window_map = dict(zip(window_cols, factual_win.tolist()))
            cf_window_map = dict(zip(window_cols, cf_win.tolist()))

            metrics = _cf_utility_metrics(factual_win, cf_win, factual_pred, cf_pred, desired_min, desired_max)

            pred_rows.append({
                "query_index": q_idx,
                DEFAULT_ITEM_COL: query_asset_id,
                DEFAULT_TIMESTAMP_COL: query_rec_time,
                "cf_index": cf_idx,
                "factual_rating": float(factual_win[-1]),
                "cf_rating": float(cf_win[-1]),
                "factual_prediction": factual_pred,
                "cf_prediction": float(cf_pred),
                "desired_min": desired_min,
                "desired_max": desired_max,
                **metrics,
            })
            summary_rows.extend([
                {
                    "query_index": q_idx,
                    DEFAULT_ITEM_COL: query_asset_id,
                    DEFAULT_TIMESTAMP_COL: query_rec_time,
                    "cf_index": cf_idx,
                    "row_type": "factual",
                    "window_line": json.dumps(factual_window_map),
                },
                {
                    "query_index": q_idx,
                    DEFAULT_ITEM_COL: query_asset_id,
                    DEFAULT_TIMESTAMP_COL: query_rec_time,
                    "cf_index": cf_idx,
                    "row_type": "counterfactual",
                    "window_line": json.dumps(cf_window_map),
                },
            ])
            ts_dfs.append(_window_to_timeseries(cf_win, window_timestamps, query_asset_id, q_idx))

        return pred_rows, summary_rows, ts_dfs, False

    # ------------------------------------------------------------------ #
    # Build the list of queries to process (after resume/skip filtering). #
    # ------------------------------------------------------------------ #
    queries_to_run = []
    for _, query_row in query_windows.iterrows():
        q_idx = int(query_row["query_index_original"])
        if args.query_index is not None and q_idx < int(args.query_index):
            continue
        if args.query_index is None and q_idx in done_query_indices:
            print(f"Skipping query-index {q_idx} (already processed)", flush=True)
            continue
        queries_to_run.append((q_idx, query_row))

    skipped_indices = []

    if n_workers == 1:
        # Sequential path — simpler to debug, no overhead.
        for q_idx, row_series in queries_to_run:
            pred_rows, summary_rows, ts_dfs, was_skipped = _run_query(q_idx, row_series)
            if was_skipped:
                skipped_indices.append(q_idx)
            elif pred_rows is not None:
                _write_results(pred_rows, summary_rows, ts_dfs)
    else:
        # Parallel path — each worker thread holds its own model copy and query context
        # via thread-local storage; file writes are serialised with write_lock.
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_qidx = {
                executor.submit(_run_query, q_idx, row_series): q_idx
                for q_idx, row_series in queries_to_run
            }
            for future in as_completed(future_to_qidx):
                q_idx_done = future_to_qidx[future]
                try:
                    pred_rows, summary_rows, ts_dfs, was_skipped = future.result()
                except Exception as exc:
                    print(f"Query {q_idx_done} raised an exception: {exc}", flush=True)
                    skipped_indices.append(q_idx_done)
                    continue
                if was_skipped:
                    skipped_indices.append(q_idx_done)
                elif pred_rows is not None:
                    with write_lock:
                        _write_results(pred_rows, summary_rows, ts_dfs)

    n_total = len(query_windows)
    n_skipped = len(skipped_indices)
    print(f"Skipped queries       : {n_skipped}/{n_total} ({100*n_skipped/n_total:.1f}%)")
    print(f"Saved counterfactuals : {out_cf}")
    print(f"Saved summary         : {out_summary}")
    print(f"Saved timeseries      : {out_timeseries}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate RFR counterfactuals for the last window of each experiment (or custom pkls)"
    )
    parser.add_argument(
        "--model-pkl",
        type=Path,
        nargs="+",
        default=DEFAULT_EXPERIMENT_PKLS,
        metavar="PKL",
        help=(
            "One or more pkl paths to process. Training/testing CSVs are auto-derived "
            "from the pkl filename. Defaults to the last window of each experiment "
            "(2020-08-28 for exp1, 2021-11-23 for exp2)."
        ),
    )
    parser.add_argument("--asset-id", type=str, default=None, help="Optional single asset to process")
    parser.add_argument("--query-index", type=int, default=None, help="Run only this specific query index")
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=20,
        help=(
            "Number of queries to process in parallel. Each worker runs a full DiCE search "
            "on a separate thread with its own model copy and query context. "
            "Use -1 to use all available CPU cores (currently %d)." % (os.cpu_count() or 1)
        ),
    )
    parser.add_argument("--method", type=str, default=DEFAULT_DICE_METHOD, choices=["genetic", "random", "kdtree"])
    parser.add_argument("--maxiterations", type=int, default=DEFAULT_MAXITERATIONS)
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
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from last completed query (reads existing output file to skip already-processed indices)",
    )
    args = parser.parse_args()

    if args.window_size is not None and int(args.window_size) != FIXED_WINDOW_SIZE:
        raise ValueError(f"This script is fixed to window-size={FIXED_WINDOW_SIZE}")

    for pkl_path in args.model_pkl:
        training_path, testing_path = _derive_data_paths(pkl_path)
        out_cf, out_summary, out_timeseries = _derive_output_paths(pkl_path, args.method)
        _run_for_pkl(pkl_path, training_path, testing_path, out_cf, out_summary, out_timeseries, args)


if __name__ == "__main__":
    main()
