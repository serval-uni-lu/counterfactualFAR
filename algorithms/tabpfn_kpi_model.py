import os

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from tabpfn import TabPFNRegressor

from utils.constants import DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL
from utils.common_util import stratified_sample_by_group
from algorithms.rfr_kpi_model import KPIFeatureTransformer


class TabPFNKPIModel:
    """
    TabPFN regressor integrates KPI generation internally.
    Mirrors RFRKPIModel/LGBMKPIModel — same fit/predict interface, same internal KPI pipeline.

    TabPFN is a pretrained in-context-learning transformer, not a trained-from-scratch
    estimator: fit() just caches the training set, predict() does the actual work in a
    single forward pass conditioned on it — cost scales with context (training set) size,
    so large windows can exhaust GPU memory. Two independent memory-bounding knobs:
      - sample_pct: fit() trains on a per-asset stratified subsample of the window
        instead of the full thing (also applied to the generalization-metrics diagnostic).
      - predict_batch_size: at recommend()-time, every candidate asset still gets a real
        score (nothing becomes unrecommendable) — predict() only scores the most recent
        date's rows (not every historical date up to it) and does so in GPU-bounded
        batches rather than one giant forward pass.
    """

    def __init__(
        self,
        k=5,
        kpi_type="full_short",
        kpi_features=None,
        random_state=42,
        device=None,
        sample_pct=None,
        predict_batch_size=2000,
        ignore_pretraining_limits=True,
    ):
        self.k = k
        self.kpi_type = kpi_type
        self.kpi_features = kpi_features
        self.random_state = random_state
        self.sample_pct = sample_pct
        self.predict_batch_size = predict_batch_size
        self.ignore_pretraining_limits = ignore_pretraining_limits

        self.device = device
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "TabPFNKPIModel defaults to GPU (device='cuda') but no CUDA device "
                    "is available. Pass device='cpu' explicitly to run on CPU anyway "
                    "(very slow at this project's window sizes), or run on a GPU node."
                )
            self._resolved_device = "cuda"
        else:
            self._resolved_device = device

        self.transformer = KPIFeatureTransformer(k=k, kpi_type=kpi_type, kpi_features=kpi_features)
        self.pipeline = Pipeline(
            steps=[
                ("kpi", self.transformer),
                ("tabpfn", TabPFNRegressor(
                    device=self._resolved_device,
                    random_state=self.random_state,
                    ignore_pretraining_limits=self.ignore_pretraining_limits,
                )),
            ]
        )
        self.model = self.pipeline.named_steps["tabpfn"]
        self.is_fitted = False

    def _generate_kpis_df(self, time_series_df):
        return self.transformer._generate_kpis_df(time_series_df)

    def _select_features_in_order(self, df):
        missing = [col for col in self.kpi_features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing KPI features for TabPFN model: {missing}")
        return df[self.kpi_features]

    def fit(self, time_series_df, y, kpi_features, artifact_label=None):
        self.kpi_features = kpi_features
        self.transformer.kpi_features = kpi_features

        kpis_df = self._generate_kpis_df(time_series_df)
        self.transformer._kpi_cache = kpis_df
        if artifact_label is not None:
            os.makedirs(os.path.dirname(artifact_label), exist_ok=True)
            if isinstance(y, pd.DataFrame):
                target_cols = [DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, "target"]
                if all(col in y.columns for col in target_cols):
                    kpis_to_save = kpis_df.merge(
                        y[target_cols],
                        on=[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL],
                        how="left",
                    )
                else:
                    kpis_to_save = kpis_df
            else:
                kpis_to_save = kpis_df
            kpis_to_save.to_csv(f"{artifact_label}.csv", index=False)

        if isinstance(y, pd.DataFrame):
            required_target_cols = [DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, "target"]
            missing_target_cols = [col for col in required_target_cols if col not in y.columns]
            if missing_target_cols:
                raise ValueError(
                    f"Missing required target columns for TabPFN training: {missing_target_cols}"
                )

            merged = kpis_df.merge(
                y[required_target_cols],
                on=[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL],
                how="inner",
            )
            train_columns = list(self.kpi_features) + [DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, "target"]
            merged = merged[train_columns].dropna()

            if self.sample_pct is not None:
                # Bounds the actual training context TabPFN conditions on at
                # predict() time (attention cost scales with context size),
                # not just a diagnostic — real recommendations train on this
                # sampled set too. Stratified per asset so none drop out.
                merged, _ = stratified_sample_by_group(
                    merged, merged["target"], self.sample_pct, seed=self.random_state
                )

            target_df = merged[[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, "target"]].copy()
            target = merged["target"].values
        else:
            target_df = None
            if isinstance(y, pd.Series):
                y = y.values
            target = np.asarray(y).reshape(-1)

        if target.shape[0] == 0:
            return

        self.pipeline.fit(time_series_df, target, kpi__target_df=target_df)
        self.model = self.pipeline.named_steps["tabpfn"]
        self.is_fitted = True

    def predict(self, time_series_df):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if self.kpi_features is None:
            raise ValueError("kpi_features must be set (call fit() first)")

        kpis_df = self._generate_kpis_df(time_series_df)

        # The KPI generator emits one row per (asset, date) it saw, but only
        # the most recent date (the recommendation date) is ever scored
        # downstream. Scoring every historical date up to it was the query
        # set that blew up GPU memory here, independent of sample_pct
        # (which only bounds the training side) — so this filter is applied
        # unconditionally, not gated behind sample_pct.
        latest_date = kpis_df[DEFAULT_TIMESTAMP_COL].max()
        kpis_df = kpis_df[kpis_df[DEFAULT_TIMESTAMP_COL] == latest_date].reset_index(drop=True)
        self.transformer.last_kpis_df_ = kpis_df.copy()

        X = self._select_features_in_order(kpis_df).values.astype(np.float32)
        preds = self._predict_in_batches(X)
        return preds.reshape(-1, 1)

    def _predict_in_batches(self, X):
        # Batches rather than samples: every candidate asset still gets a
        # real score (nothing becomes unrecommendable), just computed in
        # GPU-memory-bounded chunks instead of one giant forward pass.
        batch_size = self.predict_batch_size
        if batch_size is None or len(X) <= batch_size:
            return self.model.predict(X)
        chunks = [self.model.predict(X[i:i + batch_size]) for i in range(0, len(X), batch_size)]
        return np.concatenate(chunks)
