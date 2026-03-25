import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from utils.constants import DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL
from algorithms.kpi_gen.ma_kpi_generator import MAKPIGenerator


class KPIFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible transformer that computes KPI features from raw time-series rows.
    """

    _kpi_cache = None  # class-level default for backward compat with pickles created before this attr existed

    def __init__(self, k=5, kpi_type="full_short", kpi_features=None):
        self.k = k
        self.kpi_type = kpi_type
        self.kpi_features = kpi_features
        self.kpi_generator = MAKPIGenerator(data=None, k=k, kpi_type=kpi_type)
        self._fit_keys = None
        self.last_kpis_df_ = None
        self._kpi_cache = None  # set by RFRKPIModel.fit() to avoid recomputing in transform()

    def _generate_kpis_df(self, time_series_df):
        self.kpi_generator.data = time_series_df
        self.kpi_generator.k = self.k
        self.kpi_generator.kpi_type = self.kpi_type
        self.kpi_generator.compute()
        return self.kpi_generator.get_kpis()

    def fit(self, X, y=None, target_df=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("KPIFeatureTransformer expects a pandas DataFrame as input")

        required_input_cols = [DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]
        missing_input_cols = [col for col in required_input_cols if col not in X.columns]
        if missing_input_cols:
            raise ValueError(f"Missing required input columns for KPI generation: {missing_input_cols}")

        self._fit_keys = None
        if isinstance(target_df, pd.DataFrame):
            required = [DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]
            if all(col in target_df.columns for col in required):
                self._fit_keys = target_df[required].drop_duplicates().reset_index(drop=True)
        return self

    def transform(self, X):
        if self.kpi_features is None:
            raise ValueError("kpi_features must be set before generating KPIs")

        if self._kpi_cache is not None:
            kpis_df = self._kpi_cache
            self._kpi_cache = None  # consume once, recompute on any subsequent call
        else:
            kpis_df = self._generate_kpis_df(X)
        self.last_kpis_df_ = kpis_df.copy()

        missing = [col for col in self.kpi_features if col not in kpis_df.columns]
        if missing:
            raise ValueError(f"Missing KPI features for RFR model: {missing}")

        if self._fit_keys is not None:
            merged = self._fit_keys.merge(
                kpis_df,
                on=[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL],
                how="inner",
            )
            self._fit_keys = None
            return merged[self.kpi_features].values.astype(np.float32)

        return kpis_df[self.kpi_features].values.astype(np.float32)


class RFRKPIModel:
    """
    Random Forest regressor that integrates KPI generation internally.
    Takes raw time series data as input, generates KPIs, and predicts profitability.
    """

    def __init__(
        self,
        n_estimators=100,
        k=5,
        kpi_type="full_short",
        kpi_features=None,
        random_state=42,
        max_features="sqrt",
        min_samples_leaf=10,
        max_depth=20,
        n_jobs=-1,
    ):
        self.n_estimators = int(n_estimators)
        self.k = k
        self.kpi_type = kpi_type
        self.kpi_features = kpi_features
        self.random_state = random_state
        self.max_features = max_features
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_depth = max_depth
        self.n_jobs = n_jobs

        self.transformer = KPIFeatureTransformer(k=k, kpi_type=kpi_type, kpi_features=kpi_features)
        self.pipeline = Pipeline(
            steps=[
                ("kpi", self.transformer),
                ("rf", RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_features=self.max_features,
                    min_samples_leaf=self.min_samples_leaf,
                    max_depth=self.max_depth,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )),
            ]
        )
        self.model = self.pipeline.named_steps["rf"]
        self.is_fitted = False

    def _generate_kpis_df(self, time_series_df):
        return self.transformer._generate_kpis_df(time_series_df)

    def _select_features_in_order(self, df):
        missing = [col for col in self.kpi_features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing KPI features for RFR model: {missing}")
        return df[self.kpi_features]

    def fit(self, time_series_df, y, kpi_features, artifact_label=None, **kwargs):
        self.kpi_features = kpi_features

        self.transformer.kpi_features = kpi_features

        kpis_df = self._generate_kpis_df(time_series_df)
        # Cache so pipeline.fit() → transform() reuses this result instead of recomputing.
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
                    f"Missing required target columns for RFR training: {missing_target_cols}"
                )

            merged = kpis_df.merge(
                y[required_target_cols],
                on=[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL],
                how="inner",
            )
            train_columns = list(self.kpi_features) + [DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, "target"]
            merged = merged[train_columns].dropna()
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
        self.model = self.pipeline.named_steps["rf"]
        self.is_fitted = True

    def predict(self, time_series_df):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if self.kpi_features is None:
            raise ValueError("kpi_features must be set (call fit() first)")

        preds = self.pipeline.predict(time_series_df)
        return preds.reshape(-1, 1)