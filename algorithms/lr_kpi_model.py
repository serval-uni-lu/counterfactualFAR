import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from utils.constants import DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL
from algorithms.rfr_kpi_model import KPIFeatureTransformer


class LRKPIModel:
    """
    Linear Regression model that integrates KPI generation internally.
    Mirrors RFRKPIModel/LGBMKPIModel — same fit/predict interface, same internal KPI pipeline.
    """

    def __init__(
        self,
        k=5,
        kpi_type="full_short",
        kpi_features=None,
        fit_intercept=True,
    ):
        self.k = k
        self.kpi_type = kpi_type
        self.kpi_features = kpi_features
        self.fit_intercept = fit_intercept

        self.transformer = KPIFeatureTransformer(k=k, kpi_type=kpi_type, kpi_features=kpi_features)
        self.pipeline = Pipeline(
            steps=[
                ("kpi", self.transformer),
                ("lr", LinearRegression(fit_intercept=self.fit_intercept)),
            ]
        )
        self.model = self.pipeline.named_steps["lr"]
        self.is_fitted = False

    def _generate_kpis_df(self, time_series_df):
        return self.transformer._generate_kpis_df(time_series_df)

    def _select_features_in_order(self, df):
        missing = [col for col in self.kpi_features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing KPI features for LR model: {missing}")
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
                    f"Missing required target columns for LR training: {missing_target_cols}"
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
        self.model = self.pipeline.named_steps["lr"]
        self.is_fitted = True

    def predict(self, time_series_df):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if self.kpi_features is None:
            raise ValueError("kpi_features must be set (call fit() first)")

        self.transformer._kpi_cache = None
        preds = self.pipeline.predict(time_series_df)
        return preds.reshape(-1, 1)
