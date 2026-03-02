import os

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from utils.constants import DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL

from algorithms.kpi_gen.ma_kpi_generator import MAKPIGenerator


class TabNetKPIModel:
    """
    TabNet regressor model that integrates KPI generation internally.
    Takes raw time series data as input, generates KPIs, and predicts profitability.
    """

    def __init__(
        self,
        k=5,
        kpi_type="full_short",
        kpi_features=None,
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
    ):
        self.k = k
        self.kpi_type = kpi_type
        self.kpi_features = kpi_features

        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse

        self.kpi_generator = MAKPIGenerator(data=None, k=k, kpi_type=kpi_type)
        self.model = None
        self.is_fitted = False

    def _generate_kpis_df(self, time_series_df):
        self.kpi_generator.data = time_series_df
        self.kpi_generator.k = self.k
        self.kpi_generator.kpi_type = self.kpi_type
        self.kpi_generator.compute()
        return self.kpi_generator.get_kpis()

    def _select_features_in_order(self, df):
        missing = [col for col in self.kpi_features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing KPI features for TabNet model: {missing}")
        return df[self.kpi_features]

    def _time_based_train_val_split(self, merged_df, val_split):
        if val_split <= 0.0 or merged_df.shape[0] < 20:
            return merged_df, None

        split_df = merged_df.sort_values([DEFAULT_TIMESTAMP_COL, DEFAULT_ITEM_COL]).reset_index(drop=True)
        unique_dates = split_df[DEFAULT_TIMESTAMP_COL].drop_duplicates().sort_values().tolist()

        if len(unique_dates) < 2:
            return split_df, None

        n_val_dates = max(1, int(round(len(unique_dates) * val_split)))
        if n_val_dates >= len(unique_dates):
            n_val_dates = len(unique_dates) - 1

        val_dates = set(unique_dates[-n_val_dates:])
        val_df = split_df[split_df[DEFAULT_TIMESTAMP_COL].isin(val_dates)].copy()
        train_df = split_df[~split_df[DEFAULT_TIMESTAMP_COL].isin(val_dates)].copy()

        if train_df.empty or val_df.empty:
            return split_df, None

        return train_df, val_df

    def _best_val_rmse(self, model):
        history = getattr(model, "history", None)
        if history is None:
            return np.inf

        if isinstance(history, dict):
            for key in ("val_0_rmse", "val_rmse", "valid_rmse"):
                if key in history and len(history[key]) > 0:
                    return float(np.min(history[key]))
            for key, value in history.items():
                if "rmse" in str(key).lower() and len(value) > 0:
                    return float(np.min(value))

        return np.inf

    def _fit_one(self, X_train, y_train, X_val, y_val, *, n_d, n_a, n_steps, learning_rate, batch_size, epochs, patience, weight_decay, device):
        model = TabNetRegressor(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=self.gamma,
            lambda_sparse=self.lambda_sparse,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=learning_rate, weight_decay=weight_decay),
            device_name=device,
            seed=42,
        )

        if X_val is not None and y_val is not None:
            model.fit(
                X_train=X_train,
                y_train=y_train,
                eval_set=[(X_val, y_val)],
                eval_name=["val"],
                eval_metric=["rmse"],
                max_epochs=epochs,
                patience=patience,
                batch_size=batch_size,
                num_workers=0,
                drop_last=False,
            )
            score = self._best_val_rmse(model)
        else:
            model.fit(
                X_train=X_train,
                y_train=y_train,
                max_epochs=epochs,
                patience=patience,
                batch_size=batch_size,
                num_workers=0,
                drop_last=False,
            )
            preds = model.predict(X_train).reshape(-1)
            score = float(np.sqrt(np.mean((preds - y_train.reshape(-1)) ** 2)))

        return model, score

    def fit(
        self,
        time_series_df,
        y,
        kpi_features,
        epochs=200,
        batch_size=2048,
        learning_rate=3e-3,
        weight_decay=1e-5,
        val_split=0.15,
        patience=25,
        device="cpu",
        artifact_label=None,
    ):
        self.kpi_features = kpi_features

        kpis_df = self._generate_kpis_df(time_series_df)
        if artifact_label is not None:
            os.makedirs("for_testing", exist_ok=True)
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
            kpis_to_save.to_csv(f"for_testing/generated_kpis_train_{artifact_label}_tabnet.csv", index=False)

        if isinstance(y, pd.DataFrame):
            required_target_cols = [DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, "target"]
            missing_target_cols = [col for col in required_target_cols if col not in y.columns]
            if missing_target_cols:
                raise ValueError(
                    f"Missing required target columns for TabNet training: {missing_target_cols}"
                )

            merged = kpis_df.merge(
                y[required_target_cols],
                on=[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL],
                how="inner",
            )

            train_columns = list(self.kpi_features) + [DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, "target"]
            merged = merged[train_columns].dropna()
            merged[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(merged[DEFAULT_TIMESTAMP_COL])

            train_df, val_df = self._time_based_train_val_split(merged, val_split)
            X_train = self._select_features_in_order(train_df).values.astype(np.float32)
            y_train = train_df["target"].values.astype(np.float32).reshape(-1, 1)

            if val_df is not None:
                X_val = self._select_features_in_order(val_df).values.astype(np.float32)
                y_val = val_df["target"].values.astype(np.float32).reshape(-1, 1)
            else:
                X_val, y_val = None, None
        else:
            X = self._select_features_in_order(kpis_df).values.astype(np.float32)
            if isinstance(y, pd.Series):
                y = y.values
            target = np.asarray(y).astype(np.float32).reshape(-1, 1)
            X_train, y_train = X, target
            X_val, y_val = None, None

        if X_train.shape[0] == 0:
            return

        self.model, _ = self._fit_one(
            X_train,
            y_train,
            X_val,
            y_val,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            weight_decay=weight_decay,
            device=device,
        )

        self.is_fitted = True

    def predict(self, time_series_df):
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        if self.kpi_features is None:
            raise ValueError("kpi_features must be set (call fit() first)")

        kpis_df = self._generate_kpis_df(time_series_df)
        X = self._select_features_in_order(kpis_df).values.astype(np.float32)
        preds = self.model.predict(X)
        return preds.reshape(-1, 1)

    def predict_with_kpis(self, time_series_df, artifact_label=None):
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        if self.kpi_features is None:
            raise ValueError("kpi_features must be set (call fit() first)")

        kpis_df = self._generate_kpis_df(time_series_df)
        
        X = self._select_features_in_order(kpis_df).values.astype(np.float32)
        preds = self.model.predict(X)
        return kpis_df, preds.reshape(-1, 1)

    def save_model(self, path_without_extension):
        if not self.is_fitted or self.model is None:
            raise ValueError("Model must be fitted before saving")
        self.model.save_model(path_without_extension)
