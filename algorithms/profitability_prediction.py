#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.

import os
import re
import fcntl
import pickle

import datetime
import random
import pandas as pd
from utils.constants import DEFAULT_TIMESTAMP_COL, DEFAULT_ITEM_COL, DEFAULT_USER_COL, DEFAULT_RATING_COL

from algorithms.algorithm import Algorithm
from algorithms.lgbm_kpi_model import LGBMKPIModel
from algorithms.rfr_kpi_model import RFRKPIModel


INTERNAL_KPI_MODELS = (RFRKPIModel, LGBMKPIModel)


class ProfitabilityPrediction(Algorithm):
    """
    Algorithm that predicts the future profitability of assets. Ranks the assets according to that value.
    """

    def __init__(self, model, data, months, indicators, train_examples_per_asset, save_for_testing=False,
                 training_sizes_path=None):
        """
        Configures the profitability prediction model.
        :param model: the model to train.
        :param data: the data for training and applying recommendations.
        :param months: how many months in the future we want to apply our model to.
        :param indicators: the technical indicators we want to use from the set of computed ones.
        :param train_examples_per_asset: the maximum number of examples per asset to use.
        """
        super().__init__(data)

        self.kpis = data.kpis
        self.months = int(months)
        self.indicators = indicators
        self.model = model
        self.train_examples_per_asset = train_examples_per_asset
        self.is_fitted = False
        self.save_for_testing = save_for_testing
        self.training_sizes_path = training_sizes_path

    def _model_tag(self):
        if isinstance(self.model, RFRKPIModel):
            return "rfr"
        if isinstance(self.model, LGBMKPIModel):
            return "lgbm"
        return "rfr"

    def _safe_fragment(self, value):
        fragment = str(value).replace(" ", "_")
        fragment = re.sub(r"[^A-Za-z0-9_.-]+", "-", fragment)
        return fragment.strip("-_")

    def _model_param_tag(self):
        if isinstance(self.model, RFRKPIModel):
            n_estimators = getattr(self.model, "n_estimators", "na")
            kpi_type = getattr(self.model, "kpi_type", "na")
            return self._safe_fragment(f"n-{n_estimators}_kpi-{kpi_type}_internal_kpis")

        if isinstance(self.model, LGBMKPIModel):
            n_estimators = getattr(self.model, "n_estimators", "na")
            kpi_type = getattr(self.model, "kpi_type", "na")
            return self._safe_fragment(f"n-{n_estimators}_kpi-{kpi_type}_internal_kpis")

        n_estimators = getattr(self.model, "n_estimators", "na")
        return self._safe_fragment(f"n-{n_estimators}")

    def _artifact_dir(self):
        return os.path.join("artifacts_for_counterfactuals", f"{self._model_tag()}_{self._model_param_tag()}")

    def _dataset_artifact_label(self, when):
        return self._safe_fragment(when)

    def _artifact_path(self, prefix, when, extension="csv"):
        name = f"{prefix}_{self._safe_fragment(when)}_{self._model_tag()}_{self._model_param_tag()}.{extension}"
        return os.path.join(self._artifact_dir(), name)

    def _dataset_artifact_path(self, prefix, when, extension="csv"):
        name = f"{prefix}_{self._safe_fragment(when)}_{self._model_tag()}_{self._model_param_tag()}.{extension}"
        return os.path.join(self._artifact_dir(), name)

    def _compute_generalization_metrics(self, kpi_train_feats, train_targets, kpi_test_feats, test_targets):
        """Compute train/test regression metrics to assess generalization.

        Uses the fitted underlying estimator directly on pre-computed KPI features,
        avoiding a full KPI regeneration pass. Works for RFR, LGBM, and external sklearn models.
        Returns a dict with R², RMSE, MAE on train and test, plus the generalization gap.
        """
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        try:
            X_train = kpi_train_feats[self.indicators].astype(float)
            X_test = kpi_test_feats[self.indicators].astype(float)
            y_train = train_targets.values.astype(float)
            y_test = test_targets.values.astype(float)

            if len(X_train) == 0 or len(X_test) == 0:
                return {}

            # For internal models use the fitted underlying estimator directly
            # (model.model is the RF/LGBM regressor, bypassing KPI regeneration).
            # For external sklearn/lgbm models call predict() on features directly.
            if isinstance(self.model, (RFRKPIModel, LGBMKPIModel)) and hasattr(self.model, "model"):
                train_preds = self.model.model.predict(X_train)
                test_preds = self.model.model.predict(X_test)
            elif not isinstance(self.model, INTERNAL_KPI_MODELS):
                train_preds = self.model.predict(X_train).flatten()
                test_preds = self.model.predict(X_test).flatten()
            else:
                return {}

            train_r2   = float(r2_score(y_train, train_preds))
            test_r2    = float(r2_score(y_test,  test_preds))
            train_rmse = float(np.sqrt(mean_squared_error(y_train, train_preds)))
            test_rmse  = float(np.sqrt(mean_squared_error(y_test,  test_preds)))
            train_mae  = float(mean_absolute_error(y_train, train_preds))
            test_mae   = float(mean_absolute_error(y_test,  test_preds))
            gap        = train_r2 - test_r2

            print(
                f"Generalization | train_R²={train_r2:.4f}  test_R²={test_r2:.4f}  "
                f"gap={gap:.4f} | train_RMSE={train_rmse:.4f}  test_RMSE={test_rmse:.4f} | "
                f"n_train={len(y_train)}  n_test={len(y_test)}",
                flush=True,
            )

            return {
                "generalization_gap_r2": gap,
                "train_r2":   train_r2,
                "test_r2":    test_r2,
                "train_rmse": train_rmse,
                "test_rmse":  test_rmse,
                "train_mae":  train_mae,
                "test_mae":   test_mae,
                "train_samples": int(len(y_train)),
                "test_samples":  int(len(y_test)),
            }
        except Exception as exc:
            print(f"WARNING: Could not compute generalization metrics: {exc}", flush=True)
            return {}

    def _save_csv_if_missing(self, df, path):
        if not os.path.exists(path):
            df.to_csv(path, index=False)

    def _save_training_size(self, train_date, train_rows):
        if self.training_sizes_path is None:
            return

        out_path = self.training_sizes_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        row = {
            "model": self._model_tag(),
            "model_params": self._model_param_tag(),
            "rec_date": pd.to_datetime(train_date).strftime("%Y-%m-%d"),
            "train_rows": int(train_rows),
        }

        lock_path = out_path + ".lock"
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            if os.path.exists(out_path):
                current = pd.read_csv(out_path)
            else:
                current = pd.DataFrame(columns=["model", "model_params", "rec_date", "train_rows"])

            exists_mask = (
                (current["model"] == row["model"])
                & (current["model_params"] == row["model_params"])
                & (current["rec_date"] == row["rec_date"])
            ) if len(current) > 0 else pd.Series([], dtype=bool)

            if len(current) == 0 or not exists_mask.any():
                updated = pd.concat([current, pd.DataFrame([row])], ignore_index=True)
                updated = updated.sort_values(["model", "model_params", "rec_date"]).reset_index(drop=True)
                updated.to_csv(out_path, index=False)

    def _generate_internal_kpis(self, time_series_df):
        if isinstance(self.model, (RFRKPIModel, LGBMKPIModel)):
            return self.model._generate_kpis_df(time_series_df)
        raise ValueError("Internal KPI generation requested for a non-internal model")

    def _validate_internal_model_contract(self):
        if not isinstance(self.model, INTERNAL_KPI_MODELS):
            return

        has_kpi_path = hasattr(self.model, "_generate_kpis_df")

        if not has_kpi_path:
            raise ValueError(
                "Internal model contract violation: missing time-series→KPI generation path. "
                "Expected RFRKPIModel/LGBMKPIModel._generate_kpis_df."
            )

        if not hasattr(self.model, "fit"):
            raise ValueError("Internal model contract violation: missing fit(...) method")

        if not hasattr(self.model, "predict"):
            raise ValueError("Internal model contract violation: missing predict(...) method")

    def train(self, train_date):
        self._validate_internal_model_contract()

        # This is the maximum training date. Meaning that no future information is considered here.
        # Considering this:
        delta = datetime.timedelta(days=self.months * 30)

        time_series_df = self.data.time_series
        split_assets = set(self.data.assets)
        time_series_df = time_series_df[time_series_df[DEFAULT_ITEM_COL].isin(split_assets)]

        # As a first step, we find the technical indicators. We use all the possible information previous
        # to the training date - the number of months we are considering.
        if isinstance(self.model, INTERNAL_KPI_MODELS):
            source_kpis = self._generate_internal_kpis(time_series_df)
        else:
            source_kpis = self.kpis

        if source_kpis is None:
            raise ValueError("KPI data is missing and the selected model does not generate KPIs internally")

        kpi_indicators = source_kpis[source_kpis[DEFAULT_ITEM_COL].isin(self.data.assets)].copy()
        # For each asset, compute the target (profitability at k months) — vectorised groupby
        kpi_indicators["final_price"] = (
            kpi_indicators.groupby(DEFAULT_ITEM_COL, sort=False)[DEFAULT_RATING_COL]
            .shift(-self.months * 21)
        )
        kpi_indicators["target"] = (
            (kpi_indicators["final_price"] - kpi_indicators[DEFAULT_RATING_COL])
            / kpi_indicators[DEFAULT_RATING_COL]
        )
        kpi_indicators = kpi_indicators[kpi_indicators[DEFAULT_RATING_COL] > 0.0]

        # Finally, we filter the indicators by date.
        full = kpi_indicators.copy()
        kpi_indicators = full[full[DEFAULT_TIMESTAMP_COL] < (train_date - delta)]  # TRAIN rows: only timestamps strictly before the training cutoff (train_date - prediction horizon)
        kpi_indicators_test = full[full[DEFAULT_TIMESTAMP_COL] >= (train_date - delta)]  # TEST/EVAL rows: timestamps on/after the cutoff

        aux_list = self.indicators.copy()
        aux_list.append(DEFAULT_ITEM_COL)
        aux_list.append(DEFAULT_TIMESTAMP_COL)
        aux_list.append("target")
        kpi_indicators = kpi_indicators[aux_list]
        kpi_indicators = kpi_indicators.dropna()
        goals = kpi_indicators["target"]
        timestamp = kpi_indicators[DEFAULT_TIMESTAMP_COL]
        items = kpi_indicators[DEFAULT_ITEM_COL]
        kpi_indicators_features = kpi_indicators[self.indicators]

        # Combine features + target into one dataframe
        training_data = kpi_indicators_features.copy()
        training_data["target"] = goals
        training_data[DEFAULT_TIMESTAMP_COL] = timestamp
        training_data[DEFAULT_ITEM_COL] = items

        aux_list_test = self.indicators.copy()
        aux_list_test.append(DEFAULT_ITEM_COL)
        aux_list_test.append(DEFAULT_TIMESTAMP_COL)
        aux_list_test.append("target")
        kpi_indicators_test = kpi_indicators_test[aux_list_test]
        kpi_indicators_test = kpi_indicators_test.dropna()

        if kpi_indicators_features.shape[0] > 0:
            self._save_training_size(train_date, kpi_indicators_features.shape[0])
            # Internal models are trained from raw time-series + aligned targets.
            if isinstance(self.model, INTERNAL_KPI_MODELS):
                training_time_series = time_series_df[time_series_df[DEFAULT_TIMESTAMP_COL] < (train_date - delta)]  # Internal-model train split over raw time-series (same temporal cutoff as KPI train rows)
                train_targets = training_data[[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, "target"]]
                artifact_prefix = os.path.join(
                    self._artifact_dir(),
                    f"kpis_train_{self._dataset_artifact_label(train_date)}_{self._model_tag()}_{self._model_param_tag()}",
                ) if self.save_for_testing else None

                self.model.fit(
                    training_time_series,
                    train_targets,
                    kpi_features=self.indicators,
                    artifact_label=artifact_prefix,
                )
            else:
                # External (precomputed-KPI) models are trained directly on KPI features.
                self.model.fit(kpi_indicators_features, goals)
            
            self.is_fitted = True

            self.generalization_metrics_ = self._compute_generalization_metrics(
                kpi_indicators_features,
                goals,
                kpi_indicators_test[aux_list_test[:-1]],  # features + item/timestamp, no target
                kpi_indicators_test["target"],
            )

            if self.save_for_testing:
                os.makedirs(self._artifact_dir(), exist_ok=True)

                if isinstance(self.model, RFRKPIModel):
                    # Save raw time-series splits so CF generation scripts can build
                    # windows and regenerate KPIs through the pipeline.
                    testing_time_series = time_series_df[
                        time_series_df[DEFAULT_TIMESTAMP_COL] >= (train_date - delta)
                    ]
                    self._save_csv_if_missing(training_time_series, self._dataset_artifact_path("training_data", train_date))
                    self._save_csv_if_missing(testing_time_series, self._dataset_artifact_path("testing_data", train_date))
                else:
                    # Save KPI-based splits for all other models.
                    self._save_csv_if_missing(training_data, self._dataset_artifact_path("training_data", train_date))
                    self._save_csv_if_missing(kpi_indicators_test, self._dataset_artifact_path("testing_data", train_date))
                self.save_fitted_model(train_date)


    def save_fitted_model(self, train_date):
        self._validate_internal_model_contract()

        def _save_pickle_object(payload, path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as handle:
                pickle.dump(payload, handle)

        if self.is_fitted:
            if isinstance(self.model, RFRKPIModel):
                pipeline_path = self._artifact_path("profitability_recommendation_pipeline", train_date, "pkl")
                _save_pickle_object(self.model, pipeline_path)

                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                initial_type = [('float_input', FloatTensorType([None, len(self.indicators)]))]
                onnx_model = convert_sklearn(self.model.model, initial_types=initial_type)
                with open(self._artifact_path("profitability_recommendation", train_date, "onnx"), "wb") as f:
                    f.write(onnx_model.SerializeToString())
            elif isinstance(self.model, LGBMKPIModel):
                # skl2onnx does not support LGBMRegressor — save pkl only.
                pipeline_path = self._artifact_path("profitability_recommendation_pipeline", train_date, "pkl")
                _save_pickle_object(self.model, pipeline_path)
            else:
                # For external sklearn models, use skl2onnx.
                # LGBMRegressor is not supported by skl2onnx — save pkl instead.
                from lightgbm import LGBMRegressor as _LGBMRegressor
                if isinstance(self.model, _LGBMRegressor):
                    pipeline_path = self._artifact_path("profitability_recommendation_pipeline", train_date, "pkl")
                    _save_pickle_object(self.model, pipeline_path)
                else:
                    from skl2onnx import convert_sklearn
                    from skl2onnx.common.data_types import FloatTensorType
                    initial_type = [('float_input', FloatTensorType([None, len(self.indicators)]))]
                    onnx_model = convert_sklearn(self.model, initial_types=initial_type)
                    with open(self._artifact_path("profitability_recommendation", train_date, "onnx"), "wb") as f:
                        f.write(onnx_model.SerializeToString())
        else:
            raise Exception("Model is not fitted yet. Cannot save an untrained model.")
        

    def recommend(self, rec_time, target_custs, repeated, only_test_customers):
        fields = [x for x in self.indicators]
        fields.append(DEFAULT_ITEM_COL)
        fields.append(DEFAULT_TIMESTAMP_COL)
        prediction_input_snapshot = None
        prediction_time_series = self.data.time_series[
            self.data.time_series[DEFAULT_ITEM_COL].isin(self.data.assets)
        ]
        prediction_time_series = prediction_time_series[
            prediction_time_series[DEFAULT_TIMESTAMP_COL] <= rec_time
        ]

        # We first obtain KPI rows at recommendation time.
        if isinstance(self.model, INTERNAL_KPI_MODELS):
            # Recommendation-time cutoff: never expose future rows to internal KPI generation.
            if self.is_fitted:
                # For fitted internal models, get KPI rows from the model prediction path to avoid computing KPIs twice.
                kpi_indicators = None
            else:
                # If not fitted, we still need KPI rows to build random baselines.
                kpis_all = self._generate_internal_kpis(prediction_time_series)
                kpi_indicators = kpis_all[fields]
        else:
            kpi_indicators = self.kpis[fields]

        if kpi_indicators is not None:
            kpi_indicators = kpi_indicators[kpi_indicators[DEFAULT_TIMESTAMP_COL] == rec_time]
            kpi_indicators = kpi_indicators[kpi_indicators[DEFAULT_ITEM_COL].isin(self.data.assets)]

        # Then, we obtain the recommendation scores:
        if self.is_fitted:
            if isinstance(self.model, INTERNAL_KPI_MODELS):
                predictions = self.model.predict(prediction_time_series)
                kpis_df = getattr(getattr(self.model, "transformer", None), "last_kpis_df_", None)
                if kpis_df is None:
                    kpis_df = self._generate_internal_kpis(prediction_time_series)
                kpis_df = kpis_df.copy()
                if len(kpis_df) != len(predictions):
                    raise ValueError(
                        "Internal model prediction shape mismatch: "
                        f"len(kpis_df)={len(kpis_df)} vs len(predictions)={len(predictions)}"
                    )
                kpis_df["score"] = predictions.flatten()
                kpis_df = kpis_df[
                    (kpis_df[DEFAULT_TIMESTAMP_COL] == rec_time)
                    & (kpis_df[DEFAULT_ITEM_COL].isin(self.data.assets))
                ]
                kpi_indicators = kpis_df[fields + ["score"]].copy()

                feature_cols = [col for col in self.indicators if col in kpi_indicators.columns]
                prediction_input_snapshot = kpi_indicators[[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL] + feature_cols].copy()
            else:
                kpi_indicators["score"] = self.model.predict(kpi_indicators[self.indicators])
                feature_cols = [col for col in self.indicators if col in kpi_indicators.columns]
                prediction_input_snapshot = kpi_indicators[[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL] + feature_cols].copy()
        else:
            kpi_indicators["score"] = kpi_indicators[DEFAULT_ITEM_COL].apply(lambda x: random.random())
            feature_cols = [col for col in self.indicators if col in kpi_indicators.columns]
            prediction_input_snapshot = kpi_indicators[[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL] + feature_cols].copy()

        if self.save_for_testing:
            score_snapshot = kpi_indicators[[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, "score"]].copy()
            score_snapshot = score_snapshot.rename(columns={"score": "prediction"})

            if prediction_input_snapshot is None:
                prediction_snapshot = score_snapshot
            else:
                prediction_snapshot = prediction_input_snapshot.merge(
                    score_snapshot,
                    on=[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL],
                    how="left",
                )

            self._save_csv_if_missing(prediction_snapshot, self._artifact_path("predictions", rec_time))

        # And, finally, we sort the assets by score:
        kpi_indicators_full = (
            kpi_indicators
            .sort_values(by="score", ascending=False)
            .rename(columns={"score": DEFAULT_RATING_COL})
        )

        kpi_indicators = (
            kpi_indicators[[DEFAULT_ITEM_COL, "score"]]
            .sort_values(by="score", ascending=False)
            .rename(columns={"score": DEFAULT_RATING_COL})
        )

        user_recommendations = []
        user_recs_full = []
        customers = (self.data.users & set(self.data.test[DEFAULT_USER_COL].unique().flatten())) if only_test_customers else self.data.users
        customers = customers & target_custs

        # num_rows = num_customers × num_items_available_at_rec_time
        for customer in customers:
            user_recommendation = kpi_indicators.copy()
            user_recommendation_full = kpi_indicators_full.copy()
            user_recommendation[DEFAULT_USER_COL] = customer
            user_recommendation_full[DEFAULT_USER_COL] = customer

            if not repeated:
                items_per_user = set(self.data.train[self.data.train[DEFAULT_USER_COL] == customer][DEFAULT_ITEM_COL].unique().flatten())
                user_recommendation = user_recommendation[~user_recommendation[DEFAULT_ITEM_COL].isin(items_per_user)]
                user_recommendation_full = user_recommendation_full[~user_recommendation_full[DEFAULT_ITEM_COL].isin(items_per_user)]
            user_recommendations.append(user_recommendation)
            user_recs_full.append(user_recommendation_full)
        
        user_recs_full = pd.concat(user_recs_full)
        if self.save_for_testing:
            self._save_csv_if_missing(user_recs_full, self._dataset_artifact_path("user_recs", rec_time))

        return pd.concat(user_recommendations)
 