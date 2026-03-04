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
import copy
import re
import fcntl
import pickle

import datetime
import random
import torch
import pandas as pd
from utils.constants import DEFAULT_TIMESTAMP_COL, DEFAULT_ITEM_COL, DEFAULT_USER_COL, DEFAULT_RATING_COL

from algorithms.algorithm import Algorithm
from algorithms.mlp_kpi_model import MLPKPIModel
from algorithms.rfr_kpi_model import RFRKPIModel
from algorithms.tabnet_kpi_model import TabNetKPIModel
from algorithms.torch_kpi_window_encoder import KPIWindowEncoder, WindowToFeatureHeadModel


INTERNAL_KPI_MODELS = (MLPKPIModel, TabNetKPIModel, RFRKPIModel)


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
        if isinstance(self.model, MLPKPIModel):
            return "mlp"
        if isinstance(self.model, TabNetKPIModel):
            return "tabnet"
        if isinstance(self.model, RFRKPIModel):
            return "rfr"
        return "rfr"

    def _safe_fragment(self, value):
        fragment = str(value).replace(" ", "_")
        fragment = re.sub(r"[^A-Za-z0-9_.-]+", "-", fragment)
        return fragment.strip("-_")

    def _model_param_tag(self):
        if isinstance(self.model, MLPKPIModel):
            hidden_sizes = getattr(self.model, "hidden_sizes", None)
            hidden_tag = "-".join(str(x) for x in hidden_sizes) if hidden_sizes else "default"
            kpi_type = getattr(self.model, "kpi_type", "na")
            return self._safe_fragment(f"hs-{hidden_tag}_kpi-{kpi_type}")

        if isinstance(self.model, TabNetKPIModel):
            kpi_type = getattr(self.model, "kpi_type", "na")
            n_d = getattr(self.model, "n_d", "na")
            n_a = getattr(self.model, "n_a", "na")
            n_steps = getattr(self.model, "n_steps", "na")
            return self._safe_fragment(f"kpi-{kpi_type}_nd-{n_d}_na-{n_a}_steps-{n_steps}")

        if isinstance(self.model, RFRKPIModel):
            n_estimators = getattr(self.model, "n_estimators", "na")
            kpi_type = getattr(self.model, "kpi_type", "na")
            return self._safe_fragment(f"n-{n_estimators}_kpi-{kpi_type}_internal_kpis")

        n_estimators = getattr(self.model, "n_estimators", "na")
        return self._safe_fragment(f"n-{n_estimators}")

    def _dataset_artifact_label(self, when):
        return self._safe_fragment(when)

    def _artifact_path(self, prefix, when, extension="csv"):
        return f"for_testing/{prefix}_{self._safe_fragment(when)}_{self._model_tag()}_{self._model_param_tag()}.{extension}"

    def _dataset_artifact_path(self, prefix, when, extension="csv"):
        dataset_tag = self._model_tag()
        if isinstance(self.model, RFRKPIModel):
            dataset_tag = "rfr_internal_kpis"
        return f"for_testing/{prefix}_{self._safe_fragment(when)}_{dataset_tag}.{extension}"

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
        if isinstance(self.model, MLPKPIModel):
            return self.model.kpi_module.generate_kpis_df(time_series_df)
        if isinstance(self.model, TabNetKPIModel):
            return self.model._generate_kpis_df(time_series_df)
        if isinstance(self.model, RFRKPIModel):
            return self.model._generate_kpis_df(time_series_df)
        raise ValueError("Internal KPI generation requested for a non-internal model")

    def _validate_internal_model_contract(self):
        if not isinstance(self.model, INTERNAL_KPI_MODELS):
            return

        if isinstance(self.model, MLPKPIModel):
            has_kpi_path = hasattr(self.model, "kpi_module") and hasattr(self.model.kpi_module, "generate_kpis_df")
        elif isinstance(self.model, TabNetKPIModel):
            has_kpi_path = hasattr(self.model, "_generate_kpis_df")
        elif isinstance(self.model, RFRKPIModel):
            has_kpi_path = hasattr(self.model, "_generate_kpis_df")
        else:
            has_kpi_path = False

        if not has_kpi_path:
            raise ValueError(
                "Internal model contract violation: missing time-series→KPI generation path. "
                "Expected MLPKPIModel.kpi_module.generate_kpis_df or TabNetKPIModel._generate_kpis_df."
            )

        if not hasattr(self.model, "fit"):
            raise ValueError("Internal model contract violation: missing fit(...) method")

        if not (hasattr(self.model, "predict_with_kpis") or hasattr(self.model, "predict")):
            raise ValueError("Internal model contract violation: missing predict(...) or predict_with_kpis(...) method")

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

        kpi_indicators = source_kpis[source_kpis[DEFAULT_ITEM_COL].isin(self.data.assets)]
        # For each asset, we get the target (profitability at k months)
        asset_dfs = []
        for asset in kpi_indicators[DEFAULT_ITEM_COL].unique():
            asset_df = kpi_indicators[kpi_indicators[DEFAULT_ITEM_COL] == asset].copy()
            asset_df["final_price"] = asset_df[DEFAULT_RATING_COL].shift(-self.months * 21)
            asset_df["target"] = (asset_df["final_price"] - asset_df[DEFAULT_RATING_COL]) / (asset_df[DEFAULT_RATING_COL])
            asset_df = asset_df[asset_df[DEFAULT_RATING_COL] > 0.0]
            asset_dfs.append(asset_df)
        kpi_indicators = pd.concat(asset_dfs)

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
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if isinstance(self.model, TabNetKPIModel):
                    self.model.fit(
                        training_time_series,
                        train_targets,
                        kpi_features=self.indicators,
                        epochs=150,
                        batch_size=2048,
                        learning_rate=3e-3,
                        weight_decay=1e-5,
                        val_split=0.1,
                        patience=20,
                        device=device,
                        artifact_label=self._dataset_artifact_label(train_date) if self.save_for_testing else None,
                    )
                elif isinstance(self.model, RFRKPIModel):
                    self.model.fit(
                        training_time_series,
                        train_targets,
                        kpi_features=self.indicators,
                        artifact_label=self._dataset_artifact_label(train_date) if self.save_for_testing else None,
                    )
                else:
                    self.model.fit(
                        training_time_series,
                        train_targets,
                        kpi_features=self.indicators,
                        epochs=150,
                        batch_size=2048,
                        learning_rate=1e-3,
                        weight_decay=1e-4,
                        val_split=0.1,
                        patience=15,
                        device=device,
                        artifact_label=self._dataset_artifact_label(train_date) if self.save_for_testing else None,
                    )
            else:
                # External (precomputed-KPI) models are trained directly on KPI features.
                self.model.fit(kpi_indicators_features, goals)
            
            self.is_fitted = True
            if self.save_for_testing:
                os.makedirs("for_testing", exist_ok=True)
            
                # Save to CSV using the same split-derived KPI datasets for both internal and external flows.
                self._save_csv_if_missing(training_data, self._dataset_artifact_path("training_data", train_date))
                self._save_csv_if_missing(kpi_indicators_test, self._dataset_artifact_path("testing_data", train_date))
                self.save_fitted_model(train_date)


    def save_fitted_model(self, train_date):
        self._validate_internal_model_contract()

        def _move_unregistered_tensors(module, device):
            for attr_name, attr_value in vars(module).items():
                if isinstance(attr_value, torch.Tensor):
                    setattr(module, attr_name, attr_value.to(device))
            for child in module.children():
                _move_unregistered_tensors(child, device)

        def _prepare_torch_model_for_export(torch_model, device):
            torch_model = torch_model.eval().to(device)
            _move_unregistered_tensors(torch_model, device)
            return torch_model

        def _save_torch_checkpoint(payload, path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(payload, path)

        def _save_pickle_object(payload, path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as handle:
                pickle.dump(payload, handle)

        def _window_size_for_export(kpi_type, k):
            periods = [21, 63, 126] if "short" in str(kpi_type).lower() else [21, 63, 126, 189]
            min_len = max(periods) + int(k)
            return max(160, min_len)

        def _export_end_to_end_onnx(torch_head_model, model_family, kpi_type, k):
            export_device = torch.device("cpu")
            head_cpu = _prepare_torch_model_for_export(copy.deepcopy(torch_head_model), export_device)
            encoder = KPIWindowEncoder(feature_names=list(self.indicators), kpi_type=kpi_type, k=int(k))
            end_to_end = WindowToFeatureHeadModel(encoder=encoder, head_model=head_cpu).eval().to(export_device)

            window_size = _window_size_for_export(kpi_type, k)
            ts_pt_path = self._artifact_path("profitability_recommendation_ts", train_date, "pt")
            _save_torch_checkpoint(
                {
                    "model_family": model_family,
                    "kpi_type": kpi_type,
                    "k": int(k),
                    "window_size": int(window_size),
                    "indicators": list(self.indicators),
                    "module": end_to_end,
                },
                ts_pt_path,
            )

            dummy_input = torch.randn(1, window_size, dtype=torch.float32, device=export_device)
            ts_onnx_path = self._artifact_path("profitability_recommendation_ts", train_date, "onnx")
            os.makedirs(os.path.dirname(ts_onnx_path), exist_ok=True)

            torch.onnx.export(
                end_to_end,
                dummy_input,
                ts_onnx_path,
                input_names=["prices"],
                output_names=["prediction"],
                dynamic_axes={"prices": {0: "batch"}, "prediction": {0: "batch"}},
                opset_version=13,
                do_constant_folding=True,
            )

            if self.save_for_testing:
                metadata_path = self._artifact_path("profitability_recommendation_ts_metadata", train_date, "csv")
                metadata_df = pd.DataFrame(
                    [
                        {
                            "model_family": model_family,
                            "kpi_type": kpi_type,
                            "k": int(k),
                            "window_size": int(window_size),
                            "n_features": int(len(self.indicators)),
                            "features": "|".join(list(self.indicators)),
                        }
                    ]
                )
                self._save_csv_if_missing(metadata_df, metadata_path)

        if self.is_fitted:
            if isinstance(self.model, MLPKPIModel):
                if self.model.network is None:
                    raise Exception("MLP network is not initialized. Cannot save .pt model.")

                export_device = torch.device("cpu")
                torch_model = _prepare_torch_model_for_export(copy.deepcopy(self.model.network), export_device)
                pt_path = self._artifact_path("profitability_recommendation", train_date, "pt")
                _save_torch_checkpoint(
                    {
                        "model_family": "mlp",
                        "kpi_type": getattr(self.model, "kpi_type", "full_short"),
                        "k": getattr(self.model, "k", 5),
                        "indicators": list(self.indicators),
                        "hidden_sizes": list(getattr(self.model, "hidden_sizes", [])),
                        "target_lower_bound": float(getattr(self.model, "target_lower_bound", -5.0)),
                        "state_dict": torch_model.state_dict(),
                    },
                    pt_path,
                )
                _export_end_to_end_onnx(
                    torch_head_model=torch_model,
                    model_family="mlp",
                    kpi_type=getattr(self.model, "kpi_type", "full_short"),
                    k=getattr(self.model, "k", 5),
                )
            elif isinstance(self.model, TabNetKPIModel):
                if self.model.model is None:
                    raise Exception("TabNet model is not initialized. Cannot save .pt model.")

                torch_model = getattr(self.model.model, "network", None)
                if torch_model is None:
                    raise Exception("TabNet internal torch network not found. Cannot save .pt model.")

                export_device = torch.device("cpu")
                torch_model = _prepare_torch_model_for_export(copy.deepcopy(torch_model), export_device)
                pt_path = self._artifact_path("profitability_recommendation", train_date, "pt")
                _save_torch_checkpoint(
                    {
                        "model_family": "tabnet",
                        "kpi_type": getattr(self.model, "kpi_type", "full_short"),
                        "k": getattr(self.model, "k", 5),
                        "indicators": list(self.indicators),
                        "n_d": getattr(self.model, "n_d", None),
                        "n_a": getattr(self.model, "n_a", None),
                        "n_steps": getattr(self.model, "n_steps", None),
                        "state_dict": torch_model.state_dict(),
                    },
                    pt_path,
                )
                _export_end_to_end_onnx(
                    torch_head_model=torch_model,
                    model_family="tabnet",
                    kpi_type=getattr(self.model, "kpi_type", "full_short"),
                    k=getattr(self.model, "k", 5),
                )
            elif isinstance(self.model, RFRKPIModel):
                pipeline_path = self._artifact_path("profitability_recommendation_pipeline", train_date, "pkl")
                _save_pickle_object(self.model, pipeline_path)

                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                initial_type = [('float_input', FloatTensorType([None, len(self.indicators)]))]
                onnx_model = convert_sklearn(self.model.model, initial_types=initial_type)
                with open(self._artifact_path("profitability_recommendation", train_date, "onnx"), "wb") as f:
                    f.write(onnx_model.SerializeToString())
            else:
                # For sklearn models, use skl2onnx
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
                if hasattr(self.model, "predict_with_kpis") and not isinstance(self.model, RFRKPIModel):
                    kpis_df, predictions = self.model.predict_with_kpis(prediction_time_series)
                else:
                    predictions = self.model.predict(prediction_time_series)
                    if isinstance(self.model, RFRKPIModel):
                        kpis_df = getattr(getattr(self.model, "transformer", None), "last_kpis_df_", None)
                        if kpis_df is None:
                            kpis_df = self._generate_internal_kpis(prediction_time_series)
                    else:
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
 