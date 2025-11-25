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

import datetime
import random
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import pandas as pd
from utils.constants import DEFAULT_TIMESTAMP_COL, DEFAULT_ITEM_COL, DEFAULT_USER_COL, DEFAULT_RATING_COL

from algorithms.algorithm import Algorithm


class ProfitabilityPrediction(Algorithm):
    """
    Algorithm that predicts the future profitability of assets. Ranks the assets according to that value.
    """

    def __init__(self, model, data, months, indicators, train_examples_per_asset):
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
        self.months = int(months) # CHANGED
        self.indicators = indicators
        self.model = model
        self.train_examples_per_asset = train_examples_per_asset
        self.is_fitted = False
    
    def train(self, train_date): # CHANGED
        # This is the maximum training date. Meaning that no future information is considered here.
        # Considering this:
        delta = datetime.timedelta(days=self.months * 30)

        # As a first step, we find the technical indicators. We use all the possible information previous
        # to the training date - the number of months we are considering.
        kpi_indicators = self.kpis[self.kpis[DEFAULT_ITEM_COL].isin(self.data.assets)]
        # For each asset, we get the target (profitability at k months)
        asset_dfs = []
        for asset in kpi_indicators[DEFAULT_ITEM_COL].unique():
            asset_df = kpi_indicators[kpi_indicators[DEFAULT_ITEM_COL] == asset]
            asset_df["final_price"] = asset_df[DEFAULT_RATING_COL].shift(-self.months * 21)
            asset_df["target"] = (asset_df["final_price"] - asset_df[DEFAULT_RATING_COL]) / (asset_df[DEFAULT_RATING_COL])
            asset_df = asset_df[asset_df[DEFAULT_RATING_COL] > 0.0]
            asset_dfs.append(asset_df)
        kpi_indicators = pd.concat(asset_dfs)

        # Finally, we filter the indicators by date.
        full = kpi_indicators.copy()
        kpi_indicators = full[full[DEFAULT_TIMESTAMP_COL] < (train_date - delta)]
        kpi_indicators_test = full[full[DEFAULT_TIMESTAMP_COL] >= (train_date - delta)]

        aux_list = self.indicators.copy()
        aux_list.append("target")
        aux_list.append("col_timestamp")
        kpi_indicators = kpi_indicators[aux_list]
        kpi_indicators = kpi_indicators.dropna()
        goals = kpi_indicators["target"]
        timestamp = kpi_indicators["col_timestamp"]
        kpi_indicators = kpi_indicators[self.indicators]

        # Combine features + target into one dataframe
        training_data = kpi_indicators.copy()
        training_data["target"] = goals
        training_data["col_timestamp"] = timestamp

        aux_list_test = self.indicators.copy()
        aux_list_test.append("target")
        aux_list_test.append("col_timestamp")
        kpi_indicators_test = kpi_indicators_test[aux_list_test]
        kpi_indicators_test = kpi_indicators_test.dropna()

        if kpi_indicators.shape[0] > 0:  # CHANGED
            self.model.fit(kpi_indicators, goals)
            self.is_fitted = True

            os.makedirs("for_testing", exist_ok=True)
        
            # Save to CSV
            training_data.to_csv(f"for_testing/training_data_{train_date}.csv", index=False)
            kpi_indicators_test.to_csv(f"for_testing/testing_data_{train_date}.csv", index=False)
            self.save_fitted_model(train_date)


    def save_fitted_model(self, train_date): # ADDED
        if self.is_fitted:
            initial_type = [('float_input', FloatTensorType([None, len(self.indicators)]))]
            onnx_model = convert_sklearn(self.model, initial_types=initial_type)
            with open(f"for_testing/profitability_recommendation_{train_date}.onnx", "wb") as f:
                f.write(onnx_model.SerializeToString())
        else:
            raise Exception("Model is not fitted yet. Cannot save an untrained model.")
        

    def recommend(self, rec_time, target_custs, repeated, only_test_customers):
        fields = [x for x in self.indicators]
        fields.append(DEFAULT_ITEM_COL)
        fields.append(DEFAULT_TIMESTAMP_COL)

        # We first obtain the KPIs
        kpi_indicators = self.kpis[fields]

        kpi_indicators = kpi_indicators[kpi_indicators[DEFAULT_TIMESTAMP_COL] == rec_time]
        kpi_indicators = kpi_indicators[kpi_indicators[DEFAULT_ITEM_COL].isin(self.data.assets)]

        # Then, we obtain the recommendation scores:
        if self.is_fitted:
            kpi_indicators["score"] = self.model.predict(kpi_indicators.drop(columns=[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]))
        else:
            kpi_indicators["score"] = kpi_indicators[DEFAULT_ITEM_COL].apply(lambda x: random.random())

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
        ) # number of items at rec_time

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
        user_recs_full.to_csv(f"for_testing/user_recs_{rec_time}.csv", index=False)

        return pd.concat(user_recommendations)
