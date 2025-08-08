#  Copyright (c) 2022. Terrier Team at University of Glasgow, http://http://terrierteam.dcs.gla.ac.uk
#
#  This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
#  If a copy of the MPL was not distributed with this  file, you can obtain one at
#  http://mozilla.org/MPL/2.0/.
from typing import Set

from utils.constants import DEFAULT_TIMESTAMP_COL, DEFAULT_ITEM_COL
from pandas import DataFrame

from data.filter.asset.asset_filter import AssetFilter


class AssetWithFullTest(AssetFilter):
    """
    Filter that keeps all assets which contain full test, i.e. those test which appear in the test set and have pricing
    information at the end of the test period.
    """
    def filter(self, time_series: DataFrame, train: DataFrame, valid: DataFrame, test: DataFrame, split_date) -> Set:
        assets_inter = set(test[DEFAULT_ITEM_COL].unique().flatten())
        max_date = time_series[DEFAULT_TIMESTAMP_COL].max()
        assets_time = set(time_series[time_series[DEFAULT_TIMESTAMP_COL] == max_date][DEFAULT_ITEM_COL].unique().flatten())
        assets = assets_inter & assets_time
        return assets