import datetime

import pandas as pd
from joblib import Parallel, delayed
from utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_TIMESTAMP_COL,
)

from algorithms.kpi_gen.indicators import *
from algorithms.kpi_gen.kpi_generator import KPIGenerator


def _compute_asset_kpis(asset_df, k, kpi_type):
    """Compute all KPIs for a single asset. Called in parallel across assets."""
    asset_df = asset_df.sort_values(by=DEFAULT_TIMESTAMP_COL, ascending=True)
    periods_3 = [21, 63, 126] if kpi_type == "short" else [21, 63, 126, 189]
    periods_4 = [1, 21, 63, 126] if kpi_type == "short" else [1, 21, 63, 126, 189]
    asset_df = avg_price(asset_df, periods=periods_3)
    asset_df = roi(asset_df, periods=periods_4)
    asset_df = volatility(asset_df, periods=periods_3)
    asset_df = moving_average_convergence_divergence(asset_df)
    asset_df = momentum(asset_df, periods=periods_3)
    asset_df = rate_of_change(asset_df, periods=periods_3)
    asset_df = relative_strength_index(asset_df)
    asset_df = detrended_close_oscillator(asset_df)
    asset_df = sharpe(asset_df, periods=periods_3)
    asset_df = min_max(asset_df, periods=periods_3)
    for column in asset_df.columns:
        if column != DEFAULT_ITEM_COL and column != DEFAULT_TIMESTAMP_COL:
            asset_df[column] = asset_df[column].rolling(k).mean()
    return asset_df.dropna()


class MAKPIGenerator(KPIGenerator):
    """
    Class for computing the basic technical indicators for the recommendation. This generator applies a moving average
    over the last days of data to generate the definitive technical indicators.
    """

    def __init__(self, data, k, kpi_type):
        super().__init__()
        self.data = data
        self.k = k
        self.kpi_type = kpi_type

    def compute(self):
        """
        Computes the desired KPIs.
        :return: a dataframe containing the KPIs.
        """
        timea = datetime.datetime.now()
        asset_groups = [grp for _, grp in self.data.groupby(DEFAULT_ITEM_COL, sort=False)]

        # prefer='threads': pandas rolling ops release the GIL so threads parallelise
        # without the serialisation overhead of spawning loky worker processes.
        asset_dfs = Parallel(n_jobs=-1, prefer='threads')(
            delayed(_compute_asset_kpis)(grp, self.k, self.kpi_type)
            for grp in asset_groups
        )

        time_elapsed = datetime.datetime.now() - timea
        print(f"Generated indicators for {len(asset_dfs)} assets ({time_elapsed})")

        full_df = pd.concat(asset_dfs)
        self.kpis = full_df
