"""
SR@k: Sharpe ratio of the equally weighted top-k portfolio over the evaluation period.

Equal amounts go into each recommended asset at the recommendation date and are held to the
evaluation date without rebalancing, so the portfolio's final return equals ROI@k. The Sharpe ratio
is the mean daily excess return over the standard deviation of daily returns, annualised with
sqrt(252) (prices are on a business-day grid; missing days carry the last price forward).
"""

import numpy as np
import pandas as pd
from utils.constants import DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_USER_COL

from metrics.metric import Metric

TRADING_DAYS = 252


def relative_prices(time_series, rec_date, eval_date):
    """Daily price of each asset divided by its price at rec_date, from rec_date to eval_date.
    Only assets priced on both dates are kept (the same assets ROI@k can score)."""
    win = time_series[time_series[DEFAULT_TIMESTAMP_COL].between(rec_date, eval_date)]
    px = win.pivot_table(index=DEFAULT_TIMESTAMP_COL, columns=DEFAULT_ITEM_COL, values=DEFAULT_RATING_COL,
                         aggfunc="last").sort_index()
    if rec_date not in px.index or eval_date not in px.index:
        return pd.DataFrame()
    px = px.loc[:, px.loc[rec_date].gt(0) & px.loc[eval_date].notna()].ffill()
    return px / px.loc[rec_date]


def sharpe_ratio(value, rf_daily=0.0):
    """Annualised Sharpe ratio of a daily value series."""
    r = value.pct_change().dropna()
    sd = r.std(ddof=1)
    if len(r) < 2 or not sd > 0:
        return np.nan
    return float((r.mean() - rf_daily) / sd * np.sqrt(TRADING_DAYS))


class PortfolioSharpe(Metric):
    """
    SR@k per customer, averaged over customers with at least one priced asset in their top-k.
    Recommended assets without a price on both dates are left out of the portfolio.
    `reference_values` gives the market reference: the equally weighted portfolio of every
    priced asset.
    """

    def __init__(self, data, rec_date, eval_date, risk_free=0.0):
        """
        :param risk_free: annual risk-free rate (0.02 for 2%).
        """
        super().__init__(data)
        self.rel = relative_prices(data.time_series, rec_date, eval_date)
        self.rf_daily = (1.0 + risk_free) ** (1.0 / TRADING_DAYS) - 1.0
        self.market = sharpe_ratio(self.rel.mean(axis=1), self.rf_daily) if not self.rel.empty else np.nan

    def evaluate_cutoffs(self, recs, cutoffs, target_custs, only_test_customers):
        customers = self.data.users & set(self.data.test[DEFAULT_USER_COL].unique().flatten()) \
            if only_test_customers else self.data.users
        customers = customers & target_custs
        top = recs[recs[DEFAULT_USER_COL].isin(customers)].groupby(DEFAULT_USER_COL).head(max(cutoffs))
        lists = top.groupby(DEFAULT_USER_COL, sort=False)[DEFAULT_ITEM_COL].agg(list)
        priced = set(self.rel.columns)

        cache = {}
        res = {}
        for cutoff in cutoffs:
            rows = []
            for cust, items in lists.items():
                key = tuple(sorted(a for a in items[:cutoff] if a in priced))
                if not key:
                    continue
                if key not in cache:
                    cache[key] = sharpe_ratio(self.rel[list(key)].mean(axis=1), self.rf_daily)
                rows.append((cust, cache[key]))
            df = pd.DataFrame(rows, columns=[DEFAULT_USER_COL, "metric"])
            res[cutoff] = (df, float(df["metric"].mean()) if len(df) else np.nan)
        return res

    def reference_values(self):
        return {"market_sharpe": self.market}

    def evaluate_indiv(self, customer_df, cutoff):
        raise NotImplementedError("use evaluate_cutoffs")

    def evaluate_indiv_cutoffs(self, customer_df, cutoffs):
        raise NotImplementedError("use evaluate_cutoffs")
