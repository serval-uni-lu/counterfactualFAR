"""
MiFID II suitability of the recommended lists ("Finance metrics", Suitability).

Asset risk class: the seven market-risk classes of the PRIIPs summary risk indicator (Delegated
Regulation (EU) 2017/653, Annex II), with the annualised volatility of daily returns as a proxy
for the VaR-equivalent volatility. Customer profiles map to the highest class they may receive.

SVR@k = 1/|U| * sum_u 1/k * sum_j 1[class(i_j) > c_max(profile(u))]

The module also provides the suitability filter (drop assets above the customer's maximum after
ranking), a wrapper that evaluates any metric on the filtered lists, and the checks on the
classification itself (stability between time points, closeness to a band boundary).
"""

import numpy as np
import pandas as pd
from utils.constants import DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL, DEFAULT_USER_COL

from metrics.metric import Metric

# PRIIPs market-risk classes: upper volatility bound of classes 1-6 (class 7 is above 80%).
PRIIPS_EDGES = [0.005, 0.05, 0.12, 0.20, 0.30, 0.80]
# Reference profile limits of the protocol.
MAX_CLASS = {"Conservative": 2, "Income": 3, "Balanced": 5, "Aggressive": 7}
# Sensitivity check: every limit one class stricter.
STRICTER_MAX_CLASS = {p: max(1, c - 1) for p, c in MAX_CLASS.items()}
VOL_WINDOW_YEARS = 5           # PRIIPs look-back for daily prices
SENSITIVITY_WINDOW_YEARS = 1   # alternative window for the sensitivity check
MIN_RETURNS = 126              # fewer daily returns than this (about 6 months): no class
TRADING_DAYS = 252
BOUNDARY_TOLERANCE = 0.10      # "within 10% of a band boundary"

CLASSIFICATION_RULE = (f"PRIIPs MRM classes on annualised daily-return volatility, edges {PRIIPS_EDGES}; "
                       f"window {VOL_WINDOW_YEARS}y, min {MIN_RETURNS} returns; limits {MAX_CLASS}")


def annualised_volatility(time_series, rec_date, years=VOL_WINDOW_YEARS, min_returns=MIN_RETURNS):
    """Annualised volatility of daily returns per asset over the `years` before rec_date (inclusive).
    Assets with fewer than `min_returns` daily returns are left out."""
    start = rec_date - pd.DateOffset(years=years)
    ts = time_series[(time_series[DEFAULT_TIMESTAMP_COL] <= rec_date) & (time_series[DEFAULT_TIMESTAMP_COL] > start)]
    px = ts.pivot_table(index=DEFAULT_TIMESTAMP_COL, columns=DEFAULT_ITEM_COL, values=DEFAULT_RATING_COL,
                        aggfunc="last").sort_index()
    rets = px.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    vol = rets.std()[rets.notna().sum() >= min_returns] * np.sqrt(TRADING_DAYS)
    return vol.dropna()


def risk_class(vol):
    """PRIIPs market-risk class (1-7) for each volatility."""
    return pd.Series(np.digitize(vol.values, PRIIPS_EDGES) + 1, index=vol.index).astype(int)


def customer_profiles(customer_info_file, rec_date):
    """Risk profile per customer: the latest record on or before rec_date (or the first record if none
    is that old). "Predicted_" labels count as the plain profile; "Not_Available" is left out."""
    info = pd.read_csv(customer_info_file, usecols=["customerID", "riskLevel", "timestamp"],
                       parse_dates=["timestamp"])
    info["riskLevel"] = info["riskLevel"].str.replace("Predicted_", "", regex=False)
    info = info.sort_values("timestamp")
    before = info[info["timestamp"] <= rec_date].drop_duplicates("customerID", keep="last")
    after = info[~info["customerID"].isin(before["customerID"])].drop_duplicates("customerID", keep="first")
    prof = pd.concat([before, after]).set_index("customerID")["riskLevel"]
    return prof[prof.isin(list(MAX_CLASS))].to_dict()


class Suitability:
    """Risk classes, customer limits and the suitability filter for one recommendation date."""

    def __init__(self, time_series, rec_date, customer_info_file, years=VOL_WINDOW_YEARS, limits=None,
                 profiles=None):
        self.limits = limits or MAX_CLASS
        self.vol = annualised_volatility(time_series, rec_date, years)
        self.asset_class = risk_class(self.vol).to_dict()
        self.profile = profiles if profiles is not None else customer_profiles(customer_info_file, rec_date)
        self.max_class = {c: self.limits[p] for c, p in self.profile.items()}
        self._cache = (None, None)

    def annotate(self, recs):
        """Adds the asset class, the customer's maximum, and whether the pair is suitable."""
        out = recs.copy()
        out["asset_class"] = out[DEFAULT_ITEM_COL].map(self.asset_class)
        out["max_class"] = out[DEFAULT_USER_COL].map(self.max_class)
        out["suitable"] = out["asset_class"] <= out["max_class"]  # False when either is missing
        return out

    def eligible(self, assets, customer):
        """Assets with a known class within the customer's maximum."""
        cap = self.max_class.get(customer)
        if cap is None:
            return []
        return [a for a in assets if self.asset_class.get(a, 99) <= cap]

    def filter(self, recs):
        """Suitability filter applied after ranking: keeps only assets whose class is known and within
        the customer's maximum, in the original order. Customers without a profile get no assets."""
        if self._cache[0] is recs:
            return self._cache[1]
        out = recs[self.annotate(recs)["suitable"].values]
        self._cache = (recs, out)
        return out


def classification_checks(suit, suit_prev=None):
    """Stability of the classification: share of assets whose class changed since the previous time
    point, and share whose volatility lies within 10% of a band boundary."""
    edges = np.array(PRIIPS_EDGES)
    near = np.min(np.abs(suit.vol.values[:, None] - edges[None, :]) / edges[None, :], axis=1) <= BOUNDARY_TOLERANCE
    out = {"risk_class_assets": len(suit.asset_class),
           "risk_class_near_boundary_share": float(near.mean()) if len(near) else np.nan}
    for c in range(1, 8):
        out[f"risk_class_{c}_share"] = float(np.mean([v == c for v in suit.asset_class.values()])) \
            if suit.asset_class else np.nan
    if suit_prev is not None:
        both = set(suit.asset_class) & set(suit_prev.asset_class)
        out["risk_class_change_share"] = float(np.mean([suit.asset_class[a] != suit_prev.asset_class[a]
                                                         for a in both])) if both else np.nan
    return out


class SuitabilityViolationRate(Metric):
    """
    SVR@k as in the protocol: per customer, violations divided by the length of the top-k list,
    averaged over customers with a risk profile. An asset without a risk class is not counted as a
    violation; its share is reported separately (`<name>_unclassified_share@k`), and the filter
    removes such assets. `reference_values` also gives the share of customers with at least one
    unsuitable asset.
    """

    def __init__(self, data, suitability, name="svr", detail_cutoff=10):
        super().__init__(data)
        self.suit = suitability
        self.name = name
        self.detail_cutoff = detail_cutoff
        self.violations = None
        self.reference = {}

    def evaluate_cutoffs(self, recs, cutoffs, target_custs, only_test_customers):
        customers = self.data.users & set(self.data.test[DEFAULT_USER_COL].unique().flatten()) \
            if only_test_customers else self.data.users
        customers = customers & target_custs
        top = recs[recs[DEFAULT_USER_COL].isin(customers)].groupby(DEFAULT_USER_COL).head(max(cutoffs))
        top = self.suit.annotate(top)
        top["rank"] = top.groupby(DEFAULT_USER_COL).cumcount() + 1
        top = top[top["max_class"].notna()]
        top["violation"] = top["asset_class"].notna() & ~top["suitable"]

        res, self.reference = {}, {}
        for cutoff in cutoffs:
            part = top[top["rank"] <= cutoff]
            g = part.groupby(DEFAULT_USER_COL)
            per_cust = g["violation"].mean()  # violations / list length
            res[cutoff] = (per_cust.rename("metric").reset_index(), float(per_cust.mean()) if len(per_cust) else np.nan)
            self.reference[f"{self.name}_customers@{cutoff}"] = float((g["violation"].sum() > 0).mean()) \
                if len(per_cust) else np.nan
            self.reference[f"{self.name}_unclassified_share@{cutoff}"] = float(part["asset_class"].isna().mean()) \
                if len(part) else np.nan

        self.violations = top[top["violation"] & (top["rank"] <= self.detail_cutoff)][
            [DEFAULT_USER_COL, DEFAULT_ITEM_COL, "rank", "asset_class", "max_class"]]
        self.reference[f"{self.name}_customers_with_profile"] = int(top[DEFAULT_USER_COL].nunique())
        self.reference[f"{self.name}_customers_without_profile"] = int(len(customers) - top[DEFAULT_USER_COL].nunique())
        self.reference[f"{self.name}_violating_pairs@{self.detail_cutoff}"] = int(len(self.violations))
        return res

    def reference_values(self):
        return self.reference

    def extra_tables(self):
        return {} if self.violations is None or self.name != "svr" else {"svr_violations": self.violations}

    def evaluate_indiv(self, customer_df, cutoff):
        raise NotImplementedError("use evaluate_cutoffs")

    def evaluate_indiv_cutoffs(self, customer_df, cutoffs):
        raise NotImplementedError("use evaluate_cutoffs")


class FilteredMetric(Metric):
    """Evaluates another metric on the suitability-filtered recommendation lists. Only the metric
    values are passed on (not the wrapped metric's reference values or detail tables)."""

    def __init__(self, metric, suitability):
        super().__init__(metric.data)
        self.metric = metric
        self.suit = suitability

    def evaluate_cutoffs(self, recs, cutoffs, target_custs, only_test_customers):
        return self.metric.evaluate_cutoffs(self.suit.filter(recs), list(cutoffs), target_custs,
                                            only_test_customers)

    def evaluate_indiv(self, customer_df, cutoff):
        raise NotImplementedError("use evaluate_cutoffs")

    def evaluate_indiv_cutoffs(self, customer_df, cutoffs):
        raise NotImplementedError("use evaluate_cutoffs")
