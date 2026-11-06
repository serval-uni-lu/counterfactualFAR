"""
Metrics of the "Finance metrics" protocol added to every evaluation window, on top of the pipeline's
nDCG@k / profitability@k / monthly_prof@k / volatility@k. Shared by recommendation.py and
compute_risk_metrics.py, so new runs and existing results use the same definitions.

Per cutoff k (also per customer in _customers.csv):
  monthly_roi@k            monthly ROI as the protocol defines it
  sharpe@k                 SR@k of the equally weighted top-k portfolio
  svr@k                    suitability violation rate on the model's ranking (unfiltered)
  svr_window1y@k, svr_stricter@k
                           the same with a 1-year volatility window / every limit one class stricter
  *_filtered@k             nDCG, monthly ROI, SR and SVR on the suitability-filtered ranking
  svr_filtered_window1y@k, svr_filtered_stricter@k
                           the filtered list checked against the alternative classifications
                           (a pass that holds only under the main one is "conditional")
Per window (reference values): market and risk-matched references, customer average, random and
popularity nDCG, %prof@k, exposure (Gini, coverage), in-house products, fairness gaps, and the
checks on the risk classification. See metrics/finance.py and metrics/suitability.py.
"""

import pandas as pd

from metrics.finance import FinanceProtocol, PortfolioMonthlyROI, customer_attributes, load_inhouse
from metrics.kpi_monthly_evaluation_metric import MonthlyKPIEvaluationMetric
from metrics.pure_ndcg import PureNDCG
from metrics.sharpe import PortfolioSharpe
from metrics.suitability import (CLASSIFICATION_RULE, SENSITIVITY_WINDOW_YEARS, STRICTER_MAX_CLASS, FilteredMetric,
                                 Suitability, SuitabilityViolationRate, classification_checks)


class MarketReference:
    """Not a ranking metric: the market's monthly ROI as monthly_prof@k computes it (each asset made
    monthly, then averaged), so it can be compared with monthly_prof@k; and the checks on the risk
    classification. The protocol-definition references are written by FinanceProtocol."""

    def __init__(self, profitability, days, checks=None):
        monthly = [pow(1.0 + v, 30 / days) - 1.0 for v in profitability.values()]
        self.values = {"market_monthly_prof": float(sum(monthly) / len(monthly)) if monthly else float("nan")}
        self.values.update(checks or {})

    def evaluate_cutoffs(self, recs, cutoffs, target_custs, only_test_customers):
        return {}

    def reference_values(self):
        return self.values


def risk_metrics(splitted_data, rec_date, eval_date, profitability, customer_info_file, risk_free=0.0,
                 prev_date=None):
    """
    :param profitability: raw return per asset between rec_date and eval_date (not padded with zeros).
    :param customer_info_file: customer_information.csv; without it, suitability and fairness are skipped.
    :param prev_date: previous time point, for the stability of the risk classes (default: 14 days before).
    :return: a list of (name, metric) pairs, in the format recommendation.test() expects.
    """
    days = (eval_date - rec_date).days
    sharpe = PortfolioSharpe(splitted_data, rec_date, eval_date, risk_free)
    out = [("monthly_roi", PortfolioMonthlyROI(splitted_data, dict(profitability), days)),
           ("sharpe", sharpe)]
    if customer_info_file is None:
        out.append(("market", MarketReference(profitability, days)))
        out.append(("finance", FinanceProtocol(splitted_data, rec_date, eval_date, dict(profitability), sharpe.rel,
                                               sharpe.rf_daily, inhouse=load_inhouse())))
        return out

    ts = splitted_data.time_series
    suit = Suitability(ts, rec_date, customer_info_file)
    suit_1y = Suitability(ts, rec_date, None, years=SENSITIVITY_WINDOW_YEARS, profiles=suit.profile)
    suit_strict = Suitability(ts, rec_date, None, limits=STRICTER_MAX_CLASS, profiles=suit.profile)
    prev = Suitability(ts, prev_date if prev_date is not None else rec_date - pd.Timedelta(days=14), None,
                       profiles=suit.profile)
    checks = classification_checks(suit, prev)
    checks["classification_rule"] = CLASSIFICATION_RULE  # non-numeric: recorded, ignored by the stats

    out += [
        ("market", MarketReference(profitability, days, checks)),
        ("finance", FinanceProtocol(splitted_data, rec_date, eval_date, dict(profitability), sharpe.rel,
                                    sharpe.rf_daily, suit=suit, attrs=customer_attributes(customer_info_file, rec_date),
                                    inhouse=load_inhouse())),
        ("svr", SuitabilityViolationRate(splitted_data, suit)),
        ("svr_window1y", SuitabilityViolationRate(splitted_data, suit_1y, name="svr_window1y")),
        ("svr_stricter", SuitabilityViolationRate(splitted_data, suit_strict, name="svr_stricter")),
        ("svr_filtered", FilteredMetric(SuitabilityViolationRate(splitted_data, suit), suit)),
        ("svr_filtered_window1y", FilteredMetric(SuitabilityViolationRate(splitted_data, suit_1y), suit)),
        ("svr_filtered_stricter", FilteredMetric(SuitabilityViolationRate(splitted_data, suit_strict), suit)),
        ("ndcg_filtered", FilteredMetric(PureNDCG(splitted_data), suit)),
        ("monthly_prof_filtered", FilteredMetric(
            MonthlyKPIEvaluationMetric(splitted_data, dict(profitability), days), suit)),
        ("monthly_roi_filtered", FilteredMetric(PortfolioMonthlyROI(splitted_data, dict(profitability), days), suit)),
        ("sharpe_filtered", FilteredMetric(PortfolioSharpe(splitted_data, rec_date, eval_date, risk_free), suit)),
    ]
    return out
