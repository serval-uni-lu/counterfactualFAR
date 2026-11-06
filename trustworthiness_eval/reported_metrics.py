"""
Figures for the reported metrics (Table 3.1.12 of "Finance metrics"), one figure per trustworthiness property.

The main figures (in "reported metrics/") are the mean over the 61 time points on the counterfactual price
series (synthetic_eval.py, attribution.py --synthetic); the same figures on actual prices (normal runs,
results/rfr/<run>_<date>_metrics.csv) are written to "reported metrics/real/". Rows of the table that measure close things share a figure: 01 accuracy (prediction,
ranking, return), 02 suitability, 03 fairness, 04 robustness, 05 explainability, 06 privacy,
07 statistical uncertainty, 08 monitoring.

Inputs, all under trustworthiness_eval/outputs/<model>/:
    all_dates/   actual prices, every time point: robustness.py, attribution.py, monitoring.py,
                 cf_quality_all_dates.py, cf_tests_all_dates.py, privacy/holdings_inference/holdings_inference.py
    synthetic/   counterfactual price series: synthetic_eval.py, attribution.py --synthetic
plus results/rfr/<run>_<date>_metrics.csv for the per-time-point metrics on actual prices.

build_recs, window_values and holdings_values are also used by synthetic_eval.py, so the metrics on the
counterfactual series are computed with the same code.

Usage (from the repo root):
    python trustworthiness_eval/reported_metrics.py
"""

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "trustworthiness_eval" / "outputs"
for p in (ROOT, ROOT / "trustworthiness_eval", ROOT / "counterfactuals_generation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

pd.set_option("future.infer_string", False)  # the data loaders call .flatten() on string columns

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


STRESS = (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30"))  # COVID-19 market fall
HORIZON = pd.DateOffset(months=6)  # horizon of each time point

from process_results import load_window_metrics  # noqa: E402
from utils.constants import DEFAULT_ITEM_COL as ITEM  # noqa: E402
from utils.constants import DEFAULT_RATING_COL as RATING  # noqa: E402
from utils.constants import DEFAULT_USER_COL as USER  # noqa: E402

HOLDINGS_CUTOFFS = [10, 20, 50, 100, "full"]

# Reference palette (dataviz skill, light mode).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]
SHAP_COLORS = ["#7c5cd6", "#c4458f", "#d4a21a"]  # violet, magenta, gold: not used by the other figures
MEAN_GREY = "#a3a29c"
REF_GREYS = ["#7d7c76", "#c3c2b7"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

# Caption under each figure: what is measured and how to read it. "\n" starts a new paragraph.
CAPTIONS = {
    "accuracy": "Prediction: the model predicts each asset's return over the next 6 months from 30 indicators computed "
                "from its past prices, scored on test rows the model was not trained on. R² is the share of the "
                "variation in the realised return that the predictions explain (1 = perfect, 0 = no better than "
                "predicting the average, below 0 = worse). RMSE and MAE are the typical size of the error, in "
                "percentage points of return; RMSE weighs large errors more. nDCG@10: whether the 10 assets "
                "ranked highest for a customer include the assets that customer bought in the next 6 months, with more "
                "credit near the top (0 = none bought, 1 = purchases fill the top); it measures agreement with customer "
                "behaviour, not the quality of the advice. Random: 10 assets drawn at random from those the customer "
                "does not hold. Monthly ROI@10 = (1 + return)^(30/days) - 1 and SR@10 (Sharpe ratio, annualised) of an "
                "equally weighted portfolio of each customer's top-10 over the next 6 months, averaged over customers. "
                "Risk-matched market: all assets the customer's risk profile allows; Market: all assets, equally "
                "weighted; Customer average: return of the assets customers acquired over the same horizon. Vol@10 "
                "(annualised volatility of the recommended assets): {vol}. %prof@10 (share of the 10 assets with a "
                "non-negative return): {prof}. Decision rules: nDCG@10 above random; ROI@10 and SR@10 above the "
                "risk-matched market.",
    "suitability": "SVR@10 (suitability violation rate): share of the 10 recommended assets whose risk class is above "
                   "the highest class the customer's risk profile allows. Risk classes are the 7 PRIIPs classes from the "
                   "5-year annualised volatility of daily returns; limits: Conservative 2, Income 3, Balanced 5, "
                   "Aggressive 7. Left: the filtered list the system shows, under the main classification and two "
                   "alternatives (1-year volatility window; every limit one class stricter). This is the only pass/fail "
                   "rule: under the main classification it must be 0 at every time point, so the mean must be 0; a pass that "
                   "holds only under the main classification is reported as conditional. Middle: the model's own top-10 before "
                   "the filter, as a share of assets and a share of customers with at least one unsuitable asset; it "
                   "shows how much the system relies on the filter. Right: assets whose class changed since the previous "
                   "time point (two weeks earlier), and assets within 10% of a class boundary, where a small price move "
                   "could change the class.",
    "fairness": "Top row, group gap: the highest minus the lowest group mean, over customer segments (Mass, Premium, "
                "Professional, ...), investment-capacity bands and risk profiles; 0 = all groups served equally. Groups "
                "under 10 customers are left out. Monthly ROI@10 is compared only within the same risk profile (gap per "
                "profile, weighted by the profile's share of customers), since conservative customers are expected to "
                "earn less. The SVR@10 gap by risk profile is high by construction: every customer sees the same "
                "unfiltered ranking and conservative customers have the strictest limit. Bottom row: Gini of exposure = "
                "how unevenly recommendations are spread over the assets (0 = every asset equally often, close to 1 = a "
                "few assets everywhere); catalogue coverage = share of all assets recommended to at least one customer; "
                "in-house ratio = share of NBG's own funds in the top-10 divided by their share among the eligible "
                "assets (1 = proportional, above 1 = favoured); %prof@10 by origin = share of recommended assets with a "
                "non-negative return, for in-house and third-party products.",
    "robustness": "Top row, market crisis: stress time points are those whose 6-month horizon overlaps the COVID-19 "
                  "fall of February–April 2020; all others are calm. Relative stress drop = (calm mean - stress mean) / "
                  "calm mean. Worst time point: the lowest value over the 61 time points. Bottom row, reference shocks "
                  "applied to the 20 trading days of prices before the recommendation date: equity (stocks and equity "
                  "funds fall 30%, volatility doubled), rates (bonds and bond funds fall 10%, a 200 bp rise with a 5-year "
                  "duration) and combined. Jaccard J@10: overlap of each customer's filtered top-10 before and after the "
                  "shock (1 = unchanged). SVR@10 under shock: violation rate of the filtered list after the risk classes "
                  "are recomputed on the shocked prices; it must stay 0. Last panel: share of assets the shock moves into "
                  "a higher risk class.",
    "shap": "SHAP splits each asset's predicted return into the contributions of the 30 indicators. Stability: "
            "attributions of the recommended assets recomputed after small changes to the price history (noise of "
            "0.1% or 0.25% on each daily price, or 2% of trading days dropped); Spearman correlation of the 30 values "
            "and overlap of the 5 most important indicators with the original attributions (1 = identical). "
            "Faithfulness: change in the prediction when the n most important indicators are set to their training "
            "median, divided by the change when n random indicators are; above 1 = the attributions point to the "
            "indicators the model relies on.",
    "shocks":"Reference shocks applied to the 20 trading days of prices before the recommendation date: equity "
              "(stocks and equity funds fall 30%, volatility doubled), rates (bonds and bond funds fall 10%, a 200 bp "
              "rise with a 5-year duration) and combined. J@10: overlap of each customer's filtered top-10 before and "
              "after the shock (1 = unchanged). Right: share of assets the shock moves into a higher risk class. "
              "Suitability holds under every shock: with the risk classes recomputed on the shocked prices, the "
              "filtered list has SVR@10 = 0 for all three scenarios at all {n} time points (max {svr}).",
    "explainability": "Counterfactuals: the smallest change to an asset's last 21 daily prices that raises its predicted "
                      "return. Validity: share that reach the requested prediction when re-scored through "
                      "the full model. Proximity: median relative price change (1% = prices moved "
                      "by 1% on average). Sparsity: share of the 21 prices changed (lower = simpler; below 50% counts as "
                      "actionable). Plausibility: share of price paths with positive prices and daily moves inside the "
                      "asset's own historical range. SHAP: each asset's predicted return split into the contribution of "
                      "the 30 indicators. Stability: attributions of the recommended assets recomputed after small "
                      "changes to the price history (noise of 0.1% or 0.25% on each daily price, or 2% of trading days "
                      "dropped); Spearman correlation of the 30 values and overlap of the 5 most important indicators "
                      "(1 = identical). Faithfulness: change in the prediction when the n most important indicators are "
                      "set to their training median, divided by the change for n random indicators; above 1 = the "
                      "attributions point to the indicators that matter.",
    "privacy": "Every customer receives the same ranking minus the assets they already hold, so the gaps in a "
               "customer's list reveal those holdings to anyone who knows the ranking. For each list length, the assets "
               "ranked above the last one shown but missing from the list are inferred as held and compared with the "
               "customer's actual holdings before the recommendation date. Holdings recovered: share of all holdings "
               "that are inferred. Customers exposed: share of "
               "customers with at least one holding revealed. The exposure arises only if the lists reach someone who "
               "does not already have the customers' transactions.",
    "uncertainty": "nDCG@10, monthly ROI@10 and SR@10 of the model's top-10 at each of the 61 time points (each on "
                   "its own scale). The time points are two weeks apart and their 6-month horizons overlap, so "
                   "neighbouring time points share most of their returns and are not independent observations (about "
                   "5 independent 6-month periods in 61 time points); the spread is therefore shown descriptively.",
    "monitoring": "The population stability index (PSI) compares the distribution of a quantity at the recommendation "
                  "date with its distribution in the model's training data, over 10 bins: PSI = sum over bins of "
                  "(current share - training share) × ln(current share / training share). Below 0.1 (dotted line) = "
                  "stable, 0.1 to 0.25 = moderate shift, above 0.25 = significant shift, which triggers a review of the "
                  "model.",
}


# --------------------------------------------------------------------------------------------- #
# Compute: the window metrics on the counterfactual lists                                        #
# --------------------------------------------------------------------------------------------- #

def build_recs(scores, split):
    """Every test customer's ranking minus the assets they held in training, as recommend() writes it."""
    order = pd.Series(scores).sort_values(ascending=False, kind="mergesort")
    held = split.train.groupby(USER)[ITEM].agg(set).to_dict()
    customers = sorted(split.users & set(split.test[USER].unique()))
    parts = []
    for u in customers:
        keep = order[~order.index.isin(held.get(u, set()))]
        parts.append(pd.DataFrame({USER: u, ITEM: keep.index, RATING: keep.values}))
    recs = pd.concat(parts, ignore_index=True)
    return recs.sort_values([USER, RATING], ascending=[False, False], kind="mergesort")


def window_values(recs, split, rec, fut, prof, vol, prev, customer_info):
    """Every per-window metric of the protocol for one set of lists, with the 61-time-point code."""
    from compute_risk_metrics import evaluate
    from metrics.kpi_evaluation_metric import KPIEvaluationMetric
    from metrics.kpi_monthly_evaluation_metric import MonthlyKPIEvaluationMetric
    from metrics.pure_ndcg import PureNDCG
    from metrics.risk_metrics import risk_metrics

    days = (fut - rec).days
    # KPIEvaluationMetric pads its dict with zeros for unpriced assets, so every metric gets a copy.
    metrics = [("ndcg", PureNDCG(split)),
               ("monthly_prof", MonthlyKPIEvaluationMetric(split, dict(prof), days)),
               ("volatility", KPIEvaluationMetric(split, dict(vol)))]
    metrics += risk_metrics(split, rec, fut, dict(prof), customer_info, 0.0, prev_date=prev)
    return evaluate(recs, metrics, split.users)


def holdings_values(recs, data_dir, date):
    from trustworthiness_eval.privacy.holdings_inference.holdings_inference import (infer_holdings, inference_metrics, load_holdings,
                                                               presence_matrix)
    present, customers, assets = presence_matrix(recs)
    held = load_holdings(data_dir, date, customers, assets)
    out = {}
    for k in HOLDINGS_CUTOFFS:
        m = inference_metrics(infer_holdings(present, k), held)
        for key in ("holdings_recall", "holdings_precision", "customers_exposed"):
            out[f"{key}@{k}"] = m[key]
    return out


# --------------------------------------------------------------------------------------------- #
# Plot                                                                                           #
# --------------------------------------------------------------------------------------------- #

class Sources:
    """Per-time-point metrics on actual prices (results/rfr) and the all_dates/ folder."""

    def __init__(self, args):
        self.tw = OUT / args.model / "all_dates"
        long, _ = load_window_metrics(str(ROOT / "results" / "rfr"), args.results_run)
        long["value"] = pd.to_numeric(long["value"], errors="coerce")
        self.series = long.pivot_table(index="split_date", columns="metric", values="value", aggfunc="first").sort_index()
        self.series.index = pd.to_datetime(self.series.index)

    def mean61(self, metric):
        s = self.series.get(metric)
        return float(s.mean()) if s is not None else np.nan

    def n61(self):
        return len(self.series)


class SyntheticSources(Sources):
    """The same interface as Sources, read from synthetic_eval.py (counterfactual price series)."""

    def __init__(self, args):
        self.tw = OUT / args.model / "synthetic"
        long = pd.read_csv(self.tw / "synthetic_metrics.csv")
        self.series = long.pivot_table(index="date", columns="metric", values="value", aggfunc="first").sort_index()
        self.series.index = pd.to_datetime(self.series.index)


def synthetic_shap(args):
    """SHAP stability and faithfulness on the counterfactual series (attribution.py --synthetic);
    None for both while they are not computed."""
    d = OUT / args.model / "synthetic"
    if not (d / "attribution_stability.csv").exists() or not (d / "attribution_faithfulness.csv").exists():
        return None, None
    return pd.read_csv(d / "attribution_stability.csv"), pd.read_csv(d / "attribution_faithfulness.csv")


def style(ax, title):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fmt(v, pct):
    if v is None or np.isnan(v):
        return ""
    if pct:
        if abs(v) >= 0.9995:
            return f"{v:.0%}"
        return f"{v:.2%}" if 0 < abs(v) < 0.01 else f"{v:.1%}"
    return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"


def label_bar(ax, bar, v, pct, size=7):
    """Value at the end of the bar: above a positive bar, below a negative one."""
    if v is None or np.isnan(v):
        ax.annotate("n/a", (bar.get_x() + bar.get_width() / 2, 0), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=size, color=INK2)
        return
    up = v >= 0
    ax.annotate(fmt(v, pct), (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 2 if up else -2),
                textcoords="offset points", ha="center", va="bottom" if up else "top", fontsize=size, color=INK2)


def figure(n, title, note, size=(4.2, 3.6), rows=1):
    """n panels per row, `rows` rows; size is the size of one panel. Returns the flat list of axes."""
    fig, axes = plt.subplots(rows, n, figsize=(size[0] * n, size[1] * rows), squeeze=False)
    fig.suptitle(title, x=0.01, y=0.98, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig._note = fig.text(0.01, 0.905, note, fontsize=8, color=INK2, ha="left")
    fig._rows = rows
    return fig, list(axes.flat)


def save(fig, path, bottom=0.2, caption="", wspace=0.38):
    """Lays out the panels and adds the caption under them; the figure grows by the caption's height
    so the panels keep their size."""
    w, h = fig.get_size_inches()
    rows = getattr(fig, "_rows", 1)
    lines = []
    for para in caption.split("\n"):
        lines += textwrap.wrap(para, width=int(w * 14.5)) or [""]
    cap_h = 0.15 * len(lines) + 0.2 if caption else 0.0
    H = h + cap_h
    row_h = h / rows
    fig.set_size_inches(w, H)
    fig.subplots_adjust(top=1 - 0.22 * row_h / H, bottom=(bottom * row_h + cap_h) / H, wspace=wspace,
                        hspace=0.6 if rows > 1 else 0.2, left=0.06, right=0.94)
    fig._suptitle.set_y(1 - 0.02 * row_h / H)
    fig._note.set_y(1 - 0.095 * row_h / H)
    if caption:
        fig.text(0.01, (cap_h - 0.08) / H, "\n".join(lines), fontsize=8, color=INK, ha="left", va="top",
                 linespacing=1.4)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"saved {path.name}")


def bars(ax, title, items, pct=False, hline=None, colors=None, width=0.7):
    """One bar per item, each labelled under it. items: [(label, value)]."""
    labels = [textwrap.fill(str(lab), 14) for lab, _ in items]
    vals = [np.nan if v is None else float(v) for _, v in items]
    colors = colors or (SERIES * 3)[:len(items)]
    x = np.arange(len(items))
    drawn = ax.bar(x, np.nan_to_num(vals, nan=0.0), width, color=colors, edgecolor="white", linewidth=1)
    for b, v in zip(drawn, vals):
        label_bar(ax, b, v, pct)
    if hline is not None:
        ax.axhline(hline[0], color=INK2, linewidth=1, linestyle="--")
        if hline[1]:
            ax.annotate(hline[1], (1.0, hline[0]), xycoords=("axes fraction", "data"), xytext=(2, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=7, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.6, len(items) - 0.4)
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    style(ax, title)


def headroom(ax, f=1.15):
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo * (f if lo < 0 else 1), hi * f if hi > 0 else hi)


MODEL, REF1, REF2 = BLUE, REF_GREYS[0], REF_GREYS[1]


def read_all_dates(args, name, n, **kw):
    """A file of the run over all time points (all_dates/). The figures and the table state how many
    dates it covers; a warning is printed when it covers fewer than the n time points."""
    d = pd.read_csv(OUT / args.model / "all_dates" / name, **kw)
    got = d.loc[d["date"] != "ALL", "date"].nunique()
    if got < n:
        print(f"warning: {name} covers {got} of {n} time points")
    return d


def cf_quality_mean(args, n):
    """Counterfactual quality at the recommendation date, averaged over the dates of
    cf_quality_all_dates.py, and the number of dates."""
    q = read_all_dates(args, "cf_quality.csv", n)
    q = q[q["scope"] == "recommendation_date"]
    return q.mean(numeric_only=True), q["date"].nunique()


def plot_privacy(args, src, out_dir, synthetic=False, prices="actual prices"):
    """Holdings inference by list length, mean over the time points: recovered, precision, customers exposed."""
    ks = [str(k) for k in HOLDINGS_CUTOFFS]
    if synthetic:
        n_h = src.n61()
        h = pd.DataFrame({col: [src.mean61(f"{col}@{k}") for k in ks]
                          for col in ("holdings_recall", "customers_exposed")}, index=ks)
    else:
        h = read_all_dates(args, "holdings_inference.csv", src.n61(), dtype={"k": str})
        n_h = h.loc[h["date"] != "ALL", "date"].nunique()
        h = h[h["date"] == "ALL"].set_index("k")
    fig, ax = figure(1, "Privacy: holdings inferred from the recommendation lists",
                     f"Mean over {n_h} time points, {prices}.", size=(9, 3.9))
    x = np.arange(len(ks))
    series = [("holdings_recall", "Holdings recovered"), ("customers_exposed", "Customers exposed")]
    w = 0.8 / len(series)
    for i, (col, name) in enumerate(series):
        ys = [h.at[k, col] for k in ks]
        drawn = ax[0].bar(x + (i - (len(series) - 1) / 2) * w, ys, w * 0.92, color=[MODEL, REF1][i],
                          edgecolor="white", label=name)
        for b, v in zip(drawn, ys):
            label_bar(ax[0], b, v, True, size=10)
    ax[0].set_xticks(x, [f"Top {k}" if k != "full" else "Full list" for k in ks])
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax[0].set_ylim(0, 1.12)
    style(ax[0], "By the length of the list the observer sees")
    ax[0].tick_params(labelsize=10)
    ax[0].title.set_fontsize(12)
    ax[0].legend(fontsize=10, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    save(fig, out_dir / "06_privacy.png", bottom=0.25, caption=CAPTIONS["privacy"])


def plot_uncertainty(src, out_dir, prices="actual prices"):
    """Spread of nDCG@10, monthly ROI@10 and SR@10 over the 61 time points: one box each, mean marked."""
    n = src.n61()
    fig, ax = figure(3, f"Spread over the {n} time points",
                     "Box: median and interquartile range; whiskers: lowest and highest time point; dot: mean. "
                     f"{prices.capitalize()}.", size=(3.2, 3.9))
    for a, (key, name, pct) in zip(ax, [("ndcg@10", "nDCG@10", False), ("monthly_roi@10", "Monthly ROI@10", True),
                                       ("sharpe@10", "SR@10", False)]):
        v = src.series.get(key).dropna().values
        a.boxplot([v], positions=[0], widths=0.5, whis=(0, 100), patch_artist=True, showfliers=False,
                  boxprops=dict(facecolor=GRID, edgecolor=INK2), medianprops=dict(color=INK, linewidth=1.5),
                  whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
        mean = float(np.mean(v))
        a.scatter([0], [mean], color=MODEL, s=36, zorder=3, edgecolor="white")
        if v.min() < 0 < v.max():
            a.axhline(0, color=INK2, linewidth=1, linestyle="--")
        a.set_xticks([])
        a.set_xlim(-0.5, 0.9)
        if pct:
            a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
        style(a, name)
    save(fig, out_dir / "07_uncertainty.png", bottom=0.12, caption=CAPTIONS["uncertainty"])


def plot_all(args, out_dir, fig_dir, synthetic=False):
    """Figures on the actual prices (synthetic=False) or on the counterfactual price series (synthetic=True),
    written to fig_dir."""
    src = SyntheticSources(args) if synthetic else Sources(args)
    out_dir = fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tw = src.tw
    m = src.mean61
    n = src.n61()
    prices = "counterfactual price series" if synthetic else "actual prices"
    MEAN_NOTE = f"Mean over the {n} time points (6-month windows), {prices}."
    acc_caption = CAPTIONS["accuracy"].format(
        vol=f"{fmt(m('volatility@10'), True)} (whole catalogue {fmt(m('volatility@1000'), True)})",
        prof=fmt(m("pct_prof@10"), True))
    if synthetic:
        acc_caption += (" The counterfactual series is noisier than actual prices (each day's price comes from a "
                        "separate counterfactual search), which inflates volatility for every list; compare Vol@10 "
                        "only with the whole catalogue on the same series.")

    # 1. Accuracy ------------------------------------------------------------------------------
    fig, ax = figure(5, "Accuracy", MEAN_NOTE, size=(3.4, 3.9))
    bars(ax[0], "Prediction: R²", [("R²", m("test_r2"))], colors=[MODEL], hline=(0, ""), width=0.4)
    bars(ax[1], "Prediction error", [("RMSE", m("test_rmse")), ("MAE", m("test_mae"))], pct=True,
         colors=[MODEL, MODEL])
    bars(ax[2], "Ranking: nDCG@10", [("Model", m("ndcg@10")), ("Random", m("ndcg_random@10"))],
         colors=[MODEL, REF1])
    bars(ax[3], "Monthly ROI@10", [("Model", m("monthly_roi@10")), ("Risk-matched market", m("market_matched_monthly_roi")),
                                   ("Market", m("market_monthly_roi")),
                                   ("Customer average", m("customer_avg_monthly_roi"))], pct=True,
         colors=[MODEL, REF1, REF2, AQUA])
    bars(ax[4], "Sharpe ratio SR@10", [("Model", m("sharpe@10")), ("Risk-matched market", m("market_matched_sharpe")),
                                       ("Market", m("market_sharpe"))], colors=[MODEL, REF1, REF2])
    for a in ax:
        headroom(a)
    save(fig, out_dir / "01_accuracy.png", bottom=0.2, caption=acc_caption)

    # 2. Suitability ---------------------------------------------------------------------------
    fig, ax = figure(3, "Suitability", MEAN_NOTE, size=(4.4, 3.9))
    bars(ax[0], "SVR@10, filtered list (main: must be 0)",
         [("Main classification", m("svr_filtered@10")), ("1-year volatility window", m("svr_filtered_window1y@10")),
          ("Limits one class stricter", m("svr_filtered_stricter@10"))], pct=True, colors=[MODEL, REF1, REF2])
    ax[0].set_ylim(0, 1)
    bars(ax[1], "Unfiltered ranking", [("SVR@10 (assets)", m("svr@10")),
                                       ("Customers with an unsuitable asset", m("svr_customers@10"))],
         pct=True, colors=[MODEL, MODEL])
    bars(ax[2], "Stability of the risk classification",
         [("Changed class since previous time point", m("risk_class_change_share")),
          ("Within 10% of a class boundary", m("risk_class_near_boundary_share"))], pct=True, colors=[REF1, REF1])
    for a in ax[1:]:
        headroom(a)
    save(fig, out_dir / "02_suitability.png", bottom=0.24, caption=CAPTIONS["suitability"])

    # 3. Fairness ------------------------------------------------------------------------------
    attrs = [("customerType", "Segment"), ("investmentCapacity", "Capacity band"), ("riskLevel", "Risk profile")]

    def sig(metric, a):
        p = src.series.get(f"fair_gap_{metric}@10_{a}_perm_p")
        return f"{(p.dropna() < 0.05).mean():.0%}" if p is not None and p.notna().any() else "n/a"

    fig, ax = figure(3, "Fairness", MEAN_NOTE + "  Under each group gap: share of time points where it is significant "
                     "(permutation test, p < 0.05).", size=(4.6, 3.9), rows=2)
    bars(ax[0], "nDCG@10 gap", [(f"{lab} ({sig('ndcg', a)})", m(f"fair_gap_ndcg@10_{a}")) for a, lab in attrs])
    bars(ax[1], "SVR@10 gap (unfiltered)", [(f"{lab} ({sig('svr', a)})", m(f"fair_gap_svr@10_{a}")) for a, lab in attrs],
         pct=True)
    bars(ax[2], "Monthly ROI@10 gap within risk profile",
         [(lab, m(f"fair_strat_gap_monthly_roi@10_{a}")) for a, lab in attrs[:2]], pct=True)
    bars(ax[3], "Exposure across assets (top-10)", [("Gini of exposure", m("gini@10")),
                                                   ("Catalogue coverage", m("coverage@10"))], colors=[MODEL, MODEL])
    bars(ax[4], "In-house share / eligible share", [("In-house ratio", m("inhouse_ratio@10"))], colors=[MODEL],
         hline=(1, "1 = proportional"), width=0.4)
    bars(ax[5], "%prof@10 by product origin", [("In-house", m("pct_prof_inhouse@10")),
                                              ("Third-party", m("pct_prof_thirdparty@10"))], pct=True,
         colors=[MODEL, REF1])
    for a in ax:
        headroom(a)
    save(fig, out_dir / "03_fairness.png", bottom=0.2, caption=CAPTIONS["fairness"])

    # 4. Robustness ----------------------------------------------------------------------------
    t = src.series.index.to_series()
    stress = (t < STRESS[1]) & ((t + HORIZON) > STRESS[0])
    shocks = (pd.read_csv(tw / "synthetic_shocks.csv") if synthetic
              else read_all_dates(args, "robustness_market_shocks.csv", src.n61()))
    n_shock = shocks["date"].nunique()
    fig, ax = figure(3, "Robustness",
                     f"Top row: {prices} over the {n} time points (stress = 6-month horizon overlaps "
                     f"{STRESS[0]:%b %Y}–{STRESS[1]:%b %Y}, {int(stress.sum())} time points). Bottom row: mean over "
                     f"the {n_shock} time points of the shocked prices.", size=(4.6, 3.9), rows=2)
    for a, (key, name, pct) in zip(ax[:3], [("ndcg@10", "nDCG@10", False), ("monthly_roi@10", "Monthly ROI@10", True),
                                           ("sharpe@10", "SR@10", False)]):
        s = src.series.get(key)
        calm, stressed = s[~stress].mean(), s[stress].mean()
        drop = (calm - stressed) / calm if calm else np.nan
        bars(a, f"{name} (relative stress drop {drop:.0%})",
             [("Calm mean", calm), ("Stress mean", stressed), ("Worst time point", s.min())], pct=pct,
             colors=[MODEL, REF1, REF2])
    scen = ["equity", "rates", "combined"]
    by = shocks.groupby("scenario").mean(numeric_only=True)
    for a, (col, name, pct) in zip(ax[3:], [("j@10_filtered_mean", "Shock: Jaccard J@10 before vs after", False),
                                           ("svr@10_shocked_filtered", "Shock: SVR@10, filtered (must be 0)", True),
                                           ("reclassified_up_share", "Shock: assets moved to a higher risk class",
                                            True)]):
        bars(a, name, [(s.capitalize(), by.at[s, col]) for s in scen], pct=pct)
        if col == "svr@10_shocked_filtered":
            a.set_ylim(0, 1)
        if col == "j@10_filtered_mean":
            a.set_ylim(0, 1.1)
    for a in ax[:3] + ax[5:]:
        headroom(a)
    save(fig, out_dir / "04_robustness.png", bottom=0.2, caption=CAPTIONS["robustness"])

    # 4b. Market shocks alone ------------------------------------------------------------------
    fig, ax = figure(2, "Robustness: market shocks", f"Mean over the {n_shock} time points, shocked prices.",
                     size=(4.2, 3.6))
    bars(ax[0], "J@10 before vs after", [(s.capitalize(), by.at[s, "j@10_filtered_mean"]) for s in scen],
         hline=(1, "1 = unchanged"))
    ax[0].set_ylim(0, 1.1)
    bars(ax[1], "Assets moved to a higher risk class",
         [(s.capitalize(), by.at[s, "reclassified_up_share"]) for s in scen], pct=True)
    headroom(ax[1])
    svr_max = shocks["svr@10_shocked_filtered"].max()
    save(fig, out_dir / "04b_market_shocks.png", bottom=0.14,
         caption=CAPTIONS["shocks"].format(svr=fmt(svr_max, True), n=n_shock))

    # 5. Explainability ------------------------------------------------------------------------
    q, n_cf = cf_quality_mean(args, n)
    if synthetic:
        st, fa = synthetic_shap(args)
    else:
        st = read_all_dates(args, "attribution_stability.csv", src.n61())
        fa = read_all_dates(args, "attribution_faithfulness.csv", src.n61())
    st = st[st["perturbation"].str.startswith(("noise", "missing"))]
    fa = fa[fa["subset"] == "recommended"]
    n_attr = st["date"].nunique()
    cf_note = (f"Counterfactuals: mean over {n_cf} time points" if n_cf > 2 else
               "Counterfactuals: mean of the 2 dates for which they were generated (2020-08-28, 2021-11-23)")
    fig, ax = figure(5, "Explainability", f"{cf_note}. SHAP: mean over {n_attr} time points, {prices}.",
                     size=(3.6, 3.9))
    bars(ax[0], "Counterfactuals: validity", [("Re-checked end to end", q["validity_rechecked_end_to_end"])],
         pct=True)
    bars(ax[1], "Counterfactuals: quality", [("Proximity (price change)", q["mean_rel_change_median"]),
                                             ("Sparsity (prices changed)", q["sparsity_share_changed_mean"]),
                                             ("Plausibility", q["plausibility_cf"])], pct=True)
    pert = [("noise_0.001", "Noise 0.1%"), ("noise_0.0025", "Noise 0.25%"), ("missing_0.02", "2% days dropped")]
    sp = st.groupby("perturbation").mean(numeric_only=True)
    bars(ax[2], "SHAP stability: Spearman", [(lab, sp.at[p, "spearman_recommended_mean"]) for p, lab in pert])
    bars(ax[3], "SHAP stability: top-5 overlap", [(lab, sp.at[p, "top5_jaccard_recommended_mean"]) for p, lab in pert])
    for a in ax[2:4]:
        a.set_ylim(0, 1.1)
    fr = fa.groupby("n_features")["ratio_top_to_random"].mean()
    bars(ax[4], "SHAP faithfulness: top vs random", [(f"Top {k}", fr.get(k)) for k in (1, 5, 10)],
         hline=(1, "1 = random"))
    for a in ax[:2] + ax[4:]:
        headroom(a)
    save(fig, out_dir / "05_explainability.png", bottom=0.24, caption=CAPTIONS["explainability"])

    # 5b. SHAP alone ---------------------------------------------------------------------------
    fig, ax = figure(3, "Explainability: SHAP attributions", f"Mean over the {n_attr} time points, {prices}.",
                     size=(3.8, 3.6))
    bars(ax[0], "Stability: Spearman (1 = identical)",
         [(lab, sp.at[p, "spearman_recommended_mean"]) for p, lab in pert], colors=SHAP_COLORS)
    bars(ax[1], "Stability: top-5 overlap (1 = identical)",
         [(lab, sp.at[p, "top5_jaccard_recommended_mean"]) for p, lab in pert], colors=SHAP_COLORS)
    for a in ax[:2]:
        a.set_ylim(0, 1.1)
    bars(ax[2], "Faithfulness: top vs random (1 = random)", [(f"Top {k}", fr.get(k)) for k in (1, 5, 10)],
         hline=(1, ""), colors=SHAP_COLORS)
    headroom(ax[2])
    for a in ax:
        a.title.set_fontsize(11)
        a.tick_params(labelsize=10)
        for t in a.texts:
            t.set_fontsize(9)
    save(fig, out_dir / "05b_shap.png", bottom=0.18, caption=CAPTIONS["shap"], wspace=0.22)

    # 6. Privacy -------------------------------------------------------------------------------
    plot_privacy(args, src, out_dir, synthetic, prices)

    # 7. Statistical uncertainty ---------------------------------------------------------------
    plot_uncertainty(src, out_dir, prices)

    # 8. Monitoring ----------------------------------------------------------------------------
    psi = pd.read_csv(tw / ("synthetic_monitoring.csv" if synthetic else "monitoring_psi.csv"))
    fig, ax = figure(1, "Monitoring: population stability index (PSI)",
                     f"Mean over {len(psi)} time points, {prices}. Review triggered at "
                     f"{int(psi['review_triggered'].sum())} of {len(psi)} time points.", size=(6, 3.9))
    bars(ax[0], "PSI against the training data", [("Model score", psi["psi_score"].mean()),
                                                  ("Highest of the 5 most important features",
                                                   psi["psi_top5_importance_max"].mean())],
         colors=[MODEL, REF1], hline=(0.25, "review > 0.25"), width=0.5)
    ax[0].axhline(0.1, color=GRID, linewidth=1, linestyle=":")
    headroom(ax[0])
    save(fig, out_dir / "08_monitoring.png", bottom=0.27, caption=CAPTIONS["monitoring"])


def write_table(args, out_dir, synthetic=False):
    """One row per reported metric, same values as the figures.
    synthetic=False: real prices, written to reported_metrics_table_real.csv / .md (no privacy rows).
    synthetic=True: counterfactual price series (synthetic_eval.py) plus the tests on the 21-day CF windows
    (cf_tests_all_dates.py), written to reported_metrics_table.csv / .md."""
    src = SyntheticSources(args) if synthetic else Sources(args)
    tw = src.tw
    m, s = src.mean61, src.series
    n = src.n61()
    data_note = ", counterfactual series" if synthetic else ""
    T61, T2 = f"mean of {n} time points{data_note}", "mean of 2 dates (counterfactuals)"
    rows = []

    def add(prop, metric, value, pct=False, ref="", check="", basis=T61):
        rows.append({"Property": prop, "Metric": metric,
                     "Value": value if isinstance(value, str) else fmt(value, pct),
                     "Reference or threshold": ref, "Check": check, "Basis": basis})

    def above(v, r):
        return "pass" if v > r else "fail"

    # Accuracy
    add("Accuracy", "R² of the 6-month return prediction (test rows)", m("test_r2"), ref="0 = predicting the average")
    add("Accuracy", "RMSE (test rows)", m("test_rmse"), pct=True, ref="percentage points of return")
    add("Accuracy", "MAE (test rows)", m("test_mae"), pct=True, ref="percentage points of return")
    add("Accuracy", "nDCG@10", m("ndcg@10"), ref=f"random {fmt(m('ndcg_random@10'), False)}",
        check=above(m("ndcg@10"), m("ndcg_random@10")))
    add("Accuracy", "Monthly ROI@10", m("monthly_roi@10"), pct=True,
        ref=f"risk-matched market {fmt(m('market_matched_monthly_roi'), True)}; market {fmt(m('market_monthly_roi'), True)}; "
            f"customer average {fmt(m('customer_avg_monthly_roi'), True)}",
        check=above(m("monthly_roi@10"), m("market_matched_monthly_roi")))
    add("Accuracy", "SR@10 (Sharpe ratio)", m("sharpe@10"),
        ref=f"risk-matched market {fmt(m('market_matched_sharpe'), False)}; market {fmt(m('market_sharpe'), False)}",
        check=above(m("sharpe@10"), m("market_matched_sharpe")))
    add("Accuracy", "Vol@10 (annualised volatility)", m("volatility@10"), pct=True)
    add("Accuracy", "%prof@10", m("pct_prof@10"), pct=True)

    # Suitability
    worst = s["svr_filtered@10"].max()
    # "A pass that holds only under one classification choice is reported as conditional."
    sens_worst = max(s["svr_filtered_window1y@10"].max(), s["svr_filtered_stricter@10"].max())
    svr_check = "fail" if worst > 0 else ("conditional pass" if sens_worst > 0 else "pass")
    add("Suitability", "SVR@10, filtered list", m("svr_filtered@10"), pct=True,
        ref=f"must be 0 at every time point (max {fmt(worst, True)}); conditional if it fails under the "
            f"sensitivity classifications", check=svr_check)
    add("Suitability", "SVR@10, filtered, 1-year volatility window", m("svr_filtered_window1y@10"), pct=True,
        ref="sensitivity check, not binding: above 0 makes the pass conditional")
    add("Suitability", "SVR@10, filtered, limits one class stricter", m("svr_filtered_stricter@10"), pct=True,
        ref="sensitivity check, not binding: above 0 makes the pass conditional")
    add("Suitability", "SVR@10, unfiltered ranking", m("svr@10"), pct=True)
    add("Suitability", "Customers with an unsuitable asset (unfiltered)", m("svr_customers@10"), pct=True)
    add("Suitability", "Assets that changed class since the previous time point", m("risk_class_change_share"),
        pct=True)
    add("Suitability", "Assets within 10% of a class boundary", m("risk_class_near_boundary_share"), pct=True)

    # Fairness
    attrs = [("customerType", "segment"), ("investmentCapacity", "capacity band"), ("riskLevel", "risk profile")]
    for metric, name, pct in [("ndcg", "nDCG@10", False), ("svr", "SVR@10 (unfiltered)", True)]:
        for a, lab in attrs:
            p = s.get(f"fair_gap_{metric}@10_{a}_perm_p")
            sig = f"significant at {(p.dropna() < 0.05).mean():.0%} of time points" if p is not None else ""
            add("Fairness", f"{name} gap by {lab}", m(f"fair_gap_{metric}@10_{a}"), pct=pct, ref=f"0 = parity; {sig}")
    for a, lab in attrs[:2]:
        add("Fairness", f"Monthly ROI@10 gap within risk profile, by {lab}",
            m(f"fair_strat_gap_monthly_roi@10_{a}"), pct=True, ref="0 = parity")
    add("Fairness", "Gini of exposure (top-10)", m("gini@10"), ref="0 = even, 1 = concentrated")
    add("Fairness", "Catalogue coverage (top-10)", m("coverage@10"), pct=True)
    add("Fairness", "In-house share / eligible share", m("inhouse_ratio@10"), ref="1 = proportional")
    add("Fairness", "%prof@10, in-house products", m("pct_prof_inhouse@10"), pct=True)
    add("Fairness", "%prof@10, third-party products", m("pct_prof_thirdparty@10"), pct=True)

    # Robustness
    t = s.index.to_series()
    stress = (t < STRESS[1]) & ((t + HORIZON) > STRESS[0])
    for key, name, pct in [("ndcg@10", "nDCG@10", False), ("monthly_roi@10", "Monthly ROI@10", True),
                           ("sharpe@10", "SR@10", False)]:
        calm, stressed = s[key][~stress].mean(), s[key][stress].mean()
        add("Robustness", f"{name}, calm vs stress",
            f"{fmt(calm, pct)} vs {fmt(stressed, pct)} (drop {(calm - stressed) / calm:.0%})",
            ref=f"worst time point {fmt(s[key].min(), pct)}")
    shocks = (pd.read_csv(tw / "synthetic_shocks.csv") if synthetic
              else read_all_dates(args, "robustness_market_shocks.csv", src.n61()))
    n_shock = shocks["date"].nunique()
    by, mx = shocks.groupby("scenario").mean(numeric_only=True), shocks.groupby("scenario").max(numeric_only=True)
    for sc in ["equity", "rates", "combined"]:
        b = f"mean of {n_shock} time points{data_note}"
        add("Robustness", f"J@10 before vs after, {sc} shock", by.at[sc, "j@10_filtered_mean"], ref="1 = unchanged",
            basis=b)
        w = mx.at[sc, "svr@10_shocked_filtered"]
        add("Robustness", f"SVR@10 under {sc} shock, filtered", by.at[sc, "svr@10_shocked_filtered"], pct=True,
            ref=f"must be 0 (max {fmt(w, True)})", check="pass" if w == 0 else "fail", basis=b)
        add("Robustness", f"Assets moved to a higher risk class, {sc} shock", by.at[sc, "reclassified_up_share"],
            pct=True, basis=b)

    # Transparency
    q, n_cf = cf_quality_mean(args, n)
    T2 = f"mean of {n_cf} time points" if n_cf > 2 else T2
    add("Transparency", "Counterfactual validity, re-checked end to end", q["validity_rechecked_end_to_end"], pct=True,
        basis=T2)
    add("Transparency", "Counterfactual proximity (median relative price change)", q["mean_rel_change_median"],
        pct=True, basis=T2)
    add("Transparency", "Counterfactual sparsity (share of the 21 prices changed)", q["sparsity_share_changed_mean"],
        pct=True, ref="below 50% = actionable", basis=T2)
    add("Transparency", "Counterfactual plausibility", q["plausibility_cf"], pct=True, basis=T2)
    if synthetic:
        st, fa = synthetic_shap(args)
    else:
        st = read_all_dates(args, "attribution_stability.csv", src.n61())
        fa = read_all_dates(args, "attribution_faithfulness.csv", src.n61())
    n_attr = st["date"].nunique() if st is not None else 0
    pending = synthetic and n_attr < n
    shap_basis = (f"PENDING: synthetic SHAP run in progress ({n_attr} of {n} time points done)" if pending
                  else f"mean of {n_attr} time points{data_note}")
    labels = [("noise_0.001", "noise 0.1%"), ("noise_0.0025", "noise 0.25%"), ("missing_0.02", "2% days dropped")]
    if pending:
        for _, lab in labels:
            add("Transparency", f"SHAP stability, Spearman ({lab})", "pending", ref="1 = identical", basis=shap_basis)
            add("Transparency", f"SHAP stability, top-5 overlap ({lab})", "pending", ref="1 = identical",
                basis=shap_basis)
        for k in (1, 5, 10):
            add("Transparency", f"SHAP faithfulness, top {k} vs random", "pending", ref="above 1 = faithful",
                basis=shap_basis)
    else:
        sp = st[st["perturbation"].str.startswith(("noise", "missing"))].groupby("perturbation").mean(numeric_only=True)
        for p, lab in labels:
            add("Transparency", f"SHAP stability, Spearman ({lab})", sp.at[p, "spearman_recommended_mean"],
                ref="1 = identical", basis=shap_basis)
            add("Transparency", f"SHAP stability, top-5 overlap ({lab})", sp.at[p, "top5_jaccard_recommended_mean"],
                ref="1 = identical", basis=shap_basis)
        fr = fa[fa["subset"] == "recommended"].groupby("n_features")["ratio_top_to_random"].mean()
        for k in (1, 5, 10):
            add("Transparency", f"SHAP faithfulness, top {k} vs random", fr.get(k), ref="above 1 = faithful",
                basis=shap_basis)

    if synthetic:
        cf_test_rows(args, add)

    # Monitoring
    psi = pd.read_csv(tw / ("synthetic_monitoring.csv" if synthetic else "monitoring_psi.csv"))
    trig = int(psi["review_triggered"].sum())
    add("Monitoring", "PSI of the model score", psi["psi_score"].mean(), ref="review above 0.25",
        basis=f"mean of {len(psi)} time points{data_note}")
    add("Monitoring", "Highest PSI among the 5 most important features", psi["psi_top5_importance_max"].mean(),
        ref=f"review above 0.25; triggered at {trig} of {len(psi)} time points",
        basis=f"mean of {len(psi)} time points{data_note}")

    if synthetic:
        # Privacy: holdings inferred from the recommendation lists built on the counterfactual series.
        for k, lab in [("10", "top-10"), ("full", "full list")]:
            add("Privacy", f"Holdings recovered, {lab}", m(f"holdings_recall@{k}"), pct=True)
            add("Privacy", f"Precision of the inferred holdings, {lab}", m(f"holdings_precision@{k}"), pct=True)
            add("Privacy", f"Customers with at least one holding exposed, {lab}", m(f"customers_exposed@{k}"),
                pct=True)
        order = ["Accuracy", "Suitability", "Fairness", "Robustness", "Transparency", "Privacy", "Monitoring"]
        rows.sort(key=lambda r: order.index(r["Property"]))

    table = pd.DataFrame(rows)
    name = "reported_metrics_table" if synthetic else "reported_metrics_table_real"
    table.to_csv(out_dir / f"{name}.csv", index=False)
    md = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in table.itertuples(index=False)]
    if synthetic:
        md = ["Data: counterfactual price series (cf_timeseries) in place of the real prices over each window's "
              "test period; asset-days without a counterfactual keep the real price. Customer profiles and "
              "transactions are those of FAR-Trans. Rows on the 21-day counterfactual windows say so in Basis. "
              "Real-data values: reported_metrics_table_real.md.", ""] + md
    (out_dir / f"{name}.md").write_text("\n".join(md) + "\n")
    print(f"saved {name}.csv / .md ({len(table)} rows)")


def cf_test_rows(args, add):
    """Rows of the tests on the 21-day counterfactual windows (cf_tests_all_dates.py)."""
    d = OUT / args.model / "all_dates"
    stab = pd.read_csv(d / "cf_test_stability.csv")
    n = stab["date"].nunique()
    b = f"mean of {n} time points, 21-day counterfactual windows"
    f = stab[stab["list"] == "filtered"].mean(numeric_only=True)
    add("Robustness", "J@10, counterfactual scores vs model scores (filtered list)", f["j@10_mean"],
        ref=f"1 = unchanged; lists unchanged for {fmt(f['share_customers_unchanged'], True)} of customers", basis=b)
    add("Robustness", "Kendall tau, counterfactual scores vs model scores", f["kendall_tau_scores"], ref="1 = same order",
        basis=b)
    fair = pd.read_csv(d / "cf_test_fairness.csv")
    names = {"explained_share": ("Share of the top-10 explained", True),
             "proximity": ("Proximity of the explanations", True),
             "sparsity": ("Sparsity of the explanations", True)}
    labs = {"customerType": "segment", "investmentCapacity": "capacity band", "riskLevel": "risk profile"}
    for metric, (name, pct) in names.items():
        for attr, lab in labs.items():
            g = fair[(fair["metric"] == metric) & (fair["attribute"] == attr)]
            add("Fairness", f"{name}: gap by {lab}", g["gap"].mean(), pct=pct,
                ref=f"0 = parity; significant at {(g['perm_p'] < 0.05).mean():.0%} of time points", basis=b)
    priv = pd.read_csv(d / "cf_test_privacy.csv").mean(numeric_only=True)
    add("Privacy", "Counterfactual windows within 0.1% of a past price window of the asset",
        priv["cf_within_0.1pct_of_past_window"], pct=True,
        ref=f"real windows: {fmt(priv['real_within_0.1pct_of_past_window'], True)}", basis=b)
    add("Privacy", "Counterfactual windows that copy a past price window exactly", priv["cf_exact_copy_of_past_window"],
        pct=True, basis=b)
    su = pd.read_csv(d / "cf_test_suitability.csv").mean(numeric_only=True)
    add("Suitability", "Counterfactuals that move the asset to a higher risk class", su["cf_class_up_share"], pct=True,
        basis=b)
    add("Suitability", "Explained assets whose counterfactual class is above the customer's limit",
        su["explained_pairs_above_customer_limit"], pct=True, basis=b)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="rfr_n-193_leaf-96_depth-30_kpi-full_short_internal_kpis")
    parser.add_argument("--results-run", default="rfr_tuned_full_short_internal_kpis")
    args = parser.parse_args()

    out_dir = OUT / args.model / "reported metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Actual prices: figures in "real/", table reported_metrics_table_real.md.
    plot_all(args, out_dir, out_dir / "real")
    write_table(args, out_dir)
    # Counterfactual price series: the main figures and reported_metrics_table.md.
    syn = OUT / args.model / "synthetic" / "synthetic_metrics.csv"
    if syn.exists() and pd.read_csv(syn)["date"].nunique() >= Sources(args).n61():
        st, _ = synthetic_shap(args)
        if st is not None and st["date"].nunique() >= Sources(args).n61():
            plot_all(args, out_dir, out_dir, synthetic=True)
        else:
            print("synthetic SHAP incomplete: figures on the counterfactual series not drawn")
        write_table(args, out_dir, synthetic=True)
    else:
        print("synthetic results incomplete: reported_metrics_table.md not written")


if __name__ == "__main__":
    main()
