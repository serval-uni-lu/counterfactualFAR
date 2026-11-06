"""
Per-window metrics of the "Finance metrics" protocol that are not plain ranking metrics:

  accuracy    monthly ROI@k as the document defines it, (1 + ROI@k)^(30/d) - 1; %prof@k; the market
              and customer-average references; the risk-matched market reference (per profile, over
              the assets eligible for it) for monthly ROI and SR; random and popularity nDCG@k.
  fairness    group gap and ratio of nDCG@k and SVR@k by segment, capacity and risk profile; the
              within-profile ROI gap (weights = share of customers per profile); equal-opportunity
              (TPR) and false-positive-rate gaps of the classification view; exposure concentration
              (Gini, catalogue coverage); in-house products.

  uncertainty customer bootstrap at the fairness cutoff (B = 1,000): 95% intervals of nDCG@k (model,
              filtered list, random, popularity), of the paired differences model - baseline, and of
              SVR@k (keys ending in _ci_low / _ci_high); for each nDCG and SVR group gap, a bootstrap
              interval and a permutation p-value (_perm_p). The interval covers which customers were
              sampled at this time point, not the market, which only varies over time points.

Everything is computed in one pass by FinanceProtocol and written as reference values (one line per
key in <run>_<date>_metrics.csv). Group-level detail goes to <run>_<date>_fairness_groups.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from utils.constants import DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_USER_COL

from metrics.metric import Metric
from metrics.sharpe import sharpe_ratio

ATTRIBUTES = ["customerType", "investmentCapacity", "riskLevel"]
MIN_GROUP_SIZE = 10          # reporting rule of the protocol
BOOT_B = 1000                # customer bootstrap resamples (and permutations) per time point
INHOUSE_FILE = Path(__file__).with_name("inhouse_assets.csv")


def customer_bootstrap(frame, b=BOOT_B, seed=0):
    """95% percentile interval of the mean of each column, resampling customers (rows) with
    replacement. Every column uses the same resamples, so a column holding the difference between
    two lists gives a paired interval. Returns {column: (low, high)}."""
    x = frame.to_numpy(dtype=float)
    if len(x) < 2:
        return {c: (np.nan, np.nan) for c in frame.columns}
    idx = np.random.default_rng(seed).integers(0, len(x), size=(b, len(x)))
    out = {}
    for j, col in enumerate(frame.columns):
        means = np.nanmean(x[:, j][idx], axis=1)
        out[col] = (float(np.nanpercentile(means, 2.5)), float(np.nanpercentile(means, 97.5)))
    return out


def _gap_of_codes(codes, x, n_groups):
    counts = np.bincount(codes, minlength=n_groups)
    sums = np.bincount(codes, weights=x, minlength=n_groups)
    means = sums[counts > 0] / counts[counts > 0]
    return means.max() - means.min() if len(means) >= 2 else np.nan


def group_gap_uncertainty(values, groups, b=BOOT_B, seed=0):
    """Customer-bootstrap 95% interval of the group gap (max - min of the group means) and the
    permutation p-value of "no difference between groups" (group labels shuffled b times).
    A gap is never negative, so its interval never contains 0: whether a gap is larger than chance
    is read from the p-value, not from the interval."""
    x = np.asarray(values, dtype=float)
    codes, uniques = pd.factorize(pd.Series(groups))
    n, k = len(x), len(uniques)
    if n < 2 or k < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    observed = _gap_of_codes(codes, x, k)
    boots = np.array([_gap_of_codes(codes[i], x[i], k) for i in rng.integers(0, n, size=(b, n))])
    perms = np.array([_gap_of_codes(rng.permutation(codes), x, k) for _ in range(b)])
    p = (1 + np.sum(perms >= observed - 1e-12)) / (b + 1)
    return float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5)), float(p)


def customer_attributes(customer_info_file, rec_date):
    """Segment, capacity band and risk profile valid at rec_date (latest record on or before it, or the
    first record if none is that old). "Predicted_" labels count as the plain ones."""
    info = pd.read_csv(customer_info_file, parse_dates=["timestamp"])
    for col in ["riskLevel", "investmentCapacity"]:
        info[col] = info[col].str.replace("Predicted_", "", regex=False)
    info = info.sort_values("timestamp")
    before = info[info["timestamp"] <= rec_date].drop_duplicates("customerID", keep="last")
    after = info[~info["customerID"].isin(before["customerID"])].drop_duplicates("customerID", keep="first")
    out = pd.concat([before, after]).set_index("customerID")[ATTRIBUTES]
    return out.replace("Not_Available", np.nan)


def load_inhouse(path=INHOUSE_FILE):
    return set(pd.read_csv(path)["ISIN"]) if Path(path).exists() else set()


def gini(counts):
    """Gini coefficient of non-negative counts (0 = equal exposure, close to 1 = concentrated)."""
    x = np.sort(np.asarray(counts, dtype=float))
    n, total = len(x), x.sum()
    if n == 0 or total == 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * total))


def gap_ratio(means):
    """Group gap (max - min) and ratio (max / min, only when every group mean is positive)."""
    means = means.dropna()
    if len(means) < 2:
        return np.nan, np.nan
    ratio = means.max() / means.min() if (means > 0).all() else np.nan
    return float(means.max() - means.min()), float(ratio)


def ndcg_per_customer(lists, positives, cutoffs):
    """nDCG@k exactly as PureNDCG computes it (IDCG over min(k, #positives)), for ranked lists given as
    {customer: [assets]}. Customers without positives get 0."""
    out = {k: {} for k in cutoffs}
    disc = 1.0 / np.log(np.arange(max(cutoffs)) + 2.0)
    cum = np.cumsum(disc)
    for u, items in lists.items():
        pos = positives.get(u, set())
        hits = np.array([a in pos for a in items[:max(cutoffs)]], dtype=float)
        for k in cutoffs:
            if not pos:
                out[k][u] = 0.0
                continue
            idcg = cum[min(k, len(pos)) - 1]
            out[k][u] = float((hits[:k] * disc[:len(hits[:k])]).sum() / idcg)
    return out


class PortfolioMonthlyROI(Metric):
    """Monthly ROI@k as the protocol defines it: (1 + ROI@k(u))^(30/d) - 1, where ROI@k(u) is the mean
    raw return of the customer's top-k assets (priced assets only). Averaged over customers.
    (monthly_prof@k in the pipeline instead makes each asset monthly first, then averages.)"""

    def __init__(self, data, profitability, days):
        super().__init__(data)
        self.prof = profitability
        self.days = days

    def evaluate_cutoffs(self, recs, cutoffs, target_custs, only_test_customers):
        customers = self.data.users & set(self.data.test[DEFAULT_USER_COL].unique().flatten()) \
            if only_test_customers else self.data.users
        top = recs[recs[DEFAULT_USER_COL].isin(customers & target_custs)].groupby(DEFAULT_USER_COL).head(max(cutoffs))
        top = top[top[DEFAULT_ITEM_COL].isin(list(self.prof))].copy()
        top["ret"] = top[DEFAULT_ITEM_COL].map(self.prof)
        top["rank"] = top.groupby(DEFAULT_USER_COL).cumcount() + 1
        res = {}
        for k in cutoffs:
            raw = top[top["rank"] <= k].groupby(DEFAULT_USER_COL)["ret"].mean()
            per = (1.0 + raw) ** (30.0 / self.days) - 1.0
            res[k] = (per.rename("metric").reset_index(), float(per.mean()) if len(per) else np.nan)
        return res

    def evaluate_indiv(self, customer_df, cutoff):
        raise NotImplementedError("use evaluate_cutoffs")

    def evaluate_indiv_cutoffs(self, customer_df, cutoffs):
        raise NotImplementedError("use evaluate_cutoffs")


class FinanceProtocol:
    """See the module docstring. Behaves like a metric whose evaluate_cutoffs stores its results and
    returns no per-cutoff entries; recommendation.test() and compute_risk_metrics.py write them out
    through reference_values() and extra_tables()."""

    def __init__(self, data, rec_date, eval_date, profitability, rel, rf_daily=0.0, suit=None,
                 attrs=None, inhouse=None, seed=0, fairness_cutoff=10):
        """
        :param profitability: raw return per priced asset between rec_date and eval_date.
        :param rel: daily prices relative to rec_date (metrics.sharpe.relative_prices).
        :param suit: a metrics.suitability.Suitability (None: suitability-based parts are skipped).
        :param attrs: customer_attributes() frame (None: fairness is skipped).
        """
        self.data = data
        self.prof = profitability
        self.rel = rel
        self.rf_daily = rf_daily
        self.days = (eval_date - rec_date).days
        self.suit = suit
        self.attrs = attrs
        self.inhouse = inhouse or set()
        self.seed = seed
        self.kf = fairness_cutoff
        self.values, self.groups = {}, None
        self._sr_cache = {}

    # ---------------------------------------------------------------- helpers
    def monthly(self, raw):
        return (1.0 + raw) ** (30.0 / self.days) - 1.0

    def roi(self, items):
        r = [self.prof[a] for a in items if a in self.prof]
        return float(np.mean(r)) if r else np.nan

    def sr(self, items):
        key = tuple(sorted(a for a in items if a in self.rel.columns))
        if not key:
            return np.nan
        if key not in self._sr_cache:
            self._sr_cache[key] = sharpe_ratio(self.rel[list(key)].mean(axis=1), self.rf_daily)
        return self._sr_cache[key]

    # ---------------------------------------------------------------- main
    def evaluate_cutoffs(self, recs, cutoffs, target_custs, only_test_customers):
        data = self.data
        customers = data.users & set(data.test[DEFAULT_USER_COL].unique().flatten()) \
            if only_test_customers else data.users
        customers = customers & target_custs
        cutoffs = sorted(cutoffs)
        top = recs[recs[DEFAULT_USER_COL].isin(customers)].groupby(DEFAULT_USER_COL).head(max(cutoffs))
        lists = top.groupby(DEFAULT_USER_COL, sort=False)[DEFAULT_ITEM_COL].agg(list).to_dict()
        positives = {u: set(data.get_positive_assets(u)) for u in customers}
        held = data.train[data.train[DEFAULT_USER_COL].isin(customers)].groupby(DEFAULT_USER_COL)[
            DEFAULT_ITEM_COL].agg(set).to_dict()
        universe = sorted(data.assets)
        v = {}

        # --- accuracy: %prof@k (monthly ROI per customer kept for the references and fairness) -----
        per_cust_roi = {}
        for k in cutoffs:
            rois, profs = {}, []
            for u, items in lists.items():
                r = [self.prof[a] for a in items[:k] if a in self.prof]
                if r:
                    rois[u] = self.monthly(np.mean(r))
                    profs.append(np.mean([x >= 0 for x in r]))
            v[f"pct_prof@{k}"] = float(np.mean(profs)) if profs else np.nan
            per_cust_roi[k] = rois

        # --- references -------------------------------------------------------------------------
        priced = [a for a in universe if a in self.prof]
        v["market_monthly_roi"] = self.monthly(np.mean([self.prof[a] for a in priced])) if priced else np.nan
        cust_avg = [self.monthly(self.roi(positives[u])) for u in lists if not np.isnan(self.roi(positives[u]))]
        v["customer_avg_monthly_roi"] = float(np.mean(cust_avg)) if cust_avg else np.nan

        if self.suit is not None:
            ref_roi, ref_sr = {}, {}
            for p, cap in self.suit.limits.items():
                elig = [a for a in priced if self.suit.asset_class.get(a, 99) <= cap]
                ref_roi[p] = self.monthly(np.mean([self.prof[a] for a in elig])) if elig else np.nan
                ref_sr[p] = self.sr(elig) if elig else np.nan
                v[f"market_monthly_roi_{p}"] = ref_roi[p]
                v[f"market_sharpe_{p}"] = ref_sr[p]
            prof_of = self.suit.profile
            with_p = [u for u in lists if u in prof_of]
            k = self.kf
            if with_p:
                m_roi = np.array([ref_roi[prof_of[u]] for u in with_p])
                m_sr = np.array([ref_sr[prof_of[u]] for u in with_p])
                own_roi = np.array([per_cust_roi[k].get(u, np.nan) for u in with_p])
                own_sr = np.array([self.sr(lists[u][:k]) for u in with_p])
                v["market_matched_monthly_roi"] = float(np.nanmean(m_roi))
                v["market_matched_sharpe"] = float(np.nanmean(m_sr))
                v[f"monthly_roi@{k}_minus_matched"] = float(np.nanmean(own_roi - m_roi))
                v[f"sharpe@{k}_minus_matched"] = float(np.nanmean(own_sr - m_sr))
                # The same comparison for the suitability-filtered lists (what a compliant system shows).
                flists = self.suit.filter(top).groupby(DEFAULT_USER_COL, sort=False)[DEFAULT_ITEM_COL].agg(list)
                f_roi = np.array([self.monthly(self.roi(flists.get(u, [])[:k])) for u in with_p])
                f_sr = np.array([self.sr(flists.get(u, [])[:k]) for u in with_p])
                v[f"monthly_roi_filtered@{k}_minus_matched"] = float(np.nanmean(f_roi - m_roi))
                v[f"sharpe_filtered@{k}_minus_matched"] = float(np.nanmean(f_sr - m_sr))

        # --- baselines: random and popularity nDCG ------------------------------------------------
        rng = np.random.default_rng(self.seed)
        pop = data.train[data.train[DEFAULT_RATING_COL] > 0].groupby(DEFAULT_ITEM_COL).size()
        pop_order = sorted(universe, key=lambda a: (-pop.get(a, 0), a))
        rand_lists, pop_lists = {}, {}
        for u in customers:
            h = held.get(u, set())
            cand = [a for a in universe if a not in h]
            rand_lists[u] = list(rng.permutation(cand)[:max(cutoffs)])
            pop_lists[u] = [a for a in pop_order if a not in h][:max(cutoffs)]
        base_nd = {}
        for name, ls in (("random", rand_lists), ("popularity", pop_lists)):
            nd = ndcg_per_customer(ls, positives, cutoffs)
            base_nd[name] = nd
            for k in cutoffs:
                v[f"ndcg_{name}@{k}"] = float(np.mean(list(nd[k].values()))) if nd[k] else np.nan

        # --- customer bootstrap at the fairness cutoff ------------------------------------------
        if self.kf in cutoffs:
            v.update(self._customer_intervals(customers, lists, positives, top, base_nd))

        # --- exposure concentration ---------------------------------------------------------------
        for k in cutoffs:
            counts = pd.Series([a for items in lists.values() for a in items[:k]]).value_counts()
            full = counts.reindex(universe, fill_value=0)
            v[f"gini@{k}"] = gini(full.values)
            v[f"coverage@{k}"] = float((full > 0).mean()) if len(full) else np.nan

        # --- in-house products -------------------------------------------------------------------
        if self.inhouse:
            k = self.kf
            share_top, share_elig, prof_in, prof_out, sr_in, sr_out = [], [], [], [], [], []
            for u, items in lists.items():
                items = items[:k]
                if not items:
                    continue
                share_top.append(np.mean([a in self.inhouse for a in items]))
                cand = [a for a in universe if a not in held.get(u, set())]
                elig = self.suit.eligible(cand, u) if self.suit is not None else cand
                if elig:
                    share_elig.append(np.mean([a in self.inhouse for a in elig]))
                ins = [a for a in items if a in self.inhouse]
                outs = [a for a in items if a not in self.inhouse]
                for grp, p_list, s_list in ((ins, prof_in, sr_in), (outs, prof_out, sr_out)):
                    r = [self.prof[a] for a in grp if a in self.prof]
                    if r:
                        p_list.append(np.mean([x >= 0 for x in r]))
                        s_list.append(self.sr(grp))
            v[f"inhouse_share@{k}"] = float(np.mean(share_top)) if share_top else np.nan
            v["inhouse_eligible_share"] = float(np.mean(share_elig)) if share_elig else np.nan
            v[f"inhouse_ratio@{k}"] = v[f"inhouse_share@{k}"] / v["inhouse_eligible_share"] \
                if share_elig and v["inhouse_eligible_share"] > 0 else np.nan
            v[f"pct_prof_inhouse@{k}"] = float(np.mean(prof_in)) if prof_in else np.nan
            v[f"pct_prof_thirdparty@{k}"] = float(np.mean(prof_out)) if prof_out else np.nan
            v[f"sharpe_inhouse@{k}"] = float(np.nanmean(sr_in)) if sr_in else np.nan
            v[f"sharpe_thirdparty@{k}"] = float(np.nanmean(sr_out)) if sr_out else np.nan

        # --- fairness at the fairness cutoff -------------------------------------------------------
        if self.attrs is not None:
            self._fairness(lists, positives, held, universe, per_cust_roi, v)

        self.values = v
        return {}

    def _customer_intervals(self, customers, lists, positives, top, base_nd):
        """95% customer-bootstrap intervals at the fairness cutoff. Customers with an empty list count
        as nDCG 0, as in PureNDCG. All columns share the resamples, so model - baseline is paired."""
        k = self.kf
        custs = sorted(customers)
        cols = {"ndcg": ndcg_per_customer({u: lists.get(u, []) for u in custs}, positives, [k])[k]}
        if self.suit is not None:
            flists = self.suit.filter(top).groupby(DEFAULT_USER_COL, sort=False)[DEFAULT_ITEM_COL].agg(list)
            cols["ndcg_filtered"] = ndcg_per_customer({u: flists.get(u, []) for u in custs}, positives, [k])[k]
        for name, nd in base_nd.items():
            cols[f"ndcg_{name}"] = nd[k]
        frame = pd.DataFrame({c: [vals.get(u, 0.0) for u in custs] for c, vals in cols.items()}, index=custs)
        for model_col in [c for c in ("ndcg", "ndcg_filtered") if c in frame]:
            for name in base_nd:
                frame[f"{model_col}_minus_{name}"] = frame[model_col] - frame[f"ndcg_{name}"]
        if self.suit is not None:
            cap = self.suit.max_class
            frame["svr"] = [float(np.mean([self.suit.asset_class.get(a, 0) > cap[u] for a in lists[u][:k]]))
                            if u in cap and lists.get(u) else np.nan for u in custs]
        out = {}
        for col, (lo, hi) in customer_bootstrap(frame, seed=self.seed).items():
            out[f"{col}@{k}_ci_low"], out[f"{col}@{k}_ci_high"] = lo, hi
        return out

    def _fairness(self, lists, positives, held, universe, per_cust_roi, v):
        k = self.kf
        nd = ndcg_per_customer(lists, positives, [k])[k]
        rows = []
        for u, items in lists.items():
            items = items[:k]
            cand_n = len(universe) - len(held.get(u, set()))
            pos = positives.get(u, set())
            pos_in_cand = len(pos - held.get(u, set()))
            hits = sum(a in pos for a in items)
            svr = np.nan
            if self.suit is not None and u in self.suit.max_class:
                cap = self.suit.max_class[u]
                svr = float(np.mean([self.suit.asset_class.get(a, 0) > cap for a in items])) if items else np.nan
            rows.append({DEFAULT_USER_COL: u, "ndcg": nd.get(u, 0.0), "monthly_roi": per_cust_roi[k].get(u, np.nan),
                         "svr": svr, "tp": hits, "pos": pos_in_cand, "fp": len(items) - hits,
                         "neg": cand_n - pos_in_cand})
        frame = pd.DataFrame(rows).set_index(DEFAULT_USER_COL).join(self.attrs, how="left")

        groups = []
        for attr in ATTRIBUTES:
            sub = frame[frame[attr].notna()]
            sizes = sub.groupby(attr).size()
            kept = sizes[sizes >= MIN_GROUP_SIZE].index
            g = sub[sub[attr].isin(kept)].groupby(attr)
            # nDCG and SVR are compared across all groups.
            for metric in ["ndcg", "svr"]:
                means = g[metric].mean()
                gap, ratio = gap_ratio(means)
                v[f"fair_gap_{metric}@{k}_{attr}"] = gap
                v[f"fair_ratio_{metric}@{k}_{attr}"] = ratio
                rows_m = sub[sub[attr].isin(kept) & sub[metric].notna()]
                lo, hi, p = group_gap_uncertainty(rows_m[metric].values, rows_m[attr].values, seed=self.seed)
                v[f"fair_gap_{metric}@{k}_{attr}_ci_low"] = lo
                v[f"fair_gap_{metric}@{k}_{attr}_ci_high"] = hi
                v[f"fair_gap_{metric}@{k}_{attr}_perm_p"] = p
                groups += [{"attribute": attr, "metric": metric, "group": grp, "n_customers": int(sizes[grp]),
                            "value": val} for grp, val in means.items()]
            # Classification view: pooled TPR and FPR per group.
            tpr = g["tp"].sum() / g["pos"].sum().replace(0, np.nan)
            fpr = g["fp"].sum() / g["neg"].sum().replace(0, np.nan)
            for metric, means in (("tpr", tpr), ("fpr", fpr)):
                gap, ratio = gap_ratio(means)
                v[f"fair_gap_{metric}@{k}_{attr}"] = gap
                v[f"fair_ratio_{metric}@{k}_{attr}"] = ratio
                groups += [{"attribute": attr, "metric": metric, "group": grp, "n_customers": int(sizes[grp]),
                            "value": val} for grp, val in means.items()]

        # Profitability only within the same risk profile, averaged with w_r = share of customers in r.
        for attr in ["customerType", "investmentCapacity"]:
            gaps, weights = [], []
            sub = frame[frame["riskLevel"].notna() & frame[attr].notna() & frame["monthly_roi"].notna()]
            for r, rows_r in sub.groupby("riskLevel"):
                sizes = rows_r.groupby(attr).size()
                means = rows_r[rows_r[attr].isin(sizes[sizes >= MIN_GROUP_SIZE].index)].groupby(attr)["monthly_roi"].mean()
                gap, _ = gap_ratio(means)
                groups += [{"attribute": attr, "metric": "monthly_roi", "risk_profile": r, "group": grp,
                            "n_customers": int(sizes[grp]), "value": val} for grp, val in means.items()]
                if not np.isnan(gap):
                    gaps.append(gap)
                    weights.append(len(rows_r))
            # Profiles without two groups of the minimum size are left out; weights renormalised.
            v[f"fair_strat_gap_monthly_roi@{k}_{attr}"] = float(np.average(gaps, weights=weights)) if gaps else np.nan
            v[f"fair_strat_profiles_used_{attr}"] = len(gaps)
        self.groups = pd.DataFrame(groups)

    def reference_values(self):
        return self.values

    def extra_tables(self):
        return {} if self.groups is None or self.groups.empty else {"fairness_groups": self.groups}
