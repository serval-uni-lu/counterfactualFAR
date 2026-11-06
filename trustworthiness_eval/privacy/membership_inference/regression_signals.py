"""
Shared building blocks for the regression membership-inference attacks
(loss.py, population_attack.py) run against this project's own fitted
profitability models — no shadow models.

Scope: the target models hold no customer information. They are trained on
market data only — each row is an asset on a date, described by technical
indicators computed from its price history — so a row being identified as a
member reveals nothing about a person. The attacks are still useful as a
memorisation audit: a model whose training rows can be told apart from unseen
ones fits them much more closely than new data, which is the generalisation gap
seen from the attacker's side. The results therefore belong with the model's
generalisation/robustness assessment rather than with customer privacy. One
caveat for that reading: non-members come from the later test period, so the
separation mixes memorisation with the market changing over time (evaluate.py's
temporal analysis shows how much depends on the distance from the cutoff).
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.constants import DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL


def load_target_model(pkl_path):
    """Unpickle an internal KPI-model pipeline (RFRKPIModel/LGBMKPIModel/TabPFNKPIModel)."""
    with open(pkl_path, "rb") as handle:
        model = pickle.load(handle)

    if not hasattr(model, "_generate_kpis_df"):
        raise ValueError(f"{pkl_path}: loaded object has no _generate_kpis_df (expected an internal KPI model)")
    if not hasattr(model, "model"):
        raise ValueError(f"{pkl_path}: loaded object has no fitted regressor at .model")
    if not getattr(model, "kpi_features", None):
        raise ValueError(f"{pkl_path}: loaded object has no kpi_features configured")
    return model


def _prepare_ts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(df[DEFAULT_TIMESTAMP_COL])
    return df.sort_values([DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]).reset_index(drop=True)


def compute_row_losses(raw_ts_df: pd.DataFrame, model, months: int, lookback_df: pd.DataFrame = None,
                       keep_features: bool = False) -> pd.DataFrame:
    """Regenerate KPI features + forward-shifted target from a raw time series using the
    model's own KPI pipeline, and return per-row squared error.

    `lookback_df`, if given, is concatenated in purely to supply extra context rows —
    trailing history the KPI generator needs, and/or future prices the target's
    forward shift needs (whichever `raw_ts_df` itself doesn't span far enough to
    cover) — the returned rows are still restricted to (item, timestamp) pairs that
    came from `raw_ts_df` itself. Without this, rows within the
    last `months*21` trading days of training_data.csv would be silently dropped for
    having no future price of their own to shift into.

    `keep_features` also returns the model's KPI feature columns, for attacks that have
    to score the same rows with other models (rmia.py's reference models).
    """
    raw_ts_df = _prepare_ts(raw_ts_df)

    if lookback_df is not None:
        lookback_df = _prepare_ts(lookback_df)
        combined = (
            pd.concat([lookback_df, raw_ts_df], ignore_index=True)
            .drop_duplicates(subset=[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL])
            .sort_values([DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL])
            .reset_index(drop=True)
        )
        target_keys = raw_ts_df[[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]].drop_duplicates()
    else:
        combined = raw_ts_df
        target_keys = None

    kpis_df = model._generate_kpis_df(combined)
    kpis_df = kpis_df.sort_values([DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]).reset_index(drop=True)

    kpis_df["final_price"] = (
        kpis_df.groupby(DEFAULT_ITEM_COL, sort=False)[DEFAULT_RATING_COL].shift(-months * 21)
    )
    kpis_df["target"] = (kpis_df["final_price"] - kpis_df[DEFAULT_RATING_COL]) / kpis_df[DEFAULT_RATING_COL]
    kpis_df = kpis_df[kpis_df[DEFAULT_RATING_COL] > 0.0]

    feature_cols = list(model.kpi_features)
    missing = [c for c in feature_cols if c not in kpis_df.columns]
    if missing:
        raise ValueError(f"Missing KPI feature columns after generation: {missing}")

    kpis_df = kpis_df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

    if target_keys is not None:
        kpis_df = kpis_df.merge(target_keys, on=[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL], how="inner")

    if kpis_df.empty:
        columns = [DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, "prediction", "target", "loss"]
        return pd.DataFrame(columns=columns + (feature_cols if keep_features else []))

    X = kpis_df[feature_cols].astype(float)
    y_true = kpis_df["target"].to_numpy(dtype=float)
    y_pred = np.asarray(model.model.predict(X), dtype=float).reshape(-1)
    loss = (y_pred - y_true) ** 2

    scored = pd.DataFrame({
        DEFAULT_ITEM_COL: kpis_df[DEFAULT_ITEM_COL].values,
        DEFAULT_TIMESTAMP_COL: kpis_df[DEFAULT_TIMESTAMP_COL].values,
        "prediction": y_pred,
        "target": y_true,
        "loss": loss,
    })
    if keep_features:
        scored = pd.concat([scored, X.reset_index(drop=True)], axis=1)
    return scored


def stratified_split(df: pd.DataFrame, group_col: str, frac: float = 0.5, seed: int = 42):
    """Split df into two disjoint parts, ~frac/~(1-frac) within each group_col group
    independently, so both parts stay representative across groups (assets) rather than
    one part accidentally favoring different assets than the other."""
    rng = np.random.default_rng(seed)
    part_a, part_b = [], []
    for _, group in df.groupby(group_col, sort=False):
        order = rng.permutation(len(group))
        cut = int(round(len(group) * frac))
        part_a.append(group.iloc[order[:cut]])
        part_b.append(group.iloc[order[cut:]])
    a = pd.concat(part_a).reset_index(drop=True) if part_a else df.iloc[0:0]
    b = pd.concat(part_b).reset_index(drop=True) if part_b else df.iloc[0:0]
    return a, b


def load_cf_timeseries(cf_path) -> pd.DataFrame:
    """Read a cf_timeseries_*.csv (generate_counterfactuals.py) as a raw time series with
    the same columns as testing_data.csv: one counterfactual price per (item, timestamp)."""
    cf_ts = pd.read_csv(cf_path)
    # Rows appended by different runs don't all share one timestamp format
    # ("2021-05-27" vs "2022-05-23 00:00:00").
    cf_ts[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(cf_ts[DEFAULT_TIMESTAMP_COL], format="mixed")
    cf_ts = cf_ts.dropna(subset=[DEFAULT_RATING_COL])
    cf_ts = cf_ts.drop_duplicates(subset=[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL])
    return cf_ts[[DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_TIMESTAMP_COL]].reset_index(drop=True)


def build_auditing_sets(training_ts: pd.DataFrame, testing_ts: pd.DataFrame, model, months: int, split_seed: int = 42,
                        cf_ts: pd.DataFrame = None, keep_features: bool = False, return_member_pool: bool = False):
    """Build the (member_df, nonmember_df, population_df):

    - The valid test-period rows are split 50/50 stratified per asset (stratified_split):
      one half is the audited non-members, the other half is reserved purely as the
      population reference (population_attack.py).
    - member_df is downsampled uniformly at random to the same size as the non-member
      half, so both classes are balanced instead of members vastly outnumbering
      non-members.
    - With `cf_ts` (load_cf_timeseries), the test-period rows are the counterfactual
      ones instead of the real ones: the non-member and population halves are both
      drawn from them, while the members stay exactly the real-data ones. Test-period
      rows with no counterfactual keep their real price, but only as context (so the
      KPI lookback and the target's forward shift still land on the right dates) —
      they are never audited.
    - `keep_features` keeps the KPI feature columns on every returned frame, and
      `return_member_pool` appends the full (not downsampled) member pool as a fourth
      return value — both for rmia.py, whose reference models train on that pool.
    """
    testing_ts = _prepare_ts(testing_ts)
    member_pool = compute_row_losses(training_ts, model, months, lookback_df=testing_ts, keep_features=keep_features)
    if cf_ts is None:
        test_scored = compute_row_losses(testing_ts, model, months, lookback_df=training_ts, keep_features=keep_features)
    else:
        keys = [DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]
        cf_ts = _prepare_ts(cf_ts)
        has_cf = testing_ts.merge(cf_ts[keys], on=keys, how="left", indicator=True)["_merge"].eq("both").to_numpy()
        context = pd.concat([training_ts, testing_ts[~has_cf]], ignore_index=True)
        test_scored = compute_row_losses(cf_ts, model, months, lookback_df=context, keep_features=keep_features)
    if test_scored.empty:
        raise ValueError("No usable test-period rows survived KPI/target reconstruction")

    nonmember_df, population_df = stratified_split(test_scored, DEFAULT_ITEM_COL, frac=0.5, seed=split_seed)

    n = len(nonmember_df)
    if len(member_pool) > n:
        member_df = member_pool.sample(n=n, random_state=split_seed).reset_index(drop=True)
    else:
        member_df = member_pool

    if return_member_pool:
        return member_df, nonmember_df, population_df, member_pool
    return member_df, nonmember_df, population_df


def summarize_rows(df: pd.DataFrame, name: str) -> str:
    """One-line diagnostic: row count, date span, and asset count,
    needed to notice a set that's unexpectedly empty, tiny, or missing a date range
    (e.g. rows near a series' edge silently dropped for lacking KPI lookback or a
    target's forward-shift price)."""
    if df.empty:
        return f"{name}: 0 rows"
    dates = pd.to_datetime(df[DEFAULT_TIMESTAMP_COL])
    return (f"{name}: {len(df)} rows, {df[DEFAULT_ITEM_COL].nunique()} assets, "
            f"dates {dates.min().date()} to {dates.max().date()}")


# Standard low-FPR checkpoints
LOW_FPR_THRESHOLDS = (0.001, 0.01, 0.1)


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
    """Highest TPR reachable at FPR <= target_fpr (the last ROC point within the budget),
    same as ml_privacy_meter's compute_attack_results — no interpolation, which would
    credit the attack with operating points no real threshold achieves."""
    return float(tpr[np.where(fpr <= target_fpr)[0][-1]])


def attack_metrics(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> dict:
    """Threshold-independent + best-threshold MIA metrics.

    Scores must already be oriented so that a HIGHER score means "more likely member"
    (callers negate loss, or pass population_score() of the negated loss, before calling this).
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    member_scores = np.asarray(member_scores, dtype=float)
    nonmember_scores = np.asarray(nonmember_scores, dtype=float)

    scores = np.concatenate([member_scores, nonmember_scores])
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])

    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_idx = int(np.argmax(tpr - fpr))
    best_threshold = float(thresholds[best_idx])
    predictions = (scores >= best_threshold).astype(int)

    return {
        "n_members": int(len(member_scores)),
        "n_nonmembers": int(len(nonmember_scores)),
        "member_score_mean": float(np.mean(member_scores)) if len(member_scores) else None,
        "nonmember_score_mean": float(np.mean(nonmember_scores)) if len(nonmember_scores) else None,
        "roc_auc": auc,
        "best_threshold": best_threshold,
        "accuracy_at_best_threshold": float((predictions == labels).mean()),
        "attack_advantage": float(tpr[best_idx] - fpr[best_idx]),
        "tpr_at_fpr": {t: tpr_at_fpr(fpr, tpr, t) for t in LOW_FPR_THRESHOLDS},
    }


def population_score(sample_signals: np.ndarray, population_signals: np.ndarray) -> np.ndarray:
    """Population attack score (ml_privacy_meter's run_population_attack): the empirical
    CDF of each sample's signal within the population's signals, i.e. the fraction of
    population samples whose signal is <= the sample's.

    Signals must be oriented "higher = more likely member" (pass -loss), so the score is
    too: a sample that beats most of the population gets a score near 1."""
    population_sorted = np.sort(np.asarray(population_signals, dtype=float).ravel())
    sample_signals = np.asarray(sample_signals, dtype=float).ravel()
    if len(population_sorted) == 0:
        return np.full(len(sample_signals), np.nan)
    return np.searchsorted(population_sorted, sample_signals, side="right") / len(population_sorted)


def resolve_paths(model: str, date: str, artifacts_dir: str = "artifacts_for_counterfactuals"):
    """File paths for one (model tag, date) window, matching the naming convention
    written by algorithms/profitability_prediction.py / generate_counterfactuals.py."""
    model_dir = Path(artifacts_dir) / model
    suffix = f"{date}_00-00-00_{model}"
    return {
        "pkl": model_dir / f"profitability_recommendation_pipeline_{suffix}.pkl",
        "training_data": model_dir / f"training_data_{suffix}.csv",
        "testing_data": model_dir / f"testing_data_{suffix}.csv",
    }


TEST_SOURCES = ("original", "counterfactual")


def attack_dir_name(attack_name: str, test_source: str) -> str:
    """Results/plots subdirectory for an attack: the counterfactual-test-data variant
    gets its own "_cf" directory so it never overwrites the original-data run."""
    return attack_name if test_source == "original" else f"{attack_name}_cf"


def resolve_cf_timeseries(model: str, date: str, method: str = "genetic",
                          cf_results_dir: str = "counterfactuals_results") -> Path:
    """Path of the counterfactual time series generate_counterfactuals.py wrote for one
    (model tag, date) window."""
    return Path(cf_results_dir) / model / f"cf_timeseries_{model}_{date}_{method}.csv"


def available_dates(model: str, artifacts_dir: str = "artifacts_for_counterfactuals") -> list:
    """Recommendation dates with a fitted pipeline available for this model tag, sorted ascending."""
    model_dir = Path(artifacts_dir) / model
    dates = []
    for pkl_path in model_dir.glob("profitability_recommendation_pipeline_*.pkl"):
        stem = pkl_path.stem.replace("profitability_recommendation_pipeline_", "")
        date = stem.split("_00-00-00_")[0]
        dates.append(date)
    return sorted(set(dates))


def load_all_date_metrics(output_dir, model: str, attack_name: str) -> dict:
    """Load every metrics.json already on disk for this model/attack, keyed by date,
    so a pooled summary / across-dates trend plot reflects every date ever run for
    this model, not just the dates requested in the current invocation (otherwise
    running one date today and another tomorrow would silently drop the first date's
    result from auc_by_date.png, or skip the plot entirely on a single-date run)."""
    model_dir = Path(output_dir) / model
    metrics_by_date = {}
    for metrics_path in sorted(model_dir.glob(f"*/{attack_name}/metrics.json")):
        date = metrics_path.parent.parent.name
        with open(metrics_path) as handle:
            metrics = json.load(handle)
        if "tpr_at_fpr" in metrics:
            # JSON only allows string keys, so this comes back as {"0.01": ...} —
            # restore float keys to match what attack_metrics() returns in-memory.
            metrics["tpr_at_fpr"] = {float(k): v for k, v in metrics["tpr_at_fpr"].items()}
        metrics_by_date[date] = metrics
    return metrics_by_date
