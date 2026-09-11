"""
Shared building blocks for the regression membership-inference attacks
(loss.py, population_attack.py) run against this project's own fitted
profitability models — no shadow models.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
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


def compute_row_losses(raw_ts_df: pd.DataFrame, model, months: int, lookback_df: pd.DataFrame = None) -> pd.DataFrame:
    """Regenerate KPI features + forward-shifted target from a raw time series using the
    model's own KPI pipeline, and return per-row squared error.

    `lookback_df`, if given, is concatenated in purely to supply extra context rows —
    trailing history the KPI generator needs, and/or future prices the target's
    forward shift needs (whichever `raw_ts_df` itself doesn't span far enough to
    cover) — the returned rows are still restricted to (item, timestamp) pairs that
    came from `raw_ts_df` itself. Without this, rows within the
    last `months*21` trading days of training_data.csv would be silently dropped for
    having no future price of their own to shift into.
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
        return pd.DataFrame(columns=[DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL, "prediction", "target", "loss"])

    X = kpis_df[feature_cols].astype(float)
    y_true = kpis_df["target"].to_numpy(dtype=float)
    y_pred = np.asarray(model.model.predict(X), dtype=float).reshape(-1)
    loss = (y_pred - y_true) ** 2

    return pd.DataFrame({
        DEFAULT_ITEM_COL: kpis_df[DEFAULT_ITEM_COL].values,
        DEFAULT_TIMESTAMP_COL: kpis_df[DEFAULT_TIMESTAMP_COL].values,
        "prediction": y_pred,
        "target": y_true,
        "loss": loss,
    })


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


def build_auditing_sets(training_ts: pd.DataFrame, testing_ts: pd.DataFrame, model, months: int, split_seed: int = 42):
    """Build the (member_df, nonmember_df, population_df):

    - The valid test-period rows are split 50/50 stratified per asset (stratified_split):
      one half is the audited non-members, the other half is reserved purely as the
      population reference (population_attack.py).
    - member_df is downsampled uniformly at random to the same size as the non-member
      half, so both classes are balanced instead of members vastly outnumbering
      non-members.
    """
    member_pool = compute_row_losses(training_ts, model, months, lookback_df=testing_ts)
    test_scored = compute_row_losses(testing_ts, model, months, lookback_df=training_ts)
    if test_scored.empty:
        raise ValueError("No usable test-period rows survived KPI/target reconstruction")

    nonmember_df, population_df = stratified_split(test_scored, DEFAULT_ITEM_COL, frac=0.5, seed=split_seed)

    n = len(nonmember_df)
    if len(member_pool) > n:
        member_df = member_pool.sample(n=n, random_state=split_seed).reset_index(drop=True)
    else:
        member_df = member_pool

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
    """TPR at a given FPR, linearly interpolated along the ROC curve."""
    return float(np.interp(target_fpr, fpr, tpr))


def attack_metrics(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> dict:
    """Threshold-independent + best-threshold MIA metrics.

    Scores must already be oriented so that a HIGHER score means "more likely member"
    (callers negate loss, or negate a population-percentile rank, before calling this).
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


def population_percentile(sample_losses: np.ndarray, population_losses: np.ndarray) -> np.ndarray:
    """Fraction of population_losses <= each sample loss (empirical CDF rank in [0, 1])."""
    population_sorted = np.sort(np.asarray(population_losses, dtype=float))
    sample_losses = np.asarray(sample_losses, dtype=float)
    if len(population_sorted) == 0:
        return np.full(len(sample_losses), np.nan)
    ranks = np.searchsorted(population_sorted, sample_losses, side="right")
    return ranks / len(population_sorted)


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
