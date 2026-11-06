"""
Evaluation of the membership inference attacks that accounts for near-duplicate rows.

Works on the scores loss.py / population_attack.py / rmia.py already saved; nothing is
retrained or rescored. For each attack and date it adds:

  1. Block membership. Neighbouring days of one asset share almost all their KPIs and
     target, so "was this exact row in training?" is not a well-posed question. The
     claim evaluated here is "was this asset's history over this period in training?":
     time is cut into blocks of `--block-days` trading days counted from the training
     cutoff, a block is (asset, time block), and its score is the mean attack score of
     `--rows-per-block` of its audited rows (the same number for every block, so a
     block is not easier to classify just because more of its rows were audited).
     Members are the blocks before the cutoff, non-members the blocks after it.
  2. Uncertainty by resampling assets. Rows of one asset are not independent, so every
     95% interval comes from a bootstrap that redraws whole assets.
  3. A temporal analysis: how the attack's verdict on a sample depends on its distance
     from the training cutoff.

Written next to each attack's results: evaluation.json and temporal_analysis.csv; the
plots evaluation_levels.png / temporal_analysis.png go with the attack's other plots.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from trustworthiness_eval.privacy.membership_inference.regression_signals import (
    LOW_FPR_THRESHOLDS,
    TEST_SOURCES,
    attack_dir_name,
    available_dates,
    resolve_paths,
    tpr_at_fpr,
)
from trustworthiness_eval.privacy.membership_inference.plots import PLOT_ROOT, plot_evaluation_levels, plot_temporal_analysis
from utils.constants import DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL

ATTACKS = ("loss_attack", "population_attack", "rmia")


def load_scores(date_dir: Path) -> pd.DataFrame:
    """Audited rows of one attack: member + non-member score files, stacked."""
    parts = []
    for name, is_member in (("member_scores.csv", True), ("nonmember_scores.csv", False)):
        part = pd.read_csv(date_dir / name, float_precision="round_trip")
        parts.append(part.assign(is_member=is_member))
    scores = pd.concat(parts, ignore_index=True)
    scores[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(scores[DEFAULT_TIMESTAMP_COL])
    return scores


def days_from_cutoff(timestamps: pd.Series, paths: dict) -> np.ndarray:
    """Trading days between each row and the last training day (<= 0 for the training
    period, >= 1 for the test period), counted on the model's own trading calendar."""
    train_days = pd.to_datetime(pd.read_csv(paths["training_data"], usecols=[DEFAULT_TIMESTAMP_COL])[DEFAULT_TIMESTAMP_COL])
    test_days = pd.to_datetime(pd.read_csv(paths["testing_data"], usecols=[DEFAULT_TIMESTAMP_COL])[DEFAULT_TIMESTAMP_COL])
    calendar = pd.DatetimeIndex(sorted(set(train_days) | set(test_days) | set(timestamps)))
    position = pd.Series(np.arange(len(calendar)), index=calendar)
    return (position.loc[timestamps].to_numpy() - position.loc[train_days.max()]).astype(int)


def distance_bin(days: np.ndarray, width: int) -> np.ndarray:
    """Signed bin of `width` trading days: -1 is the last `width` training days, +1 the
    first `width` test days."""
    days = np.asarray(days)
    return np.where(days <= 0, -((-days) // width + 1), (days - 1) // width + 1)


def weighted_metrics(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray = None) -> dict:
    fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
    metrics = {"roc_auc": float(auc(fpr, tpr)), "attack_advantage": float(np.max(tpr - fpr))}
    for target in LOW_FPR_THRESHOLDS:
        metrics[f"tpr_at_fpr_{target:g}"] = tpr_at_fpr(fpr, tpr, target)
    return metrics


def asset_bootstrap_weights(asset_codes: np.ndarray, n_assets: int, n_boot: int, seed: int):
    """Yield one row-weight vector per replicate: each row counts as many times as its
    asset was drawn when redrawing `n_assets` assets with replacement."""
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, n_assets, n_assets), minlength=n_assets)[asset_codes].astype(float)


def with_intervals(point: dict, replicates: list) -> dict:
    out = {}
    for key, value in point.items():
        draws = np.array([r[key] for r in replicates])
        out[key] = {"value": value, "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975))}
    return out


def evaluate_level(frame: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Metrics + asset-bootstrap intervals for one set of scored units (rows or blocks)."""
    scores = frame["signal"].to_numpy(dtype=float)
    labels = frame["is_member"].to_numpy(dtype=int)
    asset_codes, assets = pd.factorize(frame[DEFAULT_ITEM_COL])
    replicates = []
    for weights in asset_bootstrap_weights(asset_codes, len(assets), n_boot, seed):
        keep = weights > 0
        if labels[keep].min() == labels[keep].max():
            continue
        replicates.append(weighted_metrics(scores[keep], labels[keep], weights[keep]))
    result = with_intervals(weighted_metrics(scores, labels), replicates)
    result.update({"n_members": int(labels.sum()), "n_nonmembers": int((1 - labels).sum()), "n_assets": int(len(assets))})
    return result


def build_blocks(scores: pd.DataFrame, block_days: int, rows_per_block: int, seed: int) -> pd.DataFrame:
    """One row per (asset, time block): the mean score of `rows_per_block` randomly
    chosen audited rows. Blocks with fewer audited rows are dropped."""
    shuffled = scores.sample(frac=1.0, random_state=seed)
    shuffled = shuffled.assign(block=distance_bin(shuffled["days_from_cutoff"].to_numpy(), block_days))
    groups = shuffled.groupby([DEFAULT_ITEM_COL, "block"], sort=False)
    chosen = shuffled[(groups.cumcount() < rows_per_block) & (groups["signal"].transform("size") >= rows_per_block)]
    blocks = chosen.groupby([DEFAULT_ITEM_COL, "block"], sort=False).agg(signal=("signal", "mean"),
                                                                         is_member=("is_member", "first")).reset_index()
    return blocks


def temporal_analysis(scores: pd.DataFrame, bin_days: int, threshold: float, n_boot: int, seed: int) -> pd.DataFrame:
    """Per distance bin and class: the share of rows the attack flags as member at
    `threshold`, the ROC-AUC of the bin's rows against every row of the other class,
    and the median loss — the first two with asset-bootstrap intervals."""
    s = scores["signal"].to_numpy(dtype=float)
    member = scores["is_member"].to_numpy(dtype=bool)
    bins = distance_bin(scores["days_from_cutoff"].to_numpy(), bin_days)
    bin_codes, bin_values = pd.factorize(bins, sort=True)
    asset_codes, assets = pd.factorize(scores[DEFAULT_ITEM_COL])
    flagged = (s >= threshold).astype(float)

    # Position of every row among the sorted scores of the OTHER class, fixed once; a
    # replicate only changes the weights, so "weight of other-class rows below this
    # row" is a lookup in that replicate's cumulative weights.
    order = {cls: np.argsort(s[member == cls], kind="stable") for cls in (True, False)}
    sorted_scores = {cls: s[member == cls][order[cls]] for cls in (True, False)}
    left = np.where(member, np.searchsorted(sorted_scores[False], s, "left"), np.searchsorted(sorted_scores[True], s, "left"))
    right = np.where(member, np.searchsorted(sorted_scores[False], s, "right"), np.searchsorted(sorted_scores[True], s, "right"))

    def per_bin(weights):
        total = np.bincount(bin_codes, weights, minlength=len(bin_values))
        beats = np.empty(len(s))
        for cls in (True, False):
            # Rows of class `cls` are compared with the other class (~cls ... not cls).
            other = np.concatenate([[0.0], np.cumsum(weights[member != cls][order[not cls]])])
            rows = member == cls
            beats[rows] = (other[left[rows]] + other[right[rows]]) / 2 / other[-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            flag_rate = np.bincount(bin_codes, weights * flagged, minlength=len(bin_values)) / total
            beat_rate = np.bincount(bin_codes, weights * beats, minlength=len(bin_values)) / total
        # A member bin's AUC is how often its rows outscore non-members; a non-member
        # bin's AUC is how often members outscore its rows.
        return flag_rate, np.where(bin_values < 0, beat_rate, 1 - beat_rate)

    flag_rate, bin_auc = per_bin(np.ones(len(s)))
    boot = [per_bin(w) for w in asset_bootstrap_weights(asset_codes, len(assets), n_boot, seed)]
    flag_draws = np.array([b[0] for b in boot])
    auc_draws = np.array([b[1] for b in boot])

    grouped = scores.assign(bin=bins).groupby("bin", sort=True)
    out = pd.DataFrame({
        "bin": bin_values,
        "is_member": bin_values < 0,
        "days_from": np.where(bin_values < 0, bin_values * bin_days + 1, (bin_values - 1) * bin_days + 1),
        "days_to": np.where(bin_values < 0, (bin_values + 1) * bin_days, bin_values * bin_days),
        "first_date": grouped[DEFAULT_TIMESTAMP_COL].min().dt.date.to_numpy(),
        "last_date": grouped[DEFAULT_TIMESTAMP_COL].max().dt.date.to_numpy(),
        "n_rows": grouped.size().to_numpy(),
        "n_assets": grouped[DEFAULT_ITEM_COL].nunique().to_numpy(),
        "flag_rate": flag_rate,
        "flag_rate_low": np.nanquantile(flag_draws, 0.025, axis=0),
        "flag_rate_high": np.nanquantile(flag_draws, 0.975, axis=0),
        "auc_vs_opposite": bin_auc,
        "auc_vs_opposite_low": np.nanquantile(auc_draws, 0.025, axis=0),
        "auc_vs_opposite_high": np.nanquantile(auc_draws, 0.975, axis=0),
        "mean_score": grouped["signal"].mean().to_numpy(),
        "median_loss": grouped["loss"].median().to_numpy(),
    })
    out["days_mid"] = (out["days_from"] + out["days_to"]) / 2
    return out


def best_threshold(scores: pd.DataFrame) -> float:
    fpr, tpr, thresholds = roc_curve(scores["is_member"].to_numpy(dtype=int), scores["signal"].to_numpy(dtype=float))
    return float(thresholds[int(np.argmax(tpr - fpr))])


def run_for_attack(model_tag: str, date: str, attack: str, test_source: str, output_dir: Path, args) -> dict:
    attack_dir = attack_dir_name(attack, test_source)
    date_dir = output_dir / model_tag / date / attack_dir
    if not (date_dir / "member_scores.csv").exists():
        print(f"[{model_tag} @ {date}] {attack_dir}: no saved scores, skipping")
        return None

    scores = load_scores(date_dir)
    scores["days_from_cutoff"] = days_from_cutoff(scores[DEFAULT_TIMESTAMP_COL], resolve_paths(model_tag, date))

    levels = {"row": evaluate_level(scores, args.n_boot, args.seed)}
    labels = {"row": "single rows"}
    for k in args.rows_per_block:
        blocks = build_blocks(scores, args.block_days, k, args.seed)
        key = f"block_{k}_rows"
        levels[key] = evaluate_level(blocks, args.n_boot, args.seed)
        labels[key] = f"asset x {args.block_days}-day blocks, {k} row{'s' if k > 1 else ''} each"

    threshold = best_threshold(scores)
    temporal = temporal_analysis(scores, args.bin_days, threshold, args.n_boot, args.seed)

    evaluation = {
        "attack": attack_dir, "n_boot": args.n_boot, "block_days": args.block_days, "bin_days": args.bin_days,
        "flag_threshold": threshold, "level_labels": labels, "levels": levels,
    }
    with open(date_dir / "evaluation.json", "w") as handle:
        json.dump(evaluation, handle, indent=2)
    temporal.to_csv(date_dir / "temporal_analysis.csv", index=False)

    plot_dir = PLOT_ROOT / attack_dir / model_tag / date
    suffix = "" if test_source == "original" else " (counterfactual test data)"
    plot_evaluation_levels([{"label": labels[k], **levels[k]} for k in levels], plot_dir / "evaluation_levels.png",
                           title=f"{attack_dir}: rows vs asset-period blocks — {model_tag} @ {date}{suffix}")
    plot_temporal_analysis(temporal, plot_dir / "temporal_analysis.png",
                           title=f"{attack_dir}: verdict vs distance from the cutoff — {model_tag} @ {date}{suffix}",
                           threshold_label="at the attack's best threshold")

    for key, level in levels.items():
        roc, tpr = level["roc_auc"], level["tpr_at_fpr_0.01"]
        print(f"[{model_tag} @ {date}] {attack_dir:22s} {labels[key]:42s} n={level['n_members']}+{level['n_nonmembers']} "
              f"AUC={roc['value']:.3f} [{roc['ci_low']:.3f}, {roc['ci_high']:.3f}] "
              f"TPR@1%FPR={tpr['value']:.3f} [{tpr['ci_low']:.3f}, {tpr['ci_high']:.3f}]")
    return evaluation


def main():
    parser = argparse.ArgumentParser(description="Block-level, asset-bootstrap and temporal evaluation of saved MIA scores.")
    parser.add_argument("--model", required=True, help="Model tag directory name under artifacts_for_counterfactuals/")
    parser.add_argument("--dates", type=str, default=None,
                         help="Comma-separated recommendation dates (YYYY-MM-DD). Defaults to every date with a fitted pkl for this model.")
    parser.add_argument("--attacks", type=str, default=",".join(ATTACKS), help=f"Comma-separated attacks to evaluate, among {', '.join(ATTACKS)}.")
    parser.add_argument("--test-source", choices=TEST_SOURCES, default="original")
    parser.add_argument("--block-days", type=int, default=63, help="Length of a membership block, in trading days (63 is about a quarter).")
    parser.add_argument("--rows-per-block", type=lambda v: [int(k) for k in v.split(",")], default=[1, 5],
                         help="Comma-separated numbers of audited rows averaged into a block's score; one evaluation each.")
    parser.add_argument("--bin-days", type=int, default=21, help="Bin width of the temporal analysis, in trading days (21 is about a month).")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap replicates (whole assets are redrawn).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="trustworthiness_eval/privacy/membership_inference/results")
    args = parser.parse_args()

    dates = args.dates.split(",") if args.dates else available_dates(args.model)
    output_dir = Path(args.output_dir)
    for date in dates:
        for attack in args.attacks.split(","):
            run_for_attack(args.model, date, attack, args.test_source, output_dir, args)


if __name__ == "__main__":
    main()
