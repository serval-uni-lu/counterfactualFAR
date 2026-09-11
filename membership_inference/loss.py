"""
LOSS membership inference attack for this project's regression models.

Audits a single already-fitted target model directly, comparing per-row loss
on training data (members) against testing data (non-members): a sample is
predicted "member" when its loss is unusually low (Yeom et al.).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from membership_inference.regression_signals import (
    attack_metrics,
    available_dates,
    build_auditing_sets,
    load_all_date_metrics,
    load_target_model,
    resolve_paths,
    summarize_rows,
)
from membership_inference.plots import (
    plot_loss_distribution,
    plot_metric_across_dates,
    plot_roc_curve,
    plot_tpr_at_low_fpr,
)

STATS_DIR = Path("stats") / "membership_inference" / "loss_attack"


def run_for_date(model_tag: str, date: str, months: int, output_dir: Path, split_seed: int) -> dict:
    paths = resolve_paths(model_tag, date)
    for key in ("pkl", "training_data", "testing_data"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing {key} for model={model_tag} date={date}: {paths[key]}")

    print(f"[{model_tag} @ {date}] loading target model")
    model = load_target_model(paths["pkl"])
    training_ts = pd.read_csv(paths["training_data"])
    testing_ts = pd.read_csv(paths["testing_data"])

    print(f"[{model_tag} @ {date}] building balanced member/non-member auditing sets")
    member_df, nonmember_df, _ = build_auditing_sets(training_ts, testing_ts, model, months, split_seed)
    print(f"[{model_tag} @ {date}] {summarize_rows(member_df, 'member')}")
    print(f"[{model_tag} @ {date}] {summarize_rows(nonmember_df, 'non-member')}")

    # "signal" is the exact value fed into attack_metrics/roc_curve (lower loss ->
    # higher signal, oriented "higher = more likely member").
    member_df = member_df.assign(signal=-member_df["loss"])
    nonmember_df = nonmember_df.assign(signal=-nonmember_df["loss"])

    metrics = attack_metrics(member_scores=member_df["signal"].to_numpy(),
                              nonmember_scores=nonmember_df["signal"].to_numpy())

    date_dir = output_dir / model_tag / date / "loss_attack"
    date_dir.mkdir(parents=True, exist_ok=True)
    member_df.to_csv(date_dir / "member_scores.csv", index=False)
    nonmember_df.to_csv(date_dir / "nonmember_scores.csv", index=False)
    with open(date_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)

    plot_dir = STATS_DIR / model_tag / date
    plot_roc_curve(member_df["signal"].to_numpy(), nonmember_df["signal"].to_numpy(),
                   plot_dir / "roc_curve.png", title=f"LOSS attack ROC — {model_tag} @ {date}")
    plot_loss_distribution(member_df["loss"].to_numpy(), nonmember_df["loss"].to_numpy(),
                            plot_dir / "loss_distribution.png", title=f"Loss distribution — {model_tag} @ {date}")
    plot_tpr_at_low_fpr(metrics["tpr_at_fpr"], plot_dir / "tpr_at_low_fpr.png",
                        title=f"TPR @ low FPR — {model_tag} @ {date}")

    print(f"[{model_tag} @ {date}] members={metrics['n_members']} nonmembers={metrics['n_nonmembers']} "
          f"AUC={metrics['roc_auc']:.4f} advantage={metrics['attack_advantage']:.4f} "
          f"TPR@1%FPR={metrics['tpr_at_fpr'][0.01]:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run LOSS membership inference attack on a fitted profitability model.")
    parser.add_argument("--model", required=True, help="Model tag directory name under artifacts_for_counterfactuals/")
    parser.add_argument("--dates", type=str, default=None,
                         help="Comma-separated recommendation dates (YYYY-MM-DD). Defaults to every date with a fitted pkl for this model.")
    parser.add_argument("--months", type=int, default=6, help="Profitability horizon in months (must match how the model was trained).")
    parser.add_argument("--split-seed", type=int, default=42,
                         help="Seed for the audit/population split and member downsampling — use the same value as population_attack.py to audit identical rows.")
    parser.add_argument("--output-dir", type=str, default="membership_inference/results")
    args = parser.parse_args()

    dates = args.dates.split(",") if args.dates else available_dates(args.model)
    if not dates:
        raise FileNotFoundError(f"No fitted pkl files found for model={args.model} under artifacts_for_counterfactuals/")

    output_dir = Path(args.output_dir)
    for date in dates:
        run_for_date(args.model, date, args.months, output_dir, args.split_seed)

    # Pool/plot across every date ever computed for this model.
    all_metrics = load_all_date_metrics(output_dir, args.model, "loss_attack")
    if len(all_metrics) > 1:
        sorted_dates = sorted(all_metrics.keys())
        pooled = {
            "dates": sorted_dates,
            "mean_roc_auc": float(np.mean([all_metrics[d]["roc_auc"] for d in sorted_dates])),
            "mean_attack_advantage": float(np.mean([all_metrics[d]["attack_advantage"] for d in sorted_dates])),
            "per_date": {d: all_metrics[d] for d in sorted_dates},
        }
        model_dir = output_dir / args.model
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "loss_attack_metrics.json", "w") as handle:
            json.dump(pooled, handle, indent=2)

        plot_metric_across_dates(
            sorted_dates,
            [all_metrics[d]["roc_auc"] for d in sorted_dates],
            [all_metrics[d]["attack_advantage"] for d in sorted_dates],
            [all_metrics[d]["tpr_at_fpr"][0.01] for d in sorted_dates],
            STATS_DIR / args.model / "auc_by_date.png",
            title=f"LOSS attack strength across dates — {args.model}",
        )

        print(f"[{args.model}] pooled across {len(sorted_dates)} dates on disk: "
              f"mean AUC={pooled['mean_roc_auc']:.4f} mean advantage={pooled['mean_attack_advantage']:.4f}")


if __name__ == "__main__":
    main()
