"""
Population membership inference attack.

Population data is a held-out random half of the real test-period rows (stratified
per asset), disjoint from the other half used as the audited non-members. Each sample is scored
by the fraction of the population whose -loss is <= its own (ml_privacy_meter's attack-P);
a score near 1 (lower loss than most of the population) means "member".
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from trustworthiness_eval.privacy.membership_inference.regression_signals import (
    TEST_SOURCES,
    attack_dir_name,
    attack_metrics,
    available_dates,
    build_auditing_sets,
    load_all_date_metrics,
    load_cf_timeseries,
    load_target_model,
    population_score,
    resolve_cf_timeseries,
    resolve_paths,
    summarize_rows,
)
from trustworthiness_eval.privacy.membership_inference.plots import (
    PLOT_ROOT,
    plot_loss_distribution,
    plot_metric_across_dates,
    plot_percentile_distribution,
    plot_roc_curve,
    plot_tpr_at_low_fpr,
)


def run_for_date(model_tag: str, date: str, months: int, output_dir: Path, split_seed: int,
                 test_source: str = "original", cf_method: str = "genetic") -> dict:
    attack_dir = attack_dir_name("population_attack", test_source)
    paths = resolve_paths(model_tag, date)
    for key in ("pkl", "training_data", "testing_data"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing {key} for model={model_tag} date={date}: {paths[key]}")

    print(f"[{model_tag} @ {date}] loading target model")
    model = load_target_model(paths["pkl"])
    training_ts = pd.read_csv(paths["training_data"])
    testing_ts = pd.read_csv(paths["testing_data"])
    cf_ts = None
    if test_source == "counterfactual":
        cf_path = resolve_cf_timeseries(model_tag, date, cf_method)
        if not cf_path.exists():
            raise FileNotFoundError(f"Missing counterfactual time series for model={model_tag} date={date}: {cf_path}")
        cf_ts = load_cf_timeseries(cf_path)
        print(f"[{model_tag} @ {date}] test-period rows taken from counterfactuals: {cf_path}")

    print(f"[{model_tag} @ {date}] building balanced member/non-member auditing sets + population reference")
    member_df, nonmember_df, population_df = build_auditing_sets(training_ts, testing_ts, model, months, split_seed, cf_ts=cf_ts)
    print(f"[{model_tag} @ {date}] {summarize_rows(member_df, 'member')}")
    print(f"[{model_tag} @ {date}] {summarize_rows(nonmember_df, 'non-member')}")
    print(f"[{model_tag} @ {date}] {summarize_rows(population_df, 'population')}")

    population_losses = population_df["loss"].to_numpy()
    # Per-sample signal is -loss ("higher = more likely member", the regression analogue of
    # true-label confidence); the score is its empirical CDF within the population's signals.
    member_score = population_score(-member_df["loss"].to_numpy(), -population_losses)
    nonmember_score = population_score(-nonmember_df["loss"].to_numpy(), -population_losses)

    member_df = member_df.assign(population_score=member_score, signal=member_score)
    nonmember_df = nonmember_df.assign(population_score=nonmember_score, signal=nonmember_score)

    metrics = attack_metrics(member_scores=member_df["signal"].to_numpy(),
                              nonmember_scores=nonmember_df["signal"].to_numpy())
    metrics["n_population"] = int(len(population_losses))
    metrics["population_loss_mean"] = float(np.mean(population_losses))

    metrics["test_source"] = test_source

    date_dir = output_dir / model_tag / date / attack_dir
    date_dir.mkdir(parents=True, exist_ok=True)
    member_df.to_csv(date_dir / "member_scores.csv", index=False)
    nonmember_df.to_csv(date_dir / "nonmember_scores.csv", index=False)
    population_df.to_csv(date_dir / "population_scores.csv", index=False)
    with open(date_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)

    plot_dir = PLOT_ROOT / attack_dir / model_tag / date
    suffix = "" if test_source == "original" else " (counterfactual test data)"
    plot_roc_curve(member_df["signal"].to_numpy(), nonmember_df["signal"].to_numpy(),
                   plot_dir / "roc_curve.png", title=f"Population attack ROC — {model_tag} @ {date}{suffix}")
    plot_loss_distribution(member_df["loss"].to_numpy(), nonmember_df["loss"].to_numpy(),
                            plot_dir / "loss_distribution.png", title=f"Loss distribution — {model_tag} @ {date}{suffix}",
                            population_loss=population_losses)
    plot_percentile_distribution(member_score, nonmember_score,
                                  plot_dir / "percentile_distribution.png",
                                  title=f"Population score — {model_tag} @ {date}{suffix}")
    plot_tpr_at_low_fpr(metrics["tpr_at_fpr"], plot_dir / "tpr_at_low_fpr.png",
                        title=f"TPR @ low FPR — {model_tag} @ {date}{suffix}")

    print(f"[{model_tag} @ {date}] members={metrics['n_members']} nonmembers={metrics['n_nonmembers']} "
          f"population={metrics['n_population']} AUC={metrics['roc_auc']:.4f} "
          f"advantage={metrics['attack_advantage']:.4f} TPR@1%FPR={metrics['tpr_at_fpr'][0.01]:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run population membership inference attack on a fitted profitability model.")
    parser.add_argument("--model", required=True, help="Model tag directory name under artifacts_for_counterfactuals/")
    parser.add_argument("--dates", type=str, default=None,
                         help="Comma-separated recommendation dates (YYYY-MM-DD). Defaults to every date with a fitted pkl for this model.")
    parser.add_argument("--months", type=int, default=6, help="Profitability horizon in months (must match how the model was trained).")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for the audit/population split of test-period rows.")
    parser.add_argument("--output-dir", type=str, default="trustworthiness_eval/privacy/membership_inference/results")
    parser.add_argument("--test-source", choices=TEST_SOURCES, default="original",
                         help="Where the non-member and population rows come from: the real testing data, or the "
                              "counterfactual time series under counterfactuals_results/ (members are real either way).")
    parser.add_argument("--cf-method", type=str, default="genetic",
                         help="Counterfactual method suffix of the cf_timeseries file (only with --test-source counterfactual).")
    args = parser.parse_args()

    dates = args.dates.split(",") if args.dates else available_dates(args.model)
    if not dates:
        raise FileNotFoundError(f"No fitted pkl files found for model={args.model} under artifacts_for_counterfactuals/")

    output_dir = Path(args.output_dir)
    for date in dates:
        run_for_date(args.model, date, args.months, output_dir, args.split_seed, args.test_source, args.cf_method)

    # Pool/plot across every date ever computed for this model.
    attack_dir = attack_dir_name("population_attack", args.test_source)
    all_metrics = load_all_date_metrics(output_dir, args.model, attack_dir)
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
        with open(model_dir / f"{attack_dir}_metrics.json", "w") as handle:
            json.dump(pooled, handle, indent=2)

        plot_metric_across_dates(
            sorted_dates,
            [all_metrics[d]["roc_auc"] for d in sorted_dates],
            [all_metrics[d]["attack_advantage"] for d in sorted_dates],
            [all_metrics[d]["tpr_at_fpr"][0.01] for d in sorted_dates],
            PLOT_ROOT / attack_dir / args.model / "auc_by_date.png",
            title=f"Population attack strength across dates — {args.model}" + ("" if args.test_source == "original" else " (counterfactual test data)"),
        )

        print(f"[{args.model}] pooled across {len(sorted_dates)} dates on disk: "
              f"mean AUC={pooled['mean_roc_auc']:.4f} mean advantage={pooled['mean_attack_advantage']:.4f}")


if __name__ == "__main__":
    main()
