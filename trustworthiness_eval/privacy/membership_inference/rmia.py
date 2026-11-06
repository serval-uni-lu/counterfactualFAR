"""
Offline RMIA (Zarifzadeh et al.) against this project's already-fitted profitability model.

The target is the saved pipeline and is never retrained. Around it, this script trains:

  - reference models: same estimator and hyperparameters as the target, fitted in pairs
    on the target's own training rows. Each pair splits those rows in two halves, so
    every row is OUT (unseen) for exactly one model per pair, i.e. it is scored against
    `--num-ref` OUT reference models.
  - one supplementary model: fitted like a reference on a random half. It is never used
    as a reference — it stands in for the target (whose paired model can't exist, since
    the target is fixed) as a model with known membership, to tune offline_a.

RMIA score of x = share of population rows z with
    [P(x|target) / P(x)] / [P(z|target) / P(z)] > 1,
where P(.) is estimated from the OUT reference models only (offline RMIA):
    P(x) = (1 + a) / 2 * mean_OUT P(x|ref) + (1 - a) / 2.

A regressor has no class probability, so the signal P(x|model) is a Gaussian likelihood
of the true target, exp(-(y - y_hat)^2 / (2 * sigma^2)) in (0, 1], with sigma^2 the
attacked model's median squared error on the population (the errors are heavy-tailed, so
the mean would be set by a few outliers and flatten every signal towards 1).

Members, non-members and population are the same rows loss.py / population_attack.py
audit (same --split-seed).

Saved under <output-dir>/<model>/<date>/rmia/:
    models/   supplementary + reference models, and which training rows each one saw
    signals/  every model's prediction on the audited and population rows
    metrics.json, member_scores.csv, nonmember_scores.csv
Plots go to stats/plots/membership_inference/rmia/<model>/<date>/, like the other attacks.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

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
from utils.constants import DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL

KEY_COLS = [DEFAULT_ITEM_COL, DEFAULT_TIMESTAMP_COL]
REF_SPLITS = ("asset", "row")
OFFLINE_A_GRID = np.round(np.arange(0.0, 1.01, 0.1), 1)
SUPPLEMENTARY = 0  # column of the supplementary model; references are columns 1..2*num_ref


def model_names(num_ref: int) -> list:
    return ["supplementary"] + [f"reference_{i:02d}" for i in range(2 * num_ref)]


def draw_memberships(pool: pd.DataFrame, num_ref: int, ref_split: str, seed: int) -> np.ndarray:
    """(n_pool, 1 + 2*num_ref) bool: which training rows each trained model sees.

    `ref_split="asset"` assigns whole assets, so an OUT model has never seen any day of
    that asset: neighbouring days of one asset have almost the same KPIs and target, and
    a model that was only denied the row itself would still have effectively seen it.
    `ref_split="row"` assigns single rows instead."""
    rng = np.random.default_rng(seed)
    if ref_split == "asset":
        units, unit_of_row = np.unique(pool[DEFAULT_ITEM_COL].to_numpy(), return_inverse=True)
    else:
        units, unit_of_row = np.arange(len(pool)), np.arange(len(pool))
    if len(units) < 2:
        raise ValueError(f"Need at least 2 {ref_split}s to split the training rows, got {len(units)}")

    def random_half():
        half = np.zeros(len(units), dtype=bool)
        half[rng.permutation(len(units))[: len(units) // 2]] = True
        return half

    unit_in = np.zeros((len(units), 1 + 2 * num_ref), dtype=bool)
    unit_in[:, SUPPLEMENTARY] = random_half()
    for pair in range(num_ref):
        unit_in[:, 1 + 2 * pair] = random_half()
        unit_in[:, 2 + 2 * pair] = ~unit_in[:, 1 + 2 * pair]
    return unit_in[unit_of_row]


def train_or_load_models(target, pool: pd.DataFrame, num_ref: int, ref_split: str, seed: int,
                         models_dir: Path, retrain: bool, log) -> tuple:
    """Fit (or reload) the supplementary + reference models. Returns (models, memberships)."""
    names = model_names(num_ref)
    feature_cols = list(target.kpi_features)
    config = {"num_ref": num_ref, "ref_split": ref_split, "seed": seed, "n_training_rows": int(len(pool)),
              "estimator": type(target.model).__name__, "features": feature_cols}
    config_path = models_dir / "config.json"
    paths = [models_dir / f"{name}.joblib" for name in names]

    if not retrain and config_path.exists() and all(p.exists() for p in paths):
        with open(config_path) as handle:
            if json.load(handle) == config:
                log(f"reusing {len(names)} trained models from {models_dir}")
                memberships = np.load(models_dir / "memberships.npy")
                return [joblib.load(p) for p in paths], memberships
        log(f"models in {models_dir} were trained with another configuration, retraining")

    models_dir.mkdir(parents=True, exist_ok=True)
    memberships = draw_memberships(pool, num_ref, ref_split, seed)
    X, y = pool[feature_cols].astype(float), pool["target"].to_numpy(dtype=float)
    models = []
    for m, (name, path) in enumerate(zip(names, paths)):
        estimator = clone(target.model)
        params = estimator.get_params()
        estimator.set_params(**{k: v for k, v in (("random_state", seed + m), ("n_jobs", -1)) if k in params})
        rows = memberships[:, m]
        log(f"training {name} on {int(rows.sum())} of {len(pool)} training rows")
        estimator.fit(X[rows], y[rows])
        joblib.dump(estimator, path, compress=3)
        models.append(estimator)

    np.save(models_dir / "memberships.npy", memberships)
    pool[KEY_COLS].to_csv(models_dir / "training_rows.csv", index=False)
    with open(config_path, "w") as handle:
        json.dump(config, handle, indent=2)
    return models, memberships


def seen_by_models(rows: pd.DataFrame, in_training: bool, pool: pd.DataFrame, memberships: np.ndarray,
                   ref_split: str) -> np.ndarray:
    """(n_rows, n_models) bool: whether each model saw each row, at the granularity the
    models were split on. With the asset split that is "saw the row's asset", for
    training-period and test-period rows alike, so P(x) is estimated the same way for
    members, non-members and population; with the row split, test-period rows were seen
    by no model."""
    if ref_split == "asset":
        assets = pool[DEFAULT_ITEM_COL].to_numpy()
        _, first = np.unique(assets, return_index=True)
        asset_in = pd.DataFrame(memberships[first], index=assets[first])
        return asset_in.reindex(rows[DEFAULT_ITEM_COL].to_numpy()).fillna(False).to_numpy(dtype=bool)
    if not in_training:
        return np.zeros((len(rows), memberships.shape[1]), dtype=bool)
    pool_index = pd.Series(np.arange(len(pool)), index=pd.MultiIndex.from_frame(pool[KEY_COLS]))
    return memberships[pool_index.loc[pd.MultiIndex.from_frame(rows[KEY_COLS])].to_numpy()]


def gaussian_signal(predictions: np.ndarray, y: np.ndarray, sigma2: float) -> np.ndarray:
    """P(x|model) for a regressor: likelihood of the true target, in (0, 1]."""
    predictions = np.asarray(predictions, dtype=float)
    y = np.asarray(y, dtype=float)
    if predictions.ndim == 2:
        y = y[:, None]
    return np.exp(-((predictions - y) ** 2) / (2 * sigma2))


def offline_ratio(target_signal: np.ndarray, ref_signals: np.ndarray, ref_out: np.ndarray, offline_a: float) -> np.ndarray:
    """P(x|target) / P(x), with P(x) approximated from the OUT reference models only."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        mean_out = (ref_signals * ref_out).sum(axis=1) / ref_out.sum(axis=1)
        mean_x = (1 + offline_a) / 2 * mean_out + (1 - offline_a) / 2
        return target_signal / mean_x


def rmia_scores(ratio_x: np.ndarray, ratio_z: np.ndarray) -> np.ndarray:
    """Share of population rows z that x dominates (ratio_x / ratio_z > 1), higher = more
    likely a member. Both ratios are >= 0, so that is ratio_x > ratio_z: rank each x
    within the sorted population ratios instead of building the n_x by n_z matrix. A 0/0
    ratio (signals underflowed to 0) dominates nothing: NaN sorts last, so it is never
    counted on the population side, and scores 0 on the x side."""
    scores = np.searchsorted(np.sort(ratio_z), ratio_x, side="left") / len(ratio_z)
    return np.where(np.isnan(ratio_x), 0.0, scores)


def run_offline_rmia(attacked_x, refs_x, out_x, y_x, attacked_z, refs_z, out_z, y_z, offline_a: float) -> np.ndarray:
    """Offline RMIA scores of the x rows against the model whose predictions are
    `attacked_*`, from raw predictions: (n,) attacked model, (n, n_refs) references."""
    sigma2 = float(np.median((np.asarray(attacked_z, dtype=float) - y_z) ** 2)) + 1e-12
    ratio_x = offline_ratio(gaussian_signal(attacked_x, y_x, sigma2), gaussian_signal(refs_x, y_x, sigma2), out_x, offline_a)
    ratio_z = offline_ratio(gaussian_signal(attacked_z, y_z, sigma2), gaussian_signal(refs_z, y_z, sigma2), out_z, offline_a)
    return rmia_scores(ratio_x, ratio_z)


def tune_offline_a(audit: dict, population: dict) -> tuple:
    """Pick offline_a by attacking the supplementary model, whose members are known: the
    audited member rows it was trained on, against the audited non-members."""
    is_member = audit["is_member"]
    sup_member = is_member & audit["seen"][:, SUPPLEMENTARY]
    rows = sup_member | ~is_member
    if not sup_member.any():
        raise ValueError("No audited member row was used to train the supplementary model, cannot tune offline_a")

    aucs = {}
    for a in OFFLINE_A_GRID:
        scores = run_offline_rmia(
            audit["predictions"][rows, SUPPLEMENTARY], audit["predictions"][rows, 1:], ~audit["seen"][rows, 1:], audit["y"][rows],
            population["predictions"][:, SUPPLEMENTARY], population["predictions"][:, 1:], ~population["seen"][:, 1:], population["y"],
            float(a),
        )
        aucs[float(a)] = float(roc_auc_score(sup_member[rows], scores))
    best_a = max(aucs, key=aucs.get)
    return best_a, aucs


def predict_all(models: list, rows: pd.DataFrame, feature_cols: list) -> np.ndarray:
    X = rows[feature_cols].astype(float)
    return np.column_stack([np.asarray(m.predict(X), dtype=float).reshape(-1) for m in models])


def run_for_date(model_tag: str, date: str, months: int, output_dir: Path, split_seed: int,
                 num_ref: int, ref_split: str, seed: int, retrain: bool,
                 test_source: str = "original", cf_method: str = "genetic") -> dict:
    attack_dir = attack_dir_name("rmia", test_source)
    tag = f"[{model_tag} @ {date}]"

    def log(message):
        print(f"{tag} {message}", flush=True)

    paths = resolve_paths(model_tag, date)
    for key in ("pkl", "training_data", "testing_data"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing {key} for model={model_tag} date={date}: {paths[key]}")

    log("loading target model")
    target = load_target_model(paths["pkl"])
    training_ts = pd.read_csv(paths["training_data"])
    testing_ts = pd.read_csv(paths["testing_data"])
    cf_ts = None
    if test_source == "counterfactual":
        cf_path = resolve_cf_timeseries(model_tag, date, cf_method)
        if not cf_path.exists():
            raise FileNotFoundError(f"Missing counterfactual time series for model={model_tag} date={date}: {cf_path}")
        cf_ts = load_cf_timeseries(cf_path)
        log(f"test-period rows taken from counterfactuals: {cf_path}")

    log("building balanced member/non-member auditing sets + population reference")
    member_df, nonmember_df, population_df, pool = build_auditing_sets(
        training_ts, testing_ts, target, months, split_seed, cf_ts=cf_ts,
        keep_features=True, return_member_pool=True)
    log(summarize_rows(member_df, "member"))
    log(summarize_rows(nonmember_df, "non-member"))
    log(summarize_rows(population_df, "population"))

    feature_cols = list(target.kpi_features)
    date_dir = output_dir / model_tag / date / attack_dir
    # The trained models only depend on the target's training rows, so the
    # counterfactual-test-data run shares them with the original-data one.
    models_dir = output_dir / model_tag / date / "rmia" / "models"
    models, memberships = train_or_load_models(target, pool, num_ref, ref_split, seed, models_dir, retrain, log)

    log("computing signals of every model on the audited and population rows")
    audit_df = pd.concat([member_df, nonmember_df], ignore_index=True)
    is_member = np.r_[np.ones(len(member_df), dtype=bool), np.zeros(len(nonmember_df), dtype=bool)]
    audit = {
        "y": audit_df["target"].to_numpy(dtype=float),
        "is_member": is_member,
        "target_prediction": audit_df["prediction"].to_numpy(dtype=float),
        "predictions": predict_all(models, audit_df, feature_cols),
        "seen": np.vstack([seen_by_models(member_df, True, pool, memberships, ref_split),
                           seen_by_models(nonmember_df, False, pool, memberships, ref_split)]),
    }
    population = {
        "y": population_df["target"].to_numpy(dtype=float),
        "target_prediction": population_df["prediction"].to_numpy(dtype=float),
        "predictions": predict_all(models, population_df, feature_cols),
        "seen": seen_by_models(population_df, False, pool, memberships, ref_split),
    }

    signal_dir = date_dir / "signals"
    signal_dir.mkdir(parents=True, exist_ok=True)
    names = model_names(num_ref)
    for name, rows, data in (("audit", audit_df, audit), ("population", population_df, population)):
        keys = rows[KEY_COLS + ["target"]].copy()
        if name == "audit":
            keys["is_member"] = is_member.astype(int)
        keys.to_csv(signal_dir / f"{name}_rows.csv", index=False)
        # Column 0 is the target model, then the supplementary and reference models.
        np.save(signal_dir / f"{name}_predictions.npy", np.column_stack([data["target_prediction"], data["predictions"]]))
        np.save(signal_dir / f"{name}_seen_by_model.npy", data["seen"])
    with open(signal_dir / "columns.json", "w") as handle:
        json.dump({"predictions": ["target"] + names, "seen_by_model": names}, handle, indent=2)

    offline_a, tuning_aucs = tune_offline_a(audit, population)
    log(f"offline_a={offline_a:.1f} (supplementary-model AUC {tuning_aucs[offline_a]:.4f})")

    scores = run_offline_rmia(
        audit["target_prediction"], audit["predictions"][:, 1:], ~audit["seen"][:, 1:], audit["y"],
        population["target_prediction"], population["predictions"][:, 1:], ~population["seen"][:, 1:], population["y"],
        offline_a,
    )
    with np.errstate(invalid="ignore"):
        out_loss = np.nanmean(np.where(~audit["seen"][:, 1:], (audit["predictions"][:, 1:] - audit["y"][:, None]) ** 2, np.nan), axis=1)

    score_cols = KEY_COLS + ["prediction", "target", "loss"]
    scored = audit_df[score_cols].assign(mean_out_reference_loss=out_loss, rmia_score=scores, signal=scores)
    member_out, nonmember_out = scored[is_member], scored[~is_member]

    metrics = attack_metrics(member_scores=member_out["signal"].to_numpy(), nonmember_scores=nonmember_out["signal"].to_numpy())
    metrics.update({
        "n_population": int(len(population_df)),
        "num_ref": num_ref,
        "n_models_trained": len(models),
        "ref_split": ref_split,
        "offline_a": offline_a,
        "offline_a_tuning_auc": tuning_aucs,
        "sigma2": float(np.median(population_df["loss"])) + 1e-12,
        "median_loss": {
            "member_target": float(member_out["loss"].median()),
            "member_out_references": float(member_out["mean_out_reference_loss"].median()),
            "nonmember_target": float(nonmember_out["loss"].median()),
            "nonmember_out_references": float(nonmember_out["mean_out_reference_loss"].median()),
        },
        "test_source": test_source,
    })

    member_out.to_csv(date_dir / "member_scores.csv", index=False)
    nonmember_out.to_csv(date_dir / "nonmember_scores.csv", index=False)
    with open(date_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)

    plot_dir = PLOT_ROOT / attack_dir / model_tag / date
    suffix = "" if test_source == "original" else " (counterfactual test data)"
    plot_roc_curve(member_out["signal"].to_numpy(), nonmember_out["signal"].to_numpy(),
                   plot_dir / "roc_curve.png", title=f"RMIA (offline) ROC — {model_tag} @ {date}{suffix}")
    plot_loss_distribution(member_out["loss"].to_numpy(), nonmember_out["loss"].to_numpy(),
                            plot_dir / "loss_distribution.png", title=f"Loss distribution — {model_tag} @ {date}{suffix}",
                            population_loss=population_df["loss"].to_numpy())
    plot_percentile_distribution(member_out["signal"].to_numpy(), nonmember_out["signal"].to_numpy(),
                                  plot_dir / "score_distribution.png",
                                  title=f"RMIA score — {model_tag} @ {date}{suffix}",
                                  xlabel="RMIA score (share of population rows the sample dominates)")
    plot_tpr_at_low_fpr(metrics["tpr_at_fpr"], plot_dir / "tpr_at_low_fpr.png",
                        title=f"TPR @ low FPR — {model_tag} @ {date}{suffix}")

    log(f"members={metrics['n_members']} nonmembers={metrics['n_nonmembers']} population={metrics['n_population']} "
        f"AUC={metrics['roc_auc']:.4f} advantage={metrics['attack_advantage']:.4f} "
        f"TPR@1%FPR={metrics['tpr_at_fpr'][0.01]:.4f} (offline_a={offline_a:.1f})")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run offline RMIA on a fitted profitability model.")
    parser.add_argument("--model", required=True, help="Model tag directory name under artifacts_for_counterfactuals/")
    parser.add_argument("--dates", type=str, default=None,
                         help="Comma-separated recommendation dates (YYYY-MM-DD). Defaults to every date with a fitted pkl for this model.")
    parser.add_argument("--months", type=int, default=6, help="Profitability horizon in months (must match how the model was trained).")
    parser.add_argument("--split-seed", type=int, default=42,
                         help="Seed for the audit/population split and member downsampling — use the same value as loss.py / population_attack.py to audit identical rows.")
    parser.add_argument("--num-ref", type=int, default=5,
                         help="Reference model pairs (2 models each), i.e. the number of OUT reference models every row is scored with.")
    parser.add_argument("--ref-split", choices=REF_SPLITS, default="asset",
                         help="Split the target's training rows between reference models by whole asset, or by single row.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the reference/supplementary models and their training splits.")
    parser.add_argument("--retrain", action="store_true", help="Retrain the reference/supplementary models even if saved ones exist.")
    parser.add_argument("--output-dir", type=str, default="trustworthiness_eval/privacy/membership_inference/results")
    parser.add_argument("--test-source", choices=TEST_SOURCES, default="original",
                         help="Where the non-member and population rows come from: the real testing data, or the "
                              "counterfactual time series under counterfactuals_results/ (members are real either way).")
    parser.add_argument("--cf-method", type=str, default="genetic",
                         help="Counterfactual method suffix of the cf_timeseries file (only with --test-source counterfactual).")
    args = parser.parse_args()
    if args.num_ref < 1:
        parser.error("--num-ref must be >= 1: offline RMIA needs at least one reference pair")

    dates = args.dates.split(",") if args.dates else available_dates(args.model)
    if not dates:
        raise FileNotFoundError(f"No fitted pkl files found for model={args.model} under artifacts_for_counterfactuals/")

    output_dir = Path(args.output_dir)
    for date in dates:
        run_for_date(args.model, date, args.months, output_dir, args.split_seed, args.num_ref, args.ref_split,
                     args.seed, args.retrain, args.test_source, args.cf_method)

    # Pool/plot across every date ever computed for this model.
    attack_dir = attack_dir_name("rmia", args.test_source)
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
        with open(model_dir / f"{attack_dir}_metrics.json", "w") as handle:
            json.dump(pooled, handle, indent=2)

        plot_metric_across_dates(
            sorted_dates,
            [all_metrics[d]["roc_auc"] for d in sorted_dates],
            [all_metrics[d]["attack_advantage"] for d in sorted_dates],
            [all_metrics[d]["tpr_at_fpr"][0.01] for d in sorted_dates],
            PLOT_ROOT / attack_dir / args.model / "auc_by_date.png",
            title=f"RMIA (offline) strength across dates — {args.model}" + ("" if args.test_source == "original" else " (counterfactual test data)"),
        )

        print(f"[{args.model}] pooled across {len(sorted_dates)} dates on disk: "
              f"mean AUC={pooled['mean_roc_auc']:.4f} mean advantage={pooled['mean_attack_advantage']:.4f}")


if __name__ == "__main__":
    main()
