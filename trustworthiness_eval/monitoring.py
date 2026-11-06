"""
Population stability index ("Finance metrics", Monitoring after deployment) of the model inputs and
of the score distribution, for every saved window of one model. Nothing is retrained.

    PSI = sum_i (a_i - e_i) * ln(a_i / e_i)

e_i: shares of the development data (the window's own training rows, kpis_train_*.csv) over ten bins
cut at the development deciles; a_i: shares of the current period (the rows scored at the
recommendation date, predictions_*.csv). The development scores are the saved regressor's
predictions on the training rows. PSI < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant
shift (triggers a review).

Usage (from the repo root):
    python trustworthiness_eval/monitoring.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "counterfactuals_generation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from generate_counterfactuals import _load_model  # noqa: E402

BINS = 10
EPS = 1e-6


def psi(dev, cur, bins=BINS):
    dev, cur = np.asarray(dev, float), np.asarray(cur, float)
    dev, cur = dev[np.isfinite(dev)], cur[np.isfinite(cur)]
    if len(dev) == 0 or len(cur) == 0:
        return np.nan
    edges = np.unique(np.quantile(dev, np.linspace(0, 1, bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(edges, dev, side="right"), minlength=len(edges) + 1) / len(dev)
    a = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1) / len(cur)
    e, a = np.clip(e, EPS, None), np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="rfr_n-193_leaf-96_depth-30_kpi-full_short_internal_kpis")
    parser.add_argument("--output-dir", default="trustworthiness_eval/outputs")
    args = parser.parse_args()

    artifacts = ROOT / "artifacts_for_counterfactuals" / args.model
    dates = sorted(p.name.split("_")[3] for p in artifacts.glob("profitability_recommendation_pipeline_*.pkl"))
    summary, per_feature = [], []
    for date in dates:
        stem = f"{date}_00-00-00_{args.model}"
        model = _load_model(artifacts / f"profitability_recommendation_pipeline_{stem}.pkl")
        feats = list(model.kpi_features)
        dev = pd.read_csv(artifacts / f"kpis_train_{stem}.csv")
        cur = pd.read_csv(artifacts / f"predictions_{stem}.csv")
        dev_score = np.asarray(model.model.predict(dev[feats])).reshape(-1)
        importance = dict(zip(feats, getattr(model.model, "feature_importances_", np.full(len(feats), np.nan))))
        f_psi = {f: psi(dev[f], cur[f]) for f in feats if f in cur}
        s_psi = psi(dev_score, cur["prediction"])
        for f, val in f_psi.items():
            per_feature.append({"date": date, "feature": f, "psi": val, "importance": importance.get(f)})
        top = sorted(feats, key=lambda f: -(importance.get(f) or 0))[:5]
        summary.append({
            "date": date, "psi_score": s_psi,
            "psi_feature_max": max(f_psi.values()), "feature_max": max(f_psi, key=f_psi.get),
            "psi_feature_mean": float(np.mean(list(f_psi.values()))),
            "psi_top5_importance_max": max(f_psi[f] for f in top if f in f_psi),
            "n_features_above_0.25": int(sum(v > 0.25 for v in f_psi.values())),
            "review_triggered": bool(s_psi > 0.25 or max(f_psi[f] for f in top if f in f_psi) > 0.25),
        })
        print(f"[{date}] PSI score {s_psi:.3f}, max feature {summary[-1]['psi_feature_max']:.3f} "
              f"({summary[-1]['feature_max']}), top-5 features max {summary[-1]['psi_top5_importance_max']:.3f}",
              flush=True)

    out_dir = ROOT / args.output_dir / args.model / "all_dates"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out_dir / "monitoring_psi.csv", index=False)
    pd.DataFrame(per_feature).to_csv(out_dir / "monitoring_psi_by_feature.csv", index=False)
    print(f"saved monitoring_psi.csv and monitoring_psi_by_feature.csv in {out_dir}")


if __name__ == "__main__":
    main()
