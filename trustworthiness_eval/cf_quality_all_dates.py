"""
Counterfactual quality at every saved recommendation date (not only the last date of each experiment).

Uses the same functions as evaluate.py (Section 5): coverage, proximity, sparsity, plausibility and actionability of the CFs at the recommendation date, and the
end-to-end re-check of every CF through the saved model. Also writes the test-period quality
(all CFs of the window) without the re-check. Summary lines broken by interrupted writes are skipped
and counted (summary_broken_lines_skipped).

Each date is saved to <output-dir>/<model>/all_dates/cf_quality_by_date/<date>.csv, so an interrupted run resumes
where it stopped; the combined file is <output-dir>/<model>/all_dates/cf_quality.csv.

Usage (from the repo root):
    python trustworthiness_eval/cf_quality_all_dates.py --dates all --jobs 8
"""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "trustworthiness_eval", ROOT / "counterfactuals_generation"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

pd.set_option("future.infer_string", False)

from common import (TS, _load_model, asset_reference, cf_quality, load_cfs, price_history,  # noqa: E402
                      rechecked_at_rec, window_stats)


def one_date(date, args, out_file):
    tag = args.model
    artifacts = ROOT / "artifacts_for_counterfactuals" / tag
    cf_dir = ROOT / "counterfactuals_results" / tag
    rec = pd.Timestamp(date)

    history = price_history(artifacts, tag, date)
    summary_path = cf_dir / f"summary_{tag}_{date}_genetic.csv"
    csv.field_size_limit(sys.maxsize)  # a broken line can open a quote that runs over many lines
    with open(summary_path, newline="") as f:
        reader = csv.reader(f)
        width = len(next(reader))
        bad_lines = sum(len(row) != width for row in reader)
    q, n_orphans = load_cfs(cf_dir, tag, date, on_bad_lines="skip")
    stats_all = window_stats(q, asset_reference(history, rec))
    at_rec = q[q[TS] == rec]
    quality = [cf_quality(at_rec, stats_all[stats_all[TS] == rec], "recommendation_date", date),
               cf_quality(q, stats_all, "test_period", date)]
    quality[1]["orphan_cfs_without_summary"] = n_orphans
    quality[1]["summary_broken_lines_skipped"] = bad_lines

    model = _load_model(artifacts / f"profitability_recommendation_pipeline_{date}_00-00-00_{tag}.pkl")
    reval, valid_rows = rechecked_at_rec(model, history, at_rec, f"recommendation_date {date}")
    ok = reval.dropna(subset=["gen_cf"]) if "gen_cf" in reval else reval.iloc[0:0]
    quality[0]["n_revalidated"] = len(ok)
    quality[0]["revalidation_errors"] = len(reval) - len(ok)
    quality[0]["coverage_queries_valid_end_to_end"] = len(valid_rows) / len(at_rec) if len(at_rec) else float("nan")
    if len(ok):
        quality[0]["validity_rechecked_generator_path"] = ok["valid_generator_path"].mean()
        quality[0]["validity_rechecked_end_to_end"] = ok["valid_end_to_end"].mean()
        quality[0]["cf_raises_score_end_to_end"] = ok["cf_raises_score_end_to_end"].mean()
        quality[0]["reported_cf_score_reproduced"] = ok["reported_cf_reproduced"].mean()

    df = pd.DataFrame(quality).rename(columns={"window": "date"})
    df.to_csv(out_file, index=False)
    r = df.iloc[0]
    print(f"[{date}] coverage {r['coverage_queries']:.1%}, validity re-checked "
          f"{r.get('validity_rechecked_end_to_end', float('nan')):.1%}, sparsity {r['sparsity_share_changed_mean']:.1%}, "
          f"plausibility {r['plausibility_cf']:.1%}", flush=True)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="rfr_n-193_leaf-96_depth-30_kpi-full_short_internal_kpis")
    parser.add_argument("--dates", default="all", help="comma-separated dates, or 'all'")
    parser.add_argument("--jobs", type=int, default=8, help="dates processed in parallel")
    parser.add_argument("--output-dir", default="trustworthiness_eval/outputs")
    args = parser.parse_args()

    artifacts = ROOT / "artifacts_for_counterfactuals" / args.model
    cf_dir = ROOT / "counterfactuals_results" / args.model
    saved = sorted(p.name.split("_")[3] for p in artifacts.glob("profitability_recommendation_pipeline_*.pkl"))
    saved = [d for d in saved if (cf_dir / f"summary_{args.model}_{d}_genetic.csv").exists()]
    dates = saved if args.dates == "all" else args.dates.split(",")

    out_dir = ROOT / args.output_dir / args.model / "all_dates"
    by_date = out_dir / "cf_quality_by_date"
    by_date.mkdir(parents=True, exist_ok=True)
    todo = [d for d in dates if not (by_date / f"{d}.csv").exists()]
    print(f"{len(dates)} dates, {len(dates) - len(todo)} already done, {len(todo)} to run", flush=True)

    Parallel(n_jobs=args.jobs, backend="loky")(delayed(one_date)(d, args, by_date / f"{d}.csv") for d in todo)

    done = [by_date / f"{d}.csv" for d in dates if (by_date / f"{d}.csv").exists()]
    all_q = pd.concat([pd.read_csv(p) for p in done], ignore_index=True)
    all_q.to_csv(out_dir / "cf_quality.csv", index=False)
    print(f"saved {out_dir / 'cf_quality.csv'} ({all_q['date'].nunique()} dates)")


if __name__ == "__main__":
    main()
