# Counterfactuals in Asset Recommendations

This project builds on [FAR-Trans: An Investment Dataset for Financial Asset Recommendation](https://github.com/JavierSanzCruza/far-trans). It focuses on profitability-based recommendation models and counterfactual explanation generation.

**Data source:** https://researchdata.gla.ac.uk/1658/

---

## Installation

Create and activate a virtual environment with Python 3.9:

```bash
pipenv --python /usr/bin/python3
pipenv install
pipenv shell
```

If Python 3.9 is not available at that path:

```bash
pipenv --python $(which python3)
pipenv install
pipenv shell
```

---

## Usage

### 1. Dataset Analysis (optional)

Analyse asset/customer profitability over time. Not required for running recommendations.

```bash
python3 run_dataset_analysis.py FAR-Trans-Data output
```

Arguments:

| Argument | Description | Example |
|---|---|---|
| `interactions` | Transactions file | `transactions.csv` |
| `time_series` | Price file | `close_prices.csv` |
| `subcommand` | `range` or `fixed_dates` | |
| → `min_date` / `max_date` | Date range (if `range`) | `2019-08-01` / `2021-02-26` |
| → `num_splits` / `num_future` | Split config (if `range`) | `28` / `13` |
| → `split_dates` / `future_dates` | Explicit dates (if `fixed_dates`) | |
| `output_dir` | Output directory | `output` |
| `summary_file` | Summary CSV filename | `assets_1.csv` |

---

### 2. Recommendations

Supported models: `rfr`, `lgbm`, `tabpfn`. `rfr`/`lgbm` use plain, untuned defaults (`RandomForestRegressor(n_estimators=n)` / `LGBMRegressor(n_estimators=n)`, everything else left at library defaults) unless you pass `tuned` (see [Hyperparameter Tuning](#2b-hyperparameter-tuning-optuna) below).

```bash
python3 run_recommendation.py FAR-Trans-Data results rfr
```

Pass `n_estimators` and/or `kpi_type` directly:

```bash
python3 run_recommendation.py FAR-Trans-Data results rfr 100 short
```

`tabpfn` is internal-only and takes a `kpi_type` parameter and, optionally, a sample fraction — no `n_estimators`, and no `tuned` mode.

```bash
python3 run_recommendation.py FAR-Trans-Data results tabpfn
```
Pass a fraction in `(0, 1]` as an extra model parameter to cap the widnow size — sampled proportionally per asset (min. 1 row/asset), so every asset stays represented. 

```bash
python3 run_recommendation.py FAR-Trans-Data results tabpfn full_short 0.25
```

**Internal vs. external KPI generation:**

- **Internal (default)** — RFR/LGBM generate technical indicators on the fly, per training window, directly from raw price windows. No precomputed file needed; this is what `RFRKPIModel`/`LGBMKPIModel` do.
- **External** — technical indicators are precomputed once for the whole dataset into `<output_dir>/kpis.csv` (computed on first run, reused on later runs) and a plain `RandomForestRegressor`/`LGBMRegressor` trains directly on those columns. Pass `external` as an extra model parameter:

  ```bash
  python3 run_recommendation.py FAR-Trans-Data results rfr 100 full_short external
  ```

  Note: the external path doesn't set `random_state`, so unlike the internal path (seeded, reproducible) its results vary between runs.

**Run a single time window directly:**

```bash
python3 recommendation.py FAR-Trans-Data prices range 2019-08-01 2021-02-26 28 13 results 6 rfr
```


---

### 2b. Hyperparameter Tuning (Optuna)

Search RFR/LGBM hyperparameters with Optuna against several **expanding-window calibration folds**, then save the config that's stable across those folds for the `tuned` model parameter to pick up.

```bash
python3 algorithms/tune_hyperparams.py FAR-Trans-Data rfr --n-trials 20
python3 algorithms/tune_hyperparams.py FAR-Trans-Data lgbm --n-trials 20
```

- Objective: `mean(fold_scores) - robustness_lambda * std(fold_scores)`, maximized, where each fold's score is `monthly_prof@10` (ROI) on that fold's own held-out validation window.
- Folds are expanding windows whose validation periods all end at or before `2019-08-01`. This keeps every calibration fold strictly separate from every window whose result gets reported, so the tuning process can never leak into a "test" result.
- The calibration horizon (`--calibration-months`, default `3`) is deliberately shorter than the real deployment horizon (6 months, `DEPLOYMENT_MONTHS`) because this dataset's pre-`2019-08-01` history isn't long enough to fit one 6-month-horizon fold. 

Apply the saved best params across all windows in both experiments:

```bash
python3 run_recommendation.py FAR-Trans-Data results rfr tuned
python3 run_recommendation.py FAR-Trans-Data results lgbm tuned
```

Requires the best-params JSON above to already exist. Produces `rfr_tuned_full_short_internal_kpis` / `lgbm_tuned_full_short_internal_kpis` results alongside (not overwriting) the untuned baseline runs.

---

### 3. Compute Average Metrics

```bash
python3 process_results.py model
```

---

### 4. Generate Counterfactuals

By default, runs the last window of each experiment (exp1: `2020-08-28`, exp2: `2021-11-23`). Training/testing CSVs and output paths are auto-derived from the model pickle filename.

```bash
python3 generate_counterfactuals.py
```

Run for a specific window:

```bash
python3 generate_counterfactuals.py \
  --model-pkl artifacts_for_counterfactuals/rfr_n-100_kpi-full_short_internal_kpis/profitability_recommendation_pipeline_2020-08-28_00-00-00_rfr_n-100_kpi-full_short_internal_kpis.pkl
```

Reproducibility: the search is seeded per query (`--seed`, default 42), but that only guarantees identical results with `--n-jobs 1` — DiCE's genetic/random explainers draw from the global `random`/`np.random` state, so concurrent worker threads interleave draws unpredictably regardless of seeding.

Re-running without `--resume` on an output directory that already has results refuses to proceed (to avoid silently overwriting them) — pass `--resume` to continue, or clear the old files first.

---

### 5. Analyse Counterfactuals

Sort each counterfactual file by `query_index` (overwrites in place):

```bash
python3 process_results.py cf --sort
```

Aggregate comparison across all assets (metric distributions + factual vs CF scatter):

```bash
python3 process_results.py cf
```

Output: `cf_summary_all_assets.png` and `cf_scatter_all_assets.png` saved to `stats/plots/cf/`.

Plot the factual vs CF price window for a specific asset and query:

```bash
python3 process_results.py cf --asset-id <ASSET_ID> --query-index <N>
```

---

### 6. Membership Inference Attacks

The valid test-period rows are split 50/50 (stratified per asset): one half is audited
as non-members, the other half is held out purely as the population reference used by
the population attack. Members are then downsampled to match the non-member count, so
both classes are balanced. `loss.py` and `population_attack.py` audit the
same member/non-member rows.

**LOSS attack** — a sample is predicted "member" when its loss is unusually low
(Yeom et al.):

```bash
python3 membership_inference/loss.py --model rfr_n-100_kpi-full_short_internal_kpis --dates 2020-08-28,2021-11-23
```

**Population attack** — each audited sample is scored by where its loss falls within
the population loss distribution:

```bash
python3 membership_inference/population_attack.py --model rfr_n-100_kpi-full_short_internal_kpis
```

Output per date: `member_scores.csv`, `nonmember_scores.csv` (plus `population_scores.csv`
for the population attack), and `metrics.json` (ROC-AUC, accuracy at the best threshold,
attack advantage).

Each run also saves plots to `stats/membership_inference/{loss_attack,population_attack}/<model>/`:
an ROC curve (linear + log-log) and a loss-distribution histogram per date, a
percentile-rank histogram per date for the population attack, and (when more than one
date is run) an AUC/attack-advantage trend plot across dates.
