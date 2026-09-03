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

Supported models: `rfr`, `lgbm`. Both use plain, untuned defaults (`RandomForestRegressor(n_estimators=n)` / `LGBMRegressor(n_estimators=n)`, everything else left at library defaults).

```bash
python3 run_recommendation.py FAR-Trans-Data results rfr
```

Pass `n_estimators` and/or `kpi_type` directly:

```bash
python3 run_recommendation.py FAR-Trans-Data results rfr 100 full_short
```

KPI generation can be _internal_ (default) or _external_ (precomputed).

**Run a single time window directly:**

```bash
python3 recommendation.py FAR-Trans-Data prices range 2019-08-01 2021-02-26 28 13 results 6 rfr
```

---

### 3. Compute Average Metrics

```bash
python3 process_results.py model
```

---

### 4. Generate Counterfactuals

By default, runs the last window of each experiment (exp1: `2020-08-28`, exp2: `2021-11-23`). Training/testing CSVs and output paths are auto-derived from the model pickle filename.

```bash
python3 generate_rfr_counterfactuals_pkl.py
```

Run for a specific window:

```bash
python3 generate_rfr_counterfactuals_pkl.py \
  --model-pkl artifacts_for_counterfactuals/rfr_n-100_kpi-full_short_internal_kpis/profitability_recommendation_pipeline_2020-08-28_00-00-00_rfr_n-100_kpi-full_short_internal_kpis.pkl
```

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
