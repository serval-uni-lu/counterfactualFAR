### Counterfactuals in asset recommendations

This project builds on "[FAR-Trans: An Investment Dataset for Financial Asset Recommendation](https://github.com/JavierSanzCruza/far-trans)" project.

However, from all implemented algorithms, we only use one for our purpose (RF). All the code lines that were changed are tagged with "# CHANGED" for fast tracking.


**For installation**

Create venv with python 3.9

```
pipenv --python /usr/bin/python3
```

Or 

If your machine does not have python 3.9 run:

```
pipenv --python $(which python3)
```

Install all dependencies 
```
pipenv install
```

Initiate the environment
```
pipenv shell
```

### Running the code
Analyze the asset/customer profitability over time (<span style="color:coral">not needed for recommendation</span>):
```
python3 run_dataset_analysis.py FAR-Trans-Data output
```

Information about args:
1. interactions       → transactions.csv
2. time_series        → close_prices.csv
3. subcommand         → range or fixed_dates
   - if "range":
     - min_date       → 2019-08-01
     - max_date       → 2021-02-26
     - num_splits     → 28
     - num_future     → 13
   - if "fixed_dates":
     - split_dates    → ...
     - future_dates   → ...
4. output_dir         → output
5. summary_file       → assets_1.csv


**Obtain all recommendations**:

Three models are accepted: rfr, mlp and tabnet.

Specify the parameters or defaults will be used.

For rfr, can be used _internal_ or _external_ kpis generation. _internal_ follows the same approach as mlp/tabnet

```
python3 run_recommendation.py FAR-Trans-Data results rfr 100 internal
```

```
python3 run_recommendation.py FAR-Trans-Data results mlp 64 32 16
```

```
python3 run_recommendation.py FAR-Trans-Data results tabnet 32 32 3
```

Run specific time recommendation
```
python3 recommendation.py Far-Trans-Data prices range 2019-08-01 2021-02-26 28 13 results 6 rfr
```

**Compute the average metrics**
```
python3 process_results.py model
```

**Generate the counterfactuals**

By default, runs the last window of each experiment (exp1: 2020-08-28, exp2: 2021-11-23).
Training/testing CSVs and output paths are auto-derived from the pkl filename.
```
python3 generate_rfr_counterfactuals_pkl.py
```

Run a single window:
```
python3 generate_rfr_counterfactuals_pkl.py \
  --model-pkl artifacts_for_counterfactuals/rfr_n-100_kpi-full_short_internal_kpis/profitability_recommendation_pipeline_2020-08-28_00-00-00_rfr_n-100_kpi-full_short_internal_kpis.pkl
```

**Analyse counterfactuals**

First, sort each CF file by query_index, and overwrites them in place
```
python3 process_results.py cf --sort
```

Aggregate comparison across all assets (metric distributions + factual vs CF scatter):
```
python3 process_results.py cf
```
Saves `cf_summary_all_assets.png` and `cf_scatter_all_assets.png` to `stats/plots/cf/`.

Plot the factual vs CF price window for a specific asset and query:
```
python3 process_results.py cf --asset-id <ASSET_ID> --query-index <N>
```