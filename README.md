
pipenv run pip install .

Create venv with python 3.9
pipenv --python /usr/bin/python3

Code lines that were changed are tagged with "# CHANGED"

python3 run_dataset_analysis.py FAR-Trans-Data output

positional:
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


For recommendations:
python3 run_recommendation.py Far-Trans-Data results
python3 recommendation.py Far-Trans-Data prices range 2019-08-01 2021-02-26 28 13 results 6 rfr 

delta smoothed kpi_type assets_time compute_metrics repetitions only_test_customers min_prices customer_file
