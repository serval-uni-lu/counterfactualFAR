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

Three models are accepted: rfr, mlp and tabnet

```
python3 run_recommendation.py FAR-Trans-Data results mlp
```

Run specific time recommendation
```
python3 recommendation.py Far-Trans-Data prices range 2019-08-01 2021-02-26 28 13 results 6 rfr
```

**Compute the average metrics**
```
python3 process_results.py
```