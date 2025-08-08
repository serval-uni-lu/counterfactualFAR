#!/usr/bin/python
import os
import sys

__author__ = "Javier Sanz-Cruzado (javier.sanz-cruzadopuig@glasgow.ac.uk)"

if __name__ == "__main__":
    """
    Script for executing the experiments in the FAR-Trans paper, except for the hybrid models.
    For the experiments with hybrid models, please, see run_hybrid_experiments_script.py.
    
    This program takes the FAR-Trans dataset and runs a series of recommender systems over
    the data (basic models, profitability prediction and collaborative filtering methods).
    Then, it evaluates them over a set of 61 dates.
    
    Input:
    - Dataset path: the path on which the FAR-Trans dataset is stored. We assume names have not been changed.
    - Output directory: the path on which the results will be stored. A directory for every model will be created.
    
    Output:
    - A directory for every recommendation algorithm.
    - For every tested date, three files will be created:
        - File 1: A recommendation file. titled "algorithm_date_recs.csv".
                Format: col_user \t col_item \t col_rating (sorted by user in ascending order, rating in descending)
        - File 2: An evaluation file, titled "algorithm_date_metrics.csv"
                Format: metric \t value
        - File 3: A customer evaluation file, where the metrics for every customer are detailed:
                Format: col_user \t metric1 \t metric2 ... \t metricN
    """

    if len(sys.argv) < 3:
        sys.stderr.write("ERROR: Invalid arguments")
        sys.stderr.write("\tdataset_path: route to the dataset.")
        sys.stderr.write("\toutput_dir: directory on which to store the results.")

    dataset_path = sys.argv[1]
    output_directory = sys.argv[2]

    # Obtain the routes for the interaction and time series files.
    interactions_file = os.path.join(dataset_path, "transactions.csv")
    time_series = os.path.join(dataset_path, "close_prices.csv")

    # Execute the algorithms:
    dates = [("2019-08-01", "2021-02-26", "28", "13", "6", output_directory),
            ("2020-09-14", "2022-05-23", "31", "13", "6", output_directory)]

    # Only run the RFR model
rfr_config = ("rfr", "rfr", "20", "full_short")

for date in dates:
    print("Starting", rfr_config[0], "for time horizon of", date[4], "month(s)")

    directory = os.path.join(date[5], rfr_config[1])
    os.makedirs(directory, exist_ok=True)

    # Build command
    exec_code = [
        "python3", "./recommendation.py",
        interactions_file,
        time_series,
        "range",         # date_format choice
        date[0],         # min_date
        date[1],         # max_date
        date[2],         # num_splits
        date[3],         # num_future
        directory,       # output_dir
        date[4],         # months
        rfr_config[0],   # model identifier
    ]

    # Add model parameters
    exec_code.extend(rfr_config[2:])  # -> ["20", "full_short"]

    # Convert list to string
    final_cmd = " ".join(exec_code)

    print("Executing:", final_cmd)
    if os.system(final_cmd) != 0:
        sys.exit(f"Error when executing {rfr_config[0]} for date {date}")