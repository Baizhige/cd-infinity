import os
import numpy as np
import pandas as pd
from datetime import datetime

def append_results_to_csv(cache_prefix, record_val_metric, record_test_metric, file_path=os.path.join("record", "comparison_study.csv")):
    # Calculate mean and standard deviation
    val_mean_std = np.array([np.mean(record_val_metric, axis=1), np.std(record_val_metric, axis=1)]).T
    test_mean_std = np.array([np.mean(record_test_metric, axis=1), np.std(record_test_metric, axis=1)]).T

    # Formatting mean(std)
    formatted_val = [f"{mean:.5f}({std:.5f})" for mean, std in val_mean_std]
    formatted_test = [f"{mean:.5f}({std:.5f})" for mean, std in test_mean_std]

    # Create a DataFrame
    df = pd.DataFrame({
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Cache Prefix": cache_prefix,
        "Source Domain Val Accuracy": formatted_val[0],
        "Source Domain Test Accuracy": formatted_test[0],
        "Target Domain Val Accuracy": formatted_val[1],
        "Target Domain Test Accuracy": formatted_test[1]
    }, index=[0])

    # Append to CSV
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)