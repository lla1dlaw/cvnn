"""
Author: Liam Laidlaw
Purpose: A script to process csv data from training complex valued neural networks.
"""

from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
import os
from pathlib import Path
import re


def pretty_print_summary(model_metrics: dict):
    """
    Prints a summarized, readable version of a model's metrics dictionary.

    For epoch-based metrics (lists), it prints a summary including the number
    of epochs and the best value (min for 'loss', max otherwise). For all
    other metrics, it prints the key-value pair.

    Args:
        model_metrics (dict): The consolidated metrics dictionary for a single model.
    """
    print("-" * 50)
    for metric_name, value in model_metrics.items():
        if isinstance(value, list):
            # This is an epoch-based metric
            num_epochs = len(value)
            if not value: # Handle empty lists
                summary = "0 epochs, no data"
            elif 'loss' in metric_name.lower():
                best_value = min(value)
                summary = f"{num_epochs} epochs, Min value: {best_value:.4f}"
            else:
                best_value = max(value)
                summary = f"{num_epochs} epochs, Max value: {best_value:.4f}"
            print(f"  - {metric_name:<25}: {summary}")
        else:
            # This is a single-value metric
            print(f"  - {metric_name:<25}: {value}")
    print("-" * 50)


def create_metrics_dictionary(file_path: str, index_col: str = None) -> dict:
    """
    Loads training results from a CSV and organizes them into a dictionary,
    indexed by a specified column.

    This function processes each row of the CSV and combines all associated
    metrics into a single dictionary for that row. Single-value metrics
    (e.g., learning_rate) are stored as values, and epoch-based metrics
    (e.g., train_acc) are stored as lists of values.

    Args:
        file_path (str): The path to the input CSV file.
        index_col (str, optional): The column to use as the primary key.
                                   If None, the first column of the CSV is used.
                                   Defaults to None.

    Returns:
        dict: A dictionary containing all processed metrics, indexed by model.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Successfully loaded '{file_path}' with {len(df)} rows.")
    except FileNotFoundError:
        print(f"❌ Error: The file '{file_path}' was not found.")
        return {}

    # If no index column is specified, use the first column as the default.
    if index_col is None:
        index_col = df.columns[0]
        print(f"ℹ️ No index column specified. Using the first column ('{index_col}') as the index.")
    # If an index column is specified, ensure it exists.
    elif index_col not in df.columns:
        print(f"❌ Error: The specified index column '{index_col}' was not found in the CSV.")
        print(f"   Available columns are: {list(df.columns)}")
        return {}

    # Define regex patterns to find epoch-based columns
    pattern1 = re.compile(r'(.+)_epoch_(\d+)')
    pattern2 = re.compile(r'epoch_(\d+)_(.+)')

    final_results = {}

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        epoch_cols_info = {}
        all_epoch_col_names = set()

        # First pass: identify all epoch-based columns and group them
        for col in row.index:
            col_str = str(col)
            match1 = pattern1.fullmatch(col_str)
            match2 = pattern2.fullmatch(col_str)
            
            if match1 or match2:
                match = match1 if match1 else match2
                metric_name = match.group(1) if match1 else match.group(2)
                epoch_num = int(match.group(2) if match1 else match.group(1))

                if metric_name not in epoch_cols_info:
                    epoch_cols_info[metric_name] = []
                epoch_cols_info[metric_name].append((epoch_num, col))
                all_epoch_col_names.add(col)
        
        # Process epoch metrics into a dictionary
        epoch_metrics = {}
        for metric_name, epoch_tuples in epoch_cols_info.items():
            sorted_tuples = sorted(epoch_tuples, key=lambda x: x[0])
            sorted_cols = [t[1] for t in sorted_tuples]
            epoch_metrics[metric_name] = row[sorted_cols].tolist()

        # Get single-value metrics (all columns that are not epoch-based)
        single_value_cols = [col for col in df.columns if col not in all_epoch_col_names]
        
        # Start with the dictionary of single-value metrics
        model_metrics = row[single_value_cols].to_dict()
        
        # Merge the dictionary of epoch-based metrics into it
        model_metrics.update(epoch_metrics)

        # Get the primary key for the outer dictionary
        model_key = row[index_col]

        # Assign the consolidated metrics dictionary to the model key
        final_results[model_key] = model_metrics
    
    print(f"✅ Finished processing. Created a metrics dictionary for {len(final_results)} models.")
    return final_results

# --- Example Usage ---
if __name__ == '__main__':
    csv_file_path = 'training_results.csv'
    
    # Create the metrics dictionary. The function will automatically use the
    # first column of your CSV as the index.
    metrics = create_metrics_dictionary(csv_file_path, index_col='name')

    # If you know the name of your model column, you can specify it like this:
    # metrics = create_metrics_dictionary(csv_file_path, index_col='your_model_column_name')

    if metrics:
        pprint(metrics)
        # Get the names of all processed models
        model_names = list(metrics.keys())
        
        # Select the first model in the dictionary to inspect
        model_to_inspect = model_names[0] 
        print(f"\n--- Metrics Summary for model: '{model_to_inspect}' ---")
        
        # Use the summary function for a clean, readable output
        pretty_print_summary(metrics[model_to_inspect])

        # You can still access the full data as before if needed
        print("\n--- Accessing full data for the same model ---")
        try:
            # Example: access a single-value metric
            lr = metrics[model_to_inspect].get('learning_rate', 'N/A')
            print(f"Learning Rate: {lr}")

            # Example: access an epoch-based metric
            train_acc_epoch_3 = metrics[model_to_inspect]['train_acc'][3]
            print(f"Training accuracy at epoch 3: {train_acc_epoch_3:.4f}")
        except (KeyError, IndexError) as e:
            print(f"Could not retrieve a specific metric. Reason: {e}")
