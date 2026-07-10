import pandas as pd
import ast
import random
import numpy as np
import os

def calculate_mean_best_weight(best_weights, interval_count):
    """Calculate mean best weight based on the rules provided"""
    
    # Parse the best_weights string to get the actual data structure
    if isinstance(best_weights, str):
        weights = ast.literal_eval(best_weights)
    else:
        weights = best_weights

    # Case 0: interval_count is '0' - return
    if interval_count == '0':
        return None  
    
    # Case 1: interval_count is 'single' - return the weight itself
    elif interval_count == 'single':
        return round(weights[0], 2)
    
    # Case 2: interval_count is 1 - calculate mean of the interval
    elif interval_count == '1':
        if len(weights) == 1 and len(weights[0]) == 2:
            # Single interval like [[0.0, 1.0]]
            start, end = weights[0]

            if round(end - start, 2) == 0.01: # for case [0.11, 0.12]
                value = random.choice(weights[0])
                return value

        else:
            # Handle case like [0.03] where it's just a single value
            return round(weights[0], 2)
        
        # Generate values from start to end with 0.01 increment
        values = np.arange(start, end + 0.01, 0.01)
        return round(np.mean(values), 2)
    
    # Case 3: interval_count > 1 - pick interval with widest gap
    else:
        max_gap = 0
        widest_interval = None
        
        for interval in weights:
            if len(interval) > 1:
                gap = interval[1] - interval[0]
                if gap > max_gap:
                    max_gap = gap
                    widest_interval = interval
        
        if widest_interval is None:  # all interval length == 1 or is the same, [[0.56], [0.64]]
            return random.choice(weights)[0]
    
        start, end = widest_interval
        values = np.arange(start, end + 0.01, 0.01)
        return round(np.mean(values), 2)

def calculate_overall_average(df):
    """Calculate the average of mean_best_weight column excluding None/NaN"""
    # .mean() in pandas automatically ignores NaN/None values
    avg = df['mean_best_weight'].mean()
    return avg

def process_single_file(input_csv, output_csv):
    """Process a single CSV file"""
    # Read the CSV file
    df = pd.read_csv(input_csv)

    # Calculate mean_best_weight for each row
    df['mean_best_weight'] = df.apply(
        lambda row: calculate_mean_best_weight(row['friendly_best_weights'], row['interval_count']),
        axis=1
    )

    # Calculate and print the overall average across all entries
    overall_avg = calculate_overall_average(df)
    print(f"Overall Average mean_best_weight: {overall_avg:.4f}")

    # Reorder columns to put 'best_weights' last
    cols = [col for col in df.columns if col != 'best_weights'] + ['best_weights']
    df = df[cols]

    # Save the result to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

    return overall_avg

if __name__ == "__main__":
    DATASET = "nfcorpus"  # acord-entire-corpus, nfcorpus, nq, msmarco
    SPLITS = ["train", "dev", "test"]
    TOP_K = 200
    RUN_DIR = ["ndcg_runs_3decimalp", "mrr_runs_3decimalp",
               "ndcg_runs_2decimalp", "mrr_runs_2decimalp" ]  # mrr_runs_3decimalp or mrr_runs_2decimalp

    # Define search method combinations
    COMBINATIONS = [
        ("bm25", "biencoder"),
        ("bm25", "qwen3"),
        ("rm3", "biencoder"),
        ("rm3", "qwen3")
    ]

    _base_data_dir = "dataset"

    print(f"Starting mean best weight calculation for dataset: {DATASET}")
    print(f"Splits: {SPLITS}")
    print(f"Search combinations: {COMBINATIONS}\n")

    for split in SPLITS:
        print(f"\n{'='*80}")
        print(f"Processing split: {split}")
        print(f"{'='*80}\n")

        for sparse_method, dense_method in COMBINATIONS:
            print(f"\n{'-'*80}")
            print(f"Combination: {sparse_method} + {dense_method}")
            print(f"{'-'*80}")

            try:
                for run_dir in RUN_DIR:
                    input_csv = f"{_base_data_dir}/{DATASET}/{run_dir}/{split}/top{TOP_K}/results_{split}_{sparse_method}_vs_{dense_method}_best_weights_friendly_intervals.csv"
                    output_csv = input_csv.replace('.csv', '_with_mean_weight.csv')

                    if not os.path.exists(input_csv):
                        print(f"Warning: Input file not found: {input_csv}")
                        continue

                    # Process the file
                    overall_avg = process_single_file(input_csv, output_csv)

            except Exception as e:
                print(f"Error processing {split} with {sparse_method}+{dense_method}: {e}")
                continue

    print(f"\n{'='*80}")
    print("All mean best weight calculations complete!")
    print(f"{'='*80}")