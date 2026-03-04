import pandas as pd
import ast
import numpy as np
import random
import os

def get_interval_info(weights_str):
    """
    Parses the weights string and determines the interval representation and count.
    Returns: (friendly_weights_list, interval_count_value)
    """
    if not isinstance(weights_str, str) or not weights_str or weights_str == '[]':
        return [], 0

    try:
        weights = ast.literal_eval(weights_str)
    except (ValueError, SyntaxError):
        return [], 0

    # Handle empty or single cases
    if len(weights) == 0:
        return [], 0
    if len(weights) == 1:
        return weights, 'single'

    # Sort weights
    weights = sorted(weights)
    
    intervals = []
    start = weights[0]
    prev = weights[0]

    # Group consecutive weights (differing by approx 0.01)
    for i in range(1, len(weights)):
        current = weights[i]
        # Using 0.011 to handle floating point tolerance
        if abs(current - prev) > 0.011:
            if start == prev:
                intervals.append([start])
            else:
                intervals.append([start, prev])
            start = current
        prev = current

    # Add the last interval
    if start == prev:
        intervals.append([start])
    else:
        intervals.append([start, prev])

    return intervals, len(intervals)

def calculate_mean_weight(friendly_weights, interval_count):
    """
    Calculates the mean best weight based on the intervals and count.
    """
    # Case 0: No weights
    if interval_count == 0:
        return None

    # Case 1: Single weight (interval_count is 'single')
    if interval_count == 'single':
        return round(friendly_weights[0], 2)

    # Case 2: Exactly one interval group (e.g., [[0.1, 0.15]])
    if interval_count == 1:
        interval = friendly_weights[0]
        
        # If it's a single point interval like [[0.03]]
        if len(interval) == 1:
            return round(interval[0], 2)
        
        start, end = interval
        
        # Special case: very small gap [0.11, 0.12] -> pick random
        if round(end - start, 2) == 0.01:
            return random.choice([start, end])
        
        # Calculate mean of range
        values = np.arange(start, end + 0.01, 0.01)
        return round(np.mean(values), 2)

    # Case 3: Multiple intervals - pick the one with the widest gap
    # interval_count is an integer > 1
    max_gap = 0
    widest_interval = None

    for interval in friendly_weights:
        if len(interval) > 1:
            gap = interval[1] - interval[0]
            if gap > max_gap:
                max_gap = gap
                widest_interval = interval

    # If all intervals are single points (e.g., [[0.5], [0.8]]), pick random
    if widest_interval is None:
        # Flatten the list of lists to pick a random value
        flat_weights = [item[0] for item in friendly_weights]
        return random.choice(flat_weights)

    start, end = widest_interval
    values = np.arange(start, end + 0.01, 0.01)
    return round(np.mean(values), 2)

def process_pipeline(input_csv, output_csv):
    print(f"Reading input: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # 1. Apply interval logic
    # We apply the function and expand the result into two columns
    print("Processing intervals...")
    interval_data = df['best_weights'].apply(get_interval_info)
    
    # Extract data into separate columns
    # 'friendly_best_weights' will hold the actual list object for calculation
    # We will convert it to string later for the CSV output
    df['temp_friendly_obj'] = interval_data.apply(lambda x: x[0])
    df['interval_count'] = interval_data.apply(lambda x: x[1])

    # 2. Calculate mean best weight
    print("Calculating mean weights...")
    df['mean_best_weight'] = df.apply(
        lambda row: calculate_mean_weight(row['temp_friendly_obj'], row['interval_count']), 
        axis=1
    )

    # 3. Filter out rows with no mean_best_weight
    initial_count = len(df)
    df = df.dropna(subset=['mean_best_weight'])
    dropped_count = initial_count - len(df)
    print(f"Dropped {dropped_count} rows with no valid weights.")

    # 4. Final Formatting
    # Convert the list object to the string representation user expects in CSV
    df['friendly_best_weights'] = df['temp_friendly_obj'].astype(str)
    
    # Calculate stats
    overall_avg = df['mean_best_weight'].mean()
    print(f"Overall Average mean_best_weight: {overall_avg:.4f}")

    # Reorder columns: Keep all original + new ones, but ensure best_weights is last
    cols = [c for c in df.columns if c not in ['best_weights', 'temp_friendly_obj']]
    cols.append('best_weights')
    
    # Save
    df[cols].to_csv(output_csv, index=False)
    print(f"Saved processed data to: {output_csv}")

import os

if __name__ == "__main__":
    DATASET = "msmarco"  # nq, msmarco, # nfcorpus Fixed dataset
    
    # Define your search space
    SPLITS = ["train"] #   , "dev", "train"
    SPARSE_METHODS = ["rm3"] # 
    DENSE_METHODS = ["biencoder"] # 
    EVAL_METHOD = "mrr" #mrr, ndcg

    for split in SPLITS:
        for sparse in SPARSE_METHODS:
            for dense in DENSE_METHODS:
                # 1. Update the input path dynamically for each iteration
                input_file = f"/extra/huaiyaom0/tr-intern/wrrf/dataset/{DATASET}/{EVAL_METHOD}_runs/{split}/top200/results_{split}_{sparse}_vs_{dense}_best_weights.csv"
                
                # 2. Check if the file actually exists before processing to avoid crashes
                if not os.path.exists(input_file):
                    print(f"Skipping: File not found -> {input_file}")
                    continue
                
                # 3. Generate output path automatically
                base, ext = os.path.splitext(input_file)
                output_file = f"{base}_final_mean{ext}"

                print(f"Processing: {sparse} vs {dense} ({split})")
                
                # 4. Run the pipeline
                process_pipeline(input_file, output_file)

    print("--- All processing tasks complete ---")