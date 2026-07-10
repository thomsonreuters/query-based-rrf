import csv
import ast
import os

def process_weights(weights_str):
    """
    Convert consecutive weights into intervals for better visualization.
    Groups weights that differ by 0.01 into ranges.
    """
    if not weights_str or weights_str == '[]':
        return weights_str
    
    
    # Parse the string representation of the list
    weights = ast.literal_eval(weights_str)
    
    # If single weight or empty, return as is
    if len(weights) <= 1:
        return weights_str
    
    # Sort weights to ensure proper ordering
    weights = sorted(weights)
    
    intervals = []
    start = weights[0]
    prev = weights[0]
    
    for i in range(1, len(weights)):
        current = weights[i]
        # If gap is larger than 0.01 (with small tolerance for floating point)
        if abs(current - prev) > 0.011:
            # Close current interval
            if start == prev:
                intervals.append([start])  # Single value interval
            else:
                intervals.append([start, prev])
            # Start new interval
            start = current
        prev = current
    
    # Add the last interval
    if start == prev:
        intervals.append([start])  # Single value interval
    else:
        intervals.append([start, prev])
    
    return str(intervals)
    

def count_intervals(weights_str):
    """
    Count the number of intervals in the weights.
    Returns: 0 for no weight, 'single' for single weight, number for interval groups
    """
    if not weights_str or weights_str == '[]':
        return 0
    
    weights = ast.literal_eval(weights_str)
    
    if len(weights) == 0:
        return 0
    elif len(weights) == 1:
        return 'single'
    
    # Sort weights to ensure proper ordering
    weights = sorted(weights)
    
    intervals = 1
    prev = weights[0]
    
    for i in range(1, len(weights)):
        current = weights[i]
        # If gap is larger than 0.01, it's a new interval
        if abs(current - prev) > 0.011:
            intervals += 1
        prev = current
    
    return intervals
    

def process_csv_file(input_file='input.csv', output_file='output.csv'):
    """
    Process the CSV file and transform best_weights column.
    """
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
    
        # Preserve original fieldnames and add new columns
        original_fieldnames = list(reader.fieldnames)
        fieldnames = original_fieldnames + ['friendly_best_weights', 'interval_count']
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                # Process the best_weights column
                if 'best_weights' in row:
                    original_weights = row['best_weights']
                    row['friendly_best_weights'] = process_weights(original_weights)
                    row['interval_count'] = count_intervals(original_weights)
                else:
                    row['interval_count'] = 0
                writer.writerow(row)

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

    print(f"Starting friendly interval processing for dataset: {DATASET}")
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
                    input_csv = f"{_base_data_dir}/{DATASET}/{run_dir}/{split}/top{TOP_K}/results_{split}_{sparse_method}_vs_{dense_method}_best_weights.csv"
                    output_csv = input_csv.replace(".csv", "_friendly_intervals.csv")

                    if not os.path.exists(input_csv):
                        print(f"Warning: Input file not found: {input_csv}")
                        continue

                    # Process the file
                    process_csv_file(input_csv, output_csv)
                    print(f"Processed: {output_csv}")

            except Exception as e:
                print(f"Error processing {split} with {sparse_method}+{dense_method}: {e}")
                continue

    print(f"\n{'='*80}")
    print("All friendly interval processing complete!")
    print(f"{'='*80}")