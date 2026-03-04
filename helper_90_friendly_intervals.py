import csv
import ast

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


input_csv = "/extra/huaiyaom0/tr-intern/wrrf/dataset/nq/mrr_runs/train/top200/results_train_bm25_vs_biencoder_best_weights"
output_csv = input_csv.replace(".csv", "_friendly_intervals.csv")

# Process the file
process_csv_file(input_csv, output_csv)