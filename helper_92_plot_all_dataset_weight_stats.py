import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plt_ticker
import seaborn as sns
import ast
import os
import sys

# Define a Logger class to redirect print statements to both console and file
class Logger(object):
    def __init__(self, filename='log_output.txt'):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Set up logging
sys.stdout = Logger()

def load_csv_data(file_path):
    # Load local file
    df = pd.read_csv(file_path)
    return df

def parse_weights(weights_str):
    """Parse weight string to list of floats"""
    if pd.isna(weights_str) or weights_str == '[]':
        return []
    try:
        return ast.literal_eval(weights_str)
    except:
        return []

def count_intervals(weights):
    """Count the number of intervals in a weight list"""
    if len(weights) <= 1:
        return len(weights)
    
    # Sort weights to ensure proper interval detection
    sorted_weights = sorted(weights)
    intervals = 1
    
    for i in range(1, len(sorted_weights)):
        # If difference is greater than 0.1, it's a new interval
        if sorted_weights[i] - sorted_weights[i-1] > 0.1:
            intervals += 1
    
    return intervals

def classify_weight_pattern(weights):
    """Classify the weight pattern into categories"""
    if len(weights) == 0:
        return "empty"
    elif len(weights) == 1:
        return "single"
    else:
        intervals = count_intervals(weights)
        if intervals == 1:
            return "multiple_single_interval"
        else:
            return "multiple_multiple_intervals"

def analyze_weights(df, config_name, total_entries_reference):
    """Analyze weight statistics using hardcoded totals for correct empty calculation"""
    
    # Detect which metric column is available
    if 'highest_mrr@10' in df.columns:
        metric_col = 'highest_mrr@10'
    elif 'highest_ndcg@10' in df.columns:
        metric_col = 'highest_ndcg@10'
    else:
        # Fallback detection
        possible_metrics = [c for c in df.columns if 'highest_' in c and ('mrr' in c or 'ndcg' in c)]
        if possible_metrics:
            metric_col = possible_metrics[0]
        else:
            # If we really can't find it, just warn and proceed (or raise error)
            print("Warning: standard metric columns not found.")
            metric_col = "unknown"
    
    print(f"Config: {config_name} - Using metric column: {metric_col}")
    
    # Parse best_weights column
    df['parsed_weights'] = df['best_weights'].apply(parse_weights)
    
    # Classify weight patterns
    df['weight_pattern'] = df['parsed_weights'].apply(classify_weight_pattern)
    df['num_intervals'] = df['parsed_weights'].apply(count_intervals)
    
    # Count PRESENT categories in the CSV
    # Note: 'empty' here would only be empty lists present in the CSV, 
    # but user mentioned filtered CSVs don't have empty rows.
    present_empty = len(df[df['weight_pattern'] == 'empty'])
    single_count = len(df[df['weight_pattern'] == 'single'])
    multiple_single_interval_count = len(df[df['weight_pattern'] == 'multiple_single_interval'])
    multiple_multiple_intervals_count = len(df[df['weight_pattern'] == 'multiple_multiple_intervals'])
    
    # Calculate TRUE empty count based on total_entries_reference
    # The CSV only contains queries that had valid weights (or patterns we analyzed).
    # The "missing" entries are the ones that had 0 metric or were filtered out.
    
    # Sum of analyzed valid patterns
    total_valid_analyzed = single_count + multiple_single_interval_count + multiple_multiple_intervals_count + present_empty
    
    # If the CSV has more entries than the reference (e.g. slight mismatch in source), cap empty at 0
    calculated_empty_count = max(0, total_entries_reference - total_valid_analyzed)
    
    # If present_empty > 0, we add it to calculated_empty_count (though user said they filtered them out)
    final_empty_count = calculated_empty_count + present_empty
    
    # Use total_entries_reference for percentages
    total_entries = total_entries_reference
    
    empty_pct = (final_empty_count / total_entries) * 100
    single_pct = (single_count / total_entries) * 100
    multiple_single_interval_pct = (multiple_single_interval_count / total_entries) * 100
    multiple_multiple_intervals_pct = (multiple_multiple_intervals_count / total_entries) * 100
    
    print(f"Total entries (Reference): {total_entries}")
    print(f"Entries in CSV: {len(df)}")
    print(f"1) Queries with empty best weights: {final_empty_count} ({empty_pct:.0f}%)")
    print(f"2) Queries with single best weight: {single_count} ({single_pct:.0f}%)")
    print(f"3) Queries with multiple best weights (single interval): {multiple_single_interval_count} ({multiple_single_interval_pct:.0f}%)")
    print(f"4) Queries with multiple best weights (multi interval): {multiple_multiple_intervals_count} ({multiple_multiple_intervals_pct:.0f}%)")
    
    # Additional detailed analysis for entries with weights
    entries_with_weights = df[df['parsed_weights'].apply(len) > 0].copy()
    
    if len(entries_with_weights) > 0:
        entries_with_weights['num_weights'] = entries_with_weights['parsed_weights'].apply(len)
        entries_with_weights['weight_mean'] = entries_with_weights['parsed_weights'].apply(
            lambda x: np.mean(x) if len(x) > 0 else np.nan
        )
        return entries_with_weights
    else:
        print("No entries with weights found!")
        return None

def create_number_of_weights_visualization(results_list):
    """Create visualization for number of weights per query in a 4x4 grid"""
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, result_data in enumerate(results_list):
        config_name = result_data['name']
        entries_with_weights = result_data['data']
        
        ax = axes[idx]
        
        if entries_with_weights is not None and len(entries_with_weights) > 0:
            weights, bins, patches = ax.hist(entries_with_weights['num_weights'], 
                                            bins=20, alpha=0.7, 
                                            edgecolor='black', 
                                            weights=np.ones(len(entries_with_weights['num_weights'])) / len(entries_with_weights['num_weights']) * 100)
            
            ax.set_title(f'{config_name}', fontsize=9)
            ax.set_xlabel('Number of Weights', fontsize=8)
            ax.set_ylabel('Percentage (%)', fontsize=8)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No data for\n{config_name}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_title(f'{config_name}', fontsize=9)
    
    # Hide any unused subplots
    for i in range(len(results_list), 16):
        axes[i].set_visible(False)
    
    plt.suptitle('Distribution of Number of Weights per Query', fontsize=16)
    plt.tight_layout()
    plt.savefig('number_of_weights_per_query.jpg', dpi=300, bbox_inches='tight')
    plt.close()

def create_mean_weight_visualization(results_list):
    """Create visualization for mean optimal weight per query in a 4x4 grid"""
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    
    # First pass: determine the maximum y-value across all datasets
    max_y = 0
    for result_data in results_list:
        entries_with_weights = result_data['data']
        if entries_with_weights is not None and len(entries_with_weights) > 0:
            weights_data = entries_with_weights['weight_mean'].dropna()
            hist_values, _ = np.histogram(weights_data, bins=20)
            hist_percentages = hist_values / len(weights_data) * 100
            max_y = max(max_y, hist_percentages.max())
    
    # Round up to nearest nice number for cleaner axis
    max_y = np.ceil(max_y / 5) * 5  # Round to nearest 5%
    
    for idx, result_data in enumerate(results_list):
        config_name = result_data['name']
        entries_with_weights = result_data['data']
        
        ax = axes[idx]
        
        if entries_with_weights is not None and len(entries_with_weights) > 0:
            weights, bins, patches = ax.hist(entries_with_weights['weight_mean'].dropna(),
                                            bins=20, alpha=0.7,
                                            edgecolor='black',
                                            weights=np.ones(len(entries_with_weights['weight_mean'].dropna())) / len(entries_with_weights['weight_mean'].dropna()) * 100)
            
            # Count of actual weights loaded
            count_str = f"{len(entries_with_weights):,}"
            ax.set_title(f'{config_name}\n({count_str} Queries)', fontsize=9)
            ax.set_xlabel('Mean Optimal Weight', fontsize=8)
            ax.set_ylabel('Fraction of Queries', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Set consistent y-axis limits and ticks
            ax.set_ylim(0, max_y)
            ax.set_yticks(np.arange(0, max_y + 1, 5))
            ax.yaxis.set_major_formatter(plt_ticker.FuncFormatter(lambda x, p: f'{int(x)}%'))
        else:
            ax.text(0.5, 0.5, f'No data for\n{config_name}',
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_title(f'{config_name}', fontsize=9)
            ax.set_ylim(0, max_y)
            ax.set_yticks(np.arange(0, max_y + 1, 5))
    
    # Hide any unused subplots
    for i in range(len(results_list), 16):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('mean_optimal_weight_per_query.jpg', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    datasets = ["msmarco", "nq", "acord-entire-corpus", "nfcorpus"]
    top_k = 200
    split = "train"
    
    SPARSE_METHODS = ["rm3", "bm25"] 
    DENSE_METHODS = ["biencoder", "qwen3"] 
    
    # Hardcoded total entries for percentage calculation
    dataset_totals = {
        "msmarco": 502939,
        "nq": 92468,
        "acord-entire-corpus": 51,
        "nfcorpus": 2590
    }
    
    # Hardcoded display names for plot titles
    dataset_display_names = {
        "msmarco": "MS MARCO Train",
        "nq": "NQ Train",
        "acord-entire-corpus": "ACORD Train",
        "nfcorpus": "NFCorpus Train"
    }

    # We will store results in a list to preserve the order for the 4x4 grid
    # Order: Iterate Datasets (Rows), then Sparse, then Dense (Columns)
    all_dataset_results = []
    
    print("Starting analysis... check log_output.txt for details.\n")
    
    for dataset in datasets:
        # Determine evaluation method based on dataset
        if dataset in ["msmarco", "nq"]:
            eval_method = "mrr"
        elif dataset in ["nfcorpus", "acord-entire-corpus"]:
            eval_method = "ndcg"
        else:
            print(f"Warning: Unknown metric logic for {dataset}, defaulting to mrr")
            eval_method = "mrr"
            
        # Get total entries count for this dataset
        total_entries_ref = dataset_totals.get(dataset, 0)
        display_name_base = dataset_display_names.get(dataset, dataset)

        for sparse in SPARSE_METHODS:
            for dense in DENSE_METHODS:
                
                # Construct file path
                file_name = f"results_{split}_{sparse}_vs_{dense}_best_weights_final_mean_with_text.csv"
                file_path = f"dataset/{dataset}/{eval_method}_runs/{split}/top{top_k}/{file_name}"
                
                # Friendly name for the plot title and logs
                config_name = f"{display_name_base}\n{sparse} vs {dense}"
                
                print(f"\nProcessing: {config_name}")
                print(f"Path: {file_path}")

                try:
                    df = load_csv_data(file_path)
                    print(f"Successfully loaded data. Shape: {df.shape}")
                    
                    # Analyze weights with reference total
                    entries_with_weights = analyze_weights(df, config_name, total_entries_ref)
                    
                    all_dataset_results.append({
                        "name": config_name,
                        "data": entries_with_weights
                    })
                    
                except Exception as e:
                    print(f"Error processing {config_name}: {e}")
                    # Append None to keep the grid alignment in the plot
                    all_dataset_results.append({
                        "name": config_name,
                        "data": None
                    })
    
    # Create separate visualizations
    if len(all_dataset_results) > 0:
        create_number_of_weights_visualization(all_dataset_results)
        create_mean_weight_visualization(all_dataset_results)
        
        print(f"\nVisualizations saved:")
        print(f"1. 'number_of_weights_per_query.jpg'")
        print(f"2. 'mean_optimal_weight_per_query.jpg'")
    else:
        print("No results to plot.")

if __name__ == "__main__":
    main()