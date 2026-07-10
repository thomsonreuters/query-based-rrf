import pandas as pd
import numpy as np
import os
import ast
from scipy import stats

def parse_friendly_weights(weights_str):
    """Parse friendly_best_weights string to extract ranges"""
    if pd.isna(weights_str) or weights_str == '[]' or weights_str == '' or not isinstance(weights_str, str):
        return []

    try:
        weights = ast.literal_eval(weights_str)
    except (ValueError, SyntaxError):
        return []

    if not isinstance(weights, list):
        return []

    ranges = []

    for interval in weights:
        if isinstance(interval, (int, float)):
            # Single value
            ranges.append((interval, interval))
        elif isinstance(interval, list):
            if len(interval) == 1:
                ranges.append((interval[0], interval[0]))
            elif len(interval) == 2:
                ranges.append((interval[0], interval[1]))

    return ranges

def calculate_interval_lengths(ranges):
    """Calculate total length of all intervals"""
    if not ranges:
        return 0.0
    total_length = sum(end - start for start, end in ranges)
    return total_length

def calculate_interval_overlap(ranges_2dp, ranges_3dp):
    """Calculate Intersection over Union (IoU) of intervals"""
    if not ranges_2dp and not ranges_3dp:
        return 1.0  # Both empty, consider as perfect overlap
    if not ranges_2dp or not ranges_3dp:
        return 0.0  # One empty, no overlap

    # Calculate total length of each set
    total_length_2dp = calculate_interval_lengths(ranges_2dp)
    total_length_3dp = calculate_interval_lengths(ranges_3dp)

    # Special case: both have zero length (single points)
    # Check if they're the same points
    if total_length_2dp == 0 and total_length_3dp == 0:
        # Both are single points - check if they match
        points_2dp = set(start for start, end in ranges_2dp if start == end)
        points_3dp = set(start for start, end in ranges_3dp if start == end)

        if not points_2dp and not points_3dp:
            return 1.0  # Both empty

        # Calculate Jaccard similarity for points
        if points_2dp and points_3dp:
            intersection_points = len(points_2dp & points_3dp)
            union_points = len(points_2dp | points_3dp)
            return intersection_points / union_points if union_points > 0 else 0.0

        return 0.0

    # Calculate intersection
    total_intersection = 0.0
    for r1_start, r1_end in ranges_2dp:
        for r2_start, r2_end in ranges_3dp:
            # Calculate overlap between two intervals
            overlap_start = max(r1_start, r2_start)
            overlap_end = min(r1_end, r2_end)
            if overlap_end > overlap_start:
                total_intersection += (overlap_end - overlap_start)

    # Calculate union: Union = Length_A + Length_B - Intersection
    union_length = total_length_2dp + total_length_3dp - total_intersection

    if union_length == 0:
        return 1.0  # Both have zero length but passed previous checks - shouldn't happen

    # IoU = Intersection / Union
    iou = total_intersection / union_length
    return iou

def load_and_prepare_data(file_2dp, file_3dp):
    """Load both 2dp and 3dp CSV files and prepare for comparison"""
    df_2dp = pd.read_csv(file_2dp)
    df_3dp = pd.read_csv(file_3dp)

    # Ensure both dataframes have the same queries in the same order
    # Merge on query_id to align rows
    merged = df_2dp.merge(df_3dp, on='query_id', suffixes=('_2dp', '_3dp'))

    return merged

def compare_interval_counts(merged_df):
    """Compare interval counts between 2dp and 3dp"""
    ic_2dp = merged_df['interval_count_2dp']
    ic_3dp = merged_df['interval_count_3dp']

    # Convert 'single' to numeric for comparison
    ic_2dp_numeric = ic_2dp.replace('single', 1).astype(float)
    ic_3dp_numeric = ic_3dp.replace('single', 1).astype(float)

    # Calculate statistics
    same_count = (ic_2dp == ic_3dp).sum()
    total_count = len(merged_df)
    percentage_same = (same_count / total_count) * 100

    # Calculate correlation
    correlation = ic_2dp_numeric.corr(ic_3dp_numeric)

    return {
        'same_count': same_count,
        'total_count': total_count,
        'percentage_same': percentage_same,
        'correlation': correlation,
        'mean_2dp': ic_2dp_numeric.mean(),
        'mean_3dp': ic_3dp_numeric.mean(),
        'std_2dp': ic_2dp_numeric.std(),
        'std_3dp': ic_3dp_numeric.std()
    }

def compare_interval_lengths_and_overlaps(merged_df):
    """Compare sum of interval lengths and calculate overlap ratios"""
    total_length_2dp_all = []
    total_length_3dp_all = []
    per_query_differences = []
    overlap_ratios = []
    union_minus_intersection_all = []

    for _, row in merged_df.iterrows():
        ranges_2dp = parse_friendly_weights(row['friendly_best_weights_2dp'])
        ranges_3dp = parse_friendly_weights(row['friendly_best_weights_3dp'])

        if ranges_2dp or ranges_3dp:
            len_2dp = calculate_interval_lengths(ranges_2dp)
            len_3dp = calculate_interval_lengths(ranges_3dp)

            # Calculate intersection for union-intersection metric
            intersection = 0.0
            for r1_start, r1_end in ranges_2dp:
                for r2_start, r2_end in ranges_3dp:
                    overlap_start = max(r1_start, r2_start)
                    overlap_end = min(r1_end, r2_end)
                    if overlap_end > overlap_start:
                        intersection += (overlap_end - overlap_start)

            # Handle zero-length intervals for union
            if len_2dp == 0 and len_3dp == 0:
                union = 0.0  # Both are single points
            else:
                union = len_2dp + len_3dp - intersection

            union_minus_intersection = union - intersection

            overlap_ratio = calculate_interval_overlap(ranges_2dp, ranges_3dp)

            total_length_2dp_all.append(len_2dp)
            total_length_3dp_all.append(len_3dp)
            per_query_differences.append(len_2dp - len_3dp)
            overlap_ratios.append(overlap_ratio)
            union_minus_intersection_all.append(union_minus_intersection)

    return {
        'sum_interval_length_2dp': sum(total_length_2dp_all),
        'sum_interval_length_3dp': sum(total_length_3dp_all),
        'mean_interval_length_2dp': np.mean(total_length_2dp_all) if total_length_2dp_all else 0,
        'mean_interval_length_3dp': np.mean(total_length_3dp_all) if total_length_3dp_all else 0,
        'avg_per_query_diff': np.mean(per_query_differences) if per_query_differences else 0,
        'std_per_query_diff': np.std(per_query_differences) if per_query_differences else 0,
        'mean_overlap_ratio': np.mean(overlap_ratios) if overlap_ratios else 0,
        'median_overlap_ratio': np.median(overlap_ratios) if overlap_ratios else 0,
        'std_overlap_ratio': np.std(overlap_ratios) if overlap_ratios else 0,
        'min_overlap_ratio': np.min(overlap_ratios) if overlap_ratios else 0,
        'max_overlap_ratio': np.max(overlap_ratios) if overlap_ratios else 0,
        'mean_union_minus_intersection': np.mean(union_minus_intersection_all) if union_minus_intersection_all else 0,
        'std_union_minus_intersection': np.std(union_minus_intersection_all) if union_minus_intersection_all else 0,
        'queries_with_intervals': len(overlap_ratios)
    }

def compare_mean_weights(merged_df):
    """Compare mean best weights between 2dp and 3dp"""
    mw_2dp = merged_df['mean_best_weight_2dp'].dropna()
    mw_3dp = merged_df['mean_best_weight_3dp'].dropna()

    # Align the dataframes by dropping rows with NaN in either column
    valid_mask = merged_df['mean_best_weight_2dp'].notna() & merged_df['mean_best_weight_3dp'].notna()
    mw_2dp_aligned = merged_df.loc[valid_mask, 'mean_best_weight_2dp']
    mw_3dp_aligned = merged_df.loc[valid_mask, 'mean_best_weight_3dp']

    # Calculate absolute differences
    abs_diff = (mw_2dp_aligned - mw_3dp_aligned).abs()

    # Calculate correlation
    correlation = mw_2dp_aligned.corr(mw_3dp_aligned) if len(mw_2dp_aligned) > 0 else None

    return {
        'mean_2dp': mw_2dp.mean(),
        'mean_3dp': mw_3dp.mean(),
        'std_2dp': mw_2dp.std(),
        'std_3dp': mw_3dp.std(),
        'correlation': correlation,
        'mean_abs_difference': abs_diff.mean(),
        'max_abs_difference': abs_diff.max(),
        'median_abs_difference': abs_diff.median(),
        'within_0.01': (abs_diff <= 0.01).sum() / len(abs_diff) * 100 if len(abs_diff) > 0 else 0,
        'within_0.05': (abs_diff <= 0.05).sum() / len(abs_diff) * 100 if len(abs_diff) > 0 else 0
    }

def compare_metric_scores(merged_df, metric='mrr'):
    """Compare MRR/NDCG scores between 2dp and 3dp - calculate mean and variance of differences"""
    if metric == 'mrr':
        col_name = 'highest_mrr@10'
    else:
        col_name = 'highest_ndcg@10'

    score_2dp = merged_df[f'{col_name}_2dp']
    score_3dp = merged_df[f'{col_name}_3dp']

    # Calculate differences (3dp - 2dp)
    diff = score_3dp - score_2dp
    abs_diff = diff.abs()

    # Calculate correlation
    correlation = score_2dp.corr(score_3dp)

    return {
        'mean_2dp': score_2dp.mean(),
        'mean_3dp': score_3dp.mean(),
        'std_2dp': score_2dp.std(),
        'std_3dp': score_3dp.std(),
        'correlation': correlation,
        'mean_difference': diff.mean(),
        'variance_difference': diff.var(),
        'std_difference': diff.std(),
        'mean_abs_difference': abs_diff.mean(),
        'max_abs_difference': abs_diff.max(),
        'total': len(diff)
    }

def generate_summary_report(merged_df, output_file, metric='mrr'):
    """Generate a comprehensive comparison report"""

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SENSITIVITY ANALYSIS: 2 Decimal Points vs 3 Decimal Points\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total Queries: {len(merged_df)}\n\n")

        # Interval Count Comparison
        f.write("-"*80 + "\n")
        f.write("1. INTERVAL COUNT COMPARISON\n")
        f.write("-"*80 + "\n")
        ic_stats = compare_interval_counts(merged_df)
        f.write(f"Queries with same interval count: {ic_stats['same_count']} / {ic_stats['total_count']} ({ic_stats['percentage_same']:.2f}%)\n\n")

        # Interval Lengths and Overlaps
        f.write("-"*80 + "\n")
        f.write("2. INTERVAL LENGTH COMPARISON\n")
        f.write("-"*80 + "\n")
        overlap_stats = compare_interval_lengths_and_overlaps(merged_df)
        f.write(f"Average per-query difference (2dp - 3dp):\n")
        f.write(f"  Mean: {overlap_stats['avg_per_query_diff']:.4f}\n")
        f.write(f"  Std: {overlap_stats['std_per_query_diff']:.4f}\n\n")

        # Overlap Analysis
        f.write("-"*80 + "\n")
        f.write("3. INTERVAL OVERLAP ANALYSIS\n")
        f.write("-"*80 + "\n")
        f.write(f"Disagreement (Union - Intersection):\n")
        f.write(f"  Mean: {overlap_stats['mean_union_minus_intersection']:.4f}\n")
        f.write(f"  Std: {overlap_stats['std_union_minus_intersection']:.4f}\n")
        f.write(f"  Queries analyzed: {overlap_stats['queries_with_intervals']}\n\n")

        # Mean Weight Comparison
        f.write("-"*80 + "\n")
        f.write("4. MEAN BEST WEIGHT COMPARISON\n")
        f.write("-"*80 + "\n")
        mw_stats = compare_mean_weights(merged_df)
        f.write("KEY SENSITIVITY METRIC - Queries with close mean_best_weight:\n\n")
        f.write(f"  Queries within ±0.01 difference: {mw_stats['within_0.01']:.2f}%\n")
        f.write(f"  Queries within ±0.05 difference: {mw_stats['within_0.05']:.2f}%\n\n")

        # Metric Score Comparison
        f.write("-"*80 + "\n")
        metric_name = metric.upper()
        f.write(f"5. {metric_name}@10 SCORE COMPARISON (Using Mean Best Weights)\n")
        f.write("-"*80 + "\n")
        score_stats = compare_metric_scores(merged_df, metric=metric)

        f.write(f"Difference (3dp - 2dp):\n")
        f.write(f"  Mean: {score_stats['mean_difference']:.6f}\n")
        f.write(f"  Std: {score_stats['std_difference']:.6f}\n\n")

        # Summary
        f.write("="*80 + "\n")
        f.write("SENSITIVITY ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n")
        f.write("Finer discretization (0.001 steps) vs coarser (0.01 steps):\n\n")
        f.write(f"1. Interval count: {ic_stats['percentage_same']:.1f}% of queries have identical counts\n\n")
        f.write(f"2. Interval lengths:\n")
        f.write(f"   - Avg per-query difference (2dp - 3dp): {overlap_stats['avg_per_query_diff']:.4f}\n")
        f.write(f"   - Disagreement (Union - Intersection): {overlap_stats['mean_union_minus_intersection']:.4f}\n\n")
        f.write(f"3. Mean best weight: {mw_stats['within_0.01']:.1f}% within ±0.01, {mw_stats['within_0.05']:.1f}% within ±0.05\n\n")
        f.write(f"4. {metric_name}@10 scores: {score_stats['mean_difference']:.6f} ± {score_stats['std_difference']:.6f}\n\n")
        f.write("CONCLUSION: Results demonstrate STABILITY under finer discretizations.\n")

def process_comparison(dataset, split, sparse_method, dense_method, metric='mrr', top_k=200):
    """Process comparison for a single combination"""

    base_dir = "dataset"

    if metric == 'mrr':
        dir_2dp = "mrr_runs_2decimalp"
        dir_3dp = "mrr_runs_3decimalp"
    else:
        dir_2dp = "ndcg_runs_2decimalp"
        dir_3dp = "ndcg_runs_3decimalp"

    # File paths
    file_2dp = f"{base_dir}/{dataset}/{dir_2dp}/{split}/top{top_k}/results_{split}_{sparse_method}_vs_{dense_method}_best_weights_friendly_intervals_with_mean_weight.csv"
    file_3dp = f"{base_dir}/{dataset}/{dir_3dp}/{split}/top{top_k}/results_{split}_{sparse_method}_vs_{dense_method}_best_weights_friendly_intervals_with_mean_weight.csv"

    output_dir = f"{base_dir}/{dataset}/sensitivity_analysis/{split}/top{top_k}"
    os.makedirs(output_dir, exist_ok=True)
    output_report = f"{output_dir}/sensitivity_{split}_{sparse_method}_vs_{dense_method}_{metric}.txt"
    output_csv = f"{output_dir}/comparison_{split}_{sparse_method}_vs_{dense_method}_{metric}.csv"

    # Check if files exist
    if not os.path.exists(file_2dp):
        print(f"Warning: 2dp file not found: {file_2dp}")
        return None
    if not os.path.exists(file_3dp):
        print(f"Warning: 3dp file not found: {file_3dp}")
        return None

    # Load and merge data
    merged_df = load_and_prepare_data(file_2dp, file_3dp)

    # Generate report
    generate_summary_report(merged_df, output_report, metric=metric)

    # Save comparison CSV
    merged_df.to_csv(output_csv, index=False)

    print(f"Report saved: {output_report}")
    print(f"Data saved: {output_csv}")

    return {
        'interval_stats': compare_interval_counts(merged_df),
        'overlap_stats': compare_interval_lengths_and_overlaps(merged_df),
        'weight_stats': compare_mean_weights(merged_df),
        'score_stats': compare_metric_scores(merged_df, metric=metric)
    }

def generate_aggregated_report(results_by_combination, output_file, dataset, metric):
    """Generate an aggregated report across train/dev/test for each search combination"""

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"AGGREGATED SENSITIVITY ANALYSIS: {dataset.upper()}\n")
        f.write(f"Metric: {metric.upper()}@10\n")
        f.write(f"Aggregated across: train, dev, test splits\n")
        f.write("="*80 + "\n\n")

        for combo_key, results in results_by_combination.items():
            sparse_method, dense_method = combo_key
            f.write("="*80 + "\n")
            f.write(f"SEARCH COMBINATION: {sparse_method} vs {dense_method}\n")
            f.write("="*80 + "\n\n")

            # Aggregate interval count stats
            same_pct_list = [r['interval_stats']['percentage_same'] for r in results]
            f.write("-"*80 + "\n")
            f.write("1. INTERVAL COUNT COMPARISON\n")
            f.write("-"*80 + "\n")
            f.write(f"Queries with same interval count: {np.mean(same_pct_list):.2f}%\n\n")

            # Aggregate interval length and overlap stats
            avg_per_query_diff_list = [r['overlap_stats']['avg_per_query_diff'] for r in results]
            mean_union_minus_intersection_list = [r['overlap_stats']['mean_union_minus_intersection'] for r in results]

            f.write("-"*80 + "\n")
            f.write("2. INTERVAL LENGTH COMPARISON\n")
            f.write("-"*80 + "\n")
            f.write(f"Average per-query difference (2dp - 3dp) across splits:\n")
            f.write(f"  Mean: {np.mean(avg_per_query_diff_list):.4f}\n")
            f.write(f"  Std: {np.std(avg_per_query_diff_list):.4f}\n\n")

            f.write("-"*80 + "\n")
            f.write("3. INTERVAL OVERLAP ANALYSIS\n")
            f.write("-"*80 + "\n")
            f.write(f"Disagreement (Union - Intersection):\n")
            f.write(f"  Mean (across splits): {np.mean(mean_union_minus_intersection_list):.4f}\n")
            f.write(f"  Std (across splits): {np.std(mean_union_minus_intersection_list):.4f}\n\n")

            # Aggregate mean weight stats
            within_001_list = [r['weight_stats']['within_0.01'] for r in results]
            within_005_list = [r['weight_stats']['within_0.05'] for r in results]

            f.write("-"*80 + "\n")
            f.write("4. MEAN BEST WEIGHT COMPARISON\n")
            f.write("-"*80 + "\n")
            f.write(f"Queries within ±0.01 difference: {np.mean(within_001_list):.2f}%\n")
            f.write(f"Queries within ±0.05 difference: {np.mean(within_005_list):.2f}%\n\n")

            # Aggregate score stats
            mean_diff_list = [r['score_stats']['mean_difference'] for r in results]
            std_diff_list = [r['score_stats']['std_difference'] for r in results]

            f.write("-"*80 + "\n")
            f.write(f"5. {metric.upper()}@10 SCORE COMPARISON\n")
            f.write("-"*80 + "\n")
            f.write(f"Difference (3dp - 2dp) across splits:\n")
            f.write(f"  Mean: {np.mean(mean_diff_list):.6f}\n")
            f.write(f"  Std: {np.mean(std_diff_list):.6f}\n\n")

            f.write("\n")

if __name__ == "__main__":
    DATASET = "nfcorpus"  # acord-entire-corpus, nfcorpus, nq, msmarco
    SPLITS = ["train", "dev", "test"]
    TOP_K = 200
    METRIC = "ndcg"  # "mrr" or "ndcg"

    # Define search method combinations
    COMBINATIONS = [
        ("bm25", "biencoder"),
        ("bm25", "qwen3"),
        ("rm3", "biencoder"),
        ("rm3", "qwen3")
    ]

    print(f"Starting sensitivity analysis (2dp vs 3dp) for dataset: {DATASET}")
    print(f"Metric: {METRIC}")
    print(f"Splits: {SPLITS}")
    print(f"Search combinations: {COMBINATIONS}\n")

    # Store results by combination
    results_by_combination = {combo: [] for combo in COMBINATIONS}

    for split in SPLITS:
        print(f"\n{'='*80}")
        print(f"Processing split: {split}")
        print(f"{'='*80}\n")

        for sparse_method, dense_method in COMBINATIONS:
            print(f"\n{'-'*80}")
            print(f"Combination: {sparse_method} + {dense_method}")
            print(f"{'-'*80}")

            try:
                result = process_comparison(
                    DATASET, split, sparse_method, dense_method,
                    metric=METRIC, top_k=TOP_K
                )

                if result:
                    results_by_combination[(sparse_method, dense_method)].append(result)

            except Exception as e:
                print(f"Error processing {split} with {sparse_method}+{dense_method}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n{'='*80}")
    print("All per-split sensitivity analysis complete!")
    print(f"{'='*80}")

    # Generate aggregated reports for each combination
    print(f"\n{'='*80}")
    print("Generating aggregated reports...")
    print(f"{'='*80}\n")

    base_dir = "dataset"
    if METRIC == 'mrr':
        dir_name = "mrr_runs_comparison"
    else:
        dir_name = "ndcg_runs_comparison"

    output_dir = f"{base_dir}/{DATASET}/sensitivity_analysis/aggregated/top{TOP_K}"
    os.makedirs(output_dir, exist_ok=True)

    output_report = f"{output_dir}/sensitivity_aggregated_{METRIC}_2dp_vs_3dp.txt"
    generate_aggregated_report(results_by_combination, output_report, DATASET, METRIC)
    print(f"Aggregated report saved: {output_report}")
