import math
from collections import defaultdict
import numpy as np
import os
import glob
import csv

def calculate_ndcg_at_k_single_query(query_results, query_qrels, k=10, use_trec_eval=True):
    """Calculate NDCG@k for a single query following the example function closely"""
    # Only consider top k
    results_top_k = query_results[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, (doc_id, rank) in enumerate(results_top_k):
        relevance = query_qrels.get(doc_id, 0)
        dcg += relevance / math.log2(i + 2)
    
    # Calculate IDCG (ideal DCG)
    ideal_relevances = sorted(query_qrels.values(), reverse=True)[:k]
    idcg = 0.0
    for i, relevance in enumerate(ideal_relevances):
        if use_trec_eval:
            idcg += relevance / math.log2(i + 2)
        else:
            idcg += (2**relevance - 1) / math.log2(i + 2)
    
    # Calculate NDCG
    if idcg > 0:
        return round(dcg / idcg, 4)
    else:
        return 0.0

def calculate_map_at_k_single_query(query_results, query_qrels, k=10):
    """Calculate MAP@k for a single query"""
    results_top_k = query_results[:k]
    
    relevant_count = 0
    precision_sum = 0.0
    
    for i, (doc_id, rank) in enumerate(results_top_k):
        if doc_id in query_qrels and query_qrels[doc_id] > 0:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i
    
    if relevant_count > 0:
        return round(precision_sum / relevant_count, 4)
    else:
        return 0.0

def calculate_mrr_at_k_single_query(query_results, query_qrels, k=10):
    """Calculate MRR@k for a single query using position-based ranking"""
    # query_results should already be sorted by rank
    # We use position in the ranked list (1-indexed) for MRR calculation
    
    for position, (doc_id, _) in enumerate(query_results[:k], 1):
        if doc_id in query_qrels and query_qrels[doc_id] > 0:
            return round(1.0 / position, 4)
    
    return 0.0  # No relevant document found in top k

def load_qrels(qrels_path):
    """Load relevance judgments from TSV file locally"""
    qrels = defaultdict(dict)
    
    with open(qrels_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query_id = parts[0]  # query-id
                doc_id = parts[1]    # corpus-id  
                score = int(parts[2]) # score
                qrels[query_id][doc_id] = score
    
    return qrels

def load_trec_results(trec_path):
    """Load TREC results and return ranked lists per query"""
    results = defaultdict(list)
    
    with open(trec_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                query_id = parts[0]
                doc_id = parts[2]
                rank = int(parts[3])
                results[query_id].append((doc_id, rank))
    
    # Sort by rank for each query
    for query_id in results:
        results[query_id].sort(key=lambda x: x[1])
    
    return results

def test_ndcg_calculation(trec_path, qrels_path, k=10):
    """Test NDCG@k calculation and return average NDCG@k"""
    
    print(f"Loading TREC results from: {trec_path}")
    print(f"Loading qrels from: {qrels_path}")
    
    # Load data
    trec_results = load_trec_results(trec_path)
    qrels = load_qrels(qrels_path)
    
    print(f"Loaded {len(trec_results)} queries from TREC file")
    print(f"Loaded {len(qrels)} queries from qrels file")
    
    # Calculate NDCG@k for each query
    ndcg_scores = []
    
    # Loop over QRELS, not trec_results
    for query_id in qrels:
        query_qrels = qrels[query_id]
        
        # If the query is in our results, calculate normally. If not, pass an empty list.
        query_results = trec_results.get(query_id, [])
        
        ndcg_score = calculate_ndcg_at_k_single_query(query_results, query_qrels, k)
        ndcg_scores.append(ndcg_score)
    
    # Calculate average NDCG@k
    if ndcg_scores:
        avg_ndcg = round(sum(ndcg_scores) / len(qrels), 3)
        return avg_ndcg
    else:
        print("No matching queries found!")
        return 0.0

def calculate_average_mrr(trec_file, qrels_file, k=10):
    """Calculate average MRR@k across all queries"""
    results = load_trec_results(trec_file)
    qrels = load_qrels(qrels_file)

    total_mrr = 0.0
    
    # Loop over QRELS
    for query_id in qrels:
        query_results = results.get(query_id, [])
        mrr = calculate_mrr_at_k_single_query(query_results, qrels[query_id], k)
        total_mrr += mrr

    avg_mrr = round(total_mrr / len(qrels), 3) 
    return avg_mrr

def calculate_average_map(trec_file, qrels_file, k=10):
    """Calculate average MAP@k across all queries"""
    results = load_trec_results(trec_file)
    qrels = load_qrels(qrels_file)
    
    total_map = 0.0
    
    # Loop over QRELS
    for query_id in qrels:
        query_results = results.get(query_id, [])
        map_score = calculate_map_at_k_single_query(query_results, qrels[query_id], k)
        total_map += map_score
    
    avg_map = round(total_map / len(qrels), 3)
    return avg_map

def bootstrap_confidence_interval(metric_scores, n_bootstrap=1000):
    """Simple bootstrap confidence interval calculation"""
    metric_scores = np.array(metric_scores)
    n_queries = len(metric_scores)
    
    # Generate bootstrap samples
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(metric_scores, size=n_queries, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    # Calculate 95% confidence interval
    lower_bound = np.percentile(bootstrap_means, 2.5)
    upper_bound = np.percentile(bootstrap_means, 97.5)
    mean_score = np.mean(metric_scores)
    
    return lower_bound, upper_bound, mean_score

def evaluate_with_bootstrap(trec_path, qrels_path):
    """Simple evaluation with bootstrap confidence intervals for both @5 and @10"""
    
    # Load data using your existing functions
    trec_results = load_trec_results(trec_path)
    qrels = load_qrels(qrels_path)
    
    # Calculate per-query scores for @5 and @10
    ndcg_scores_5, mrr_scores_5, map_scores_5 = [], [], []
    ndcg_scores_10, mrr_scores_10, map_scores_10 = [], [], []
    
    # Loop over QRELS to ensure we account for every query
    for query_id in qrels:
        query_qrels = qrels[query_id]
        
        # Fetch results, default to empty list if query is missing from TREC
        query_results = trec_results.get(query_id, [])
        
        # @5 metrics
        ndcg_scores_5.append(calculate_ndcg_at_k_single_query(query_results, query_qrels, k=5))
        mrr_scores_5.append(calculate_mrr_at_k_single_query(query_results, query_qrels, k=5))
        map_scores_5.append(calculate_map_at_k_single_query(query_results, query_qrels, k=5))
        
        # @10 metrics
        ndcg_scores_10.append(calculate_ndcg_at_k_single_query(query_results, query_qrels, k=10))
        mrr_scores_10.append(calculate_mrr_at_k_single_query(query_results, query_qrels, k=10))
        map_scores_10.append(calculate_map_at_k_single_query(query_results, query_qrels, k=10))
    

    # Calculate bootstrap confidence intervals
    ndcg_lower_5, ndcg_upper_5, ndcg_mean_5 = bootstrap_confidence_interval(ndcg_scores_5)
    mrr_lower_5, mrr_upper_5, mrr_mean_5 = bootstrap_confidence_interval(mrr_scores_5)
    map_lower_5, map_upper_5, map_mean_5 = bootstrap_confidence_interval(map_scores_5)
    ndcg_lower_10, ndcg_upper_10, ndcg_mean_10 = bootstrap_confidence_interval(ndcg_scores_10)
    mrr_lower_10, mrr_upper_10, mrr_mean_10 = bootstrap_confidence_interval(mrr_scores_10)
    map_lower_10, map_upper_10, map_mean_10 = bootstrap_confidence_interval(map_scores_10)
    
    return (ndcg_lower_5, ndcg_upper_5, ndcg_mean_5, mrr_lower_5, mrr_upper_5, mrr_mean_5, 
            map_lower_5, map_upper_5, map_mean_5, ndcg_lower_10, ndcg_upper_10, ndcg_mean_10, 
            mrr_lower_10, mrr_upper_10, mrr_mean_10, map_lower_10, map_upper_10, map_mean_10)

def process_single_dataset(dataset, split, sparse, dense, trec_path, qrels_path):
    """Process a single combination of dataset, split, and retrievers and return results"""
    print(f"\n{'='*60}")
    print(f"Processing: {dataset} ({split}) | Sparse: {sparse} | Dense: {dense}")
    print(f"File: {trec_path}")
    print(f"{'='*60}")
    
    # Calculate metrics for @5 and @10
    avg_ndcg_5 = test_ndcg_calculation(trec_path, qrels_path, k=5)
    avg_mrr_5 = calculate_average_mrr(trec_path, qrels_path, k=5)
    avg_map_5 = calculate_average_map(trec_path, qrels_path, k=5)
    avg_ndcg_10 = test_ndcg_calculation(trec_path, qrels_path, k=10)
    avg_mrr_10 = calculate_average_mrr(trec_path, qrels_path, k=10)
    avg_map_10 = calculate_average_map(trec_path, qrels_path, k=10)
    
    bootstrap_results = evaluate_with_bootstrap(trec_path, qrels_path)
    
    return {
        'dataset': dataset,
        'split': split,
        'sparse': sparse,
        'dense': dense,
        'avg_ndcg_5': avg_ndcg_5,
        'avg_mrr_5': avg_mrr_5,
        'avg_map_5': avg_map_5,
        'avg_ndcg_10': avg_ndcg_10,
        'avg_mrr_10': avg_mrr_10,
        'avg_map_10': avg_map_10,
        'bootstrap_ndcg_5': bootstrap_results[2],
        'bootstrap_mrr_5': bootstrap_results[5],
        'bootstrap_map_5': bootstrap_results[8],
        'bootstrap_ndcg_10': bootstrap_results[11],
        'bootstrap_mrr_10': bootstrap_results[14],
        'bootstrap_map_10': bootstrap_results[17],
        'ndcg_lower_5': bootstrap_results[0], 
        'ndcg_upper_5': bootstrap_results[1],
        'mrr_lower_5': bootstrap_results[3], 
        'mrr_upper_5': bootstrap_results[4],
        'map_lower_5': bootstrap_results[6],
        'map_upper_5': bootstrap_results[7],
        'ndcg_lower_10': bootstrap_results[9], 
        'ndcg_upper_10': bootstrap_results[10],
        'mrr_lower_10': bootstrap_results[12], 
        'mrr_upper_10': bootstrap_results[13],
        'map_lower_10': bootstrap_results[15],
        'map_upper_10': bootstrap_results[16]
    }

def process_multiple_datasets(datasets_config, folder):
    """
    Process multiple datasets, matching files via glob, and return consolidated sorted results.
    Saves output to metrics.txt and metrics.csv in the target folder.
    """
    all_results = []
    
    for dataset_info in datasets_config:
        if isinstance(dataset_info, dict):
            dataset = dataset_info['dataset']
            split = dataset_info['split']
        else:
            # Assume it's a tuple (dataset, split)
            dataset, split = dataset_info
            
        qrels_path = f"dataset/{dataset}/qrels/{split}.tsv"
        
        # Iterate the folder to find all combinations for this dataset/split
        search_pattern = os.path.join(folder, f"{dataset}_*_*_{split}*.trec")
        matching_files = glob.glob(search_pattern)
        
        if not matching_files:
            print(f"No TREC files found matching pattern: {search_pattern}")
            continue
            
        for trec_path in matching_files:
            basename = os.path.basename(trec_path).replace('.trec', '')
            
            sparse_retriever = "unknown"
            dense_retriever = "unknown"
            
            # The filename starts with the dataset name, so we remove it to get the rest
            prefix = f"{dataset}_"
            if basename.startswith(prefix):
                remainder = basename[len(prefix):] # e.g., "rm3_biencoder_test_057"
                parts = remainder.split('_')
                
                # The first two elements after the dataset are sparse and dense
                if len(parts) >= 2:
                    sparse_retriever = parts[0]
                    dense_retriever = parts[1]
            
            # Note: Assuming process_single_dataset is defined elsewhere in your code
            result = process_single_dataset(dataset, split, sparse_retriever, dense_retriever, trec_path, qrels_path)
            all_results.append(result)
            
    # Sort results by dataset -> split -> sparse -> dense
    all_results.sort(key=lambda x: (x['dataset'], x['split'], x['sparse'], x['dense']))
    
    # Define output file paths
    txt_path = os.path.join(folder, "metrics.txt")
    csv_path = os.path.join(folder, "metrics.csv")
    
    # Prepare CSV headers
    csv_headers = [
        "Dataset", "Split", "Sparse", "Dense", 
        "NDCG@5", "NDCG@10", "MRR@5", "MRR@10", "MAP@5", "MAP@10"
    ]

    # Open both files for writing
    with open(txt_path, 'w', encoding='utf-8') as txt_file, \
         open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_headers)

        # Helper to print to screen and write to txt file
        def print_and_save(line):
            print(line)
            txt_file.write(line + "\n")

        # Print/Save summary table headers
        print_and_save(f"\n{'='*230}")
        print_and_save("SUMMARY RESULTS")
        print_and_save(f"{'='*230}")
        print_and_save(f"{'Dataset':<15} {'Split':<8} {'Sparse':<10} {'Dense':<12} {'NDCG@5':<30} {'NDCG@10':<30} {'MRR@5':<30} {'MRR@10':<30} {'MAP@5':<30} {'MAP@10':<30}")
        print_and_save(f"{'-'*230}")
        
        for result in all_results:
            # Format the strings
            ndcg_formatted_5 = f"{result['bootstrap_ndcg_5']:.3f} [{result['ndcg_lower_5']:.3f}, {result['ndcg_upper_5']:.3f}]"
            ndcg_formatted_10 = f"{result['bootstrap_ndcg_10']:.3f} [{result['ndcg_lower_10']:.3f}, {result['ndcg_upper_10']:.3f}]"
            mrr_formatted_5 = f"{result['bootstrap_mrr_5']:.3f} [{result['mrr_lower_5']:.3f}, {result['mrr_upper_5']:.3f}]"
            mrr_formatted_10 = f"{result['bootstrap_mrr_10']:.3f} [{result['mrr_lower_10']:.3f}, {result['mrr_upper_10']:.3f}]"
            map_formatted_5 = f"{result['bootstrap_map_5']:.3f} [{result['map_lower_5']:.3f}, {result['map_upper_5']:.3f}]"     
            map_formatted_10 = f"{result['bootstrap_map_10']:.3f} [{result['map_lower_10']:.3f}, {result['map_upper_10']:.3f}]"
            
            # Print and save to TXT
            row_str = f"{result['dataset']:<15} {result['split']:<8} {result['sparse']:<10} {result['dense']:<12} {ndcg_formatted_5:<30} {ndcg_formatted_10:<30} {mrr_formatted_5:<30} {mrr_formatted_10:<30} {map_formatted_5:<30}  {map_formatted_10:<30}"
            print_and_save(row_str)
            
            # Write row to CSV
            csv_writer.writerow([
                result['dataset'], 
                result['split'], 
                result['sparse'], 
                result['dense'], 
                ndcg_formatted_5, 
                ndcg_formatted_10, 
                mrr_formatted_5, 
                mrr_formatted_10, 
                map_formatted_5, 
                map_formatted_10
            ])
    
    print(f"\nResults saved to:\n  - {txt_path}\n  - {csv_path}")
    return all_results

if __name__ == "__main__":
    
    datasets_config = [
        {"dataset": "acord-entire-corpus", "split": "test"},
        {"dataset": "msmarco", "split": "dev"},
        {"dataset": "nq", "split": "dev"},
        {"dataset": "nfcorpus", "split": "test"}
    ]
    
    _base_results_dir = os.environ.get("BASE_RESULTS_DIR", "/extra/huaiyaom0/tr-intern/wrrf/results")
    folder_path = f"{_base_results_dir}/01-standard-rrf"
    
    # Process all datasets
    results = process_multiple_datasets(datasets_config, folder_path)
    print(f"result from: {folder_path}")