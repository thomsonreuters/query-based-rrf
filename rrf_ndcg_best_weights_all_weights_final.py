import csv
import os
import math
import numpy as np
from collections import defaultdict
from pathlib import Path

# Configuration
BASE_PATH = 'dataset'

def read_file_content(file_path):
    """Read file content from local filesystem"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return open(file_path, 'r', encoding='utf-8')

def write_csv_file(file_path, data, fieldnames):
    """Write CSV file to local filesystem"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"File saved locally: {file_path}")

def load_qrels(qrels_file):
    """Load relevance judgments from TSV file"""
    qrels = defaultdict(dict)
    with read_file_content(qrels_file) as f:
        # Handle header safely
        first = True
        for line in f:
            if first:
                first = False
                if 'query' in line.lower() or 'id' in line.lower():
                    continue
            
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query_id = parts[0]
                doc_id = parts[1]
                score = int(parts[2])
                qrels[query_id][doc_id] = score
    return qrels

def load_trec_results(trec_file):
    """Load TREC results and return ranked lists per query"""
    results = defaultdict(list)
    with read_file_content(trec_file) as f:
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

def calculate_ndcg_at_10_single_query(query_results, query_qrels, k=10):
    """Calculate NDCG@k for a single query (Baseline Python implementation)"""
    # Only consider top k
    results_top_k = query_results[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, (doc_id, _) in enumerate(results_top_k):
        relevance = query_qrels.get(doc_id, 0)
        if i == 0:
            dcg += relevance
        else:
            dcg += relevance / math.log2(i + 2)
    
    # Calculate IDCG (ideal DCG)
    ideal_relevances = sorted(query_qrels.values(), reverse=True)[:k]
    idcg = 0.0
    for i, relevance in enumerate(ideal_relevances):
        if i == 0:
            idcg += relevance
        else:
            idcg += relevance / math.log2(i + 2)
    
    if idcg > 0:
        return dcg / idcg
    return 0.0

def calculate_average_ndcg(results, qrels, k=10):
    """Calculate average NDCG@10 across all queries (Baseline)"""
    ndcg_scores = []
    for query_id in results:
        if query_id in qrels:
            ndcg = calculate_ndcg_at_10_single_query(results[query_id], qrels[query_id], k)
            ndcg_scores.append(ndcg)
    return sum(ndcg_scores) / len(qrels) if len(qrels) > 0 else 0.0

def optimize_ndcg_numpy(sparse_file, dense_file, qrels_file, rr_k=60, ndcg_k=10):
    """
    Optimized RRF Grid Search using NumPy for NDCG@10.
    """
    # 1. Load Data
    print("Loading data...")
    qrels = load_qrels(qrels_file)
    sparse_results = load_trec_results(sparse_file)
    dense_results = load_trec_results(dense_file)
    
    # 2. Baseline Calculations
    print("Calculating baselines...")
    sparse_avg = calculate_average_ndcg(sparse_results, qrels, ndcg_k)
    dense_avg = calculate_average_ndcg(dense_results, qrels, ndcg_k)
    print(f"  Sparse Baseline NDCG@10: {sparse_avg:.4f}")
    print(f"  Dense Baseline NDCG@10:  {dense_avg:.4f}")

    # 3. Pre-compute Discount Factors for NDCG
    # Formula: rel if i=0 else rel/log2(i+2)
    # Ranks are 0-indexed in array: 0, 1, 2...
    # i=0: val=1.0. i=1: 1/log2(3). i=2: 1/log2(4).
    discounts = np.zeros(ndcg_k)
    discounts[0] = 1.0
    if ndcg_k > 1:
        discounts[1:] = 1.0 / np.log2(np.arange(1, ndcg_k) + 2)
    
    # 4. Optimization
    print("Starting NumPy optimization...")
    results_data = []
    
    # Weight vectors: shape (101, 1)
    w_sparse_vals = np.round(np.linspace(0, 1, 101).reshape(-1, 1), 2)
    w_dense_vals = 1.0 - w_sparse_vals
    
    all_queries = set(sparse_results.keys()) | set(dense_results.keys())
    total_queries = len(all_queries)
    
    for idx, query_id in enumerate(all_queries):
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{total_queries} queries...")

        # 4a. Get Qrels and IDCG for this query
        query_qrels = qrels.get(query_id, {})
        if not query_qrels:
            # No relevance judgments exist for this query
            # Skip or record 0. We record 0 to keep count correct.
            # (Logic matches calculate_average_ndcg which skips missing queries in denominator,
            # but usually we want to output a row for every retrieved query)
            pass
            
        # Calculate IDCG (Ideal DCG)
        ideal_relevances = sorted(query_qrels.values(), reverse=True)[:ndcg_k]
        idcg = 0.0
        for i, rel in enumerate(ideal_relevances):
            idcg += rel * discounts[i]
            
        if idcg == 0.0:
            # If IDCG is 0, NDCG is always 0.
            results_data.append({
                'query_id': query_id,
                'query_text': f'Query_{query_id}',
                'best_weights': [0.5], 'highest_ndcg@10': 0.0,
                'weights': w_sparse_vals.flatten().tolist(),
                'all_weights_ndcg@10': [0.0] * 101
            })
            continue

        # 4b. Build document union and score vectors
        s_list = sparse_results.get(query_id, [])
        d_list = dense_results.get(query_id, [])
        
        doc_to_index = {}
        unique_docs = []
        current_idx = 0
        
        s_scores_map = {}
        d_scores_map = {}
        
        for doc_id, rank in s_list:
            s_scores_map[doc_id] = 1.0 / (rr_k + rank)
            if doc_id not in doc_to_index:
                doc_to_index[doc_id] = current_idx
                unique_docs.append(doc_id)
                current_idx += 1
                
        for doc_id, rank in d_list:
            d_scores_map[doc_id] = 1.0 / (rr_k + rank)
            if doc_id not in doc_to_index:
                doc_to_index[doc_id] = current_idx
                unique_docs.append(doc_id)
                current_idx += 1
        
        if not unique_docs:
            results_data.append({
                'query_id': query_id,
                'query_text': f'Query_{query_id}',
                'best_weights': [0.5], 'highest_ndcg@10': 0.0,
                'weights': w_sparse_vals.flatten().tolist(),
                'all_weights_ndcg@10': [0.0] * 101
            })
            continue

        num_docs = len(unique_docs)
        vec_s = np.zeros(num_docs)
        vec_d = np.zeros(num_docs)
        
        # Build relevance vector for all docs in pool
        # shape: (num_docs,)
        doc_relevances = np.zeros(num_docs)
        
        for doc_id in unique_docs:
            idx_in_vec = doc_to_index[doc_id]
            # Fill score vectors
            if doc_id in s_scores_map: vec_s[idx_in_vec] = s_scores_map[doc_id]
            if doc_id in d_scores_map: vec_d[idx_in_vec] = d_scores_map[doc_id]
            # Fill relevance vector
            if doc_id in query_qrels:
                doc_relevances[idx_in_vec] = query_qrels[doc_id]

        # 4c. Vectorized Calculation
        # Calculate RRF Scores: (101, num_docs)
        all_scores = w_sparse_vals * vec_s[None, :] + w_dense_vals * vec_d[None, :]
        
        # Sort and get indices of top K docs for each weight
        # argsort is ascending, so we use -all_scores to sort descending
        # We only need top K
        top_k_indices = np.argsort(-all_scores, axis=1)[:, :ndcg_k]
        
        # Retrieve relevances for these top K docs
        # shape: (101, ndcg_k)
        # We use advanced indexing: For every row i, take indices from top_k_indices[i]
        # Since doc_relevances is 1D, we can just flatten or broadcast?
        # Correct NumPy way: doc_relevances[top_k_indices] works if top_k_indices is int array
        top_k_relevances = doc_relevances[top_k_indices]
        
        # Calculate DCG for all 101 weights
        # shape: (101,)
        dcg_values = np.sum(top_k_relevances * discounts[None, :], axis=1)
        
        # Calculate NDCG
        ndcg_values = dcg_values / idcg
        ndcg_values = np.round(ndcg_values, 4)
        
        max_ndcg = np.max(ndcg_values)
        best_weight_indices = np.where(ndcg_values == max_ndcg)[0]
        
        if max_ndcg > 0:
            best_ws = w_sparse_vals[best_weight_indices].flatten().tolist()
        else:
            best_ws = [0.5]

        results_data.append({
            'query_id': query_id,
            'query_text': f'Query_{query_id}',
            'best_weights': best_ws,
            'highest_ndcg@10': float(max_ndcg),
            'weights': w_sparse_vals.flatten().tolist(),
            'all_weights_ndcg@10': ndcg_values.tolist()
        })
        
    return results_data, sparse_avg, dense_avg, len(all_queries)

def main_single(dataset_name, split, sparse_method_name, dense_method_name, top_k):
    
    # group = "original-paper" # Or param
    
    # Construct Paths Dynamically
    # Adjust strict path structure to match your specific example in prompt if needed.
    # User example: dataset/{dataset}/search_results/{group}/bm25/top{top_k}/results_{split}.trec
    
    sparse_file = os.path.join(BASE_PATH, dataset_name, "search_results", sparse_method_name, f"top{top_k}", f"results_{split}.trec")
    dense_file = os.path.join(BASE_PATH, dataset_name, "search_results", dense_method_name, f"top{top_k}", f"results_{split}.trec")
    qrels_file = os.path.join(BASE_PATH, dataset_name, "qrels", f"{split}.tsv")
    
    # Output path
    output_dir = os.path.join(BASE_PATH, dataset_name, "ndcg_runs", split, f"top{top_k}")
    output_file = os.path.join(output_dir, f"results_{split}_{sparse_method_name}_vs_{dense_method_name}_best_weights.csv")
    
    print(f"Processing: {dataset_name} | {split}")
    print(f"  Sparse: {sparse_file}")
    print(f"  Dense:  {dense_file}")
    
    if not os.path.exists(sparse_file) or not os.path.exists(dense_file):
        print("Error: Input files not found.")
        return

    # Run Optimized NumPy version
    results_data, sparse_avg, dense_avg, total_queries = optimize_ndcg_numpy(
        sparse_file, dense_file, qrels_file, ndcg_k=10
    )
    
    fieldnames = ['query_id', 'query_text', 'best_weights', 'highest_ndcg@10', 'weights', 'all_weights_ndcg@10']
    write_csv_file(output_file, results_data, fieldnames)
    
    avg_ndcg = sum(row['highest_ndcg@10'] for row in results_data) / total_queries if total_queries > 0 else 0.0
    
    sum_weights = 0.0
    valid_weight_count = 0
    for row in results_data:
        if row['highest_ndcg@10'] > 0:
            sum_weights += row['best_weights'][0]
            valid_weight_count += 1
            
    avg_weight = sum_weights / valid_weight_count if valid_weight_count > 0 else 0.0
    
    print(f"\nOptimization complete!")
    print(f"{sparse_method_name} Avg NDCG@10: {sparse_avg:.4f}")
    print(f"{dense_method_name} Avg NDCG@10:  {dense_avg:.4f}")
    print(f"Optimized Avg NDCG@10: {avg_ndcg:.4f}")
    print(f"Avg Best Weight (for {sparse_method_name}): {avg_weight:.2f}")

if __name__ == "__main__":
    
    TOP_K = 200
    DATASET = "nfcorpus" # acord-entire-corpus, nfcorpus
    
    # Setup all lists to iterate over
    SPLITS = ["dev", "test", "train"]
    SPARSE_METHODS = ["bm25", "rm3"] 
    DENSE_METHODS = ["biencoder", "qwen3"]
    
    # Iterate through Splits -> Sparse -> Dense
    for split in SPLITS:
        for sparse_method in SPARSE_METHODS:
            for dense_method in DENSE_METHODS:
                
                print(f"=== Running RRF NDCG: {split.upper()} | {sparse_method} + {dense_method} ===")
                
                # Pass the current 'split' variable to the function
                main_single(DATASET, split, sparse_method, dense_method, TOP_K)
                
                print("-" * 50)