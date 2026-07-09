import csv
import os
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
    qrels = defaultdict(set)
    with read_file_content(qrels_file) as f:
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
                if score > 0:
                    qrels[query_id].add(doc_id)
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
    # Sort just to be safe, though usually TREC files are sorted
    for query_id in results:
        results[query_id].sort(key=lambda x: x[1])
    return results

def reciprocal_rank_fusion(bm25_results, dense_results, bm25_weight, k=60):
    """
    Perform weighted reciprocal rank fusion (From Original Code)
    """
    dense_weight = round(1.0 - bm25_weight, 2)
    fused_results = defaultdict(dict)
    
    # Get all queries
    all_queries = set(bm25_results.keys()) | set(dense_results.keys())
    
    for query_id in all_queries:
        doc_scores = defaultdict(float)
        
        # Add BM25 scores
        if query_id in bm25_results:
            for doc_id, rank in bm25_results[query_id]:
                rrf_score = 1.0 / (k + rank)
                doc_scores[doc_id] += bm25_weight * rrf_score
  
        # Add dense retriever scores
        if query_id in dense_results:
            for doc_id, rank in dense_results[query_id]:
                rrf_score = 1.0 / (k + rank)
                doc_scores[doc_id] += dense_weight * rrf_score
      
        # Sort documents by fused score (descending)
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        fused_results[query_id] = [(doc_id, rank + 1) for rank, (doc_id, score) in enumerate(sorted_docs)]
    
    return fused_results

def calculate_mrr_at_10_single_query(query_results, query_qrels, k=10):
    """Calculate MRR@k for a single query using position-based ranking (From Original Code)"""
    for position, (doc_id, _) in enumerate(query_results[:k], 1):
        if doc_id in query_qrels:
            return round(1.0 / position, 4)
    return 0.0

def calculate_average_mrr(results, qrels, k=10):
    """Calculate average MRR@k across all queries"""
    total_mrr = 0.0
    count = 0
    for query_id in results:
        if query_id in qrels:
            mrr = calculate_mrr_at_10_single_query(results[query_id], qrels[query_id], k)
            total_mrr += mrr
            count += 1
    return total_mrr / count if count > 0 else 0.0

def optimize_rrf_iterative(sparse_file, dense_file, qrels_file):
    """
    RRF Optimization using the iterative method from Original Code 
    (Sorting + Python Loops instead of NumPy vectorization)
    """
    # 1. Load Data
    print("Loading data...")
    qrels = load_qrels(qrels_file)
    sparse_results = load_trec_results(sparse_file)
    dense_results = load_trec_results(dense_file)
    
    # 2. Baseline Calculations
    print("Calculating baselines...")
    sparse_avg = calculate_average_mrr(sparse_results, qrels)
    dense_avg = calculate_average_mrr(dense_results, qrels)
    print(f"  Sparse Baseline MRR@10: {sparse_avg:.4f}")
    print(f"  Dense Baseline MRR@10:  {dense_avg:.4f}")

    # 3. Optimization
    print("Starting Iterative Optimization...")
    results_data = []
    
    all_queries = set(sparse_results.keys()) | set(dense_results.keys())
    total_queries = len(all_queries)
    query_count = 0
    
    for query_id in all_queries:
        query_count += 1
        
        if query_count % 1000 == 0:
            print(f"  Processed {query_count}/{total_queries} queries...")

        best_weights = []
        best_mrr = 0.0
        
        all_weights = []
        all_mrr_values = []
        
        # Grid search over weights 0.00 to 1.00
        for i in range(101):
            weight = round(i * 0.01, 2)
            all_weights.append(weight)
            
            # Isolate data for this specific query (as done in original code)
            single_query_sparse = {query_id: sparse_results.get(query_id, [])}
            single_query_dense = {query_id: dense_results.get(query_id, [])}
            
            # Perform Fusion
            fused_results = reciprocal_rank_fusion(single_query_sparse, single_query_dense, weight)
            
            # Calculate MRR
            if query_id in fused_results and query_id in qrels:
                mrr = calculate_mrr_at_10_single_query(fused_results[query_id], qrels[query_id])
            else:
                mrr = 0.0
            
            all_mrr_values.append(mrr)
            
            # Track best weights
            if mrr > best_mrr:
                best_mrr = mrr
                best_weights = [weight]
            elif mrr == best_mrr and mrr > 0:
                best_weights.append(weight)
                
        # If no relevant docs found for any weight, best_mrr is 0.
        # We handle the 'best_weights' logic same as original code.
        
        results_data.append({
            'query_id': query_id,
            'query_text': f'Query_{query_id}',
            'best_weights': best_weights,
            'highest_mrr@10': best_mrr,
            'weights': all_weights,
            'all_weights_mrr@10': all_mrr_values
        })
        
    return results_data, sparse_avg, dense_avg, total_queries

def main_single(dataset_name, split, sparse_method_name, dense_method_name):
    
    top_k = 200
    
    sparse_file = os.path.join(BASE_PATH, dataset_name, f"search_results/{sparse_method_name}/top{top_k}", f"results_{split}.trec")
    dense_file = os.path.join(BASE_PATH, dataset_name, f"search_results/{dense_method_name}/top{top_k}", f"results_{split}.trec")
    qrels_file = os.path.join(BASE_PATH, dataset_name, "qrels", f"{split}.tsv")
    
    output_dir = os.path.join(BASE_PATH, dataset_name, "mrr_runs", split, f"top{top_k}")
    output_file = os.path.join(output_dir, f"results_{split}_{sparse_method_name}_vs_{dense_method_name}_best_weights.csv")
    
    print(f"Processing: {dataset_name} | {split}")
    
    # Run Optimization (Iterative/Original Method)
    results_data, sparse_avg, dense_avg, total_queries = optimize_rrf_iterative(
        sparse_file, dense_file, qrels_file
    )
    
    fieldnames = ['query_id', 'query_text', 'best_weights', 'highest_mrr@10', 'weights', 'all_weights_mrr@10']
    write_csv_file(output_file, results_data, fieldnames)
    
    
    
    sum_weights = 0.0
    valid_weight_count = 0

    # avg_mrr = sum(row['highest_mrr@10'] for row in results_data) / total_queries if total_queries > 0 else 0.0
    avg_mrr = sum(row['highest_mrr@10'] for row in results_data) / valid_weight_count if valid_weight_count > 0 else 0.0

    for row in results_data:
        if row['highest_mrr@10'] > 0 and row['best_weights']:
            sum_weights += row['best_weights'][0]
            valid_weight_count += 1
            
    avg_weight = sum_weights / valid_weight_count if valid_weight_count > 0 else 0.0
    
    print(f"\nOptimization complete!")
    print(f"{sparse_method_name} Avg MRR@10: {sparse_avg:.4f}")
    print(f"{dense_method_name} Avg MRR@10:  {dense_avg:.4f}")
    print(f"Optimized Avg MRR@10: {avg_mrr:.4f}")
    print(f"Avg Best Weight (for {sparse_method_name}): {avg_weight:.2f}")
    
if __name__ == "__main__":
    DATASET = "nq" #nq, msmarco
    SPLIT = "train"
    SPARSE_METHOD = "bm25"
    DENSE_METHOD = "biencoder"
    
    main_single(DATASET, SPLIT, SPARSE_METHOD, DENSE_METHOD)