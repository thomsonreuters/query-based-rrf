import csv
import os
# import boto3
import io
from collections import defaultdict
from pathlib import Path

# S3 Configuration
# s3_client = boto3.client('s3')
is_client = ""
bucket_name = 'a207918-ml-workspace-practicallawxohj-use1'
base_path = 'labs_pl-intern-irisma-summer25/dataset'

def is_s3_path(path):
    """Check if path is an S3 URI"""
    return path.startswith('s3://')

def read_file_content(file_path):
    """Read file content from S3 or local filesystem"""
    if is_s3_path(file_path):
        # Extract S3 key from full S3 path
        s3_key = file_path.replace(f's3://{bucket_name}/', '')
        
        # Read from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        content = response['Body'].read().decode('utf-8')
        return io.StringIO(content)
    else:
        # Read from local filesystem
        return open(file_path, 'r')

def write_csv_file(file_path, data, fieldnames):
    """Write CSV file to S3 or local filesystem"""
    if is_s3_path(file_path):
        # Extract S3 key from full S3 path
        s3_key = file_path.replace(f's3://{bucket_name}/', '')
        
        # Write to S3
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        print(f"File saved to S3: s3://{bucket_name}/{s3_key}")
    else:
        # Write to local filesystem
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        print(f"File saved locally: {file_path}")

def load_qrels(qrels_file):
    """Load relevance judgments from TSV file (S3 or local)"""
    qrels = defaultdict(set)
    with read_file_content(qrels_file) as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query_id = parts[0]
                doc_id = parts[1]
                score = int(parts[2])
                if score > 0:  # Relevant document
                    qrels[query_id].add(doc_id)
    return qrels

def load_trec_results(trec_file):
    """Load TREC results and return ranked lists per query (S3 or local)"""
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

def reciprocal_rank_fusion(bm25_results, dense_results, bm25_weight, k=60):
    """
    Perform weighted reciprocal rank fusion
    RRF score = w1 * 1/(k + rank1) + w2 * 1/(k + rank2)
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
    """Calculate MRR@k for a single query using position-based ranking"""
    # query_results should already be sorted by rank
    # We use position in the ranked list (1-indexed) for MRR calculation
    
    for position, (doc_id, _) in enumerate(query_results[:k], 1):
        if doc_id in query_qrels:
            return round(1.0 / position, 4)
    
    return 0.0  # No relevant document found in top k

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


def reciprocal_rank_fusion_optimization_per_query(bm25_file, dense_file, qrels_file):
    """
    Find optimal BM25 weight for weighted RRF using grid search for each query individually
    Track all weights that achieve the highest MRR
    """

    global num_queries

    # Load data
    qrels = load_qrels(qrels_file)
    num_queries = len(qrels)
    bm25_results = load_trec_results(bm25_file)
    dense_results = load_trec_results(dense_file)
    
    # Calculate average MRR@10 for BM25 and dense separately
    bm25_avg_mrr = calculate_average_mrr(bm25_results, qrels)
    dense_avg_mrr = calculate_average_mrr(dense_results, qrels)
    
    print(f"  BM25 Average MRR@10: {bm25_avg_mrr:.4f}")
    print(f"  Dense Average MRR@10: {dense_avg_mrr:.4f}")

    input(f"press to continue ...")
    
    # Get all queries
    all_queries = set(bm25_results.keys()) | set(dense_results.keys())
    
    results_data = []
    query_count = 0
    
    for query_id in all_queries:
        query_count += 1
        
        # Only print progress for every 1000 queries
        if query_count % 1000 == 0:
            print(f"  Processed {query_count}/{len(all_queries)} queries. Currently optimizing for query: {query_id}")
        
        best_weights = []  # List to store all weights that achieve the highest MRR
        best_mrr = 0.0
        
        # Initialize lists to store all weights and their corresponding MRR values
        all_weights = []
        all_mrr_values = []
        
        # Grid search over BM25 weights from 0.0 to 1.0 with step 0.01
        for i in range(101):  # 0 to 100 inclusive
            bm25_weight = round(i * 0.01, 2)
            all_weights.append(bm25_weight)  # Add weight to list
            
            # Create single-query results for fusion
            single_query_bm25 = {query_id: bm25_results.get(query_id, [])}
            single_query_dense = {query_id: dense_results.get(query_id, [])}
            
            # Perform weighted RRF for this query
            fused_results = reciprocal_rank_fusion(single_query_bm25, single_query_dense, bm25_weight)
            
            # Calculate MRR@10 for this query
            if query_id in fused_results and query_id in qrels:
                mrr = round(calculate_mrr_at_10_single_query(fused_results[query_id], qrels[query_id]),4)
            else:
                mrr = 0.0
            
            all_mrr_values.append(mrr)  # Add MRR to list
            
            # Track all weights that achieve the highest MRR
            if mrr > best_mrr:
                # Found a higher MRR - clear previous weights and start fresh
                best_mrr = mrr
                best_weights = [bm25_weight]  # Clear and start with new weight
            elif mrr == best_mrr and mrr > 0:
                # Same MRR as current best - add to the list
                best_weights.append(bm25_weight)
        
        # Store results for this query
        results_data.append({
            'query_id': query_id,
            'query_text': f'Query_{query_id}',  # Placeholder since we don't have query text
            'best_weights': best_weights,
            'highest_mrr@10': best_mrr,
            'weights': all_weights,
            'all_weights_mrr@10': all_mrr_values
        })
        
        # Only print detailed results for every 1000 queries
        if query_count % 1000 == 0:
            print(f"    Best weights: {best_weights}, Best MRR@10: {best_mrr:.4f}")
    
    # Print final count if it doesn't align with 1000 interval
    if query_count % 1000 != 0:
        print(f"  Completed processing all {query_count} queries.")
    
    return results_data, bm25_avg_mrr, dense_avg_mrr

def main_single(dataset_name, split):
    """Main function to run RRF optimization for a single dataset and split"""
    
    # Configuration: Set to True for S3, False for local
    use_s3 = False  # Change this to switch between S3 and local

    top_k = 200
  
    if use_s3:
        # S3 file paths
        bm25_file = f"s3://{bucket_name}/{base_path}/{dataset_name}/bm25_top{top_k}_results/results_{split}.trec"
        dense_file = f"s3://{bucket_name}/{base_path}/{dataset_name}/biencoder_top{top_k}_results/results_{split}.trec"
        qrels_file = f"s3://{bucket_name}/{base_path}/{dataset_name}/qrels/{split}.tsv"
        output_file = f"s3://{bucket_name}/{base_path}/{dataset_name}/mrr_runs/{split}/top{top_k}/results_{split}_best_weights_all_weights.csv"
    else:
        # Local file paths
        bm25_file = f"dataset/{dataset_name}/search_results/bm25/top{top_k}/results_{split}.trec"
        dense_file = f"dataset/{dataset_name}/search_results/biencoder/top{top_k}/results_{split}.trec"
        qrels_file = f"dataset/{dataset_name}/qrels/{split}.tsv"
        output_file = f"{dataset_name}_mrr_runs/{split}/top{top_k}/results_{split}_best_weights_all_weights_verify_original.csv"
    
    print(f"Processing dataset: {dataset_name}, split: {split}")
    print(f"  BM25: {bm25_file}")
    print(f"  Dense: {dense_file}")
    print(f"  Qrels: {qrels_file}")
    

    # Run optimization per query
    results_data, bm25_avg_mrr, dense_avg_mrr = reciprocal_rank_fusion_optimization_per_query(
        bm25_file, dense_file, qrels_file
    )
    
    # Write results
    fieldnames = ['query_id', 'query_text', 'best_weights', 'highest_mrr@10', 'weights', 'all_weights_mrr@10']
    write_csv_file(output_file, results_data, fieldnames)
    
    # Calculate overall statistics
    total_queries = num_queries
    avg_mrr = sum(row['highest_mrr@10'] for row in results_data) / total_queries if total_queries > 0 else 0.0
    # Calculate average of first best weight for each query (for backward compatibility)
    avg_weight = sum(row['best_weights'][0] if row['best_weights'] else 0.0 for row in results_data) / total_queries if total_queries > 0 else 0.0
    
    print(f"\nOptimization complete!")
    print(f"Total queries processed: {total_queries}")
    print(f"BM25 Average MRR@10 (before RRF): {bm25_avg_mrr:.4f}")
    print(f"Dense Average MRR@10 (before RRF): {dense_avg_mrr:.4f}")
    print(f"Average first best weight: {avg_weight:.2f}")
    print(f"Average MRR@10 (after RRF): {avg_mrr:.4f}")
    print(f"Results saved to: {output_file}")
    
    return results_data


# Usage examples
if __name__ == "__main__":
    
    # Single dataset processing
    dataset_name = "nq"  # or "nfcorpus", etc.
    split = "train"
    
    print("=== Single Dataset Processing ===")
    results = main_single(dataset_name, split)
    
    # # Multiple datasets processing
    # datasets = ["nq", "nfcorpus"]  # Add more datasets as needed
    # splits = ["dev", "test"]  # Add more splits as needed
    
    # print("\n=== Multiple Datasets Processing ===")
    # for dataset in datasets:
    #     for split in splits:
    #         print(f"\n--- Processing {dataset} {split} ---")
    #         main_single(dataset, split)