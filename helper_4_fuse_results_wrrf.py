import csv
import pandas as pd
import os
import glob
import itertools

def parse_trec_file(file_path):
    """
    Parse a TREC format file and return a nested dictionary structure.
    """
    results = {}
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        return results

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) >= 6:
                query_id = parts[0]
                passage_id = parts[2]
                rank = int(parts[3])
                score = float(parts[4])
                run_id = parts[5]
                
                if query_id not in results:
                    results[query_id] = {}
                
                results[query_id][passage_id] = {
                    'rank': rank,
                    'score': score,
                    'run_id': run_id
                }
    
    return results

def fuse_search_results_fixed_weight(sparse_file, dense_file, sparse_weight, output_file, k=60, run_id="fused_results"):
    """
    Fuse search results using a fixed weight for the sparse retriever across all queries.
    Does NOT require a weight CSV.
    """
    sparse_results = parse_trec_file(sparse_file)
    dense_results = parse_trec_file(dense_file)
    
    all_queries = set(sparse_results.keys()) | set(dense_results.keys())
    fused_results = []

    for query_id in sorted(all_queries):     
        sparse_passages = set(sparse_results.get(query_id, {}).keys())
        dense_passages = set(dense_results.get(query_id, {}).keys())
        all_passages = sparse_passages | dense_passages
        
        passage_scores = {}
        for psg_id in all_passages:
            wrrf_score = 0.0
            
            if query_id in sparse_results and psg_id in sparse_results[query_id]:
                rank_s = sparse_results[query_id][psg_id]['rank']
                wrrf_score += sparse_weight * (1.0 / (k + rank_s))
            
            if query_id in dense_results and psg_id in dense_results[query_id]:
                rank_d = dense_results[query_id][psg_id]['rank']
                wrrf_score += (1 - sparse_weight) * (1.0 / (k + rank_d))
            
            passage_scores[psg_id] = wrrf_score
        
        sorted_passages = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)
        
        for new_rank, (psg_id, wrrf_score) in enumerate(sorted_passages, 1):
            fused_results.append({
                'query_id': query_id,
                'psg_id': psg_id,
                'rank': new_rank,
                'score': wrrf_score,
                'run_id': run_id
            })
    
    with open(output_file, 'w') as f:
        for result in fused_results:
            line = f"{result['query_id']} Q0 {result['psg_id']} {result['rank']} {result['score']:.8f} {result['run_id']}"
            f.write(line + '\n')
    
    print(f"Fused results saved to: {output_file}")

def fuse_search_results_predicted_weights(sparse_file, dense_file, weight_csv_file, output_file, k=60, run_id="fused_results"):
    """
    Fuse search results using predicted weights from CSV file for each query.
    """
    sparse_results = parse_trec_file(sparse_file)
    dense_results = parse_trec_file(dense_file)
    
    weights_df = pd.read_csv(weight_csv_file)
    query_weights = dict(zip(weights_df['query_id'].astype(str), weights_df['predicted']))
    
    all_queries = set(sparse_results.keys()) | set(dense_results.keys())
    fused_results = []
    
    for query_id in sorted(all_queries):
        if str(query_id) not in query_weights:
            continue

        sparse_weight = query_weights.get(str(query_id), 0.5)
        sparse_passages = set(sparse_results.get(query_id, {}).keys())
        dense_passages = set(dense_results.get(query_id, {}).keys())
        all_passages = sparse_passages | dense_passages
        
        passage_scores = {}
        for psg_id in all_passages:
            wrrf_score = 0.0
            if query_id in sparse_results and psg_id in sparse_results[query_id]:
                rank_s = sparse_results[query_id][psg_id]['rank']
                wrrf_score += sparse_weight * (1.0 / (k + rank_s))
            
            if query_id in dense_results and psg_id in dense_results[query_id]:
                rank_d = dense_results[query_id][psg_id]['rank']
                wrrf_score += (1 - sparse_weight) * (1.0 / (k + rank_d))
            
            passage_scores[psg_id] = wrrf_score
        
        sorted_passages = sorted(passage_scores.items(), key=lambda x: x[1], reverse=True)
        for new_rank, (psg_id, wrrf_score) in enumerate(sorted_passages, 1):
            fused_results.append({
                'query_id': query_id,
                'psg_id': psg_id,
                'rank': new_rank,
                'score': wrrf_score,
                'run_id': run_id
            })
    
    with open(output_file, 'w') as f:
        for result in fused_results:
            line = f"{result['query_id']} Q0 {result['psg_id']} {result['rank']} {result['score']:.8f} {result['run_id']}"
            f.write(line + '\n')
    
    print(f"Fused results saved to: {output_file}")

if __name__ == "__main__":
    
    use_fixed_weight = False  # Toggle this to switch logic
    sparse_weight = 0.5
    top_k = 200
    _base_experiment_dir = os.environ.get("BASE_EXPERIMENT_DIR", "/extra/huaiyaom0/tr-intern/wrrf/experiment")
    experiment = f"{_base_experiment_dir}/roberta/roberta-experiment-1-mean-best-weight-1"
    
    if use_fixed_weight:

        # Define which retrievers to use
        # sparse_name = "bm25" 
        sparse_name = "rm3"   # e.g., "bm25" or "rm3"
        
        dense_name = "biencoder"
        # dense_name = "qwen3"  # e.g., "biencoder" or "qwen3"
        # --- FIXED WEIGHT LOGIC: Iterate datasets directly ---
        
        datasets = [
            ("acord-entire-corpus", "test"),
            ("msmarco", "dev"),
            ("nfcorpus", "test"),
            ("nq", "dev"),
        ]
        
        for dataset_name, split in datasets:
            sparse_file = f"dataset/{dataset_name}/search_results/{sparse_name}/top{top_k}/results_{split}.trec"
            dense_file = f"dataset/{dataset_name}/search_results/{dense_name}/top{top_k}/results_{split}.trec"
            output_file = f"results/{experiment}/{dataset_name}_{sparse_name}_{dense_name}_{split}_{str(sparse_weight).replace('.', '')}.trec"

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            fuse_search_results_fixed_weight(
                sparse_file=sparse_file,
                dense_file=dense_file,
                sparse_weight=sparse_weight,
                output_file=output_file,
            )
            print(f"Completed {dataset_name} using fixed weight: {sparse_weight} ({sparse_name} + {dense_name})")

    else:
            subfolder = "04-roberta"
            folder_path = f"{experiment}/predictions"
            prediction_files = glob.glob(os.path.join(folder_path, "*.csv")) 

            # 1. Define all possible retrievers to look for in the filename
            sparse_retrievers = ["bm25", "rm3"]
            dense_retrievers = ["biencoder", "qwen3"]

            # 2. Use a dictionary to cleanly map filename keywords to (dataset_name, split)
            dataset_mapping = {
            # "acord-original-paper": ("acord-original-paper", "test"),
                "acord-entire-corpus": ("acord-entire-corpus", "test"),
                "msmarco": ("msmarco", "dev"),
                "nfcorpus": ("nfcorpus", "test"),
                "nq": ("nq", "dev"),
            }

            for file_path in prediction_files:
                file_name = os.path.basename(file_path).lower()
                
                # Extract dataset_name and split based on the mapping
                dataset_name, split = next(
                    ((d_name, d_split) for key, (d_name, d_split) in dataset_mapping.items() if key in file_name), 
                    (None, None)
                )
                
                # Skip the file if it doesn't match any known dataset
                if not dataset_name:
                    continue
                    
                # 3. Dynamically identify the retrievers from the file name
                sparse_name = next((s for s in sparse_retrievers if s in file_name), None)
                dense_name = next((d for d in dense_retrievers if d in file_name), None)

                # Skip if the filename doesn't contain exactly one valid sparse and dense retriever
                if not sparse_name or not dense_name:
                    print(f"Skipping {file_name}: Could not identify both sparse and dense retrievers.")
                    continue
                    
                sparse_file = f"dataset/{dataset_name}/search_results/{sparse_name}/top{top_k}/results_{split}.trec"
                dense_file = f"dataset/{dataset_name}/search_results/{dense_name}/top{top_k}/results_{split}.trec"
                _base_results_dir = os.environ.get("BASE_RESULTS_DIR", "/extra/huaiyaom0/tr-intern/wrrf/results")
                output_file = f"{_base_results_dir}/{subfolder}/{dataset_name}_{sparse_name}_{dense_name}_{split}.trec"
            
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                fuse_search_results_predicted_weights(
                    sparse_file=sparse_file,
                    dense_file=dense_file,
                    weight_csv_file=file_path,
                    output_file=output_file,
                    run_id=subfolder
                )
                print(f"Completed {dataset_name} using predicted weights ({sparse_name} + {dense_name})")