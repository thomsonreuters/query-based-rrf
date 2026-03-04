import os

def filter_trec_file(input_file, output_file, max_results=200):
    """
    Filter TREC file to keep only first max_results for each query
    
    Args:
        input_file: path to input TREC file
        output_file: path to output TREC file
        max_results: maximum number of results to keep per query (default: 200)
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    query_counts = {}
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
                
            # Parse TREC format: query_id Q0 doc_id rank score run_name
            parts = line.split()
            if len(parts) >= 6:
                query_id = parts[0]
                
                # Count results for this query
                if query_id not in query_counts:
                    query_counts[query_id] = 0
                
                # Only keep if under the limit
                if query_counts[query_id] < max_results:
                    outfile.write(line + '\n')
                    query_counts[query_id] += 1

# Usage example
if __name__ == "__main__":
    top_k = 200
    dataset = "acord-entire-corpus"  # Options: acord, msmarco, nq, nfcorpus, trec-covid
    split = "dev"
    retriever = "qwen3" # bm25, biencoder, rm3, qwen3
    input_file = f"dataset/{dataset}/search_results/{retriever}/top1000/results_{split}.trec"  # Replace with your input file path
    # input_file = f"dataset/{dataset}/search_results/{retriever}/top1000/results_{split}.trec"

    
    output_file = f"dataset/{dataset}/search_results/{retriever}/top{top_k}/results_{split}.trec"  # Replace with your output file path
    # output_file = f"experiments/roberta/roberta-experiment-1-mean-best-weight/experiments/trained-on-specific-dataset-top400/prediction-wrrf-results/top200-after-wrrf/{dataset}_{split}.trec"

    filter_trec_file(input_file, output_file, top_k)
    print(f"Filtered TREC file saved to {output_file}")