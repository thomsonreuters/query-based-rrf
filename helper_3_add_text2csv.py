import json
import pandas as pd
import os
import io

def read_file_content(file_path):
    return open(file_path, 'r', encoding='utf-8')

def add_query_text_to_csv(json_file_path, csv_file_path):
    """
    Replace placeholder query text in CSV with actual query text from JSONL file based on query_id.
    
    Args:
        json_file_path (str): Path to the queries.jsonl file (local or S3)
        csv_file_path (str): Path to the queries.csv file (local only for now)
    """

    # Read JSONL file and create a dictionary mapping _id to text
    query_text_map = {}
    with read_file_content(json_file_path) as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data = json.loads(line)
                query_text_map[data['_id']] = data['text']
    
    # Read CSV file (assuming local for output)
    df = pd.read_csv(csv_file_path)
    
    # Replace placeholder query_text with actual query text based on query_id
    df['query_text'] = df['query_id'].astype(str).map(query_text_map)
    
    # Create output filename
    file_dir = os.path.dirname(csv_file_path)
    file_name = os.path.basename(csv_file_path)
    name_without_ext = os.path.splitext(file_name)[0]
    file_ext = os.path.splitext(file_name)[1]
    
    output_filename = f"{name_without_ext}_with_text{file_ext}"
    output_file_path = os.path.join(file_dir, output_filename)
    
    # Save the new CSV file
    df.to_csv(output_file_path, index=False)
    
    print(f"New CSV file saved as: {output_file_path}")
    print(f"Replaced query text for {df['query_text'].notna().sum()} out of {len(df)} queries")
    
    # Show queries without matching text
    missing_queries = df[df['query_text'].isna()]
    if len(missing_queries) > 0:
        print(f"Warning: {len(missing_queries)} queries had no matching text in JSON file")
        # print("Missing query IDs:", missing_queries['query_id'].tolist())


# --- Configuration ---
dataset_name = "nfcorpus"
eval_method = "ndcg"
top_k = 200

# Iteration lists
splits = ["dev", "train", "test"]
SPARSE_METHODS = ["rm3", "bm25"] 
DENSE_METHODS = ["biencoder", "qwen3"] 

# Path to the query source (usually constant for the dataset)
query_file = f"dataset/{dataset_name}/queries.jsonl"

# --- Main Iteration Loop ---
for current_split in splits:
    for sparse_method in SPARSE_METHODS:
        for dense_method in DENSE_METHODS:
            
            # Construct the input CSV path based on current combination
            input_csv = (
                f"dataset/{dataset_name}/{eval_method}_runs/{current_split}/top{top_k}/"
                f"results_{current_split}_{sparse_method}_vs_{dense_method}_best_weights_final_mean.csv"
            )
            
            print(f"Processing: {current_split} | {sparse_method} vs {dense_method}")

            # Check if file exists before processing
            if os.path.exists(input_csv):
                try:
                    add_query_text_to_csv(query_file, input_csv)
                except Exception as e:
                    print(f"Error processing {input_csv}: {e}")
            else:
                print(f"Skipping: File not found ({input_csv})")
            
            print("-" * 50)