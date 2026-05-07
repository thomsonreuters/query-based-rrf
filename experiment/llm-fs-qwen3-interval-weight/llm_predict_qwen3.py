import pandas as pd
import torch
import os
import json
import time
import bm25s
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import re

WARMUP_ITERS = 3

# Enable TF32 on Ampere+ GPUs for any fp32 matmul (consistent with the encoder paths;
# negligible effect here since most compute is bf16/4-bit, but kept for hygiene).
torch.set_float32_matmul_precision('high')

PROMPT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "interval_weight_prompt.json")
with open(PROMPT_CONFIG_PATH, "r") as f:
    PROMPT_CONFIG = json.load(f)

# --- Classes and Methods ---

def retrieve_top_k_qwen_batch(query_texts, model, corpus_embeddings, metadata_df, k=10):
    """Batched version of dense retrieval"""
    query_embeddings = model.encode(
        query_texts, 
        convert_to_tensor=True, 
        normalize_embeddings=True
    )
    
    # corpus_embeddings is already on device in the main loop
    cos_scores = util.cos_sim(query_embeddings, corpus_embeddings)
    top_results = torch.topk(cos_scores, k=k, dim=1)
    
    batch_results = []
    for b_idx in range(len(query_texts)):
        scores = top_results.values[b_idx]
        indices = top_results.indices[b_idx]
        
        results = []
        for score, idx in zip(scores, indices):
            idx = idx.item()
            match_info = metadata_df.iloc[idx].to_dict()
            match_info['retrieval_score'] = round(score.item(), 4)
            match_info['source'] = 'qwen'
            results.append(match_info)
        batch_results.append(results)
        
    return batch_results

class BM25Retriever:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        if 'query_id' in self.df.columns:
            self.df['query_id'] = self.df['query_id'].astype(str)
        self.df['query_text'] = self.df['query_text'].fillna("").astype(str)
        
        self.target_metric = 'highest_ndcg@10' if 'highest_ndcg@10' in self.df.columns else 'highest_mrr@10'
        
        corpus_texts = self.df['query_text'].tolist()
        corpus_tokens = bm25s.tokenize(corpus_texts)
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)

    def search_similar_queries_batch(self, input_query_texts, top_k=10):
        """Batched version of BM25 retrieval"""
        query_tokens = bm25s.tokenize(input_query_texts)
        doc_indices, scores = self.retriever.retrieve(query_tokens, k=top_k)
        
        batch_results = []
        for b_idx in range(len(input_query_texts)):
            results = []
            for i in range(top_k):
                idx = int(doc_indices[b_idx, i])
                score = float(scores[b_idx, i])
                row = self.df.iloc[idx]
                
                results.append({
                    "retrieval_score": score,
                    "query_id": str(row['query_id']),
                    "query_text": row['query_text'],
                    self.target_metric: row.get(self.target_metric, None),
                    "friendly_best_weights": row.get('friendly_best_weights', None),
                    "source": "bm25"
                })
            batch_results.append(results)
            
        return batch_results

# --- Prediction Logic ---

def generate_prediction_prompt(test_query, context_queries, metric_name):
    """Formats the context and the prompt for Qwen to predict the weight."""

    system_prompt = PROMPT_CONFIG["system_prompt"]

    user_prompt = PROMPT_CONFIG["user_context_header"].format(metric_name=metric_name)

    for i, q in enumerate(context_queries, 1):
        user_prompt += PROMPT_CONFIG["user_context_item"].format(
            index=i,
            query_text=q['query_text'],
            friendly_best_weights=q['friendly_best_weights'],
            metric_name=metric_name,
            metric_value=q.get(metric_name, 'N/A'),
        )

    user_prompt += PROMPT_CONFIG["user_task"]
    user_prompt += PROMPT_CONFIG["user_target"].format(test_query=test_query)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return messages

def extract_weight(response_text):
    """Attempts to parse a float between 0 and 1 from the model output using Regex."""
    try:
        # input(f"response text is: {response_text}, press to continue ....")
        match = re.search(r'\d*\.\d+', response_text)
        
        if match:
            val = float(match.group())
            if 0.0 <= val <= 1.0:
                return round(val, 2)
            else:
                 return max(0.0, min(1.0, round(val, 2)))
        
        match_int = re.search(r'\b[01]\b', response_text)
        if match_int:
             return float(match_int.group())

        return 0.50

    except Exception:
        return 0.50


def main():
    # --- Global Configuration ---
    DATASETS = ["acord-entire-corpus", "nfcorpus", "nq", "msmarco"]
    COMBINATIONS = ["bm25_vs_biencoder", "bm25_vs_qwen3", "rm3_vs_biencoder", "rm3_vs_qwen3"]
    TOP_K = 5
    BATCH_SIZE = 1 # Process queries in batches for massive speedup
    MODEL = "Qwen/Qwen3-32B"
    
    # 1. Load Heavy Models ONCE before the loops
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print(F"Loading LLM {MODEL} for weight prediction...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    # Set padding configurations for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 
    
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        quantization_config=quantization_config
    )
    
    print("Loading Sentence Transformer (Qwen3) for embedding retrieval...")
    qwen_embedder = SentenceTransformer('Qwen/Qwen3-8B', device="cuda:0", trust_remote_code=True)

    # 2. Iterate through Datasets and Combinations
    for DATASET in DATASETS:
        for COMBINATION in COMBINATIONS:
            print(f"\n{'='*60}")
            print(f"Processing Dataset: {DATASET} | Combination: {COMBINATION}")
            print(f"{'='*60}\n")
            
            # --- Dynamic Configuration per loop ---
            SPLIT = "test" if DATASET in ["acord-entire-corpus", "nfcorpus"] else "dev"
            metric = "ndcg" if DATASET in ["acord-entire-corpus", "nfcorpus"] else "mrr"
            # model_folder = MODEL.split("/")[-1].lower()

            BASE_TRAIN_DIR = f"../../dataset/{DATASET}/qwen3-embeddings/train/top200"
            TRAIN_EMBEDDINGS_PATH = os.path.join(BASE_TRAIN_DIR, "train_query_embeddings.pt")
            TRAIN_METADATA_PATH = os.path.join(BASE_TRAIN_DIR, f"train_{COMBINATION}_query_metadata.pkl")
            TRAIN_CSV_PATH = f"../../dataset/{DATASET}/{metric}_runs/train/top200/results_train_{COMBINATION}_best_weights_final_mean_with_text.csv"
            
            TEST_CSV_PATH = f"../../dataset/{DATASET}/{metric}_runs/{SPLIT}/top200/results_{SPLIT}_{COMBINATION}_best_weights_final_mean_with_text.csv"
            OUTPUT_PATH = f"predictions/{DATASET}_{SPLIT}_{COMBINATION}_predictions.csv"

            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

            # --- Safety Checks for Files ---
            missing_files = []
            for path in [TEST_CSV_PATH, TRAIN_CSV_PATH, TRAIN_EMBEDDINGS_PATH, TRAIN_METADATA_PATH]:
                if not os.path.exists(path):
                    missing_files.append(path)
            
            if missing_files:
                print(f"Skipping {DATASET} - {COMBINATION} due to missing files:")
                for f in missing_files:
                    print(f"  - {f}")
                continue

            # --- Load Dataset-Specific Data ---
            print("Loading test dataset...")
            test_df = pd.read_csv(TEST_CSV_PATH)
            if 'query_id' in test_df.columns:
                test_df['query_id'] = test_df['query_id'].astype(str)
            
            target_metric = 'highest_ndcg@10' if 'highest_ndcg@10' in test_df.columns else 'highest_mrr@10'

            print("Initializing BM25 Retriever...")
            bm25_retriever = BM25Retriever(TRAIN_CSV_PATH)

            print("Loading Corpus Embeddings and Metadata...")
            # Load directly to the GPU once to avoid transfer overhead per query.
            # Cast to bf16 so it matches the embedder's encode() output dtype.
            corpus_embeddings = torch.load(TRAIN_EMBEDDINGS_PATH).to(
                qwen_embedder.device, dtype=torch.bfloat16
            )
            metadata_df = pd.read_pickle(TRAIN_METADATA_PATH)

            # --- Prediction Loop ---
            predictions = []
            print(f"Starting prediction process for {len(test_df)} queries...")

            def _predict_one(query_text):
                """Full per-query pipeline: retrieve similar → prompt → tokenize → generate → parse."""
                bm25_hits = bm25_retriever.search_similar_queries_batch([query_text], top_k=TOP_K)[0]
                dense_hits = retrieve_top_k_qwen_batch(
                    [query_text], qwen_embedder, corpus_embeddings, metadata_df, k=TOP_K
                )[0]

                merged_dict = {}
                for hit in bm25_hits + dense_hits:
                    qid = str(hit['query_id'])
                    if qid not in merged_dict:
                        merged_dict[qid] = hit
                    else:
                        merged_dict[qid]['source'] += f" & {hit['source']}"
                context_queries = list(merged_dict.values())

                messages = generate_prediction_prompt(query_text, context_queries, target_metric)
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
                input_length = inputs.input_ids.shape[1]

                with torch.inference_mode():
                    outputs = llm.generate(
                        **inputs,
                        max_new_tokens=10,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )

                generated_tokens = outputs[:, input_length:]
                response_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                return extract_weight(response_text)

            # Warmup (not timed) — primes CUDA kernels and any first-call caches.
            if len(test_df) > 0:
                warm_text = str(test_df.iloc[0]['query_text'])
                for _ in range(WARMUP_ITERS):
                    _predict_one(warm_text)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            # Per-query timed loop
            for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
                query_id = row['query_id']
                query_text = str(row['query_text'])
                actual_weight = row['friendly_best_weights'] if 'friendly_best_weights' in test_df.columns else None
                tgt_metric = row[target_metric] if target_metric in test_df.columns else None

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                predicted_weight = _predict_one(query_text)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                latency_ms = (time.perf_counter() - t0) * 1000.0

                predictions.append({
                    "query_id": query_id,
                    "query_text": query_text,
                    "predicted": predicted_weight,
                    "actual": actual_weight,
                    target_metric: tgt_metric,
                    "latency_ms": latency_ms,
                })
                

            # --- Save Results ---
            print(f"Saving predictions to {OUTPUT_PATH}...")
            results_df = pd.DataFrame(predictions)
            results_df.to_csv(OUTPUT_PATH, index=False)
            print(f"Finished processing {DATASET} - {COMBINATION}!\n")

    print("All datasets and combinations completed successfully!")

if __name__ == "__main__":
    main()