import pandas as pd
import numpy as np
import torch
import os
import json
import time
import bm25s
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sentence_transformers import SentenceTransformer, util
from llm_backend import BedrockBackend, LocalQwen3Backend, LocalMistralBackend
from tqdm import tqdm
import re

PROMPT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "mean_best_weight_prompt.json")
with open(PROMPT_CONFIG_PATH, "r") as f:
    PROMPT_CONFIG = json.load(f)

TOP_K = 5


def _retrieve_dense_one(query_text, model, corpus_embeddings, metadata_df, k):
    query_embedding = model.encode([query_text], convert_to_tensor=True, normalize_embeddings=True)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)
    top_results = torch.topk(cos_scores, k=k, dim=1)
    results = []
    for score, idx in zip(top_results.values[0], top_results.indices[0]):
        match_info = metadata_df.iloc[idx.item()].to_dict()
        match_info['retrieval_score'] = round(score.item(), 4)
        match_info['source'] = 'qwen'
        results.append(match_info)
    return results


class BM25Retriever:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        if 'query_id' in self.df.columns:
            self.df['query_id'] = self.df['query_id'].astype(str)
        self.df['query_text'] = self.df['query_text'].fillna("").astype(str)
        self.target_metric = 'highest_ndcg@10' if 'highest_ndcg@10' in self.df.columns else 'highest_mrr@10'
        corpus_tokens = bm25s.tokenize(self.df['query_text'].tolist())
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)

    def search_one(self, query_text, top_k):
        query_tokens = bm25s.tokenize([query_text])
        doc_indices, scores = self.retriever.retrieve(query_tokens, k=top_k)
        results = []
        for i in range(top_k):
            idx = int(doc_indices[0, i])
            row = self.df.iloc[idx]
            results.append({
                "retrieval_score": float(scores[0, i]),
                "query_id": str(row['query_id']),
                "query_text": row['query_text'],
                self.target_metric: row.get(self.target_metric, None),
                "mean_best_weight": row.get('mean_best_weight', None),
                "source": "bm25",
            })
        return results


def generate_prediction_prompt(test_query, context_queries, metric_name):
    system_prompt = PROMPT_CONFIG["system_prompt"]
    user_prompt = PROMPT_CONFIG["user_context_header"].format(metric_name=metric_name)
    for i, q in enumerate(context_queries, 1):
        user_prompt += PROMPT_CONFIG["user_context_item"].format(
            index=i,
            query_text=q['query_text'],
            mean_best_weight=q['mean_best_weight'],
            metric_name=metric_name,
            metric_value=q.get(metric_name, 'N/A'),
        )
    user_prompt += PROMPT_CONFIG["user_task"]
    user_prompt += PROMPT_CONFIG["user_target"].format(test_query=test_query)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def extract_weight(response_text):
    try:
        match = re.search(r'\d*\.\d+', response_text)
        if match:
            val = float(match.group())
            return round(max(0.0, min(1.0, val)), 2)
        match_int = re.search(r'\b[01]\b', response_text)
        if match_int:
            return float(match_int.group())
        return 0.50
    except Exception:
        return 0.50


def _run_one(
    query_text,
    bm25_retriever,
    qwen_embedder,
    corpus_embeddings,
    metadata_df,
    llm_backend,
    target_metric,
):
    t0 = time.perf_counter()
    bm25_hits = bm25_retriever.search_one(query_text, top_k=TOP_K)
    dense_hits = _retrieve_dense_one(query_text, qwen_embedder, corpus_embeddings, metadata_df, k=TOP_K)

    merged_dict = {}
    for hit in bm25_hits + dense_hits:
        qid = str(hit['query_id'])
        if qid not in merged_dict:
            merged_dict[qid] = hit
        else:
            merged_dict[qid]['source'] += f" & {hit['source']}"
    context_queries = list(merged_dict.values())

    messages = generate_prediction_prompt(query_text, context_queries, target_metric)

    response_text = llm_backend.generate(messages)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return extract_weight(response_text), latency_ms


def main():
    DATASETS = ["acord-entire-corpus", "nfcorpus", "nq", "msmarco"]
    COMBINATIONS = ["bm25_vs_biencoder", "bm25_vs_qwen3", "rm3_vs_biencoder", "rm3_vs_qwen3"]
    BACKEND = "local_qwen3"  # "bedrock", "local_qwen3", or "local_mistral"

    if BACKEND == "bedrock":
        llm_backend = BedrockBackend(model_id="qwen.qwen3-32b-v1:0")
    elif BACKEND == "local_qwen3":
        llm_backend = LocalQwen3Backend(model="Qwen/Qwen3-32B", max_new_tokens=10)
    elif BACKEND == "local_mistral":
        llm_backend = LocalMistralBackend(model="mistralai/Ministral-3-14B-Instruct-2512", max_new_tokens=100)
    else:
        raise ValueError(f"Unknown BACKEND: {BACKEND}")

    _base = os.environ.get("BASE_DATA_DIR", "/home/sagemaker-user/query-aware-rrf/query-based-rrf/dataset")
    _output_base = os.environ.get("OUTPUT_DIR", os.path.dirname(os.path.abspath(__file__)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading Sentence Transformer (Qwen3-8B) for embedding retrieval...")
    qwen_embedder = SentenceTransformer(
        'Qwen/Qwen3-8B',
        device=device,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )

    for DATASET in DATASETS:
        for COMBINATION in COMBINATIONS:
            print(f"\n{'='*60}")
            print(f"Processing Dataset: {DATASET} | Combination: {COMBINATION}")
            print(f"{'='*60}\n")

            SPLIT = "test" if DATASET in ["acord-entire-corpus", "nfcorpus"] else "dev"
            metric = "ndcg" if DATASET in ["acord-entire-corpus", "nfcorpus"] else "mrr"

            BASE_TRAIN_DIR = f"{_base}/{DATASET}/qwen3-embeddings/train/top200"
            TRAIN_EMBEDDINGS_PATH = os.path.join(BASE_TRAIN_DIR, "train_query_embeddings.pt")
            TRAIN_METADATA_PATH = os.path.join(BASE_TRAIN_DIR, f"train_{COMBINATION}_query_metadata.pkl")
            TRAIN_CSV_PATH = f"{_base}/{DATASET}/{metric}_runs/train/top200/results_train_{COMBINATION}_best_weights_final_mean_with_text.csv"
            TEST_CSV_PATH = f"{_base}/{DATASET}/{metric}_runs/{SPLIT}/top200/results_{SPLIT}_{COMBINATION}_best_weights_final_mean_with_text.csv"

            OUTPUT_PATH = os.path.join(_output_base, "predictions", f"{DATASET}_{SPLIT}_{COMBINATION}_predictions.csv")
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

            missing = [p for p in [TEST_CSV_PATH, TRAIN_CSV_PATH, TRAIN_EMBEDDINGS_PATH, TRAIN_METADATA_PATH] if not os.path.exists(p)]
            if missing:
                print(f"Skipping {DATASET} - {COMBINATION} due to missing files:")
                for f in missing:
                    print(f"  - {f}")
                continue

            print("Loading test dataset...")
            test_df = pd.read_csv(TEST_CSV_PATH)
            if 'query_id' in test_df.columns:
                test_df['query_id'] = test_df['query_id'].astype(str)
            target_metric = 'highest_ndcg@10' if 'highest_ndcg@10' in test_df.columns else 'highest_mrr@10'

            print("Initializing BM25 Retriever...")
            bm25_retriever = BM25Retriever(TRAIN_CSV_PATH)

            print("Loading corpus embeddings and metadata...")
            corpus_embeddings = torch.load(TRAIN_EMBEDDINGS_PATH, weights_only=False).to(
                device=qwen_embedder.device, dtype=torch.bfloat16
            )
            metadata_df = pd.read_pickle(TRAIN_METADATA_PATH)

            # The pickle metadata only carries friendly_best_weights; pull mean_best_weight from the train CSV.
            metadata_df['query_id'] = metadata_df['query_id'].astype(str)
            mean_weight_lookup = bm25_retriever.df.set_index('query_id')['mean_best_weight']
            metadata_df['mean_best_weight'] = metadata_df['query_id'].map(mean_weight_lookup)

            predictions = []
            latency_ms_list = []
            print(f"Starting prediction for {len(test_df)} queries...")

            for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
                query_id = str(row['query_id'])
                query_text = str(row['query_text'])
                actual_weight = row.get('mean_best_weight', None)
                tgt_metric_val = row.get(target_metric, None)

                predicted, latency_ms = _run_one(
                    query_text,
                    bm25_retriever,
                    qwen_embedder,
                    corpus_embeddings,
                    metadata_df,
                    llm_backend,
                    target_metric,
                )
                latency_ms_list.append(latency_ms)

                predictions.append({
                    "query_id": query_id,
                    "query_text": query_text,
                    "predicted": predicted,
                    "actual": actual_weight,
                    target_metric: tgt_metric_val,
                    "latency_ms": latency_ms,
                })

            print(
                f"Latency [{DATASET}/{COMBINATION}] n={len(latency_ms_list)}, "
                f"mean={np.mean(latency_ms_list):.1f}ms, "
                f"median={np.median(latency_ms_list):.1f}ms, "
                f"p95={np.percentile(latency_ms_list, 95):.1f}ms"
            )

            print(f"Saving predictions to {OUTPUT_PATH}...")
            pd.DataFrame(predictions).to_csv(OUTPUT_PATH, index=False)
            print(f"Finished {DATASET} - {COMBINATION}!\n")

    print("All datasets and combinations completed successfully!")


if __name__ == "__main__":
    main()
