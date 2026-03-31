import os
import sys
import json
import asyncio
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from helper_5_ir_metrics import process_single_dataset

import boto3
from labs_pl_common.llm import Proxy, SystemPrompt, UserPrompt, KnownModel, ModelProps, ProxyProps
from labs_pl_common.gcs import GCSProps
from labs_pl_common.aws import AWSProps


# ---------------------------------------------------------------------------
# LLM client abstraction (supports TR Proxy and direct OpenAI-compatible endpoints)
# ---------------------------------------------------------------------------

class LLMClient:
    """Thin wrapper so get_llm_scores works with any backend."""

    @staticmethod
    async def from_proxy(gcs, aws, proxy_props, model_props):
        client = LLMClient()
        client._backend = "proxy"
        client._proxy = await Proxy.get(gcs=gcs, aws=aws, proxy=proxy_props, model=model_props)
        return client

    @staticmethod
    def from_bedrock(model_id, aws_profile, region="us-east-1"):
        client = LLMClient()
        client._backend = "bedrock"
        session = boto3.Session(profile_name=aws_profile)
        client._bedrock = session.client("bedrock-runtime", region_name=region)
        client._model_id = model_id
        return client

    async def invoke(self, system_message, user_message):
        if self._backend == "proxy":
            messages = [
                SystemPrompt(system_message),
                UserPrompt(user_message),
            ]
            response = await self._proxy.invoke(messages=messages)
            return response.content
        elif self._backend == "bedrock":
            request_body = {
                "messages": [
                    {"role": "user", "content": f"{system_message}\n\n{user_message}"},
                ],
                "max_tokens": 32,
            }
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._bedrock.invoke_model(
                    modelId=self._model_id,
                    body=json.dumps(request_body),
                    contentType="application/json",
                    accept="application/json",
                ),
            )
            response_body = json.loads(response["body"].read())
            return response_body["choices"][0]["message"]["content"]
        else:
            raise NotImplementedError(f"Backend '{self._backend}' is not yet implemented.")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE = """You are an evaluator assessing the retrieval effectiveness of dense retrieval and sparse retrieval for finding the correct answer.

## Task:
Given a question and two top1 search results (one from sparse retrieval, one from dense retrieval), score each retrieval method from **0 to 5** based on whether the correct answer is likely to appear in top2 , top3 , etc.

### **Scoring Criteria:**
1. **Direct hit --> 5 points**
    - If the retrieved document directly answers the question, assign **5 points**.
2. **Good wrong result (High likelihood correct answer is nearby) --> 3-4 points**
    - If the top1 result is **conceptually close** to the correct answer (e.g., mentions relevant entities, related events, partial answer), it indicates the search method is in the right direction.
    - Give **4** if it's very close, **3** if somewhat close.
3. **Bad wrong result (Low likelihood correct answer is nearby) --> 1-2 points**
    - If the top1 result is **loosely related but misleading** (e.g., shares keywords but changes context), correct answers might not be in top2, top3.
    - Give **2** if there's a small chance correct answers are nearby, **1** if unlikely.
4. **Completely off-track --> 0 points**
    - If the result is **totally unrelated**, it means the retrieval method is failing.

---
### **Given Data :**
- **Question :** "{question}"
- **sparse retrieval Top1 Result:** "{sparse_reference}"
- **dense retrieval Top1 Result:** "{dense_reference}"
---
### ** Output Format :**
Return two integers separated by a space:
- **First number:** sparse retrieval score.
- **Second number:** dense retrieval score.
- Example output: 3 4
(Sparse : 3, Dense : 4)
**Do not output any other text.**"""


def build_user_message(query_text, sparse_doc_text, dense_doc_text):
    return (
        f'- **Question :** "{query_text}"\n'
        f'- **sparse retrieval Top1 Result:** "{sparse_doc_text}"\n'
        f'- **dense retrieval Top1 Result:** "{dense_doc_text}"'
    )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache(cache_file):
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)
            return df.set_index('query_id')[['sparse_score', 'dense_score']].to_dict('index')
        except (pd.errors.EmptyDataError, KeyError):
            return {}
    return {}


def save_to_cache(cache_file, query_id, sparse_score, dense_score):
    entry = pd.DataFrame([{'query_id': query_id, 'sparse_score': sparse_score, 'dense_score': dense_score}])
    header = not os.path.exists(cache_file)
    entry.to_csv(cache_file, mode='a', header=header, index=False)


# ---------------------------------------------------------------------------
# TREC file parsing
# ---------------------------------------------------------------------------

def parse_trec_file(file_path):
    """Returns {query_id: {doc_id: {'rank': int, 'score': float}}}"""
    results = {}
    if not os.path.exists(file_path):
        print(f"Warning: TREC file not found: {file_path}")
        return results
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            query_id, doc_id, rank, score = parts[0], parts[2], int(parts[3]), float(parts[4])
            if query_id not in results:
                results[query_id] = {}
            results[query_id][doc_id] = {'rank': rank, 'score': score}
    return results


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(corpus_path):
    """Returns {doc_id: text} from a jsonl file with _id and text fields."""
    corpus = {}
    with open(corpus_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            doc_id = str(doc['_id'])
            title = doc.get('title', '')
            text = doc.get('text', '')
            corpus[doc_id] = f"{title} {text}".strip() if title else text
    return corpus


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------

def minmax_normalize_scores(query_docs):
    """
    Min-max normalize scores for a single query's retrieved docs.
    query_docs: {doc_id: {'rank': int, 'score': float}}
    Returns: {doc_id: normalized_score}
    """
    if not query_docs:
        return {}
    scores = np.array([v['score'] for v in query_docs.values()])
    min_s, max_s = scores.min(), scores.max()
    if max_s == min_s:
        return {doc_id: 0.0 for doc_id in query_docs}
    return {
        doc_id: (query_docs[doc_id]['score'] - min_s) / (max_s - min_s)
        for doc_id in query_docs
    }


# ---------------------------------------------------------------------------
# Alpha computation
# ---------------------------------------------------------------------------

def compute_alpha(sparse_score, dense_score):
    """
    alpha = sparse_score / (sparse_score + dense_score)
    alpha in [0, 1]; higher alpha = more weight on sparse.
    Falls back to 0.5 if both scores are zero.
    """
    if sparse_score == 0 and dense_score == 0:
        return 0.5
    elif sparse_score == 5 and dense_score < 5:
        return 1.0
    elif dense_score == 5 and sparse_score < 5:
        return 0.0
    else:
        total = sparse_score + dense_score
        if total == 0:
            return 0.5
        return sparse_score / total


# ---------------------------------------------------------------------------
# LLM scoring
# ---------------------------------------------------------------------------

async def get_llm_scores(llm_client, query_id, query_text, sparse_doc_text, dense_doc_text,
                         cache, cache_file, semaphore=None, failed_log_file=None,
                         max_retries=5, initial_delay=2):
    """
    Returns (query_id, sparse_score, dense_score) for a query, using cache if available.
    Retries on failure with exponential backoff. Logs failed queries instead of using fallback.
    Returns None scores if all retries are exhausted.
    """
    if query_id in cache:
        entry = cache[query_id]
        return query_id, entry['sparse_score'], entry['dense_score']

    user_message = build_user_message(query_text, sparse_doc_text, dense_doc_text)

    last_error = None
    for attempt in range(max_retries):
        try:
            async with semaphore if semaphore else asyncio.Semaphore(1):
                content = await llm_client.invoke(SYSTEM_MESSAGE, user_message)
            parts = content.strip().split()
            sparse_score = int(parts[0])
            dense_score = int(parts[1])
            return query_id, sparse_score, dense_score
        except Exception as e:
            last_error = e
            delay = initial_delay * (2 ** attempt)
            print(f"  [query {query_id}] attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)

    print(f"  [query {query_id}] all retries exhausted. Logging as failed.")
    if failed_log_file:
        with open(failed_log_file, 'a') as f:
            f.write(f"{query_id}\t{last_error}\n")
    return query_id, None, None


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

async def run_dynamic_alpha(
    dataset_name,
    sparse_retriever,
    dense_retriever,
    split,
    base_data_dir,
    output_dir,
    llm_client,
    model_name,
    top_k=200,
    concurrency=20,
    batch_size=60,
):
    """
    For one dataset × retriever combo × split:
      1. Parse TREC files for sparse and dense retrievers
      2. Load corpus for doc texts
      3. For each query: min-max normalize scores, call LLM for (sparse_score, dense_score),
         compute alpha, compute final hybrid score per doc, rank docs
      4. Write ranked results to TREC output file
    """
    combo = f"{sparse_retriever}_vs_{dense_retriever}"
    print(f"\n[{dataset_name}] {combo} | split={split}")

    sparse_trec = os.path.join(
        base_data_dir, dataset_name, "search_results", sparse_retriever, f"top{top_k}", f"results_{split}.trec"
    )
    dense_trec = os.path.join(
        base_data_dir, dataset_name, "search_results", dense_retriever, f"top{top_k}", f"results_{split}.trec"
    )
    corpus_path = os.path.join(base_data_dir, dataset_name, "corpus.jsonl")
    queries_path = os.path.join(base_data_dir, dataset_name, "queries.jsonl")

    sparse_results = parse_trec_file(sparse_trec)
    dense_results = parse_trec_file(dense_trec)

    if not sparse_results or not dense_results:
        print(f"  Skipping: missing TREC files.")
        return

    print(f"  Loading corpus from {corpus_path}...")
    corpus = load_corpus(corpus_path)

    print(f"  Loading queries from {queries_path}...")
    queries = {}
    with open(queries_path, 'r') as f:
        for line in f:
            q = json.loads(line)
            queries[str(q['_id'])] = q['text']

    cache_dir = os.path.join(output_dir, "llm_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{dataset_name}_{model_name}_{combo}_{split}_cache.csv")
    failed_log_file = os.path.join(cache_dir, f"{dataset_name}_{model_name}_{combo}_{split}_failed.tsv")
    cache = load_cache(cache_file)

    all_query_ids = sorted(set(sparse_results.keys()) | set(dense_results.keys()))

    query_inputs = []
    for query_id in all_query_ids:
        sparse_q = sparse_results.get(query_id, {})
        dense_q = dense_results.get(query_id, {})
        sparse_top1 = min(sparse_q, key=lambda d: sparse_q[d]['rank']) if sparse_q else None
        dense_top1 = min(dense_q, key=lambda d: dense_q[d]['rank']) if dense_q else None
        sparse_doc_text = corpus.get(sparse_top1, "") if sparse_top1 else ""
        dense_doc_text = corpus.get(dense_top1, "") if dense_top1 else ""
        query_inputs.append((query_id, queries.get(query_id, ""), sparse_doc_text, dense_doc_text))

    n_batches = (len(query_inputs) + batch_size - 1) // batch_size
    print(f"  Processing {len(query_inputs)} queries in {n_batches} batches (batch_size={batch_size}, concurrency={concurrency})...")

    llm_scores = {}
    for i in tqdm(range(0, len(query_inputs), batch_size), desc=f"{dataset_name}/{combo}/{split}", total=n_batches):
        batch = query_inputs[i:i + batch_size]
        semaphore = asyncio.Semaphore(concurrency)
        batch_tasks = [
            get_llm_scores(llm_client, qid, qt, sd, dd, cache, cache_file, semaphore, failed_log_file)
            for qid, qt, sd, dd in batch
        ]
        batch_results = await asyncio.gather(*batch_tasks)
        for qid, s, d in batch_results:
            if s is not None and d is not None:
                save_to_cache(cache_file, qid, s, d)
            llm_scores[qid] = (s, d)

    fused_rows = []

    for query_id in tqdm(all_query_ids, desc=f"{dataset_name}/{combo}/{split}"):
        sparse_q = sparse_results.get(query_id, {})
        dense_q = dense_results.get(query_id, {})

        norm_sparse = minmax_normalize_scores(sparse_q)
        norm_dense = minmax_normalize_scores(dense_q)

        sparse_score, dense_score = llm_scores[query_id]
        if sparse_score is None or dense_score is None:
            continue
        alpha = compute_alpha(sparse_score, dense_score)

        all_docs = set(norm_sparse.keys()) | set(norm_dense.keys())
        doc_final_scores = {}
        for doc_id in all_docs:
            s = norm_sparse.get(doc_id, 0.0)
            d = norm_dense.get(doc_id, 0.0)
            doc_final_scores[doc_id] = alpha * s + (1 - alpha) * d

        ranked = sorted(doc_final_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, score) in enumerate(ranked, 1):
            fused_rows.append((query_id, doc_id, rank, score))

    os.makedirs(output_dir, exist_ok=True)
    run_id = f"dynamic_alpha_{model_name}"
    output_file = os.path.join(output_dir, f"{dataset_name}_{combo}_{split}.trec")
    with open(output_file, 'w') as f:
        for query_id, doc_id, rank, score in fused_rows:
            f.write(f"{query_id} Q0 {doc_id} {rank} {score:.8f} {run_id}\n")

    print(f"  Output written to: {output_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    BASE_DATA_DIR = "/Users/a6128162/Repos/query-based-rrf/data/input"
    OUTPUT_DIR = "/Users/a6128162/Repos/query-based-rrf/data/output/dynamic-alpha-tuning-qwen3_32"

    BACKEND = "bedrock"  # "proxy" or "bedrock"

    if BACKEND == "proxy":
        MODEL = KnownModel.GPT_5_2
        gcs = GCSProps(
            AUTH_URL="https://entitlement-qa.gcs.int.thomsonreuters.com/v1/token",
            AUTH_SECRET_ID="a207918-labs-plsearchmod-ras-search-credentials",
        )
        aws = AWSProps()
        proxy_props = ProxyProps(
            PROFILE="practical_chat_profile2",
            URL="https://llm-proxy-qa.4649.aws-int.thomsonreuters.com",
        )
        model_props = ModelProps(MODEL=MODEL, MAX_TOKENS=256)
        llm_client = await LLMClient.from_proxy(gcs, aws, proxy_props, model_props)
        model_name = MODEL.name
    else:
        llm_client = LLMClient.from_bedrock(
            model_id="qwen.qwen3-32b-v1:0",
            aws_profile="a204383-ml-workspace-practicallawqw7t-prod-use1",
        )
        model_name = "qwen3-32b"

    DATASETS = [
        # ("msmarco",             "dev"),
        # ("nq",                  "dev"),
        # ("acord-entire-corpus", "test"),
        ("nfcorpus",            "test"),
    ]

    SPARSE_RETRIEVERS = [ "bm25", "rm3" ]
    DENSE_RETRIEVERS  = ["biencoder", "qwen3"]

    for dataset_name, split in DATASETS:
        for sparse_retriever in SPARSE_RETRIEVERS:
            for dense_retriever in DENSE_RETRIEVERS:
                await run_dynamic_alpha(
                    dataset_name=dataset_name,
                    sparse_retriever=sparse_retriever,
                    dense_retriever=dense_retriever,
                    split=split,
                    base_data_dir=BASE_DATA_DIR,
                    output_dir=os.path.join(OUTPUT_DIR, dataset_name),
                    llm_client=llm_client,
                    model_name=model_name,
                    top_k=200,
                )

        # Calculate the evaluation metrics for all generated TREC files for this dataset and split
        dataset_output_dir = os.path.join(OUTPUT_DIR, dataset_name)
        qrels_path = os.path.join(BASE_DATA_DIR, dataset_name, "qrels", f"{split}.tsv")
        combo_results = []
        for sparse_retriever in SPARSE_RETRIEVERS:
            for dense_retriever in DENSE_RETRIEVERS:
                trec_file = os.path.join(dataset_output_dir, f"{dataset_name}_{sparse_retriever}_vs_{dense_retriever}_{split}.trec")
                if os.path.exists(trec_file) and os.path.exists(qrels_path):
                    result = process_single_dataset(dataset_name, split, sparse_retriever, dense_retriever, trec_file, qrels_path)
                    combo_results.append(result)

        if combo_results:
            import csv
            metrics_csv = os.path.join(dataset_output_dir, "metrics.csv")
            with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=combo_results[0].keys())
                writer.writeheader()
                writer.writerows(combo_results)
            print(f"\nMetrics saved to: {metrics_csv}")


if __name__ == "__main__":
    asyncio.run(main())
