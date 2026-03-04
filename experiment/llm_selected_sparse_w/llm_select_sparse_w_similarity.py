import pandas as pd
import numpy as np
import ast
import requests
import json
import os
import time
import random
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle
from typing import List, Tuple, Dict
from scipy.stats import wilcoxon

 
class AzureModel:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        workspace_id = "PracticalLawxOhJ"
        payload = {
            "workspace_id": workspace_id,
            "model_name": model_name,
            "oai_access": "apim"
        }
        url = "https://aiplatform.gcs.int.thomsonreuters.com/v1/openai/token"
        try:
            resp = requests.post(url, headers=None, json=payload)
            resp.raise_for_status()
            credentials = resp.json()

            if "openai_key" in credentials:
                self.model = AzureChatOpenAI(
                    openai_api_version="2024-05-01-preview",
                    azure_deployment=model_name,
                    openai_api_type="azure_ad",
                    azure_endpoint=credentials["openai_endpoint"],
                    openai_api_key=credentials["openai_key"],
                    timeout=240,
                    max_retries=3,
                )
            else:
                raise ValueError("Incorrect Workspace ID or Model Name")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching credentials: {e}")
            self.model = None
        except ValueError as e:
            print(f"Error processing credentials: {e}")
            self.model = None

    def run_azure(self, system_message, human_message, max_retries=5, initial_delay=2):
        """
        Calls the Azure LLM with retry logic and exponential backoff.
        Handles content filtering errors by returning a specific marker.
        """
        if not self.model:
            print("Model not initialized. Cannot run query.")
            return None
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]

        for attempt in range(max_retries):
            try:
                message = self.model.invoke(messages)
                return message.content
            except Exception as e:
                # Check for content filter error specifically
                if 'content_filter' in str(e):
                    print(f"Content filter triggered for a query. Skipping with default value. Error: {e}")
                    return "_CONTENT_FILTER_"

                if attempt + 1 == max_retries:
                    print(f"Error during model invocation after {max_retries} attempts: {e}")
                    return None

                # Exponential backoff with jitter for other errors
                delay = (initial_delay * 2 ** attempt) + random.uniform(0, 1)
                print(
                    f"Error during model invocation (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
        return None


class QuerySimilarityRetriever:
    def __init__(self, training_df: pd.DataFrame, cache_dir: str = None):
        """
        Initialize the similarity retriever with training data.
        
        Args:
            training_df: DataFrame with columns ['query_text', 'mean_best_weight', 'query_id']
                        Optionally also 'friendly_best_weights' for range-based methods
            cache_dir: Directory to cache embeddings and BM25 index
        """
        self.training_df = training_df
        self.cache_dir = cache_dir
        
        # Initialize models
        print("Loading sentence transformer model...")
        self.dense_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Prepare training queries
        self.training_queries = training_df['query_text'].tolist()
        self.training_weights = training_df['mean_best_weight'].tolist()
        self.training_ids = training_df['query_id'].tolist()
        
        # Also prepare friendly_best_weights if available
        self.training_friendly_weights = None
        if 'friendly_best_weights' in training_df.columns:
            self.training_friendly_weights = training_df['friendly_best_weights'].tolist()
        
        # Initialize retrievers
        self._setup_dense_retriever()
        self._setup_sparse_retriever()
        
    def _setup_dense_retriever(self):
        """Setup dense retriever with cached embeddings."""
        embeddings_cache_path = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            embeddings_cache_path = os.path.join(self.cache_dir, 'training_embeddings.pkl')
        
        if embeddings_cache_path and os.path.exists(embeddings_cache_path):
            print("Loading cached training embeddings...")
            with open(embeddings_cache_path, 'rb') as f:
                self.training_embeddings = pickle.load(f)
        else:
            print("Computing training query embeddings...")
            self.training_embeddings = self.dense_model.encode(self.training_queries, show_progress_bar=True)
            
            if embeddings_cache_path:
                print(f"Caching embeddings to {embeddings_cache_path}")
                with open(embeddings_cache_path, 'wb') as f:
                    pickle.dump(self.training_embeddings, f)
    
    def _setup_sparse_retriever(self):
        """Setup BM25 sparse retriever."""
        print("Setting up BM25 retriever...")
        # Tokenize queries for BM25
        tokenized_queries = [query.lower().split() for query in self.training_queries]
        self.bm25 = BM25Okapi(tokenized_queries)
    
    def find_similar_queries(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Find similar queries using both dense and sparse retrieval with interleaving.
        
        Args:
            query: Query text to find similar queries for
            top_k: Target number of results (may return top_k + 1 if both methods contribute unique queries in final iteration)
            
        Returns:
            List of tuples: (similar_query_text, weight, query_id)
        """
        # Dense retrieval - get all results sorted by similarity
        query_embedding = self.dense_model.encode([query])
        dense_similarities = np.dot(self.training_embeddings, query_embedding.T).flatten()
        dense_sorted_indices = np.argsort(dense_similarities)[::-1]  # Descending order
        
        # Sparse retrieval - get all results sorted by BM25 score
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        sparse_sorted_indices = np.argsort(sparse_scores)[::-1]  # Descending order
        
        # Interleave results: alternately add 1 from dense and 1 from sparse until we reach top_k
        unique_indices = set()
        result_indices = []  # To maintain order for final results
        dense_idx = sparse_idx = 0
        
        while len(unique_indices) < top_k:
            # Add from dense if available and not already included
            if dense_idx < len(dense_sorted_indices):
                candidate = dense_sorted_indices[dense_idx]
                if candidate not in unique_indices:
                    unique_indices.add(candidate)
                    result_indices.append(candidate)
                dense_idx += 1
            
            # Add from sparse if available, not already included, and we haven't reached top_k
            if len(unique_indices) < top_k and sparse_idx < len(sparse_sorted_indices):
                candidate = sparse_sorted_indices[sparse_idx]
                if candidate not in unique_indices:
                    unique_indices.add(candidate)
                    result_indices.append(candidate)
                sparse_idx += 1
            
            # Break if we can't add any more from either method
            if dense_idx >= len(dense_sorted_indices) and sparse_idx >= len(sparse_sorted_indices):
                break
        
        # Get similar queries with their weights
        similar_queries = []
        for idx in result_indices:
            similar_queries.append((
                self.training_queries[idx],
                self.training_weights[idx],
                self.training_ids[idx]
            ))
        
        return similar_queries
    
    def find_similar_queries_with_ranges(self, query: str, top_k: int = 3) -> List[Tuple[str, str, str]]:
        """
        Find similar queries using both dense and sparse retrieval with interleaving.
        Returns friendly_best_weights instead of mean_best_weight.
        
        Args:
            query: Query text to find similar queries for
            top_k: Target number of results (may return top_k + 1 if both methods contribute unique queries in final iteration)
            
        Returns:
            List of tuples: (similar_query_text, friendly_best_weights, query_id)
        """
        if self.training_friendly_weights is None:
            raise ValueError("friendly_best_weights not available in training data")
        
        # Dense retrieval - get all results sorted by similarity
        query_embedding = self.dense_model.encode([query])
        dense_similarities = np.dot(self.training_embeddings, query_embedding.T).flatten()
        dense_sorted_indices = np.argsort(dense_similarities)[::-1]  # Descending order
        
        # Sparse retrieval - get all results sorted by BM25 score
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        sparse_sorted_indices = np.argsort(sparse_scores)[::-1]  # Descending order
        
        # Interleave results: alternately add 1 from dense and 1 from sparse until we reach top_k
        unique_indices = set()
        result_indices = []  # To maintain order for final results
        dense_idx = sparse_idx = 0
        
        while len(unique_indices) < top_k:
            # Add from dense if available and not already included
            if dense_idx < len(dense_sorted_indices):
                candidate = dense_sorted_indices[dense_idx]
                if candidate not in unique_indices:
                    unique_indices.add(candidate)
                    result_indices.append(candidate)
                dense_idx += 1
            
            # Add from sparse if available, not already included, and we haven't reached top_k
            if len(unique_indices) < top_k and sparse_idx < len(sparse_sorted_indices):
                candidate = sparse_sorted_indices[sparse_idx]
                if candidate not in unique_indices:
                    unique_indices.add(candidate)
                    result_indices.append(candidate)
                sparse_idx += 1
            
            # Break if we can't add any more from either method
            if dense_idx >= len(dense_sorted_indices) and sparse_idx >= len(sparse_sorted_indices):
                break
        
        # Get similar queries with their friendly_best_weights
        similar_queries = []
        for idx in result_indices:
            similar_queries.append((
                self.training_queries[idx],
                self.training_friendly_weights[idx],
                self.training_ids[idx]
            ))
        
        return similar_queries


def load_cache(cache_file):
    """Loads the cache file if it exists, using query_id as the key."""
    if os.path.exists(cache_file):
        try:
            # Use query_id as the index for the cache
            return pd.read_csv(cache_file).set_index('query_id')['score'].to_dict()
        except (pd.errors.EmptyDataError, KeyError):
            # Handle empty file or missing 'query_id' column
            return {}
    return {}


def save_to_cache(cache_file, query_id, score):
    """Saves a single result to the cache file using query_id."""
    new_entry = pd.DataFrame([{'query_id': query_id, 'score': score}])
    # Add header only if the file doesn't exist
    header = not os.path.exists(cache_file)
    new_entry.to_csv(cache_file, mode='a', header=header, index=False)


def get_score_for_weight(row, weight, all_weights_col):
    """Gets the score for a given weight from the specified all_weights column."""
    try:
        # Try the new format first (e.g., 'all_weights_ndcg@10' or 'all_weights_mrr@10')
        if all_weights_col in row and pd.notna(row[all_weights_col]):
            scores = ast.literal_eval(row[all_weights_col])
            if not scores:  # Handle empty list case
                return 0.0
            # Calculate index based on weight, assuming scores are for 0.00, 0.01, ..., 1.00
            idx = int(round(weight * 100))
            if 0 <= idx < len(scores):
                return scores[idx]
        # Fallback to old format (ndcg_distribution with 0.1 increments)
        elif 'ndcg_distribution' in row:
            idx = int(round(weight * 10))
            ndcg_dist = ast.literal_eval(row['ndcg_distribution'])
            if 0 <= idx < len(ndcg_dist):
                return ndcg_dist[idx]
        return 0.0
    except (ValueError, SyntaxError, TypeError, IndexError):
        return 0.0


def get_theoretical_limit(row, highest_score_col):
    """Gets the theoretical limit score for a row from the specified highest score column."""
    try:
        # Try new format first
        if highest_score_col in row:
            return row[highest_score_col]
        # Fallback to old format
        elif 'ndcg_distribution' in row:
            ndcg_dist = ast.literal_eval(row['ndcg_distribution'])
            return max(ndcg_dist) if ndcg_dist else 0.0
        return 0.0
    except (ValueError, SyntaxError, TypeError):
        return 0.0


DOMAIN_DESCRIPTIONS = {
    'acord': "This is an expert-annotated legal IR benchmark for contract clause retrieval (e.g., Limitation of Liability, Indemnification), with 114 queries and 126k query-clause pairs rated 1–5 stars; it focuses on matching nuanced legal drafting precedents.",
    'trec-covid': "This is a biomedical/clinical IR benchmark of COVID-19-related topics and relevance judgments over scientific articles (e.g., CORD-19), emphasizing terminology-heavy, expert-focused queries.",
    'nq': "This is an open-domain QA benchmark from real Google queries with answers grounded in Wikipedia, mixing short factoid and longer explanatory information needs.",
    'nfcorpus': "This is a non-factoid biomedical/health IR dataset with consumer health queries and longer, descriptive answers from curated medical sources.",
    'msmarco': "This is a large-scale web passage/document ranking dataset derived from real Bing queries with human relevance labels, covering diverse, noisy, and conversational search intents."
}

# System message templates for different prompt versions
SYSTEM_MESSAGES = {
    'original': (
        "You are an expert in search relevance. Your task is to predict the optimal 'sparse_w' for a given user query. "
        "The 'sparse_w' is a weight from 0.0 to 1.0 (in 0.01 increments) that balances two search components: a sparse lexical search (like BM25) and a dense vector search. "
        "A sparse_w of 0.0 means 100% dense search, and 1.0 means 100% sparse search. "
        "The search domain is as follows: {domain_description}\n\n"
        "Consider the query's nature: queries with specific keywords, entities, or identifiers might benefit from a higher sparse_w. Broader, more conceptual queries might benefit from a lower sparse_w. "
        "Respond with only the float value for the predicted sparse_w (e.g., 0.72)."
    ),
    'joel': (
        "You have deep knowledge of search queries and their structure. Your task is to come up with the weight for a BM25 retriever in reciprocal rank fusion. "
        "Assign a weight called selected_weight as a value between 0.00 and 1.00 (in 0.01 increments) to the BM25 retriever according to patterns you identify in the query. "
        "1 - selected_weight will be the weight assigned to the dense retriever in the reciprocal rank fusion. "
        "The search domain is as follows: {domain_description}\n\n"
        "For queries that would benefit more from lexical matching to candidate documents due to the presence of entities or other important keywords, assign a higher weight. "
        "For queries that would benefit more from semantic matching to candidate documents, assign a lower weight. "
        "Respond with only the float value for the selected_weight (e.g., 0.72)."
    ),
    'combined': (
        "You are an expert in search relevance. Your task is to set selected_weight, the weight for a BM25 (sparse lexical) retriever within a reciprocal rank fusion of BM25 and a dense vector retriever.\n\n"
        "Definition:\n"
        "selected_weight is a float between 0.00 and 1.00, in 0.01 increments.\n"
        "selected_weight applies to BM25; the dense retriever weight is 1.0 - selected_weight.\n"
        "The dense retriever is sentence-transformers/all-MiniLM-L6-v2.\n\n"
        "Domain Context:\n"
        "{domain_description}\n\n"
        "Guidance:\n"
        "Prefer higher selected_weight for queries with specific terms, identifiers, names, or operators where exact lexical matching matters.\n"
        "Prefer lower selected_weight for broader, concept-driven, or open-ended queries where semantic similarity is more important.\n"
        "Consider query length, presence of entities, and domain-specific identifiers when deciding.\n"
        "Note: all-MiniLM-L6-v2 is a general-purpose encoder; it may underperform on niche jargon and precise identifiers compared to BM25.\n\n"
        "Output:\n"
        "Respond with only the float value for selected_weight (e.g., 0.72). No explanations or extra text."
    ),
    'no_guidance': (
        "You are an expert in search relevance. Your task is to set 'sparse_w', the weight for a BM25 (sparse lexical) retriever within a reciprocal rank fusion of BM25 and a dense vector retriever.\n\n"
        "Definitions:\n"
        "'sparse_w' is a float between 0.00 and 1.00, in 0.01 increments that weights two search components.\n"
        "'sparse_w' applies to BM25; the dense retriever weight is (1.0 - 'sparse_w').\n"
        "A 'sparse_w' of 0.0 means 100% dense search, and 1.0 means 100% sparse search.\n"
        "The dense retriever is sentence-transformers/all-MiniLM-L6-v2.\n\n"
        "Domain Context:\n"
        "{domain_description}\n\n"
        "Output:\n"
        "Respond with only the float value for 'sparse_w' (e.g., 0.72). No explanations or extra text."
    )
}


def evaluate_theoretical_limit(test_df, highest_score_col):
    """Theoretical limit using the best weight for each query."""
    print("Running Theoretical Limit...")
    scores = [get_theoretical_limit(row, highest_score_col) for _, row in test_df.iterrows()]
    return np.mean(scores) if scores else 0.0


def evaluate_baseline(test_df, weight, method_name, all_weights_col):
    """Generic baseline evaluation for fixed weights."""
    print(f"Running {method_name}...")
    scores = [get_score_for_weight(row, weight, all_weights_col) for _, row in test_df.iterrows()]
    return np.mean(scores) if scores else 0.0


def evaluate_llm_method(test_df, azure_model, cache_file, prompt_version, domain_description, all_weights_col,
                        train_df=None, method_name="", train_avg_weight=None):
    """Generic LLM evaluation method that handles both with/without examples."""
    print(f"Running {method_name}...")
    if not azure_model.model:
        return 0.0

    cache = load_cache(cache_file)
    scores = list(cache.values())

    system_message = SYSTEM_MESSAGES[prompt_version].format(domain_description=domain_description)

    if 'query_id' not in test_df.columns:
        print("Error: 'query_id' column not found in test_df. Cannot proceed with LLM evaluation.")
        return 0.0

    cached_ids = set(map(type(test_df['query_id'].iloc[0]), cache.keys()))
    to_process_df = test_df[~test_df['query_id'].isin(cached_ids)]

    if not to_process_df.empty:
        print(f"Processing {len(to_process_df)} queries not found in cache...")
        query_col = 'query_text' if 'query_text' in test_df.columns else 'query'

        examples_str = ""
        if train_df is not None:
            weight_label = 'selected_weight' if prompt_version in ['joel', 'combined'] else 'sparse_w'
            weight_col_name = 'mean_best_weight'
            train_query_col = 'query_text' if 'query_text' in train_df.columns else 'query'

            if weight_col_name in train_df.columns:
                target_weights = np.arange(0, 1.05, 0.05)
                example_rows = []
                for weight in target_weights:
                    closest_row_idx = (train_df[weight_col_name] - weight).abs().idxmin()
                    example_rows.append(train_df.loc[closest_row_idx])

                seen_indices = set()
                unique_example_rows = []
                for row in example_rows:
                    if row.name not in seen_indices:
                        unique_example_rows.append(row)
                        seen_indices.add(row.name)

                examples = [f"Query: {r[train_query_col]}\nBest {weight_label}: {r[weight_col_name]:.2f}"
                            for r in unique_example_rows]
                examples_str = "\n\n".join(examples)
                print(f"Created {len(unique_example_rows)} stratified examples for prompts.")

        for _, row in tqdm(to_process_df.iterrows(), total=len(to_process_df), desc="Querying LLM"):
            query_text = row[query_col]
            query_id = row['query_id']

            human_message = ""
            weight_label = 'selected_weight' if prompt_version in ['joel', 'combined'] else 'sparse_w'

            if train_avg_weight is not None:
                human_message += f"Note: The average best {weight_label} across the training data is {train_avg_weight:.2f}.\n\n"

            if examples_str:
                human_message += f"Here are some examples:\n\n{examples_str}\n\n"

            human_message += f"Now predict for this query:\n{query_text}"

            response = azure_model.run_azure(system_message, human_message)

            if response == "_CONTENT_FILTER_":
                score = get_score_for_weight(row, 0.5, all_weights_col)
            else:
                try:
                    predicted_w = round(float(response.strip()), 2)
                    score = get_score_for_weight(row, predicted_w, all_weights_col)
                except (ValueError, TypeError):
                    print(f"Could not parse LLM response for query_id '{query_id}'. Using 0.5 as default.")
                    score = get_score_for_weight(row, 0.5, all_weights_col)

            scores.append(score)
            save_to_cache(cache_file, query_id, score)

    return np.mean(scores) if scores else 0.0


def evaluate_similarity_based_llm_method(test_df, azure_model, cache_file, prompt_version, domain_description, 
                                        all_weights_col, similarity_retriever, method_name="", top_k=3, train_avg_weight=None):
    """LLM evaluation method using similar query examples from training data."""
    print(f"Running {method_name}...")
    if not azure_model.model:
        return 0.0

    cache = load_cache(cache_file)
    scores = list(cache.values())

    system_message = SYSTEM_MESSAGES[prompt_version].format(domain_description=domain_description)

    if 'query_id' not in test_df.columns:
        print("Error: 'query_id' column not found in test_df. Cannot proceed with LLM evaluation.")
        return 0.0

    cached_ids = set(map(type(test_df['query_id'].iloc[0]), cache.keys()))
    to_process_df = test_df[~test_df['query_id'].isin(cached_ids)]

    if not to_process_df.empty:
        print(f"Processing {len(to_process_df)} queries not found in cache...")
        query_col = 'query_text' if 'query_text' in test_df.columns else 'query'

        for _, row in tqdm(to_process_df.iterrows(), total=len(to_process_df), desc="Querying LLM with Similar Examples"):
            query_text = row[query_col]
            query_id = row['query_id']

            # Find similar queries using the specified top_k
            similar_queries = similarity_retriever.find_similar_queries(query_text, top_k=top_k)
            
            # Format similar examples
            weight_label = 'selected_weight' if prompt_version in ['joel', 'combined'] else 'sparse_w'
            examples = []
            for sim_query, sim_weight, sim_id in similar_queries:
                examples.append(f"Query: {sim_query}\nBest {weight_label}: {sim_weight:.2f}")
            
            examples_str = "\n\n".join(examples)

            human_message = ""
            
            if train_avg_weight is not None:
                human_message += f"Note: The average best {weight_label} across the training data is {train_avg_weight:.2f}.\n\n"

            human_message += f"Here are some similar queries and their optimal weights:\n\n{examples_str}\n\n"
            human_message += f"Now predict for this query:\n{query_text}"

            response = azure_model.run_azure(system_message, human_message)

            if response == "_CONTENT_FILTER_":
                score = get_score_for_weight(row, 0.5, all_weights_col)
            else:
                try:
                    predicted_w = round(float(response.strip()), 2)
                    score = get_score_for_weight(row, predicted_w, all_weights_col)
                except (ValueError, TypeError):
                    print(f"Could not parse LLM response for query_id '{query_id}'. Using 0.5 as default.")
                    score = get_score_for_weight(row, 0.5, all_weights_col)

            scores.append(score)
            save_to_cache(cache_file, query_id, score)

    return np.mean(scores) if scores else 0.0


def evaluate_similarity_based_llm_method_with_ranges(test_df, azure_model, cache_file, prompt_version, domain_description, 
                                                   all_weights_col, similarity_retriever, method_name="", top_k=3, train_avg_weight=None):
    """
    LLM evaluation method using similar query examples with friendly_best_weights ranges instead of mean_best_weight.
    Shows the LLM the actual weight ranges where optimal performance occurred.
    """
    print(f"Running {method_name}...")
    if not azure_model.model:
        return 0.0

    # Check if friendly_best_weights is available
    if similarity_retriever.training_friendly_weights is None:
        print("Error: friendly_best_weights not available in training data. Cannot use range-based method.")
        return 0.0

    cache = load_cache(cache_file)
    scores = list(cache.values())

    system_message = SYSTEM_MESSAGES[prompt_version].format(domain_description=domain_description)

    if 'query_id' not in test_df.columns:
        print("Error: 'query_id' column not found in test_df. Cannot proceed with LLM evaluation.")
        return 0.0

    cached_ids = set(map(type(test_df['query_id'].iloc[0]), cache.keys()))
    to_process_df = test_df[~test_df['query_id'].isin(cached_ids)]

    if not to_process_df.empty:
        print(f"Processing {len(to_process_df)} queries not found in cache...")
        query_col = 'query_text' if 'query_text' in test_df.columns else 'query'

        for _, row in tqdm(to_process_df.iterrows(), total=len(to_process_df), desc="Querying LLM with Range Examples"):
            query_text = row[query_col]
            query_id = row['query_id']

            # Find similar queries with ranges using the new method
            similar_queries = similarity_retriever.find_similar_queries_with_ranges(query_text, top_k=top_k)
            
            # Format similar examples with ranges and explanations
            weight_label = 'selected_weight' if prompt_version in ['joel', 'combined'] else 'sparse_w'
            examples = []
            for sim_query, sim_ranges, sim_id in similar_queries:
                # Parse and format the ranges for better readability
                try:
                    ranges = ast.literal_eval(sim_ranges)
                    if isinstance(ranges, list):
                        if len(ranges) == 1 and isinstance(ranges[0], (int, float)):
                            # Single value case: [0.52]
                            range_desc = f"exactly {ranges[0]:.2f}"
                        else:
                            # Range case: [[0.57, 0.72]] or multiple ranges: [[0.41, 0.45], [0.59, 0.61]]
                            range_parts = []
                            for r in ranges:
                                if isinstance(r, list) and len(r) == 2:
                                    range_parts.append(f"{r[0]:.2f}-{r[1]:.2f}")
                                elif isinstance(r, (int, float)):
                                    range_parts.append(f"{r:.2f}")
                            range_desc = " and ".join(range_parts)
                    else:
                        range_desc = str(sim_ranges)
                except (ValueError, SyntaxError):
                    range_desc = str(sim_ranges)
                
                examples.append(f"Query: {sim_query}\nOptimal {weight_label} range: {range_desc}")
            
            examples_str = "\n\n".join(examples)

            # Enhanced human message with explanation of what ranges represent
            human_message = ""
            
            if train_avg_weight is not None:
                human_message += f"Note: The average best {weight_label} across the training data is {train_avg_weight:.2f}.\n\n"

            human_message += f"""Here are some similar queries and their optimal weight ranges:

{examples_str}

Note: The ranges shown above represent the {weight_label} values where each similar query achieved its best performance (highest NDCG or MRR). When you see a range like "0.57-0.72", it means the query performed optimally across that entire range of weights. When you see a single value like "0.52", it means that specific weight was the single best performer.

Based on these similar examples, what would be the optimal {weight_label} for the following query?

Query: "{query_text}"

Please provide only a single decimal number between 0.0 and 1.0 as your answer."""

            response = azure_model.run_azure(system_message, human_message)

            if response == "_CONTENT_FILTER_":
                score = get_score_for_weight(row, 0.5, all_weights_col)
            else:
                try:
                    predicted_w = round(float(response.strip()), 2)
                    score = get_score_for_weight(row, predicted_w, all_weights_col)
                except (ValueError, TypeError):
                    print(f"Could not parse LLM response for query_id '{query_id}'. Using 0.5 as default.")
                    score = get_score_for_weight(row, 0.5, all_weights_col)

            scores.append(score)
            save_to_cache(cache_file, query_id, score)

    return np.mean(scores) if scores else 0.0


def load_combined_training_data():
    """
    Load and combine training data from nq and msmarco datasets for trec-covid special case.
    Returns combined DataFrame and average weight.
    """
    nq_train_path = '/home/smcilroy/projects/labs_pl-intern-irisma/weighted_rrf/dataset/nq/mrr_runs/train/results_train_best_weights_cleaned.csv'
    msmarco_train_path = '/home/smcilroy/projects/labs_pl-intern-irisma/weighted_rrf/dataset/msmarco/mrr_runs/train/results_train_best_weights_cleaned.csv'
    
    combined_dfs = []
    all_weights = []
    
    # Load NQ training data
    if os.path.exists(nq_train_path):
        print(f"Loading NQ training data from {nq_train_path}...")
        nq_df = pd.read_csv(nq_train_path)
        if 'mean_best_weight' in nq_df.columns:
            # Add dataset source identifier
            nq_df['source_dataset'] = 'nq'
            combined_dfs.append(nq_df)
            all_weights.extend(nq_df['mean_best_weight'].tolist())
            print(f"Loaded {len(nq_df)} NQ training queries")
    
    # Load MSMARCO training data
    if os.path.exists(msmarco_train_path):
        print(f"Loading MSMARCO training data from {msmarco_train_path}...")
        msmarco_df = pd.read_csv(msmarco_train_path)
        if 'mean_best_weight' in msmarco_df.columns:
            # Add dataset source identifier
            msmarco_df['source_dataset'] = 'msmarco'
            combined_dfs.append(msmarco_df)
            all_weights.extend(msmarco_df['mean_best_weight'].tolist())
            print(f"Loaded {len(msmarco_df)} MSMARCO training queries")
    
    if not combined_dfs:
        print("Warning: No training data found for NQ or MSMARCO")
        return None, None
    
    # Combine the datasets
    combined_df = pd.concat(combined_dfs, ignore_index=True)
    combined_avg_weight = np.mean(all_weights) if all_weights else None
    
    print(f"Combined training dataset: {len(combined_df)} total queries")
    print(f"Combined average weight: {combined_avg_weight:.4f}")
    
    return combined_df, combined_avg_weight


def get_dataset_name_from_path(file_path):
    """Extract dataset name from the directory that appears after 'dataset' in the path."""
    path_parts = file_path.replace('\\', '/').split('/')
    
    # Find the index of 'dataset' in the path
    try:
        dataset_index = path_parts.index('dataset')
        # Return the directory name that comes after 'dataset'
        if dataset_index + 1 < len(path_parts):
            return path_parts[dataset_index + 1]
    except ValueError:
        # 'dataset' not found in path
        pass
    
    # Fallback: return filename without extension
    return os.path.basename(file_path).replace('.csv', '')


def validate_dataframe(df, all_weights_col):
    """Validates that the all_weights column has 101 values or is empty."""
    print(f"Validating dataframe using column '{all_weights_col}'...")
    if all_weights_col not in df.columns:
        print(f"Warning: Validation column '{all_weights_col}' not found in dataframe.")
        return

    invalid_rows = []
    for index, row in df.iterrows():
        if pd.notna(row[all_weights_col]):
            try:
                scores = ast.literal_eval(row[all_weights_col])
                if not isinstance(scores, list):
                    invalid_rows.append((index, "Not a list"))
                elif scores and len(scores) != 101:
                    invalid_rows.append((index, f"Incorrect length: {len(scores)}"))
            except (ValueError, SyntaxError):
                print(f"Warning: Could not parse '{all_weights_col}' for row index {index}.")

    if invalid_rows:
        print(f"Warning: Found {len(invalid_rows)} rows where '{all_weights_col}' is invalid.")
        for index, reason in invalid_rows[:5]:
            print(f"  - Row index {index}: {reason}.")
        if len(invalid_rows) > 5:
            print(f"  ... and {len(invalid_rows) - 5} more.")
    else:
        print("Dataframe validation passed.")


def load_llm_method_scores(test_df, cache_file, all_weights_col):
    """
    Load LLM method scores from cache file, matching with test queries.
    Returns array of scores in the same order as test_df.
    """
    if not os.path.exists(cache_file):
        print(f"Warning: Cache file {cache_file} not found")
        return None
    
    try:
        cache_df = pd.read_csv(cache_file)
        if 'query_id' not in cache_df.columns or 'score' not in cache_df.columns:
            print(f"Warning: Cache file {cache_file} missing required columns")
            return None
        
        # Create mapping from query_id to score
        cache_dict = cache_df.set_index('query_id')['score'].to_dict()
        
        # Get scores for test queries in order
        scores = []
        for _, row in test_df.iterrows():
            query_id = row['query_id']
            if query_id in cache_dict:
                scores.append(cache_dict[query_id])
            else:
                # If not in cache, calculate using baseline weight 0.5
                score = get_score_for_weight(row, 0.5, all_weights_col)
                scores.append(score)
        
        return np.array(scores)
    except Exception as e:
        print(f"Error loading cache file {cache_file}: {e}")
        return None


def get_baseline_scores(test_df, weight, all_weights_col):
    """
    Calculate baseline scores for all test queries using a fixed weight.
    Returns array of scores in the same order as test_df.
    """
    scores = []
    for _, row in test_df.iterrows():
        score = get_score_for_weight(row, weight, all_weights_col)
        scores.append(score)
    return np.array(scores)


def perform_wilcoxon_test(llm_scores, baseline_scores, method_name, baseline_name):
    """
    Perform Wilcoxon signed-rank test comparing LLM method to baseline.
    Returns dictionary with test results.
    """
    if llm_scores is None or len(llm_scores) == 0:
        return {
            'method': method_name,
            'baseline': baseline_name,
            'n_queries': 0,
            'llm_mean': 0.0,
            'baseline_mean': 0.0,
            'mean_difference': 0.0,
            'statistic': None,
            'p_value': None,
            'significant': False,
            'error': 'No LLM scores available'
        }
    
    if len(llm_scores) != len(baseline_scores):
        return {
            'method': method_name,
            'baseline': baseline_name,
            'n_queries': len(llm_scores),
            'llm_mean': np.mean(llm_scores),
            'baseline_mean': np.mean(baseline_scores),
            'mean_difference': np.mean(llm_scores) - np.mean(baseline_scores),
            'statistic': None,
            'p_value': None,
            'significant': False,
            'error': 'Score arrays have different lengths'
        }
    
    # Calculate differences
    differences = llm_scores - baseline_scores
    
    # Basic statistics
    llm_mean = np.mean(llm_scores)
    baseline_mean = np.mean(baseline_scores)
    mean_diff = np.mean(differences)
    
    try:
        # Perform Wilcoxon signed-rank test
        # alternative='greater' tests if LLM method is significantly better than baseline
        statistic, p_value = wilcoxon(llm_scores, baseline_scores, alternative='greater')
        
        return {
            'method': method_name,
            'baseline': baseline_name,
            'n_queries': len(llm_scores),
            'llm_mean': llm_mean,
            'baseline_mean': baseline_mean,
            'mean_difference': mean_diff,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'error': None
        }
    
    except Exception as e:
        return {
            'method': method_name,
            'baseline': baseline_name,
            'n_queries': len(llm_scores),
            'llm_mean': llm_mean,
            'baseline_mean': baseline_mean,
            'mean_difference': mean_diff,
            'statistic': None,
            'p_value': None,
            'significant': False,
            'error': str(e)
        }


def compare_methods_to_baselines(test_df, dataset_name, model_name, all_weights_col, train_avg_weight=None):
    """
    Compare all LLM methods to both Hybrid Baseline and Train Average Weight Baseline.
    Returns list of statistical test results.
    """
    print(f"\n=== STATISTICAL ANALYSIS FOR {dataset_name.upper()} ===")
    
    # Define baselines
    baselines = [('Hybrid Baseline', 0.5)]
    if train_avg_weight is not None:
        baselines.append(('Train Avg Weight Baseline', train_avg_weight))
    
    # Define LLM methods to test (based on available cache files)
    cache_dir = os.path.join(os.path.dirname(test_df.attrs.get('source_path', '.')), 'llm_cache')
    if not hasattr(test_df, 'attrs') or 'source_path' not in test_df.attrs:
        # Fallback: try to determine cache directory from dataset name
        if dataset_name == 'trec-covid':
            cache_dir = '/home/smcilroy/projects/labs_pl-intern-irisma/weighted_rrf/dataset/trec-covid/ndcg_runs/test/llm_cache'
        elif dataset_name == 'acord':
            cache_dir = '/home/smcilroy/projects/labs_pl-intern-irisma/weighted_rrf/dataset/acord/ndcg_runs/test/llm_cache'
        elif dataset_name == 'nfcorpus':
            cache_dir = '/home/smcilroy/projects/labs_pl-intern-irisma/weighted_rrf/dataset/nfcorpus/ndcg_runs/test/llm_cache'
        elif dataset_name == 'nq':
            cache_dir = '/home/smcilroy/projects/labs_pl-intern-irisma/weighted_rrf/dataset/nq/mrr_runs/dev/llm_cache'
        elif dataset_name == 'msmarco':
            cache_dir = '/home/smcilroy/projects/labs_pl-intern-irisma/weighted_rrf/dataset/msmarco/mrr_runs/dev/llm_cache'
    
    # Define LLM methods based on available cache files
    llm_methods = [
        ('LLM No_Guidance - No Examples', f'{dataset_name}_{model_name}_no_guidance_no_examples_cache.csv'),
        ('LLM Combined - No Examples', f'{dataset_name}_{model_name}_combined_no_examples_cache.csv'),
        ('LLM No_Guidance - With Stratified Examples', f'{dataset_name}_{model_name}_no_guidance_with_stratified_examples_cache.csv'),
        ('LLM No_Guidance - Similarity Based k=20', f'{dataset_name}_{model_name}_no_guidance_similarity_based_k20_cache.csv'),
        ('LLM No_Guidance - Similarity Ranges k=20', f'{dataset_name}_{model_name}_no_guidance_similarity_ranges_k20_cache.csv')
    ]
    
    statistical_results = []
    
    # Test each LLM method against each baseline
    for method_name, cache_filename in llm_methods:
        cache_file = os.path.join(cache_dir, cache_filename)
        
        # Load LLM scores
        llm_scores = load_llm_method_scores(test_df, cache_file, all_weights_col)
        
        if llm_scores is not None:
            print(f"\nTesting {method_name}:")
            
            for baseline_name, baseline_weight in baselines:
                # Get baseline scores
                baseline_scores = get_baseline_scores(test_df, baseline_weight, all_weights_col)
                
                # Perform statistical test
                result = perform_wilcoxon_test(llm_scores, baseline_scores, method_name, baseline_name)
                statistical_results.append(result)
                
                # Print result
                if result['error']:
                    print(f"  vs {baseline_name}: ERROR - {result['error']}")
                else:
                    significance = "SIGNIFICANT" if result['significant'] else "NOT SIGNIFICANT"
                    print(f"  vs {baseline_name}: p={result['p_value']:.6f} ({significance})")
                    print(f"    LLM mean: {result['llm_mean']:.4f}, Baseline mean: {result['baseline_mean']:.4f}")
                    print(f"    Mean difference: {result['mean_difference']:.4f}")
        else:
            print(f"\nSkipping {method_name}: Cache file not found or invalid")
    
    return statistical_results


def generate_statistical_report(all_statistical_results):
    """
    Generate comprehensive statistical report across all datasets.
    """
    print(f"\n{'='*80}")
    print("COMPREHENSIVE STATISTICAL SIGNIFICANCE REPORT")
    print(f"{'='*80}")
    
    # Group results by dataset
    by_dataset = {}
    for result in all_statistical_results:
        dataset_key = result.get('dataset', 'unknown')
        if dataset_key not in by_dataset:
            by_dataset[dataset_key] = []
        by_dataset[dataset_key].append(result)
    
    # Summary statistics
    total_tests = len(all_statistical_results)
    significant_tests = sum(1 for r in all_statistical_results if r.get('significant', False))
    
    print(f"Total statistical tests performed: {total_tests}")
    print(f"Statistically significant results (p < 0.05): {significant_tests}")
    print(f"Significance rate: {significant_tests/total_tests*100:.1f}%")
    
    # Detailed results by dataset
    for dataset_name, results in by_dataset.items():
        print(f"\n{'-'*60}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'-'*60}")
        
        # Group by method
        by_method = {}
        for result in results:
            method = result['method']
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(result)
        
        for method_name, method_results in by_method.items():
            print(f"\n{method_name}:")
            
            for result in method_results:
                if result['error']:
                    print(f"  vs {result['baseline']}: ERROR - {result['error']}")
                else:
                    status = "✓ SIGNIFICANT" if result['significant'] else "✗ Not significant"
                    print(f"  vs {result['baseline']}: {status}")
                    print(f"    p-value: {result['p_value']:.6f}")
                    print(f"    Mean improvement: {result['mean_difference']:.4f}")
                    print(f"    Sample size: {result['n_queries']} queries")
    
    # Summary of which methods are consistently better
    print(f"\n{'='*60}")
    print("METHODS SIGNIFICANTLY BETTER THAN BOTH BASELINES")
    print(f"{'='*60}")
    
    # Find methods that are significantly better than both baselines across datasets
    method_performance = {}
    
    for result in all_statistical_results:
        if result['error']:
            continue
            
        method = result['method']
        baseline = result['baseline']
        dataset = result.get('dataset', 'unknown')
        
        if method not in method_performance:
            method_performance[method] = {}
        if dataset not in method_performance[method]:
            method_performance[method][dataset] = {}
        
        method_performance[method][dataset][baseline] = result['significant']
    
    # Check which methods beat both baselines
    for method, datasets in method_performance.items():
        consistently_better = []
        for dataset, baselines in datasets.items():
            if len(baselines) >= 2:  # Has results for both baselines
                if all(baselines.values()):  # Significant against all baselines
                    consistently_better.append(dataset)
        
        if consistently_better:
            print(f"\n{method}:")
            print(f"  Significantly better than both baselines on: {', '.join(consistently_better)}")
    
    # NEW: Comprehensive Summary Table
    print(f"\n{'='*80}")
    print("COMPREHENSIVE SUMMARY TABLE")
    print("For each method-dataset combination, shows significance vs baselines:")
    print("H = Hybrid Baseline, T = Train Avg Weight Baseline, B = Both")
    print(f"{'='*80}")
    
    # Organize data for the summary table
    summary_data = {}
    datasets_found = set()
    methods_found = set()
    
    for result in all_statistical_results:
        if result['error']:
            continue
            
        method = result['method']
        baseline = result['baseline']
        dataset = result.get('dataset', 'unknown')
        significant = result['significant']
        
        datasets_found.add(dataset)
        methods_found.add(method)
        
        if method not in summary_data:
            summary_data[method] = {}
        if dataset not in summary_data[method]:
            summary_data[method][dataset] = {}
        
        # Store significance results
        if baseline == 'Hybrid Baseline':
            summary_data[method][dataset]['hybrid'] = significant
        elif baseline == 'Train Avg Weight Baseline':
            summary_data[method][dataset]['train_avg'] = significant
    
    # Sort datasets and methods for consistent display
    sorted_datasets = sorted(datasets_found)
    sorted_methods = sorted(methods_found)
    
    # Print table header
    header = f"{'Method':<50}"
    for dataset in sorted_datasets:
        header += f"{dataset.upper():<12}"
    print(header)
    print("-" * len(header))
    
    # Print each method's results
    for method in sorted_methods:
        row = f"{method:<50}"
        
        for dataset in sorted_datasets:
            if method in summary_data and dataset in summary_data[method]:
                data = summary_data[method][dataset]
                hybrid_sig = data.get('hybrid', False)
                train_avg_sig = data.get('train_avg', False)
                
                # Determine status symbol
                if hybrid_sig and train_avg_sig:
                    status = "B"  # Both baselines
                elif hybrid_sig:
                    status = "H"  # Hybrid only
                elif train_avg_sig:
                    status = "T"  # Train avg only
                else:
                    status = "-"  # Neither
                
                # Handle case where only one baseline exists
                if 'train_avg' not in data:  # Only hybrid baseline tested
                    status = "H" if hybrid_sig else "-"
                elif 'hybrid' not in data:  # Only train avg baseline tested
                    status = "T" if train_avg_sig else "-"
            else:
                status = "N/A"  # No data available
            
            row += f"{status:<12}"
        
        print(row)
    
    # Print legend
    print(f"\nLegend:")
    print(f"  H   = Significantly better than Hybrid Baseline only")
    print(f"  T   = Significantly better than Train Avg Weight Baseline only")
    print(f"  B   = Significantly better than BOTH baselines")
    print(f"  -   = Not significantly better than any baseline")
    print(f"  N/A = No data available for this method-dataset combination")
    
    # Count and report summary statistics for the table
    total_combinations = 0
    both_better = 0
    hybrid_only = 0
    train_only = 0
    neither = 0
    
    for method in summary_data:
        for dataset in summary_data[method]:
            total_combinations += 1
            data = summary_data[method][dataset]
            hybrid_sig = data.get('hybrid', False)
            train_avg_sig = data.get('train_avg', False)
            
            if hybrid_sig and train_avg_sig:
                both_better += 1
            elif hybrid_sig:
                hybrid_only += 1
            elif train_avg_sig:
                train_only += 1
            else:
                neither += 1
    
    if total_combinations > 0:
        print(f"\nSummary Statistics:")
        print(f"  Total method-dataset combinations: {total_combinations}")
        print(f"  Better than both baselines: {both_better} ({both_better/total_combinations*100:.1f}%)")
        print(f"  Better than hybrid only: {hybrid_only} ({hybrid_only/total_combinations*100:.1f}%)")
        print(f"  Better than train avg only: {train_only} ({train_only/total_combinations*100:.1f}%)")
        print(f"  Not better than either: {neither} ({neither/total_combinations*100:.1f}%)")


def run_evaluation(test_csv_path, train_csv_path=None, model_name='gpt-4o', prompt_versions=None):
    """Run all evaluation methods on a dataset."""
    if prompt_versions is None:
        prompt_versions = ['original', 'joel', 'combined', 'no_guidance']
    dataset_name = get_dataset_name_from_path(test_csv_path)
    print(f"\n=== EVALUATING DATASET: {dataset_name} (Model: {model_name}) ===")

    domain_description = DOMAIN_DESCRIPTIONS.get(dataset_name, "a general search task")
    print(f"Domain Context: {domain_description}")

    # Determine metric and column names based on path/training data presence
    if train_csv_path and 'mrr_runs' in test_csv_path:
        print("Dataset type: MRR (with training data)")
        all_weights_col = 'all_weights_mrr@10'
        highest_score_col = 'highest_mrr@10'
        run_type = 'dev'
    else:
        print("Dataset type: NDCG (no training data)")
        all_weights_col = 'all_weights_ndcg@10'
        highest_score_col = 'highest_ndcg@10'
        run_type = 'test'

    try:
        test_df = pd.read_csv(test_csv_path)
    except FileNotFoundError:
        print(f"Error: Test file '{test_csv_path}' not found.")
        return None

    base_dir = os.path.dirname(test_csv_path)
    correct_queries_path = os.path.join(base_dir, f'results_{run_type}_best_weights_cleaned.csv')
    try:
        print(f"Attempting to load correct query texts from '{correct_queries_path}'...")
        correct_queries_df = pd.read_csv(correct_queries_path)

        if 'query_id' in test_df.columns and 'query_id' in correct_queries_df.columns and 'query_text' in correct_queries_df.columns:
            query_text_map = correct_queries_df.set_index('query_id')['query_text'].to_dict()
            original_query_text = test_df['query_text'].copy()
            test_df['query_text'] = test_df['query_id'].map(query_text_map).fillna(test_df['query_text'])
            num_updated = (test_df['query_text'] != original_query_text).sum()
            num_missing = test_df['query_text'].isnull().sum()
            print(f"Successfully updated {num_updated} 'query_text' entries based on 'query_id'.")
            if num_missing > 0:
                print(f"Warning: {num_missing} queries in the test set did not have a matching 'query_id' in the correction file.")
        else:
            print("Warning: Could not perform query text correction. 'query_id' or 'query_text' columns are missing.")
    except FileNotFoundError:
        print(f"Info: Correct query file '{correct_queries_path}' not found. Using original 'query_text'.")
    except Exception as e:
        print(f"An error occurred while loading correct queries: {e}")

    validate_dataframe(test_df, all_weights_col)

    train_df = None
    train_avg_weight = None
    similarity_retriever = None
    
    # Special case handling for trec-covid dataset
    if dataset_name == 'trec-covid':
        print("SPECIAL CASE: trec-covid detected - using combined NQ + MSMARCO training data")
        combined_train_df, combined_avg_weight = load_combined_training_data()
        
        if combined_train_df is not None and combined_avg_weight is not None:
            train_df = combined_train_df
            train_avg_weight = combined_avg_weight
            print(f"Using combined training data for trec-covid. Combined average weight: {train_avg_weight:.4f}")
            
            # Initialize similarity retriever with combined data
            cache_dir = os.path.join(os.path.dirname(test_csv_path), 'similarity_cache')
            similarity_retriever = QuerySimilarityRetriever(train_df, cache_dir)
            print("Similarity retriever initialized with combined NQ + MSMARCO data.")
        else:
            print("Warning: Could not load combined training data for trec-covid.")
    
    elif train_csv_path and os.path.exists(train_csv_path):
        try:
            train_df = pd.read_csv(train_csv_path)
            if 'mean_best_weight' in train_df.columns:
                train_avg_weight = train_df['mean_best_weight'].mean()
                print(f"Training data loaded from {train_csv_path}. Average best weight: {train_avg_weight:.4f}")
                
                # Initialize similarity retriever
                cache_dir = os.path.join(os.path.dirname(test_csv_path), 'similarity_cache')
                similarity_retriever = QuerySimilarityRetriever(train_df, cache_dir)
                print("Similarity retriever initialized successfully.")
            else:
                print("Training data loaded, but 'mean_best_weight' column not found.")
        except FileNotFoundError:
            print(f"Warning: Training file '{train_csv_path}' not found. Running without examples.")

    cache_dir = os.path.join(os.path.dirname(test_csv_path), 'llm_cache')
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Test set size: {len(test_df)}")
    if train_df is not None:
        print(f"Training set size: {len(train_df)}")

    azure_model = AzureModel(model_name)
    results = {}

    results['Theoretical Limit'] = evaluate_theoretical_limit(test_df, highest_score_col)
    print(f"Theoretical Limit Average Score: {results['Theoretical Limit']:.4f}")

    results['Dense Only'] = evaluate_baseline(test_df, 0.0, "Dense Only (sparse_w = 0.0)", all_weights_col)
    print(f"Dense Only Average Score: {results['Dense Only']:.4f}")

    results['Sparse Only'] = evaluate_baseline(test_df, 1.0, "Sparse Only (sparse_w = 1.0)", all_weights_col)
    print(f"Sparse Only Average Score: {results['Sparse Only']:.4f}")

    results['Hybrid Baseline'] = evaluate_baseline(test_df, 0.5, "Hybrid Baseline (sparse_w = 0.5)", all_weights_col)
    print(f"Hybrid Baseline Average Score: {results['Hybrid Baseline']:.4f}")

    if train_avg_weight is not None:
        method_name = "Train Avg Weight Baseline"
        results[method_name] = evaluate_baseline(test_df, train_avg_weight, method_name, all_weights_col)
        print(f"{method_name} (w={train_avg_weight:.2f}) Average Score: {results[method_name]:.4f}")

    # Original LLM methods (for comparison)
    for prompt_version in prompt_versions:
        cache_file = os.path.join(cache_dir, f'{dataset_name}_{model_name}_{prompt_version}_no_examples_cache.csv')
        method_name = f'LLM {prompt_version.title()} - No Examples ({model_name})'
        results[method_name] = evaluate_llm_method(
            test_df, azure_model, cache_file, prompt_version, domain_description, all_weights_col,
            train_df=None, method_name=method_name, train_avg_weight=train_avg_weight
        )
        print(f"{method_name} Average Score: {results[method_name]:.4f}")

    if train_df is not None:
        # Original stratified examples method
        for prompt_version in prompt_versions:
            cache_file = os.path.join(cache_dir, f'{dataset_name}_{model_name}_{prompt_version}_with_stratified_examples_cache.csv')
            method_name = f'LLM {prompt_version.title()} - With Stratified Examples ({model_name})'
            results[method_name] = evaluate_llm_method(
                test_df, azure_model, cache_file, prompt_version, domain_description, all_weights_col,
                train_df=train_df, method_name=method_name, train_avg_weight=train_avg_weight
            )
            print(f"{method_name} Average Score: {results[method_name]:.4f}")

        # NEW: Similarity-based LLM methods with configurable top_k
        if similarity_retriever is not None:
            # You can experiment with different top_k values here
            top_k_values = [20]  # Default to 3, but can be expanded to [3, 5, 7] for experiments
            
            for top_k in top_k_values:
                for prompt_version in prompt_versions:
                    # Original similarity method (using mean_best_weight)
                    cache_file = os.path.join(cache_dir, f'{dataset_name}_{model_name}_{prompt_version}_similarity_based_k{top_k}_cache.csv')
                    method_name = f'LLM {prompt_version.title()} - Similarity Based k={top_k} ({model_name})'
                    results[method_name] = evaluate_similarity_based_llm_method(
                        test_df, azure_model, cache_file, prompt_version, domain_description, 
                        all_weights_col, similarity_retriever, method_name=method_name, top_k=top_k,
                        train_avg_weight=train_avg_weight
                    )
                    print(f"{method_name} Average Score: {results[method_name]:.4f}")
                    
                    # NEW: Range-based similarity method (using friendly_best_weights)
                    if similarity_retriever.training_friendly_weights is not None:
                        range_cache_file = os.path.join(cache_dir, f'{dataset_name}_{model_name}_{prompt_version}_similarity_ranges_k{top_k}_cache.csv')
                        range_method_name = f'LLM {prompt_version.title()} - Similarity Ranges k={top_k} ({model_name})'
                        results[range_method_name] = evaluate_similarity_based_llm_method_with_ranges(
                            test_df, azure_model, range_cache_file, prompt_version, domain_description, 
                            all_weights_col, similarity_retriever, method_name=range_method_name, top_k=top_k,
                            train_avg_weight=train_avg_weight
                        )
                        print(f"{range_method_name} Average Score: {results[range_method_name]:.4f}")

    return results


def run_evaluation_with_statistics(test_csv_path, train_csv_path=None, model_name='gpt-4o', prompt_versions=None):
    """
    Run evaluation with statistical analysis included.
    This is a wrapper around run_evaluation that adds statistical testing.
    """
    # First run the normal evaluation
    results = run_evaluation(test_csv_path, train_csv_path, model_name, prompt_versions)
    
    if results is None:
        return None, []
    
    # Extract dataset info for statistical analysis
    dataset_name = get_dataset_name_from_path(test_csv_path)
    
    # Load test data for statistical analysis
    try:
        test_df = pd.read_csv(test_csv_path)
        # Store source path for cache directory determination
        test_df.attrs['source_path'] = test_csv_path
    except FileNotFoundError:
        print(f"Error: Could not reload test file for statistical analysis")
        return results, []
    
    # Determine metric column and train average weight
    if train_csv_path and 'mrr_runs' in test_csv_path:
        all_weights_col = 'all_weights_mrr@10'
    else:
        all_weights_col = 'all_weights_ndcg@10'
    
    # Get train average weight if available
    train_avg_weight = None
    if train_csv_path and os.path.exists(train_csv_path):
        try:
            train_df = pd.read_csv(train_csv_path)
            if 'mean_best_weight' in train_df.columns:
                train_avg_weight = train_df['mean_best_weight'].mean()
        except:
            pass
    elif dataset_name == 'trec-covid':
        # Special case: use combined training data average
        _, combined_avg_weight = load_combined_training_data()
        train_avg_weight = combined_avg_weight
    
    # Run statistical analysis
    statistical_results = compare_methods_to_baselines(
        test_df, dataset_name, model_name, all_weights_col, train_avg_weight
    )
    
    # Add dataset info to results
    for result in statistical_results:
        result['dataset'] = dataset_name
    
    return results, statistical_results


if __name__ == '__main__':
    # Define datasets as tuples of (test_path, train_path) where train_path can be None
    datasets = [
        # ACORD dataset (start with this one as requested)
        (
            '/home/smcilroy/projects/labs_pl-intern-irisma/weighted_rrf/dataset/acord/ndcg_runs/test/results_test_all_queries.csv',
            '/home/smcilroy/projects/labs_pl-intern-irisma/weighted_rrf/dataset/acord/ndcg_runs/train/results_train_best_weights_cleaned.csv'
        ),
        # Other datasets
        (
            '/home/smcilroy/projects/labs_pl-intern-irisma/weighted_rrf/dataset/trec-covid/ndcg_runs/test/results_test_all_queries.csv',
            None
        ),
        (
            '/home/smcilroy/projects/labs_pl-intern-irisma/weighted_rrf/dataset/nfcorpus/ndcg_runs/test/results_test_all_queries.csv',
            '/home/smcilroy/projects/labs_pl-intern-irisma/weighted_rrf/dataset/nfcorpus/ndcg_runs/train/results_train_best_weights_cleaned.csv'
        ),
        # MRR datasets (with training data)
        (
            '/home/smcilroy/projects/labs_pl-intern-irisma/weighted_rrf/dataset/nq/mrr_runs/dev/results_dev_all_queries.csv',
            '/home/smcilroy/projects/labs_pl-intern-irisma/weighted_rrf/dataset/nq/mrr_runs/train/results_train_best_weights_cleaned.csv'
        ),
        (
            '/home/smcilroy/projects/labs_pl-intern-irisma/weighted_rrf/dataset/msmarco/mrr_runs/dev/results_dev_all_queries.csv',
            '/home/smcilroy/projects/labs_pl-intern-irisma/weighted_rrf/dataset/msmarco/mrr_runs/train/results_train_best_weights_cleaned.csv'
        )
    ]

    # --- Script Parameters ---
    # Model versions to test
    model_versions = ['gpt-4o']  # Add more models like ['gpt-4o', 'gpt-5']
    # Prompt versions to run
    prompt_versions_to_run = ['no_guidance']  # Start with combined, can add ['original', 'joel', 'combined']
    # --- End of Parameters ---

    all_results = {}
    all_statistical_results = []

    for model_name in model_versions:
        print(f"\n{'=' * 60}")
        print(f"TESTING MODEL: {model_name}")
        print(f"{'=' * 60}")

        for test_path, train_path in datasets:
            dataset_name = get_dataset_name_from_path(test_path)
            dataset_key = f"{dataset_name}_{model_name}"
            
            # Run evaluation with statistical analysis
            result, statistical_results = run_evaluation_with_statistics(
                test_csv_path=test_path,
                train_csv_path=train_path,
                model_name=model_name,
                prompt_versions=prompt_versions_to_run
            )
            
            if result:
                all_results[dataset_key] = result
                all_statistical_results.extend(statistical_results)

    # Print comprehensive summary
    if len(all_results.keys()) > 1:
        print(f"\n{'=' * 80}")
        print("COMPREHENSIVE SUMMARY ACROSS ALL DATASETS AND MODELS")
        print(f"{'=' * 80}")

        # Create a DataFrame for easy viewing
        summary_df = pd.DataFrame(all_results).T
        # Reorder columns to a more logical sequence if possible
        cols = summary_df.columns.tolist()
        ordered_cols = sorted(cols, key=lambda x: ('LLM' not in x, 'Baseline' not in x, x))
        summary_df = summary_df[ordered_cols]
        print(summary_df.to_string())

    elif len(all_results) == 1:
        print(f"\n{'=' * 60}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'=' * 60}")
        for results in all_results.values():
            for method, score in results.items():
                print(f"{method}: {score:.4f}")

    # Generate comprehensive statistical report
    if all_statistical_results:
        generate_statistical_report(all_statistical_results)
    else:
        print("\nNo statistical results to report.")
