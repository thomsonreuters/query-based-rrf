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


def run_evaluation(test_csv_path, train_csv_path=None, model_name='gpt-4o', prompt_versions=None):
    """Run all evaluation methods on a dataset."""
    if prompt_versions is None:
        prompt_versions = ['original', 'joel', 'combined']
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
    
    # Special case handling for trec-covid dataset
    if dataset_name == 'trec-covid':
        print("SPECIAL CASE: trec-covid detected - using combined NQ + MSMARCO training data")
        combined_train_df, combined_avg_weight = load_combined_training_data()
        
        if combined_train_df is not None and combined_avg_weight is not None:
            train_df = combined_train_df
            train_avg_weight = combined_avg_weight
            print(f"Using combined training data for trec-covid. Combined average weight: {train_avg_weight:.4f}")
        else:
            print("Warning: Could not load combined training data for trec-covid.")
    
    elif train_csv_path and os.path.exists(train_csv_path):
        try:
            train_df = pd.read_csv(train_csv_path)
            if 'mean_best_weight' in train_df.columns:
                train_avg_weight = train_df['mean_best_weight'].mean()
                print(f"Training data loaded from {train_csv_path}. Average best weight: {train_avg_weight:.4f}")
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

    for prompt_version in prompt_versions:
        cache_file = os.path.join(cache_dir, f'{dataset_name}_{model_name}_{prompt_version}_no_examples_cache.csv')
        method_name = f'LLM {prompt_version.title()} - No Examples ({model_name})'
        results[method_name] = evaluate_llm_method(
            test_df, azure_model, cache_file, prompt_version, domain_description, all_weights_col,
            train_df=None, method_name=method_name, train_avg_weight=train_avg_weight
        )
        print(f"{method_name} Average Score: {results[method_name]:.4f}")

    if train_df is not None:
        for prompt_version in prompt_versions:
            cache_file = os.path.join(cache_dir, f'{dataset_name}_{model_name}_{prompt_version}_with_stratified_examples_cache.csv')
            method_name = f'LLM {prompt_version.title()} - With Stratified Examples ({model_name})'
            results[method_name] = evaluate_llm_method(
                test_df, azure_model, cache_file, prompt_version, domain_description, all_weights_col,
                train_df=train_df, method_name=method_name, train_avg_weight=train_avg_weight
            )
            print(f"{method_name} Average Score: {results[method_name]:.4f}")

    return results


if __name__ == '__main__':
    # Define datasets as tuples of (test_path, train_path) where train_path can be None
    datasets = [
        # NDCG datasets (no training data)
        (
            '/home/sagemaker-user/labs_pl-intern-irisma/weighted_rrf/dataset/acord/ndcg_runs/test/results_test_all_queries.csv',
            '/home/sagemaker-user/labs_pl-intern-irisma/weighted_rrf/dataset/acord/ndcg_runs/train/results_train_best_weights_cleaned.csv'
        ),
        (
            '/home/sagemaker-user/labs_pl-intern-irisma/weighted_rrf/dataset/trec-covid/ndcg_runs/test/results_test_all_queries.csv',
            None
        ),
        (
            '/home/sagemaker-user/labs_pl-intern-irisma/weighted_rrf/dataset/nfcorpus/ndcg_runs/test/results_test_all_queries.csv',
            '/home/sagemaker-user/labs_pl-intern-irisma/weighted_rrf/dataset/nfcorpus/ndcg_runs/train/results_train_best_weights_cleaned.csv'
        ),
        # MRR datasets (with training data)
        (
            '/home/sagemaker-user/labs_pl-intern-irisma/weighted_rrf/dataset/nq/mrr_runs/dev/results_dev_all_queries.csv',
            '/home/sagemaker-user/labs_pl-intern-irisma/weighted_rrf/dataset/nq/mrr_runs/train/results_train_best_weights_cleaned.csv'
        ),
        (
            '/home/sagemaker-user/labs_pl-intern-irisma/weighted_rrf/dataset/msmarco/mrr_runs/dev/results_dev_all_queries.csv',
            '/home/sagemaker-user/labs_pl-intern-irisma/weighted_rrf/dataset/msmarco/mrr_runs/train/results_train_best_weights_cleaned.csv'
        )
    ]

    # --- Script Parameters ---
    # Model versions to test
    model_versions = ['gpt-5']  # Add more models like ['gpt-4o', gpt-5]
    # Prompt versions to run
    prompt_versions_to_run = ['original']  # Subset this list as needed, e.g., ['original', 'joel', 'combined']
    # --- End of Parameters ---

    all_results = {}

    for model_name in model_versions:
        print(f"\n{'=' * 60}")
        print(f"TESTING MODEL: {model_name}")
        print(f"{'=' * 60}")

        for test_path, train_path in datasets:
            dataset_name = get_dataset_name_from_path(test_path)
            dataset_key = f"{dataset_name}_{model_name}"
            result = run_evaluation(
                test_csv_path=test_path,
                train_csv_path=train_path,
                model_name=model_name,
                prompt_versions=prompt_versions_to_run
            )
            if result:
                all_results[dataset_key] = result

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
