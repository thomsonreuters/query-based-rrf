#!/usr/bin/env python3
"""
Create corpus.jsonl and queries.jsonl for the timing-samples dataset
by extracting the exact queries and documents that were sampled.

This script ensures correspondence by:
1. Reading query IDs and their source datasets from the CSV files
2. Extracting only those specific queries from source datasets
3. Extracting only documents that appear in the sampled TREC results
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Set, Tuple, Any


def get_sampled_queries_from_csv(timing_samples_dir: Path) -> Dict[str, Set[str]]:
    """
    Extract query IDs and their source datasets from the CSV files.
    Returns: {dataset_name: set_of_query_ids}
    """
    # Read any of the CSV files (they all have the same queries)
    csv_file = (timing_samples_dir / "ndcg_runs" / "test" / "top200" /
                "results_test_bm25_vs_biencoder_best_weights_final_mean_with_text.csv")

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)

    # Group by source dataset
    dataset_query_map = {}
    for dataset in df['source_dataset'].unique():
        query_ids = set(df[df['source_dataset'] == dataset]['query_id'].astype(str).unique())
        dataset_query_map[dataset] = query_ids
        print(f"{dataset}: {len(query_ids)} queries")

    return dataset_query_map


def get_sampled_documents_from_trec(timing_samples_dir: Path) -> Set[str]:
    """
    Extract all unique document IDs from all TREC files.
    """
    all_doc_ids = set()
    retrievers = ["bm25", "biencoder", "rm3", "qwen3"]

    for retriever in retrievers:
        trec_file = timing_samples_dir / "search_results" / retriever / "top200" / "results_test.trec"
        if not trec_file.exists():
            print(f"Warning: {trec_file} not found")
            continue

        with open(trec_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    all_doc_ids.add(parts[2])

    print(f"Total unique documents across all retrievers: {len(all_doc_ids)}")
    return all_doc_ids


def verify_queries_exist(
    dataset_query_map: Dict[str, Set[str]],
    base_data_dir: Path
) -> Dict[str, Dict[str, Any]]:
    """
    Verify that all sampled queries exist in source datasets and extract their data.
    Returns: {query_id: query_data_dict}
    """
    all_queries = {}

    for dataset, query_ids in dataset_query_map.items():
        queries_file = base_data_dir / dataset / "queries.jsonl"
        if not queries_file.exists():
            raise FileNotFoundError(f"Source queries file not found: {queries_file}")

        found_ids = set()
        with open(queries_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data["_id"] in query_ids:
                    all_queries[data["_id"]] = data
                    found_ids.add(data["_id"])

        missing = query_ids - found_ids
        if missing:
            print(f"WARNING: {len(missing)} queries from {dataset} not found in source: {list(missing)[:5]}...")
        else:
            print(f"✓ All {len(query_ids)} queries from {dataset} found in source")

    return all_queries


def map_docs_to_source_datasets(
    doc_ids: Set[str],
    base_data_dir: Path
) -> Dict[str, Any]:
    """
    Find documents in source datasets and extract their data.
    Returns: {doc_id: doc_data_dict}
    """
    all_docs = {}
    datasets = ["acord-entire-corpus", "nfcorpus", "msmarco", "nq"]

    remaining_docs = doc_ids.copy()

    for dataset in datasets:
        corpus_file = base_data_dir / dataset / "corpus.jsonl"
        if not corpus_file.exists():
            print(f"Warning: {corpus_file} not found")
            continue

        found_in_dataset = 0
        with open(corpus_file, 'r') as f:
            for line in f:
                if not remaining_docs:  # All docs found
                    break
                data = json.loads(line)
                if data["_id"] in remaining_docs:
                    all_docs[data["_id"]] = data
                    remaining_docs.remove(data["_id"])
                    found_in_dataset += 1

        if found_in_dataset > 0:
            print(f"Found {found_in_dataset} documents in {dataset}")

    if remaining_docs:
        print(f"WARNING: {len(remaining_docs)} documents not found in any source dataset")
        print(f"  Sample missing IDs: {list(remaining_docs)[:5]}")

    return all_docs


def create_queries_jsonl(queries_data: Dict[str, Any], output_file: Path) -> None:
    """Create queries.jsonl from extracted query data."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for query_id in sorted(queries_data.keys()):
            f.write(json.dumps(queries_data[query_id]) + '\n')

    print(f"Created {output_file} with {len(queries_data)} queries")


def create_corpus_jsonl(docs_data: Dict[str, Any], output_file: Path) -> None:
    """Create corpus.jsonl from extracted document data."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        # Sort by doc_id for consistent output
        for doc_id in sorted(docs_data.keys()):
            f.write(json.dumps(docs_data[doc_id]) + '\n')

    print(f"Created {output_file} with {len(docs_data)} documents")


def verify_correspondence(
    timing_samples_dir: Path,
    queries_data: Dict[str, Any],
    docs_data: Dict[str, Any]
) -> None:
    """Verify that created files match what's in TREC/CSV files."""
    print("\n=== Verification ===")

    # Verify queries match CSV
    csv_file = (timing_samples_dir / "ndcg_runs" / "test" / "top200" /
                "results_test_bm25_vs_biencoder_best_weights_final_mean_with_text.csv")
    df = pd.read_csv(csv_file)
    csv_query_ids = set(df['query_id'].astype(str).unique())
    jsonl_query_ids = set(queries_data.keys())

    if csv_query_ids == jsonl_query_ids:
        print(f"✓ Query IDs match perfectly: {len(csv_query_ids)} queries")
    else:
        missing_in_jsonl = csv_query_ids - jsonl_query_ids
        extra_in_jsonl = jsonl_query_ids - csv_query_ids
        if missing_in_jsonl:
            print(f"✗ Missing in queries.jsonl: {missing_in_jsonl}")
        if extra_in_jsonl:
            print(f"✗ Extra in queries.jsonl: {extra_in_jsonl}")

    # Verify all TREC query IDs are covered
    trec_file = timing_samples_dir / "search_results" / "bm25" / "top200" / "results_test.trec"
    trec_query_ids = set()
    with open(trec_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                trec_query_ids.add(parts[0])

    if trec_query_ids == jsonl_query_ids:
        print(f"✓ TREC query IDs match perfectly")
    else:
        print(f"✗ TREC vs queries.jsonl mismatch")
        print(f"  In TREC but not in jsonl: {trec_query_ids - jsonl_query_ids}")
        print(f"  In jsonl but not in TREC: {jsonl_query_ids - trec_query_ids}")


def main():
    import os

    # Check for override from command line
    if "_OVERRIDE_DATA_DIR" in os.environ:
        base_data_dir = Path(os.environ["_OVERRIDE_DATA_DIR"])
    else:
        base_repo = os.getenv("QBR_REPO_PATH", "/Users/a6128162/Repos/query-based-rrf")
        base_data_dir = Path(os.path.join(base_repo, "data/input"))

    timing_samples_dir = base_data_dir / "timing-samples"

    print("=== Step 1: Extract sampled queries from CSV ===")
    dataset_query_map = get_sampled_queries_from_csv(timing_samples_dir)

    print("\n=== Step 2: Extract sampled documents from TREC ===")
    sampled_doc_ids = get_sampled_documents_from_trec(timing_samples_dir)

    print("\n=== Step 3: Verify and extract query data from sources ===")
    queries_data = verify_queries_exist(dataset_query_map, base_data_dir)

    print("\n=== Step 4: Extract document data from sources ===")
    docs_data = map_docs_to_source_datasets(sampled_doc_ids, base_data_dir)

    print("\n=== Step 5: Create queries.jsonl ===")
    queries_output = timing_samples_dir / "queries.jsonl"
    create_queries_jsonl(queries_data, queries_output)

    print("\n=== Step 6: Create corpus.jsonl ===")
    corpus_output = timing_samples_dir / "corpus.jsonl"
    create_corpus_jsonl(docs_data, corpus_output)

    # Verify correspondence
    verify_correspondence(timing_samples_dir, queries_data, docs_data)

    print("\n✓ Done! Created:")
    print(f"  - {queries_output}")
    print(f"  - {corpus_output}")


if __name__ == "__main__":
    import argparse
    import os
    import sys

    # Check if being called with arguments
    if len(sys.argv) > 1:
        base_repo = os.getenv("QBR_REPO_PATH", "/Users/a6128162/Repos/query-based-rrf")
        default_data_dir = os.path.join(base_repo, "data/input")

        parser = argparse.ArgumentParser(description="Create corpus.jsonl and queries.jsonl for timing-samples")
        parser.add_argument("--data-dir", default=default_data_dir,
                            help=f"Base data directory (default: $QBR_REPO_PATH/data/input or {default_data_dir})")
        args = parser.parse_args()

        # Override the path in main()
        os.environ["_OVERRIDE_DATA_DIR"] = args.data_dir

    main()