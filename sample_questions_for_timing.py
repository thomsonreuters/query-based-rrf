#!/usr/bin/env python3
"""
Sample <100 questions from each dataset for timing model inference.
Creates a consolidated dataset with queries from all available datasets.
"""

import os
import pandas as pd
from pathlib import Path
import json
import random
from typing import Dict, List

def sample_questions_from_datasets(
    base_data_dir: str = None,
    num_samples: int = 100,
    output_file: str = "sampled_questions_for_timing.json",
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Sample questions from each dataset in data/input for timing experiments.

    Args:
        base_data_dir: Base directory containing the data (uses config if not provided)
        num_samples: Maximum number of questions to sample per dataset (default: 100)
        output_file: Output JSON file for sampled questions
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping dataset names to lists of sampled questions
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Use the actual data/input path
    if base_data_dir is None:
        base_data_dir = "/Users/a6128162/Repos/query-based-rrf/data/input"

    # Define datasets and their configurations
    datasets = ["acord-entire-corpus", "msmarco", "nfcorpus", "nq"]
    combinations = ["bm25_vs_biencoder", "bm25_vs_qwen3", "rm3_vs_biencoder", "rm3_vs_qwen3"]

    sampled_questions = {}

    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")

        # Determine metric and split based on dataset
        metric = "ndcg" if dataset in ["acord-entire-corpus", "nfcorpus"] else "mrr"
        test_split = "test" if dataset in ["acord-entire-corpus", "nfcorpus"] else "dev"

        all_questions = set()

        # Try to collect questions from various sources
        for combo in combinations:
            # Check train files
            train_path = Path(base_data_dir) / dataset / f"{metric}_runs" / "train" / "top200" / f"results_train_{combo}_best_weights_final_mean_with_text.csv"
            if train_path.exists():
                try:
                    df = pd.read_csv(train_path)
                    # Use query_text column which is what the actual CSVs have
                    if 'query_text' in df.columns:
                        all_questions.update(df['query_text'].dropna().unique())
                    elif 'query' in df.columns:
                        all_questions.update(df['query'].dropna().unique())
                    elif 'question' in df.columns:
                        all_questions.update(df['question'].dropna().unique())
                    elif 'text' in df.columns:
                        all_questions.update(df['text'].dropna().unique())
                    print(f"  Found {len(df)} entries in {combo} train file")
                except Exception as e:
                    print(f"  Error reading {train_path}: {e}")

            # Check test/dev files
            test_path = Path(base_data_dir) / dataset / f"{metric}_runs" / test_split / "top200" / f"results_{test_split}_{combo}_best_weights_final_mean_with_text.csv"
            if test_path.exists():
                try:
                    df = pd.read_csv(test_path)
                    # Use query_text column which is what the actual CSVs have
                    if 'query_text' in df.columns:
                        all_questions.update(df['query_text'].dropna().unique())
                    elif 'query' in df.columns:
                        all_questions.update(df['query'].dropna().unique())
                    elif 'question' in df.columns:
                        all_questions.update(df['question'].dropna().unique())
                    elif 'text' in df.columns:
                        all_questions.update(df['text'].dropna().unique())
                    print(f"  Found {len(df)} entries in {combo} {test_split} file")
                except Exception as e:
                    print(f"  Error reading {test_path}: {e}")

        # Convert to list and sample
        question_list = list(all_questions)

        if question_list:
            # Sample up to num_samples questions
            sample_size = min(len(question_list), num_samples)
            sampled = random.sample(question_list, sample_size)
            sampled_questions[dataset] = sampled
            print(f"  Sampled {len(sampled)} unique questions from {dataset}")
        else:
            print(f"  WARNING: No questions found for {dataset}")
            sampled_questions[dataset] = []

    # Save sampled questions to JSON file
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_questions, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*50}")
    print("Sampling Summary:")
    print(f"{'='*50}")
    total_questions = 0
    for dataset, questions in sampled_questions.items():
        count = len(questions)
        total_questions += count
        print(f"{dataset}: {count} questions")
    print(f"{'='*50}")
    print(f"Total questions sampled: {total_questions}")
    print(f"Output saved to: {output_path}")

    return sampled_questions


def create_timing_dataset(
    sampled_questions: Dict[str, List[str]] = None,
    input_file: str = "sampled_questions_for_timing.json",
    output_file: str = "timing_dataset.csv"
):
    """
    Create a CSV dataset for timing experiments from sampled questions.

    Args:
        sampled_questions: Dictionary of sampled questions (loads from file if None)
        input_file: Input JSON file with sampled questions
        output_file: Output CSV file for timing experiments
    """
    if sampled_questions is None:
        with open(input_file, 'r', encoding='utf-8') as f:
            sampled_questions = json.load(f)

    # Create a dataframe with all questions
    rows = []
    for dataset, questions in sampled_questions.items():
        for question in questions:
            rows.append({
                'dataset': dataset,
                'question': question,
                'question_length': len(question),
                'word_count': len(question.split())
            })

    df = pd.DataFrame(rows)

    # Save to CSV
    df.to_csv(output_file, index=False)

    print(f"\nTiming dataset created: {output_file}")
    print(f"Shape: {df.shape}")

    if not df.empty:
        print(f"\nDataset distribution:")
        print(df['dataset'].value_counts())
        print(f"\nQuestion length statistics:")
        print(df[['question_length', 'word_count']].describe())
    else:
        print("WARNING: No questions were sampled - dataset is empty")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sample questions from datasets for timing experiments")
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Base data directory (uses config.BASE_DATA_DIR if not specified)')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Maximum number of questions to sample per dataset (default: 100)')
    parser.add_argument('--output-json', type=str, default='sampled_questions_for_timing.json',
                       help='Output JSON file for sampled questions')
    parser.add_argument('--output-csv', type=str, default='timing_dataset.csv',
                       help='Output CSV file for timing experiments')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--create-csv', action='store_true',
                       help='Also create CSV dataset for timing')

    args = parser.parse_args()

    # Sample questions from datasets
    sampled_questions = sample_questions_from_datasets(
        base_data_dir=args.data_dir,
        num_samples=args.num_samples,
        output_file=args.output_json,
        seed=args.seed
    )

    # Optionally create CSV dataset
    if args.create_csv and sampled_questions:
        create_timing_dataset(
            sampled_questions=sampled_questions,
            output_file=args.output_csv
        )