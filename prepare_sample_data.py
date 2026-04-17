#!/usr/bin/env python3
"""
Prepare sample data for quick experiment testing.
Creates a subset of 10 queries from each dataset for timing experiments.
"""

import os
import pandas as pd
from pathlib import Path

def create_sample_data(base_data_dir="/home/sagemaker-user/query-aware-rrf/query-based-rrf/data",
                       num_samples=10):
    """
    Create sample CSV files with limited queries for testing.
    """
    datasets = ["acord-entire-corpus", "msmarco", "nfcorpus", "nq"]
    combinations = ["bm25_vs_biencoder", "bm25_vs_qwen3", "rm3_vs_biencoder", "rm3_vs_qwen3"]

    for dataset in datasets:
        metric = "ndcg" if dataset in ["acord-entire-corpus", "nfcorpus"] else "mrr"
        test_split = "test" if dataset in ["acord-entire-corpus", "nfcorpus"] else "dev"

        for combo in combinations:
            # Train file
            train_path = Path(base_data_dir) / dataset / f"{metric}_runs" / "train" / "top200" / f"results_train_{combo}_best_weights_final_mean_with_text.csv"
            if train_path.exists():
                df = pd.read_csv(train_path)
                sample_df = df.head(num_samples)
                sample_path = train_path.parent / f"sample_{num_samples}_{train_path.name}"
                sample_df.to_csv(sample_path, index=False)
                print(f"Created: {sample_path}")

            # Test file
            test_path = Path(base_data_dir) / dataset / f"{metric}_runs" / test_split / "top200" / f"results_{test_split}_{combo}_best_weights_final_mean_with_text.csv"
            if test_path.exists():
                df = pd.read_csv(test_path)
                sample_df = df.head(num_samples)
                sample_path = test_path.parent / f"sample_{num_samples}_{test_path.name}"
                sample_df.to_csv(sample_path, index=False)
                print(f"Created: {sample_path}")

if __name__ == "__main__":
    create_sample_data()