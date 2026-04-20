#!/usr/bin/env python3
"""
Build a single 'timing-samples' dataset under data/input/ by sampling up to
100 queries per source dataset (400 total) and merging into one set of
combination CSVs that mirrors the other datasets' structure.

Sampling strategy per source dataset (greedy):
  1. Take up to 100 from test/.
  2. If still under 100, fill the remainder from dev/ (no duplicates).
  3. If combined total is still < 100, that's fine — keep what's available.

Output: {base_data_dir}/timing-samples/ndcg_runs/test/top200/
          results_test_{combo}_best_weights_final_mean_with_text.csv  (×4)
"""

import os
import random
import pandas as pd
from pathlib import Path


SOURCE_DATASETS = [
    ("msmarco",             "mrr"),
    ("nq",                  "mrr"),
    ("acord-entire-corpus", "ndcg"),
    ("nfcorpus",            "ndcg"),
]

COMBINATIONS = [
    "bm25_vs_biencoder",
    "bm25_vs_qwen3",
    "rm3_vs_biencoder",
    "rm3_vs_qwen3",
]

QUERY_COLUMN_CANDIDATES = ["query_text", "query", "question", "text"]


def _query_column(df: pd.DataFrame) -> str | None:
    for col in QUERY_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _unique_queries_in_split(base: Path, dataset: str, metric: str, split: str) -> set[str]:
    queries: set[str] = set()
    for combo in COMBINATIONS:
        path = base / dataset / f"{metric}_runs" / split / "top200" / f"results_{split}_{combo}_best_weights_final_mean_with_text.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        col = _query_column(df)
        if col:
            queries.update(df[col].dropna().unique())
    return queries


def _sample_greedy(base: Path, dataset: str, metric: str, num_samples: int) -> dict[str, set[str]]:
    """
    Greedily sample up to num_samples queries, test first then dev for the remainder.
    Returns {split: set_of_sampled_queries} — may contain one or both splits.
    """
    result: dict[str, set[str]] = {}

    test_q = _unique_queries_in_split(base, dataset, metric, "test")
    test_sampled = set(random.sample(sorted(test_q), min(len(test_q), num_samples)))
    if test_sampled:
        result["test"] = test_sampled

    remaining = num_samples - len(test_sampled)
    if remaining > 0:
        dev_q = _unique_queries_in_split(base, dataset, metric, "dev") - test_sampled
        dev_sampled = set(random.sample(sorted(dev_q), min(len(dev_q), remaining)))
        if dev_sampled:
            result["dev"] = dev_sampled

    return result


def create_sample_data(
    base_data_dir: str = "/Users/a6128162/Repos/query-based-rrf/data/input",
    num_samples: int = 100,
    seed: int = 42,
):
    random.seed(seed)
    base = Path(base_data_dir)
    out_dir = base / "timing-samples" / "ndcg_runs" / "test" / "top200"
    out_dir.mkdir(parents=True, exist_ok=True)

    combined: dict[str, list[pd.DataFrame]] = {c: [] for c in COMBINATIONS}
    total_queries = 0

    for dataset, metric in SOURCE_DATASETS:
        print(f"\n=== {dataset} ===")
        split_queries = _sample_greedy(base, dataset, metric, num_samples)

        if not split_queries:
            print(f"  WARNING: no queries found, skipping")
            continue

        n_total = sum(len(q) for q in split_queries.values())
        total_queries += n_total
        for split, qs in split_queries.items():
            print(f"  {split}: {len(qs)} queries")

        for combo in COMBINATIONS:
            frames = []
            for split, qs in split_queries.items():
                path = base / dataset / f"{metric}_runs" / split / "top200" / f"results_{split}_{combo}_best_weights_final_mean_with_text.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                col = _query_column(df)
                if col is None:
                    continue
                filtered = df[df[col].isin(qs)].copy()
                filtered["source_dataset"] = dataset
                frames.append(filtered)

            if frames:
                combined[combo].append(pd.concat(frames, ignore_index=True))

    print(f"\n--- Writing timing-samples ({total_queries} total queries) ---")
    for combo in COMBINATIONS:
        if not combined[combo]:
            print(f"  SKIP (no data): {combo}")
            continue
        merged = pd.concat(combined[combo], ignore_index=True)
        out_path = out_dir / f"results_test_{combo}_best_weights_final_mean_with_text.csv"
        merged.to_csv(out_path, index=False)
        print(f"  wrote {len(merged)} rows -> {out_path}")

    print(f"\nDone. Output: {out_dir.parent.parent}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build timing-samples dataset")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_sample_data(
        base_data_dir=args.data_dir or "/Users/a6128162/Repos/query-based-rrf/data/input",
        num_samples=args.num_samples,
        seed=args.seed,
    )
