#!/usr/bin/env python3
"""
Build timing-samples dataset under data/input/timing-samples/ by sampling up to
100 queries per source dataset (test-first, dev to fill the remainder), keeping
only queries that have search results from all four retrievers.

Output mirrors the structure of existing datasets:
  timing-samples/
    search_results/{bm25,biencoder,rm3,qwen3}/top200/results_test.trec  (×4)
    ndcg_runs/test/top200/results_test_{combo}_best_weights_final_mean_with_text.csv  (×4)
"""

import random
import pandas as pd
from pathlib import Path


RETRIEVERS = ["bm25", "biencoder", "rm3", "qwen3"]

COMBINATIONS = [
    "bm25_vs_biencoder",
    "bm25_vs_qwen3",
    "rm3_vs_biencoder",
    "rm3_vs_qwen3",
]

# (metric, splits_in_priority_order)
# msmarco and nq have no test split in combination CSVs — only dev and train exist.
DATASET_CONFIG = {
    "acord-entire-corpus": ("ndcg", ["test", "dev"]),
    "nfcorpus":            ("ndcg", ["test", "dev"]),
    "msmarco":             ("mrr",  ["dev"]),
    "nq":                  ("mrr",  ["dev"]),
}


def _valid_query_ids_in_split(base: Path, dataset: str, metric: str, split: str) -> set[str]:
    """
    Return query_ids present in ALL 4 combination files AND all 4 TREC files for this split.
    Returns an empty set if any required file is missing.
    """
    per_combo: list[set[str]] = []
    for combo in COMBINATIONS:
        path = (
            base / dataset / f"{metric}_runs" / split / "top200"
            / f"results_{split}_{combo}_best_weights_final_mean_with_text.csv"
        )
        if not path.exists():
            return set()
        df = pd.read_csv(path, usecols=["query_id"])
        per_combo.append(set(df["query_id"].dropna().astype(str).unique()))
    valid = set.intersection(*per_combo)

    for retriever in RETRIEVERS:
        trec_path = (
            base / dataset / "search_results" / retriever / "top200"
            / f"results_{split}.trec"
        )
        if not trec_path.exists():
            return set()
        trec_qids: set[str] = set()
        with open(trec_path) as f:
            for line in f:
                fields = line.split(None, 1)
                if fields:
                    trec_qids.add(fields[0])
        valid &= trec_qids

    return valid


def _sample_queries(base: Path, num_samples: int) -> dict[str, list[tuple[str, str]]]:
    """
    Sample up to num_samples (query_id, split) pairs per dataset.
    Returns {dataset: [(query_id, split), ...]}.
    """
    result: dict[str, list[tuple[str, str]]] = {}

    for dataset, (metric, priority_splits) in DATASET_CONFIG.items():
        print(f"\n=== {dataset} ===")
        sampled: list[tuple[str, str]] = []
        sampled_ids: set[str] = set()

        for split in priority_splits:
            needed = num_samples - len(sampled)
            if needed <= 0:
                break
            valid = _valid_query_ids_in_split(base, dataset, metric, split) - sampled_ids
            if not valid:
                print(f"  {split}: 0 valid queries (missing files or no intersection)")
                continue
            chosen = random.sample(sorted(valid), min(len(valid), needed))
            sampled.extend((qid, split) for qid in chosen)
            sampled_ids.update(chosen)
            print(f"  {split}: sampled {len(chosen)} / {len(valid)} valid queries")

        result[dataset] = sampled
        print(f"  total: {len(sampled)}")

    return result


def _write_combination_csvs(
    base: Path,
    out_dir: Path,
    sampled: dict[str, list[tuple[str, str]]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    combined: dict[str, list[pd.DataFrame]] = {c: [] for c in COMBINATIONS}

    for dataset, query_split_pairs in sampled.items():
        if not query_split_pairs:
            continue
        metric = DATASET_CONFIG[dataset][0]

        split_to_qids: dict[str, set[str]] = {}
        for qid, split in query_split_pairs:
            split_to_qids.setdefault(split, set()).add(qid)

        for combo in COMBINATIONS:
            for split, qids in split_to_qids.items():
                path = (
                    base / dataset / f"{metric}_runs" / split / "top200"
                    / f"results_{split}_{combo}_best_weights_final_mean_with_text.csv"
                )
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                df["query_id"] = df["query_id"].astype(str)
                filtered = df[df["query_id"].isin(qids)].copy()
                filtered["source_dataset"] = dataset
                combined[combo].append(filtered)

    for combo in COMBINATIONS:
        if not combined[combo]:
            print(f"  SKIP (no data): {combo}")
            continue
        merged = pd.concat(combined[combo], ignore_index=True)
        out_path = out_dir / f"results_test_{combo}_best_weights_final_mean_with_text.csv"
        merged.to_csv(out_path, index=False)
        print(f"  {combo}: {len(merged)} rows -> {out_path.name}")


def _write_trec_files(
    base: Path,
    out_base: Path,
    sampled: dict[str, list[tuple[str, str]]],
) -> None:
    for retriever in RETRIEVERS:
        out_dir = out_base / "search_results" / retriever / "top200"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "results_test.trec"

        with open(out_path, "w") as out_f:
            for dataset, query_split_pairs in sampled.items():
                if not query_split_pairs:
                    continue

                split_to_qids: dict[str, set[str]] = {}
                for qid, split in query_split_pairs:
                    split_to_qids.setdefault(split, set()).add(qid)

                for split, qids in split_to_qids.items():
                    trec_path = (
                        base / dataset / "search_results" / retriever / "top200"
                        / f"results_{split}.trec"
                    )
                    if not trec_path.exists():
                        print(f"  WARNING: missing {trec_path.relative_to(base)}")
                        continue
                    with open(trec_path) as in_f:
                        for line in in_f:
                            if line.strip() and line.split()[0] in qids:
                                out_f.write(line)

        print(f"  {retriever} -> {out_path.relative_to(out_base)}")


def create_sample_data(
    base_data_dir: str = "/Users/a6128162/Repos/query-based-rrf/data/input",
    num_samples: int = 100,
    seed: int = 42,
) -> None:
    random.seed(seed)
    base = Path(base_data_dir)
    out_base = base / "timing-samples"

    print("=== Sampling queries ===")
    sampled = _sample_queries(base, num_samples=num_samples)
    total = sum(len(v) for v in sampled.values())
    print(f"\nTotal sampled: {total} queries")

    print("\n=== Writing combination CSVs ===")
    _write_combination_csvs(base, out_base / "ndcg_runs" / "test" / "top200", sampled)

    print("\n=== Writing TREC search result files ===")
    _write_trec_files(base, out_base, sampled)

    print(f"\nDone. Output: {out_base}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build timing-samples dataset")
    parser.add_argument("--data-dir", default="/Users/a6128162/Repos/query-based-rrf/data/input")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_sample_data(
        base_data_dir=args.data_dir,
        num_samples=args.num_samples,
        seed=args.seed,
    )
