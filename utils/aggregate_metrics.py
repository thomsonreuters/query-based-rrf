#!/usr/bin/env python3
"""
Aggregate retrieval-metric CSVs into one row per (dataset, model).

Input CSVs (one per dataset) must have columns:
    Method, Dataset, Split, Sparse, Dense, <metric@5>, <metric@10>

The 7th column (index 6) is used as the metric — either NDCG@10 or MRR@10.
Rows are averaged over the four retriever combos per (Dataset, Method).

Output columns: dataset, model, <metric_name>
If input files carry different metric names, a "metric_type" column is added.
"""

import argparse
import sys

import pandas as pd

DATASET_ORDER = ["acord", "msmarco", "nfcorpus", "nq"]
DATASET_ALIASES = {"acord-entire-corpus": "acord"}
VALID_COMBOS = {"bm25_vs_biencoder", "bm25_vs_qwen3", "rm3_vs_biencoder", "rm3_vs_qwen3"}
REQUIRED_COLUMNS = {"Method", "Dataset", "Split", "Sparse", "Dense"}
DATASET_EXPECTED_METRIC = {
    "acord":    "NDCG@10",
    "msmarco":  "MRR@10",
    "nfcorpus": "NDCG@10",
    "nq":       "MRR@10",
}


def load_and_validate(path: str) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    df = df.dropna(how="all").reset_index(drop=True)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        sys.exit(f"ERROR [{path}]: missing columns: {sorted(missing)}")
    if len(df.columns) < 8:
        sys.exit(f"ERROR [{path}]: expected at least 8 columns, got {len(df.columns)}")

    # Normalize dataset aliases (e.g. "acord-entire-corpus" → "acord").
    df["Dataset"] = df["Dataset"].replace(DATASET_ALIASES)

    # Column index 7 must be NDCG@10 or MRR@10; cross-check against the dataset.
    metric_col = df.columns[7]
    for dataset in df["Dataset"].unique():
        expected = DATASET_EXPECTED_METRIC.get(dataset)
        if expected is not None and metric_col != expected:
            sys.exit(
                f"ERROR [{path}]: column 7 is '{metric_col}' but dataset "
                f"'{dataset}' expects '{expected}' — check column order."
            )

    # Extract point estimate from values like "0.132 [0.098, 0.166]".
    df[metric_col] = df[metric_col].astype(str).str.extract(r"([\d.]+)")[0].astype(float)

    return df, metric_col


def filter_to_valid_combos(df: pd.DataFrame, path: str) -> pd.DataFrame:
    df = df.copy()
    df["combo"] = df["Sparse"] + "_vs_" + df["Dense"]
    mask = df["combo"].isin(VALID_COMBOS)
    n_excluded = (~mask).sum()
    if n_excluded:
        excluded = df.loc[~mask, "combo"].unique().tolist()
        print(
            f"Warning [{path}]: {n_excluded} row(s) excluded — combo not in VALID_COMBOS: {excluded}",
            file=sys.stderr,
        )
    return df[mask].copy()


def aggregate(df: pd.DataFrame, metric_col: str, path: str) -> pd.DataFrame:
    result = (
        df.groupby(["Dataset", "Method"], sort=False)
        .agg(metric=(metric_col, "mean"), n_combos=("combo", "count"))
        .reset_index()
    )

    incomplete = result[result["n_combos"] != len(VALID_COMBOS)]
    if not incomplete.empty:
        print(
            f"Warning [{path}]: {len(incomplete)} group(s) have fewer than {len(VALID_COMBOS)} combos:\n"
            + incomplete[["Dataset", "Method", "n_combos"]].to_string(index=False),
            file=sys.stderr,
        )

    return (
        result.drop(columns=["n_combos"])
        .rename(columns={"Dataset": "dataset", "Method": "model", "metric": metric_col})
    )


def sort_by_dataset_then_model(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    datasets_in_data = df["dataset"].unique().tolist()
    unknown = sorted(set(datasets_in_data) - set(DATASET_ORDER))
    order = DATASET_ORDER + unknown
    if unknown:
        print(f"Warning: unknown dataset(s) appended at end: {unknown}", file=sys.stderr)
    df = df.copy()
    df["dataset"] = pd.Categorical(df["dataset"], categories=order, ordered=True)
    return df.sort_values(["dataset", "model"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average NDCG@10/MRR@10 over retriever combos, grouped by (dataset, model)."
    )
    parser.add_argument("inputs", nargs="+", help="One or more input CSV files")
    parser.add_argument("-o", "--output", help="Path to output CSV file (default: stdout)")
    args = parser.parse_args()

    frames = []
    metric_names = set()

    for path in args.inputs:
        df, metric_col = load_and_validate(path)
        metric_names.add(metric_col)
        df = filter_to_valid_combos(df, path)
        df = aggregate(df, metric_col, path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # If files carry different metric names, disambiguate with a metric_type column.
    if len(metric_names) > 1:
        combined = combined.rename(columns={next(iter(metric_names)): "metric"})
        for col in list(metric_names)[1:]:
            if col in combined.columns:
                combined["metric"] = combined["metric"].combine_first(combined[col])
                combined = combined.drop(columns=[col])
        metric_col_out = "metric"
    else:
        metric_col_out = next(iter(metric_names))

    combined[metric_col_out] = combined[metric_col_out].round(4)
    combined = sort_by_dataset_then_model(combined, metric_col_out)

    if args.output:
        combined.to_csv(args.output, index=False)
        print(f"Saved {len(combined)} rows → {args.output}")
    else:
        print(combined.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
