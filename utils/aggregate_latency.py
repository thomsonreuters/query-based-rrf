#!/usr/bin/env python3
"""
Aggregate per-combo latency rows into one row per (dataset, model).

Input CSV columns expected:
    model, dataset, infra, num of queries, num of param,
    sparse, dense, total time (ms), avg latency (ms)

The combo is derived from `sparse` and `dense` as "{sparse}_vs_{dense}".
Only rows whose combo is in VALID_COMBOS are kept; `avg latency (ms)` and
`total time (ms)` are averaged across the (up to 4) combos per group.
"""

import argparse
import sys

import pandas as pd

DATASET_ORDER = ["acord", "msmarco", "nfcorpus", "nq"]
VALID_COMBOS = {"bm25_vs_biencoder", "bm25_vs_qwen3", "rm3_vs_biencoder", "rm3_vs_qwen3"}
# Models with no per-query inference cost — added to every dataset with latency 0.
# These canonical names must match what model_mapping.json maps to.
ZERO_COST_MODELS = ["rrf", "mow"]
REQUIRED_COLUMNS = {
    "model", "dataset", "infra", "num of queries", "num of param",
    "sparse", "dense", "total time (ms)", "avg latency (ms)",
}


def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        sys.exit(f"ERROR: missing columns in input file: {sorted(missing)}")
    # Strip whitespace from all string columns so e.g. "msmarco " == "msmarco".
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    # Drop fully-empty rows (blank separator rows in spreadsheet exports).
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def build_combo_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["combo"] = df["sparse"] + "_vs_" + df["dense"]  # whitespace already stripped by load_and_validate
    return df


def filter_to_valid_combos(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["combo"].isin(VALID_COMBOS)
    n_excluded = (~mask).sum()
    if n_excluded:
        excluded = df.loc[~mask, "combo"].unique().tolist()
        print(
            f"Warning: {n_excluded} row(s) excluded — combo not in VALID_COMBOS: {excluded}",
            file=sys.stderr,
        )
    return df[mask].copy()


def aggregate_over_combos(df: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["dataset", "model"]

    # infra and num of param are constant per (dataset, model) — take first.
    # num of queries can differ across combos (different pipeline coverage) — average.
    agg_spec = {col: "first" for col in ["infra", "num of param"]}
    agg_spec["num of queries"] = "mean"
    agg_spec["avg latency (ms)"] = "mean"
    agg_spec["total time (ms)"] = "mean"
    agg_spec["combo"] = "count"  # repurposed as a validation counter

    result = (
        df.groupby(group_keys, sort=False)
        .agg(agg_spec)
        .rename(columns={"combo": "_n_combos"})
        .reset_index()
    )

    incomplete = result[result["_n_combos"] != len(VALID_COMBOS)]
    if not incomplete.empty:
        print(
            f"Warning: {len(incomplete)} group(s) have fewer than {len(VALID_COMBOS)} combos:\n"
            + incomplete[["dataset", "model", "_n_combos"]].to_string(index=False),
            file=sys.stderr,
        )

    result = result.drop(columns=["_n_combos"])
    result["avg latency (ms)"] = result["avg latency (ms)"].round(4)
    result = result[["dataset", "model", "avg latency (ms)"]]

    # Append zero-cost models for every dataset present in the data.
    datasets = result["dataset"].unique().tolist()
    zero_rows = pd.DataFrame([
        {"dataset": ds, "model": model, "avg latency (ms)": 0.0}
        for ds in datasets
        for model in ZERO_COST_MODELS
    ])
    return pd.concat([result, zero_rows], ignore_index=True)


def sort_by_dataset_then_model(df: pd.DataFrame) -> pd.DataFrame:
    datasets_in_data = df["dataset"].unique().tolist()
    unknown = sorted(set(datasets_in_data) - set(DATASET_ORDER))
    order = DATASET_ORDER + unknown  # put unknown datasets at the end
    if unknown:
        print(f"Warning: unknown dataset(s) appended at end: {unknown}", file=sys.stderr)

    df = df.copy()
    df["dataset"] = pd.Categorical(df["dataset"], categories=order, ordered=True)
    return df.sort_values(["dataset", "model"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average avg latency (ms) over retriever combos, grouped by (dataset, model)."
    )
    parser.add_argument("input", help="Path to input CSV file")
    parser.add_argument(
        "-o", "--output",
        help="Path to output CSV file (default: print to stdout)",
    )
    args = parser.parse_args()

    df = load_and_validate(args.input)
    df = build_combo_column(df)
    df = filter_to_valid_combos(df)
    df = aggregate_over_combos(df)
    df = sort_by_dataset_then_model(df)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} rows → {args.output}")
    else:
        print(df.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
