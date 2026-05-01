#!/usr/bin/env python3
"""
Join aggregated retrieval metrics with timing data into one table.

Steps:
  1. Aggregate metrics CSV(s) → (dataset, model, <metric>) via aggregate_metrics logic
  2. Aggregate timing CSV     → (dataset, model, avg latency (ms)) via aggregate_latency logic
  3. Normalize model names using an optional JSON mapping file
  4. Inner-join on (dataset, model) and write output

Model name mapping (--mapping):
  A JSON object where every key is an alias (as it appears in either input)
  and every value is the canonical name to use in the output.
  Example:
    {
      "DAT - qwen3":  "DAT-qwen3",
      "moderbert regression": "modern-bert-regression"
    }
  Models with no entry are kept as-is.

Use --print-unmatched to list models that appear in only one source after mapping
(useful for building the mapping file iteratively).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Make sibling imports work whether run as a script or module.
sys.path.insert(0, str(Path(__file__).parent))

from aggregate_latency import (
    load_and_validate as load_timing,
    build_combo_column,
    filter_to_valid_combos as filter_combos_timing,
    aggregate_over_combos,
    sort_by_dataset_then_model,
)
from aggregate_metrics import (
    load_and_validate as load_metrics,
    filter_to_valid_combos as filter_combos_metrics,
    aggregate as aggregate_metrics,
)


def get_timing(path: str) -> pd.DataFrame:
    df = load_timing(path)
    df = build_combo_column(df)
    df = filter_combos_timing(df)
    df = aggregate_over_combos(df)
    return df[["dataset", "model", "avg latency (ms)"]]


def get_metrics(paths: list[str]) -> pd.DataFrame:
    frames = []
    metric_names = set()
    for path in paths:
        df, metric_col = load_metrics(path)
        metric_names.add(metric_col)
        df = filter_combos_metrics(df, path)
        df = aggregate_metrics(df, metric_col, path)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    if len(metric_names) > 1:
        # Melt mixed metric columns into a generic "metric" column.
        combined = combined.rename(columns={next(iter(metric_names)): "metric"})
        for col in list(metric_names)[1:]:
            if col in combined.columns:
                combined["metric"] = combined["metric"].combine_first(combined[col])
                combined = combined.drop(columns=[col])
        metric_col_out = "metric"
    else:
        metric_col_out = next(iter(metric_names))

    combined[metric_col_out] = combined[metric_col_out].round(4)
    return combined, metric_col_out


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str], col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].replace(mapping)
    return df


def load_mapping(path: str) -> dict[str, str]:
    with open(path) as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        sys.exit(f"ERROR: {path} must be a JSON object (dict)")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join retrieval metrics and latency into one results table."
    )
    parser.add_argument("--metrics", nargs="+", required=True,
                        help="One or more metrics CSV files (one per dataset)")
    parser.add_argument("--timing", required=True,
                        help="Timing CSV file")
    parser.add_argument("--mapping",
                        help="JSON file mapping model name aliases to canonical names")
    parser.add_argument("--print-unmatched", action="store_true",
                        help="Print models that appear in only one source and exit")
    parser.add_argument("-o", "--output",
                        help="Output CSV path (default: stdout)")
    args = parser.parse_args()

    mapping = load_mapping(args.mapping) if args.mapping else {}

    timing_df = get_timing(args.timing)
    metrics_df, metric_col = get_metrics(args.metrics)

    timing_df = apply_mapping(timing_df, mapping, "model")
    metrics_df = apply_mapping(metrics_df, mapping, "model")

    if args.print_unmatched:
        timing_models  = set(zip(timing_df["dataset"],  timing_df["model"]))
        metrics_models = set(zip(metrics_df["dataset"], metrics_df["model"]))
        only_timing  = sorted(timing_models  - metrics_models)
        only_metrics = sorted(metrics_models - timing_models)
        if only_timing:
            print("In timing only (need mapping or missing from metrics):")
            for ds, m in only_timing:
                print(f"  [{ds}] {m!r}")
        if only_metrics:
            print("In metrics only (need mapping or missing from timing):")
            for ds, m in only_metrics:
                print(f"  [{ds}] {m!r}")
        if not only_timing and not only_metrics:
            print("All models matched.")
        return

    result = timing_df.merge(metrics_df, on=["dataset", "model"], how="inner")

    unmatched_timing  = set(zip(timing_df["dataset"],  timing_df["model"]))  \
                      - set(zip(result["dataset"],     result["model"]))
    unmatched_metrics = set(zip(metrics_df["dataset"], metrics_df["model"])) \
                      - set(zip(result["dataset"],     result["model"]))

    if unmatched_timing:
        print("Warning: these timing rows had no matching metric row (dropped):", file=sys.stderr)
        for ds, m in sorted(unmatched_timing):
            print(f"  [{ds}] {m!r}", file=sys.stderr)
    if unmatched_metrics:
        print("Warning: these metric rows had no matching timing row (dropped):", file=sys.stderr)
        for ds, m in sorted(unmatched_metrics):
            print(f"  [{ds}] {m!r}", file=sys.stderr)

    result = sort_by_dataset_then_model(result)

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Saved {len(result)} rows → {args.output}")
    else:
        print(result.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
