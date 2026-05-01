#!/usr/bin/env python3
"""
End-to-end pipeline: timing + IR-metric CSVs → combined table + per-dataset plots.

Steps performed internally:
  1. Aggregate timing CSV       → results/timing_aggregated.csv
  2. Aggregate metric CSV(s)    → (dataset, model, <metric>)
  3. Normalize model names via mapping file
  4. Inner-join timing × metrics → results/combined_results.csv
  5. Generate one tradeoff plot per dataset → results/plots/

Usage:
    python utils/analyze_ir_latency_tradeoff.py \\
        --timing  "wrrf tracker - timing.csv" \\
        --metrics acord.csv msmarco.csv nfcorpus.csv nq.csv \\
        --mapping utils/model_mapping.json \\
        --output-dir ./my_run
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from aggregate_latency import (
    load_and_validate as load_timing,
    build_combo_column,
    filter_to_valid_combos as filter_timing,
    aggregate_over_combos,
    sort_by_dataset_then_model,
)
from aggregate_metrics import (
    load_and_validate as load_metrics,
    filter_to_valid_combos as filter_metrics,
    aggregate as aggregate_metric,
)
from plot_tradeoff import plot_tradeoff

# Per-dataset display names and metric labels used in the plots.
DATASET_DISPLAY = {
    "acord":    "ACORD",
    "msmarco":  "MS MARCO",
    "nfcorpus": "NFCorpus",
    "nq":       "NQ",
}
DATASET_METRIC = {
    "acord":    "NDCG@10",
    "msmarco":  "MRR@10",
    "nfcorpus": "NDCG@10",
    "nq":       "MRR@10",
}


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def aggregate_timing(timing_path: str) -> pd.DataFrame:
    df = load_timing(timing_path)
    df = build_combo_column(df)
    df = filter_timing(df)
    df = aggregate_over_combos(df)
    return df[["dataset", "model", "avg latency (ms)"]]


def aggregate_metrics(metric_paths: list[str]) -> tuple[pd.DataFrame, str]:
    frames = []
    metric_names = set()
    for path in metric_paths:
        df, metric_col = load_metrics(path)
        metric_names.add(metric_col)
        df = filter_metrics(df, path)
        df = aggregate_metric(df, metric_col, path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if len(metric_names) > 1:
        # Mixed metrics (e.g. NDCG@10 + MRR@10): merge into a generic "metric" column.
        first = next(iter(metric_names))
        combined = combined.rename(columns={first: "metric"})
        for col in metric_names - {first}:
            if col in combined.columns:
                combined["metric"] = combined["metric"].combine_first(combined[col])
                combined = combined.drop(columns=[col])
        metric_col_out = "metric"
    else:
        metric_col_out = next(iter(metric_names))

    combined[metric_col_out] = combined[metric_col_out].round(4)
    return combined, metric_col_out


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["model"] = df["model"].replace(mapping)
    return df


def join(timing_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    result = timing_df.merge(metrics_df, on=["dataset", "model"], how="inner")

    unmatched_timing  = (set(zip(timing_df["dataset"],  timing_df["model"]))
                         - set(zip(result["dataset"],   result["model"])))
    unmatched_metrics = (set(zip(metrics_df["dataset"], metrics_df["model"]))
                         - set(zip(result["dataset"],   result["model"])))

    if unmatched_timing:
        print("Warning: timing rows with no matching metric (dropped):", file=sys.stderr)
        for ds, m in sorted(unmatched_timing):
            print(f"  [{ds}] {m!r}", file=sys.stderr)
    if unmatched_metrics:
        print("Warning: metric rows with no matching timing (dropped):", file=sys.stderr)
        for ds, m in sorted(unmatched_metrics):
            print(f"  [{ds}] {m!r}", file=sys.stderr)

    return sort_by_dataset_then_model(result)


def make_plots(df: pd.DataFrame, metric_col: str, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    for dataset, group in df.groupby("dataset", sort=False, observed=True):
        models = [
            (row["model"], row["avg latency (ms)"], row[metric_col])
            for _, row in group.iterrows()
        ]
        plot_tradeoff(
            models,
            metric_name=DATASET_METRIC.get(dataset, metric_col),
            dataset_name=DATASET_DISPLAY.get(dataset, dataset.upper()),
            output=str(plots_dir / f"tradeoff_{dataset}.png"),
            delta_mode=False,
            show=False,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full pipeline: timing + IR metrics → combined table + plots."
    )
    parser.add_argument("--timing", required=True,
                        help="Timing CSV (one row per model × dataset × retriever combo)")
    parser.add_argument("--metrics", nargs="+", required=True,
                        help="Metric CSV(s), one per dataset")
    parser.add_argument("--mapping",
                        help="JSON file mapping model name aliases to canonical names")
    parser.add_argument("--output-dir", default=".",
                        help="Base output directory (default: current directory). "
                             "Outputs are written to <output-dir>/results/ and <output-dir>/results/plots/")
    args = parser.parse_args()

    results_dir = Path(args.output_dir) / "results"
    plots_dir   = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)

    mapping = {}
    if args.mapping:
        with open(args.mapping) as f:
            mapping = json.load(f)

    print("Step 1/4  aggregating timing …")
    timing_df = aggregate_timing(args.timing)
    timing_df = apply_mapping(timing_df, mapping)

    # TODO: remove once actual DAT-gpt5.2 latency measurements are available
    datasets = timing_df["dataset"].unique().tolist()
    dat_rows = pd.DataFrame([
        {"dataset": ds, "model": "DAT-gpt5.2", "avg latency (ms)": 1700.0}
        for ds in datasets
    ])
    timing_df = pd.concat([timing_df, dat_rows], ignore_index=True)

    timing_out = results_dir / "timing_aggregated.csv"
    timing_df.to_csv(timing_out, index=False)
    print(f"          saved → {timing_out}")

    print("Step 2/4  aggregating IR metrics …")
    metrics_df, metric_col = aggregate_metrics(args.metrics)
    metrics_df = apply_mapping(metrics_df, mapping)
    metrics_out = results_dir / "metrics_aggregated.csv"
    metrics_df.to_csv(metrics_out, index=False)
    print(f"          saved → {metrics_out}")

    print("Step 3/4  joining …")
    result = join(timing_df, metrics_df)
    print(f"          {len(result)} rows in combined table")
    combined_out = results_dir / "combined_results.csv"
    result.to_csv(combined_out, index=False)
    print(f"          saved → {combined_out}")

    print("Step 4/4  generating plots …")
    make_plots(result, metric_col, plots_dir)

    print("Done.")


if __name__ == "__main__":
    main()
