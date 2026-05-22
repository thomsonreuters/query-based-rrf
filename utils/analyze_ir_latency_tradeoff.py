#!/usr/bin/env python3
"""
End-to-end pipeline: timing + IR-metric CSVs → combined table + per-dataset plots.

Steps performed internally:
  1. Load and prepare timing CSV (pre-aggregation)
  2. Aggregate metric CSV(s)    → (dataset, model, <metric>)
  3. Apply model-name mapping to both
  4. Exclude unwanted models from both
  5. Fill missing llm-fs-* combo rows with estimated latency
  6. Aggregate timing           → results/timing_aggregated.csv
  7. Inner-join timing × metrics → results/combined_results.csv
  8. Generate a 2×2 tradeoff grid (one subplot per dataset) → results/plots/tradeoff_all.svg/pdf

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
    VALID_COMBOS,
)
from aggregate_metrics import (
    load_and_validate as load_metrics,
    filter_to_valid_combos as filter_metrics,
    aggregate as aggregate_metric,
)
from plot_tradeoff import plot_tradeoff_grid

# Models excluded from both timing and metrics before any processing.
EXCLUDED_MODELS = {"DAT-gpt5.2"}

# Estimated per-query latency (ms) used to fill missing combo rows for llm-fs-* models.
# Replace with measured values once timing is complete.
LLM_FS_ESTIMATED_LATENCY_MS = 269.0
LLM_FS_PREFIX = "llm-fs-"

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

def prepare_timing(timing_path: str) -> pd.DataFrame:
    """Load, build combo column, and filter timing to valid combos (pre-aggregation)."""
    df = load_timing(timing_path)
    df = build_combo_column(df)
    df = filter_timing(df)
    return df


def fill_llm_fs_timing(timing_df: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    For llm-fs-* models, replace zero-latency rows and add missing combo rows with
    LLM_FS_ESTIMATED_LATENCY_MS so the combo average is computed over all 4 combos
    using only real or estimated values — never zeros.

    Datasets to fill for are taken from metrics_df, ensuring models entirely absent
    from timing still get the correct per-dataset rows.
    """
    df = timing_df.copy()

    # Replace 0ms rows for llm-fs-* models with the estimated latency.
    llm_fs_mask = df["model"].str.startswith(LLM_FS_PREFIX) & (df["avg latency (ms)"] == 0)
    n_replaced = llm_fs_mask.sum()
    if n_replaced:
        df.loc[llm_fs_mask, "avg latency (ms)"] = LLM_FS_ESTIMATED_LATENCY_MS
        print(
            f"          replaced {n_replaced} zero-latency llm-fs-* row(s) "
            f"with {LLM_FS_ESTIMATED_LATENCY_MS} ms (estimated)",
            file=sys.stderr,
        )

    # Add rows for combos entirely absent from the timing sheet.
    llm_fs_models = [m for m in metrics_df["model"].unique() if m.startswith(LLM_FS_PREFIX)]
    datasets = metrics_df["dataset"].unique().tolist()
    rows = []
    for model in llm_fs_models:
        for dataset in datasets:
            existing = set(
                df.loc[(df["model"] == model) & (df["dataset"] == dataset), "combo"]
            )
            for combo in VALID_COMBOS - existing:
                sparse, dense = combo.split("_vs_")
                rows.append({
                    "model": model, "dataset": dataset,
                    "sparse": sparse, "dense": dense, "combo": combo,
                    "avg latency (ms)": LLM_FS_ESTIMATED_LATENCY_MS,
                    "total time (ms)": float("nan"),
                    "infra": "", "num of queries": float("nan"), "num of param": "",
                })

    if rows:
        print(
            f"          added {len(rows)} missing llm-fs-* combo row(s) "
            f"with {LLM_FS_ESTIMATED_LATENCY_MS} ms (estimated)",
            file=sys.stderr,
        )
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    return df


def aggregate_metrics(metric_paths: list[str], mapping: dict[str, str]) -> tuple[pd.DataFrame, str]:
    frames = []
    metric_names = set()
    for path in metric_paths:
        df, metric_col = load_metrics(path)
        metric_names.add(metric_col)
        df = filter_metrics(df, path)
        # Apply mapping before groupby so aliases (e.g. two Method names for the
        # same canonical model) are merged into one group and averaged over all 4 combos.
        df["Method"] = df["Method"].replace(mapping)
        df = aggregate_metric(df, metric_col, path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if len(metric_names) > 1:
        first = next(iter(metric_names))
        combined = combined.rename(columns={first: "metric"})
        for col in sorted(metric_names - {first}):
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


def exclude_models(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["model"].isin(EXCLUDED_MODELS)].copy()


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
    datasets = []
    for dataset, group in df.groupby("dataset", sort=False, observed=True):
        models = list(group[["model", "avg latency (ms)", metric_col]].itertuples(index=False, name=None))
        datasets.append((
            DATASET_METRIC.get(dataset, metric_col),
            DATASET_DISPLAY.get(dataset, dataset.upper()),
            models,
        ))
    plot_tradeoff_grid(
        datasets,
        output=str(plots_dir / "tradeoff_all.svg"),
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
                        help="Base output directory (default: current directory).")
    args = parser.parse_args()

    results_dir = Path(args.output_dir) / "results"
    plots_dir   = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)

    mapping = {}
    if args.mapping:
        with open(args.mapping) as f:
            mapping = json.load(f)

    print("Step 1/4  loading timing …")
    timing_raw = prepare_timing(args.timing)
    timing_raw = apply_mapping(timing_raw, mapping)
    timing_raw = exclude_models(timing_raw)

    print("Step 2/4  aggregating IR metrics …")
    metrics_df, metric_col = aggregate_metrics(args.metrics, mapping)
    metrics_df = exclude_models(metrics_df)
    metrics_out = results_dir / "metrics_aggregated.csv"
    metrics_df.to_csv(metrics_out, index=False)
    print(f"          saved → {metrics_out}")

    # Fill any missing llm-fs-* combo rows before averaging over combos.
    timing_raw = fill_llm_fs_timing(timing_raw, metrics_df)

    timing_df = aggregate_over_combos(timing_raw)[["dataset", "model", "avg latency (ms)"]]
    timing_out = results_dir / "timing_aggregated.csv"
    timing_df.to_csv(timing_out, index=False)
    print(f"          saved → {timing_out}")

    print("Step 3/4  joining …")
    result = join(timing_df, metrics_df)
    print(f"          {len(result)} rows in combined table")
    combined_out = results_dir / "combined_results.csv"
    result.to_csv(combined_out, index=False)
    print(f"          saved → {combined_out}")

    print("Step 4/4  generating 2×2 plot …")
    make_plots(result, metric_col, plots_dir)

    print("Done.")


if __name__ == "__main__":
    main()
