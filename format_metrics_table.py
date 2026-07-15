#!/usr/bin/env python3
"""
Format an IR metrics CSV into a summary table with "mean [lower, upper]" cells.

Detects metrics from columns named avg_{metric}_{cutoff} and pairs them with
{metric}_lower_{cutoff} / {metric}_upper_{cutoff} bootstrap confidence interval columns.

Usage:
    python format_metrics_table.py input.csv [output_stem]

Output:
    {output_stem}.csv   — quoted CSV safe for Excel/spreadsheets
    {output_stem}.md    — Markdown table
    (default stem: input file path without extension + "_formatted")
"""

import csv
import sys
from pathlib import Path

KEY_COLS = ["dataset", "split", "sparse", "dense"]
DECIMALS = 3

NDCG_FIRST_DATASETS = {"acord-entire-corpus", "nfcorpus"}
MRR_FIRST_DATASETS  = {"nq", "msmarco"}


def detect_metrics(fieldnames: list[str]) -> dict[tuple[str, str], tuple[str, str, str]]:
    """Return {(metric, cutoff): (mean_col, lower_col, upper_col)} for all detectable metrics."""
    metrics = {}
    for col in fieldnames:
        if not col.startswith("avg_"):
            continue
        suffix = col[len("avg_"):]           # e.g. "ndcg_5" or "mrr_10"
        parts = suffix.rsplit("_", 1)
        if len(parts) != 2:
            continue
        metric, cutoff = parts
        lower_col = f"{metric}_lower_{cutoff}"
        upper_col = f"{metric}_upper_{cutoff}"
        if lower_col in fieldnames and upper_col in fieldnames:
            metrics[(metric, cutoff)] = (col, lower_col, upper_col)
    return metrics


def fmt(value: str, label: str, row_key: str) -> str:
    if value is None or value.strip() == "":
        print(f"WARNING: missing value for '{label}' in row {row_key}", file=sys.stderr)
        return "NaN"
    try:
        return f"{float(value):.{DECIMALS}f}"
    except ValueError:
        print(f"WARNING: non-numeric value '{value}' for '{label}' in row {row_key}", file=sys.stderr)
        return "NaN"


def cell(mean_col: str, lower_col: str, upper_col: str, row: dict, row_key: str) -> str:
    mean  = fmt(row[mean_col],  mean_col,  row_key)
    lower = fmt(row[lower_col], lower_col, row_key)
    upper = fmt(row[upper_col], upper_col, row_key)
    if "NaN" in (mean, lower, upper):
        return "NaN"
    return f"{mean} [{lower}, {upper}]"


def order_metric_cols(metric_cols: list[str], datasets: set[str]) -> list[str]:
    """Put ndcg or mrr first depending on dataset group; map and others go last."""
    ndcg_cols  = [c for c in metric_cols if c.startswith("ndcg@")]
    mrr_cols   = [c for c in metric_cols if c.startswith("mrr@")]
    other_cols = [c for c in metric_cols if not c.startswith(("ndcg@", "mrr@"))]

    if datasets and datasets <= MRR_FIRST_DATASETS:
        ordered = mrr_cols + ndcg_cols + other_cols
    elif datasets and not (datasets <= NDCG_FIRST_DATASETS):
        # mixed: datasets span both groups — warn and fall back to ndcg-first
        print(
            f"WARNING: datasets {datasets} span both ordering groups; defaulting to ndcg-first.",
            file=sys.stderr,
        )
        ordered = ndcg_cols + mrr_cols + other_cols
    else:
        ordered = ndcg_cols + mrr_cols + other_cols

    return ordered


def build_output_rows(
    reader: csv.DictReader,
    metrics: dict[tuple[str, str], tuple[str, str, str]],
) -> tuple[list[str], list[dict]]:
    sorted_metrics = sorted(metrics.keys(), key=lambda x: (x[0], int(x[1])))
    all_metric_cols = [f"{metric}@{cutoff}" for metric, cutoff in sorted_metrics]

    raw_rows = list(reader)
    datasets = {row["dataset"] for row in raw_rows}
    ordered_metric_cols = order_metric_cols(all_metric_cols, datasets)
    out_cols = KEY_COLS + ordered_metric_cols

    rows = []
    for row in raw_rows:
        out_row = {k: row[k] for k in KEY_COLS}
        row_key = ",".join(row[k] for k in KEY_COLS)
        for (metric, cutoff) in sorted_metrics:
            mean_col, lower_col, upper_col = metrics[(metric, cutoff)]
            col_label = f"{metric}@{cutoff}"
            out_row[col_label] = cell(mean_col, lower_col, upper_col, row, row_key)
        rows.append(out_row)

    return out_cols, rows


def write_csv(path: Path, cols: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote CSV  → {path}")


def write_markdown(path: Path, cols: list[str], rows: list[dict]) -> None:
    col_widths = {c: max(len(c), max(len(r[c]) for r in rows)) for c in cols}

    def md_row(values: dict) -> str:
        cells = [values[c].ljust(col_widths[c]) for c in cols]
        return "| " + " | ".join(cells) + " |"

    def separator() -> str:
        dashes = ["-" * col_widths[c] for c in cols]
        return "| " + " | ".join(dashes) + " |"

    with open(path, "w") as f:
        f.write(md_row({c: c for c in cols}) + "\n")
        f.write(separator() + "\n")
        for row in rows:
            f.write(md_row(row) + "\n")
    print(f"Wrote MD   → {path}")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} input.csv [output_stem]", file=sys.stderr)
        sys.exit(1)

    in_path = Path(sys.argv[1]).expanduser()
    if not in_path.exists():
        print(f"File not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    stem = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else in_path.with_name(in_path.stem + "_formatted")

    with open(in_path) as f:
        reader = csv.DictReader(f)
        metrics = detect_metrics(reader.fieldnames)

        if not metrics:
            print("No metrics detected. Expected columns like avg_ndcg_10, ndcg_lower_10, ndcg_upper_10.", file=sys.stderr)
            sys.exit(1)

        print(f"Detected metrics: {sorted(metrics.keys())}")
        cols, rows = build_output_rows(reader, metrics)

    write_csv(Path(str(stem) + ".csv"), cols, rows)
    write_markdown(Path(str(stem) + ".md"), cols, rows)


if __name__ == "__main__":
    main()
