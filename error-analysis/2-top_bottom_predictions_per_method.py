#!/usr/bin/env python3
"""
Build one good/weak predictions file per method per dataset, reusing the
already-sorted CSVs from 1-sort_scores_by_error.py (sorted descending by
gap: largest gap first, smallest gap last).

For a given method + dataset, there are up to 4 retriever-combo CSVs
(bm25_vs_biencoder, bm25_vs_qwen3, rm3_vs_biencoder, rm3_vs_qwen3). From
each combo file's existing sort order (primary_gap, secondary_gap columns
from step 1):
  - the head N% rows (largest gap)  -> weak candidates
  - the tail N% rows (smallest gap) -> good candidates
are taken and appended across all 4 combos, reusing those columns as-is
(no gap recomputation). The same query_id can appear in more than one
combo's slice; duplicates are resolved by keeping the most extreme
occurrence: worst (largest primary_gap, then largest secondary_gap) for
the weak set, best (smallest primary_gap, then smallest secondary_gap)
for the good set.

Output: top_bottom_predictions/<method>/.../prediction-scores/{dataset}_top_bottom_predictions.csv
(a separate tree from sorted_scores, so this script's own output is never
picked up as an input combo file on a later run) with an added
`prediction_quality` column (good_prediction / weak_prediction) and a
`source_combo_file` column (originating combo CSV, for traceability).

Usage:
    python 2-top_bottom_predictions_per_method.py
    python 2-top_bottom_predictions_per_method.py --methods 01-standard-rrf 03
    python 2-top_bottom_predictions_per_method.py --datasets msmarco nq
    python 2-top_bottom_predictions_per_method.py --pct 5
    python 2-top_bottom_predictions_per_method.py --output-root /path/to/results
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SORTED_DIRNAME = "sorted_scores"
OUTPUT_DIRNAME = "top_bottom_predictions"
DATASETS = ["acord-entire-corpus", "msmarco", "nfcorpus", "nq"]
OUTPUT_SUFFIX = "_top_bottom_predictions.csv"


def discover_methods(sorted_dir: Path):
    return sorted(p for p in sorted_dir.iterdir() if p.is_dir())


def method_matches(method_dir: Path, selected):
    if selected is None:
        return True
    name = method_dir.name
    for sel in selected:
        if sel == name:
            return True
        if sel.isdigit() and name.startswith(f"{int(sel):02d}-"):
            return True
    return False


def dataset_of(filename: str):
    for ds in sorted(DATASETS, key=len, reverse=True):
        if filename == ds or filename.startswith(ds + "-") or filename.startswith(ds + "_"):
            return ds
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=None,
                         help="Method folder names or numeric prefixes. Default: all.")
    parser.add_argument("--datasets", nargs="+", default=None, choices=DATASETS,
                         help="Dataset names to include. Default: all.")
    parser.add_argument("--pct", type=float, default=5,
                         help="Percent of rows taken from each combo's head (weak) and tail (good). Default: 5.")
    parser.add_argument("--output-root", type=Path, default=ROOT,
                         help=f"Directory containing '{SORTED_DIRNAME}/' (step 1's output) and under which "
                              f"'{OUTPUT_DIRNAME}/' is created. Default: this script's directory.")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    sorted_dir = output_root / SORTED_DIRNAME
    output_dir = output_root / OUTPUT_DIRNAME

    if not sorted_dir.exists():
        print(f"{sorted_dir} does not exist. Run 1-sort_scores_by_error.py first.")
        return

    methods = [m for m in discover_methods(sorted_dir) if method_matches(m, args.methods)]
    if not methods:
        print(f"No matching method folders found under {sorted_dir}/.")
        return

    selected_datasets = set(args.datasets) if args.datasets else set(DATASETS)

    n_written = 0
    for method_dir in methods:
        for scores_dir in sorted(method_dir.glob("**/prediction-scores")):
            by_dataset = {}
            for csv_path in sorted(scores_dir.glob("*.csv")):
                ds = dataset_of(csv_path.name)
                if ds in selected_datasets:
                    by_dataset.setdefault(ds, []).append(csv_path)

            for ds, csv_paths in by_dataset.items():
                weak_parts, good_parts = [], []
                for csv_path in csv_paths:
                    df = pd.read_csv(csv_path)
                    n = len(df)
                    k = max(1, round(n * args.pct / 100))
                    k = min(k, n // 2) if n >= 2 else n

                    weak = df.head(k).copy()
                    weak["source_combo_file"] = csv_path.name
                    weak_parts.append(weak)

                    good = df.tail(k).copy()
                    good["source_combo_file"] = csv_path.name
                    good_parts.append(good)

                weak_df = pd.concat(weak_parts, ignore_index=True)
                weak_df = weak_df.sort_values(["primary_gap", "secondary_gap"], ascending=[False, False])
                weak_df = weak_df.drop_duplicates(subset="query_id", keep="first")
                weak_df["prediction_quality"] = "weak_prediction"

                good_df = pd.concat(good_parts, ignore_index=True)
                good_df = good_df.sort_values(["primary_gap", "secondary_gap"], ascending=[True, True])
                good_df = good_df.drop_duplicates(subset="query_id", keep="first")
                good_df["prediction_quality"] = "good_prediction"

                combined = pd.concat([weak_df, good_df], ignore_index=True)

                out_dir = output_dir / scores_dir.relative_to(sorted_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{ds}{OUTPUT_SUFFIX}"
                combined.to_csv(out_path, index=False)
                n_written += 1
                print(f"wrote {out_path.relative_to(output_root)} "
                      f"({len(weak_df)} weak + {len(good_df)} good from {len(csv_paths)} combo(s))")

    print(f"\nDone. {n_written} file(s) written under {output_dir}/")


if __name__ == "__main__":
    main()
