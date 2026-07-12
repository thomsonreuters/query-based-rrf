#!/usr/bin/env python3
"""
Sort per-query prediction CSVs by the distance from `predicted` to the
nearest endpoint of the best-weight region(s) in `friendly_best_weights`,
and write them to ./sorted_scores, mirroring the original method/combo
folder structure.

`friendly_best_weights` is a list of segments, each either:
  - a bare float / single-value list, e.g. 0.65 or [0.65]  -> a single point
  - a two-value list, e.g. [0.30, 0.35]                    -> an interval
The primary sort key is the min gap across segments: 0 if `predicted` falls
inside an interval segment, else the distance to the nearest endpoint/point.
Many rows tie at gap 0 (predicted already inside the interval), so a
secondary key breaks ties: |predicted - center of that segment|. This keeps
the ordering deterministic and ranks "how centered" a good prediction is.

Usage:
    python sort_scores_by_error.py
    python sort_scores_by_error.py --methods 01-standard-rrf 03-ridge-regression
    python sort_scores_by_error.py --methods 01 03 10
    python sort_scores_by_error.py --datasets msmarco nq
    python sort_scores_by_error.py --ascending
    python sort_scores_by_error.py --output-root /path/to/results
"""
import argparse
import ast
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT_DIRNAME = "sorted_scores"
DATASETS = ["acord-entire-corpus", "msmarco", "nfcorpus", "nq"]


def discover_methods():
    return sorted(
        p for p in ROOT.iterdir()
        if p.is_dir() and p.name != OUTPUT_DIRNAME and re.match(r"^\d", p.name)
    )


def method_matches(method_dir: Path, selected):
    if selected is None:
        return True
    name = method_dir.name
    for sel in selected:
        if sel == name:
            return True
        if sel.isdigit():
            num_prefix = f"{int(sel):02d}-"
            if name.startswith(num_prefix):
                return True
    return False


def gap_key(predicted: float, friendly_best_weights: str) -> tuple:
    """(primary_gap, secondary_gap) for the segment that minimizes primary_gap
    (and, among ties, secondary_gap). primary_gap is 0 inside an interval,
    else distance to the nearest endpoint. secondary_gap is the distance to
    that segment's center, used only to break primary_gap ties."""
    segments = ast.literal_eval(friendly_best_weights)
    keys = []
    for seg in segments:
        if isinstance(seg, (list, tuple)):
            lo, hi = (seg[0], seg[0]) if len(seg) == 1 else (min(seg), max(seg))
        else:
            lo = hi = seg
        primary = 0.0 if lo <= predicted <= hi else min(abs(predicted - lo), abs(predicted - hi))
        secondary = abs(predicted - (lo + hi) / 2)
        keys.append((primary, secondary))
    return min(keys)


def dataset_of(filename: str):
    for ds in sorted(DATASETS, key=len, reverse=True):
        if filename == ds or filename.startswith(ds + "-") or filename.startswith(ds + "_"):
            return ds
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Method folder names or numeric prefixes (e.g. 01-standard-rrf or 01). Default: all.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None, choices=DATASETS,
        help="Dataset names to include. Default: all.",
    )
    parser.add_argument(
        "--ascending", action="store_true",
        help="Sort smallest error first (default: largest error first).",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT,
        help=f"Directory under which '{OUTPUT_DIRNAME}/' is created. Default: this script's directory.",
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_dir = output_root / OUTPUT_DIRNAME

    methods = [m for m in discover_methods() if method_matches(m, args.methods)]
    if not methods:
        print("No matching method folders found.")
        return

    selected_datasets = set(args.datasets) if args.datasets else set(DATASETS)

    n_written = 0
    for method_dir in methods:
        for scores_dir in sorted(method_dir.glob("**/prediction-scores")):
            for csv_path in sorted(scores_dir.glob("*.csv")):
                ds = dataset_of(csv_path.name)
                if ds is None or ds not in selected_datasets:
                    continue

                df = pd.read_csv(csv_path)
                keys = df.apply(
                    lambda row: gap_key(row["predicted"], row["friendly_best_weights"]), axis=1
                )
                df["primary_gap"] = keys.apply(lambda k: k[0])
                df["secondary_gap"] = keys.apply(lambda k: k[1])
                df = df.sort_values(
                    ["primary_gap", "secondary_gap"], ascending=[args.ascending, args.ascending]
                )

                rel_path = csv_path.relative_to(ROOT)
                out_path = output_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out_path, index=False)
                n_written += 1
                print(f"wrote {out_path.relative_to(output_root)}")

    print(f"\nDone. {n_written} CSV(s) written under {output_dir}/")


if __name__ == "__main__":
    main()
