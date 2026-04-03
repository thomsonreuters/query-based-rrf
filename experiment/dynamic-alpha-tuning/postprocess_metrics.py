import csv
import argparse
from pathlib import Path


def format_metric(mean: float, lower: float, upper: float) -> str:
    return f"{mean:.3f} [{lower:.3f}, {upper:.3f}]"


def process_metrics_csv(input_file: str) -> list[dict]:
    rows = []
    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed = {
                "Dataset": row["dataset"],
                "Split": row["split"],
                "Sparse": row["sparse"],
                "Dense": row["dense"],
                "NDCG@5": format_metric(
                    float(row["avg_ndcg_5"]),
                    float(row["ndcg_lower_5"]),
                    float(row["ndcg_upper_5"]),
                ),
                "NDCG@10": format_metric(
                    float(row["avg_ndcg_10"]),
                    float(row["ndcg_lower_10"]),
                    float(row["ndcg_upper_10"]),
                ),
                "MRR@5": format_metric(
                    float(row["avg_mrr_5"]),
                    float(row["mrr_lower_5"]),
                    float(row["mrr_upper_5"]),
                ),
                "MRR@10": format_metric(
                    float(row["avg_mrr_10"]),
                    float(row["mrr_lower_10"]),
                    float(row["mrr_upper_10"]),
                ),
                "MAP@5": format_metric(
                    float(row["avg_map_5"]),
                    float(row["map_lower_5"]),
                    float(row["map_upper_5"]),
                ),
                "MAP@10": format_metric(
                    float(row["avg_map_10"]),
                    float(row["map_lower_10"]),
                    float(row["map_upper_10"]),
                ),
            }
            rows.append(processed)
    return rows


def save_table(rows: list[dict], output_file: str) -> None:
    headers = ["Dataset", "Split", "Sparse", "Dense", "NDCG@5", "NDCG@10", "MRR@5", "MRR@10", "MAP@5", "MAP@10"]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process metrics.csv into a formatted table.")
    parser.add_argument("input_file", help="Path to metrics.csv")
    args = parser.parse_args()

    output_file = Path(args.input_file).parent / "metrics_processed.csv"
    rows = process_metrics_csv(args.input_file)
    save_table(rows, str(output_file))
    print(f"Saved to {output_file}")
