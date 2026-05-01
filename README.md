


## Configuration

All scripts resolve data, results, and model paths through environment variables. Set these before running any script:

| Variable | Used for | Default (cluster) |
|---|---|---|
| `BASE_DATA_DIR` | Dataset CSV files | `/extra/huaiyaom0/tr-intern/wrrf/dataset` |
| `BASE_RESULTS_DIR` | Output `.trec` / metrics files | `/extra/huaiyaom0/tr-intern/wrrf/results` |
| `BASE_EXPERIMENT_DIR` | Saved model checkpoints | `/extra/huaiyaom0/tr-intern/wrrf/experiment` |

Copy `.env.local`, fill in your paths, and save it as `.env`:

```bash
cp .env.local .env
# edit .env with your paths
```

Scripts load `.env` automatically via `utils/env.py`. You can also add the variables directly to your shell profile (`~/.zshrc` / `~/.bashrc`) or set them in your SageMaker environment settings.

---

## Data Collection

### Optimal Fusion Weights (Ground Truth)

To collect the optimal fusion weights for each dataset, use one of the following scripts:

* `rrf_mrr_best_weights_all_weights_final.py`
* `rrf_ndcg_best_weights_all_weights_final.py`

Choose the script based on the dataset characteristics:

* If each query has **only one relevant document**, use the **MRR-based** script.
* If each query has **multiple relevant documents**, use the **nDCG-based** script.

**Note**: A query can have more than one optimal weight that achieves the highest MRR or nDCG score. However, some queries may not have any optimal weights because their MRR or nDCG score is 0 across all weight ranges.

### Mean Optimal Weight

---

## Fusion Strategy Evaluation
### Fuse Search Results Use Specific Weight

After obtaining the predicted optimal weights from the model, use:

```
helper_4_fuse_results_wrrf.py
```

This script fuses search results from different retrievers, including:

* **Sparse retrievers**: BM25, RM3
* **Dense retrievers**: Bi-encoder, Qwen3

The fused results are saved in **TREC format**.

Finally, use:

```
helper_5_ir_metrics.py
```

to compute evaluation metrics such as **nDCG** and **MRR**.

---


## Experiments

### Experiment 1: Standard RRF

In this setting, both sparse retrievers (`bm25`, `rm3`) and dense retrievers (`biencoder`, `qwen3`) use equal fusion weights (0.5).

To run this experiment, use the script:

```
helper_4_fuse_results_wrrf.py
```

Set the following parameters in the script:

```python
use_fixed_weight = True  # Enable fixed fusion weights
sparse_weight = 0.5
```

Then manually switch the retriever names as needed:

```python
sparse_name = "rm3"      # Options: "bm25" or "rm3"
dense_name = "qwen3"     # Options: "biencoder" or "qwen3"
```

---

### Experiment 2: Mean Optimal Weight

For each dataset, use the **mean optimal fusion weight** collected from its training set to fuse the search results.

The execution logic is the same as in **Experiment 1 (Standard RRF)** — the only difference is that you set `sparse_weight` to the dataset-specific mean optimal weight instead of `0.5`.

---

### Experiment 3: Ridge Regression

See the directory:  
[`/experiment/ridge-regression/ridge-regression-mean-best-weight`](./experiment/ridge-regression/ridge-regression-mean-best-weight)

---

### Experiment 4: RoBERTa Regression

See the directory:  
[`/experiment/roberta/roberta-experiment-mean-best-weight`](./experiment/roberta/roberta-experiment-mean-best-weight)

---

### Experiment 5: LLM-Based Weight Prediction

See the directory:  
[`/experiment/llm_selected_sparse_w`](./experiment/llm_selected_sparse_w)

---

## IR Performance–Latency Tradeoff Analysis

`utils/analyze_ir_latency_tradeoff.py` is a single end-to-end script that takes raw timing and metric spreadsheets, joins them, and produces aggregated CSVs and per-dataset scatter plots.

### Required inputs

| Argument       | Description                                                                                                                                                                                                                                                                                                                         |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--timing`     | Timing CSV with one row per `(model × dataset × retriever combo)`. Must have columns: `model`, `dataset`, `sparse`, `dense`, `avg latency (ms)`.                                                                                                                                                                                    |
| `--metrics`    | One CSV per dataset (e.g. acord, msmarco, nfcorpus, nq). Must have columns: `Method`, `Dataset`, `Split`, `Sparse`, `Dense`, followed by metric columns — the 7th column (`NDCG@10` or `MRR@10`) is used. Metric values may include confidence intervals (`"0.132 [0.098, 0.166]"`); the point estimate is extracted automatically. |
| `--mapping`    | `utils/model_mapping.json` — maps model name variants across sheets to a single canonical name.                                                                                                                                                                                                                                     |
| `--output-dir` | Base output directory (default: `./`).                                                                                                                                                                                                                                                                                              |

Both inputs are averaged over the four retriever combos (`bm25_vs_biencoder`, `bm25_vs_qwen3`, `rm3_vs_biencoder`, `rm3_vs_qwen3`). Models with no per-query inference cost (RRF, MOW) are assigned `latency = 0`.

### Outputs

All outputs are written under `<output-dir>/results/`:

```
results/
├── timing_aggregated.csv     # avg latency per (dataset, model)
├── metrics_aggregated.csv    # avg metric per (dataset, model)
├── combined_results.csv      # inner join of the two above
└── plots/
    ├── tradeoff_acord.png
    ├── tradeoff_msmarco.png
    ├── tradeoff_nfcorpus.png
    └── tradeoff_nq.png
```

### Example

```bash
python utils/analyze_ir_latency_tradeoff.py \
    --timing  "wrrf tracker - timing.csv" \
    --metrics acord.csv msmarco.csv nfcorpus.csv nq.csv \
    --mapping utils/model_mapping.json \
    --output-dir ./results
```

