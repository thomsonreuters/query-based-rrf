# Repository Overview: query-based-rrf

## Project Goal

Predict optimal per-query fusion weights for combining sparse retrievers (BM25, RM3) with dense
retrievers (Bi-encoder, Qwen3) using **Weighted Reciprocal Rank Fusion (WRRF)** in information
retrieval.

Instead of a fixed weight for all queries, a model predicts the best weight per query based on its
text. A weight of `1.0` means fully sparse (BM25/RM3); `0.0` means fully dense.

---

## Repository Structure

```
query-based-rrf/
├── README.md
├── REPO_OVERVIEW.md
├── helper_3_add_text2csv.py
├── helper_4_fuse_results_wrrf.py
├── helper_5_ir_metrics.py
├── helper_80_split_search_results.py
├── helper_81_get_top_k_results.py
├── helper_90_friendly_intervals.py
├── helper_91_mean_best_weight.py
├── helper_92_plot_all_dataset_weight_stats.py
├── helper_93_interval_distribution.py
├── get_mean_best_weight.py
└── experiment/
    ├── ridge-regression/
    │   └── ridge-regression-mean-best-weight/
    │       ├── config.yaml
    │       ├── train.py
    │       ├── test.py
    │       └── run.py
    ├── roberta/
    │   └── roberta-experiment-mean-best-weight/
    │       ├── config.yaml
    │       ├── train.py
    │       ├── test.py
    │       ├── run.py
    │       ├── run_mul.py
    │       └── README.md
    ├── llm_selected_sparse_w/
    │   ├── llm_select_sparse_w.py
    │   ├── llm_select_sparse_w_similarity.py
    │   └── run.py
    └── dynamic-alpha-tuning/
        └── dynamic_alpha_tuning.py
```

---

## Datasets & Combinations

| Dataset                       | Metric  | Test split | Used in             |
|-------------------------------|---------|------------|---------------------|
| MS MARCO                      | MRR@10  | `dev`      | Ridge, RoBERTa, LLM |
| NQ                            | MRR@10  | `dev`      | Ridge, RoBERTa, LLM |
| ACORD (`acord-entire-corpus`) | nDCG@10 | `test`     | Ridge, RoBERTa, LLM |
| NFCorpus                      | nDCG@10 | `test`     | Ridge, RoBERTa, LLM |
| TREC-COVID                    | nDCG@10 | `test`     | LLM only            |

Retriever combos (Ridge & RoBERTa): `bm25_vs_biencoder`, `bm25_vs_qwen3`, `rm3_vs_biencoder`,
`rm3_vs_qwen3` → **16 runs per model**.

---

## Data Preparation (Shared)

Run these steps once before any experiment.

### Step 1 — Split TREC results by qrels

```bash
python helper_80_split_search_results.py
```

Reads query IDs from `qrels/{train,dev,test}.tsv` and splits a single TREC result file into
per-split TREC files.

### Step 2 — Filter to top-200 per query

```bash
python helper_81_get_top_k_results.py
```

Filters each TREC file to the top 200 results per query (default).

### Step 3 — Add query text

```bash
python helper_3_add_text2csv.py
```

Reads `queries.jsonl` (fields: `_id`, `text`) and populates the `query_text` column in the CSV
files, producing `*_with_text.csv` outputs.

### Step 4 — Compute training labels

```bash
# Group weights into intervals, then compute per-query mean:
python helper_90_friendly_intervals.py
python helper_91_mean_best_weight.py
```

The `mean_best_weight` column (float in [0, 1]) is the regression target used by all models.

Expected data layout after all preparation steps:

```
dataset/
└── {dataset}/
    └── {mrr_runs|ndcg_runs}/
        ├── train/top200/
        ├── dev/top200/        # MS MARCO & NQ only
        └── test/top200/
            └── results_{split}_{combo}_best_weights_final_mean_with_text.csv
```

---

## Downstream IR Evaluation (Shared)

After obtaining predicted weights from any model:

```bash
# Fuse sparse + dense retrieval results using predicted weights (outputs TREC format)
python helper_4_fuse_results_wrrf.py

# Compute nDCG@10, MRR@10, MAP@10 from fused results
python helper_5_ir_metrics.py
```

### Analysis & Visualization

```bash
python helper_92_plot_all_dataset_weight_stats.py   # weight distribution across datasets
python helper_93_interval_distribution.py           # stacked bar chart of weight patterns
```

---

## Experiment 1: Ridge Regression

**Path:** `experiment/ridge-regression/ridge-regression-mean-best-weight/`

### Architecture

TF-IDF (unigrams–4-grams, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`) →
Ridge Regression (`alpha=100`). Lightweight, interpretable baseline.

### Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy pyyaml psutil
```

### Configuration

Key settings in `config.yaml`:

```yaml
model:
  ngram_range: [1, 4]
  min_df: 2
  max_df: 0.95
  sublinear_tf: true
regression:
  alpha: 100
  fit_intercept: true
  positive: false
training:
  seed: 42
  test_size: 0.1      # 10% held out as validation during training
```

### How to Run

```bash
cd experiment/ridge-regression/ridge-regression-mean-best-weight/
```

**Run all 16 combinations** (4 datasets × 4 combos) sequentially:

```bash
python run.py
```

`run.py` auto-selects the metric (`mrr` vs `ndcg`) and test split (`dev` vs `test`) per dataset,
then calls `train.py` → `test.py` for each combination.

**Run a single combination** by setting paths in `config.yaml`, then:

```bash
python train.py
```

> `test.py` cannot be run standalone — its `__main__` prints "Please run testing via run.py".
> Import `test_model(model_dir, test_file, output_base_dir)` directly if needed.

**Before running**, update `base_data_dir` in `run.py` (line 7):

```python
base_data_dir = "/extra/huaiyaom0/tr-intern/wrrf/dataset"  # change to your dataset root
```

### Expected Outputs

Saved under `experiments/{name}_{timestamp}/` during **training**:

```
config.yaml
script_train.py                   # copy of train.py for reproducibility
results.json                      # MAE, MSE, RMSE, R², Pearson/Spearman with 95% bootstrap CI
train_predictions.csv
val_predictions.csv
tfidf_vectorizer.pkl
ridge_regression_model.pkl
all_features_coefficients.csv
top_30_coefficients.png
coefficient_distribution.png
```

Saved under `predictions/` (working directory) during **testing**:

```
predictions/{dataset}_{combo}_{split}.csv
  columns: query_id, query_text, actual, predicted, absolute_error
```

---

## Experiment 2: RoBERTa

**Path:** `experiment/roberta/roberta-experiment-mean-best-weight/`

### Architecture

`roberta-large` encoder → pooler output → Dropout(0.01) → Linear(1024 → 1) → scalar weight.
MSE loss. Trained with HuggingFace `Trainer` (FP16 enabled, early stopping).

### Installation

```bash
pip install pandas numpy scikit-learn scipy pyyaml torch transformers
```

> PyTorch installation varies by platform and CUDA version.
> See https://pytorch.org/get-started/locally/ for the correct command for your environment.

### Configuration

Key settings in `config.yaml`:

```yaml
model:
  name: roberta-large       # or roberta-base
  max_length: 64
  dropout: 0.01
training:
  num_train_epochs: 10
  learning_rate: 2.0e-05
  per_device_train_batch_size: 256
  warmup_steps: 500
  early_stopping_patience: 10
  eval_steps: 200
  seed: 42
```

### How to Run

```bash
cd experiment/roberta/roberta-experiment-mean-best-weight/
```

**Option A — Multi-GPU** (recommended for all 16 combinations):

Edit `AVAILABLE_GPUS` in `run_mul.py` (line 113), then:

```python
AVAILABLE_GPUS = ["0", "1", "2", "4"]   # set to your available GPU IDs
```

```bash
python run_mul.py
```

`run_mul.py` uses `multiprocessing` with the `spawn` start method (required for PyTorch CUDA).
Each worker claims a GPU from a shared queue and runs one dataset–combo pair in parallel.

**Option B — Sequential** (all combinations on one GPU):

```bash
python run.py
```

**Run a single combination** by setting paths in `config.yaml`, then:

```bash
python train.py

# To test, edit model_path at line 186 in test.py, then:
python test.py
```

**Before running**, update `base_data_dir` in the relevant runner:

| Script       | Path type | Value                                     |
|--------------|-----------|-------------------------------------------|
| `run.py`     | absolute  | `/extra/huaiyaom0/tr-intern/wrrf/dataset` |
| `run_mul.py` | relative  | `../../../wrrf/dataset`                   |

### Expected Outputs

Saved under `experiments/{name}_{timestamp}/` during **training**:

```
config.yaml
script_train.py                 # copy of train.py for reproducibility
results.json                    # MAE, MSE, RMSE, R², Pearson, Spearman, MAPE (point estimates)
checkpoints/                    # HuggingFace Trainer intermediate checkpoints
best_model/
  ├── pytorch_model.bin
  ├── config.json
  └── tokenizer files
validation_predictions.csv      # predictions on the 10% validation split held out during training
```

Saved under `predictions/` (working directory) during **testing**:

```
predictions/{experiment_name}_{split}.csv
  # '_vs' is stripped from experiment name, e.g. msmarco-bm25_biencoder_dev.csv
  columns: all original CSV columns + predicted, error (mean_best_weight renamed to actual)
```

---

## Experiment 3: LLM (Azure GPT-4o)

**Path:** `experiment/llm_selected_sparse_w/`

### Architecture

Two scripts that prompt an Azure-hosted GPT-4o to predict the sparse weight directly from query
text, without any training:

| Script                              | Description                                                                                                                              |
|-------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `llm_select_sparse_w.py`            | Core LLM evaluation: 3 prompt versions, optional stratified few-shot examples from training data                                         |
| `llm_select_sparse_w_similarity.py` | Extends the above with sentence-transformer + BM25 example retrieval, additional `no_guidance` prompt, and Wilcoxon significance testing |

### Installation

```bash
# llm_select_sparse_w.py:
pip install pandas numpy requests tqdm langchain-core langchain-openai

# llm_select_sparse_w_similarity.py (adds):
pip install sentence-transformers rank-bm25 scipy
```

### Prerequisites

Access to the TR internal Azure AI Platform (`aiplatform.gcs.int.thomsonreuters.com`) with
workspace `PracticalLawxOhJ`. Credentials are fetched automatically at runtime via a POST
request — no manual API key setup is required on the TR network.

### Configuration

Edit the `__main__` block at the bottom of each script:

```python
# Which LLM(s) to evaluate:
model_versions = ['gpt-4o']           # e.g. ['gpt-4o', 'gpt-5']

# Which prompt style(s) to evaluate:
prompt_versions_to_run = ['combined'] # see prompt versions table below

# Dataset paths — list of (test_path, train_path) tuples:
datasets = [
    ('/path/to/dataset/nq/mrr_runs/dev/results_dev_all_queries.csv',
     '/path/to/dataset/nq/mrr_runs/train/results_train_best_weights_cleaned.csv'),
    # use train_path=None for zero-shot datasets (e.g. trec-covid)
    ('/path/to/dataset/trec-covid/ndcg_runs/test/results_test_all_queries.csv', None),
    ...
]
```

**Prompt versions:**

| Version       | Available in           | Style                                                               |
|---------------|------------------------|---------------------------------------------------------------------|
| `original`    | both scripts           | "predict sparse_w" framing with domain context                      |
| `joel`        | both scripts           | "assign selected_weight to BM25" framing                            |
| `combined`    | both scripts           | Detailed guidance with explicit BM25 vs dense tradeoff instructions |
| `no_guidance` | similarity script only | No domain guidance; default in `llm_select_sparse_w_similarity.py`  |

**Special case:** For `trec-covid`, training data is automatically loaded by combining the NQ and
MS MARCO training sets (paths hardcoded inside `load_combined_training_data()`).

### How to Run

```bash
cd experiment/llm_selected_sparse_w/

# Basic LLM experiment:
python llm_select_sparse_w.py

# LLM + similarity-based example retrieval + Wilcoxon tests:
python llm_select_sparse_w_similarity.py
```

### Caching

Results are cached per `query_id` under `{data_dir}/llm_cache/`. Interrupted runs resume
automatically by skipping already-processed queries.

Cache files written by `llm_select_sparse_w.py`:
```
{dataset}_{model}_{prompt}_no_examples_cache.csv
{dataset}_{model}_{prompt}_with_stratified_examples_cache.csv
```

Additional cache files written by `llm_select_sparse_w_similarity.py`:
```
{dataset}_{model}_{prompt}_similarity_based_k20_cache.csv
{dataset}_{model}_{prompt}_similarity_ranges_k20_cache.csv
```

### Expected Outputs

Console summary table comparing:
- Theoretical Limit (oracle best weight per query)
- Dense Only (`sparse_w = 0.0`)
- Sparse Only (`sparse_w = 1.0`)
- Hybrid Baseline (`sparse_w = 0.5`)
- Train Avg Weight Baseline
- LLM variants (no examples / with stratified examples, per prompt version)

`llm_select_sparse_w_similarity.py` additionally prints Wilcoxon test p-values comparing each
LLM method against the baselines.

---

## Experiment 4: Dynamic Alpha Tuning

**Path:** `experiment/dynamic-alpha-tuning/`

### Architecture

For each query, uses an LLM to score the quality of the top-1 result from each retriever (sparse
and dense) on a 0–5 scale, then computes a per-query alpha weight:

```
alpha = sparse_score / (sparse_score + dense_score)
final_score = alpha * norm_sparse + (1 - alpha) * norm_dense
```

Scores are min-max normalized per query per retriever before fusion.

### Supported LLM Backends

| Backend | Details |
|---------|---------|
| `proxy` | internal LLM proxy, GPT-5.2 |
| `bedrock` | AWS Bedrock, `qwen.qwen3-32b-v1:0` via boto3 |

Switch backend by setting `BACKEND` in `main()`.

### How to Run

```bash
uv run python experiment/dynamic-alpha-tuning/dynamic_alpha_tuning.py
```

### Configuration

Edit the constants in `main()`:

```python
BACKEND = "bedrock"          # "proxy" or "bedrock"
BASE_DATA_DIR = "..."        # path to data/input
OUTPUT_DIR = "..."           # path to data/output
DATASETS = [...]             # list of (dataset_name, split) tuples
SPARSE_RETRIEVERS = [...]    # e.g. ["bm25", "rm3"]
DENSE_RETRIEVERS = [...]     # e.g. ["biencoder", "qwen3"]
```

### Caching

LLM scores are cached per query under `{OUTPUT_DIR}/llm_cache/`. Failed queries are logged to
`*_failed.tsv`. Re-running retries only failed queries.

### Expected Outputs

```
{OUTPUT_DIR}/{dataset}/
├── {dataset}_{sparse}_vs_{dense}_{split}.trec   # ranked results in TREC format
├── metrics.csv                                   # nDCG, MRR, MAP @5 and @10 with 95% bootstrap CIs
└── llm_cache/
    ├── {dataset}_{model}_{combo}_{split}_cache.csv
    └── {dataset}_{model}_{combo}_{split}_failed.tsv
```

---

## Key Design Patterns

- **Config-driven (YAML):** All hyperparameters in `config.yaml`; `run.py` overrides per-run
  values (`experiment.name`, `data.train_file`, `data.test_file`) without modifying the file.
- **ExperimentTracker:** Auto-creates a timestamped directory under `experiments/` and logs
  config, metrics, and artifacts on every run.
- **Bootstrap confidence intervals:** 95% CIs reported for MAE, MSE, RMSE, R² in Ridge
  regression only. RoBERTa reports point estimates (MAE, MSE, RMSE, R², Pearson, Spearman,
  MAPE) without CIs.
- **Predictions clipped to [0, 1]:** Enforced in both Ridge `test.py` and RoBERTa `test.py`.
- **LLM result caching:** Per-query cache written after each LLM call; safe to interrupt and
  resume.
- **Metric/split auto-selection:** `run.py` uses `"mrr"` metric and `"dev"` split for MS MARCO
  and NQ; `"ndcg"` metric and `"test"` split for ACORD and NFCorpus.
