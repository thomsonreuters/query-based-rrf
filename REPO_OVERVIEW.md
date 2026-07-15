# Repository Overview: query-based-rrf

For project motivation, the latency/quality decision framework (tiers T0–T3b), and top-level
usage instructions, see [`README.md`](./README.md) — that is the canonical entry point. This
document covers implementation details that aren't in the README: the full directory layout,
the data-prep pipeline, per-experiment config/architecture specifics, output file formats, and
known inconsistencies worth cleaning up.

---

## Repository Structure

```
query-based-rrf/
├── README.md
├── REPO_OVERVIEW.md
├── .env.local                        # template — copy to .env and fill in your paths
├── format_metrics_table.py
├── get_mean_best_weight.py
├── helper_3_add_text2csv.py
├── helper_4_fuse_results_wrrf.py
├── helper_5_ir_metrics.py
├── helper_80_split_search_results.py
├── helper_81_get_top_k_results.py
├── helper_90_friendly_intervals.py
├── helper_91_mean_best_weight.py
├── helper_92_plot_all_dataset_weight_stats.py
├── helper_93_interval_distribution.py
├── rrf_mrr_best_weights_all_weights_final.py
├── rrf_ndcg_best_weights_all_weights_final.py
├── utils/
│   ├── env.py                        # .env.local loader (not used by most scripts — see Known Inconsistencies)
│   ├── aggregate_latency.py
│   ├── aggregate_metrics.py
│   ├── combine_results.py
│   ├── compute_latency_ci.py
│   ├── compute_latency_ci_per_tier.py
│   ├── model_mapping.json
│   ├── analyze_ir_latency_tradeoff.py  # documented in README
│   └── plot_tradeoff.py                # documented in README
├── difference-analysis/
│   └── roberta/                      # PCA graphs + per-dataset diff CSVs (RM3 vs Qwen3)
└── experiment/
    ├── ridge-regression/
    │   └── ridge-regression-mean-best-weight/       (config.yaml, run.py, train.py, test.py)
    ├── roberta-regression/
    │   └── roberta-experiment-mean-best-weight/     (config.yaml, run.py, run_mul.py, train.py, test.py, README.md)
    ├── roberta-interval-weight/                     (config.yaml, run_mul.py, train.py, test.py)
    ├── modern-bert-regression/                      (config.yaml, run.py, run_mul.py, train.py, test.py)
    ├── modern-bert-interval-weight/                 (config.yaml, run_mul.py, train.py, test.py)
    ├── modern-bert-passage-conditioned/              (config.yaml, run_mul.py, train.py, test.py)
    ├── llm-fs-ministral-mean-best-weight/            (llm_predict_ministral3.py, llm_predict_ministral3_mean_best_weight_infer.py)
    ├── llm-fs-ministral-interval-weight/              (llm_predict_ministral3.py, llm_predict_ministral3_interval_weight_infer.py)
    ├── llm-fs-qwen3-mean-best-weight/                (llm_predict_qwen3.py, llm_predict_qwen3_mean_best_weight_infer.py)
    ├── llm-fs-qwen3-interval-weight/                 (llm_predict_qwen3.py, llm_predict_qwen3_interval_weight_infer.py)
    ├── dynamic-alpha-tuning/
    │   ├── dynamic_alpha_tuning.py     # current entry point (batched, cached, env-driven paths)
    │   ├── dat-infer.py                # single-dataset latency probe (hardcoded paths)
    │   ├── dat-infer-gpt.py            # gpt-5.2 latency probe via internal AI Platform backend
    │   ├── postprocess_metrics.py      # formats metrics.csv → "mean [lo, hi]" strings
    │   └── test_bedrock.py             # Bedrock connectivity smoke test
    ├── llm_backend.py                  # shared LLM backend classes (Bedrock / local HF)
    ├── test_llm_backend.py
    ├── experiment-results.md
    ├── mean_best_weight_prompt.json
    └── interval_weight_prompt.json
```

---

## Datasets

| Dataset                       | Metric  | Eval split |
|-------------------------------|---------|------------|
| MS MARCO                      | MRR@10  | `dev`      |
| NQ                            | MRR@10  | `dev`      |
| ACORD (`acord-entire-corpus`) | nDCG@10 | `test`     |
| NFCorpus                      | nDCG@10 | `test`     |

Retriever combos (all experiments): `bm25_vs_biencoder`, `bm25_vs_qwen3`, `rm3_vs_biencoder`,
`rm3_vs_qwen3` → 16 dataset×combo runs per model.

> TREC-COVID is **not** used anywhere in the current codebase (no experiment script references
> it outside a stale comment) — it was dropped from a previous iteration of this repo.

---

## Data Preparation Pipeline

Run once before any experiment. None of these five scripts take CLI args — each has constants
(dataset name, retriever, split, top-k) hardcoded near the top or bottom of the file that you
edit and rerun per dataset/combo.

1. **Split by qrels** — `helper_80_split_search_results.py`: splits one full `.trec` run into
   train/dev/test files by query-id membership against `qrels/{split}.tsv`.
2. **Truncate to top-K** — `helper_81_get_top_k_results.py`: truncates each query's result list
   to the top 200 (default) from a larger top-K run (e.g. top1000).
3. **Attach query text** — `helper_3_add_text2csv.py`: joins `query_text` from `queries.jsonl`
   into the `best_weights_final_mean.csv` files, producing `*_with_text.csv`.
4. **Bucket into intervals** — `helper_90_friendly_intervals.py`: groups a query's raw
   `best_weights` list into interval ranges (`friendly_best_weights` + `interval_count`).
5. **Collapse to a scalar label** — `helper_91_mean_best_weight.py` (or the root-level
   `get_mean_best_weight.py`, which does the same interval-collapsing logic in one pass):
   produces the `mean_best_weight` regression target used by Ridge/RoBERTa/ModernBERT.

Steps 4–5 read `BASE_DATA_DIR` (env var, falls back to a hardcoded cluster path) but — unlike
steps 1–3 — are currently hardcoded to a single dataset/combo/split (`nq` /
`bm25_vs_biencoder` / `train`) rather than looping over all 16; re-point the hardcoded path at
the top of each file per dataset/combo you need to process.

> **Known bug:** `helper_90_friendly_intervals.py`'s output path is computed via
> `input_csv.replace(".csv", "_friendly_intervals.csv")`, but the hardcoded input path in the
> script has no `.csv` in it — so the replace is a no-op and the script silently overwrites its
> own input file instead of writing a new one. Worth fixing before relying on this step.

### Downstream analysis & visualization

- `helper_92_plot_all_dataset_weight_stats.py` — classifies each query's weight pattern
  (empty / single point / one interval / multiple intervals) across all 4 datasets × 4 combos
  and plots two 4×4 grids.
- `helper_93_interval_distribution.py` — a stacked-bar chart of weight-pattern categories;
  note the percentages are hardcoded constants in the script, not computed from a CSV — update
  them by hand if the underlying data changes.
- `format_metrics_table.py` — formats a raw IR-metrics CSV (mean + bootstrap bounds) into
  `"mean [lower, upper]"` cells, writing both `.csv` and `.md` tables.

(Fusing weighted results and computing IR metrics — `helper_4_fuse_results_wrrf.py` /
`helper_5_ir_metrics.py` — and collecting ground-truth weights —
`rrf_mrr_best_weights_all_weights_final.py` / `rrf_ndcg_best_weights_all_weights_final.py` —
are already documented in the README's "Data Collection" and "Fusion Strategy Evaluation"
sections.)

---

## Experiment Implementation Details

### T1b — Ridge Regression

**Path:** `experiment/ridge-regression/ridge-regression-mean-best-weight/`

TF-IDF (`ngram_range=[1,4]`, `min_df=2`, `max_df=0.95`, `sublinear_tf=true`, English stopwords)
→ `Ridge(alpha=100)`. `run.py` loops all 16 dataset/combo pairs, resolving
`BASE_DATA_DIR` from the environment (hardcoded fallback if unset).

- **Training** writes `experiments/{name}_{timestamp}/`: `config.yaml`, `script_train.py`
  (copy of `train.py` for reproducibility), `tfidf_vectorizer.pkl`, `ridge_regression_model.pkl`,
  `train_predictions.csv`, `val_predictions.csv`, `all_features_coefficients.csv`,
  `top_30_coefficients.png`, `coefficient_distribution.png`, `results.json`.
- **Testing** runs true per-query (batch-size-1) timed inference (5 warmup iterations,
  `time.perf_counter()`), clips predictions to `[0.00, 1.0]`, appends results into that
  experiment's `results.json`, and writes `predictions/{dataset}_{combo}_{split}.csv` with
  columns `query_id, query_text, actual, predicted, latency_ms, absolute_error`.

### T2a — RoBERTa (query-only)

Both variants share a `RobertaModel.from_pretrained(...) → pooler_output → Dropout → Linear(hidden_size, 1)`
architecture, `roberta-large` by default, `max_length=64`, `lr=2e-5`, 10 epochs, batch size 256,
`fp16=True`, early stopping (patience 10).

- **`experiment/roberta-regression/roberta-experiment-mean-best-weight/`** — plain
  `nn.MSELoss()` against the scalar `mean_best_weight` label. Has both `run.py` (sequential,
  one GPU) and `run_mul.py` (multi-GPU: `multiprocessing.Pool` + a `Manager().Queue()` of GPU
  ids, each worker sets `CUDA_VISIBLE_DEVICES` from the queue — edit `AVAILABLE_GPUS` at the
  bottom of the file for your hardware).
- **`experiment/roberta-interval-weight/`** — same architecture, but trained against the
  `friendly_best_weights` interval label (parsed into `[left, right]`, widest interval picked if
  multiple) with a custom **interval-aware satisficing loss**:
  `relu(left - pred)² + relu(pred - right)²` (zero loss inside the interval, squared-hinge
  penalty outside it). Metrics use the interval midpoint as the point-estimate proxy for
  R²/Pearson/Spearman. Multi-GPU only (`run_mul.py`, no sequential `run.py`).

Both write `experiments/{name}_{timestamp}/{checkpoints/, best_model/, logs/, config.yaml,
results.json, validation_predictions.csv}` during training, and
`predictions/{experiment_name}_{split}.csv` during testing (`_vs` stripped from the experiment
name; split inferred from the test file path).

### T2a/T2b — ModernBERT (query-only and passage-conditioned)

Three variants, all built on `answerdotai/ModernBERT-large`, `max_length=64` except where noted:

- **`experiment/modern-bert-regression/`** — uses HuggingFace's native
  `AutoModelForSequenceClassification(..., num_labels=1, problem_type="regression")` head (MSE
  loss handled internally), `bf16=True`. Has both `run.py` and `run_mul.py`.
- **`experiment/modern-bert-interval-weight/`** — a custom `ModernBertRegression` module
  wrapping `AutoModel` + manual `Linear(hidden_size, 1)` head (CLS-token pooling, since
  ModernBERT doesn't always expose a pooler), same interval-aware satisficing loss as
  `roberta-interval-weight`, `fp16=True`. Multi-GPU only.
- **`experiment/modern-bert-passage-conditioned/`** — same architecture/loss as
  `modern-bert-interval-weight`, but the model input is
  `f"{query_text} {sep} {sparse_top1_passage_text} {sep} {dense_top1_passage_text}"` — the top-1
  document per retriever (read from the `.trec` files) with its text loaded from
  `corpus.jsonl`. Requires extra config keys (`data.corpus_path`,
  `data.{sparse,dense}_trec_{train,test}`), `max_length=1024` to fit query + 2 passages, and a
  reduced `batch_size=8` to avoid OOM. Multi-GPU only.

All three `run_mul.py` scripts share the same GPU-queue multiprocessing pattern as RoBERTa's;
only the hardcoded `AVAILABLE_GPUS` list (and, for passage-conditioned, the active `datasets`
list) differs per file.

### T3b — LLM Few-Shot (Ministral / Qwen3)

**Paths:** `experiment/llm-fs-{ministral,qwen3}-{mean-best-weight,interval-weight}/`

Shared backend (`experiment/llm_backend.py`): `BedrockBackend` (AWS Bedrock, e.g.
`mistral.ministral-3-14b-instruct` / `qwen.qwen3-32b-v1:0`), `LocalQwen3Backend` (local HF
`Qwen/Qwen3-32B`, bf16), `LocalMistralBackend` (local HF Ministral-3-14B, bf16). Each script's
`BACKEND` constant selects among `"bedrock"` / `"local_qwen3"` / `"local_mistral"` — the main
batch-scoring scripts default to local backends, the `*_infer.py` latency-benchmark variants
default to `"bedrock"`.

Few-shot exemplars are retrieved, not random: BM25 top-5 (`bm25s`) plus dense top-5
(`SentenceTransformer('Qwen/Qwen3-8B')` cosine similarity) over training-set queries, merged and
deduped by `query_id` (up to 10 context examples). Retrieval here is query-similarity retrieval
for exemplar selection — only query text goes into the prompt, not retrieved passage content.
Prompt templates live in `mean_best_weight_prompt.json` / `interval_weight_prompt.json`
(`interval-weight` variants swap the target label to `friendly_best_weights`). Output is parsed
with a regex, falling back to 0.50 if unparseable.

- `llm_predict_{ministral3,qwen3}.py` — full batched scoring pass over all 16 dataset/combo
  pairs (`BATCH_SIZE=8`, no latency capture). Writes
  `predictions/{DATASET}_{SPLIT}_{COMBINATION}_predictions.csv`.
- `llm_predict_{ministral3,qwen3}_{mean_best_weight,interval_weight}_infer.py` — a separate,
  unbatched, single-query latency-benchmarking pass (captures `latency_ms`, reports
  mean/median/p95) over the same data — not a train/inference split of the pipeline, just a
  timing-focused variant of the same scoring logic.
- No result caching in this tier (unlike Dynamic Alpha Tuning below) — each run scores the full
  dataset/combo from scratch and writes one CSV at the end.

### T3a — Dynamic Alpha Tuning

**Path:** `experiment/dynamic-alpha-tuning/`

Per query, an LLM scores the top-1 result from each retriever (sparse and dense) on a 0–5 scale;
`compute_alpha(sparse_score, dense_score)` derives α ∈ [0, 1] (α=1 if sparse=5, α=0 if dense=5,
else `sparse / (sparse + dense)`, 0.5 fallback). Final score = `α·norm_sparse + (1-α)·norm_dense`,
written out as a re-ranked `.trec` file.

- **`dynamic_alpha_tuning.py`** — current entry point: batched/concurrent
  (`batch_size=60`, `concurrency=20`, `asyncio.Semaphore`), disk-backed cache
  (`llm_cache/{dataset}_{model}_{combo}_{split}_cache.csv` + a failed-query log so interrupted
  runs resume), env-driven paths (`BASE_DATA_DIR`/`BASE_RESULTS_DIR`), default backend
  `local_qwen3`. Covers all 4 datasets × 4 combos. Output:
  `{OUTPUT_DIR}/{dataset}/{dataset}_{sparse}_vs_{dense}_{split}.trec` +
  `metrics.csv` (via `helper_5_ir_metrics.process_single_dataset`).
- **`dat-infer.py`** — a companion single-dataset latency probe: same scoring logic, but
  sequential (no batching/caching) with per-query `time.perf_counter()` timing. Paths are
  hardcoded rather than env-driven; only `("msmarco", "dev")` is active.
- **`dat-infer-gpt.py`** — the same latency-probe structure, but scores via an internal
  AI-Platform-authenticated OpenAI endpoint (`gpt-5.2`) instead of `llm_backend.py` — a
  separate, self-contained `LLMClient` implementation. Only `msmarco` is currently uncommented,
  though on-disk output evidence shows it was previously run as a full 4-dataset sweep.
- **`postprocess_metrics.py`** — CLI: `python postprocess_metrics.py <metrics.csv>`. Formats
  bootstrap-CI metric columns into `"mean [lower, upper]"` strings, writing
  `metrics_processed.csv` alongside the input.
- **`test_bedrock.py`** — a Bedrock connectivity smoke test (sends one trivial prompt) to
  validate credentials before relying on the Bedrock backend path.

Backends available across this tier: AWS Bedrock (`qwen.qwen3-32b-v1:0`), local HF Qwen3-32B
(default), local HF Ministral-3-14B (available, not default), and the internal AI-Platform
GPT-5.2 endpoint (only in `dat-infer-gpt.py`).

---

## `utils/` Reference

| Script | Purpose |
|---|---|
| `env.py` | Walks up from `utils/` to find and load `.env.local` via `python-dotenv` (`override=False`) |
| `aggregate_latency.py` | Averages avg/total latency (ms) across the 4 retriever combos per (dataset, model); adds zero-cost rows for `rrf`/`mow` |
| `aggregate_metrics.py` | Averages the NDCG@10/MRR@10 metric column (parsed from `"mean [lo, hi]"` strings) across the 4 combos per (dataset, method) |
| `combine_results.py` | Orchestrates the two scripts above, applies `model_mapping.json` alias normalization, inner-joins timing + metrics on (dataset, model) |
| `compute_latency_ci.py` | Student's-t 95% CIs on per-query latency, pooled at the fine-grained sub-tier level (T1b, T2a, T2b, T3a, T3b) |
| `compute_latency_ci_per_tier.py` | Same, pooled at the coarser tier level (T1, T2, T3) |
| `model_mapping.json` | Flat `{alias: canonical_name}` dict normalizing inconsistent raw model-name strings across result sheets |
| `analyze_ir_latency_tradeoff.py` / `plot_tradeoff.py` | Documented in README's "IR Performance–Latency Tradeoff Analysis" section |

---

## Known Inconsistencies / Tech Debt

- **`helper_90_friendly_intervals.py`** silently overwrites its own input file instead of
  writing `*_friendly_intervals.csv` — see note under Data Preparation Pipeline above.
- **`utils/env.py` is effectively unused.** Most scripts resolve `BASE_DATA_DIR` /
  `BASE_RESULTS_DIR` / `BASE_EXPERIMENT_DIR` via their own
  `os.environ.get("VAR", "<hardcoded fallback>")` call rather than importing `utils/env.py`, so
  the `.env.local` loading mechanism the README describes isn't actually wired into most entry
  points yet.
- **Hardcoded fallback paths are inconsistent across the codebase.** Most scripts fall back to
  a shared cluster path; all four `experiment/llm-fs-*/*_infer.py` latency-benchmark scripts
  (both mean-best-weight and interval-weight variants, ministral and qwen3) and
  `experiment/dynamic-alpha-tuning/dat-infer{,-gpt}.py` still fall back to (or are entirely
  hardcoded to) an older SageMaker-specific path from a previous environment. The non-`_infer`
  `llm_predict_*.py` scripts are a separate inconsistency: three of the four use a hardcoded
  relative `../../dataset/...` path with no env var at all, while only
  `llm-fs-qwen3-interval-weight/llm_predict_qwen3.py` reads `BASE_DATA_DIR`. Worth
  standardizing all of these on `utils/env.py` + `.env.local`.
- **TREC-COVID** is not used by any current script — drop any remaining references to it in
  other docs/notes rather than carrying it forward as an active dataset.
