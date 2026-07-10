# Experiment query counts & average scores

Averages = sum of `predicted_{metric}@10` divided by the dataset's **total** query count (unique qrels ids); queries missing from `prediction-scores/` (zero-metric) count as 0.

## Query counts per dataset-combination

- **dataset queries (qrels)**: unique query-ids in the dataset's qrels for the eval split -- the true eval-set size, identical across retriever combinations.
- **used**: queries in `prediction-scores/` (after removing zero-metric queries), i.e. what the averages are over.
- **zero-removed**: dataset queries (qrels) - used.
- **mean-opt weight**: dataset-specific train-set mean optimal fusion weight (`avg_mean_best_weight` rounded to 0.01; the weight used by `02-mean-optimal-weight`).

| dataset | combination | metric | dataset queries (qrels) | used | zero-removed | mean-opt weight |
|---------|-------------|:--:|:--:|:--:|:--:|:--:|
| acord-entire-corpus | bm25_vs_biencoder | ndcg@10 | 57 | 52 | 5 | 0.49 |
| acord-entire-corpus | bm25_vs_qwen3 | ndcg@10 | 57 | 56 | 1 | 0.47 |
| acord-entire-corpus | rm3_vs_biencoder | ndcg@10 | 57 | 50 | 7 | 0.57 |
| acord-entire-corpus | rm3_vs_qwen3 | ndcg@10 | 57 | 53 | 4 | 0.48 |
| msmarco | bm25_vs_biencoder | mrr@10 | 6980 | 4874 | 2106 | 0.36 |
| msmarco | bm25_vs_qwen3 | mrr@10 | 6980 | 5149 | 1831 | 0.32 |
| msmarco | rm3_vs_biencoder | mrr@10 | 6980 | 4865 | 2115 | 0.35 |
| msmarco | rm3_vs_qwen3 | mrr@10 | 6980 | 5115 | 1865 | 0.30 |
| nfcorpus | bm25_vs_biencoder | ndcg@10 | 323 | 250 | 73 | 0.49 |
| nfcorpus | bm25_vs_qwen3 | ndcg@10 | 323 | 262 | 61 | 0.39 |
| nfcorpus | rm3_vs_biencoder | ndcg@10 | 323 | 248 | 75 | 0.51 |
| nfcorpus | rm3_vs_qwen3 | ndcg@10 | 323 | 264 | 59 | 0.41 |
| nq | bm25_vs_biencoder | mrr@10 | 2893 | 2108 | 785 | 0.40 |
| nq | bm25_vs_qwen3 | mrr@10 | 2893 | 2304 | 589 | 0.31 |
| nq | rm3_vs_biencoder | mrr@10 | 2893 | 2035 | 858 | 0.34 |
| nq | rm3_vs_qwen3 | mrr@10 | 2893 | 2272 | 621 | 0.27 |

The **p** column is the two-sided paired t-test p-value of per-query score vs `01-standard-rrf` over the full eval set (`< 0.001`, `< 0.05`, or the actual value when p ≥ 0.05). The baseline, Oracle and Headroom rows have no test.

## Average ndcg@10 — acord-entire-corpus

| experiment | **BM25(M)** | p | **BM25(Q)** | p | **RM3(M)** | p | **RM3(Q)** | p |
|---|---|---|---|---|---|---|---|---|
| Oracle (per-query best w) | 0.203 | — | 0.221 | — | 0.204 | — | 0.231 | — |
| Headroom (Oracle − RRF) | 0.071 | — | 0.083 | — | 0.078 | — | 0.088 | — |
| 01-standard-rrf | 0.132 | — | 0.138 | — | 0.126 | — | 0.143 | — |
| 02-mean-optimal-weight | 0.132 | 0.703 | 0.137 | 0.382 | 0.132 | 0.241 | 0.143 | 0.590 |
| 03-ridge-regression | 0.138 | 0.069 | 0.137 | 0.336 | 0.130 | 0.484 | 0.145 | 0.855 |
| 04-roberta-regression | 0.126 | 0.611 | 0.154 | 0.307 | 0.125 | 0.882 | 0.154 | 0.487 |
| 05-modern-bert-interval-weight | 0.125 | 0.559 | 0.156 | 0.259 | 0.125 | 0.887 | 0.155 | 0.439 |
| 06-modern-bert-passage-conditioned | 0.143 | 0.060 | 0.155 | < 0.05 | 0.135 | 0.369 | 0.159 | < 0.05 |
| 07-dat-qwen3 | — | — | — | — | — | — | — | — |
| 08-dat-ministral | — | — | — | — | — | — | — | — |
| 09-llm-fs-qwen3-interval-weight | 0.137 | 0.571 | 0.143 | 0.768 | 0.126 | 0.930 | 0.142 | 0.768 |
| 10-llm-fs-ministral-interval-weight | 0.139 | 0.419 | 0.151 | 0.085 | 0.132 | 0.623 | 0.157 | 0.141 |

## Average mrr@10 — msmarco

| experiment | **BM25(M)** | p | **BM25(Q)** | p | **RM3(M)** | p | **RM3(Q)** | p |
|---|---|---|---|---|---|---|---|---|
| Oracle (per-query best w) | 0.443 | — | 0.457 | — | 0.437 | — | 0.453 | — |
| Headroom (Oracle − RRF) | 0.157 | — | 0.166 | — | 0.163 | — | 0.175 | — |
| 01-standard-rrf | 0.286 | — | 0.292 | — | 0.274 | — | 0.278 | — |
| 02-mean-optimal-weight | 0.295 | < 0.001 | 0.314 | < 0.001 | 0.289 | < 0.001 | 0.309 | < 0.001 |
| 03-ridge-regression | 0.296 | < 0.001 | 0.317 | < 0.001 | 0.289 | < 0.001 | 0.310 | < 0.001 |
| 04-roberta-regression | 0.300 | < 0.001 | 0.321 | < 0.001 | 0.295 | < 0.001 | 0.312 | < 0.001 |
| 05-modern-bert-interval-weight | 0.305 | < 0.001 | 0.326 | < 0.001 | 0.297 | < 0.001 | 0.318 | < 0.001 |
| 06-modern-bert-passage-conditioned | 0.304 | < 0.001 | 0.327 | < 0.001 | 0.296 | < 0.001 | 0.325 | < 0.001 |
| 07-dat-qwen3 | — | — | — | — | — | — | — | — |
| 08-dat-ministral | — | — | — | — | — | — | — | — |
| 09-llm-fs-qwen3-interval-weight | 0.304 | < 0.001 | 0.335 | < 0.001 | 0.295 | < 0.001 | 0.328 | < 0.001 |
| 10-llm-fs-ministral-interval-weight | 0.303 | < 0.001 | 0.350 | < 0.001 | 0.299 | < 0.001 | 0.346 | < 0.001 |

## Average ndcg@10 — nfcorpus

| experiment | **BM25(M)** | p | **BM25(Q)** | p | **RM3(M)** | p | **RM3(Q)** | p |
|---|---|---|---|---|---|---|---|---|
| Oracle (per-query best w) | 0.405 | — | 0.455 | — | 0.423 | — | 0.473 | — |
| Headroom (Oracle − RRF) | 0.054 | — | 0.068 | — | 0.068 | — | 0.076 | — |
| 01-standard-rrf | 0.351 | — | 0.387 | — | 0.356 | — | 0.397 | — |
| 02-mean-optimal-weight | 0.348 | 0.060 | 0.397 | < 0.001 | 0.355 | 0.513 | 0.403 | < 0.05 |
| 03-ridge-regression | 0.349 | 0.147 | 0.397 | < 0.001 | 0.356 | 0.972 | 0.402 | 0.051 |
| 04-roberta-regression | 0.350 | 0.610 | 0.398 | < 0.001 | 0.356 | 0.927 | 0.402 | 0.088 |
| 05-modern-bert-interval-weight | 0.350 | 0.674 | 0.399 | < 0.001 | 0.360 | 0.142 | 0.405 | < 0.05 |
| 06-modern-bert-passage-conditioned | 0.349 | 0.309 | 0.407 | < 0.001 | 0.359 | 0.246 | 0.403 | < 0.05 |
| 07-dat-qwen3 | — | — | — | — | — | — | — | — |
| 08-dat-ministral | — | — | — | — | — | — | — | — |
| 09-llm-fs-qwen3-interval-weight | 0.351 | 0.991 | 0.403 | < 0.05 | 0.356 | 0.971 | 0.406 | 0.057 |
| 10-llm-fs-ministral-interval-weight | 0.349 | 0.751 | 0.404 | < 0.05 | 0.347 | 0.225 | 0.400 | 0.612 |

## Average mrr@10 — nq

| experiment | **BM25(M)** | p | **BM25(Q)** | p | **RM3(M)** | p | **RM3(Q)** | p |
|---|---|---|---|---|---|---|---|---|
| Oracle (per-query best w) | 0.525 | — | 0.549 | — | 0.484 | — | 0.531 | — |
| Headroom (Oracle − RRF) | 0.147 | — | 0.157 | — | 0.186 | — | 0.220 | — |
| 01-standard-rrf | 0.378 | — | 0.391 | — | 0.298 | — | 0.312 | — |
| 02-mean-optimal-weight | 0.381 | 0.238 | 0.417 | < 0.001 | 0.332 | < 0.001 | 0.378 | < 0.001 |
| 03-ridge-regression | 0.383 | 0.057 | 0.416 | < 0.001 | 0.330 | < 0.001 | 0.377 | < 0.001 |
| 04-roberta-regression | 0.387 | < 0.05 | 0.418 | < 0.001 | 0.333 | < 0.001 | 0.384 | < 0.001 |
| 05-modern-bert-interval-weight | 0.387 | < 0.001 | 0.423 | < 0.001 | 0.333 | < 0.001 | 0.393 | < 0.001 |
| 06-modern-bert-passage-conditioned | 0.406 | < 0.001 | 0.426 | < 0.001 | 0.341 | < 0.001 | 0.397 | < 0.001 |
| 07-dat-qwen3 | — | — | — | — | — | — | — | — |
| 08-dat-ministral | — | — | — | — | — | — | — | — |
| 09-llm-fs-qwen3-interval-weight | 0.391 | < 0.05 | 0.433 | < 0.001 | 0.347 | < 0.001 | 0.411 | < 0.001 |
| 10-llm-fs-ministral-interval-weight | 0.386 | 0.173 | 0.447 | < 0.001 | 0.371 | < 0.001 | 0.446 | < 0.001 |

## Headroom recovery (%)

(Method − RRF) / (Oracle − RRF) × 100. `01-standard-rrf` = 0% and the oracle = 100% by construction; values below 0 mean worse than standard RRF. Best method per column in **bold** (baseline/placeholder rows excluded). M = biencoder (MiniLM), Q = Qwen3.

### ndcg@10 — acord-entire-corpus

| method | **BM25(M)** | **BM25(Q)** | **RM3(M)** | **RM3(Q)** |
|---|---|---|---|---|
| 01-standard-rrf | 0.0 | 0.0 | 0.0 | 0.0 |
| 02-mean-optimal-weight | -1.1 | -1.2 | 7.1 | 0.4 |
| 03-ridge-regression | 8.2 | -1.5 | 4.8 | 2.2 |
| 04-roberta-regression | -8.9 | 19.1 | -1.4 | 12.5 |
| 05-modern-bert-interval-weight | -10.3 | **21.4** | -1.3 | 13.9 |
| 06-modern-bert-passage-conditioned | **15.5** | 20.0 | **11.9** | **18.4** |
| 07-dat-qwen3 | — | — | — | — |
| 08-dat-ministral | — | — | — | — |
| 09-llm-fs-qwen3-interval-weight | 7.2 | 5.7 | 0.0 | -0.5 |
| 10-llm-fs-ministral-interval-weight | 10.0 | 15.4 | 7.1 | 15.7 |

### mrr@10 — msmarco

| method | **BM25(M)** | **BM25(Q)** | **RM3(M)** | **RM3(Q)** |
|---|---|---|---|---|
| 01-standard-rrf | 0.0 | 0.0 | 0.0 | 0.0 |
| 02-mean-optimal-weight | 5.6 | 13.7 | 8.7 | 17.9 |
| 03-ridge-regression | 6.5 | 15.1 | 8.9 | 18.3 |
| 04-roberta-regression | 9.0 | 17.9 | 12.5 | 19.5 |
| 05-modern-bert-interval-weight | **12.0** | 20.9 | 13.8 | 22.9 |
| 06-modern-bert-passage-conditioned | 11.8 | 21.2 | 13.0 | 27.2 |
| 07-dat-qwen3 | — | — | — | — |
| 08-dat-ministral | — | — | — | — |
| 09-llm-fs-qwen3-interval-weight | 11.5 | 26.2 | 12.5 | 28.9 |
| 10-llm-fs-ministral-interval-weight | 10.9 | **35.4** | **15.1** | **39.0** |

### ndcg@10 — nfcorpus

| method | **BM25(M)** | **BM25(Q)** | **RM3(M)** | **RM3(Q)** |
|---|---|---|---|---|
| 01-standard-rrf | 0.0 | 0.0 | 0.0 | 0.0 |
| 02-mean-optimal-weight | -5.3 | 15.1 | -1.0 | 7.9 |
| 03-ridge-regression | -3.9 | 15.2 | 0.0 | 7.4 |
| 04-roberta-regression | -2.1 | 16.4 | 0.3 | 7.0 |
| 05-modern-bert-interval-weight | -2.3 | 17.7 | **6.4** | 10.6 |
| 06-modern-bert-passage-conditioned | -4.6 | **29.3** | 5.4 | 8.5 |
| 07-dat-qwen3 | — | — | — | — |
| 08-dat-ministral | — | — | — | — |
| 09-llm-fs-qwen3-interval-weight | **-0.1** | 23.7 | 0.3 | **12.8** |
| 10-llm-fs-ministral-interval-weight | -3.1 | 24.9 | -12.3 | 3.9 |

### mrr@10 — nq

| method | **BM25(M)** | **BM25(Q)** | **RM3(M)** | **RM3(Q)** |
|---|---|---|---|---|
| 01-standard-rrf | 0.0 | 0.0 | 0.0 | 0.0 |
| 02-mean-optimal-weight | 1.9 | 16.0 | 18.2 | 30.2 |
| 03-ridge-regression | 3.0 | 15.8 | 17.2 | 29.8 |
| 04-roberta-regression | 5.7 | 16.6 | 18.5 | 32.9 |
| 05-modern-bert-interval-weight | 6.1 | 20.3 | 18.6 | 37.1 |
| 06-modern-bert-passage-conditioned | **18.8** | 22.0 | 22.8 | 38.6 |
| 07-dat-qwen3 | — | — | — | — |
| 08-dat-ministral | — | — | — | — |
| 09-llm-fs-qwen3-interval-weight | 9.0 | 26.6 | 26.0 | 45.2 |
| 10-llm-fs-ministral-interval-weight | 5.1 | **35.0** | **39.1** | **61.1** |
