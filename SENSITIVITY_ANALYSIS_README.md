# Sensitivity Analysis: 2 Decimal Places vs 3 Decimal Places

## Overview 

This sensitivity analysis measures the effect of **finer grid search granularity** on RRF (Reciprocal Rank Fusion) weight optimization. We compare two discretization levels:
- **2dp**: Weight steps of 0.01 (101 possible values: 0.00, 0.01, ..., 1.00)
- **3dp**: Weight steps of 0.001 (1001 possible values: 0.000, 0.001, ..., 1.000)



## Metric Definitions

**Interval Length Difference**: Per-query difference in total optimal weight interval length (`length_3dp - length_2dp`). 

**Interval Disagreement**: Per-query non-overlapping regions (`Union - Intersection`). Values near zero indicate high agreement.

**NDCG@10 Difference**: Per-query NDCG score difference (`NDCG_3dp - NDCG_2dp`) using mean best weights. Values near zero indicate negligible impact.

All metrics report mean ± standard deviation across queries, aggregated across train/dev/test splits.

---

## Results: All Datasets

Results are aggregated across **train, dev, and test** splits for each dataset.


### acord-entire-corpus

| Search Method | Interval Length Diff (Mean ± Std) | Interval Disagreement (Mean ± Std) | NDCG@10 Diff (Mean ± Std) |
|---------------|-----------------------------------|------------------------------------|---------------------------------|
| **bm25 vs biencoder** | 0.0037 ± 0.0043 | 0.0109 ± 0.0048 | 0.000503 ± 0.000008 |
| **bm25 vs qwen3** | 0.0068 ± 0.0017 | 0.0094 ± 0.0008 | 0.000280 ± 0.000002 |
| **rm3 vs biencoder** | 0.0025 ± 0.0033 | 0.0100 ± 0.0039 | 0.000699 ± 0.000028 |
| **rm3 vs qwen3** | 0.0050 ± 0.0022 | 0.0087 ± 0.0005 | 0.000414 ± 0.000005 |

### nfcorpus

| Search Method | Interval Length Diff (Mean ± Std) | Interval Disagreement (Mean ± Std) | NDCG@10 Diff (Mean ± Std) |
|---------------|-----------------------------------|------------------------------------|---------------------------------|
| **bm25 vs biencoder** | 0.0035 ± 0.0000 | 0.0068 ± 0.0000 | 0.000248 ± 0.000007 |
| **bm25 vs qwen3** | 0.0040 ± 0.0000 | 0.0061 ± 0.0000 | 0.000227 ± 0.000006 |
| **rm3 vs biencoder** | 0.0037 ± 0.0000 | 0.0070 ± 0.0000 | 0.000293 ± 0.000010 |
| **rm3 vs qwen3** | 0.0037 ± 0.0000 | 0.0069 ± 0.0000 | 0.000329 ± 0.000013 |

### nq 

| Search Method | Interval Length Diff (Mean ± Std) | Interval Disagreement (Mean ± Std) | NDCG@10 Diff (Mean ± Std) |
|---------------|-----------------------------------|------------------------------------|---------------------------------|
| **bm25 vs biencoder** | 0.0033 ± 0.0001 | 0.0043 ± 0.0003 | 0.000358 ± 0.000079 |
| **bm25 vs qwen3** | 0.0035 ± 0.0002 | 0.0046 ± 0.0000 | 0.000365 ± 0.000096 |
| **rm3 vs biencoder** | 0.0035 ± 0.0000 | 0.0047 ± 0.0000 | 0.000413 ± 0.000098 |
| **rm3 vs qwen3** | 0.0039 ± 0.0000 | 0.0050 ± 0.0000 | 0.000387 ± 0.000098 |

### msmarco

| Search Method | Interval Length Diff (Mean ± Std) | Interval Disagreement (Mean ± Std) | NDCG@10 Diff (Mean ± Std) |
|---------------|-----------------------------------|------------------------------------|---------------------------------|
| **bm25 vs biencoder** | 0.0040 ± 0.0004 | 0.0071 ± 0.0027 | 0.000471 ± 0.007253 |
| **bm25 vs qwen3** | 0.0044 ± 0.0009 | 0.0061 ± 0.0014 | 0.000336 ± 0.006467 |
| **rm3 vs biencoder** | 0.0043 ± 0.0008 | 0.0061 ± 0.0011 | 0.000447 ± 0.006909 |
| **rm3 vs qwen3** | 0.0046 ± 0.0011 | 0.0059 ± 0.0012 | 0.000419 ± 0.006362 |

