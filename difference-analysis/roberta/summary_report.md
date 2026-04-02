# Metric Differences Summary Report

This report summarizes the metric differences (Predicted Metric - Baseline 0.5) for processed CSV files.

| File Name | Metric | Total Queries | > 0 Count (%) | < 0 Count (%) | == 0 Count (%) |
|-----------|--------|---------------|---------------|---------------|----------------|
| msmarco_rm3_qwen3_dev.csv | MRR | 5115 | 1910 (37.34%) | 595 (11.63%) | 2610 (51.03%) |
| nfcorpus_rm3_qwen3_test.csv | NDCG | 323 | 96 (29.72%) | 62 (19.20%) | 165 (51.08%) |
| acord-entire-corpus_rm3_qwen3_test.csv | NDCG | 57 | 33 (57.89%) | 18 (31.58%) | 6 (10.53%) |
| nq_rm3_qwen3_dev.csv | MRR | 2272 | 961 (42.30%) | 208 (9.15%) | 1103 (48.55%) |
