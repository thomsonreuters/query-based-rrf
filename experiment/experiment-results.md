# Experiment query counts & average scores

Averages = sum of `predicted_{metric}@10` divided by the dataset's **total** query count (unique qrels ids); queries missing from `prediction-scores/` (not retrieved, or zero-metric) count as 0.

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

## Average ndcg@10 — acord-entire-corpus

| experiment | **BM25(M)** | **BM25(Q)** | **RM3(M)** | **RM3(Q)** |
|---|---|---|---|---|
| Oracle (per-query best w) | 0.203 | 0.221 | 0.204 | 0.231 |
| Headroom (Oracle − RRF) | 0.071 | 0.083 | 0.078 | 0.088 |
| 01-standard-rrf | 0.132 | 0.138 | 0.126 | 0.143 |
| 02-mean-optimal-weight | 0.132 | 0.137 | 0.132 | 0.143 |
| 03-ridge-regression | 0.138 | 0.137 | 0.130 | 0.145 |
| 04-roberta-regression | 0.126 | 0.154 | 0.125 | 0.154 |
| 05-modern-bert-interval-weight | 0.125 | 0.156 | 0.125 | 0.155 |
| 06-modern-bert-passage-conditioned | 0.143 | 0.155 | 0.135 | 0.159 |
| 07-dat-qwen3 | — | — | — | — |
| 08-dat-ministral | — | — | — | — |
| 09-llm-fs-qwen3-interval-weight | 0.137 | 0.143 | 0.126 | 0.142 |
| 10-llm-fs-ministral-interval-weight | 0.139 | 0.151 | 0.132 | 0.157 |

## Average mrr@10 — msmarco

| experiment | **BM25(M)** | **BM25(Q)** | **RM3(M)** | **RM3(Q)** |
|---|---|---|---|---|
| Oracle (per-query best w) | 0.443 | 0.457 | 0.437 | 0.453 |
| Headroom (Oracle − RRF) | 0.157 | 0.166 | 0.163 | 0.175 |
| 01-standard-rrf | 0.286 | 0.292 | 0.274 | 0.278 |
| 02-mean-optimal-weight | 0.295 | 0.314 | 0.289 | 0.309 |
| 03-ridge-regression | 0.296 | 0.317 | 0.289 | 0.310 |
| 04-roberta-regression | 0.300 | 0.321 | 0.295 | 0.312 |
| 05-modern-bert-interval-weight | 0.305 | 0.326 | 0.297 | 0.318 |
| 06-modern-bert-passage-conditioned | 0.304 | 0.327 | 0.296 | 0.325 |
| 07-dat-qwen3 | — | — | — | — |
| 08-dat-ministral | — | — | — | — |
| 09-llm-fs-qwen3-interval-weight | 0.304 | 0.335 | 0.295 | 0.328 |
| 10-llm-fs-ministral-interval-weight | 0.303 | 0.350 | 0.299 | 0.346 |

## Average ndcg@10 — nfcorpus

| experiment | **BM25(M)** | **BM25(Q)** | **RM3(M)** | **RM3(Q)** |
|---|---|---|---|---|
| Oracle (per-query best w) | 0.405 | 0.455 | 0.423 | 0.473 |
| Headroom (Oracle − RRF) | 0.054 | 0.068 | 0.068 | 0.076 |
| 01-standard-rrf | 0.351 | 0.387 | 0.356 | 0.397 |
| 02-mean-optimal-weight | 0.348 | 0.397 | 0.355 | 0.403 |
| 03-ridge-regression | 0.349 | 0.397 | 0.356 | 0.402 |
| 04-roberta-regression | 0.350 | 0.398 | 0.356 | 0.402 |
| 05-modern-bert-interval-weight | 0.350 | 0.399 | 0.360 | 0.405 |
| 06-modern-bert-passage-conditioned | 0.349 | 0.407 | 0.359 | 0.403 |
| 07-dat-qwen3 | — | — | — | — |
| 08-dat-ministral | — | — | — | — |
| 09-llm-fs-qwen3-interval-weight | 0.351 | 0.403 | 0.356 | 0.406 |
| 10-llm-fs-ministral-interval-weight | 0.349 | 0.404 | 0.347 | 0.400 |

## Average mrr@10 — nq

| experiment | **BM25(M)** | **BM25(Q)** | **RM3(M)** | **RM3(Q)** |
|---|---|---|---|---|
| Oracle (per-query best w) | 0.525 | 0.549 | 0.484 | 0.531 |
| Headroom (Oracle − RRF) | 0.147 | 0.157 | 0.186 | 0.220 |
| 01-standard-rrf | 0.378 | 0.391 | 0.298 | 0.312 |
| 02-mean-optimal-weight | 0.381 | 0.417 | 0.332 | 0.378 |
| 03-ridge-regression | 0.383 | 0.416 | 0.330 | 0.377 |
| 04-roberta-regression | 0.387 | 0.418 | 0.333 | 0.384 |
| 05-modern-bert-interval-weight | 0.387 | 0.423 | 0.333 | 0.393 |
| 06-modern-bert-passage-conditioned | 0.406 | 0.426 | 0.341 | 0.397 |
| 07-dat-qwen3 | — | — | — | — |
| 08-dat-ministral | — | — | — | — |
| 09-llm-fs-qwen3-interval-weight | 0.391 | 0.433 | 0.347 | 0.411 |
| 10-llm-fs-ministral-interval-weight | 0.386 | 0.447 | 0.371 | 0.446 |

## Significance — paired t-test (p-values)

Per-query significance of each method vs two baselines — `01-standard-rrf` (RRF) and `02-mean-optimal-weight` (MOW) — over the full eval set (0-fill for missing queries). Shown as `< 0.001`, else the actual value; `—` = self-comparison / no data.

### ndcg@10 — acord-entire-corpus

<table>
<thead>
<tr><th rowspan='2'>experiment</th><th colspan='2'>BM25(M)</th><th colspan='2'>BM25(Q)</th><th colspan='2'>RM3(M)</th><th colspan='2'>RM3(Q)</th></tr>
<tr><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th></tr>
</thead>
<tbody>
<tr><td>01-standard-rrf</td><td>—</td><td>0.703</td><td>—</td><td>0.382</td><td>—</td><td>0.241</td><td>—</td><td>0.590</td></tr>
<tr><td>02-mean-optimal-weight</td><td>0.703</td><td>—</td><td>0.382</td><td>—</td><td>0.241</td><td>—</td><td>0.590</td><td>—</td></tr>
<tr><td>03-ridge-regression</td><td>0.069</td><td>0.057</td><td>0.336</td><td>0.288</td><td>0.484</td><td>0.402</td><td>0.855</td><td>0.233</td></tr>
<tr><td>04-roberta-regression</td><td>0.611</td><td>0.644</td><td>0.307</td><td>0.213</td><td>0.882</td><td>0.586</td><td>0.487</td><td>0.412</td></tr>
<tr><td>05-modern-bert-interval-weight</td><td>0.559</td><td>0.589</td><td>0.259</td><td>0.177</td><td>0.887</td><td>0.594</td><td>0.439</td><td>0.368</td></tr>
<tr><td>06-modern-bert-passage-conditioned</td><td>0.060</td><td>0.057</td><td>0.030</td><td>0.013</td><td>0.369</td><td>0.781</td><td>0.020</td><td>0.011</td></tr>
<tr><td>07-dat-qwen3</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>08-dat-ministral</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>09-llm-fs-qwen3-interval-weight</td><td>0.571</td><td>0.510</td><td>0.768</td><td>0.580</td><td>0.930</td><td>0.462</td><td>0.768</td><td>0.880</td></tr>
<tr><td>10-llm-fs-ministral-interval-weight</td><td>0.419</td><td>0.361</td><td>0.085</td><td>0.039</td><td>0.623</td><td>0.920</td><td>0.141</td><td>0.068</td></tr>
</tbody>
</table>

### mrr@10 — msmarco

<table>
<thead>
<tr><th rowspan='2'>experiment</th><th colspan='2'>BM25(M)</th><th colspan='2'>BM25(Q)</th><th colspan='2'>RM3(M)</th><th colspan='2'>RM3(Q)</th></tr>
<tr><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th></tr>
</thead>
<tbody>
<tr><td>01-standard-rrf</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td></tr>
<tr><td>02-mean-optimal-weight</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td><td>—</td></tr>
<tr><td>03-ridge-regression</td><td>&lt; 0.001</td><td>0.012</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>0.742</td><td>&lt; 0.001</td><td>0.058</td></tr>
<tr><td>04-roberta-regression</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
<tr><td>05-modern-bert-interval-weight</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
<tr><td>06-modern-bert-passage-conditioned</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
<tr><td>07-dat-qwen3</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>08-dat-ministral</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>09-llm-fs-qwen3-interval-weight</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>0.017</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
<tr><td>10-llm-fs-ministral-interval-weight</td><td>&lt; 0.001</td><td>0.007</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
</tbody>
</table>

### ndcg@10 — nfcorpus

<table>
<thead>
<tr><th rowspan='2'>experiment</th><th colspan='2'>BM25(M)</th><th colspan='2'>BM25(Q)</th><th colspan='2'>RM3(M)</th><th colspan='2'>RM3(Q)</th></tr>
<tr><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th></tr>
</thead>
<tbody>
<tr><td>01-standard-rrf</td><td>—</td><td>0.060</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>0.513</td><td>—</td><td>0.037</td></tr>
<tr><td>02-mean-optimal-weight</td><td>0.060</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>0.513</td><td>—</td><td>0.037</td><td>—</td></tr>
<tr><td>03-ridge-regression</td><td>0.147</td><td>0.183</td><td>&lt; 0.001</td><td>0.483</td><td>0.972</td><td>0.274</td><td>0.051</td><td>0.261</td></tr>
<tr><td>04-roberta-regression</td><td>0.610</td><td>0.266</td><td>&lt; 0.001</td><td>0.318</td><td>0.927</td><td>0.705</td><td>0.088</td><td>0.548</td></tr>
<tr><td>05-modern-bert-interval-weight</td><td>0.674</td><td>0.558</td><td>&lt; 0.001</td><td>0.443</td><td>0.142</td><td>0.091</td><td>0.015</td><td>0.432</td></tr>
<tr><td>06-modern-bert-passage-conditioned</td><td>0.309</td><td>0.884</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>0.246</td><td>0.173</td><td>0.029</td><td>0.845</td></tr>
<tr><td>07-dat-qwen3</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>08-dat-ministral</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>09-llm-fs-qwen3-interval-weight</td><td>0.991</td><td>0.577</td><td>0.004</td><td>0.227</td><td>0.971</td><td>0.888</td><td>0.057</td><td>0.411</td></tr>
<tr><td>10-llm-fs-ministral-interval-weight</td><td>0.751</td><td>0.821</td><td>0.002</td><td>0.154</td><td>0.225</td><td>0.270</td><td>0.612</td><td>0.571</td></tr>
</tbody>
</table>

### mrr@10 — nq

<table>
<thead>
<tr><th rowspan='2'>experiment</th><th colspan='2'>BM25(M)</th><th colspan='2'>BM25(Q)</th><th colspan='2'>RM3(M)</th><th colspan='2'>RM3(Q)</th></tr>
<tr><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th></tr>
</thead>
<tbody>
<tr><td>01-standard-rrf</td><td>—</td><td>0.238</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td></tr>
<tr><td>02-mean-optimal-weight</td><td>0.238</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td><td>—</td></tr>
<tr><td>03-ridge-regression</td><td>0.057</td><td>0.062</td><td>&lt; 0.001</td><td>0.706</td><td>&lt; 0.001</td><td>0.047</td><td>&lt; 0.001</td><td>0.350</td></tr>
<tr><td>04-roberta-regression</td><td>0.001</td><td>0.001</td><td>&lt; 0.001</td><td>0.096</td><td>&lt; 0.001</td><td>0.786</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
<tr><td>05-modern-bert-interval-weight</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>0.675</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
<tr><td>06-modern-bert-passage-conditioned</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
<tr><td>07-dat-qwen3</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>08-dat-ministral</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>09-llm-fs-qwen3-interval-weight</td><td>0.004</td><td>0.016</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>0.002</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
<tr><td>10-llm-fs-ministral-interval-weight</td><td>0.173</td><td>0.359</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
</tbody>
</table>

## Significance — Wilcoxon signed-rank (p-values)

Per-query significance of each method vs two baselines — `01-standard-rrf` (RRF) and `02-mean-optimal-weight` (MOW) — over the full eval set (0-fill for missing queries). Shown as `< 0.001`, else the actual value; `—` = self-comparison / no data.

### ndcg@10 — acord-entire-corpus

<table>
<thead>
<tr><th rowspan='2'>experiment</th><th colspan='2'>BM25(M)</th><th colspan='2'>BM25(Q)</th><th colspan='2'>RM3(M)</th><th colspan='2'>RM3(Q)</th></tr>
<tr><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th></tr>
</thead>
<tbody>
<tr><td>01-standard-rrf</td><td>—</td><td>0.441</td><td>—</td><td>0.658</td><td>—</td><td>0.118</td><td>—</td><td>0.679</td></tr>
<tr><td>02-mean-optimal-weight</td><td>0.441</td><td>—</td><td>0.658</td><td>—</td><td>0.118</td><td>—</td><td>0.679</td><td>—</td></tr>
<tr><td>03-ridge-regression</td><td>0.139</td><td>0.156</td><td>0.496</td><td>0.310</td><td>0.304</td><td>0.594</td><td>0.865</td><td>0.138</td></tr>
<tr><td>04-roberta-regression</td><td>0.575</td><td>0.688</td><td>0.225</td><td>0.133</td><td>0.852</td><td>0.481</td><td>0.330</td><td>0.205</td></tr>
<tr><td>05-modern-bert-interval-weight</td><td>0.492</td><td>0.582</td><td>0.208</td><td>0.119</td><td>0.800</td><td>0.453</td><td>0.325</td><td>0.199</td></tr>
<tr><td>06-modern-bert-passage-conditioned</td><td>0.080</td><td>0.102</td><td>0.059</td><td>0.024</td><td>0.280</td><td>0.850</td><td>0.008</td><td>0.005</td></tr>
<tr><td>07-dat-qwen3</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>08-dat-ministral</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>09-llm-fs-qwen3-interval-weight</td><td>0.638</td><td>0.645</td><td>0.911</td><td>0.675</td><td>0.815</td><td>0.480</td><td>0.898</td><td>0.663</td></tr>
<tr><td>10-llm-fs-ministral-interval-weight</td><td>0.648</td><td>0.464</td><td>0.184</td><td>0.078</td><td>0.513</td><td>0.968</td><td>0.166</td><td>0.078</td></tr>
</tbody>
</table>

### mrr@10 — msmarco

<table>
<thead>
<tr><th rowspan='2'>experiment</th><th colspan='2'>BM25(M)</th><th colspan='2'>BM25(Q)</th><th colspan='2'>RM3(M)</th><th colspan='2'>RM3(Q)</th></tr>
<tr><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th></tr>
</thead>
<tbody>
<tr><td>01-standard-rrf</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td></tr>
<tr><td>02-mean-optimal-weight</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td><td>—</td></tr>
<tr><td>03-ridge-regression</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>0.622</td><td>&lt; 0.001</td><td>0.035</td></tr>
<tr><td>04-roberta-regression</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
<tr><td>05-modern-bert-interval-weight</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
<tr><td>06-modern-bert-passage-conditioned</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
<tr><td>07-dat-qwen3</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>08-dat-ministral</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>09-llm-fs-qwen3-interval-weight</td><td>&lt; 0.001</td><td>0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>0.048</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
<tr><td>10-llm-fs-ministral-interval-weight</td><td>&lt; 0.001</td><td>0.068</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>0.044</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
</tbody>
</table>

### ndcg@10 — nfcorpus

<table>
<thead>
<tr><th rowspan='2'>experiment</th><th colspan='2'>BM25(M)</th><th colspan='2'>BM25(Q)</th><th colspan='2'>RM3(M)</th><th colspan='2'>RM3(Q)</th></tr>
<tr><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th></tr>
</thead>
<tbody>
<tr><td>01-standard-rrf</td><td>—</td><td>0.156</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>0.820</td><td>—</td><td>0.001</td></tr>
<tr><td>02-mean-optimal-weight</td><td>0.156</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>0.820</td><td>—</td><td>0.001</td><td>—</td></tr>
<tr><td>03-ridge-regression</td><td>0.335</td><td>0.289</td><td>&lt; 0.001</td><td>0.650</td><td>0.779</td><td>0.360</td><td>0.002</td><td>0.836</td></tr>
<tr><td>04-roberta-regression</td><td>0.200</td><td>0.036</td><td>&lt; 0.001</td><td>0.292</td><td>0.445</td><td>0.250</td><td>0.014</td><td>0.606</td></tr>
<tr><td>05-modern-bert-interval-weight</td><td>0.048</td><td>0.005</td><td>&lt; 0.001</td><td>0.383</td><td>0.147</td><td>0.062</td><td>&lt; 0.001</td><td>0.226</td></tr>
<tr><td>06-modern-bert-passage-conditioned</td><td>0.841</td><td>0.428</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>0.325</td><td>0.194</td><td>&lt; 0.001</td><td>0.866</td></tr>
<tr><td>07-dat-qwen3</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>08-dat-ministral</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>09-llm-fs-qwen3-interval-weight</td><td>0.591</td><td>0.257</td><td>0.005</td><td>0.168</td><td>0.617</td><td>0.573</td><td>0.033</td><td>0.507</td></tr>
<tr><td>10-llm-fs-ministral-interval-weight</td><td>0.820</td><td>0.427</td><td>0.019</td><td>0.322</td><td>0.810</td><td>0.980</td><td>0.737</td><td>0.315</td></tr>
</tbody>
</table>

### mrr@10 — nq

<table>
<thead>
<tr><th rowspan='2'>experiment</th><th colspan='2'>BM25(M)</th><th colspan='2'>BM25(Q)</th><th colspan='2'>RM3(M)</th><th colspan='2'>RM3(Q)</th></tr>
<tr><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th><th>vs RRF</th><th>vs MOW</th></tr>
</thead>
<tbody>
<tr><td>01-standard-rrf</td><td>—</td><td>0.001</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td></tr>
<tr><td>02-mean-optimal-weight</td><td>0.001</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td><td>—</td><td>&lt; 0.001</td><td>—</td></tr>
<tr><td>03-ridge-regression</td><td>&lt; 0.001</td><td>0.086</td><td>&lt; 0.001</td><td>0.373</td><td>&lt; 0.001</td><td>0.058</td><td>&lt; 0.001</td><td>0.171</td></tr>
<tr><td>04-roberta-regression</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>0.157</td><td>&lt; 0.001</td><td>0.140</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
<tr><td>05-modern-bert-interval-weight</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>0.179</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
<tr><td>06-modern-bert-passage-conditioned</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
<tr><td>07-dat-qwen3</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>08-dat-ministral</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>09-llm-fs-qwen3-interval-weight</td><td>0.007</td><td>0.026</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
<tr><td>10-llm-fs-ministral-interval-weight</td><td>0.480</td><td>0.853</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr>
</tbody>
</table>

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
