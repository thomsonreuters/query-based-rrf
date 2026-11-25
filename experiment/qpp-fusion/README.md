# QPP-driven fusion baselines

Baselines that set the per-query WRRF weight from **query performance prediction (QPP)**,
added in response to Reviewer Sg6v (W2: "prior work uses QPP to adapt fusion strength; the paper
only compares against DAT"). The supervised variants also use the retrieval-internal agreement
features suggested by Reviewer 25bQ (top-k overlap, documents exclusive to one channel, RRF score
margin). The script also reports a **tuned static w\*** (the single weight that maximises the
mean training metric), which 25bQ asked for as a stronger static baseline than MOW.

Everything runs on CPU from the scores column of the top-200 run files. No index, corpus, GPU or
LLM is needed.

## Predictors (`qpp_predictors.py`)

Computed separately on each retriever's top-200 list:

| Name | Definition |
|---|---|
| `sd@k` | std of top-k scores (σ_k) |
| `nqc@k` | std(top-k) / \|mean(top-200)\| (NQC, with the list mean standing in for the corpus score) |
| `wig@k` | (mean(top-k) − mean(top-200)) / \|mean(top-200)\| |
| `smv@k` | mean over top-k of s·\|ln(s/mean(top-k))\| / \|mean(top-200)\| |
| `max` | top-1 score |
| `gap` | (s₁ − s₂) / \|s₁\| |

A run file has no corpus-level score, and none exists for a dense retriever, so the mean of the
retrieved list is used as the reference score. This makes NQC, WIG, SMV and gap scale-free: they
are unchanged if all scores in a list are multiplied by a constant.

## Weight predictors (`qpp_fusion.py`)

| Method | Labels? | Rule |
|---|---|---|
| **QPP-U [p]** | no | Each predictor is mapped to [0, 1] by its empirical CDF over that retriever's training queries; w = F_s / (F_s + F_d) |
| **QPP-C [p]** | yes | w = σ(a·(F_s − F_d) + b); a, b fitted with the satisficing loss (Eq. 3) |
| **QPP-LR** | yes | σ(linear) on all predictors of both lists + agreement features + query length; satisficing loss + L2 (λ chosen by 5-fold CV) |
| **QPP-GBR** | yes | gradient-boosted trees on the same features, MSE against the interval midpoint |

`QPP-U [nqc@100]` is the fixed, label-free default. The `[train-best]` rows choose the predictor
with the best training-split metric, which uses training labels. Every predictor is reported in
`qpp_predictors.csv`.

Reference rows are computed with the same code: Standard RRF, Mean Optimal Weight, Tuned static
w\* and Oracle. Fusion goes through `helper_4_fuse_results_wrrf.py` (k = 60) and scoring through
`helper_5_ir_metrics.py`, as for the paper's other methods.

## Inputs

Paths follow the repo convention under `--data_root` (default `$BASE_DATA_DIR`, else `./dataset`):

```
{dataset}/search_results/{bm25,rm3,biencoder,qwen3}/top200/results_{train,<eval>}.trec
{dataset}/qrels/{train,<eval>}.tsv
{dataset}/queries.jsonl                       (optional: query text / length feature)
{dataset}/{mrr|ndcg}_runs/{split}/top200/results_{split}_{sparse}_vs_{dense}_best_weights.csv
```

If a `*_best_weights.csv` sweep file is missing, it is built with the repo's sweep code. The MRR
sweep uses a vectorised port that `test_qpp.py` checks gives output identical to
`rrf_mrr_best_weights_all_weights_final.py`. Pairs with missing run files are skipped.

## Run

```bash
python experiment/qpp-fusion/test_qpp.py
python experiment/qpp-fusion/qpp_fusion.py --datasets acord-entire-corpus nq
```

Outputs go to `experiments/qpp-fusion/` (git-ignored):

- `summary.csv`: per dataset × pair × method: mean metric@10, bootstrap 95% CI, paired two-sided
  t-test p vs RRF and vs MOW, headroom recovery %, and per-query latency (ms, 95% CI).
- `qpp_predictors.csv`: QPP-U / QPP-C for every predictor (training metric, eval metric from the
  sweep, p vs RRF, calibration a, b).
- `qpp_lr_coefficients.json`: QPP-LR weights on standardised features.
- `predictions/*.csv` (`query_id, query_text, predicted`) and `fused/<method>/*.trec`. These are
  in the format `helper_5_ir_metrics.process_multiple_datasets` reads.

Latency covers weight prediction from the two in-memory score lists (batch size 1, 20 warm-up
queries). It excludes reading run files and the fusion itself, which every WRRF method shares.
