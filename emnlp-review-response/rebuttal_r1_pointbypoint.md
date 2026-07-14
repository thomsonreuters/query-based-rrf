# Response to Reviewer reaf

<details>
<summary><b>Original review (Reviewer reaf) — click to expand</b></summary>

> **Paper Summary:**
> This paper presents an empirical study characterising the optimal reciprocal rank fusion weight in
> hybrid retrieval (sparse and dense) as an interval rather than a single point and evaluating
> various dynamic per-query weight-prediction methods to demonstrate that while substantial
> performance headroom exists, current approaches struggle to recover it due to the inherent
> unpredictability of per-query optima. Experiments were conducted on four collections, two of which
> are large-scale and the other two are small. The proposed study was performed by comparing several
> predictors and comparators. The results show that none of the compared predictors reach the
> per-query oracle; there is strong headroom. Overall, the paper is well written.
>
> **Summary Of Strengths:**
> - Introduces an insightful perspective by reframing the per-query optimal weight as a set-valued
>   interval, providing a solid structural explanation for the robustness of the standard uniform RRF
>   baseline.
> - Conducts a comprehensive and well-organised evaluation that benchmarks a wide spectrum of ten
>   fusion weight-prediction methods across four diverse datasets that cover approaches ranging from
>   simple constants to fine-tuned encoder LMs and LLMs.
> - The proposed tiered decision framework is practical and effectively maps real-world practitioner
>   constraints, including latency budgets, GPU access, and label availability, to the most
>   appropriate method class.
> - Provides an honest and realistic assessment of the field that clearly identifies the core
>   bottleneck of predicting weights from the query alone, rather than just claiming a marginal
>   state-of-the-art improvement.
>
> **Summary Of Weaknesses:**
> - While the paper excellently diagnoses the problem and measures the performance headroom, it fails
>   to propose or evaluate any novel algorithmic method that actually recovers a significant portion
>   of this headroom (capping at <25%), which leaves the core problem largely unsolved.
> - The study is restricted to evaluating a single sparse-dense pair (k=2), which limits the
>   generalizability of the findings to more complex multi-retriever ensembles (k>2) where the
>   optimum becomes a multi-dimensional simplex.
> - Per-query optimisation-related references are missing.
> - Statistical significance testing was not performed.
>
> **Comments Suggestions And Typos:**
> - Please remove the zero before the decimal point in Table 1.
>
> **Confidence:** 4 · **Soundness:** 3.5 · **Excitement:** 3 · **Overall Assessment:** 3.5 =
> Borderline Conference · **Reproducibility:** 4 · **Datasets:** 4 (Useful) · **Software:** 4
> (Useful).

</details>

---

We thank Reviewer reaf for the careful reading and for recognizing the paper's structural reframing, comprehensive evaluation, and honest assessment of the field. We address each point below.

---

## W1 — "No novel method recovers the headroom (<25%)."

By design, the paper is a diagnostic, structural study rather than a proposal for a single new
predictor (§3.3). On recovery specifically:

- The ≤25% figure was computed by averaging each score only over queries with a nonzero achievable metric —
  i.e. queries where some fusion weight can surface a relevant document, the subset where the
  weight has any room to help. We updated our reporting to include all queries
  rather than only those with a non-empty optimal-weight set. Under this evaluation the recovered
  headroom is substantially higher: query-adaptive methods reach up to 61% on NQ (RM3+Qwen3)
  and exceed 25% on several MSMARCO/NQ configurations.
- Headroom nonetheless persists on the harder configurations, and no method reaches the oracle,
  illustrating the difficulty of per-query fusion-weight prediction.



---

## W2 — "Restricted to a single sparse–dense pair (k=2); generalizability to k>2 is unclear."

We agree, and we ran additional experiments in the three-retriever setting. We fuse
BM25 + RM3 + Qwen3 (Qwen3 is the strongest dense retriever on every dataset; BM25 and RM3
trade places across datasets), and re-run two of the strongest query-adaptive methods from the k=2 setting against a k=3 RRF baseline. We select two methods that were the strongest query-adaptive predictors at k=2 and that sit in different tiers of our decision framework (§6): the passage-conditioned ModernBERT predictor (Tier 2, a fine-tuned transformer encoder running on GPU at ~8 ms/query, conditioned on each retriever's top-1 passage) and the few-shot Ministral predictor (Tier 3, in-context LLM inference at ~225 ms/query, conditioned on the query alone). This pairs the strongest fine-tuned discriminative encoder with the strongest generative LLM predictor, spanning both ends of the cost–adaptivity spectrum rather than re-testing two variants of the same approach. These two also recover the most headroom at k=2: across the 16 k=2 configurations they achieve the best score in 10, including 7 of the 8 columns on the large MSMARCO and NQ benchmarks, so they are the natural candidates to carry into the k=3 setting. Statistical significance is a per-query paired t-test comparing each method to the equal-weight RRF baseline, which we selected for its robustness across sample sizes and findings from prior work
(Urbano et al., 2019, [arXiv:1905.11096](https://arxiv.org/abs/1905.11096);
Ihemelandu and Ekstrand, 2023, [arXiv:2305.02461](https://arxiv.org/abs/2305.02461);
Urbano, 2026, [arXiv:2604.25349](https://arxiv.org/abs/2604.25349)).

| dataset | metric | best k=2 method | k=3 RRF | k=3 query-adaptive (best) | oracle ceiling k=2 → k=3 |
|---|:--:|:--:|:--:|:--:|:--:|
| NFCorpus | ndcg@10 | 0.409 | 0.384 | 0.398 (p<0.001) | 0.473 → 0.482 |
| MSMARCO | mrr@10 | 0.350 | 0.247 | 0.340 (p<0.001) | 0.457 → 0.477 |
| NQ | mrr@10 | 0.447 | 0.314 | 0.435 (p<0.001) | 0.549 → 0.578 |
| ACORD | ndcg@10 | 0.159 | 0.132 | 0.148 (p=0.21) | 0.231 → 0.244 |

Three takeaways follow. 
First, query-adaptive methods extend beyond k=2 and still beat standard RRF (p < 0.001 on both large benchmarks). 
Second, the headroom grows with k: the per-query oracle ceiling rises when the third retriever is added (+0.009 to +0.029 absolute over the best k=2 pair), and the oracle−RRF gap is larger at k=3 than for any k=2 pair (e.g. NQ 0.264 vs ≤0.220, MSMARCO 0.230 vs ≤0.175), so the k=2 study is a lower bound on achievable headroom. 
Third, recovering that headroom becomes harder as k grows, because the space of candidate weight settings explodes. With a grid step size of 0.01, the k=2 setting affords 101 possible combinations. The k=3 setting has 5,151 possible settings.  Predicting the per-query optimum thus becomes more challenging.  Because RM3 and BM25 are similar retrievers, it may well be the case that the simpler k=2 setting with BM25 and Qwen captures most of the range of effective fusion weights predictable from the query, and adding the third retriever provides little value.  In practice most hybrid search systems combine two retrievers that focus on lexical and semantic matching respectively.

---

## W3 — "Per-query optimisation-related references are missing."

Thank you. We will strengthen the coverage of per-query optimisation. We already connect to
query performance prediction (Arabzadeh et al., 2024) and to hybrid-fusion analysis (Bruch et al.,
2023; Cormack et al., 2009) in Related Work, but we agree the *per-query weight-optimisation* line
is under-cited. We will add:

- **Per-query / learned combination weighting:** Vogt & Cottrell (1999, deriving the optimal
  weight for linearly combining two retrieval systems); and Sheldon et al. (2011, LambdaMerge, supervised per-query merging that learns to
  weight multiple result lists to optimise a retrieval metric, though for query reformulations
  rather than sparse–dense retrievers).

We welcome any specific references the reviewer has in mind, and will position our per-query
interval analysis relative to this work in the related-work section.

---

## W4 — "Statistical significance testing was not performed."

We have added per-query significance testing of every fusion method against standard (unweighted)
RRF using a paired t-test over 128 tests (8 query-adaptive methods × 16
dataset combinations). We control the false discovery rate via the Benjamini–Hochberg procedure
(Benjamini & Hochberg, 1995), following recent work showing FDR control is appropriate for the
many-comparison setting typical in IR system evaluation (Otero et al., 2025).

Of the 128 comparisons, 75 are significant at α = 0.05 and 74 remain significant after FDR
correction (method significantly better than RRF):

| dataset | eval queries | sig. improvements over RRF (of 32) | survive FDR (q ≤ 0.05) |
|---|:--:|:--:|:--:|
| MSMARCO | 6,980 | 31 / 32 | 31 / 32 |
| NQ | 2,893 | 30 / 32 | 30 / 32 |
| NFCorpus | 323 | 10 / 32 | 10 / 32 |
| ACORD | 57 | 4 / 32 | 3 / 32 |

The gains from per-query weighting are real, not noise. On both large benchmarks (MSMARCO, NQ) nearly every improvement over RRF survives FDR correction, including the small-looking gains. On the very small ACORD (57 test and 51 train queries) only 3 of 32 reach significance under FDR, which we might expect for such a small dataset. We additionally verified that many query-adaptive methods beat not just 50-50 RRF but the dataset-specific mean-optimal weight (48/128 significant, 46 surviving FDR).  Ultimately our contribution is a framework for practitioners to determine the most appropriate fusion method given their dataset characteristics (mean optimal weight intervals observed in a train split and whether cross-dataset fixed weights fall into the optimal weight intervals for most queries) and deployment requirements (acceptable latency, quality, and available infrastructure).

---

## C1 — "Remove the zero before the decimal point in Table 1."

Fixed — thank you. We have updated Table 1 (and checked the other tables for consistency).
