# Response to Reviewer zkNL

<details>
<summary><b>Original review (Reviewer zkNL) — click to expand</b></summary>

> **Paper Summary:**
> The paper presents a comprehensive empirical study of query-adaptive Weighted Reciprocal Rank
> Fusion (WRRF) in hybrid information retrieval. Rather than introducing another fusion model, it
> challenges the common assumption that every query has a unique optimal fusion weight. The authors
> demonstrate that the optimal weight generally forms an interval rather than a single point, and
> evaluate a wide spectrum of prediction strategies ranging from constant weights to encoder models
> and LLM-based predictors. The study further analyzes latency-performance trade-offs and proposes a
> practical decision framework for selecting fusion methods under deployment constraints.
>
> **Summary Of Strengths:**
> The paper addresses an interesting and relevant question in hybrid retrieval. The observation that
> optimal WRRF weights frequently form intervals instead of single values is novel and provides a
> different perspective on query-adaptive fusion. The experimental evaluation is extensive, covering
> multiple datasets, retriever combinations, prediction strategies, oracle analyses, and latency
> measurements. The analysis goes beyond reporting retrieval metrics by investigating why standard
> RRF remains competitive and how much of the theoretical improvement current methods can actually
> recover. The latency-aware evaluation and deployment recommendations also increase the practical
> relevance of the work. Finally, the paper is generally well written and easy to follow.
>
> **Summary Of Weaknesses:**
> The main limitation of the paper is its limited methodological novelty. The contribution is
> primarily an empirical characterization rather than a new retrieval or fusion method. Most
> evaluated predictors are existing models, and the proposed interval-based formulation is mainly
> exploited through modified supervision rather than a fundamentally new optimization framework. In
> addition, the oracle construction relies on a discretized grid search with a fixed step size, but
> the sensitivity of the discovered intervals to this discretization is not investigated. The paper
> also lacks statistical significance tests for the reported retrieval improvements, making it
> difficult to assess whether some of the observed gains are statistically meaningful. Finally, the
> experiments are limited to two-way sparse–dense fusion, leaving open the question of whether the
> same conclusions generalize to more complex fusion settings.
>
> **Comments Suggestions And Typos:**
> - The paper would benefit from a more explicit discussion of the practical implications of the
>   interval-valued formulation. While the analysis convincingly shows that optimal weights often
>   form intervals, it remains somewhat unclear how this observation should influence the design of
>   future retrieval systems beyond the proposed interval-aware supervision.
> - The oracle construction relies on a grid search with a discretization step of 0.01. It would
>   strengthen the paper to include a sensitivity analysis showing that the interval statistics
>   remain stable under finer discretization, thereby demonstrating that the conclusions are not
>   artifacts of the chosen resolution.
> - The experimental section would also be strengthened by reporting statistical significance tests
>   for the retrieval results in addition to the absolute performance values. Since many of the
>   reported improvements are relatively small, significance testing would help readers assess
>   whether these differences are meaningful.
> - The discussion could also elaborate on how the findings might extend to fusion involving more
>   than two retrieval systems. Although this limitation is acknowledged, a brief discussion of the
>   expected challenges or possible formulations in higher-dimensional weight spaces would increase
>   the impact of the paper.
> - Finally, there are several minor writing issues throughout the manuscript. For example, in
>   Section 3 the sentence "The spectrum is not a proposal for a single new method; it a set of
>   candidates..." is missing the "is". I also noticed occasional grammatical inconsistencies such as
>   "ModernBERT is exhibits the most expressive power" and several places where articles are missing
>   or wording could be made more fluent. A careful proofreading would further improve readability.
>
> **Confidence:** 4 · **Soundness:** 2.5 · **Excitement:** 2 · **Overall Assessment:** 2 = Resubmit
> next cycle · **Reproducibility:** 1 · **Datasets:** 1 (No usable datasets submitted) · **Software:**
> 1 (No usable software released).

</details>

---

We thank Reviewer zkNL for the careful reading. We address each point below.


## W1 — Novelty and contribution framing

This is a diagnostic study. §3.3 indicates ("The spectrum is not a proposal for a single new method"). Our main contribution is what the reviewer already credits as novel, that the per-query optimal WRRF weight is a set-valued interval, not a point.

We also extensively analyze quality-latency trade-off across query-adaptive methods, standard RRF, and a competitive mean-optimal-weight baseline. In many cases a CPU-only method for fusion weight assignment recovers much of the achievable headroom at a fraction of the cost of an LLM call. Some datasets have harder-to-predict fusion weights. On these, even LLM methods recover only a small amount of headroom. This motivates our tiered decision framework in §6, which maps availability of labeled training data, GPU access, and latency constraints to the right method class.


## W2 — Discretization sensitivity

We reran the oracle grid search at 0.001 against the paper's 0.01 grid across all 16 retriever configurations. The effect on interval length, interval disagreement, and the chosen retrieval metric for each dataset at the mean best weight is negligible: length shifts by only 0.002–0.007, disagreement is 0.004–0.011, and the retrieval metric changes by at most 0.0007. The retrieval metrics (NDCG and MRR) are piecewise-constant in w, so a finer grid sharpens interval endpoints without creating or removing them.


## W3 — Statistical significance testing

We added per-query significance testing of every method against standard RRF: a paired t-test across 128 comparisons (8 methods × 16 configurations), FDR-controlled via Benjamini–Hochberg (1995). 75 of 128 are significant at α = 0.05, and 74 survive FDR correction:

| dataset | eval queries | sig. improvements over RRF (of 32) | survive FDR (q ≤ 0.05) |
|---|:--:|:--:|:--:|
| MSMARCO | 6,980 | 31 / 32 | 31 / 32 |
| NQ | 2,893 | 30 / 32 | 30 / 32 |
| NFCorpus | 323 | 10 / 32 | 10 / 32 |
| ACORD | 57 | 4 / 32 | 3 / 32 |

Nearly every improvement survives FDR on the two large benchmarks (MSMARCO, NQ); on small ACORD only 3 of 32 remain significant.

We also verified that many query-adaptive methods beat not just RRF but the dataset-specific mean-optimal weight (48/128 significant, 46 surviving FDR). Ultimately our contribution is a framework for practitioners to determine the most appropriate fusion method given their dataset characteristics and deployment requirements.


## W4 — Generalization beyond k=2

We ran a three-retriever pilot: BM25 + RM3 + Qwen3 with the two strongest methods — ModernBERT passage-conditioned (Tier 2) and Ministral few-shot (Tier 3) — against a k=3 RRF baseline, with a per-query paired t-test.

| dataset | metric | best k=2 method | k=3 RRF | k=3 query-adaptive (best) | oracle ceiling k=2 → k=3 |
|---|:--:|:--:|:--:|:--:|:--:|
| NFCorpus | ndcg@10 | 0.409 | 0.384 | 0.398 (p<0.001) | 0.473 → 0.482 |
| MSMARCO | mrr@10 | 0.350 | 0.247 | 0.340 (p<0.001) | 0.457 → 0.477 |
| NQ | mrr@10 | 0.447 | 0.314 | 0.435 (p<0.001) | 0.549 → 0.578 |
| ACORD | ndcg@10 | 0.159 | 0.132 | 0.148 (p=0.21) | 0.231 → 0.244 |

Query-adaptive methods are significantly better than RRF at k=3 (p<0.001). Headroom grows with k. The oracle ceiling rises 0.009–0.029 absolute over the best k=2 pair and the oracle−RRF gap widens (e.g. NQ 0.264 vs ≤0.220, MSMARCO 0.230 vs ≤0.175), so k=2 is a lower bound. Recovery via prediction gets harder, as the candidate space explodes from 101 settings at k=2 to 5,151 at k=3. Because RM3 and BM25 are similar retrievers, the simpler k=2 setting with BM25 and Qwen may capture most of the range of effective fusion weights predictable from the query such that adding the third retriever provides little value.


## C1 — Practical implications

In addition to our contribution of an interval-aware loss, the interval structure gives three design implications that our decision framework (§6) operationalizes.

Before deploying any predictor, a practitioner can measure on a train split: the fraction of queries whose optimal interval already contains 0.5 (Figure 1) and RRF headroom relative to the oracle (§6.1). When many queries are already optimal at 0.5 or the headroom is small, per-query prediction has limited utility, and standard RRF (T0) or a dataset-level constant (T1a) is the right choice with no query-time model warranted.

Since the optimum is a range, any prediction inside it is optimal, so a cheap predictor often suffices. CPU-only (T1) and encoder (T2) methods recover headroom comparable to the far more expensive LLM predictors on responsive datasets (§5.1, Table 2).

A higher tier is warranted when headroom is large and query-dependent (MS MARCO, NQ). Even there the bottleneck is predicting of the optimum from the query, not the method (§5.2). This turns "which predictor is best" into "does this dataset need per-query weighting, and at what tier."


## C2 — Writing / typos

We thank the reviewer for catching these and have fixed them.
