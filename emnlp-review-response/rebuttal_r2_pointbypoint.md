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

We thank Reviewer zkNL for the careful reading and for recognizing the interval observation as
novel, the evaluation as extensive, and the latency-aware framework as practically relevant. We
address each point below, in the order raised. 

---

## W1 — "Limited methodological novelty; empirical characterization rather than a new method."

We appreciate the reviewer raising this, and we understand the expectation of a new fusion or optimization method. We would like to clarify our intended contribution. By design, the paper is a diagnostic and structural study rather than a proposal for a single new fusion model, and the paper states this explicitly (§3.3: "The spectrum is not a proposal for a single new method"). The contribution is not another predictor but the finding the reviewer credits as novel: that the per-query optimal WRRF weight is a set-valued interval, not a point. 
Additionally, we contribute an analysis of the quality-latency trade-off across query-adaptive methods, compared to standard 50-50 RRF and the mean optimal weight setting (set according to training data).  We show that quality and latency do not trade off monotonically: on responsive datasets, a CPU-only method (the mean optimal weight or a lightweight regressor) recovers much of the achievable headroom at a fraction of the cost of an LLM call, while on harder configurations even LLM-based methods recover only a small fraction of it. This motivates the tiered decision framework we present in §6, which maps a practitioner's constraints, labeled training data, GPU access, latency budget, to the appropriate method class.



---

## W2 — "Oracle relies on a fixed-step grid search; sensitivity of the intervals to discretization is not investigated."

We thank the reviewer for this suggestion. It is a valuable robustness check, and we agree the sensitivity should be shown rather than assumed. We added a discretization sensitivity analysis comparing the 0.01 grid used in the paper (101 weights) against a 10× finer 0.001 grid (1001
weights), measuring per query (i) the change in the
total optimal-interval length, (ii) the disagreement between the two intervals (union −
intersection), and (iii) the resulting change in the retrieval metric at the mean best weight.
Across all four datasets and their four retriever pairs (16 configurations), the effect is negligible:

- interval length changes by only 0.002–0.007 on the [0,1] weight scale,
- interval disagreement is 0.004–0.011 (the coarse and fine intervals almost entirely coincide), and
- the retrieval metric changes by < 0.001 (≤ 0.0007) at the best weight.

This is expected: the retrieval metric is piecewise-constant in w, so a finer grid only sharpens
the interval endpoints rather than creating or removing intervals. The measurements confirm that the 0.01 grid already localizes them to within ~0.01 with
no meaningful effect on the metric, so the interval statistics reported in the paper are not artifacts of the chosen resolution. 

---

## W3 — "Lacks statistical significance tests for the reported improvements."

We have added per-query significance testing of every fusion method against standard
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

The gains from per-query weighting are real, not noise. On both large benchmarks (MSMARCO, NQ) nearly every improvement over RRF survives FDR correction, including the small-looking gains. On the very small ACORD (57 test and 51 train queries) only 3 of 32 reach significance under FDR, which we might expect for such a small dataset. 
 
We additionally verified that many query-adaptive methods beat not just 50-50 RRF but the dataset-specific mean-optimal weight (48/128 significant, 46 surviving FDR). Ultimately our contribution is a framework for practitioners to determine the most appropriate fusion method given their dataset characteristics (mean optimal weight intervals observed in a train split and whether cross-dataset fixed weights fall into the optimal weight intervals for most queries) and deployment requirements (acceptable latency, quality, and available infrastructure).

---

## W4 — "Experiments limited to two-way sparse–dense fusion; generalization to k>2 is open."

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

Three takeaways follow. First, query-adaptive methods extend beyond k=2 and still beat standard
RRF (p < 0.001 on both large benchmarks). Second, the headroom grows with k: the per-query oracle
ceiling rises when the third retriever is added (+0.009 to +0.029 absolute over the best k=2 pair),
and the oracle−RRF gap is larger at k=3 than for any k=2 pair (e.g. NQ 0.264 vs ≤0.220,
MSMARCO 0.230 vs ≤0.175), so the k=2 study is a lower bound on achievable headroom. Third,
recovering that headroom becomes harder as k grows, because the space of candidate weight settings
explodes. With a grid step size of 0.01, the k=2 setting affords 101 possible combinations. The k=3
setting has 5,151 possible settings. Predicting the per-query optimum thus becomes more challenging.
Because RM3 and BM25 are similar retrievers, it may well be the case that the simpler k=2 setting
with BM25 and Qwen captures most of the range of effective fusion weights predictable from the
query, and adding the third retriever provides little value. In practice most hybrid search systems
combine two retrievers that focus on lexical and semantic matching respectively.

---

## C1 — Practical implications of the interval-valued formulation

The interval structure carries three concrete design implications beyond interval-aware supervision, and our decision framework (§6) operationalizes them.

First, it provides a signal for whether dynamic weighting is worth doing at all. Before deploying any predictor, a practitioner can measure two quantities on a labeled train split: the fraction of queries whose optimal interval already contains 0.5 (Figure 1), and the dataset-level headroom (Oracle−RRF)/Oracle (§6.1). When many queries are already optimal at 0.5 or the headroom is small, as on ACORD and NFCorpus, per-query prediction cannot pay off, and standard RRF (T0) or a single dataset-level constant (T1a) is the right choice with no query-time model warranted.

Second, the interval width relaxes the precision a predictor needs. Because the optimum is a range rather than a point, any prediction that lands inside the interval is optimal, so a cheap low-capacity predictor is often sufficient. This is why our CPU-only T1a and T1b as well as encoder-based T2 methods recover headroom comparable to the far more expensive LLM predictors on responsive datasets (§5.1, Table 2): the task does not require hitting an exact point.

Third, only when the headroom is large and the optimum is genuinely query dependent, with few queries optimal at 0.5 and a wide oracle-to-RRF gap as on MS MARCO and NQ, does a higher tier become justified, and even then the bottleneck is the predictability of the optimum from the query rather than the choice of method (§5.2). The interval view thus reframes the practitioner's question from which predictor is best to whether this dataset needs per query weighting at all and, if so, at what tier, which is the mapping our framework provides.

---

## C2 — Writing / typos

We thank the reviewer for catching these. We have fixed the specific instances noted and have done a full proofreading pass to correct the missing articles.
