# Response to Reviewer TX8Z

<details>
<summary><b>Original review (Reviewer TX8Z) — click to expand</b></summary>

> **Paper Summary:**
> The paper studies Weighted Reciprocal Rank Fusion (WRRF) for combining sparse and dense
> retrievers in hybrid IR. The authors evaluate their approach against other weight-prediction
> strategies across four datasets and four sparse–dense retriever pairs. Each method is compared to
> a per-query oracle to compute "headroom recovery."
>
> **Summary Of Strengths:**
> - The paper is written really well and it is easy to read.
> - The experiment design was quite thorough with four datasets, four retriever pairings, varying
>   model architectures and ten weight-prediction methods.
>
> **Summary Of Weaknesses:**
> - The figures, equations and tables used really small fonts which made it hard to read the
>   results.
> - The authors' proposed metric only work for single sparse-dense pair only. Although the authors
>   have acknowledged this in the limitation, many systems increasingly use 3+ systems (e.g. BM25 +
>   multiple dense retrievers + a cross-encoder reranker score) which means that grid-search becomes
>   combinatorially expensive.
> - The paper reports average performance but does not include statistical significance testing
>   between methods. Since many improvements are relatively small, significance tests would help
>   determine whether the observed gains are meaningful.
> - Code/data release is not mentioned which means that reproducing the paper will be difficult.
> - The paper concludes that current methods recover only a small fraction of the available
>   headroom, but provides limited investigation and discussion on why prediction is difficult. For
>   instance, which queries are mispredicted, and are there a correlation between query properties
>   (e.g., length, entity density, ambiguity) and prediction error. Those analysis would provide
>   deeper insights for other researchers and strengthen your conclusion beyond reporting the
>   average performance.
>
> **Comments Suggestions And Typos:** See strengths and weaknesses.
>
> **Confidence:** 3 · **Soundness:** 3 · **Excitement:** 3 · **Overall Assessment:** 3 = Findings
> · **Reproducibility:** 2 · **Datasets:** 1 (No usable datasets submitted) · **Software:** 1 (No
> usable software released).

</details>

---

We thank Reviewer TX8Z for the positive assessment and for finding the paper well written and the
experimental design thorough. We address each point below, in the order raised. 

---

## W1 — Small fonts in figures, equations, and tables

We thank the reviewer for flagging this, and we agree the current tables are too dense to read comfortably. In the revision we will enlarge them.

---

## W2 — "Metric works for a single sparse–dense pair only; 3+ systems make grid search combinatorially expensive."

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

## W3 — "Reports average performance but no statistical significance testing between methods."

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

## W4 — Code/data release not mentioned (reproducibility)

We thank the reviewer for raising this. We are committed to full reproducibility. We now have approval from our institution to open-source the code. An anonymized version of the open-source repository is available for review at:

**[https://anonymous.4open.science/r/query-based-rrf-BCC8](https://anonymous.4open.science/r/query-based-rrf-BCC8)**

---


## W5. Limited investigation of why prediction is difficult

We would first gently note that recovery is not uniformly small. Under an evaluation over all queries,
rather than only those with a non-empty optimal-weight set, query-adaptive methods recover up to 61
percent of the available headroom on NQ (RM3+Qwen3) and exceed 25 percent on several MSMARCO and NQ
configurations (revised reporting, §5). That said, we agree that measuring the headroom does not
by itself explain why the remaining gap is hard to close. To investigate why, we ran a query-level error analysis relating
query properties to prediction difficulty.

We define prediction difficulty for a query as the gap between the predicted fusion weight and the
nearest edge of that query's oracle-optimal weight interval, and zero if the prediction already falls
inside the interval. Within each retriever pair we label the hardest 5 percent of queries as weakly
predicted and the easiest 5 percent as well predicted, then pool these across the four retriever pairs
and the query-adaptive methods.

We report three query properties here, computed as follows. Average term rarity is the mean over the
query's tokens of the negative log of the fraction of queries in the collection that contain the token,
so a higher value means the query is built from rarer vocabulary. Word count is the number of word
tokens in the query. Entity count is the number of named entities detected in the query by a spaCy NER
model.

On MSMARCO, weakly predicted queries have consistently higher values than well predicted queries on all
three properties, and every difference is statistically significant (point-biserial correlation, pooled
across methods):

| query property | well predicted (mean) | weakly predicted (mean) | p |
|---|:--:|:--:|:--:|
| average term rarity | 5.09 | 5.24 | < 0.001 |
| word count | 5.82 | 6.10 | < 0.001 |
| entity count | 0.24 | 0.29 | 0.002 |

The direction holds across every method class we examined. The effects are consistent but small, so
these surface properties explain only a small part of the difficulty, and the per-query optimum remains
largely unpredictable from the query alone. Ambiguity, the reviewer's specific hypothesis, was a weaker
and inconsistent signal, reaching at most marginal significance on MSMARCO and no significant
relationship on NQ.

