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


## W1 — Small fonts in figures, equations, and tables

We agree some tables are too dense. In the revision we will enlarge them.


## W2 — Generalization beyond k=2

We ran a three-retriever pilot: BM25 + RM3 + Qwen3 with the two strongest methods — ModernBERT passage-conditioned (Tier 2) and Ministral few-shot (Tier 3) — against a k=3 RRF baseline, with a per-query paired t-test.

| dataset | metric | best k=2 method | k=3 RRF | k=3 query-adaptive (best) | oracle ceiling k=2 → k=3 |
|---|:--:|:--:|:--:|:--:|:--:|
| NFCorpus | ndcg@10 | 0.409 | 0.384 | 0.398 (p<0.001) | 0.473 → 0.482 |
| MSMARCO | mrr@10 | 0.350 | 0.247 | 0.340 (p<0.001) | 0.457 → 0.477 |
| NQ | mrr@10 | 0.447 | 0.314 | 0.435 (p<0.001) | 0.549 → 0.578 |
| ACORD | ndcg@10 | 0.159 | 0.132 | 0.148 (p=0.21) | 0.231 → 0.244 |

Query-adaptive methods are significantly better than RRF at k=3 (p<0.001). Headroom grows with k. The oracle ceiling rises 0.009–0.029 absolute over the best k=2 pair and the oracle−RRF gap widens (e.g. NQ 0.264 vs ≤0.220, MSMARCO 0.230 vs ≤0.175), so k=2 is a lower bound. Recovery via prediction gets harder, as the candidate space explodes from 101 settings at k=2 to 5,151 at k=3. Because RM3 and BM25 are similar retrievers, the simpler k=2 setting with BM25 and Qwen may capture most of the range of effective fusion weights predictable from the query such that adding the third retriever provides little value.


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


## W4 — Code/data release not mentioned (reproducibility)

We thank the reviewer for raising this. We are committed to full reproducibility. We now have approval from our institution to open-source the code. An anonymized version of the open-source repository is available for review at:

**[https://anonymous.4open.science/r/query-based-rrf-BCC8](https://anonymous.4open.science/r/query-based-rrf-BCC8)**



## W5. Limited investigation of why prediction is difficult

We would first gently note that recovery is not uniformly small. Under an evaluation over all queries,
rather than only those with a non-empty optimal-weight set, strong query-adaptive methods recover up to 61% 
of the available headroom on NQ (RM3+Qwen3) and exceed 25 percent on several MSMARCO and NQ
configurations. That said, a gap remains. To investigate, we ran a query-level error analysis relating query 
properties to prediction difficulty across three methods (mean optimal weight, 
ModernBERT passage-conditioned, and LLM few-shot Ministral).

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

The effects are consistent but small, so these surface properties explain only part of the difficulty, and the per-query optimum remains difficult to predict from the query alone. 

