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

Query-adaptive methods are significantly better than RRF at k=3 (p<0.001). Headroom grows with k. The oracle ceiling rises 0.009–0.029 absolute over the best k=2 pair and the oracle−RRF gap widens (e.g. NQ 0.264 vs ≤0.220, MSMARCO 0.230 vs ≤0.175), so k=2 is a lower bound. Recovery via prediction gets harder, as the candidate space explodes from 101 settings at k=2 to 5,151 at k=3. 


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
properties to prediction difficulty across three methods: mean optimal weight,
ModernBERT passage-conditioned, and LLM few-shot Ministral.

We define prediction difficulty for a query as the gap between the predicted fusion weight and the
nearest edge of that query's oracle-optimal weight interval, and zero if the prediction already falls
inside the interval. We use all four sparse and dense retriever combinations (BM25 or RM3 paired with
MiniLM or Qwen3). Within each combination we rank queries by this gap and label the hardest 5% as weakly
predicted and the easiest 5% as well predicted. For each method and dataset we then concatenate these
sets across the four combinations and drop duplicate queries, keeping the most extreme instance of any
query flagged by more than one combination.

We report four query properties. Average term rarity is the mean over the query's tokens of the negative
log of the fraction of collection queries containing the token, so a higher value means rarer
vocabulary. Word count is the number of word tokens. Entity count is the number of named entities
detected by a spaCy NER model. Ambiguity follows the CLAMBER taxonomy (ACL 2024): we prompt an LLM to
mark each query as unambiguous or as one of eight ambiguity types, and report the rate at which queries
are flagged ambiguous. For MSMARCO, each cell below shows the well-predicted mean → weakly-predicted
mean, with the point-biserial p against the weakly-predicted label, and bold marks p < 0.05.

| query property (MSMARCO) | 02 mean-optimal | 06 ModernBERT (passage-cond.) | 10 LLM few-shot (Ministral) |
|---|:--:|:--:|:--:|
| average term rarity | 5.13 → 5.28 (**0.007**) | 5.13 → 5.18 (0.38) | 5.00 → 5.25 (**<0.001**) |
| word count | 5.79 → 6.25 (**0.001**) | 5.84 → 5.92 (0.60) | 5.85 → 6.13 (**0.042**) |
| entity count | 0.25 → 0.30 (0.084) | 0.26 → 0.28 (0.49) | 0.21 → 0.30 (**0.001**) |
| ambiguous-query rate | 0.57 → 0.59 (0.67) | 0.56 → 0.58 (0.54) | 0.60 → 0.59 (0.74) |

Taking MSMARCO as an example, the relationship between query properties and prediction difficulty is
small and method-dependent. For the mean-optimal-weight baseline and the LLM few-shot method, weakly
predicted queries are significantly longer and built from rarer terms than well predicted ones, and for
the LLM method they also contain more named entities. For ModernBERT passage-conditioned, no surface
property significantly separates the two groups. The ambiguity signal the reviewer specifically asked
about does not distinguish weakly from well predicted queries for any method (all p > 0.5). Even where
significant, effects are small (|r| ≤ 0.13), so these surface properties explain only part of the
difficulty, and the per-query optimum remains hard to predict from the query alone.

