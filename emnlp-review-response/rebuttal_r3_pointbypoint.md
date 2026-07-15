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



## W5 — Limited investigation of why prediction is difficult

Recovery is not uniformly small: evaluated over all queries rather than only those with a non-empty optimal-weight set, strong query-adaptive methods recover up to 61% of headroom on NQ (RM3+Qwen3) and exceed 25% on several MSMARCO/NQ configurations. A gap remains, so we ran a query-level analysis linking query properties to prediction difficulty for three methods: mean optimal weight, ModernBERT passage-conditioned, and LLM few-shot Ministral.

For each query we measure the gap between the predicted weight and the nearest edge of its oracle-optimal interval (zero if inside), pooled across all four retriever pairs. We label the hardest 5% "weakly predicted" and easiest 5% "well predicted," and test four properties: term rarity (mean negative log document frequency), word count, entity count (spaCy NER), and ambiguity rate (CLAMBER taxonomy, Zhang et al. 2024, via LLM classification). For MSMARCO, each cell reports the well-predicted vs. weakly-predicted mean and p from a point-biserial correlation (correlating the binary label with the property; equivalent to a two-sample t-test), whose magnitude |r| we cite below; bold marks p < 0.05.

| property (MSMARCO) | mean-optimal | ModernBERT (passage-cond.) | LLM few-shot (Ministral) |
|---|:--:|:--:|:--:|
| term rarity | 5.13 vs. 5.28 (**0.007**) | 5.13 vs. 5.18 (0.38) | 5.00 vs. 5.25 (**<0.001**) |
| word count | 5.79 vs. 6.25 (**0.001**) | 5.84 vs. 5.92 (0.60) | 5.85 vs. 6.13 (**0.042**) |
| entity count | 0.25 vs. 0.30 (0.084) | 0.26 vs. 0.28 (0.49) | 0.21 vs. 0.30 (**0.001**) |
| ambiguous rate | 0.57 vs. 0.59 (0.67) | 0.56 vs. 0.58 (0.54) | 0.60 vs. 0.59 (0.74) |

The relationship is small and method-dependent. Weakly predicted queries are longer and rarer for mean-optimal and LLM baselines (also more entity-rich for the LLM); no property separates ModernBERT's groups. Ambiguity does not separate weakly from well predicted queries for any method (all p > 0.5). Effects are small even where significant (|r| ≤ 0.13). These properties explain only part of the difficulty, and the per-query optimum remains hard to predict from the query alone.

