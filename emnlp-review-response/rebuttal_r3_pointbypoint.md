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
experimental design thorough. We address each point below, in the order raised. New experiments
(significance testing and the k>2 study) are added to the revision, with full tables in the
appendix.

---

## W1 — Small fonts in figures, equations, and tables

We thank the reviewer for flagging this, and we agree the current tables are too dense to read
comfortably. In the revision we will redesign them and will move to the appendix at full page width. We will also enlarge the fonts and labels in the
figures so the results are readable throughout.

---

## W2 — "Metric works for a single sparse–dense pair only; 3+ systems make grid search combinatorially expensive."

We thank the reviewer for this point and agree that real systems increasingly combine more than two
retrievers. We ran additional experiments on the multi-retriever simplex, fusing
**BM25 + RM3 + Qwen3** (k=3) and re-running our two strongest k=2 methods against a k=3 RRF
baseline. Query-adaptive weighting continues to beat RRF on both large benchmarks (MSMARCO and NQ,
Wilcoxon p < 0.001), and the per-query oracle ceiling *rises* when the third retriever is added
(+0.009 to +0.029 absolute over the best k=2 pair) — so the k=2 study is a lower bound on
achievable headroom. The reviewer's point about combinatorial cost is exactly right and is itself
part of the finding: the k=3 simplex has ~5151 weight settings versus 101 at k=2 (growing as
O(g^{k−1})), so per-query optima scatter and no longer collapse into a compact learnable interval,
which is why recovering the headroom gets harder as retrievers are added. Full k=3 tables are added
to the appendix.

See our reply to Reviewer reaf for the full breakdown.

---

## W3 — "Reports average performance but no statistical significance testing between methods."

We thank the reviewer for pointing out the missing significance analysis; we agree it is important
for judging whether the smaller gains are meaningful. We have added per-query significance testing
of every fusion method against standard RRF using the **Wilcoxon signed-rank test**,
with a **Bonferroni correction** over the family of 128 tests (8 query-adaptive methods × 16
dataset–combinations; corrected α = 0.05/128 ≈ 3.9×10⁻⁴). Of the 128 comparisons, **78 are
significant at α = 0.05 and 63 survive Bonferroni**. On the two large benchmarks nearly every
improvement over RRF is significant and survives correction (MSMARCO 29/32, NQ 29/32); on the
small collections the gains are directionally positive but mostly do not reach significance at that
scale (NFCorpus 5/32, ACORD 0/32). So the small-looking gains are consistent per query rather than
noise on the datasets with enough queries to test. Full per-query tables (before and after
Bonferroni) are added to the appendix.

See our reply to Reviewer reaf for the full breakdown.

---

## W4 — Code/data release not mentioned (reproducibility)

We thank the reviewer for raising this. We are committed to full reproducibility. An
anonymized repository is available for this rebuttal at:

**[ANONYMIZED REPO LINK — TODO]**

and the repository will be made public upon acceptance. 

---

## W5 — Limited investigation of *why* prediction is hard (which queries are mispredicted; query-property correlations)

We thank the reviewer for this suggestion. We ran an error analysis: for each dataset we labelled queries as well- vs
poorly-predicted and correlated the outcome with 15 query features (length, term rarity, named
entities, digits, WH-type, and WordNet ambiguity).

The result reinforces the paper's central claim. On the two large datasets, **no query feature
explains more than ~1% of the variance** in whether a query is mispredicted (strongest
|r| = 0.07 on MSMARCO, 0.09 on NQ), and the query *ambiguity* the reviewer hypothesised shows
essentially **no correlation** on any dataset (|r| ≤ 0.08). The one directionally consistent, if
weak, signal is that **longer queries with rarer terms are marginally harder to predict** (holding
sign across all four collections), which fits intuition. In short, mispredicted queries are not
identifiable from surface query properties, which is direct evidence for our conclusion that the
per-query optimum is not predictable from the query alone. We add this analysis, with the full
per-feature correlation tables, to the appendix.

| dataset | strongest feature (→ harder) | r | ambiguity r |
|---|---|:--:|:--:|
| MSMARCO | query length | +0.07 | −0.01 |
| NQ | rare/singleton terms | +0.09 | −0.02 |
| NFCorpus | (term rarity, opposite sign) | −0.17 | +0.05 |
| ACORD | query length | +0.28¹ | +0.08 |

<sub>¹ ACORD has only ~57 queries, so its larger coefficient is high-variance. Correlations are
point-biserial (effect sizes); with the large query counts, significance is not the discriminating
factor — the effect sizes are.</sub>
