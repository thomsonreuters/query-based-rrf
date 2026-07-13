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
address each point below, in the order raised. New experiments (grid-sensitivity, significance
testing, and the k>2 study) are added to the revision, with full tables in the appendix.

---

## W1 — "Limited methodological novelty; empirical characterization rather than a new method."

We appreciate the reviewer raising this, and we understand the expectation of a new fusion or
optimization method. We would like to clarify our intended contribution. By design, the paper is a
diagnostic and structural study rather than a proposal for a single new fusion model, the paper
states this explicitly (§3.3: "The spectrum is not a proposal for a single new method"). The contribution is not another predictor but the finding the
reviewer credits as novel: that the per-query optimal WRRF weight is a **set-valued interval, not
a point**, together with the oracle-based localization of *where* the achievable headroom is and
*why* it is hard to recover, the bottleneck is the predictability of the per-query optimum from
the query alone (§5.2). Establishing this
reframes per-query fusion as a set-valued problem and explains RRF's robustness, which we believe
is a contribution.

We give a fuller treatment of the novelty/contribution framing, including corrected
headroom-recovery numbers under a full-eval-set convention, in our reply to Reviewer reaf; see
that response for the full breakdown.

---

## W2 — "Oracle relies on a fixed-step grid search; sensitivity of the intervals to discretization is not investigated."

We thank the reviewer for this suggestion. It is a valuable robustness check, and we agree the
sensitivity should be shown rather than assumed. We added a discretization sensitivity analysis
comparing the 0.01 grid used in the paper (101 weights) against a 10× finer 0.001 grid (1001
weights), measuring per query (i) the change in the
total optimal-interval length, (ii) the disagreement between the two intervals (union −
intersection), and (iii) the resulting change in the retrieval metric at the mean best weight.
Across the datasets and retriever pairs tested, the effect is negligible:

- interval length changes by only **0.003–0.007** on the [0,1] weight scale,
- interval disagreement is **0.004–0.011** (the coarse and fine intervals almost entirely coincide), and
- the retrieval metric changes by **< 0.001** (≤ 0.0007) at the best weight.

This is expected: the retrieval metric is piecewise-constant in w, so a finer grid only sharpens
the interval endpoints rather than creating or removing intervals. The measurements confirm that the 0.01 grid already localizes them to within ~0.01 with
no meaningful effect on the metric, so the interval statistics reported in the paper are not
artifacts of the chosen resolution. We add this analysis to the appendix.

---

## W3 — "Lacks statistical significance tests for the reported improvements."

We thank the reviewer for pointing out the missing significance analysis; we agree it is important
for judging whether the smaller gains are meaningful. We have added per-query significance testing
of every fusion method against standard (unweighted) RRF using the **Wilcoxon signed-rank test**, with a **Bonferroni correction** over the family of
128 tests (8 query-adaptive methods × 16 dataset–combinations; corrected α = 0.05/128 ≈ 3.9×10⁻⁴).
Of the 128 comparisons, **78 are significant at α = 0.05 and 63 survive Bonferroni**. On the two
large benchmarks nearly every improvement over RRF is significant and survives correction
(MSMARCO 29/32, NQ 29/32); on the small collections the gains are directionally positive but
mostly do not reach significance at that scale (NFCorpus 5/32, ACORD 0/32). So the small-looking
gains the reviewer flagged are consistent per query rather than noise on the datasets with enough
queries to test. Full per-query tables (before and after Bonferroni) are added to the appendix.

See our reply to Reviewer reaf for the full breakdown.

---

## W4 — "Experiments limited to two-way sparse–dense fusion; generalization to k>2 is open."

We thank the reviewer for this point and agree the k>2 case is important. We ran additional
experiments on the multi-retriever simplex. Fusing **BM25 + RM3 + Qwen3** (k=3)
and re-running our two strongest k=2 methods against a k=3 RRF baseline, query-adaptive weighting
continues to beat RRF on both large benchmarks (MSMARCO and NQ, Wilcoxon p < 0.001), and the
per-query oracle ceiling *rises* when the third retriever is added (+0.009 to +0.029 absolute over
the best k=2 pair). So the k=2 study is a lower bound on achievable headroom, and the approach is
not restricted to a single sparse–dense pair. Recovering that headroom does get harder at k=3
(the simplex has ~5151 weight settings vs. 101 at k=2, so per-query optima scatter and no longer
collapse into a compact learnable interval).
Full k=3 tables will be added to the appendix.

See our reply to Reviewer reaf for the full breakdown.

---

## C1 — Practical implications of the interval-valued formulation

<!-- TODO: pending decision — one rebuttal paragraph vs. new revision subsection. Candidate
implications: report/tune weight ranges not points; interval width as confidence/abstention
signal; skip per-query prediction on RRF-easy queries (0.5 ∈ I_q); tiered framework already
operationalizes this. -->

---

## C2 — Writing / typos

We thank the reviewer for catching these. We have fixed the specific instances noted and have done a
full proofreading pass to correct the missing articles.
