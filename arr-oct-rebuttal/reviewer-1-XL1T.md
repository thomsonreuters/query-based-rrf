# Reviewer 1 (XL1T)

Submission 1609, ARR October. Confidence 3, Soundness 3, Excitement 3, Overall 3 (Findings).

Quoted text is the reviewer's own wording. Headings are short labels added for reference.

## Weaknesses

### W1. Disjoint intervals are merged, not kept as a complete set

> The paper considers existing methods for weight prediction and merges multiple disjoint intervals. The author did not explain why they did not consider the complete set of optimal intervals.

### W2. Evaluation only at top-10

> The evaluations focus only on top-10 results.

### W3. No sensitivity analysis for the smoothing constant k

> The smoothing constant k in Equation 1 is fixed at 60. The paper did not present any sensitivity analysis for k.

### W4. No explanation of why dynamic weighting gives limited gains

> Even though the weight intervals dominate across all datasets, the headroom and statistical significance vary substantially across datasets. The paper did not discuss why dynamic weighting offers limited gains.

## Suggestions

### S1. Report smaller cutoffs (top-3, top-5)

> The effectiveness of fusion strategies may vary with the retrieval depth. Reporting performance at smaller cutoffs, such as top-3 and top-5, would provide a more comprehensive understanding

Relates to W2.

---

# Response to Reviewer XL1T (draft)

## Opening

Dear Reviewer XL1T:

Thank you very much for your time and for your careful review. We are glad that you find our observation interesting: optimal fusion weights often form intervals, not single points. We are particularly encouraged by your recognition of our analysis across 4 datasets and 4 sparse-dense settings, and of the latency spectrum that helps balance quality and cost. Below, we address your comments point by point.

## W1

> **W1:** The paper considers existing methods for weight prediction and merges multiple disjoint intervals. The author did not explain why they did not consider the complete set of optimal intervals.

Thank you for raising this and giving us an opportunity to clarify how we handle queries with more than one optimal interval.

We read "disjoint intervals" as the case in Section 3 where the best weights of one query form two or more separate ranges. For example, one ACORD query reaches its best score for weights in [0.30, 0.37] and also in [0.50, 1.00].

We do not merge these ranges in training or in prediction. Each method uses them as follows:

- **Trained models (regression and encoders):** we keep only the widest range, $I_q$, as the target. In the example, this is [0.50, 1.00].
- **LLM methods:** the prompt shows all ranges of each example query.
- **Oracle:** it is not affected, because every weight in every range gives the same best score.

We keep one range for the trained models for two reasons. First, such queries are rare: under 5% of queries in most settings. Second, there is no standard loss for a target with several separate ranges. The natural choice is not convex, so it is harder to train.

Thank you again for this helpful comment. We will add this explanation to Section 3 in the revised version. If you meant something else by "merge", we are glad to clarify.

## W2 and S1

> **W2:** The evaluations focus only on top-10 results.
>
> **S1:** The effectiveness of fusion strategies may vary with the retrieval depth. Reporting performance at smaller cutoffs, such as top-3 and top-5, would provide a more comprehensive understanding

Thank you for this helpful suggestion. We used top-10 in the paper because it is the standard cutoff for these benchmarks, for example MRR@10 on MS MARCO and nDCG@10 in BEIR. Following your comment, we also evaluated the methods at top-3 and top-5.

The setup is the same as in Table 2. We only change the cutoff of the metric. No model is retrained: each method keeps the weight it predicted from top-10 labels.

**Table A:** Results at top-3 and top-5, averaged over the four retriever pairs as in Table 2. Best method per column in bold.

| Method | MS MARCO MRR@3 | MS MARCO MRR@5 | NQ MRR@3 | NQ MRR@5 | ACORD nDCG@3 | ACORD nDCG@5 | NFCorpus nDCG@3 | NFCorpus nDCG@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Oracle (per-query best w) | .414 | .433 | .492 | .510 | .242 | .230 | .518 | .482 |
| Headroom (Oracle − RRF) | .169 | .168 | .182 | .181 | .123 | .106 | .088 | .081 |
| Standard RRF | .245 | .265 | .310 | .329 | .119 | .123 | .430 | .401 |
| Mean Optimal Weight | .260 | .283 | .339 | .359 | .118 | .125 | .435 | .406 |
| Ridge Regressor (MBW) | .261 | .284 | .339 | .359 | .122 | .126 | .435 | .406 |
| RoBERTa Regression (IW) | .265 | .289 | .342 | .362 | .130 | .137 | .434 | .407 |
| ModernBERT (IW) | .270 | .293 | .346 | .367 | .129 | .136 | **.438** | .411 |
| ModernBERT Passage Cond. | .271 | .294 | .354 | .375 | **.135** | .139 | .436 | .410 |
| Qwen3 FS (IW) | .276 | .298 | .360 | .380 | .120 | .132 | .436 | **.412** |
| Ministral3 FS (IW) | **.286** | **.308** | **.378** | **.397** | .129 | **.140** | .427 | .404 |

As shown in Table A, our conclusions stay the same at top-3 and top-5:

- **MS MARCO and NQ:** every method beats standard RRF at both cutoffs. The order of the methods is the same as at top-10. The best method gains .041 (top-3) and .043 (top-5) on MS MARCO, compared with .043 at top-10. On NQ it gains .068 at both cutoffs, compared with .067 at top-10. The gains over RRF are significant in 54 of 56 settings at top-3 and in 53 of 56 at top-5 (paired t-test, p < .05).
- **ACORD and NFCorpus:** the pattern is the same as at top-10. The best method beats standard RRF at both cutoffs, with smaller gains than on MS MARCO and NQ.
- **Headroom:** the gap between the oracle and RRF does not shrink. It stays about the same on MS MARCO and NQ, and it grows on ACORD (.123 at top-3, .080 at top-10) and NFCorpus (.088 at top-3, .066 at top-10).
- **Intervals:** our main finding also holds. At top-3 and top-5, the best weight is a single point for fewer than 10% of queries in every setting.

Thank you again for this helpful suggestion. We will add these results, with the two DAT baselines, to the revised paper. If you would like to see other metrics or cutoffs, such as Recall@100 or nDCG@100, we are happy to include them as well.

## W3

> **W3:** The smoothing constant k in Equation 1 is fixed at 60. The paper did not present any sensitivity analysis for k.

Thank you for raising this point. We used k = 60 because it is the default value set by the authors of RRF (Cormack et al., 2009), and it is the value that later work and most systems use. The effect of this constant has also been studied in detail by Bruch et al. (2023), which we cite. Our paper studies the fusion weight, so we kept k at its standard value.

Following your comment, we also repeated the weight sweep with k = 10, 20, 60 and 100.

The setup is the same as in the paper. We only change k. No model is retrained: each method keeps the weight it predicted with k = 60. For MS MARCO and NQ, this analysis covers the two MiniLM pairs, BM(M) and RM(M).

**Table B:** Results for different values of k. Each value is an average over the retriever pairs.

| Dataset | Row | k = 10 | k = 20 | k = 60 | k = 100 |
| --- | --- | --- | --- | --- | --- |
| MS MARCO (MRR@10) | Standard RRF | .295 | .289 | .280 | .277 |
| | ModernBERT Passage Cond. | .314 | .310 | .300 | .295 |
| | Oracle (per-query best w) | .412 | .425 | .440 | .443 |
| | Headroom (Oracle − RRF) | .117 | .136 | .160 | .166 |
| NQ (MRR@10) | Standard RRF | .357 | .350 | .334 | .328 |
| | ModernBERT Passage Cond. | .390 | .383 | .366 | .356 |
| | Oracle (per-query best w) | .468 | .481 | .497 | .500 |
| | Headroom (Oracle − RRF) | .110 | .131 | .162 | .172 |
| ACORD (nDCG@10) | Standard RRF | .138 | .137 | .135 | .133 |
| | ModernBERT Passage Cond. | .145 | .148 | .148 | .145 |
| | Oracle (per-query best w) | .199 | .203 | .215 | .217 |
| | Headroom (Oracle − RRF) | .061 | .066 | .080 | .084 |
| NFCorpus (nDCG@10) | Standard RRF | .376 | .377 | .372 | .370 |
| | ModernBERT Passage Cond. | .384 | .383 | .379 | .377 |
| | Oracle (per-query best w) | .431 | .435 | .439 | .440 |
| | Headroom (Oracle − RRF) | .055 | .058 | .067 | .069 |

As shown in Table B, our conclusions stay the same for every k:

- **k has a small effect on standard RRF.** A smaller k helps a little. From k = 60 to k = 10, RRF gains .015 on MS MARCO, .023 on NQ, .003 on ACORD and .004 on NFCorpus.
- **The headroom from the weight stays large.** At every k, the gap between the oracle and RRF is at least .117 on MS MARCO, .110 on NQ, .061 on ACORD and .055 on NFCorpus. This is several times larger than the effect of k.
- **Predicted weights still help.** The same predicted weights beat standard RRF at every k on every dataset, without retraining.
- **Intervals:** our main finding also holds. On MS MARCO and NQ, the best weight is a single point for about 2% of queries or fewer at every k.

Thank you again for this helpful comment. We will add this analysis to the revised paper.

## W4

> **W4:** Even though the weight intervals dominate across all datasets, the headroom and statistical significance vary substantially across datasets. The paper did not discuss why dynamic weighting offers limited gains.

Thank you for this comment. We discuss these reasons in several places in Section 5. We bring them together here.

**Why the headroom and the significance differ across datasets**

- **Headroom (Section 5.1):** it is largest on MS MARCO and NQ (0.15 to 0.22) and smaller on ACORD and NFCorpus (0.05 to 0.09). MS MARCO and NQ have about one relevant document per query and use MRR@10, so moving that one document changes the score a lot. ACORD and NFCorpus have about 30 relevant documents per query and use nDCG@10, so a change in the weight moves the score less.
- **Significance (Section 5.1):** it follows the headroom and the number of test queries. It is easy to detect on MS MARCO (6,980 queries) and NQ (2,893 queries). It is hard to detect on ACORD (57 queries) and NFCorpus (323 queries), where both are small.

**Why dynamic weighting recovers only part of the headroom**

1. **Unreachable queries (Section 5.2):** for 18 to 30% of queries on MS MARCO, NQ and NFCorpus, no weight brings a relevant document into the top 10. No method can help here.
2. **Queries where RRF is already optimal (Section 5.2):** on MS MARCO and NQ, the weight 0.5 is already a best weight for 24 to 38% of queries. A predictor can only match RRF here. Together with point 1, this is about half or more of the queries on these two datasets.
3. **Narrow targets (Section 5.2, Table 6):** on the remaining queries, the optimal interval is several times narrower. These are the queries where a predictor must move away from 0.5, and they are the hardest to hit.
4. **The weight is hard to predict from the query (Section 5.2):** in our error analysis, none of 15 query features explains more than about 1% of the errors. Our interval loss also closes only a small part of the gap. So the limit is predicting the interval, not the loss.
5. **A single tuned weight is already strong (Section 5.3):** on MS MARCO and NQ, the best weights are skewed toward the dense retriever. A dataset-level constant captures much of the gain, which leaves less for per-query methods.

Thank you again for this comment. In the revised paper, we will collect these points into one paragraph in Section 5, so that they are easier to find.
