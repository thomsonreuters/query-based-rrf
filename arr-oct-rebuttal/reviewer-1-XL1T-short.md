# Reviewer 1 (XL1T)

Submission 1609, ARR October. Confidence 3, Soundness 3, Excitement 3, Overall 3 (Findings).

Quoted text is the reviewer's own wording. Headings are short labels added for reference.

## Weaknesses

### W1. Disjoint intervals are merged, not kept as a complete set

> The paper considers existing methods for weight prediction and merges multiple disjoint intervals. The author did not explain why they did not consider the complete set of optimal intervals.

(interesting idea, it's less common, so we take take widest interval; no exact share in the reply, the exact count reaches 8.8%)

### W2. Evaluation only at top-10

> The evaluations focus only on top-10 results.

### W3. No sensitivity analysis for the smoothing constant k

> The smoothing constant k in Equation 1 is fixed at 60. The paper did not present any sensitivity analysis for k.

### W4. No explanation of why dynamic weighting gives limited gains

> Even though the weight intervals dominate across all datasets, the headroom and statistical significance vary substantially across datasets. The paper did not discuss why dynamic weighting offers limited gains.

(significant in large dataset, small dataset in specific domain, create by experts so small number of samples)

## Suggestions

### S1. Report smaller cutoffs (top-3, top-5)

> The effectiveness of fusion strategies may vary with the retrieval depth. Reporting performance at smaller cutoffs, such as top-3 and top-5, would provide a more comprehensive understanding

Relates to W2.

---

# Response to Reviewer XL1T

Dear Reviewer XL1T:

Thank you very much for your careful review. We are glad that you find our interval observation interesting, and we appreciate your recognition of our analysis across 4 datasets and 4 sparse-dense settings and of the latency spectrum.

W1. Thank you for raising this interesting point. We do not merge disjoint intervals in training or in prediction. Trained models use the widest interval, I_q, as the target. The LLM prompt and the oracle use all intervals. One interval is enough because all intervals give the same best score, so a prediction inside the widest one is already optimal. It is also a good target because on the evaluation queries, it holds about 70% of the optimal weights of such a query on average. And such queries are rare (Table 3). We will add this explanation to Section 3.

W2 and S1. Thank you for this helpful suggestion. Our conclusions stay the same at top-3 and top-5. Top-10 is a widely used cutoff for these benchmarks. For example, the official MS MARCO metric is MRR@10, and BEIR uses nDCG@10.

- MS MARCO and NQ: all seven of our methods beat standard RRF at both cutoffs. On MS MARCO, standard RRF scores .245 (MRR@3) and .265 (MRR@5), and the best method, Ministral3 FS, scores .286 and .308. On NQ, standard RRF scores .310 and .329, and Ministral3 FS scores .378 and .397. So the best method gains .041 (top-3) and .043 (top-5) on MS MARCO and .068 on NQ at both cutoffs, compared with .043 and .067 at top-10. The gains are significant in 54 of 56 settings at top-3 and 53 of 56 at top-5 (paired t-test, p < .05).
- ACORD and NFCorpus: the pattern is the same as at top-10 (see W4).

W3. Thank you for raising this point. We used k = 60 because it is the default value of RRF. Cormack et al. (2009) report that k = 60 "was near-optimal, but that the choice was not critical". As you suggested, we also tested k = 10, 20, 60 and 100. Our conclusions stay the same.

- In our experiments, a smaller k helps RRF. On MS MARCO, standard RRF scores .300 at k = 10, .292 at k = 20, .282 at k = 60 and .280 at k = 100 (MRR@10). On NQ, it is .373, .362, .345 and .337. Bruch et al. (2023) also find RRF sensitive to its parameters.
- The predicted weights add to this gain. On MS MARCO and NQ, all seven of our methods stay above RRF on average at every k, and the gain is significant in at least 51 of 56 settings. For ModernBERT Passage Cond., it is +.026 to +.034 on MS MARCO and +.040 to +.048 on NQ. So k and the weight give two separate gains.

W4. Thank you for this comment. The intervals hold on all four datasets; only the size of the gain differs. Section 5 gives most of the reasons; we will collect them in one paragraph:

- Headroom: MS MARCO and NQ have about one relevant document per query (MRR@10), so moving it changes the score a lot. ACORD and NFCorpus have 30 to 40 on average (nDCG@10), so one change in the ranking moves the score less.
- Significance: it follows the size of the gain and the number of queries: 6,980 on MS MARCO and 2,893 on NQ, but 323 on NFCorpus and 57 on ACORD. Both are specialized-domain datasets, and ACORD is expert-annotated and naturally small.
- Many queries cannot gain: for 18 to 30% of queries on MS MARCO, NQ and NFCorpus, no weight brings a relevant document into the top 10. For another 24 to 38% on MS MARCO and NQ, the weight 0.5 is already a best weight.
- Narrow targets: on the other queries, the optimal interval is 2 to 9 times narrower (Table 6).
- Hard to predict: on MS MARCO and NQ, none of 15 query features explains more than about 1% of the errors.
- Mean Optimal Weight already helps: on MS MARCO and NQ, it captures about half of the best method's gain. Per-query methods still improve on it: 46 of 128 comparisons are significant after FDR correction.

This response gives the main numbers. The full tables will be in the revised paper. Our code is at the link in the abstract. Thank you again for your helpful comments.