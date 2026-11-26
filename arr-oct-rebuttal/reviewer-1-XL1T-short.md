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

**W1.** Thank you for raising this interesting point. We do not merge disjoint intervals in training or in prediction. Trained models use the widest interval, $I_q$, as the target. The LLM prompt and the oracle use all intervals. One interval is enough: all intervals give the same best score, so a prediction inside the widest one is already optimal. It is also the easiest target: on the evaluation queries, it holds about 70% of the optimal weights of such a query on average. And such queries are rare (Table 3). Using all intervals in the loss is a natural next step. We will add this explanation to Section 3.

**W2 and S1.** Thank you for this helpful suggestion. Our conclusions stay the same at top-3 and top-5 (Table B). Top-10 is a widely used cutoff for these benchmarks.

- **MS MARCO and NQ:** every method beats standard RRF at both cutoffs. The best method gains .041 (top-3) and .043 (top-5) on MS MARCO and .068 on NQ at both cutoffs, compared with .043 and .067 at top-10. The gains are significant in 70 of 72 settings at top-3 and 69 of 72 at top-5 (paired t-test, p < .05).
- **ACORD and NFCorpus:** the pattern is the same as at top-10 (see W4).

**Table B:** ACORD and NFCorpus encoder rows use the training settings in our reply to Reviewer 25bQ (W5). DAT rows are from our rerun (see our general response).

|Method|MS MARCO MRR@3|MRR@5|NQ MRR@3|MRR@5|ACORD nDCG@3|nDCG@5|NFCorpus nDCG@3|nDCG@5|
|---|---|---|---|---|---|---|---|---|
|Oracle|.414|.433|.492|.510|.242|.230|.518|.482|
|Standard RRF|.245|.265|.310|.329|.119|.123|.430|.401|
|Mean Optimal Weight|.260|.283|.339|.359|.118|.125|.435|.406|
|Ridge Regressor|.261|.284|.339|.359|.122|.126|.435|.406|
|RoBERTa (IW)|.265|.289|.342|.362|.130|.131|.435|.407|
|ModernBERT (IW)|.270|.293|.346|.367|.130|.131|.436|.407|
|ModernBERT Passage Cond.|.271|.294|.354|.375|.135|.129|.435|.407|
|DAT Qwen3|.257|.279|.344|.361|.135|.140|.434|.411|
|DAT Ministral3|.275|.297|.375|.394|.117|.131|.433|.408|
|Qwen3 FS (IW)|.276|.298|.360|.380|.120|.132|.436|.412|
|Ministral3 FS (IW)|.286|.308|.378|.397|.129|.140|.427|.404|

**W3.** Thank you for raising this point. We used k = 60 because it is the default value of RRF. Cormack et al. (2009) report that k = 60 "was near-optimal, but that the choice was not critical". We also tested k = 10, 20, 60 and 100 (Table C). Our conclusions stay the same.

- **In our experiments, a smaller k helps RRF.** Bruch et al. (2023) also find RRF sensitive to its parameters.
- **The predicted weights add to this gain.** On MS MARCO and NQ, all seven of our methods stay above RRF on average at every k, and the gain is significant in at least 51 of 56 settings. For the method in Table C, it is +.026 to +.034 on MS MARCO and +.040 to +.048 on NQ. So k and the weight give two separate gains.

**Table C:** MRR@10 and nDCG@10. PC is ModernBERT Passage Cond., with the settings of Table B. The runs were regenerated for this sweep, so k = 60 can differ from Table 2 in the third decimal.

|k|MS MARCO RRF|PC|NQ RRF|PC|ACORD RRF|PC|NFCorpus RRF|PC|
|---|---|---|---|---|---|---|---|---|
|10|.300|.331|.373|.418|.138|.132|.376|.385|
|20|.292|.326|.362|.410|.137|.134|.377|.383|
|60|.282|.313|.343|.388|.135|.136|.372|.378|
|100|.280|.306|.337|.377|.133|.137|.370|.376|

**W4.** Thank you for this comment. The intervals hold on all four datasets; only the size of the gain differs. Section 5 gives most of the reasons; we will collect them in one paragraph:

- **Headroom:** MS MARCO and NQ have about one relevant document per query (MRR@10), so moving it changes the score a lot. ACORD and NFCorpus have 30 to 40 on average (nDCG@10), so one change in the ranking moves the score less.
- **Significance:** it follows the size of the gain and the number of queries: 6,980 on MS MARCO and 2,893 on NQ, but 323 on NFCorpus and 57 on ACORD. Both are specialized-domain datasets, and ACORD is expert-annotated and naturally small.
- **Many queries cannot gain:** for 18 to 30% of queries on MS MARCO, NQ and NFCorpus, no weight brings a relevant document into the top 10. For another 24 to 38% on MS MARCO and NQ, the weight 0.5 is already a best weight.
- **Narrow targets:** on the other queries, the optimal interval is 2 to 9 times narrower (Table 6).
- **Hard to predict:** on MS MARCO and NQ, none of 15 query features explains more than about 1% of the errors.
- **Mean Optimal Weight already helps:** on MS MARCO and NQ, it captures about half of the best method's gain. Per-query methods still improve on it: 49 of 128 comparisons are significant after FDR correction.

Our code is at the link in the abstract. Thank you again for your helpful comments.
