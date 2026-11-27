# Reviewer 3 (Sg6v)

Submission 1609, ARR October. Confidence 3, Soundness 2.5, Excitement 2.5, Overall 3 (Findings).

Quoted text is the reviewer's own wording. Headings are short labels added for reference.

## Weaknesses

### W1. No rigorous explanation of why plateaus appear

> The paper observes that optimal weights are intervals, but does not rigorously answer why plateaus appear.

(we don't provide a proof, characterization, nudged the weight, more than small jump)

### W2. QPP-driven adaptive fusion baselines are missing

> Prior work uses Query‑Performance‑Prediction (QPP) to adapt fusion strength. The paper only compares against DAT, ignoring existing QPP‑driven adaptive fusion approaches.

(usually used to find retriver perfor best for the query, inspiring by that, no direct comparision, still do QPP to pick retriveal, not compete, use either one to use a or b, intent detection, pick 2 out of 10, additional, separate pipeline stage. np net, bm25 which one to use)

### W3. Satisficing loss gives very limited empirical improvement

> The proposed satisficing loss only brings very limited empirical improvements.

## Suggestions

The reviewer's Comments, Suggestions and Typos section says only:

> W1-W3

That is, the reviewer listed no separate suggestions; the request is to address the three weaknesses above.

---

# Response to Reviewer Sg6v (draft)

## W1

**W1 (why plateaus appear).** Thank you for this question.

**Why plateaus exist.** This follows directly from the form of WRRF (Equation 1):

- The fused score of each document is a linear function of the weight $w$.
- So the score difference between any two documents is also linear in $w$. Two documents can swap order at most once as $w$ goes from 0 to 1.
- A query has a finite set of candidate documents, so there are finitely many swap points. Between two neighbouring swap points the ranking is fixed, so MRR@10 and nDCG@10 are fixed too.
- The metric can therefore change only at swap points: it is a step function of $w$. The best weights are whole flat pieces of this function, plus possibly single swap points where documents tie. These are the optimal intervals of Section 3.

This argument holds for any two retrievers and any metric computed from the ranking. It shows that plateaus must exist. It does not say how wide they are; that is what the paper measures (Tables 3 and 6).

**Why plateaus are often wide.** MRR@10 and nDCG@10 look only at the top 10. Most swaps happen between documents below the top 10, so they cannot change the metric. We counted this exactly, without the weight grid, for every reachable eval query. **Table I** gives the median per query, over the four retriever pairs:

|Dataset|Document pairs that swap|Distinct swap points|Swap points where the top-10 list changes|Swap points where the metric changes|
|---|---|---|---|---|
|MS MARCO|28,649|23,757|68|7|
|NQ|28,917|23,979|71|7|
|NFCorpus|21,918|19,544|54|12|
|ACORD|21,742|19,605|94|21|

- A typical query has about 20,000 swap points. Over all queries, the metric changes at 0.03% of the swap points on MS MARCO and NQ, 0.09% on NFCorpus and 0.12% on ACORD. The metric is flat everywhere else.
- Fewer metric changes go with wider intervals in Table 6: the median width is .220 on MS MARCO and .340 on NQ, against .075 on NFCorpus and .030 on ACORD.

The first part is exact. The second part is an empirical observation on our four datasets.

## W2

**W2 (QPP-driven adaptive fusion).** Thank you for this suggestion.

We added a QPP-driven fusion baseline on all 16 settings. It uses NQC (Shtok et al., 2012), a standard post-retrieval predictor, computed on the sparse list and on the dense list from the retrieval scores only. A linear fit on the training queries maps each NQC value to the expected score of that retriever. The sparse weight in WRRF is the sparse prediction divided by the sum of the two predictions.

**Table J:** MRR@10 / nDCG@10. On ACORD and NFCorpus, the ModernBERT row uses the training settings in our reply to Reviewer 25bQ (W5).

|Method|MS MARCO|NQ|ACORD|NFCorpus|
|---|---|---|---|---|
|Standard RRF|.282|.345|.135|.373|
|Mean Optimal Weight|.302|.377|.136|.376|
|DAT Qwen3|.289|.370|.143|.377|
|DAT Ministral3|.302|.396|.139|.370|
|NQC fusion (new)|.307|.395|.137|.373|
|ModernBERT Passage Cond.|.313|.392|.136|.378|
|Ministral3 FS (IW)|.325|.412|.145|.375|
|Oracle|.448|.522|.215|.439|

- **Level with DAT, with no LLM call.** NQC fusion improves over standard RRF in all 8 settings on MS MARCO and NQ (paired t-test, p < .001). It stays at the RRF level on ACORD and NFCorpus. This is the same pattern as our methods in Table 2.
- **On every dataset, one of our methods is above NQC fusion:** Ministral3 FS on MS MARCO, NQ and ACORD, and passage-conditioned ModernBERT on NFCorpus.
- **Most of the headroom stays open.** NQC fusion recovers about 15% of the headroom on MS MARCO and 28% on NQ. So a retrieval-score signal also leaves the gap to the oracle. This supports our main finding: the optimal weight of a query is hard to predict, and that is the open problem.
- **Other predictors.** WIG and SMV give similar results.

Related Work will also cover per-query strategy selection (Arabzadeh et al., 2021) and QPP-weighted fusion (Santra et al., 2026).

## W3

**W3 (satisficing loss).** Thank you for raising this point.

We use the satisficing loss because it fits the target. The best weight of a query is a range, not a single value. MSE penalizes a prediction inside this range, although it is already optimal. The satisficing loss does not. It adds no hyperparameter and no training cost.

Table A in our general response compares it with MSE on the same ModernBERT, with only the loss changed:

- On MS MARCO and NQ, the satisficing loss is .006 higher, and the difference is significant.
- The gain is small next to the headroom. On MS MARCO, MSE recovers 18% of the headroom, and the satisficing loss 21%. On NQ, it is 19% and 23%.

We do not present the loss as a source of large gains. Section 5.2 reports that it "closes only a small fraction of the recovery gap" (Table 7). The main limit is how well the optimal range can be predicted from the query, not the choice of loss.

Thank you again for your helpful comments.
