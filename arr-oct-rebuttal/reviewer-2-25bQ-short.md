# Reviewer 2 (25bQ)

Submission 1609, ARR October. Confidence 4, Soundness 3, Excitement 3, Overall 3 (Findings).

Quoted text is the reviewer's own wording. Headings and the "Asks" lists are short condensations added for reference.

## Weaknesses

### W1. Abstract: "74 of 128 dynamic fusion methods" is inconsistent

> The numbers in the abstract are inconsistent: the abstract writes "74 of 128 dynamic fusion methods", but 128 = 8 methods × 16 configurations (F.4, L994–995), i.e. it is the number of tests rather than the number of methods, which contradicts "eight strategies" in the same sentence.

### W2. Small gain from query-level adaptivity; tuned static constant baseline missing

> The true increment of query-level adaptivity is small, and the strong baseline "tuned static constant" is missing. Averaged over the datasets in Table 2, MOW already recovers 12% (MS MARCO), 18% (NQ), 1% (ACORD) and 5% (NFCorpus) of the headroom, while the best dynamic method recovers 26% / 38% / 16% / 9%; that is, per-query adaptivity adds only +0.023 / +0.035 / +0.012 / +0.003 over "one constant per dataset". Moreover, MOW is "the mean of the per-query optimal weights", which is not the same as "the w* that maximises the dataset-average metric"; the latter is the strongest static WRRF after tuning.

Asks:
- Add the tuned static baseline: the single w* that maximises the dataset-average metric.

### W3. Interval-width numbers in the text disagree with Table 6

> The interval-width numbers are inconsistent with Table 6. (L476–479 and Table 6, p.13) The text says "intervals where 0.5 ∈ Iq are on average 4–11× wider", whereas the aggregate values in Table 6 are MS MARCO 4.7×, NQ 4.2×, NFCorpus 8.9× and ACORD 2.3×: the maximum is 8.9, not 11, and ACORD is below 4.

### W4. Interval finding does not invalidate point estimation; Lsat is a standard ε-insensitive loss

> The interval finding does not invalidate point estimation, and the equivalence of the proposed loss weakens the methodological novelty. The paper states that the framing of predicting a per-query weight as a point is "itself wrong" (L057–059), but when the optimal set is an interval, any maximiser is still a valid choice; interval optima show that point losses (e.g. MSE) give incomplete supervision, not that point estimation is invalid. Moreover, Lsat (Eq. 3) is exactly a squared ε-insensitive loss centred at the interval midpoint m = (l+r)/2 with ε = (r−l)/2, which is zero whenever |ŵ − m| ≤ ε. In other words, the "loss tailored to the interval structure" is a standard ε-insensitive loss rewritten. Suggested: (i) narrow the claim to "point losses supervise interval optima inefficiently" rather than "the point-estimation framing is wrong"; (ii) state the relation to the ε-SVR / interval-regression loss and what is new relative to it; (iii) if the main contribution is empirical characterisation, position it as such in the introduction and abstract.

Asks:
- (i) Narrow the claim to "point losses supervise interval optima inefficiently".
- (ii) State the relation of Lsat to the ε-SVR / interval-regression loss and what is new.
- (iii) If the main contribution is empirical characterisation, position it that way in the introduction and abstract.

### W5. Weak results on small datasets are confounded with under-training

> The weak results on the small datasets are confounded with the training budget. App. B.2 (L887–891) uses 500 warm-up steps, batch size 256 and at most 10 epochs, whereas the training splits in Table 4 contain only 51 queries (ACORD) and 2,590 (NFCorpus), i.e. at most about 10 and 110 parameter updates — both below the warm-up length, so training ends while still in warm-up. The small gains and sparse stars on ACORD/NFCorpus therefore cannot be attributed solely to query-level unpredictability; under-training is an equally plausible explanation. Please state whether warm-up, learning rate or step count were adjusted for the small datasets, and add a "sufficiently trained" control (e.g. a fixed floor of ≥500 steps, or training by steps rather than epochs).

Asks:
- State whether warm-up, learning rate or step count were adjusted for the small datasets.
- Add a "sufficiently trained" control (fixed floor of ≥500 steps, or training by steps rather than epochs).

### W6. Recovery ratio may not be comparable for DAT

> The recovery scale may not be comparable for DAT. The WRRF oracle bounds only what is attainable within a rank-based WRRF family. As described in its own paper, DAT scores the top-1 results of both channels and calibrates a weighting factor through "effectiveness score normalisation" (Hsu & Tzeng, 2025), i.e. its fusion rule differs from WRRF. If so, DAT's recovery ratio conflates weight-prediction quality with a change of fusion rule. Please state the exact fusion formula used for DAT here, and provide an oracle/reference matched to each fusion rule (or compare recovery only within the same family).

Asks:
- State the exact fusion formula used for DAT.
- Provide an oracle/reference matched to each fusion rule, or compare recovery only within the same family.

## Suggestions

### S1. Test retrieval-internal features for query-level predictability

> On "query-level unpredictability": §5.2 (L489–500) and F.6 use only 15 surface features (|r| ≤ 0.09), which can only rule out surface features, not retrieval-internal signals. I suggest trying a few retrieval-internal features: the score gap between the two channels, top-k overlap (Jaccard), the RRF score margin, and the number of documents exclusive to one channel — the fact that ModernBERT passage-conditioned outperforms query-only is precisely a hint that usable signal exists there.

### S2. Robustness of the interval structure to cutoff, k, and metric

> Whether the interval structure holds under candidate cutoff 200→400 and k = 10→100; the overlap of the optimal intervals for the same query under MRR@10 and nDCG@10.

### S3. Matched point-vs-interval comparison on the same encoder

> Matched point-vs-interval comparison within the same encoder: §3.5.2 (L268–273) and App. B.2 (L893) state that both the point (MSE) and the interval (Lsat) variants were implemented, yet only the "(IW)" rows appear in Table 2 and Table 7. Please add the point-vs-interval comparison on the same backbone and data; this is the direct test of whether interval information itself helps.

### S4. Add a cheap neighbour-aggregation baseline

> A cheap adaptive baseline is missing: the LLM pipeline already retrieves 5 + 5 neighbours from the training set (App. C.1), yet there is no baseline that directly aggregates the weight labels of those same neighbours (e.g. their mean/median w̄, or kNN regression). Such a baseline is needed to separate the value of retrieving similar queries from the value of LLM generation.

### S5. Report single-channel retrieval effectiveness

> Please report single-channel retrieval effectiveness (BM25 only / RM3 only; MiniLM only / Qwen3 only), so that readers can tell what a query with w→0 or w→1 actually means.

## Typos

### T1. Abstract wording (same issue as W1)

> The "128 dynamic fusion methods" in the abstract at L019–020 (see point 1) and "eight strategies" should be made consistent.

### T2. Table 1 caption does not give n

> The n in the Table 1 caption "Latency reported as mean with 95% CI over n within-dataset means" is not given.

---

# Response to Reviewer 25bQ (draft)

## Opening

Dear Reviewer 25bQ:

Thank you very much for your careful and detailed review, and for recognizing the interval view, our statistical testing and our candid analysis. Below, we give the conclusion and the main numbers for each point. The complete results will be in the revised paper. The conventions of the numbers are in our general response. Four numbers in a row are for MS MARCO / NQ / ACORD / NFCorpus.

## W1, W3, T1, T2 and S5

W1, W3, T1, T2 and S5. We will fix all of them. The abstract will say "method-setting comparisons" (128 = 8 methods × 16 settings). The text and the caption will match Table 6. The Table 1 caption will give n (16, or 32 where a sub-tier pools two methods). Table 2 will list each retriever alone.

## W2

W2 (tuned static constant). We added the tuned constant w*, the best single weight on the training queries. It is a strong static baseline, and the per-query oracle is still far above it. w* reaches .334 / .420 / .147 / .381. The oracle is .058 to .114 above it, and the best per-query methods are within .010 of it. So the gain from a per-query weight is there, and it is hard to predict (Section 5.2).

## W4

W4 (point estimation; ε-insensitive loss). We will follow all three suggestions. (i) and (iii): L057–059 will read "We show that optimal WRRF weights typically form intervals, not points, so point losses supervise them inefficiently." The abstract and the introduction will present the paper as an empirical characterisation. (ii): Section 3.5.2 (L274) introduces L_sat as "built on the ε-insensitive tube of SVR". We will add the exact form you give. Its one difference from the standard loss is that ε is not a fixed hyperparameter: each query has its own, given by its optimal interval I_q. We will also add the link to interval regression (Rigaill et al., 2013; Drouin et al., 2017).

## W5

W5 (training budget). Appendix B gives one general configuration and does not list the settings for each dataset. We will add them. The encoders on ACORD and NFCorpus use batch size 16 (8 for the passage-conditioned model), warm-up over the first 10% of the steps, and evaluation after every epoch. Training runs for 30 steps on ACORD (3 warm-up) and 1,150 to 1,190 on NFCorpus (115 to 119 warm-up), well past warm-up. We also trained by steps, as you suggest (1,000 steps, 100 warm-up, three seeds). RoBERTa and ModernBERT stay within .003 of Table 2 on both datasets. The passage-conditioned model scores .140 on ACORD (.148 in Table 2).

## W6

W6 (DAT recovery). We now report DAT inside its own family, because it uses its own fusion rule. DAT fuses scores, not ranks (Section 2): s(d) = α · s_sparse(d) + (1 − α) · s_dense(d), with min-max normalized retrieval scores. α = S_sparse / (S_sparse + S_dense), where S are the LLM scores (0 to 5) of the top-1 results (Eq. 6 and 7 of the DAT paper, with its special cases). Sweeping α from 0 to 1 gives an equal-weight baseline (α = 0.5) and a per-query oracle for this family. Against them, with the DAT scores of Table 2, DAT recovers −6 / 7 / 3 / −2 % (Qwen3) and 4 / 24 / −3 / −14 % (Ministral3) of its own headroom.

## S1

S1 (retrieval-internal features). These features add a small gain over the query text. We tested the four families you list and the top-1 score of each channel, on all 16 settings. A regressor on them is .007 (MS MARCO) and .011 (NQ) above the Ridge Regressor on query TF-IDF. On these two datasets, the strongest feature, the top-1 score of the dense channel, correlates with misprediction at |r| = .23 and .32. The surface features of Appendix F.6 have .09 at most.

## S2

S2 (depth, k, metric). Intervals stay the most common optimal set at depth 400 and at every k we tested. At k = 60, the optimal set is an interval (single or multi) for 70.8% to 76.3% of the queries at depth 200 (from Table 3) and for 70.8% to 75.0% at depth 400. For k = 10, 20 and 100, this share is 68% to 80%. MRR@10 and nDCG@10 share an optimal weight for 85.8% to 99.8% of the reachable queries.

## S3

S3 (point vs interval). With the same ModernBERT and only the loss changed, the interval variant is .006 above MSE on MS MARCO and NQ (BM(Q) pair), and the difference is significant. Point 5 of our general response gives the details.

## S4

S4 (baseline on the neighbours' labels). The neighbours carry useful signal, mainly through their intervals. We used the same 5 + 5 neighbours (App. C.1) with no LLM call. On MS MARCO and NQ, the interval vote (the weight inside the optimal intervals of the most neighbours) scores .321 and .409. This is above the median of the neighbours' w̄ (.312 and .389) and Qwen3 FS (.315 and .396), and within .004 of Ministral3 FS (.325 and .412). So most of the gain of the few-shot methods over Mean Optimal Weight comes from the neighbours, and their intervals give more than their midpoints. This is in line with our interval view.

## Closing

Thank you again for your careful comments and suggestions.
