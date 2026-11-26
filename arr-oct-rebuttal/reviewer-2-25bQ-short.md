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

Thank you very much for your careful and detailed review. We are encouraged by your recognition of the interval view, our statistical testing, and our detailed and candid analysis of the results. Below, we address your comments point by point. The conventions of the tables are in our general response.

## W1, W3, T1, T2 and S5 (paper updates, no new experiment)

**W1, W3, T1, T2 and S5.** Thank you for reading the paper so carefully and for pointing these out. We will fix all of them in the revised paper:

- **W1 and T1 (abstract):** "dynamic fusion methods" becomes "method-setting comparisons" (128 = 8 methods × 16 settings).
- **W3 (interval width):** the text and the caption will match Table 6.
- **T2 (Table 1 caption):** n = 16 (4 datasets × 4 retriever pairs), or 32 where a sub-tier pools two methods.
- **S5 (single retrievers):** we will add each retriever alone to Table 7.

## W2

**W2 (tuned static constant).** Thank you for this careful point and for the concrete suggestion. We added the tuned constant w* (Table D). It is a strong static baseline, and the per-query oracle is still far above it.

**Table D:** w* is the single weight with the best average metric on the training queries.

|Method|MS MARCO|NQ|ACORD|NFCorpus|
|---|---|---|---|---|
|Standard RRF|.282|.345|.135|.373|
|Mean Optimal Weight|.302|.377|.136|.376|
|Tuned constant w*|.334|.420|.147|.381|
|Oracle (per-query best w)|.448|.522|.215|.439|

- **Per-query weights still have room:** the oracle is .058 to .113 above w* on every dataset. The best per-query methods are within .010 of w*. So the gain from a per-query weight is there, and it is hard to predict (Section 5.2).

## W4

**W4 (point estimation; ε-insensitive loss).** Thank you for these three clear suggestions. We will follow all three in the revised paper:

- **(i) and (iii) The claim and the positioning:** L057–059 will read "We show that optimal WRRF weights typically form intervals, not points, so point losses supervise them inefficiently." The abstract and the introduction will present the paper as an empirical characterisation of the optimal fusion weight.
- **(ii) The loss:** Section 3.5.2 (L274) introduces $\mathcal{L}_{\text{sat}}$ as "built on the ε-insensitive tube of SVR" (Smola and Schölkopf, 2004). In the form you give, the one difference from the standard loss is ε: it is not a fixed hyperparameter. Each query has its own, given by its optimal interval $\mathcal{I}_q$. We will add the exact form you give, and the link to interval regression (Rigaill et al., 2013; Drouin et al., 2017), to that section.

## W5

**W5 (training budget).** Thank you for this careful analysis and for the concrete suggestion. Appendix B gives one general configuration and does not list the settings for each dataset. We will add them. The encoders on ACORD and NFCorpus use batch size 16 (8 for the passage-conditioned model), warm-up over the first 10% of the training steps, and evaluation after every epoch. The other settings are as in Appendix B.

- **Steps:** training runs for 30 steps on ACORD (3 warm-up) and 1,150 to 1,190 on NFCorpus (115 to 119 warm-up). The passage-conditioned model has twice as many.
- **Results:** RoBERTa, ModernBERT and the passage-conditioned model score .137 / .139 / .136 on ACORD and .376 / .377 / .378 on NFCorpus. Standard RRF is .135 and .373.

## W6

**W6 (DAT recovery).** Thank you for this careful point. We now report DAT inside each family (Table E), because it uses its own fusion rule. On MS MARCO and NQ, the two views agree within 4 points.

- **Formula:** DAT fuses scores, not ranks (Section 2): $s(d) = \alpha\,\tilde{s}_{\text{sparse}}(d) + (1-\alpha)\,\tilde{s}_{\text{dense}}(d)$. The retrieval scores are min-max normalized per query. $\alpha = S_{\text{sparse}} / (S_{\text{sparse}} + S_{\text{dense}})$, where the two $S$ are the LLM scores (0 to 5) of the top-1 results, with the special cases of the DAT paper (its Eq. 6 and 7).
- **Matched references:** for DAT's rule, we swept α from 0 to 1. This gives an equal-weight baseline (α = 0.5) and a per-query oracle for its family. We also used DAT's α as the weight in WRRF, so that it shares the oracle of all other methods.

**Table E:** recovery of DAT (%), from the averages. DAT was rerun for this analysis; our general response gives the change from Table 2.

|Recovery (%)|MS MARCO|NQ|ACORD|NFCorpus|
|---|---|---|---|---|
|DAT Qwen3, own rule and own oracle|−1|7|6|5|
|DAT Qwen3, weight used in WRRF|0|7|8|4|
|DAT Ministral3, own rule and own oracle|13|33|−8|−2|
|DAT Ministral3, weight used in WRRF|16|33|3|5|

For DAT's family, the baseline is .297 / .360 / .141 / .378 and the oracle is .432 / .507 / .205 / .434.

## --- Part 2 of 2 (second comment) ---

## S1

**S1 (retrieval-internal features).** Thank you for this suggestion. Retrieval-internal features add a small gain over the query text (Table F). We tested the four feature families you list, and the top-1 score of each channel (standardized within its list), on all 16 settings.

**Table F:** the new row is a regressor on these features only.

|Method|MS MARCO|NQ|ACORD|NFCorpus|
|---|---|---|---|---|
|Ridge Regressor (MBW), query TF-IDF (Table 2)|.303|.377|.137|.376|
|Retrieval features (new)|.310|.388|.138|.379|

- **As a predictor:** the gain over the query text is .007 on MS MARCO and .011 on NQ. The oracle (Table D) is .138 and .134 above the new row.
- **Correlation with misprediction (as in Appendix F.6):** on MS MARCO and NQ, the strongest feature is the top-1 score of the dense channel, with |r| = .23 and .32 (mean over the retriever pairs). The strongest of the four you list has .13 and .23. The surface features have .09 at most.

## S2

**S2 (depth, k, metric).** Thank you for this suggestion. Intervals stay the most common optimal set at depth 400 and at every k we tested. Under the two metrics, the optimal sets of a query overlap in most cases (Table G).

**Table G:** k = 60, over the four retriever pairs. An interval is a single or multi-interval optimal set. Both depths are cut from the same depth-400 runs, so depth 200 can differ slightly from Table 3.

| |MS MARCO|NQ|ACORD|NFCorpus|
|---|---|---|---|---|
|Interval, depth 200 (% of queries)|70.9|74.6|76.3|73.1|
|Interval, depth 400 (% of queries)|70.8|74.7|75.0|73.1|
|Depths 200 and 400 share an optimal weight (% of reachable queries)|99.5|99.6|92.4|97.6|
|MRR@10 and nDCG@10 share an optimal weight (% of reachable queries)|99.8|99.0|85.8|96.1|

- **k:** for k = 10, 20 and 100, the share of intervals is 68% to 80%.
- **Metric:** when the two sets share a weight, the nDCG@10 set lies inside the MRR@10 set. With one relevant document, the two sets are the same.

## S3

**S3 (point vs interval).** Thank you for this suggestion. The interval variant is higher on all four datasets, and the difference is significant on MS MARCO and NQ (Table A in our general response).

## S4

**S4 (baseline on the neighbours' labels).** Thank you for this suggestion. The neighbours carry useful signal, mainly through their intervals. Ministral3 FS stays the highest row on MS MARCO, NQ and ACORD (Table H).

**Table H:** the kNN rows retrieve the 5 + 5 neighbours as in App. C.1 and make no LLM call.

|Method|MS MARCO|NQ|ACORD|NFCorpus|
|---|---|---|---|---|
|kNN, mean of $\bar{w}$|.306|.382|.138|.377|
|kNN, median of $\bar{w}$|.312|.389|.138|.377|
|kNN, interval vote|.321|.409|.138|.378|
|Qwen3 FS (IW)|.315|.396|.137|.379|
|Ministral3 FS (IW)|.325|.412|.145|.375|

- **Mean and median of $\bar{w}$:** on MS MARCO and NQ, they are .004 to .012 above Mean Optimal Weight (Table D). Both LLM methods are significantly above them (paired t-test on the average over the four pairs, p < .05).
- **Interval vote:** this rule predicts the weight that lies in the optimal intervals of the most neighbours. On MS MARCO and NQ, it is above Qwen3 FS and within .004 of Ministral3 FS.
- **What this separates:** on MS MARCO and NQ, most of the gain of the few-shot methods over Mean Optimal Weight comes from the neighbours. Their intervals give more than their midpoints (.321 against .312, and .409 against .389), which is in line with our interval view.

## Closing

Thank you again for your careful comments and suggestions.
