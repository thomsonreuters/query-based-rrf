## reviewer 1
Official Review of Submission1609 by Reviewer XL1T
Official Reviewby Reviewer XL1T16 Nov 2026, 13:02 (modified: 24 Nov 2026, 09:47)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer XL1T, AuthorsRevisions
Paper Summary:
The paper studies per-query Weighted Reciprocal Rank Fusion (WRRF) methods for combining sparse and dense retrieval results. For each query, the authors find an optimal interval of weights instead of having a single optimal fusion weight per query. The paper considers four strategies for predicting fusion weights -- dataset-level constant weights, linear regression, fine-tuned encoder models, and in-context learning with LLMs, and compares against standard RRF, Dynamic Alpha Tuning (DAT), and an oracle that selects the best weight for each query. The paper evaluates the WRRF on four benchmarks -- MS MARCO, Natural Questions, NFCorpus, and ACORD—using four combinations of two sparse and two dense retrievers. The experiments also examine the trade-off between retrieval effectiveness and inference latency. The results show that optimal weights frequently form intervals rather than unique points. Dynamic fusion improves over standard RRF in many settings, although the gains vary across datasets and prediction methods.

Summary Of Strengths:
The paper presents an interesting observation that optimal fusion weights may form intervals rather than unique points.
The results and model behavior are rigorously analysed and supported by sound reasoning using 4 datasets and 4 sparse-dense hybrid model settings.
The paper categorizes static and dynamic fusion strategies into a measured latency spectrum that guides balancing the performance and inference latency budgets.
Summary Of Weaknesses:
The paper considers existing methods for weight prediction and merges multiple disjoint intervals. The author did not explain why they did not consider the complete set of optimal intervals.
The evaluations focus only on top-10 results.
The smoothing constant k in Equation 1 is fixed at 60. The paper did not present any sensitivity analysis for k.
Even though the weight intervals dominate across all datasets, the headroom and statistical significance vary substantially across datasets. The paper did not discuss why dynamic weighting offers limited gains.
Comments Suggestions And Typos:
The effectiveness of fusion strategies may vary with the retrieval depth. Reporting performance at smaller cutoffs, such as top-3 and top-5, would provide a more comprehensive understanding

Confidence: 3 =  Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design.
Soundness: 3 = Acceptable: This study provides sufficient support for its main claims. Some minor points may need extra support or details.
Excitement: 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.
Overall Assessment: 3 = Findings: I think this paper could be accepted to the Findings of the ACL.
Ethical Concerns:
There are no concerns with this submission

Needs Ethics Review: No
Reproducibility: 4 = They could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.
Datasets: 1 = No usable datasets submitted.
Software: 1 = No usable software released.
Knowledge Of Or Educated Guess At Author Identity: No, I do not have even an educated guess about author identity
Publication Ethics Policy Compliance: I used a privacy-preserving tool exclusively for the use case(s) approved by PEC policy, such as language edits


## reviewer 2
Official Review of Submission1609 by Reviewer 25bQ
Official Reviewby Reviewer 25bQ08 Nov 2026, 20:25 (modified: 24 Nov 2026, 09:47)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer 25bQ, AuthorsRevisions
Paper Summary:
This paper studies the fusion weight of Weighted Reciprocal Rank Fusion (WRRF) in hybrid information retrieval. Standard RRF applies a fixed weight to all retrieval channels; recent query-aware methods treat the "per-query optimal fusion weight" as a single scalar (a point estimate) to be predicted. The authors question this assumption and show empirically, through a systematic study, that the per-query optimal fusion weight is usually an interval rather than a point.

On 4 datasets × 4 sparse–dense retriever pairs, with the per-query oracle as the upper bound, the authors evaluate 8 classes of dynamic weight-prediction strategies, reporting absolute quality, the recovery rate of the headroom over standard RRF, and the per-query inference latency.

Main findings: (1) optimal weights are mostly intervals, and can be multi-interval; (2) 74 of 128 settings remain significantly better than standard RRF after Benjamini–Hochberg FDR correction; (3) methods differ markedly in quality and latency, so the authors propose a tiered decision framework (T0–T3) that maps practical constraints to an appropriate method.

Summary Of Strengths:
The paper characterises the optimal fusion weight as a "set-valued interval object" and exposes the intrinsic mismatch of the query-aware point-estimation framing — a prediction that falls inside the optimal interval is still penalised by a point loss such as MSE. This framing has direct implications for subsequent work on fusion weights.

The paper uses paired t-tests with BH-FDR correction (128 tests) and reports mean latency with 95% t-confidence intervals; the 0.01 vs 0.001 grid-resolution sensitivity analysis rules out the possibility that the interval conclusion is an artefact.

The paper provides a four-way anatomy of the optimal weight set, a decomposition of unreachable queries, interval-width statistics, an error analysis, and a three-retriever extension, and candidly points out facts such as a median recovery of only 12.2% and frequently negative recovery under the low-capacity dense retriever.

Summary Of Weaknesses:
The numbers in the abstract are inconsistent: the abstract writes "74 of 128 dynamic fusion methods", but 128 = 8 methods × 16 configurations (F.4, L994–995), i.e. it is the number of tests rather than the number of methods, which contradicts "eight strategies" in the same sentence.

The true increment of query-level adaptivity is small, and the strong baseline "tuned static constant" is missing. Averaged over the datasets in Table 2, MOW already recovers 12% (MS MARCO), 18% (NQ), 1% (ACORD) and 5% (NFCorpus) of the headroom, while the best dynamic method recovers 26% / 38% / 16% / 9%; that is, per-query adaptivity adds only +0.023 / +0.035 / +0.012 / +0.003 over "one constant per dataset". Moreover, MOW is "the mean of the per-query optimal weights", which is not the same as "the w* that maximises the dataset-average metric"; the latter is the strongest static WRRF after tuning.

The interval-width numbers are inconsistent with Table 6. (L476–479 and Table 6, p.13) The text says "intervals where 0.5 ∈ Iq are on average 4–11× wider", whereas the aggregate values in Table 6 are MS MARCO 4.7×, NQ 4.2×, NFCorpus 8.9× and ACORD 2.3×: the maximum is 8.9, not 11, and ACORD is below 4.

The interval finding does not invalidate point estimation, and the equivalence of the proposed loss weakens the methodological novelty. The paper states that the framing of predicting a per-query weight as a point is "itself wrong" (L057–059), but when the optimal set is an interval, any maximiser is still a valid choice; interval optima show that point losses (e.g. MSE) give incomplete supervision, not that point estimation is invalid. Moreover, Lsat (Eq. 3) is exactly a squared ε-insensitive loss centred at the interval midpoint m = (l+r)/2 with ε = (r−l)/2, which is zero whenever |ŵ − m| ≤ ε. In other words, the "loss tailored to the interval structure" is a standard ε-insensitive loss rewritten. Suggested: (i) narrow the claim to "point losses supervise interval optima inefficiently" rather than "the point-estimation framing is wrong"; (ii) state the relation to the ε-SVR / interval-regression loss and what is new relative to it; (iii) if the main contribution is empirical characterisation, position it as such in the introduction and abstract.

The weak results on the small datasets are confounded with the training budget. App. B.2 (L887–891) uses 500 warm-up steps, batch size 256 and at most 10 epochs, whereas the training splits in Table 4 contain only 51 queries (ACORD) and 2,590 (NFCorpus), i.e. at most about 10 and 110 parameter updates — both below the warm-up length, so training ends while still in warm-up. The small gains and sparse stars on ACORD/NFCorpus therefore cannot be attributed solely to query-level unpredictability; under-training is an equally plausible explanation. Please state whether warm-up, learning rate or step count were adjusted for the small datasets, and add a "sufficiently trained" control (e.g. a fixed floor of ≥500 steps, or training by steps rather than epochs).

The recovery scale may not be comparable for DAT. The WRRF oracle bounds only what is attainable within a rank-based WRRF family. As described in its own paper, DAT scores the top-1 results of both channels and calibrates a weighting factor through "effectiveness score normalisation" (Hsu & Tzeng, 2025), i.e. its fusion rule differs from WRRF. If so, DAT's recovery ratio conflates weight-prediction quality with a change of fusion rule. Please state the exact fusion formula used for DAT here, and provide an oracle/reference matched to each fusion rule (or compare recovery only within the same family).

Comments Suggestions And Typos:
On "query-level unpredictability": §5.2 (L489–500) and F.6 use only 15 surface features (|r| ≤ 0.09), which can only rule out surface features, not retrieval-internal signals. I suggest trying a few retrieval-internal features: the score gap between the two channels, top-k overlap (Jaccard), the RRF score margin, and the number of documents exclusive to one channel — the fact that ModernBERT passage-conditioned outperforms query-only is precisely a hint that usable signal exists there.

Whether the interval structure holds under candidate cutoff 200→400 and k = 10→100; the overlap of the optimal intervals for the same query under MRR@10 and nDCG@10.

Matched point-vs-interval comparison within the same encoder: §3.5.2 (L268–273) and App. B.2 (L893) state that both the point (MSE) and the interval (Lsat) variants were implemented, yet only the "(IW)" rows appear in Table 2 and Table 7. Please add the point-vs-interval comparison on the same backbone and data; this is the direct test of whether interval information itself helps.

A cheap adaptive baseline is missing: the LLM pipeline already retrieves 5 + 5 neighbours from the training set (App. C.1), yet there is no baseline that directly aggregates the weight labels of those same neighbours (e.g. their mean/median w̄, or kNN regression). Such a baseline is needed to separate the value of retrieving similar queries from the value of LLM generation.

Please report single-channel retrieval effectiveness (BM25 only / RM3 only; MiniLM only / Qwen3 only), so that readers can tell what a query with w→0 or w→1 actually means.

Typos:

The "128 dynamic fusion methods" in the abstract at L019–020 (see point 1) and "eight strategies" should be made consistent.
The n in the Table 1 caption "Latency reported as mean with 95% CI over n within-dataset means" is not given.
Confidence: 4 = Quite sure. I tried to check the important points carefully. It's unlikely, though conceivable, that I missed something that should affect my ratings.
Soundness: 3 = Acceptable: This study provides sufficient support for its main claims. Some minor points may need extra support or details.
Excitement: 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.
Overall Assessment: 3 = Findings: I think this paper could be accepted to the Findings of the ACL.
Limitations And Societal Impact:
The authors discuss two points in the Limitations section (Section "Limitations"): only English benchmarks are evaluated (the shape of the intervals may change in multilingual settings), and LLM predictions are sampled at temperature 0.1, which causes slight variation in absolute numbers.

No societal-impact discussion is required.

Ethical Concerns:
There are no concerns with this submission

Needs Ethics Review: No
Reproducibility: 4 = They could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.
Datasets: 1 = No usable datasets submitted.
Software: 3 = Potentially useful: Someone might find the new software useful for their work.
Knowledge Of Or Educated Guess At Author Identity: No, I do not have even an educated guess about author identity
Publication Ethics Policy Compliance: I used a privacy-preserving tool exclusively for the use case(s) approved by PEC policy, such as language edits


## reviewer 3
Official Review of Submission1609 by Reviewer Sg6v
Official Reviewby Reviewer Sg6v03 Nov 2026, 02:51 (modified: 24 Nov 2026, 09:47)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer Sg6v, AuthorsRevisions
Paper Summary:
This paper conducts an empirical investigation of query‑adaptive Weighted Reciprocal Rank Fusion (WRRF) for hybrid sparse‑and‑dense retrieval. Its central claim is that, for most queries, optimal WRRF fusion weights form continuous intervals rather than isolated single points, which undermines existing point‑estimation‑based dynamic fusion such as DAT.

Summary Of Strengths:
The dominant interval‑shaped optimal weight for WRRF is a noteworthy empirical pattern.
Oracle‑derived headroom and recovery ratio help quantify how much potential existing methods can capture.
Tiered latency‑constrained decision framework is useful for IR engineers.
Summary Of Weaknesses:
The paper observes that optimal weights are intervals, but does not rigorously answer why plateaus appear.
Prior work uses Query‑Performance‑Prediction (QPP) to adapt fusion strength. The paper only compares against DAT, ignoring existing QPP‑driven adaptive fusion approaches.
The proposed satisficing loss only brings very limited empirical improvements.
Comments Suggestions And Typos:
W1-W3

Confidence: 3 =  Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design.
Soundness: 2.5
Excitement: 2.5
Overall Assessment: 3 = Findings: I think this paper could be accepted to the Findings of the ACL.
Ethical Concerns:
There are no concerns with this submission.

Needs Ethics Review: No
Reproducibility: 3 = They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.
Datasets: 3 = Potentially useful: Someone might find the new datasets useful for their work.
Software: 3 = Potentially useful: Someone might find the new software useful for their work.
Knowledge Of Or Educated Guess At Author Identity: Yes, I learned it during review process (e.g. checking literature)
Publication Ethics Policy Compliance: I did not use any generative AI tools for this review