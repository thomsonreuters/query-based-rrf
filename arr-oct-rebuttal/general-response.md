# General response to the chairs and all reviewers (draft)

Submission 1609, ARR October. One comment, limit 5,000 characters, plain text only (no Markdown, no tables). The loss numbers (point 5) live here; the replies to Reviewers 25bQ (S3) and Sg6v (W3) point to them. The versions with tables are in with-tables/.

---

Dear Program Chairs, Senior Area Chairs, Area Chairs and Reviewers:

We sincerely thank you for your time and for your careful reviews. Below, we summarize the strengths the reviewers recognized and the main points of our responses.

Strengths recognized by the reviewers:

- The interval finding: optimal fusion weights form intervals, not single points. This is "an interesting observation" (XL1T) and "a noteworthy empirical pattern" (Sg6v), with "direct implications for subsequent work on fusion weights" (25bQ).
- Rigorous analysis: results on 4 datasets and 4 sparse-dense settings (XL1T), with statistical testing and a grid sensitivity analysis (25bQ).
- Oracle headroom and recovery ratio: they "help quantify how much potential existing methods can capture" (Sg6v).
- Latency tiers: they guide the balance between performance and latency (XL1T) and are "useful for IR engineers" (Sg6v).

Main points of our responses:

1. Why the optimal weights form intervals (Reviewers Sg6v and XL1T). The fused score of each document is linear in the weight, so the metric can change only where two documents swap order. It is a step function of the weight, and the optimal intervals are its best flat pieces. Most swaps happen below the top 10 and cannot change the metric. For a query with several optimal intervals, trained models use the widest one, and the LLM prompt and the oracle use all of them.
2. Evaluation settings (Reviewers XL1T and 25bQ). Our conclusions stay the same at top-3 and top-5 and for k from 10 to 100. Intervals stay the most common optimal set at depth 400 and at every k we tested.
3. Size of the gain from per-query weights (Reviewers XL1T, 25bQ and Sg6v). The intervals hold on all four datasets; only the size of the gain differs. The gain from a per-query weight is there, and it is hard to predict (Section 5.2). We will collect the reasons in one paragraph of Section 5.
4. Signals from the retrieval lists (Reviewers 25bQ and Sg6v). Retrieval-internal features add a small gain over the query text. A QPP-driven fusion baseline (NQC) improves over standard RRF on MS MARCO and NQ with no LLM call. With both, most of the headroom stays open.
5. Interval loss against point loss on the same encoder (Reviewers 25bQ and Sg6v). The interval variant is higher on MS MARCO and NQ, and the difference is significant. On the BM(Q) pair, the satisficing loss scores .326 on MS MARCO and .423 on NQ (Table 2). With the same backbone (ModernBERT), data and settings, and only the loss changed, it is .006 above MSE on both (paired t-test, p < .05). On ACORD and NFCorpus, it scores .140 and .378 (Table 2), and it is .003 and .001 above MSE (four pairs). The differences on NQ, ACORD and NFCorpus are means over three seeds.
6. Claim and positioning (Reviewer 25bQ). Following the reviewer's suggestions, the revised paper says that point losses supervise interval optima inefficiently, presents the work as an empirical characterisation, and states the relation of our loss to the ε-insensitive loss.
7. Static baseline and training settings (Reviewer 25bQ). We add the tuned constant w*. It is a strong static baseline, and the per-query oracle is still far above it. We also list the training settings for ACORD and NFCorpus, and add a control trained by steps.

In our responses, unless we say otherwise, MS MARCO and NQ use MRR@10, ACORD and NFCorpus use nDCG@10, and scores are averages over the four retriever pairs, as in Table 2. Four numbers in a row are for MS MARCO / NQ / ACORD / NFCorpus.

Our responses give the main numbers. We will apply all of the changes and add all of the new results, with the full tables, to the revised paper. Our code is at the link in the abstract.

Thank you again for your time and for your helpful comments.
