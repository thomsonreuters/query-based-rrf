## Optimal Fusion Weights

To collect the optimal fusion weights for each dataset, use one of the following scripts:

* `rrf_mrr_best_weights_all_weights_final.py`
* `rrf_ndcg_best_weights_all_weights_final.py`

Choose the script based on the dataset characteristics:

* If each query has **only one relevant document**, use the **MRR-based** script.
* If each query has **multiple relevant documents**, use the **nDCG-based** script.

---

## Fuse Results

After obtaining the predicted optimal weights from the model, use:

```
helper_4_fuse_results_wrrf.py
```

This script fuses search results from different retrievers, including:

* **Sparse retrievers**: BM25, RM3
* **Dense retrievers**: Bi-encoder, Qwen3

The fused results are saved in **TREC format**.

Finally, use:

```
helper_5_ir_metrics.py
```

to compute evaluation metrics such as **nDCG** and **MRR**.

