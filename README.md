


## Data Collection

### Optimal Fusion Weights (Ground Truth)

To collect the optimal fusion weights for each dataset, use one of the following scripts:

* `rrf_mrr_best_weights_all_weights_final.py`
* `rrf_ndcg_best_weights_all_weights_final.py`

Choose the script based on the dataset characteristics:

* If each query has **only one relevant document**, use the **MRR-based** script.
* If each query has **multiple relevant documents**, use the **nDCG-based** script.

---

## Fusion Strategy Evaluation
### Fuse Search Results Use Specific Weight

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

---


## Experiments

### Experiment 1: Standard RRF

In this setting, both sparse retrievers (`bm25`, `rm3`) and dense retrievers (`biencoder`, `qwen3`) use equal fusion weights (0.5).

To run this experiment, use the script:

```
helper_4_fuse_results_wrrf.py
```

Set the following parameters in the script:

```python
use_fixed_weight = True  # Enable fixed fusion weights
sparse_weight = 0.5
```

Then manually switch the retriever names as needed:

```python
sparse_name = "rm3"      # Options: "bm25" or "rm3"
dense_name = "qwen3"     # Options: "biencoder" or "qwen3"
```

---

### Experiment 2: Mean Optimal Weight

For each dataset, use the **mean optimal fusion weight** collected from its training set to fuse the search results.

The execution logic is the same as in **Experiment 1 (Standard RRF)** — the only difference is that you set `sparse_weight` to the dataset-specific mean optimal weight instead of `0.5`.

---

### Experiment 3: Ridge Regression

See the directory:  
[`/experiment/ridge-regression/ridge-regression-mean-best-weight`](./experiment/ridge-regression/ridge-regression-mean-best-weight)

---

### Experiment 4: RoBERTa Regression

See the directory:  
[`/experiment/roberta/roberta-experiment-mean-best-weight`](./experiment/roberta/roberta-experiment-mean-best-weight)

---

### Experiment 5: LLM-Based Weight Prediction

See the directory:  
[`/experiment/llm_selected_sparse_w`](./experiment/llm_selected_sparse_w)

