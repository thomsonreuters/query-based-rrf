"""Per-query optimal-weight labels: load the repo's weight-sweep CSVs, or build them.

The sweep CSVs are the ones written by `rrf_mrr_best_weights_all_weights_final.py` and
`rrf_ndcg_best_weights_all_weights_final.py`, at

    {data_root}/{dataset}/{mrr|ndcg}_runs/{split}/top200/results_{split}_{sparse}_vs_{dense}_best_weights.csv

with columns query_id, query_text, best_weights, highest_<m>@10, weights, all_weights_<m>@10.
If a file is missing it is built with the same code: the nDCG sweep calls the repo function
directly; the MRR sweep uses `sweep_mrr_fast`, a vectorised port of the repo's pure-Python loop
that gives identical output (checked in test_qpp.py) and is fast enough for the MS MARCO and NQ
training splits.
"""
import csv
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import rrf_mrr_best_weights_all_weights_final as mrr_sweep  # noqa: E402
import rrf_ndcg_best_weights_all_weights_final as ndcg_sweep  # noqa: E402

WEIGHTS = [round(i * 0.01, 2) for i in range(101)]

# Evaluation split and metric per dataset, as in the paper.
DATASETS = {
    "msmarco": ("dev", "mrr"),
    "nq": ("dev", "mrr"),
    "acord-entire-corpus": ("test", "ndcg"),
    "nfcorpus": ("test", "ndcg"),
}


def run_path(data_root, dataset, retriever, split, top_k=200):
    return os.path.join(data_root, dataset, "search_results", retriever, f"top{top_k}", f"results_{split}.trec")


def qrels_path(data_root, dataset, split):
    return os.path.join(data_root, dataset, "qrels", f"{split}.tsv")


def sweep_path(data_root, dataset, split, sparse, dense, metric, top_k=200):
    return os.path.join(data_root, dataset, f"{metric}_runs", split, f"top{top_k}",
                        f"results_{split}_{sparse}_vs_{dense}_best_weights.csv")


# ----------------------------------------------------------------------------------------------
# MRR sweep, vectorised. Matches mrr_sweep.optimize_rrf_iterative exactly:
#  - union of documents in insertion order (sparse list in rank order, then dense-only documents);
#  - fused score = w * 1/(60+r_s) + round(1-w, 2) * 1/(60+r_d), summed in that order;
#  - stable descending sort (Python's sorted(..., reverse=True) keeps ties in insertion order);
#  - MRR@10 rounded to 4 decimals; best_weights = all weights attaining a positive maximum.
# ----------------------------------------------------------------------------------------------

def load_run_compact(path):
    """{qid: (ranks int32[n], doc_hashes int64[n], scores float64[n])}, sorted by rank (stable).

    Streams the file and keeps NumPy arrays instead of one Python tuple per line (~20 bytes per
    line instead of ~150), so the MS MARCO training runs (~100M lines each) fit in memory. Doc
    ids are Python hashes, compared only within this process."""
    run, chunks = {}, defaultdict(list)
    cur, buf = None, ([], [], [])

    def flush():
        if cur is not None and buf[0]:
            chunks[cur].append((np.array(buf[0], dtype=np.int32), np.array(buf[1], dtype=np.int64),
                                np.array(buf[2], dtype=np.float64)))

    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            if p[0] != cur:
                flush()
                cur, buf = p[0], ([], [], [])
            buf[0].append(int(p[3]))
            buf[1].append(hash(p[2]))
            buf[2].append(float(p[4]))
    flush()
    for qid, parts in chunks.items():
        r, h, s = (np.concatenate(x) for x in zip(*parts)) if len(parts) > 1 else parts[0]
        o = np.argsort(r, kind="stable")
        run[qid] = (r[o], h[o], s[o])
    return run


def sweep_mrr_fast(sparse_file, dense_file, qrels_file, rr_k=60, k=10):
    qrels = {q: {hash(d) for d in docs} for q, docs in mrr_sweep.load_qrels(qrels_file).items()}  # score > 0
    sparse = load_run_compact(sparse_file)
    dense = load_run_compact(dense_file)
    empty = (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64), np.zeros(0))
    w_s = np.array(WEIGHTS).reshape(-1, 1)
    w_d = np.array([round(1.0 - w, 2) for w in WEIGHTS]).reshape(-1, 1)
    rows = []
    for qid in set(sparse) | set(dense):
        index, vs, vd = {}, [], []
        s_r, s_h, _ = sparse.get(qid, empty)
        d_r, d_h, _ = dense.get(qid, empty)
        for doc, rank in zip(s_h.tolist(), s_r.tolist()):
            if doc not in index:
                index[doc] = len(vs)
                vs.append(0.0)
                vd.append(0.0)
            vs[index[doc]] += 1.0 / (rr_k + rank)
        for doc, rank in zip(d_h.tolist(), d_r.tolist()):
            if doc not in index:
                index[doc] = len(vs)
                vs.append(0.0)
                vd.append(0.0)
            vd[index[doc]] += 1.0 / (rr_k + rank)
        rel = qrels.get(qid, set())
        values = [0.0] * len(WEIGHTS)
        if rel and index:
            is_rel = np.zeros(len(vs), dtype=bool)
            for doc in rel:
                if doc in index:
                    is_rel[index[doc]] = True
            if is_rel.any():
                # Python computes 0.0 + w*s, then += round(1-w)*d. For absent entries the extra
                # term is an exact 0.0, so this expression gives bit-identical sums.
                scores = w_s * np.array(vs)[None, :] + w_d * np.array(vd)[None, :]
                top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
                hit = is_rel[top]
                for i in range(len(WEIGHTS)):
                    pos = np.flatnonzero(hit[i])
                    if pos.size:
                        values[i] = round(1.0 / (pos[0] + 1), 4)
        best = max(values)
        rows.append({
            "query_id": qid, "query_text": f"Query_{qid}",
            "best_weights": [w for w, v in zip(WEIGHTS, values) if v == best] if best > 0 else [],
            "highest_mrr@10": best, "weights": WEIGHTS, "all_weights_mrr@10": values,
        })
    return rows


def build_sweep(data_root, dataset, split, sparse, dense, metric):
    sf, df = run_path(data_root, dataset, sparse, split), run_path(data_root, dataset, dense, split)
    qf = qrels_path(data_root, dataset, split)
    if metric == "mrr":
        rows = sweep_mrr_fast(sf, df, qf)
    else:
        rows, *_ = ndcg_sweep.optimize_ndcg_numpy(sf, df, qf, ndcg_k=10)
    out = sweep_path(data_root, dataset, split, sparse, dense, metric)
    fields = ["query_id", "query_text", "best_weights", f"highest_{metric}@10", "weights", f"all_weights_{metric}@10"]
    ndcg_sweep.write_csv_file(out, rows, fields)
    return out


def load_sweep(data_root, dataset, split, sparse, dense, metric, build=True):
    """Returns {query_id: {"best_weights": [...], "best": float, "curve": np.array(101)}}."""
    path = sweep_path(data_root, dataset, split, sparse, dense, metric)
    if not os.path.exists(path):
        if not build:
            raise FileNotFoundError(path)
        print(f"  building weight sweep: {path}")
        build_sweep(data_root, dataset, split, sparse, dense, metric)
    csv.field_size_limit(1 << 30)
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["query_id"]] = {
                "best_weights": json.loads(row["best_weights"]),
                "best": float(row[f"highest_{metric}@10"]),
                "curve": np.array(json.loads(row[f"all_weights_{metric}@10"]), dtype=np.float64),
            }
    return out


# ----------------------------------------------------------------------------------------------
# Interval labels, following get_mean_best_weight.py: consecutive grid weights (gap <= 0.011)
# form an interval; the primary interval I_q is the widest one (first on ties); the point target
# is the mean of the grid weights inside it.
# ----------------------------------------------------------------------------------------------

def intervals(best_weights):
    ws = sorted(best_weights)
    if not ws:
        return []
    out, start, prev = [], ws[0], ws[0]
    for w in ws[1:]:
        if abs(w - prev) > 0.011:
            out.append((start, prev))
            start = w
        prev = w
    out.append((start, prev))
    return out


def primary_interval(best_weights, rng=random):
    ivs = intervals(best_weights)
    if not ivs:
        return None
    widest = max(ivs, key=lambda iv: iv[1] - iv[0])
    if widest[1] - widest[0] == 0 and len(ivs) > 1:
        widest = rng.choice(ivs)  # all single points: pick one at random, as the repo does
    return widest


def mean_best_weight(best_weights, rng=random):
    """Same branches as get_mean_best_weight.calculate_mean_weight."""
    ivs = intervals(best_weights)
    if not ivs:
        return None
    if len(best_weights) == 1:
        return round(best_weights[0], 2)
    if len(ivs) == 1:
        lo, hi = ivs[0]
        if round(hi - lo, 2) == 0.01:
            return rng.choice([lo, hi])
    else:
        wide = [iv for iv in ivs if iv[1] > iv[0]]
        if not wide:
            return rng.choice([iv[0] for iv in ivs])
        lo, hi = max(wide, key=lambda iv: iv[1] - iv[0])
    # round() on the numpy scalar (not a Python float) to match the repo's half-even behaviour.
    return float(round(np.mean(np.arange(lo, hi + 0.01, 0.01)), 2))


def curve_value(curve, w):
    """Metric of a query at weight w, read from its 101-point sweep (w rounded to the grid)."""
    return float(curve[int(round(w * 100))])


def load_qrels_any(path):
    qrels = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                qrels[p[0]][p[1]] = int(p[2])
    return qrels
