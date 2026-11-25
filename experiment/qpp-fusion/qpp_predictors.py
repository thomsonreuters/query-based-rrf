"""Post-retrieval query performance predictors (QPP) computed from one ranked list.

Every predictor reads only the retrieval scores of a single retriever's top-n list for a query
(the scores column of the TREC run file), so it works the same way for sparse (BM25, RM3) and
dense (cosine) retrievers and needs no index or corpus statistics at fusion time.

Classic score-based predictors normalise by the score of the whole corpus treated as one document
(NQC, Shtok et al. 2012; WIG, Zhou & Croft 2007; SMV, Tao & Wu 2014). That score is not available
from a run file, and it has no counterpart for a dense retriever, so we use the mean score of the
retrieved top-n list as the reference score instead. This keeps every predictor scale-free: it
does not change if all scores of a list are multiplied by a constant, which matters because BM25
scores grow with query length while cosine scores do not.

    sd@k    std of the top-k scores (sigma_k; Perez-Iglesias & Araujo 2010). Not scale-free.
    nqc@k   std(top-k) / |mean(top-n)|
    wig@k   (mean(top-k) - mean(top-n)) / |mean(top-n)|
    smv@k   mean_{i<=k} s_i |ln(s_i / mean(top-k))| / |mean(top-n)|
    max     top-1 score. Not scale-free.
    gap     (s_1 - s_2) / |s_1|
"""
import math

import numpy as np

EPS = 1e-12

# name -> (function, k). k=None means the predictor has no cutoff.
PREDICTOR_SPECS = {
    "sd@10": ("sd", 10),
    "sd@100": ("sd", 100),
    "nqc@10": ("nqc", 10),
    "nqc@100": ("nqc", 100),
    "wig@5": ("wig", 5),
    "wig@10": ("wig", 10),
    "smv@10": ("smv", 10),
    "smv@100": ("smv", 100),
    "max": ("max", None),
    "gap": ("gap", None),
}
PREDICTORS = list(PREDICTOR_SPECS)


def _ref(s):
    r = abs(float(s.mean()))
    return r if r > EPS else EPS


def sd(s, k):
    return float(s[:k].std())


def nqc(s, k):
    return float(s[:k].std()) / _ref(s)


def wig(s, k):
    return (float(s[:k].mean()) - float(s.mean())) / _ref(s)


def smv(s, k):
    sk = s[:k]
    mu = float(sk.mean())
    if mu <= EPS:
        return 0.0
    ratio = sk / mu
    ok = ratio > EPS  # ln is undefined for non-positive scores; they contribute nothing
    return float(np.sum(sk[ok] * np.abs(np.log(ratio[ok])))) / len(sk) / _ref(s)


def max_score(s):
    return float(s[0])


def gap(s):
    return float(s[0] - s[1]) / max(abs(float(s[0])), EPS) if len(s) > 1 else 0.0


_FUNCS = {"sd": sd, "nqc": nqc, "wig": wig, "smv": smv}


def predict(scores, name):
    """Value of predictor `name` for one list of scores sorted in descending order.

    Returns NaN for an empty list (the retriever returned nothing for this query)."""
    s = np.asarray(scores, dtype=np.float64)
    if s.size == 0:
        return math.nan
    fn, k = PREDICTOR_SPECS[name]
    if fn == "max":
        return max_score(s)
    if fn == "gap":
        return gap(s)
    return _FUNCS[fn](s, k)


def all_predictors(scores):
    return {name: predict(scores, name) for name in PREDICTORS}


# ----------------------------------------------------------------------------------------------
# Cross-list ("agreement") features: how much the two retrievers agree on this query. These are
# the retrieval-internal signals suggested by Reviewer 25bQ (top-k overlap, exclusive documents,
# RRF score margin); they are used only by the supervised QPP models.
# ----------------------------------------------------------------------------------------------

def _rr_of(doc, ranks):
    r = ranks.get(doc)
    return 1.0 / r if r else 0.0


def agreement_features(sparse_docs, dense_docs, k_rrf=60):
    """sparse_docs / dense_docs: document ids in rank order (rank 1 first)."""
    f = {}
    for k in (10, 50):
        a, b = set(sparse_docs[:k]), set(dense_docs[:k])
        f[f"jaccard@{k}"] = len(a & b) / len(a | b) if (a or b) else 0.0
    rs = {d: i + 1 for i, d in enumerate(sparse_docs)}
    rd = {d: i + 1 for i, d in enumerate(dense_docs)}
    f["top1_agree"] = float(bool(sparse_docs) and bool(dense_docs) and sparse_docs[0] == dense_docs[0])
    f["rr_sparse_top1_in_dense"] = _rr_of(sparse_docs[0], rd) if sparse_docs else 0.0
    f["rr_dense_top1_in_sparse"] = _rr_of(dense_docs[0], rs) if dense_docs else 0.0

    # Standard RRF (w = 0.5) over the union, to describe the fused list a fixed weight would give.
    fused = {}
    for d, r in rs.items():
        fused[d] = fused.get(d, 0.0) + 0.5 / (k_rrf + r)
    for d, r in rd.items():
        fused[d] = fused.get(d, 0.0) + 0.5 / (k_rrf + r)
    order = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    top10 = [d for d, _ in order[:10]]
    n10 = max(len(top10), 1)
    f["rrf_top10_only_sparse"] = sum(d not in rd for d in top10) / n10
    f["rrf_top10_only_dense"] = sum(d not in rs for d in top10) / n10
    sc = [s for _, s in order] + [0.0] * 11
    scale = 2.0 * (k_rrf + 1)  # makes the margins roughly unit-scale
    f["rrf_margin_1_2"] = (sc[0] - sc[1]) * scale
    f["rrf_margin_10_11"] = (sc[9] - sc[10]) * scale
    f["n_sparse"] = len(sparse_docs) / 200.0
    f["n_dense"] = len(dense_docs) / 200.0
    return f


AGREEMENT_FEATURES = [
    "jaccard@10", "jaccard@50", "top1_agree", "rr_sparse_top1_in_dense", "rr_dense_top1_in_sparse",
    "rrf_top10_only_sparse", "rrf_top10_only_dense", "rrf_margin_1_2", "rrf_margin_10_11",
    "n_sparse", "n_dense",
]
