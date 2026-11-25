"""Checks for the QPP baseline: python experiment/qpp-fusion/test_qpp.py

1. labels.sweep_mrr_fast gives the same output as the repo's rrf_mrr_best_weights loop, on
   synthetic runs built to contain score ties, duplicate documents and empty lists.
2. labels.mean_best_weight agrees with get_mean_best_weight.calculate_mean_weight.
3. The scale-free predictors do not change when a list's scores are multiplied by a constant.
"""
import os
import random
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labels as L  # noqa: E402
import qpp_predictors as Q  # noqa: E402

import get_mean_best_weight as G  # noqa: E402
import rrf_mrr_best_weights_all_weights_final as M  # noqa: E402


def _write_synthetic(tmp, rng, n_q=60, n_docs=80, depth=40):
    sp, de, qr = (os.path.join(tmp, f) for f in ("sparse.trec", "dense.trec", "qrels.tsv"))
    with open(sp, "w") as fs, open(de, "w") as fd, open(qr, "w") as fq:
        fq.write("query-id\tcorpus-id\tscore\n")
        for q in range(n_q):
            for f, tag in ((fs, "bm25"), (fd, "dense")):
                if rng.random() < 0.05:
                    continue  # empty list for this query
                docs = rng.sample(range(n_docs), depth)
                if rng.random() < 0.1:
                    docs[5] = docs[3]  # duplicate document in one list
                for r, d in enumerate(docs, 1):
                    f.write(f"q{q} Q0 d{d} {r} {100 - r} {tag}\n")
            for d in rng.sample(range(n_docs), rng.choice([0, 1, 2, 3])):
                fq.write(f"q{q}\td{d}\t{rng.choice([0, 1, 1, 2])}\n")
    return sp, de, qr


def test_fast_mrr_sweep_matches_repo():
    rng = random.Random(0)
    with tempfile.TemporaryDirectory() as tmp:
        sp, de, qr = _write_synthetic(tmp, rng)
        ref, *_ = M.optimize_rrf_iterative(sp, de, qr)
        new = L.sweep_mrr_fast(sp, de, qr)
    ref = {r["query_id"]: r for r in ref}
    new = {r["query_id"]: r for r in new}
    assert ref.keys() == new.keys()
    for q in ref:
        for key in ("best_weights", "highest_mrr@10", "all_weights_mrr@10"):
            assert ref[q][key] == new[q][key], (q, key, ref[q][key], new[q][key])
    print(f"ok  fast MRR sweep identical to repo loop on {len(ref)} synthetic queries")


def test_mean_best_weight_matches_repo():
    rng = random.Random(1)
    n = 0
    for _ in range(3000):
        pts = sorted({round(x * 0.01, 2) for x in rng.sample(range(101), rng.choice([1, 2, 3, 8, 30]))})
        if rng.random() < 0.5:  # make contiguous runs
            a = rng.randrange(0, 90)
            pts = sorted(set(pts) | {round(x * 0.01, 2) for x in range(a, a + rng.randrange(1, 12))})
        friendly, count = G.get_interval_info(str(pts))
        seed = rng.random()
        random.seed(seed)
        ref = G.calculate_mean_weight(friendly, count)
        new = L.mean_best_weight(pts, random.Random(seed))
        # Both make at most one draw, from the same list, with the same seed.
        assert ref == new, (pts, friendly, ref, new)
        n += 1
    print(f"ok  mean_best_weight agrees with get_mean_best_weight on {n} random weight sets")


def test_scale_free_predictors():
    rng = np.random.default_rng(0)
    s = np.sort(rng.gamma(2.0, 3.0, 200))[::-1]
    for name in ("nqc@10", "nqc@100", "wig@5", "wig@10", "smv@10", "smv@100", "gap"):
        a, b = Q.predict(s, name), Q.predict(7.3 * s, name)
        assert abs(a - b) < 1e-9 * max(1.0, abs(a)), (name, a, b)
    assert np.isnan(Q.predict([], "nqc@100"))
    print("ok  scale-free predictors are invariant to score scaling")


if __name__ == "__main__":
    test_fast_mrr_sweep_matches_repo()
    test_mean_best_weight_matches_repo()
    test_scale_free_predictors()
