#!/usr/bin/env python
"""QPP-driven dynamic fusion baselines for WRRF (reviewer Sg6v, W2).

For every dataset x (sparse, dense) pair this script
  1. computes post-retrieval QPP predictors (qpp_predictors.py) on each retriever's top-200 list,
     for the training split and the evaluation split;
  2. turns them into a per-query sparse weight w_q with four methods (below);
  3. fuses the evaluation runs with helper_4_fuse_results_wrrf.py (k = 60) and scores them with
     helper_5_ir_metrics.py, exactly as for the other methods in the paper;
  4. reports mean metric, bootstrap 95% CI, paired two-sided t-test vs standard RRF and vs MOW,
     and headroom recovery (Method - RRF) / (Oracle - RRF).

Methods (all CPU; predictions rounded to the 0.01 grid):
  QPP-U[p]   unsupervised, no labels. Each retriever's predictor p is mapped to [0, 1] by its
             empirical CDF over the training queries (run lists only, no qrels), and
             w_q = F_s(p_s) / (F_s(p_s) + F_d(p_d)). The retriever that looks more confident
             relative to its own typical query gets more weight. Reported for the fixed default
             predictor (nqc@100) and for every predictor in qpp_predictors.csv.
  QPP-C[p]   QPP-U's signal with a 2-parameter calibration fitted on training labels:
             w_q = sigmoid(a (F_s - F_d) + b), minimising the satisficing loss (paper Eq. 3)
             against the primary optimal interval [l_q, r_q]. a = 0 reduces to a constant
             weight, so this nests a tuned dataset-level constant.
  QPP-LR     linear model with sigmoid output on all predictors of both lists, the cross-list
             agreement features, and query length; satisficing loss + L2, strength by 5-fold CV.
  QPP-GBR    gradient-boosted trees on the same features, MSE against the interval midpoint.
For QPP-U and QPP-C, the "[train-best]" rows pick the predictor with the highest training-split
metric (this uses training labels).

Reference rows computed the same way: Standard RRF (0.5), Mean Optimal Weight (mean of training
midpoints, reachable queries), Tuned static w* (the single weight maximising the mean training
metric; reviewer 25bQ), and the per-query Oracle.

  python experiment/qpp-fusion/qpp_fusion.py --datasets acord-entire-corpus nq
  python experiment/qpp-fusion/qpp_fusion.py --datasets nq --sparse bm25 --dense qwen3

Pairs whose run files are missing are skipped with a message.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labels as L  # noqa: E402  (also puts the repo root on sys.path)
import qpp_predictors as Q  # noqa: E402

import helper_4_fuse_results_wrrf as fuse  # noqa: E402
import helper_5_ir_metrics as ir  # noqa: E402

DEFAULT_PREDICTOR = "nqc@100"
SEED = 42


# ----------------------------------------------------------------------------------------------
# Run files -> per-query score lists
# ----------------------------------------------------------------------------------------------

_RUN_CACHE = {}


def load_run(path):
    """{qid: (doc_hashes int64[n], scores float64[n])}, ordered by rank. Cached per path.

    Doc ids are stored as Python hashes so the MS MARCO training runs fit in memory; they are
    only compared with each other inside this process."""
    if path not in _RUN_CACHE:
        _RUN_CACHE[path] = {q: (h, s) for q, (_, h, s) in L.load_run_compact(path).items()}
    return _RUN_CACHE[path]


def load_query_lengths(data_root, dataset):
    path = os.path.join(data_root, dataset, "queries.jsonl")
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                out[q["_id"]] = q.get("text", "")
    return out


EMPTY = (np.zeros(0, dtype=np.int64), np.zeros(0))


def query_features(s_list, d_list, qtext=None):
    """All features for one query, from the two (doc_hashes, scores) lists."""
    f = {}
    for prefix, (_, sc) in (("s_", s_list), ("d_", d_list)):
        for name in Q.PREDICTORS:
            f[prefix + name] = Q.predict(sc, name)
    f.update(Q.agreement_features(list(s_list[0]), list(d_list[0])))
    if qtext is not None:
        f["log_qlen"] = math.log1p(len(qtext.split()))
    return f


def features_for(qids, s_run, d_run, qtexts):
    return {q: query_features(s_run.get(q, EMPTY), d_run.get(q, EMPTY),
                              None if qtexts is None else qtexts.get(q, "")) for q in qids}


# ----------------------------------------------------------------------------------------------
# Weight predictors
# ----------------------------------------------------------------------------------------------

class ECDF:
    """Empirical CDF fitted on training values; NaN (empty list) maps to 0."""

    def __init__(self, values):
        v = np.asarray([x for x in values if not math.isnan(x)], dtype=np.float64)
        self.sorted = np.sort(v) if v.size else np.zeros(1)

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        out = np.searchsorted(self.sorted, np.nan_to_num(x, nan=-np.inf), side="right") / len(self.sorted)
        return np.where(np.isnan(x), 0.0, out)


def qpp_ratio(fs, fd):
    tot = fs + fd
    return np.where(tot > 0, fs / np.where(tot > 0, tot, 1.0), 0.5)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def sat_loss_fit(X, lo, hi, lam, init_b):
    """min_theta mean(relu(lo - p)^2 + relu(p - hi)^2) + lam * |beta|^2,  p = sigmoid(X beta + b)."""
    n, d = X.shape

    def fun(theta):
        beta, b = theta[:d], theta[d]
        p = sigmoid(X @ beta + b)
        under, over = np.maximum(lo - p, 0.0), np.maximum(p - hi, 0.0)
        loss = np.mean(under ** 2 + over ** 2) + lam * beta @ beta
        dp = (2.0 * over - 2.0 * under) * p * (1.0 - p) / n
        grad = np.concatenate([X.T @ dp + 2.0 * lam * beta, [dp.sum()]])
        return loss, grad

    theta0 = np.zeros(d + 1)
    theta0[d] = init_b
    res = optimize.minimize(fun, theta0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    return res.x


def sat_loss(p, lo, hi):
    return float(np.mean(np.maximum(lo - p, 0) ** 2 + np.maximum(p - hi, 0) ** 2))


def logit(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return math.log(p / (1 - p))


class Standardizer:
    def __init__(self, X):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-12] = 1.0

    def __call__(self, X):
        return (X - self.mu) / self.sd


class SatLinear:
    """Linear model, sigmoid output, satisficing loss + L2 chosen by k-fold CV."""

    LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)

    def fit(self, X, lo, hi, init_b, folds=5):
        self.scaler = Standardizer(X)
        Z = self.scaler(X)
        rng = np.random.default_rng(SEED)
        fold = rng.permutation(len(Z)) % folds
        cv = {}
        for lam in self.LAMBDAS:
            losses = []
            for k in range(folds):
                tr, va = fold != k, fold == k
                if tr.sum() < 2 or va.sum() < 1:
                    continue
                th = sat_loss_fit(Z[tr], lo[tr], hi[tr], lam, init_b)
                losses.append(sat_loss(sigmoid(Z[va] @ th[:-1] + th[-1]), lo[va], hi[va]))
            cv[lam] = float(np.mean(losses)) if losses else math.inf
        self.lam = min(cv, key=cv.get)
        self.theta = sat_loss_fit(Z, lo, hi, self.lam, init_b)
        return self

    def predict(self, X):
        return sigmoid(self.scaler(X) @ self.theta[:-1] + self.theta[-1])


def make_gbr(n):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
        min_samples_leaf=max(5, min(50, n // 20)), early_stopping=n >= 2000, random_state=SEED)


# ----------------------------------------------------------------------------------------------
# Evaluation, through the repo's own fusion (helper_4) and metric (helper_5) code
# ----------------------------------------------------------------------------------------------

def per_query_metric(trec_path, qrels_path, metric, k=10):
    run = ir.load_trec_results(trec_path)
    qrels = ir.load_qrels(qrels_path)
    fn = ir.calculate_ndcg_at_k_single_query if metric == "ndcg" else ir.calculate_mrr_at_k_single_query
    return {q: fn(run.get(q, []), qrels[q], k) for q in qrels}


def write_predictions(path, weights, qtexts):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "query_text", "predicted"])
        for q, v in weights.items():
            w.writerow([q, "" if qtexts is None else qtexts.get(q, ""), f"{v:.2f}"])


def bootstrap_ci(x, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_p(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.allclose(a, b):
        return 1.0
    return float(stats.ttest_rel(a, b).pvalue)


def time_per_query(fn, items, warmup=20):
    """Mean per-query latency (ms) with batch size 1, and its 95% t-CI half width."""
    for it in items[:warmup]:
        fn(it)
    ts = []
    for it in items:
        t0 = time.perf_counter()
        fn(it)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts = np.asarray(ts)
    half = float(stats.t.ppf(0.975, len(ts) - 1) * ts.std(ddof=1) / math.sqrt(len(ts))) if len(ts) > 1 else 0.0
    return float(ts.mean()), half


# ----------------------------------------------------------------------------------------------
# One dataset x retriever pair
# ----------------------------------------------------------------------------------------------

def run_pair(args, dataset, sparse, dense):
    eval_split, metric = L.DATASETS[dataset]
    root = args.data_root
    paths = {(r, sp): L.run_path(root, dataset, r, sp) for r in (sparse, dense) for sp in ("train", eval_split)}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        print(f"[skip] {dataset} {sparse}+{dense}: missing {', '.join(missing)}")
        return None
    tag = f"{dataset}_{sparse}_vs_{dense}"
    print(f"\n=== {tag} (train -> {eval_split}, {metric.upper()}@10) ===")
    rng = random.Random(SEED)
    qtexts = load_query_lengths(root, dataset)

    sweep_tr = L.load_sweep(root, dataset, "train", sparse, dense, metric)
    sweep_ev = L.load_sweep(root, dataset, eval_split, sparse, dense, metric)

    s_tr, d_tr = load_run(paths[(sparse, "train")]), load_run(paths[(dense, "train")])
    s_ev, d_ev = load_run(paths[(sparse, eval_split)]), load_run(paths[(dense, eval_split)])
    qids_tr = sorted(set(s_tr) | set(d_tr))
    qids_ev = sorted(set(s_ev) | set(d_ev))
    t0 = time.time()
    F_tr = features_for(qids_tr, s_tr, d_tr, qtexts)
    F_ev = features_for(qids_ev, s_ev, d_ev, qtexts)
    print(f"  features: {len(F_tr)} train / {len(F_ev)} eval queries ({time.time() - t0:.1f}s)")
    feat_names = list(next(iter(F_tr.values())).keys())

    # Training labels: primary interval and midpoint, reachable queries only.
    lab = []
    for q in qids_tr:
        sw = sweep_tr.get(q)
        if sw and sw["best"] > 0 and sw["best_weights"]:
            lo, hi = L.primary_interval(sw["best_weights"], rng)
            lab.append((q, lo, hi, L.mean_best_weight(sw["best_weights"], rng)))
    lab_q = [x[0] for x in lab]
    lo = np.array([x[1] for x in lab])
    hi = np.array([x[2] for x in lab])
    mid = np.array([x[3] for x in lab])
    print(f"  {len(lab)} reachable training queries with labels")

    # Per-retriever ECDFs over training queries (no labels needed).
    ecdf = {}
    for p in Q.PREDICTORS:
        for side in ("s_", "d_"):
            ecdf[side + p] = ECDF([F_tr[q][side + p] for q in qids_tr])

    def mat(F, qs, names):
        X = np.array([[F[q][n] for n in names] for q in qs], dtype=np.float64)
        for j, n in enumerate(names):
            if n in ecdf:
                X[:, j] = ecdf[n](X[:, j])
        return np.nan_to_num(X, nan=0.0)

    # Mean training metric of a weight vector, read from the training sweep (selection only).
    curves_tr = {q: sweep_tr[q]["curve"] for q in qids_tr if q in sweep_tr}

    def train_metric(wmap):
        return float(np.mean([L.curve_value(c, wmap.get(q, 0.5)) for q, c in curves_tr.items()]))

    methods = {}  # name -> (eval weights {qid: w}, single-query predict fn for timing)

    # --- static references
    mow = round(float(np.mean(mid)), 2) if len(mid) else 0.5
    mean_curve = np.mean(np.stack(list(curves_tr.values())), axis=0)
    best_idx = np.flatnonzero(mean_curve == mean_curve.max())
    tuned = L.WEIGHTS[int(best_idx[np.argmin(np.abs(best_idx - 50))])]
    methods["Standard RRF"] = ({q: 0.5 for q in qids_ev}, None)
    methods["Mean Optimal Weight"] = ({q: mow for q in qids_ev}, None)
    methods["Tuned static w*"] = ({q: tuned for q in qids_ev}, None)
    print(f"  MOW = {mow:.2f}, tuned static w* = {tuned:.2f}")

    # --- QPP-U and QPP-C for every predictor
    per_pred = []
    u_fns, c_fns = {}, {}
    for p in Q.PREDICTORS:
        fs_tr = ecdf["s_" + p](np.array([F_tr[q]["s_" + p] for q in qids_tr]))
        fd_tr = ecdf["d_" + p](np.array([F_tr[q]["d_" + p] for q in qids_tr]))
        fs_ev = ecdf["s_" + p](np.array([F_ev[q]["s_" + p] for q in qids_ev]))
        fd_ev = ecdf["d_" + p](np.array([F_ev[q]["d_" + p] for q in qids_ev]))
        u_tr = dict(zip(qids_tr, np.round(qpp_ratio(fs_tr, fd_tr), 2)))
        u_ev = dict(zip(qids_ev, np.round(qpp_ratio(fs_ev, fd_ev), 2)))

        idx = {q: i for i, q in enumerate(qids_tr)}
        x_lab = np.array([fs_tr[idx[q]] - fd_tr[idx[q]] for q in lab_q]).reshape(-1, 1)
        a, b = sat_loss_fit(x_lab, lo, hi, 1e-4, logit(mow)) if len(lab) else (0.0, 0.0)
        c_tr = dict(zip(qids_tr, np.round(sigmoid(a * (fs_tr - fd_tr) + b), 2)))
        c_ev = dict(zip(qids_ev, np.round(sigmoid(a * (fs_ev - fd_ev) + b), 2)))

        def u_fn(item, p=p):
            s, d, _ = item
            fs = ecdf["s_" + p](Q.predict(s[1], p))
            fd = ecdf["d_" + p](Q.predict(d[1], p))
            return round(float(qpp_ratio(fs, fd)), 2)

        def c_fn(item, p=p, a=a, b=b):
            s, d, _ = item
            x = ecdf["s_" + p](Q.predict(s[1], p)) - ecdf["d_" + p](Q.predict(d[1], p))
            return round(float(sigmoid(a * x + b)), 2)

        u_fns[p], c_fns[p] = u_fn, c_fn
        per_pred.append({"predictor": p, "u_train": train_metric(u_tr), "c_train": train_metric(c_tr),
                         "c_a": float(a), "c_b": float(b), "u_ev": u_ev, "c_ev": c_ev})

    by = {r["predictor"]: r for r in per_pred}
    best_u = max(per_pred, key=lambda r: r["u_train"])["predictor"]
    best_c = max(per_pred, key=lambda r: r["c_train"])["predictor"]
    methods[f"QPP-U [{DEFAULT_PREDICTOR}]"] = (by[DEFAULT_PREDICTOR]["u_ev"], u_fns[DEFAULT_PREDICTOR])
    methods[f"QPP-U [train-best: {best_u}]"] = (by[best_u]["u_ev"], u_fns[best_u])
    methods[f"QPP-C [train-best: {best_c}]"] = (by[best_c]["c_ev"], c_fns[best_c])
    print(f"  train-best predictor: QPP-U {best_u}, QPP-C {best_c}")

    # --- multi-feature supervised models
    if len(lab) >= 10:
        X_lab = mat(F_tr, lab_q, feat_names)
        X_ev = mat(F_ev, qids_ev, feat_names)
        lin = SatLinear().fit(X_lab, lo, hi, logit(mow), folds=min(5, len(lab)))
        methods["QPP-LR (sat. loss)"] = (dict(zip(qids_ev, np.round(lin.predict(X_ev), 2))),
                                         lambda it: round(float(lin.predict(mat({0: query_features(*it)}, [0], feat_names))[0]), 2))
        gbr = make_gbr(len(lab)).fit(X_lab, mid)
        methods["QPP-GBR (MSE)"] = (dict(zip(qids_ev, np.round(np.clip(gbr.predict(X_ev), 0, 1), 2))),
                                    lambda it: round(float(np.clip(gbr.predict(mat({0: query_features(*it)}, [0], feat_names)), 0, 1)[0]), 2))
        print(f"  QPP-LR lambda = {lin.lam:g}")
        coefs = dict(zip(feat_names, lin.theta[:-1].tolist()))
    else:
        coefs = {}
        print("  fewer than 10 labelled training queries: skipping QPP-LR / QPP-GBR")

    # --- fuse + evaluate every method through helper_4 / helper_5
    qrels_ev = L.qrels_path(root, dataset, eval_split)
    out = os.path.join(args.out_dir)
    pq = {}
    for name, (wmap, _) in methods.items():
        slug = name.split(" [")[0].split(" (")[0].replace(" ", "-").replace("*", "star").lower()
        if "[" in name:
            slug += "_" + name.split("[")[1].rstrip("]").replace("train-best: ", "best-").replace("@", "")
        pred_csv = os.path.join(out, "predictions", f"{tag}_{eval_split}_{slug}.csv")
        trec = os.path.join(out, "fused", slug, f"{dataset}_{sparse}_{dense}_{eval_split}.trec")
        os.makedirs(os.path.dirname(trec), exist_ok=True)
        write_predictions(pred_csv, wmap, qtexts)
        with open(os.devnull, "w") as devnull:
            stdout, sys.stdout = sys.stdout, devnull
            try:
                fuse.fuse_search_results_predicted_weights(paths[(sparse, eval_split)], paths[(dense, eval_split)],
                                                           pred_csv, trec, k=60, run_id=slug)
            finally:
                sys.stdout = stdout
        pq[name] = per_query_metric(trec, qrels_ev, metric)

    qs = list(pq["Standard RRF"].keys())
    rrf = np.array([pq["Standard RRF"][q] for q in qs])
    mowv = np.array([pq["Mean Optimal Weight"][q] for q in qs])
    oracle = np.array([sweep_ev[q]["best"] if q in sweep_ev else 0.0 for q in qs])
    o_mean, r_mean = float(oracle.mean()), float(rrf.mean())

    # Timing: weight prediction from the two in-memory score lists (+ query text), batch size 1.
    timing_items = [(s_ev.get(q, EMPTY), d_ev.get(q, EMPTY), None if qtexts is None else qtexts.get(q, ""))
                    for q in qids_ev]
    rows = [{"dataset": dataset, "sparse": sparse, "dense": dense, "split": eval_split, "metric": f"{metric}@10",
             "method": "Oracle (per-query best w)", "mean": o_mean, "ci_lo": "", "ci_hi": "", "p_vs_rrf": "",
             "p_vs_mow": "", "recovery_pct": 100.0, "latency_ms": "", "latency_ci95_ms": "", "n_queries": len(qs)}]
    for name, (wmap, fn) in methods.items():
        v = np.array([pq[name][q] for q in qs])
        lo_ci, hi_ci = bootstrap_ci(v)
        lat, half = time_per_query(fn, timing_items) if fn else (0.0, 0.0)
        rows.append({
            "dataset": dataset, "sparse": sparse, "dense": dense, "split": eval_split, "metric": f"{metric}@10",
            "method": name, "mean": float(v.mean()), "ci_lo": lo_ci, "ci_hi": hi_ci,
            "p_vs_rrf": "" if name == "Standard RRF" else paired_p(v, rrf),
            "p_vs_mow": "" if name in ("Standard RRF", "Mean Optimal Weight") else paired_p(v, mowv),
            "recovery_pct": 100.0 * (float(v.mean()) - r_mean) / (o_mean - r_mean) if o_mean > r_mean else math.nan,
            "latency_ms": lat, "latency_ci95_ms": half, "n_queries": len(qs),
            "mean_weight": float(np.mean(list(wmap.values()))),
        })
    for r in rows:
        star = "*" if isinstance(r["p_vs_rrf"], float) and r["p_vs_rrf"] < 0.05 and r["mean"] > r_mean else " "
        rec = f"{r['recovery_pct']:6.1f}%" if isinstance(r["recovery_pct"], float) and not math.isnan(r["recovery_pct"]) else ""
        print(f"  {r['method']:<34} {r['mean']:.4f}{star} rec {rec}")

    pred_rows = []
    for r in per_pred:
        for kind in ("u", "c"):
            wmap = r[f"{kind}_ev"]
            v = np.array([L.curve_value(sweep_ev[q]["curve"], wmap.get(q, 0.5)) if q in sweep_ev else 0.0 for q in qs])
            pred_rows.append({"dataset": dataset, "sparse": sparse, "dense": dense, "predictor": r["predictor"],
                              "variant": "QPP-U" if kind == "u" else "QPP-C", "train_metric": r[f"{kind}_train"],
                              "eval_metric_from_sweep": float(v.mean()), "p_vs_rrf": paired_p(v, rrf),
                              "calib_a": r["c_a"] if kind == "c" else "", "calib_b": r["c_b"] if kind == "c" else ""})
    return rows, pred_rows, coefs


def write_rows(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main():
    # helper_4 orders tied documents by Python set iteration, which depends on the per-process
    # string-hash seed; pin it so fused runs (and the doc-id hashes used here) are reproducible.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(L.DATASETS), choices=list(L.DATASETS))
    ap.add_argument("--sparse", nargs="+", default=["bm25", "rm3"])
    ap.add_argument("--dense", nargs="+", default=["biencoder", "qwen3"])
    ap.add_argument("--data_root", default=os.environ.get("BASE_DATA_DIR", os.path.join(L.REPO_ROOT, "dataset")))
    ap.add_argument("--out_dir", default=os.path.join(L.REPO_ROOT, "experiments", "qpp-fusion"))
    args = ap.parse_args()
    np.random.seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    summary, preds, coefs = [], [], {}
    for ds in args.datasets:
        for s in args.sparse:
            for d in args.dense:
                res = run_pair(args, ds, s, d)
                if res:
                    summary += res[0]
                    preds += res[1]
                    coefs[f"{ds}_{s}_vs_{d}"] = res[2]
        _RUN_CACHE.clear()
    write_rows(os.path.join(args.out_dir, "summary.csv"), summary)
    write_rows(os.path.join(args.out_dir, "qpp_predictors.csv"), preds)
    with open(os.path.join(args.out_dir, "qpp_lr_coefficients.json"), "w") as f:
        json.dump(coefs, f, indent=1)


if __name__ == "__main__":
    main()
