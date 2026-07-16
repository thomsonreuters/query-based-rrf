#!/usr/bin/env python3
"""
For each dataset and a list of methods, compute per-query text features
(length, rarity, WH-type, ambiguity, entities, ...) and correlate each
feature with the good_prediction / weak_prediction label from
top_bottom_predictions/<method>/.../{dataset}_top_bottom_predictions.csv.

The unit of analysis is a ROW of that file (one query x retriever-combo
instance), not a deduplicated query, so a query flagged weak/good by
several combos contributes several observations -- consistent with how
the exploratory analysis (see 3-analysis_msmarco_good_weak.md) was done.

Correlation is point-biserial (== Pearson r with a 0/1 label), the same
statistic used in Mothe & Tanguy, "Linguistic features to predict query
difficulty" (SIGIR 2005), which correlated linguistic features (including
average WordNet polysemy) against recall/precision. Here the outcome is
the binary weak_prediction label instead. For features that are themselves
binary (e.g. has_digit, has_entity), Pearson r on 0/1 data is equivalent to
the phi coefficient -- a legitimate effect size -- but its p-value relies on
a large-sample t-approximation that's unreliable when a binary feature is
rare (small contingency-table cell counts). So binary features additionally
get a Fisher's exact test p-value and odds ratio, computed alongside (not
instead of) the Pearson r/p, so nothing is hidden.

Output: query_features_analysis/{dataset}_feature_correlations.csv with
one row per (method, feature), plus a pooled "ALL_METHODS_POOLED" method
that concatenates rows across all selected methods for that dataset. Also
query_features_analysis/{dataset}_feature_group_means.csv with the raw
per-feature mean (or rate, for binary features) split by good_prediction
vs. weak_prediction, for interpreting the direction/magnitude behind each r.

Usage:
    python 3-query-features-analysis.py --datasets msmarco --methods 01 10
    python 3-query-features-analysis.py --datasets msmarco nq --methods all
    python 3-query-features-analysis.py --datasets msmarco --methods 01 02 03 04 05 06 09 10 --rare-threshold 3
    python 3-query-features-analysis.py --datasets msmarco --output-root /path/to/results
    python 3-query-features-analysis.py --datasets acord-entire-corpus --with-ambiguity-llm

--with-ambiguity-llm adds an LLM-scored ambiguity category (0-8, via
utils.label_queries -- see utils.py for the CLAMBER-derived prompt) computed
ONLY for the queries in top_bottom_predictions/ (not the full sorted_scores
corpus, since LLM calls cost money/time unlike the local text features
above). Results are cached in {dataset}_ambiguity_labels.csv so re-running
only labels new queries. Adds a binary `is_ambiguous` column to the normal
point-biserial feature set. `ambiguity_category` itself is a 9-level nominal
variable, not continuous/binary, so it's scored with Cramer's V + chi-square
instead of point-biserial r -- its row lands in the same
{dataset}_feature_correlations.csv, with the point-biserial-only columns
(is_binary, r, p_value_pearson, p_value_fisher_exact, odds_ratio) blank for
that row, and the categorical-only columns (cramers_v, chi2, dof) blank for
every other row.

Adding a new feature later: continuous/binary features go in FEATURE_COLUMNS
(compute_correlations scores them via point-biserial r); nominal/categorical
features go in CATEGORICAL_FEATURE_COLUMNS (compute_categorical_correlations
scores them via Cramer's V). Both lists are static registries -- a feature
column missing from df (e.g. an optional one computed only under some flag)
is silently skipped, so main()'s loop doesn't need special-casing per feature.
"""
import argparse
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, pearsonr

import utils

ROOT = Path(__file__).resolve().parent
SORTED_DIRNAME = "sorted_scores"
TOP_BOTTOM_DIRNAME = "top_bottom_predictions"
OUTPUT_DIRNAME = "query_features_analysis"
DATASETS = ["acord-entire-corpus", "msmarco", "nfcorpus", "nq"]
TOP_BOTTOM_SUFFIX = "_top_bottom_predictions.csv"

WH_WORDS = {"what", "who", "how", "why", "when", "where", "which", "whom", "whose"}
STOPWORDS = {
    "what", "who", "how", "why", "when", "where", "which", "is", "are", "was", "were",
    "do", "does", "did", "the", "a", "an", "of", "in", "on", "for", "to", "and", "or",
    "you", "your", "it", "can", "will", "be", "have", "has", "with", "that", "this",
    "from", "at", "as", "by",
}

TOKEN_RE = re.compile(r"[a-z0-9']+")


# ---------------------------------------------------------------------------
# Discovery helpers (methods / datasets / files), mirroring scripts 1 & 2.
# ---------------------------------------------------------------------------

def discover_methods(top_bottom_dir: Path):
    return sorted(p for p in top_bottom_dir.iterdir() if p.is_dir())


def method_matches(method_dir: Path, selected):
    if selected is None:
        return True
    name = method_dir.name
    for sel in selected:
        if sel == name:
            return True
        if sel.isdigit() and name.startswith(f"{int(sel):02d}-"):
            return True
    return False


def dataset_of(filename: str):
    for ds in sorted(DATASETS, key=len, reverse=True):
        if filename == ds or filename.startswith(ds + "-") or filename.startswith(ds + "_"):
            return ds
    return None


def find_top_bottom_file(method_dir: Path, dataset: str):
    for scores_dir in method_dir.glob("**/prediction-scores"):
        candidate = scores_dir / f"{dataset}{TOP_BOTTOM_SUFFIX}"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Corpus construction: union of unique (query_id, query_text) across all
# combo files, for the given dataset, under the given methods' sorted_scores
# trees. Used only to compute corpus-wide word document frequencies for the
# rarity feature -- query text itself is method-agnostic (verified: same
# query_id -> same query_text across every method for a given dataset+combo).
# ---------------------------------------------------------------------------

def build_dataset_corpus(dataset: str, method_dirs, sorted_dir: Path) -> pd.DataFrame:
    texts = {}
    for method_dir in method_dirs:
        sorted_method_dir = sorted_dir / method_dir.name
        if not sorted_method_dir.exists():
            continue
        for scores_dir in sorted_method_dir.glob("**/prediction-scores"):
            for csv_path in scores_dir.glob("*.csv"):
                if csv_path.name.endswith(TOP_BOTTOM_SUFFIX):
                    continue
                if dataset_of(csv_path.name) != dataset:
                    continue
                df = pd.read_csv(csv_path, usecols=["query_id", "query_text"])
                for qid, text in zip(df["query_id"], df["query_text"]):
                    texts.setdefault(qid, text)
    return pd.DataFrame(list(texts.items()), columns=["query_id", "query_text"])


# ---------------------------------------------------------------------------
# LLM ambiguity feature (opt-in via --with-ambiguity-llm). Scoped to only the
# queries in top_bottom_predictions/ -- unlike the corpus above, LLM calls
# cost money/time, so this does not touch the full sorted_scores corpus.
# ---------------------------------------------------------------------------

def gather_top_bottom_queries(dataset: str, method_dirs) -> pd.DataFrame:
    """Union of unique (query_id, query_text) across method_dirs' top_bottom_predictions
    files for `dataset`."""
    texts = {}
    for method_dir in method_dirs:
        path = find_top_bottom_file(method_dir, dataset)
        if path is None:
            continue
        df = pd.read_csv(path, usecols=["query_id", "query_text"])
        for qid, text in zip(df["query_id"], df["query_text"]):
            texts.setdefault(qid, text)
    return pd.DataFrame(list(texts.items()), columns=["query_id", "query_text"])


def load_or_compute_ambiguity_labels(
    dataset: str, queries_df: pd.DataFrame, output_dir: Path,
    model_name: str, concurrency: int, chunk_size: int,
) -> pd.DataFrame:
    """Returns queries_df's query_ids mapped to an LLM-scored ambiguity_category (0-8),
    caching results in {dataset}_ambiguity_labels.csv so re-running only labels queries
    not already validly cached. Each chunk is appended to the cache as soon as it
    completes (see utils.label_queries' on_chunk), so a failure partway through a large
    run doesn't lose already-completed labels."""
    cache_path = output_dir / f"{dataset}_ambiguity_labels.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        cached_valid = cached[cached["ambiguity_category"].notna()]
    else:
        cached_valid = pd.DataFrame(columns=["query_id", "query_text", "ambiguity_category", "raw_response"])

    already_done = set(cached_valid["query_id"])
    remaining = queries_df[~queries_df["query_id"].isin(already_done)]
    print(f"[{dataset}] ambiguity labels: {len(already_done)} cached, {len(remaining)} to label")

    if remaining.empty:
        return cached_valid.set_index("query_id")[["ambiguity_category"]]

    accumulated = [cached_valid]

    def on_chunk(chunk_result: pd.DataFrame):
        accumulated.append(chunk_result)
        pd.concat(accumulated, ignore_index=True).to_csv(cache_path, index=False)
        print(f"[{dataset}] ambiguity labels: checkpointed {len(chunk_result)} more "
              f"({sum(len(c) for c in accumulated)} total)")

    utils.label_queries(remaining, model_name=model_name, concurrency=concurrency,
                         chunk_size=chunk_size, on_chunk=on_chunk)

    final = pd.read_csv(cache_path)
    return final[final["ambiguity_category"].notna()].set_index("query_id")[["ambiguity_category"]]


def compute_categorical_correlation(df: pd.DataFrame, feature_col: str, alpha: float,
                                     label_col="prediction_quality") -> dict:
    """Cramer's V + chi-square test of independence between a nominal feature
    (e.g. ambiguity_category, a 9-level category, not continuous/binary) and
    the good/weak label. Not point-biserial r -- that statistic assumes an
    ordinal/continuous feature, which a set of unordered categories is not."""
    sub = df[[feature_col, label_col]].dropna()
    if sub.empty or sub[feature_col].nunique() < 2 or sub[label_col].nunique() < 2:
        return {"feature": feature_col, "cramers_v": float("nan"), "p_value_categorical": float("nan"),
                "chi2": float("nan"), "dof": float("nan"), "n": len(sub),
                f"significant_p<{alpha}": False}

    table = pd.crosstab(sub[feature_col], sub[label_col])
    chi2, p_value, dof, _ = chi2_contingency(table)
    n = table.values.sum()
    min_dim = min(table.shape) - 1
    cramers_v = (chi2 / (n * min_dim)) ** 0.5 if min_dim > 0 else float("nan")
    return {"feature": feature_col, "cramers_v": cramers_v, "p_value_categorical": p_value,
            "chi2": chi2, "dof": dof, "n": int(n),
            f"significant_p<{alpha}": bool(pd.notna(p_value) and p_value < alpha)}


def tokenize(text: str):
    return TOKEN_RE.findall(str(text).lower())


def compute_doc_frequencies(corpus_df: pd.DataFrame):
    doc_freq = Counter()
    for text in corpus_df["query_text"]:
        for tok in set(tokenize(text)):
            doc_freq[tok] += 1
    return doc_freq, len(corpus_df)


# ---------------------------------------------------------------------------
# Per-query feature functions. Each takes the tokenized query (and, where
# needed, corpus stats or a spaCy doc) and returns a single scalar. Kept
# small and independently testable rather than one monolithic computation.
# ---------------------------------------------------------------------------

def word_count(tokens):
    return len(tokens)


def char_count(text):
    return len(str(text))


def avg_word_length(tokens):
    return sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0


def is_wh_question(tokens):
    return bool(tokens) and tokens[0] in WH_WORDS


def has_digit(tokens):
    return any(t.isdigit() for t in tokens)


def n_digit_tokens(tokens):
    return sum(1 for t in tokens if t.isdigit())


def content_word_frac(tokens):
    return sum(1 for t in tokens if t not in STOPWORDS) / len(tokens) if tokens else 0.0


def avg_rarity(tokens, doc_freq, n_docs):
    if not tokens:
        return 0.0
    return sum(-math.log(doc_freq.get(t, 1) / n_docs) for t in tokens) / len(tokens)


def frac_rare_words(tokens, doc_freq, rare_threshold):
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if doc_freq.get(t, 0) <= rare_threshold) / len(tokens)


def has_singleton_word(tokens, doc_freq):
    return any(doc_freq.get(t, 0) == 1 for t in tokens)


def ambiguity_score(tokens, wordnet):
    """Average WordNet polysemy (# synsets per word), default 1 for OOV words.
    Formula from Mothe & Tanguy, SIGIR 2005 (their 'SYNSETS' feature)."""
    if not tokens:
        return 0.0
    return sum(len(wordnet.synsets(t)) or 1 for t in tokens) / len(tokens)


def ends_with_question_mark(text):
    return str(text).strip().endswith("?")


# Continuous/binary features, scored by compute_correlations() (point-biserial).
# To add a new one: compute it (in compute_query_features, or merge its own
# frame in main()), then add its column name here -- nothing else changes.
FEATURE_COLUMNS = [
    "word_count", "char_count", "avg_word_length", "is_wh_question", "has_digit",
    "n_digit_tokens", "content_word_frac", "avg_rarity", "frac_rare_words",
    "has_singleton_word", "ambiguity_score", "n_entities", "has_entity",
    "n_proper_nouns", "ends_with_question_mark",
    "is_ambiguous",  # derived from ambiguity_category (see load_or_compute_ambiguity_labels)
]

# Nominal (unordered-category) features, scored by compute_categorical_correlations()
# (Cramer's V + chi-square, not point-biserial -- see compute_categorical_correlation's
# docstring for why). Same registry pattern as FEATURE_COLUMNS.
CATEGORICAL_FEATURE_COLUMNS = ["ambiguity_category"]


def compute_query_features(corpus_df: pd.DataFrame, rare_threshold: int) -> pd.DataFrame:
    """Returns a DataFrame indexed by query_id with one column per FEATURE_COLUMNS entry."""
    import spacy
    from nltk.corpus import wordnet

    doc_freq, n_docs = compute_doc_frequencies(corpus_df)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

    rows = []
    texts = corpus_df["query_text"].astype(str).tolist()
    for (qid, text), spacy_doc in zip(
        zip(corpus_df["query_id"], texts), nlp.pipe(texts, batch_size=256)
    ):
        tokens = tokenize(text)
        n_ent = len(spacy_doc.ents)
        rows.append({
            "query_id": qid,
            "word_count": word_count(tokens),
            "char_count": char_count(text),
            "avg_word_length": avg_word_length(tokens),
            "is_wh_question": is_wh_question(tokens),
            "has_digit": has_digit(tokens),
            "n_digit_tokens": n_digit_tokens(tokens),
            "content_word_frac": content_word_frac(tokens),
            "avg_rarity": avg_rarity(tokens, doc_freq, n_docs),
            "frac_rare_words": frac_rare_words(tokens, doc_freq, rare_threshold),
            "has_singleton_word": has_singleton_word(tokens, doc_freq),
            "ambiguity_score": ambiguity_score(tokens, wordnet),
            "n_entities": n_ent,
            "has_entity": n_ent > 0,
            "n_proper_nouns": sum(1 for t in spacy_doc if t.pos_ == "PROPN"),
            "ends_with_question_mark": ends_with_question_mark(text),
        })
    return pd.DataFrame(rows).set_index("query_id")


# ---------------------------------------------------------------------------
# Correlation computation.
# ---------------------------------------------------------------------------

def is_binary_series(x: pd.Series) -> bool:
    vals = x.dropna().unique()
    return len(vals) > 0 and set(vals).issubset({0, 1})


def fisher_exact_for_binary(x: pd.Series, y: pd.Series):
    """2x2 contingency table (feature x label, both 0/1) -> (odds_ratio, p_value)."""
    table = pd.crosstab(x, y).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    return fisher_exact(table.values)


def compute_correlations(df: pd.DataFrame, feature_cols, label_col="prediction_quality",
                          positive_label="weak_prediction", alpha=0.05) -> pd.DataFrame:
    """Point-biserial correlation (Pearson r on the 0/1 label) between each
    feature and the weak_prediction label -- for binary features this r is
    the phi coefficient. NaN r/p when a feature has no variance or too few
    non-null rows. Binary features additionally get a Fisher's exact p-value
    and odds ratio (more reliable than the Pearson t-approximation when a
    binary feature is rare), computed alongside, not instead of, Pearson.
    Columns in feature_cols not present in df are silently skipped -- lets
    the caller pass a static registry regardless of which optional features
    were actually computed for this run."""
    y = (df[label_col] == positive_label).astype(float)
    rows = []
    for feat in feature_cols:
        if feat not in df.columns:
            continue
        x = df[feat].astype(float)
        mask = x.notna() & y.notna()
        n = int(mask.sum())
        binary = is_binary_series(x[mask])
        if n < 3 or x[mask].nunique() < 2 or y[mask].nunique() < 2:
            r, p_pearson = float("nan"), float("nan")
        else:
            r, p_pearson = pearsonr(x[mask], y[mask])

        odds_ratio, p_fisher = float("nan"), float("nan")
        if binary and n >= 3 and x[mask].nunique() == 2 and y[mask].nunique() == 2:
            odds_ratio, p_fisher = fisher_exact_for_binary(x[mask], y[mask])

        primary_p = p_fisher if binary and pd.notna(p_fisher) else p_pearson
        rows.append({
            "feature": feat,
            "is_binary": binary,
            "r": r,
            "p_value_pearson": p_pearson,
            "p_value_fisher_exact": p_fisher,
            "odds_ratio": odds_ratio,
            "n": n,
            f"significant_p<{alpha}": bool(pd.notna(primary_p) and primary_p < alpha),
        })
    result = pd.DataFrame(rows)
    return result.reindex(result["r"].abs().sort_values(ascending=False, na_position="last").index)


def compute_group_means(df: pd.DataFrame, feature_cols, label_col="prediction_quality") -> pd.DataFrame:
    """Raw per-feature mean (a rate, for binary features) split by label group --
    gives the direction/magnitude behind each r, since a correlation coefficient
    alone doesn't show the underlying values. Columns not present in df are
    silently skipped, same as compute_correlations."""
    present_cols = [c for c in feature_cols if c in df.columns]
    means = df.groupby(label_col)[present_cols].mean().T
    means.index.name = "feature"
    means["diff (weak - good)"] = means.get("weak_prediction", float("nan")) - means.get("good_prediction", float("nan"))
    return means.reset_index()


def compute_categorical_correlations(df: pd.DataFrame, feature_cols, alpha=0.05,
                                      label_col="prediction_quality") -> pd.DataFrame:
    """Cramer's V + chi-square for each nominal feature in feature_cols that's
    present in df (see compute_categorical_correlation for the per-feature
    statistic and why it differs from compute_correlations' point-biserial r).
    Same registry-driven, skip-if-absent pattern as compute_correlations."""
    rows = [
        compute_categorical_correlation(df, feat, alpha, label_col=label_col)
        for feat in feature_cols if feat in df.columns
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", required=True, choices=DATASETS,
                         help="Dataset(s) to analyze.")
    parser.add_argument("--methods", nargs="+", default=None,
                         help="Method folder names or numeric prefixes (e.g. 01-standard-rrf or 01). "
                              "Default: all methods found under top_bottom_predictions/.")
    parser.add_argument("--rare-threshold", type=int, default=3,
                         help="Doc-frequency <= this counts as 'rare' for frac_rare_words. Default: 3.")
    parser.add_argument("--alpha", type=float, default=0.05,
                         help="Significance threshold for flagging correlations. Default: 0.05.")
    parser.add_argument("--output-root", type=Path, default=ROOT,
                         help=f"Directory containing '{SORTED_DIRNAME}/' and '{TOP_BOTTOM_DIRNAME}/' (steps 1 & 2's "
                              f"output) and under which '{OUTPUT_DIRNAME}/' is created. "
                              f"Default: this script's directory.")
    parser.add_argument("--with-ambiguity-llm", action="store_true",
                         help="Add an LLM-scored ambiguity_category feature (0-8, CLAMBER-derived prompt), "
                              "computed only for queries in top_bottom_predictions/. Default: off.")
    parser.add_argument("--ambiguity-model", default="gpt-4o-mini", choices=list(utils.MODEL_CHOICES),
                         help="Model used for ambiguity labeling. Default: gpt-4o-mini.")
    parser.add_argument("--ambiguity-concurrency", type=int, default=5,
                         help="Concurrent LLM calls in flight. Default: 5.")
    parser.add_argument("--ambiguity-chunk-size", type=int, default=50,
                         help="Queries per checkpointed chunk. Default: 50.")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    sorted_dir = output_root / SORTED_DIRNAME
    top_bottom_dir = output_root / TOP_BOTTOM_DIRNAME
    output_dir = output_root / OUTPUT_DIRNAME

    if not top_bottom_dir.exists():
        print(f"{top_bottom_dir} does not exist. Run 2-top_bottom_predictions_per_method.py first.")
        return

    all_methods = discover_methods(top_bottom_dir)
    selected_methods = [m for m in all_methods if method_matches(m, args.methods)]
    if not selected_methods:
        print(f"No matching method folders found under {top_bottom_dir}/.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        method_dirs = [m for m in selected_methods if find_top_bottom_file(m, dataset) is not None]
        if not method_dirs:
            print(f"[{dataset}] no method has a top_bottom_predictions file for this dataset, skipping.")
            continue

        print(f"\n=== {dataset}: building corpus + features from {len(method_dirs)} method(s) ===")
        corpus_df = build_dataset_corpus(dataset, method_dirs, sorted_dir)
        print(f"[{dataset}] corpus size: {len(corpus_df)} unique queries")
        feats_df = compute_query_features(corpus_df, args.rare_threshold)

        ambiguity_labels_df = None
        if args.with_ambiguity_llm:
            top_bottom_queries = gather_top_bottom_queries(dataset, method_dirs)
            ambiguity_labels_df = load_or_compute_ambiguity_labels(
                dataset, top_bottom_queries, output_dir,
                args.ambiguity_model, args.ambiguity_concurrency, args.ambiguity_chunk_size,
            )

        result_frames = []
        means_frames = []
        pooled_parts = []
        for method_dir in method_dirs:
            path = find_top_bottom_file(method_dir, dataset)
            df = pd.read_csv(path)
            df = df.merge(feats_df, on="query_id", how="left")
            if ambiguity_labels_df is not None:
                df = df.merge(ambiguity_labels_df, on="query_id", how="left")
                df["is_ambiguous"] = df["ambiguity_category"].apply(
                    lambda v: float("nan") if pd.isna(v) else float(v != 0)
                )

            corr = pd.concat([
                compute_correlations(df, FEATURE_COLUMNS, alpha=args.alpha),
                compute_categorical_correlations(df, CATEGORICAL_FEATURE_COLUMNS, alpha=args.alpha),
            ], ignore_index=True)
            corr.insert(0, "method", method_dir.name)
            corr.insert(1, "dataset", dataset)
            result_frames.append(corr)

            means = compute_group_means(df, FEATURE_COLUMNS)
            means.insert(0, "method", method_dir.name)
            means.insert(1, "dataset", dataset)
            means_frames.append(means)

            pooled_parts.append(df)

        pooled_df = pd.concat(pooled_parts, ignore_index=True)
        pooled_corr = pd.concat([
            compute_correlations(pooled_df, FEATURE_COLUMNS, alpha=args.alpha),
            compute_categorical_correlations(pooled_df, CATEGORICAL_FEATURE_COLUMNS, alpha=args.alpha),
        ], ignore_index=True)
        pooled_corr.insert(0, "method", "ALL_METHODS_POOLED")
        pooled_corr.insert(1, "dataset", dataset)
        result_frames.append(pooled_corr)

        pooled_means = compute_group_means(pooled_df, FEATURE_COLUMNS)
        pooled_means.insert(0, "method", "ALL_METHODS_POOLED")
        pooled_means.insert(1, "dataset", dataset)
        means_frames.append(pooled_means)

        final = pd.concat(result_frames, ignore_index=True)
        out_path = output_dir / f"{dataset}_feature_correlations.csv"
        final.to_csv(out_path, index=False)
        print(f"wrote {out_path.relative_to(output_root)}")

        final_means = pd.concat(means_frames, ignore_index=True)
        means_path = output_dir / f"{dataset}_feature_group_means.csv"
        final_means.to_csv(means_path, index=False)
        print(f"wrote {means_path.relative_to(output_root)}")

        print(f"\n--- {dataset}: ALL_METHODS_POOLED feature values by group ---")
        print(pooled_means.drop(columns=["method", "dataset"]).to_string(index=False))

        print(f"\n--- {dataset}: ALL_METHODS_POOLED correlations (top by |r|) ---")
        print(pooled_corr.drop(columns=["method", "dataset"]).to_string(index=False))

    print(f"\nDone. Results under {output_dir}/")


if __name__ == "__main__":
    main()
