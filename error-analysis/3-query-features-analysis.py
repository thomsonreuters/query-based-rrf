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
"""
import argparse
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact, pearsonr

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


FEATURE_COLUMNS = [
    "word_count", "char_count", "avg_word_length", "is_wh_question", "has_digit",
    "n_digit_tokens", "content_word_frac", "avg_rarity", "frac_rare_words",
    "has_singleton_word", "ambiguity_score", "n_entities", "has_entity",
    "n_proper_nouns", "ends_with_question_mark",
]


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
    binary feature is rare), computed alongside, not instead of, Pearson."""
    y = (df[label_col] == positive_label).astype(float)
    rows = []
    for feat in feature_cols:
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
    alone doesn't show the underlying values."""
    means = df.groupby(label_col)[feature_cols].mean().T
    means.index.name = "feature"
    means["diff (weak - good)"] = means.get("weak_prediction", float("nan")) - means.get("good_prediction", float("nan"))
    return means.reset_index()


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

        result_frames = []
        means_frames = []
        pooled_parts = []
        for method_dir in method_dirs:
            path = find_top_bottom_file(method_dir, dataset)
            df = pd.read_csv(path)
            df = df.merge(feats_df, on="query_id", how="left")

            corr = compute_correlations(df, FEATURE_COLUMNS, alpha=args.alpha)
            corr.insert(0, "method", method_dir.name)
            corr.insert(1, "dataset", dataset)
            result_frames.append(corr)

            means = compute_group_means(df, FEATURE_COLUMNS)
            means.insert(0, "method", method_dir.name)
            means.insert(1, "dataset", dataset)
            means_frames.append(means)

            pooled_parts.append(df)

        pooled_df = pd.concat(pooled_parts, ignore_index=True)
        pooled_corr = compute_correlations(pooled_df, FEATURE_COLUMNS, alpha=args.alpha)
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
