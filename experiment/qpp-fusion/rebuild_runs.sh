#!/usr/bin/env bash
# Rebuild the top-200 first-stage runs for one dataset with the retrieval-share recipe, written
# into the layout this repo reads:  $DATA_ROOT/<dataset>/search_results/<retriever>/top200/results_<split>.trec
#
# Use this only if the paper's original run files are unavailable. Rebuilt Qwen3 runs match the
# originals to about +-0.002 (see retrieval-share/README.md, section 5), so numbers from rebuilt
# runs are not exactly comparable to the existing Table 2 rows.
#
#   RSHARE=/path/to/retrieval-share DATA_ROOT=dataset WORK=/scratch/wrrf \
#     bash experiment/qpp-fusion/rebuild_runs.sh nq bm25 rm3 biencoder qwen3
#
# Datasets: acord-entire-corpus | nq | msmarco | nfcorpus. Retrievers: bm25 rm3 biencoder qwen3.
# Needs Java 21 + pyserini 1.6.0 for bm25/rm3, and one GPU (>= 24 GB for qwen3) for the dense
# retrievers. Qwen3 corpus encoding takes ~20 GPU-hours for NQ and ~35 for MS MARCO on RTX 4090s;
# it writes shards and resumes if interrupted.
set -euo pipefail
DS=$1; shift
RETRIEVERS=("$@")
: "${RSHARE:?set RSHARE to the retrieval-share folder}"
DATA_ROOT=${DATA_ROOT:-dataset}
WORK=${WORK:-work}
THREADS=${THREADS:-$(nproc)}
PY=${PYTHON:-python}

D=$(realpath "$DATA_ROOT/$DS")
OUT=$(realpath -m "$DATA_ROOT/$DS/search_results")
mkdir -p "$WORK"
WORK=$(realpath "$WORK")

case $DS in
  msmarco|nq) SPLITS=(dev train) ;;
  acord-entire-corpus|nfcorpus) SPLITS=(test dev train) ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
case $DS in  # title handling differs by dataset (retrieval-share README, section 4)
  nq|nfcorpus) IDX_ARGS=(--doc_format text --title_field); BM25_FIELDS=(--fields contents=1.0 title=1.0) ;;
  *) IDX_ARGS=(--doc_format text); BM25_FIELDS=() ;;
esac

cd "$RSHARE"
for R in "${RETRIEVERS[@]}"; do
  echo "=== $DS / $R ==="
  case $R in
    bm25|rm3)
      [ -d "$WORK/index/$DS/lucene" ] || $PY sparse_pyserini.py index --data_dir "$D" --index_dir "$WORK/index/$DS" \
          "${IDX_ARGS[@]}" --threads "$THREADS"
      EXTRA=(); [ "$R" = bm25 ] && EXTRA=("${BM25_FIELDS[@]}")
      $PY sparse_pyserini.py search --data_dir "$D" --index_dir "$WORK/index/$DS" --method "$R" "${EXTRA[@]}" \
          --splits "${SPLITS[@]}" --topk 200 --threads "$THREADS" --out_dir "$OUT/$R/top200"
      ;;
    biencoder)
      M=sentence-transformers/all-MiniLM-L6-v2; E="$WORK/emb/${DS}_minilm"
      $PY dense_retrieve.py encode-corpus --data_dir "$D" --emb_dir "$E" --model $M --dtype fp32 \
          --store_dtype fp32 --doc_format title_bracket_text --batch_size 128 --token_budget 30000
      $PY dense_retrieve.py encode-queries --data_dir "$D" --emb_dir "$E" --model $M --dtype fp32 --splits "${SPLITS[@]}"
      $PY dense_retrieve.py search --emb_dir "$E" --splits "${SPLITS[@]}" --topk 200 --out_dir "$OUT/biencoder/top200" --tag biencoder
      ;;
    qwen3)
      M=Qwen/Qwen3-Embedding-8B; E="$WORK/emb/${DS}_qwen3"
      QA=(--model $M --dtype fp16 --max_seq_length 512 --batch_size 32 --token_budget 12000)
      $PY dense_retrieve.py encode-corpus --data_dir "$D" --emb_dir "$E" "${QA[@]}" --last_position_batch 16 \
          --doc_format title_space_text --store_dtype fp32
      if [ "$DS" = acord-entire-corpus ]; then
        $PY dense_retrieve.py encode-queries --data_dir "$D" --emb_dir "$E" "${QA[@]}" --last_position_batch 16 --splits "${SPLITS[@]}"
      else
        $PY dense_retrieve.py encode-queries --data_dir "$D" --emb_dir "$E" "${QA[@]}" --query_prompt_name query --splits "${SPLITS[@]}"
      fi
      if ls "$E"/*.bad.json >/dev/null 2>&1; then $PY dense_retrieve.py repair --data_dir "$D" --emb_dir "$E"; fi
      $PY dense_retrieve.py search --emb_dir "$E" --splits "${SPLITS[@]}" --topk 200 --out_dir "$OUT/qwen3/top200" --tag Qwen3-Embedding-8B
      ;;
    *) echo "unknown retriever $R"; exit 1 ;;
  esac
done

# Check against the paper's single-retriever scores (retrieval-share README, section 5).
case $DS in msmarco|nq) MET=mrr; EV=dev ;; *) MET=ndcg; EV=test ;; esac
for R in "${RETRIEVERS[@]}"; do
  $PY eval_run.py --metric $MET --run "$OUT/$R/top200/results_$EV.trec" --qrels "$D/qrels/$EV.tsv"
done
