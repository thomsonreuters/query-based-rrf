import os

def load_split_queries(qrels_dir):
    splits = {}
    for split_name in ['train', 'dev', 'test']:
        tsv_path = os.path.join(qrels_dir, f'{split_name}.tsv')
        if not os.path.exists(tsv_path):
            print(f"Skipping {split_name}: file {tsv_path} not found.")
            continue
        split_qids = set()
        with open(tsv_path, 'r', encoding='utf-8') as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 1:
                    continue
                query_id = parts[0]
                split_qids.add(query_id)
        splits[split_name] = split_qids
        print(f"{split_name}: {len(split_qids)} unique queries loaded.")
    return splits

def check_overlap(splits):
    names = list(splits.keys())
    overlap_found = False
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            set1, set2 = splits[names[i]], splits[names[j]]
            overlap = set1 & set2
            if overlap:
                overlap_found = True
                print(f"WARNING: Overlap between {names[i]} and {names[j]}: {len(overlap)} query-ids.")
                if len(overlap) <= 10:
                    print("Overlapping query-ids:", ', '.join(list(overlap)))
                else:
                    print("First 10 overlapping query-ids:", ', '.join(list(overlap)[:10]))
    if not overlap_found:
        print("No overlap between train, dev, and test query-ids.")
    return overlap_found

def split_trec_results_by_query(trec_result_path, splits, output_dir):
    out_files = {
        split: open(os.path.join(output_dir, f'results_{split}.trec'), 'w', encoding='utf-8')
        for split in splits
    }

    with open(trec_result_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            query_id = line.split()[0]
            for split, qids in splits.items():
                if query_id in qids:
                    out_files[split].write(line)
                    break  # Each query_id should belong to only one split

    for f in out_files.values():
        f.close()

def split_trec_by_qrels(trec_result_path, qrels_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    splits = load_split_queries(qrels_dir)
    overlap_found = check_overlap(splits)
    if overlap_found:
        print("Please resolve overlaps in your qrels files before proceeding!")
    split_trec_results_by_query(trec_result_path, splits, output_dir)
    print("Splitting complete. Files saved in", output_dir)

# Example usage:
# split_trec_by_qrels('results.trec', 'qrels/', 'split_results/')

dataset = 'acord-entire-corpus'
retriever = 'qwen3' #bm25, biencoder, rm3, qwen3
top_k = 1000

split_trec_by_qrels(
    trec_result_path=f'dataset/{dataset}/{dataset}_{retriever}_top{top_k}.trec',
    qrels_dir=f'dataset/{dataset}/qrels',
    output_dir=f'dataset/{dataset}/search_results/{retriever}/top{top_k}'
)