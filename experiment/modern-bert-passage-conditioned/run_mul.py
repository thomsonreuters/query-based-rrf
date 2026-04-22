import os
import multiprocessing as mp

worker_gpu_id = None

def init_worker(gpu_queue):
    global worker_gpu_id
    worker_gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_gpu_id)

def process_experiment(task_args):
    from train import main as train_model
    from test import run_test as test_model
    
    dataset, combo, base_data_dir, config_template = task_args
    
    metric = "ndcg" if dataset in ["acord-entire-corpus", "nfcorpus"] else "mrr"
    test_split = "test" if dataset in ["acord-entire-corpus", "nfcorpus"] else "dev"
    
    print(f"\n{'='*80}")
    print(f"🚀 [GPU {worker_gpu_id}] PIPELINE: Dataset={dataset} | Combo={combo} | Split={test_split}")
    print(f"{'='*80}")
    
    train_file = f"{base_data_dir}/{dataset}/{metric}_runs/train/top200/results_train_{combo}_best_weights_final_mean_with_text.csv"
    test_file = f"{base_data_dir}/{dataset}/{metric}_runs/{test_split}/top200/results_{test_split}_{combo}_best_weights_final_mean_with_text.csv"
    
    if not os.path.exists(train_file):
        print(f"⚠️ [GPU {worker_gpu_id}] Skipping: Train file not found -> {train_file}")
        return False
    
    # Parse retrievers from combo (e.g., 'bm25_vs_biencoder')
    sparse_retriever, dense_retriever = combo.split('_vs_')

    # Construct the base paths necessary for querying TREC lists and passages
    corpus_path = f"{base_data_dir}/{dataset}/corpus.jsonl"
    sparse_trec_train = f"{base_data_dir}/{dataset}/search_results/{sparse_retriever}/top200/results_train.trec"
    dense_trec_train = f"{base_data_dir}/{dataset}/search_results/{dense_retriever}/top200/results_train.trec"
    sparse_trec_test = f"{base_data_dir}/{dataset}/search_results/{sparse_retriever}/top200/results_{test_split}.trec"
    dense_trec_test = f"{base_data_dir}/{dataset}/search_results/{dense_retriever}/top200/results_{test_split}.trec"

    # Inject these custom fields down to train/test pipelines
    overrides = {
        'experiment.name': f"{dataset}-{combo}",
        'data.train_file': train_file,
        'data.test_file': test_file,
        'data.corpus_path': corpus_path,
        'data.sparse_trec_train': sparse_trec_train,
        'data.dense_trec_train': dense_trec_train,
        'data.sparse_trec_test': sparse_trec_test,
        'data.dense_trec_test': dense_trec_test
    }
    
    try:
        print(f"\n[1/2] [GPU {worker_gpu_id}] Starting Training for {dataset}-{combo}...")
        exp_dir = train_model(config_template, overrides=overrides)
        
        if os.path.exists(test_file):
            print(f"\n[2/2] [GPU {worker_gpu_id}] Starting Testing using model in {exp_dir}...")
            test_model(exp_dir, test_file_path=test_file)
        else:
            print(f"⚠️ [GPU {worker_gpu_id}] Error: Test file not found -> {test_file}")
            
        return True
    except Exception as e:
        print(f"❌ [GPU {worker_gpu_id}] Error during {dataset}-{combo}: {str(e)}")
        return False

def run_pipeline(gpus):
    base_data_dir = os.environ.get("BASE_DATA_DIR", "/extra/huaiyaom0/tr-intern/wrrf/dataset")
    config_template = "config.yaml"
    
    # datasets = ["acord-entire-corpus", "msmarco", "nfcorpus", "nq"]
    datasets = ["nfcorpus", "nq", "msmarco"]
    combinations = ["bm25_vs_biencoder", "bm25_vs_qwen3", "rm3_vs_biencoder", "rm3_vs_qwen3"]

    tasks = []
    for dataset in datasets:
        for combo in combinations:
            tasks.append((dataset, combo, base_data_dir, config_template))

    if len(gpus) > 1:
        print(f"🔥 Starting multiprocessing pool across {len(gpus)} GPUs: {gpus}")
        mp.set_start_method('spawn', force=True) 
        
        m = mp.Manager()
        gpu_queue = m.Queue()
        for gpu in gpus:
            gpu_queue.put(gpu)
            
        with mp.Pool(processes=len(gpus), initializer=init_worker, initargs=(gpu_queue,)) as pool:
            pool.map(process_experiment, tasks)
            
    else:
        gpu = gpus[0] if gpus else "0"
        print(f"🐢 Running sequentially on GPU: {gpu}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        
        global worker_gpu_id
        worker_gpu_id = gpu
        
        for task in tasks:
            process_experiment(task)

if __name__ == "__main__":
    AVAILABLE_GPUS = ["1", "2", "3", "4"]
    run_pipeline(AVAILABLE_GPUS)