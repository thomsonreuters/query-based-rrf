import os
import multiprocessing as mp

# Global variable for the worker processes to keep track of their assigned GPU
worker_gpu_id = None

def init_worker(gpu_queue):
    """
    Initializer for the multiprocessing pool. 
    Grabs a GPU ID from the queue and sets it for the current process environment.
    """
    global worker_gpu_id
    worker_gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_gpu_id)

def process_experiment(task_args):
    """
    Worker function to run a single experiment.
    Imports are done locally to ensure PyTorch respects the newly set CUDA_VISIBLE_DEVICES.
    """
    # MOVED IMPORTS HERE: Import locally to avoid prematurely initializing CUDA 
    # in the main process before the worker sets its specific GPU.
    from train import main as train_model
    from test import run_test as test_model
    
    dataset, combo, base_data_dir, config_template = task_args
    
    # Determine metric and test split based on dataset
    metric = "ndcg" if dataset in ["acord-entire-corpus", "nfcorpus"] else "mrr"
    test_split = "test" if dataset in ["acord-entire-corpus", "nfcorpus"] else "dev"
    
    print(f"\n{'='*80}")
    print(f"🚀 [GPU {worker_gpu_id}] PIPELINE: Dataset={dataset} | Combo={combo} | Split={test_split}")
    print(f"{'='*80}")
    
    # Construct exact file paths
    train_file = f"{base_data_dir}/{dataset}/{metric}_runs/train/top200/results_train_{combo}_best_weights_final_mean_with_text.csv"
    test_file = f"{base_data_dir}/{dataset}/{metric}_runs/{test_split}/top200/results_{test_split}_{combo}_best_weights_final_mean_with_text.csv"
    
    if not os.path.exists(train_file):
        print(f"⚠️ [GPU {worker_gpu_id}] Skipping: Train file not found -> {train_file}")
        return False
    
    # Dynamically override the base config.yaml for this specific run
    overrides = {
        'experiment.name': f"{dataset}-{combo}",
        'data.train_file': train_file,
        'data.test_file': test_file
    }
    
    try:
        # --- 1. TRAIN ---
        print(f"\n[1/2] [GPU {worker_gpu_id}] Starting Training for {dataset}-{combo}...")
        exp_dir = train_model(config_template, overrides=overrides)
        
        # --- 2. TEST ---
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
    
    datasets = ["acord-entire-corpus", "msmarco", "nfcorpus", "nq"]
    combinations = ["bm25_vs_biencoder", "bm25_vs_qwen3", "rm3_vs_biencoder", "rm3_vs_qwen3"]

    # Build the list of tasks to process
    tasks = []
    for dataset in datasets:
        for combo in combinations:
            tasks.append((dataset, combo, base_data_dir, config_template))

    if len(gpus) > 1:
        print(f"🔥 Starting multiprocessing pool across {len(gpus)} GPUs: {gpus}")
        # 'spawn' is mandatory for PyTorch multiprocessing to avoid CUDA initialization errors
        mp.set_start_method('spawn', force=True) 
        
        m = mp.Manager()
        gpu_queue = m.Queue()
        for gpu in gpus:
            gpu_queue.put(gpu)
            
        # Create a pool of workers matching the number of GPUs
        with mp.Pool(processes=len(gpus), initializer=init_worker, initargs=(gpu_queue,)) as pool:
            # Map tasks to the pool, chunking them out as GPUs become available
            pool.map(process_experiment, tasks)
            
    else:
        # Fallback to sequential execution if only 1 GPU (or an empty list) is provided
        gpu = gpus[0] if gpus else "0"
        print(f"🐢 Running sequentially on GPU: {gpu}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        
        global worker_gpu_id
        worker_gpu_id = gpu
        
        for task in tasks:
            process_experiment(task)

if __name__ == "__main__":
    # Specify your available GPUs here. 
    # Example: ["0", "1", "2", "3"] will run 4 experiments concurrently.
    # Example: ["2", "3"] will run 2 experiments concurrently on GPU 2 and 3.
    # Example: ["0"] will run all experiments sequentially on GPU 0.
    AVAILABLE_GPUS = ["1", "2"]
    
    run_pipeline(AVAILABLE_GPUS)