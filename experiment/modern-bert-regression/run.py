import os
# We will import the adjusted main functions from your files
from train import main as train_model
from test import run_test as test_model

def run_pipeline():
    # Base paths and parameters
    base_data_dir = os.environ.get("BASE_DATA_DIR", "/extra/huaiyaom0/tr-intern/wrrf/dataset")
    config_template = "config.yaml"
    
    datasets = ["acord-entire-corpus", "msmarco", "nfcorpus", "nq"]
    combinations = ["bm25_vs_biencoder", "bm25_vs_qwen3", "rm3_vs_biencoder", "rm3_vs_qwen3"]

    for dataset in datasets:
        # Determine metric and test split based on dataset
        metric = "ndcg" if dataset in ["acord-entire-corpus", "nfcorpus"] else "mrr"
        test_split = "test" if dataset in ["acord-entire-corpus", "nfcorpus"] else "dev"
        
        for combo in combinations:
            print(f"\n{'='*80}")
            print(f"🚀 PIPELINE: Dataset={dataset} | Combo={combo} | Split={test_split}")
            print(f"{'='*80}")
            
            # Construct exact file paths
            train_file = f"{base_data_dir}/{dataset}/{metric}_runs/train/top200/results_train_{combo}_best_weights_final_mean_with_text.csv"
            test_file = f"{base_data_dir}/{dataset}/{metric}_runs/{test_split}/top200/results_{test_split}_{combo}_best_weights_final_mean_with_text.csv"
            
            if not os.path.exists(train_file):
                print(f"⚠️ Skipping: Train file not found -> {train_file}")
                continue
            
            # Dynamically override the base config.yaml for this specific run
            overrides = {
                'experiment.name': f"{dataset}-{combo}",
                'data.train_file': train_file,
                'data.test_file': test_file
            }
            
            # --- 1. TRAIN ---
            print("\n[1/2] Starting Training...")
            # train_model passes overrides and returns the experiment directory
            exp_dir = train_model(config_template, overrides=overrides)
            
            # --- 2. TEST ---
            if os.path.exists(test_file):
                print(f"\n[2/2] Starting Testing using model in {exp_dir}...")
                test_model(exp_dir, test_file_path=test_file)
            else:
                print(f"⚠️ Error: Test file not found -> {test_file}")

if __name__ == "__main__":
    run_pipeline()