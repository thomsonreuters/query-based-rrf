import pandas as pd
import numpy as np
import pickle
import json
import os
import time
from train import compute_metrics, Config
import glob

WARMUP_ITERS = 5

def test_model(model_dir, test_file, output_base_dir):
    """Test the trained model on a test file and save csv results"""
    
    config_path = f"{model_dir}/config.yaml"

    # Load configuration
    config = Config(config_path)
    
    # Load trained model and vectorizer
    print("Loading trained model and vectorizer...")
    with open(os.path.join(model_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(os.path.join(model_dir, 'ridge_regression_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # Load test dataset
    print(f"Loading test dataset: {test_file}")
    test_df = pd.read_csv(test_file)
    print(f"Test data shape: {test_df.shape}")
    
    # Extract features for ALL entries
    X_test_text = test_df['query_text'].astype(str)

    # Per-query (batch size 1) inference: time vectorize → predict → scalar
    # This matches real serving, where a single query's TF-IDF transform and
    # Ridge.predict are the per-request cost.
    print("Running per-query inference...")
    n = len(X_test_text)
    y_test_pred = np.zeros(n, dtype=np.float64)
    latencies_ms = np.zeros(n, dtype=np.float64)

    # Warmup (not timed)
    if n > 0:
        warm_text = X_test_text.iloc[0]
        for _ in range(WARMUP_ITERS):
            _ = float(model.predict(vectorizer.transform([warm_text]))[0])

    for i, text in enumerate(X_test_text):
        t0 = time.perf_counter()
        X = vectorizer.transform([text])
        pred = float(model.predict(X)[0])
        latencies_ms[i] = (time.perf_counter() - t0) * 1000.0
        y_test_pred[i] = pred

    # Clip predictions to valid range [0, 1] and round to 2 decimal places
    y_test_pred = np.round(np.clip(y_test_pred, 0.00, 1.0), 2)
    
    # Filter for evaluation metrics (only entries with valid mean_best_weight)
    valid_mask = test_df['mean_best_weight'].notna()
    y_test_valid = test_df.loc[valid_mask, 'mean_best_weight']
    y_test_pred_valid = y_test_pred[valid_mask.values]
    
    print(f"Total samples: {len(test_df)}")
    print(f"Valid samples for evaluation: {len(y_test_valid)}")
    
    # Compute metrics with confidence intervals (only on valid samples)
    print("Computing metrics with bootstrap confidence intervals...")
    test_metrics = compute_metrics(y_test_valid, y_test_pred_valid)
    
    # Prepare test results
    test_file_name = test_file
    test_results = {
        'test_file': test_file_name,
        'test_samples': len(X_test_text),
        'valid_samples_for_evaluation': len(y_test_valid),
        'test_metrics': test_metrics,
        'test_target_stats': {
            'min': float(y_test_valid.min()),
            'max': float(y_test_valid.max()),
            'mean': float(y_test_valid.mean()),
            'std': float(y_test_valid.std())
        }
    }
    
    # Load existing results and add test results
    results_file = os.path.join(model_dir, 'results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    
    # Add test results
    if 'test_results' not in results:
        results['test_results'] = []
    
    results['test_results'].append(test_results)
    
    # Save updated results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # --- NEW LOGIC FOR CSV & TREC OUTPUT FILENAMES ---
    path_parts = test_file.split(os.sep)
    dataset_name = path_parts[-5] if len(path_parts) >= 5 else "unknown" 
    split_name = path_parts[-3] if len(path_parts) >= 3 else "unknown"
    
    # Extract combo from filename (e.g., bm25_vs_biencoder -> bm25_biencoder)
    file_basename = os.path.basename(test_file)
    combo_part = "unknown_combo"
    for c in ["bm25_vs_biencoder", "bm25_vs_qwen3", "rm3_vs_biencoder", "rm3_vs_qwen3"]:
        if c in file_basename:
            # combo_part = c.replace("_vs_", "_")
            combo_part = c
            break

    # Construct the requested path formats
    os.makedirs(output_base_dir, exist_ok=True)
    csv_filename = f"{dataset_name}_{combo_part}_{split_name}.csv"
    
    output_csv_path = os.path.join(output_base_dir, csv_filename)
    # ------------------------------------------
    
    # Create test predictions DataFrame
    test_predictions_df = pd.DataFrame({
        'query_id': test_df['query_id'] if 'query_id' in test_df.columns else range(len(test_df)),
        'query_text': X_test_text,
        'actual': test_df['mean_best_weight'],
        'predicted': y_test_pred,
        'latency_ms': latencies_ms,
        'absolute_error': np.round(np.where(valid_mask, np.abs(test_df['mean_best_weight'] - y_test_pred), np.nan), 2)
    })
    
    # 1. Save detailed info as CSV
    test_predictions_df.to_csv(output_csv_path, index=False)



    # Print summary with confidence intervals
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test file: {test_file_name}")
    print(f"Model directory: {model_dir}")
    print(f"Total samples: {len(test_df)}")
    
    print(f"\nTEST METRICS (with 95% CI):")
    for metric in ['mae', 'mse', 'rmse', 'r2']:
        value = test_metrics[metric]
        ci_lower = test_metrics[f'{metric}_ci_lower']
        ci_upper = test_metrics[f'{metric}_ci_upper']
        print(f"  {metric}: {value:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print(f"\nResults saved to: {results_file}")
    print(f"CSV output saved to: {output_csv_path}")



if __name__ == "__main__":
    print("Please run testing via run.py")