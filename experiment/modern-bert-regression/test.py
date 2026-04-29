import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yaml
import argparse
import os
import json
import time


# TO THIS:
from train import Config, RegressionDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Enable TF32 on Ampere+ GPUs for fp32 matmul (free ~10–20% on the fp32 path).
torch.set_float32_matmul_precision('high')

def load_model(model_path, config):
    """Load model and tokenizer"""
    # Check if it's experiment directory
    if os.path.exists(os.path.join(model_path, 'best_model')):
        model_path = os.path.join(model_path, 'best_model')

    # Use Auto classes instead of Roberta specific ones
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load in bf16 on CUDA — memory-bandwidth-bound at batch=1, ~1.5–2× speedup
    # via Tensor Cores on Ampere+. Required when FA2 is enabled.
    use_fa2 = config.get('testing.use_flash_attention_2', False)
    if use_fa2 and not torch.cuda.is_available():
        raise RuntimeError("testing.use_flash_attention_2=true but CUDA is not available")
    load_kwargs = {}
    if torch.cuda.is_available():
        load_kwargs['torch_dtype'] = torch.bfloat16
    if use_fa2:
        load_kwargs['attn_implementation'] = 'flash_attention_2'

    model = AutoModelForSequenceClassification.from_pretrained(model_path, **load_kwargs)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    return model, tokenizer


WARMUP_ITERS = 5


def test_model(model, dataset, tokenizer, config):
    """Per-query (batch size 1) inference. Times tokenize → forward → scalar-on-host."""
    model.eval()
    device = next(model.parameters()).device
    max_length = config.get('model.max_length', 64)

    # FA2's HF wrapper calls Tensor.item() inside forward, which torch.compile
    # cannot trace across — graph-breaks every call and re-traces. Skip compile
    # when FA2 is on; SDPA + compile is the recommended combo anyway.
    if device.type == 'cuda' and not config.get('testing.use_flash_attention_2', False):
        model = torch.compile(model)

    n = len(dataset)
    predictions = np.zeros(n, dtype=np.float64)
    labels = np.zeros(n, dtype=np.float64)
    latencies_ms = np.zeros(n, dtype=np.float64)

    def _run_one(text):
        enc = tokenizer(
            text, truncation=True, padding='max_length',
            max_length=max_length, return_tensors='pt'
        )
        input_ids = enc['input_ids'].to(device, non_blocking=True)
        attention_mask = enc['attention_mask'].to(device, non_blocking=True)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
        return float(logits.squeeze().float().cpu().item())
    start = time.time()
    with torch.inference_mode():
        if n > 0:
            warm_text = dataset.get_input_text(0)
            for _ in range(WARMUP_ITERS):
                _run_one(warm_text)
            if device.type == 'cuda':
                torch.cuda.synchronize()

        for i in range(n):
            text = dataset.get_input_text(i)
            labels[i] = dataset.get_label(i)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = _run_one(text)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies_ms[i] = (time.perf_counter() - t0)
            predictions[i] = round(pred, 2)
    end = time.time()
    print("=======> end-start", end - start)
    valid_mask = ~np.isnan(labels)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]

    if len(valid_labels) > 0:
        metrics = {
            'MAE': mean_absolute_error(valid_labels, valid_predictions),
            'MSE': mean_squared_error(valid_labels, valid_predictions),
            'RMSE': np.sqrt(mean_squared_error(valid_labels, valid_predictions)),
            'R2': r2_score(valid_labels, valid_predictions)
        }
    else:
        metrics = {'MAE': None, 'MSE': None, 'RMSE': None, 'R2': None}

    return metrics, predictions, labels, latencies_ms

# UPDATE: Added config as a parameter to access experiment.name
def save_results(model_path, metrics, predictions, labels, latencies_ms, dataset, test_file_path, config):
    """Save metrics and predictions to files"""
    # Load existing results.json
    results_file = os.path.join(model_path, 'results.json')
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Initialize test_results as list if it doesn't exist
    if 'test_results' not in results:
        results['test_results'] = []
    
    # Create new test result
    new_test_result = {
        'test_file_name': test_file_path,
        'metrics': metrics
    }
    
    # Check if this test file path already exists, if so replace it
    found = False
    for i, test_result in enumerate(results['test_results']):
        if test_result['test_file_name'] == test_file_path:
            results['test_results'][i] = new_test_result
            found = True
            break
    
    # If not found, append new result
    if not found:
        results['test_results'].append(new_test_result)
    
    # Save updated results.json back to the model path
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Test metrics added to: {results_file}")
    
    # Save predictions with all original columns
    results_df = dataset.data.copy()
    results_df['predicted'] = predictions
    results_df['latency_ms'] = latencies_ms

    # Only calculate error for entries with valid labels
    valid_mask = ~np.isnan(labels)
    results_df['error'] = np.nan
    results_df.loc[valid_mask, 'error'] = np.abs(labels[valid_mask] - predictions[valid_mask])
    
    # Rename mean_best_weight to actual for clarity
    if 'mean_best_weight' in results_df.columns:
        results_df = results_df.rename(columns={'mean_best_weight': 'actual'})
    
    # UPDATE: Build the new filename and save location
    experiment_name = config.get('experiment.name', 'experiment')
    
    # Determine split ('dev' or 'test') by looking at the file path string
    split = 'dev' if '/dev/' in test_file_path.replace('\\', '/') or '_dev_' in test_file_path else 'test'
    
    # Create the 'predictions' directory in the current working directory
    predictions_dir = "predictions"
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Format: {experiment.name}_{split}_predictions.csv
    exp_name = experiment_name.replace("_vs", "")
    predictions_filename = f"{exp_name}_{split}.csv"
    predictions_file = os.path.join(predictions_dir, predictions_filename)
    
    results_df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to: {predictions_file}")

def run_test(model_path, test_file_path=None):
    config = Config(f'{model_path}/config.yaml')
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model(model_path, config)
    
    # Use the injected test_file_path if provided, else fallback to config
    test_file_path = test_file_path or config.get('data.test_file')
    
    # Determine the split dynamically for RegressionDataset as well
    split = 'dev' if '/dev/' in test_file_path.replace('\\', '/') or '_dev_' in test_file_path else 'test'
    
    # Load test data
    test_dataset = RegressionDataset(
        test_file_path,
        tokenizer,
        config.get('model.max_length', 64),
        split=split
    )
    print("## ", len(test_dataset), "queries")
    # Test
    print("Testing (batch size 1, per-query timing)...")
    metrics, predictions, labels, latencies_ms = test_model(model, test_dataset, tokenizer, config)

    # UPDATE: Pass the config object down to save_results
    save_results(model_path, metrics, predictions, labels, latencies_ms, test_dataset, test_file_path, config)
    
    # Print results
    print("\nResults:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric}: {value:.4f}")
    
    # Show some examples
    print(f"\nSample predictions:")
    for i in range(min(5, len(test_dataset))):
        text = test_dataset.data.iloc[i]['query_text']
        actual = labels[i]
        predicted = predictions[i]
        print(f"'{text[:50]}...' -> Actual: {actual:.2f}, Predicted: {predicted:.2f}")
    return latencies_ms


if __name__ == "__main__":
    import glob

    _base_experiment_dir = os.environ.get("BASE_EXPERIMENT_DIR", "/extra/huaiyaom0/tr-intern/wrrf/experiment")
    _base_data_dir = os.environ.get("BASE_DATA_DIR", "/extra/huaiyaom0/tr-intern/wrrf/dataset")
    experiments_dir = f"{_base_experiment_dir}/modern-bert-regression/experiments"

    datasets = ["acord-entire-corpus", "msmarco", "nfcorpus", "nq"]
    combos = ["bm25_vs_biencoder", "bm25_vs_qwen3", "rm3_vs_biencoder", "rm3_vs_qwen3"]

    for dataset in datasets:
        split = "test" if dataset in ["acord-entire-corpus", "nfcorpus"] else "dev"
        metric = "ndcg" if dataset in ["acord-entire-corpus", "nfcorpus"] else "mrr"
        for combo in combos:
            matches = sorted(glob.glob(f"{experiments_dir}/{dataset}-{combo}_*"))
            if not matches:
                print(f"Skipping {dataset}/{combo}: no experiment folder found.")
                continue
            model_path = matches[-1]
            test_file_path = f"{_base_data_dir}/{dataset}/{metric}_runs/{split}/top200/results_{split}_{combo}_best_weights_final_mean_with_text.csv"
            latencies_ms = run_test(model_path, test_file_path=test_file_path)
            print(f"=======> Total Latencies for dataset-combo {dataset}-{combo}, split {split}, metric {metric}", sum(latencies_ms), "seconds")
