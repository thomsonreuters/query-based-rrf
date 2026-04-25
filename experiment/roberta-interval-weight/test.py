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

# Import reusable classes and functions from train.py
from train import Config, RegressionDataset, RobertaRegression

# Enable TF32 on Ampere+ GPUs for fp32 matmul (free ~10–20% on the fp32 path).
torch.set_float32_matmul_precision('high')

def load_model(model_path, config):
    """Load model and tokenizer"""
    # Check if it's experiment directory
    if os.path.exists(os.path.join(model_path, 'best_model')):
        model_path = os.path.join(model_path, 'best_model')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = RobertaRegression(
        model_name=config.get('model.name', 'roberta-base'),
        dropout=config.get('model.dropout', 0.1)
    )

    model.load_state_dict(torch.load(
        os.path.join(model_path, 'pytorch_model.bin'),
        map_location='cuda' if torch.cuda.is_available() else 'cpu'
    ))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    # Cast to bf16 on CUDA — memory-bandwidth-bound at batch=1, ~1.5–2× speedup
    # via Tensor Cores on Ampere+ (A10G/L4/L40S/A100/H100; not T4).
    if device == 'cuda':
        model = model.to(torch.bfloat16)

    return model, tokenizer

WARMUP_ITERS = 5


def test_model(model, dataset, tokenizer, config):
    """Per-query (batch size 1) inference. Times tokenize → forward → scalar-on-host."""
    model.eval()
    device = next(model.parameters()).device
    max_length = config.get('model.max_length', 64)

    if device.type == 'cuda':
        model = torch.compile(model)

    n = len(dataset)
    predictions = np.zeros(n, dtype=np.float64)
    labels = np.zeros((n, 2), dtype=np.float64)
    latencies_ms = np.zeros(n, dtype=np.float64)

    def _run_one(text):
        enc = tokenizer(
            text, truncation=True, padding='max_length',
            max_length=max_length, return_tensors='pt'
        )
        input_ids = enc['input_ids'].to(device, non_blocking=True)
        attention_mask = enc['attention_mask'].to(device, non_blocking=True)
        outputs = model(input_ids, attention_mask)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        return float(logits.squeeze().float().cpu().item())

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
            raw_pred = _run_one(text)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies_ms[i] = (time.perf_counter() - t0) * 1000.0
            predictions[i] = round(max(0.0, min(1.0, raw_pred)), 2)
    
    # Filter out entries where labels are NaN or missing for metric calculation
    # Since labels is 2D, we check if any value in the row is NaN
    valid_mask = ~np.isnan(labels).any(axis=1)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    
    # Only calculate metrics if we have valid labels
    if len(valid_labels) > 0:
        left_interval = valid_labels[:, 0]
        right_interval = valid_labels[:, 1]
        
        error_left = np.maximum(0, left_interval - valid_predictions)
        error_right = np.maximum(0, valid_predictions - right_interval)
        interval_errors = error_left + error_right
        
        mse = np.mean(interval_errors ** 2)
        midpoints = (left_interval + right_interval) / 2.0
        
        metrics = {
            'Interval_MAE': np.mean(interval_errors),
            'Interval_MSE': mse,
            'Interval_RMSE': np.sqrt(mse),
            'R2_Midpoint': r2_score(midpoints, valid_predictions)
        }
    else:
        metrics = {
            'Interval_MAE': None,
            'Interval_MSE': None,
            'Interval_RMSE': None,
            'R2_Midpoint': None
        }

    return metrics, predictions, labels, latencies_ms

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
    results_df['actual_interval'] = [str(lbl) for lbl in labels]
    
    # Calculate interval error for valid labels
    labels_np = np.array(labels)
    valid_mask = ~np.isnan(labels_np).any(axis=1)
    
    interval_errors = np.full(len(labels_np), np.nan)
    if valid_mask.any():
        left_np = labels_np[valid_mask, 0]
        right_np = labels_np[valid_mask, 1]
        preds_np = predictions[valid_mask]
        
        interval_errors[valid_mask] = np.maximum(0, left_np - preds_np) + np.maximum(0, preds_np - right_np)
        
    results_df['interval_error'] = interval_errors
    
    # Determine split ('dev' or 'test') by looking at the file path string
    split = 'dev' if '/dev/' in test_file_path.replace('\\', '/') or '_dev_' in test_file_path else 'test'
    
    # Create the 'predictions' directory in the current working directory
    predictions_dir = "predictions"
    os.makedirs(predictions_dir, exist_ok=True)
    
    experiment_name = config.get('experiment.name', 'experiment')
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
    
    # Test
    print("Testing (batch size 1, per-query timing)...")
    metrics, predictions, labels, latencies_ms = test_model(model, test_dataset, tokenizer, config)

    # Pass the config object down to save_results
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
        actual = labels[i]  # This is now a list [left, right]
        predicted = predictions[i]
        print(f"'{text[:50]}...' -> Actual Interval: {actual}, Predicted: {predicted:.2f}")


if __name__ == "__main__":
    model_path = ""
    if model_path:
        run_test(model_path)