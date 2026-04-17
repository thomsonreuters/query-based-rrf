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

from train import Config, RegressionDataset, ModernBertRegression

def load_model(model_path, config):
    if os.path.exists(os.path.join(model_path, 'best_model')):
        model_path = os.path.join(model_path, 'best_model')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = ModernBertRegression(
        model_name=config.get('model.name', 'answerdotai/ModernBERT-base'),
        dropout=config.get('model.dropout', 0.1)
    )
    
    model.load_state_dict(torch.load(
        os.path.join(model_path, 'pytorch_model.bin'), 
        map_location='cuda' if torch.cuda.is_available() else 'cpu'
    ))
    
    return model, tokenizer

def test_model(model, dataset, batch_size=32):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions, labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch['input_ids'], batch['attention_mask'])
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            preds_batch = logits.squeeze().cpu().numpy()
            if preds_batch.ndim == 0:
                preds_batch = [preds_batch.item()]
            
            clamped_rounded = [round(max(0.0, min(1.0, x)), 2) for x in preds_batch]
            predictions.extend(clamped_rounded)
            labels.extend(batch['labels'].cpu().numpy().tolist())
    
    predictions = np.array(predictions)
    labels = np.array(labels) 
    
    valid_mask = ~np.isnan(labels).any(axis=1)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    
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
    
    return metrics, predictions, labels

def save_results(model_path, metrics, predictions, labels, dataset, test_file_path, config):
    results_file = os.path.join(model_path, 'results.json')
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    if 'test_results' not in results:
        results['test_results'] = []
    
    new_test_result = {
        'test_file_name': test_file_path,
        'metrics': metrics
    }
    
    found = False
    for i, test_result in enumerate(results['test_results']):
        if test_result['test_file_name'] == test_file_path:
            results['test_results'][i] = new_test_result
            found = True
            break
    
    if not found:
        results['test_results'].append(new_test_result)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Test metrics added to: {results_file}")
    
    results_df = dataset.data.copy()
    results_df['predicted'] = np.round(predictions, 2)
    results_df['actual_interval'] = [str(lbl) for lbl in labels]
    
    labels_np = np.array(labels)
    valid_mask = ~np.isnan(labels_np).any(axis=1)
    
    interval_errors = np.full(len(labels_np), np.nan)
    if valid_mask.any():
        left_np = labels_np[valid_mask, 0]
        right_np = labels_np[valid_mask, 1]
        preds_np = predictions[valid_mask]
        
        interval_errors[valid_mask] = np.maximum(0, left_np - preds_np) + np.maximum(0, preds_np - right_np)
        
    results_df['interval_error'] = interval_errors
    
    split = 'dev' if '/dev/' in test_file_path.replace('\\', '/') or '_dev_' in test_file_path else 'test'
    
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
    
    print("Loading model...")
    model, tokenizer = load_model(model_path, config)
    
    test_file_path = test_file_path or config.get('data.test_file')
    split = 'dev' if '/dev/' in test_file_path.replace('\\', '/') or '_dev_' in test_file_path else 'test'
    
    # Provide the necessary paths for passage conditioning
    test_dataset = RegressionDataset(
        test_file_path,
        tokenizer,
        config.get('model.max_length', 512),
        split=split,
        corpus_path=config.get('data.corpus_path'),
        sparse_trec=config.get('data.sparse_trec_test'),
        dense_trec=config.get('data.dense_trec_test')
    )
    
    print("Testing...")
    metrics, predictions, labels = test_model(model, test_dataset, config.get('testing.batch_size', 32))
    
    save_results(model_path, metrics, predictions, labels, test_dataset, test_file_path, config)
    
    print("\nResults:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"{metric}: {value:.4f}")
    
    print(f"\nSample predictions:")
    for i in range(min(5, len(test_dataset))):
        text = test_dataset.data.iloc[i]['query_text']
        actual = labels[i]
        predicted = predictions[i]
        print(f"'{text[:50]}...' -> Actual Interval: {actual}, Predicted: {predicted:.2f}")


if __name__ == "__main__":
    model_path = ""
    if model_path:
        run_test(model_path)