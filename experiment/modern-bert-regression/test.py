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


# TO THIS:
from train import Config, RegressionDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_path, config):
    """Load model and tokenizer"""
    # Check if it's experiment directory
    if os.path.exists(os.path.join(model_path, 'best_model')):
        model_path = os.path.join(model_path, 'best_model')

    # Use Auto classes instead of Roberta specific ones
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load the model directly from the saved directory
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Move model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    return model, tokenizer


def test_model(model, dataset, batch_size=128):
    """Test model and return metrics and predictions"""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    predictions, labels = [], []
    
    # Get the device the model is currently on
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in dataloader:
            # FIX: Move batch data to the GPU!
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # FIX: Use keyword arguments
            outputs = model(
                input_ids=batch['input_ids'], 
                attention_mask=batch['attention_mask']
            )
            
            # FIX: Access logits directly
            logits = outputs.logits 
            
            predictions.extend([round(x, 2) for x in logits.squeeze().cpu().numpy().tolist()])
            labels.extend(batch['labels'].cpu().numpy().tolist())
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Filter out entries where labels are NaN or missing for metric calculation
    valid_mask = ~np.isnan(labels)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    
    # Only calculate metrics if we have valid labels
    if len(valid_labels) > 0:
        metrics = {
            'MAE': mean_absolute_error(valid_labels, valid_predictions),
            'MSE': mean_squared_error(valid_labels, valid_predictions),
            'RMSE': np.sqrt(mean_squared_error(valid_labels, valid_predictions)),
            'R2': r2_score(valid_labels, valid_predictions)
        }
    else:
        metrics = {
            'MAE': None,
            'MSE': None,
            'RMSE': None,
            'R2': None
        }
    
    return metrics, predictions, labels

# UPDATE: Added config as a parameter to access experiment.name
def save_results(model_path, metrics, predictions, labels, dataset, test_file_path, config):
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
    
    # Test
    print("Testing...")
    metrics, predictions, labels = test_model(model, test_dataset, config.get('testing.batch_size', 128))
    
    # UPDATE: Pass the config object down to save_results
    save_results(model_path, metrics, predictions, labels, test_dataset, test_file_path, config)
    
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


if __name__ == "__main__":
    model_path = "/extra/huaiyaom0/tr-intern/wrrf/experiment/roberta/roberta-experiment-1-mean-best-weight-1/experiments/msmarco-bm25_vs_biencoder_20260226_193002"
    if model_path:
        run_test(model_path)