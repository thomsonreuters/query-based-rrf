import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    RobertaTokenizer, 
    RobertaModel, 
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback
)
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import math
import yaml
import os
import json
import shutil
import inspect
from datetime import datetime

class Config:
    def __init__(self, config_path, overrides=None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Apply any dynamic overrides passed from the runner script
        if overrides:
            for key, value in overrides.items():
                keys = key.split('.')
                current = self.config
                for k in keys[:-1]:
                    current = current.setdefault(k, {})
                current[keys[-1]] = value
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value

class ExperimentTracker:
    def __init__(self, base_dir="experiments"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def start_experiment(self, config, script_path=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = config.get('experiment.name', 'experiment')
        self.exp_dir = os.path.join(self.base_dir, f"{exp_name}_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.exp_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config.config, f, default_flow_style=False)

        # Save script copy
        if script_path:
            script_name = os.path.basename(script_path)
            shutil.copy2(script_path, os.path.join(self.exp_dir, f"script_{script_name}"))
            print(f"Script saved: {script_name}")
        
        # Initialize results log
        self.results = {
            'experiment_name': exp_name,
            'timestamp': timestamp,
            'config': config.config,
            'script_path': script_path,
            'metrics': {}
        }
        
        print(f"Experiment started: {self.exp_dir}")
        return self.exp_dir
    
    def log_metrics(self, metrics, step=None):
        if step is not None:
            if 'training_steps' not in self.results:
                self.results['training_steps'] = {}
            self.results['training_steps'][step] = metrics
        else:
            self.results['final_metrics'] = metrics
    
    def save_results(self):
        with open(os.path.join(self.exp_dir, 'results.json'), 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

class RegressionDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=64, split='train'):
        self.data = pd.read_csv(csv_file)

        if split == 'train':
            # Filter out rows with missing mean_best_weight
            self.data = self.data.dropna(subset=['mean_best_weight'])
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)

    def get_input_text(self, idx):
        return str(self.data.iloc[idx]['query_text'])

    def get_label(self, idx):
        try:
            return float(self.data.iloc[idx]['mean_best_weight'])
        except (ValueError, TypeError):
            return float('nan')

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['query_text'])
        label = float(self.data.iloc[idx]['mean_best_weight'])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

class RobertaRegression(nn.Module):
    def __init__(self, model_name='roberta-base', dropout=0.1):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.roberta.config.hidden_size, 1) 
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits.squeeze(), labels)
        
        return {'loss': loss, 'logits': logits}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    
    mae = mean_absolute_error(labels, predictions)
    mse = mean_squared_error(labels, predictions)
    rmse = math.sqrt(mse)
    r2 = r2_score(labels, predictions)
    
    pearson_corr, pearson_p = pearsonr(labels, predictions)
    spearman_corr, spearman_p = spearmanr(labels, predictions)
    
    # Fix MAPE calculation - avoid division by tiny values
    epsilon = 1e-3  # Reasonable epsilon for [0,1] range
    mape = np.mean(np.abs((labels - predictions) / np.maximum(labels, epsilon))) * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'pearson_correlation': pearson_corr,
        'spearman_correlation': spearman_corr,
        'mape': mape,
    }


def save_validation_results(trainer, eval_dataset, exp_dir):
    """Save validation predictions to CSV file"""
    model = trainer.model
    model.eval()
    
    dataloader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
    predictions, labels = [], []
    
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in dataloader:
            # Move tensors to the same device as the model
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            outputs = model(batch['input_ids'], batch['attention_mask'])
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            predictions.extend([round(x, 2) for x in logits.squeeze().cpu().numpy().tolist()])
            labels.extend(batch['labels'].cpu().numpy().tolist())
    
    # Get original data indices for validation set
    val_indices = eval_dataset.indices
    original_data = eval_dataset.dataset.data.iloc[val_indices].copy()
    
    # Add predictions and errors
    original_data['predicted'] = predictions
    original_data['error'] = np.abs(np.array(labels) - np.array(predictions))
    
    # Rename for clarity
    if 'mean_best_weight' in original_data.columns:
        original_data = original_data.rename(columns={'mean_best_weight': 'actual'})
    
    # Save to CSV
    val_results_file = os.path.join(exp_dir, 'validation_predictions.csv')
    original_data.to_csv(val_results_file, index=False)
    print(f"Validation predictions saved to: {val_results_file}")

# UPDATE: Add overrides=None to the parameters
def main(config_path, overrides=None):
    # UPDATE: Pass overrides to the Config initialization
    config = Config(config_path, overrides=overrides)

    # Get the current script path
    current_script = inspect.getfile(inspect.currentframe())
    
    # Initialize experiment tracker
    tracker = ExperimentTracker()
    exp_dir = tracker.start_experiment(config, script_path=current_script)
    
    # Set seed
    torch.manual_seed(config.get('training.seed'))
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config.get('model.name'))
    
    # Load dataset
    full_train_dataset = RegressionDataset(
        config.get('data.train_file'), 
        tokenizer, 
        max_length=config.get('model.max_length')
    )
    
    # Split dataset
    total_size = len(full_train_dataset)
    train_size = int(config.get('data.train_split') * total_size)
    eval_size = total_size - train_size
    
    train_dataset, eval_dataset = random_split(full_train_dataset, [train_size, eval_size])
    
    # Initialize model
    model = RobertaRegression(
        model_name=config.get('model.name'),
        dropout=config.get('model.dropout')
    )
    
    # Update output directories to use experiment directory
    output_dir = os.path.join(exp_dir, "checkpoints")
    best_model_dir = os.path.join(exp_dir, "best_model")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get('training.num_train_epochs'),
        per_device_train_batch_size=config.get('training.per_device_train_batch_size'),
        per_device_eval_batch_size=config.get('training.per_device_eval_batch_size'),
        # gradient_accumulation_steps=config.get('training.gradient_accumulation_steps'), 
        learning_rate=float(config.get('training.learning_rate')),
        warmup_steps=config.get('training.warmup_steps'),
        weight_decay=config.get('training.weight_decay'),
        logging_dir=os.path.join(exp_dir, 'logs'),
        logging_steps=config.get('training.logging_steps'),
        eval_strategy="steps",
        eval_steps=config.get('training.eval_steps'),
        save_strategy="steps",
        save_steps=config.get('training.save_steps'),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=config.get('training.save_total_limit'),
        greater_is_better=False,
        fp16=True,  # ADD THIS LINE
        report_to="none"  # <--- ADD THIS LINE HERE
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=config.get('training.early_stopping_patience')
        )]
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()

    # Save validation results  # ADD THIS LINE
    save_validation_results(trainer, eval_dataset, exp_dir)  # ADD THIS LINE
    
    # Save the best model
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    torch.save(model.state_dict(), f"{best_model_dir}/pytorch_model.bin")
    
    # Log final metrics
    final_metrics = trainer.evaluate()
    tracker.log_metrics(final_metrics)
    tracker.save_results()
    
    print(f"Training completed! Results saved to {exp_dir}")
    
    # UPDATE: Return the experiment directory
    return exp_dir

if __name__ == "__main__":
    config_file = 'config.yaml' 
    main(config_file)

# CUDA_VISIBLE_DEVICES=0 python