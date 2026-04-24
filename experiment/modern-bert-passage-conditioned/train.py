import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer, 
    AutoModel, 
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
import ast
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

def _parse_interval_label(weight_str):
    """Parse friendly_best_weights string → (left, right). Picks widest interval."""
    try:
        parsed_weights = ast.literal_eval(weight_str)
        if isinstance(parsed_weights[0], (int, float)):
            val = float(parsed_weights[0])
            return val, val
        best_interval = parsed_weights[0]
        max_diff = -1.0
        for interval in parsed_weights:
            if len(interval) == 1:
                l, r = float(interval[0]), float(interval[0])
            elif len(interval) >= 2:
                l, r = float(interval[0]), float(interval[1])
            else:
                continue
            diff = r - l
            if diff > max_diff:
                max_diff = diff
                best_interval = [l, r]
        return best_interval[0], best_interval[1]
    except Exception:
        return 0.0, 0.0


class RegressionDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512, split='train', corpus_path=None, sparse_trec=None, dense_trec=None):
        self.data = pd.read_csv(csv_file, dtype={'query_id': str})

        if split == 'train':
            # Filter out rows with missing friendly_best_weights
            self.data = self.data.dropna(subset=['friendly_best_weights'])
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load top-1 doc mappings
        self.sparse_top1 = self._get_top1_trec(sparse_trec)
        self.dense_top1 = self._get_top1_trec(dense_trec)
        
        # Memory-efficient loading: only grab corpus texts for documents we actually need
        needed_docs = set(self.sparse_top1.values()).union(set(self.dense_top1.values()))
        self.corpus = self._load_corpus(corpus_path, needed_docs)
        
    def _get_top1_trec(self, trec_file):
        top1_docs = {}
        if not trec_file or not os.path.exists(trec_file):
            return top1_docs
            
        print(f"Loading top-1 docs from {trec_file}...")
        with open(trec_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    qid = str(parts[0])
                    docid = str(parts[2])
                    # TREC is sorted by score/rank, so first occurrence is top-1
                    if qid not in top1_docs:
                        top1_docs[qid] = docid
        return top1_docs

    def _load_corpus(self, corpus_path, needed_docids):
        corpus = {}
        if not corpus_path or not os.path.exists(corpus_path) or not needed_docids:
            return corpus
            
        print(f"Loading corpus subset from {corpus_path}...")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                doc_id = str(doc.get('_id', doc.get('id', '')))
                if doc_id in needed_docids:
                    title = doc.get('title', '')
                    text = doc.get('text', doc.get('contents', ''))
                    corpus[doc_id] = f"{title} {text}".strip()
        print(f"Loaded {len(corpus)} documents into memory.")
        return corpus
    
    def __len__(self):
        return len(self.data)

    def _separator(self):
        sep = getattr(self.tokenizer, 'sep_token', None)
        if not sep:
            sep = getattr(self.tokenizer, 'eos_token', '[SEP]')
        return sep

    def get_input_text(self, idx):
        row = self.data.iloc[idx]
        query_text = str(row['query_text'])
        query_id = str(row['query_id'])
        sparse_docid = self.sparse_top1.get(query_id)
        dense_docid = self.dense_top1.get(query_id)
        sparse_text = self.corpus.get(sparse_docid, "")
        dense_text = self.corpus.get(dense_docid, "")
        sep = self._separator()
        return f"{query_text} {sep} {sparse_text} {sep} {dense_text}"

    def get_label(self, idx):
        weight_str = str(self.data.iloc[idx]['friendly_best_weights'])
        return _parse_interval_label(weight_str)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        query_text = str(row['query_text'])
        query_id = str(row['query_id'])
        weight_str = str(row['friendly_best_weights'])
        
        # Retrieve passage texts
        sparse_docid = self.sparse_top1.get(query_id)
        dense_docid = self.dense_top1.get(query_id)
        
        sparse_text = self.corpus.get(sparse_docid, "")
        dense_text = self.corpus.get(dense_docid, "")
        
        # Determine the correct separator token for the model
        sep = getattr(self.tokenizer, 'sep_token', None)
        if not sep:
            sep = getattr(self.tokenizer, 'eos_token', '[SEP]')
            
        # Format: query [SEP] top-1 sparse passage [SEP] top-1 dense passage
        combined_input = f"{query_text} {sep} {sparse_text} {sep} {dense_text}"
        
        # Parse friendly_best_weights based on provided rules
        try:
            parsed_weights = ast.literal_eval(weight_str)
            
            if isinstance(parsed_weights[0], (int, float)):
                val = float(parsed_weights[0])
                left, right = val, val
            else:
                best_interval = parsed_weights[0]
                max_diff = -1.0
                
                for interval in parsed_weights:
                    if len(interval) == 1:
                        l, r = float(interval[0]), float(interval[0])
                    elif len(interval) >= 2:
                        l, r = float(interval[0]), float(interval[1])
                    else:
                        continue
                        
                    diff = r - l
                    if diff > max_diff:
                        max_diff = diff
                        best_interval = [l, r]
                        
                left, right = best_interval[0], best_interval[1]
        except Exception:
            left, right = 0.0, 0.0
        
        label = [left, right]

        encoding = self.tokenizer(
            combined_input,
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

class ModernBertRegression(nn.Module):
    def __init__(self, model_name='answerdotai/ModernBERT-base', dropout=0.1,
                 attn_implementation=None, torch_dtype=None):
        super().__init__()
        encoder_kwargs = {}
        if attn_implementation is not None:
            encoder_kwargs['attn_implementation'] = attn_implementation
        if torch_dtype is not None:
            encoder_kwargs['torch_dtype'] = torch_dtype
        self.model = AutoModel.from_pretrained(model_name, **encoder_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
            
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)
        
        preds = logits.squeeze(-1)

        loss = None
        if labels is not None:
            left_interval = labels[:, 0]
            right_interval = labels[:, 1]
            
            loss_left = torch.relu(left_interval - preds) ** 2
            loss_right = torch.relu(preds - right_interval) ** 2
            loss = (loss_left + loss_right).mean()
        
        return {'loss': loss, 'logits': logits}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    
    left_interval = labels[:, 0]
    right_interval = labels[:, 1]
    
    error_left = np.maximum(0, left_interval - predictions)
    error_right = np.maximum(0, predictions - right_interval)
    interval_errors = error_left + error_right
    
    mae = np.mean(interval_errors)
    mse = np.mean(interval_errors ** 2)
    rmse = math.sqrt(mse)
    
    midpoints = (left_interval + right_interval) / 2.0
    r2 = r2_score(midpoints, predictions)
    
    pearson_corr, pearson_p = pearsonr(midpoints, predictions)
    spearman_corr, spearman_p = spearmanr(midpoints, predictions)
    
    epsilon = 1e-3
    mape = np.mean(interval_errors / np.maximum(midpoints, epsilon)) * 100
    
    return {
        'interval_mae': mae,
        'interval_mse': mse,
        'interval_rmse': rmse,
        'r2_midpoint': r2,
        'pearson_correlation': pearson_corr,
        'spearman_correlation': spearman_corr,
        'mape': mape,
    }

def save_validation_results(trainer, eval_dataset, exp_dir):
    model = trainer.model
    model.eval()
    
    dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    predictions, labels = [], []
    
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            outputs = model(batch['input_ids'], batch['attention_mask'])
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            preds_batch = logits.squeeze().cpu().numpy()
            if preds_batch.ndim == 0:
                preds_batch = [preds_batch.item()]
            
            clamped_rounded = [round(max(0.0, min(1.0, x)), 2) for x in preds_batch]
            predictions.extend(clamped_rounded)
            labels.extend(batch['labels'].cpu().numpy().tolist())
    
    val_indices = eval_dataset.indices
    original_data = eval_dataset.dataset.data.iloc[val_indices].copy()
    
    original_data['predicted'] = predictions
    
    labels_np = np.array(labels)
    left_np = labels_np[:, 0]
    right_np = labels_np[:, 1]
    preds_np = np.array(predictions)
    
    original_data['actual_interval'] = [str(lbl) for lbl in labels]
    original_data['interval_error'] = np.maximum(0, left_np - preds_np) + np.maximum(0, preds_np - right_np)
    
    val_results_file = os.path.join(exp_dir, 'validation_predictions.csv')
    original_data.to_csv(val_results_file, index=False)
    print(f"Validation predictions saved to: {val_results_file}")

def main(config_path, overrides=None):
    config = Config(config_path, overrides=overrides)

    current_script = inspect.getfile(inspect.currentframe())
    
    tracker = ExperimentTracker()
    exp_dir = tracker.start_experiment(config, script_path=current_script)
    
    torch.manual_seed(config.get('training.seed'))
    
    tokenizer = AutoTokenizer.from_pretrained(config.get('model.name'))
    
    # Pass path configurations to construct textual inputs properly
    full_train_dataset = RegressionDataset(
        config.get('data.train_file'), 
        tokenizer, 
        max_length=config.get('model.max_length'),
        corpus_path=config.get('data.corpus_path'),
        sparse_trec=config.get('data.sparse_trec_train'),
        dense_trec=config.get('data.dense_trec_train')
    )
    
    total_size = len(full_train_dataset)
    train_size = int(config.get('data.train_split') * total_size)
    eval_size = total_size - train_size
    
    train_dataset, eval_dataset = random_split(full_train_dataset, [train_size, eval_size])
    
    model = ModernBertRegression(
        model_name=config.get('model.name'),
        dropout=config.get('model.dropout')
    )
    
    output_dir = os.path.join(exp_dir, "checkpoints")
    best_model_dir = os.path.join(exp_dir, "best_model")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get('training.num_train_epochs'),
        per_device_train_batch_size=config.get('training.per_device_train_batch_size'),
        per_device_eval_batch_size=config.get('training.per_device_eval_batch_size'),
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
        fp16=True, 
        report_to="none" 
    )
    
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
    
    print("Starting training...")
    trainer.train()

    save_validation_results(trainer, eval_dataset, exp_dir)
    
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    torch.save(model.state_dict(), f"{best_model_dir}/pytorch_model.bin")
    
    final_metrics = trainer.evaluate()
    tracker.log_metrics(final_metrics)
    tracker.save_results()
    
    print(f"Training completed! Results saved to {exp_dir}")
    
    return exp_dir

if __name__ == "__main__":
    config_file = 'config.yaml' 
    main(config_file)