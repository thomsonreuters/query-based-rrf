import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import yaml
import os
import json
import shutil
import pickle
import inspect
from datetime import datetime
import math
import psutil

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

def bootstrap_confidence_interval(y_true, y_pred, metric_func, n_bootstrap=1000, confidence=0.95):
    """Calculate confidence interval using bootstrap sampling"""
    np.random.seed(42)  # For reproducibility
    n_samples = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Calculate metric
        score = metric_func(y_true_boot, y_pred_boot)
        bootstrap_scores.append(score)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_scores, lower_percentile)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile)
    
    return ci_lower, ci_upper

def compute_metrics(y_true, y_pred):
    """Compute comprehensive metrics for regression with confidence intervals"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    pearson_corr, pearson_p = pearsonr(y_true, y_pred)
    spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    
    # # Fix MAPE calculation - avoid division by tiny values
    # epsilon = 1e-3  # Reasonable epsilon for [0,1] range
    # mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, epsilon))) * 100
    # Use relative epsilon or handle zero values explicitly
    epsilon = np.finfo(float).eps
    non_zero_mask = np.abs(y_true) > epsilon
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask.values]) / y_true[non_zero_mask])) * 100
    else:
        mape = np.inf  # or handle appropriately
    
    # Calculate 95% confidence intervals using bootstrap
    mae_ci = bootstrap_confidence_interval(y_true, y_pred, mean_absolute_error)
    mse_ci = bootstrap_confidence_interval(y_true, y_pred, mean_squared_error)
    rmse_ci = bootstrap_confidence_interval(y_true, y_pred, lambda yt, yp: math.sqrt(mean_squared_error(yt, yp)))
    r2_ci = bootstrap_confidence_interval(y_true, y_pred, r2_score)
    
    return {
        'mae': mae,
        'mae_ci_lower': mae_ci[0],
        'mae_ci_upper': mae_ci[1],
        'mse': mse,
        'mse_ci_lower': mse_ci[0],
        'mse_ci_upper': mse_ci[1],
        'rmse': rmse,
        'rmse_ci_lower': rmse_ci[0],
        'rmse_ci_upper': rmse_ci[1],
        'r2': r2,
        'r2_ci_lower': r2_ci[0],
        'r2_ci_upper': r2_ci[1],
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'mape': mape,
    }

def plot_top_coefficients(model, feature_names, top_n=20, save_path='top_coefficients.png'):
    """
    Plot top N coefficients by absolute value in descending order
    """
    # Create DataFrame with features and coefficients
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_
    })
    
    # Calculate absolute values and sort
    feature_importance['abs_coefficient'] = np.abs(feature_importance['coefficient'])
    top_features = feature_importance.nlargest(top_n, 'abs_coefficient')
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar plot
    colors = ['red' if coef < 0 else 'blue' for coef in top_features['coefficient']]
    bars = plt.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
    
    # Customize the plot
    plt.yticks(range(len(top_features)), top_features['feature'], fontsize=10)
    plt.xlabel('Coefficient Value', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Features by Absolute Coefficient Value\n(Blue: Positive, Red: Negative)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for better readability
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add coefficient values on bars
    for i, (bar, coef) in enumerate(zip(bars, top_features['coefficient'])):
        plt.text(coef + (0.01 if coef > 0 else -0.01), i, f'{coef:.3f}', 
                va='center', ha='left' if coef > 0 else 'right', fontsize=9)
    
    # Invert y-axis to show highest coefficients at top
    plt.gca().invert_yaxis()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # Close the figure to free memory
    
    print(f"Plot saved as: {save_path}")
    
    return top_features

def save_feature_analysis(model, feature_names, exp_dir, config):
    """Save comprehensive feature analysis"""
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_,
        'abs_coefficient': np.abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)
    
    # Save all features with coefficients
    feature_importance.to_csv(
        os.path.join(exp_dir, 'all_features_coefficients.csv'), 
        index=False
    )
    
    # Generate plots if configured
    if config.get('analysis.save_coefficient_plots', True):
        # Plot top features
        plot_top_coefficients(
            model, feature_names, top_n=30, 
            save_path=os.path.join(exp_dir, 'top_30_coefficients.png')
        )
        
        # Plot coefficient distribution
        plt.figure(figsize=(10, 6))
        plt.hist(model.coef_, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Coefficient Value', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Distribution of All Coefficients', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'coefficient_distribution.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("Feature analysis plots saved")
    
    return feature_importance

def main(config_path, overrides=None):
    # Pass overrides to Config
    config = Config(config_path, overrides)
    
    # Get the current script path
    current_script = inspect.getfile(inspect.currentframe())
    
    # Initialize experiment tracker
    tracker = ExperimentTracker()
    exp_dir = tracker.start_experiment(config, script_path=current_script)
    
    # Set random seed
    np.random.seed(config.get('training.seed', 42))
    
    # Load training dataset
    print("Loading training dataset...")
    train_df = pd.read_csv(config.get('data.train_file'))
    
    print(f"Training data shape: {train_df.shape}")
    
    # Extract features and target variables
    X_train_text = train_df['query_text'].astype(str)
    y_train = train_df['mean_best_weight']
    
    # Split training data into train/validation if specified
    train_split = config.get('data.train_split', 0.9)
    if train_split < 1.0:
        X_train_text, X_val_text, y_train, y_val = train_test_split(
            X_train_text, y_train, 
            train_size=train_split,
            random_state=config.get('training.random_state', 42)
        )
        print(f"Split training data: {len(X_train_text)} train, {len(X_val_text)} validation")
    
    # Log data statistics
    data_stats = {
        'train_samples': len(X_train_text),
        'train_target_stats': {
            'min': float(y_train.min()),
            'max': float(y_train.max()),
            'mean': float(y_train.mean()),
            'std': float(y_train.std())
        }
    }
    
    # Initialize TfidfVectorizer with config parameters
    print("Initializing TF-IDF vectorizer...")
    vectorizer_params = {
        'max_features': config.get('model.max_features'),
        'stop_words': config.get('model.stop_words'),
        'min_df': config.get('model.min_df'),
        'max_df': config.get('model.max_df'),
        'ngram_range': tuple(config.get('model.ngram_range')),
        'sublinear_tf': config.get('model.sublinear_tf')
    }
    
    # Remove None values
    vectorizer_params = {k: v for k, v in vectorizer_params.items() if v is not None}
    
    vectorizer = TfidfVectorizer(**vectorizer_params)
    
    # Fit and transform training data
    print("Fitting TF-IDF vectorizer and transforming data...")
    print(f"Memory usage before TF-IDF: {psutil.virtual_memory().percent}%")
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    print(f"TF-IDF matrix memory usage: {X_train_tfidf.data.nbytes / 1024**2:.2f} MB")
    
    if train_split < 1.0:
        X_val_tfidf = vectorizer.transform(X_val_text)
    
    print(f"TF-IDF Training Matrix Shape: {X_train_tfidf.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Save vectorizer
    with open(os.path.join(exp_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Initialize Ridge Regression model
    print("Training Ridge Regression model...")
    regression_params = {
        'alpha': config.get('regression.alpha', 1.0),
        'fit_intercept': config.get('regression.fit_intercept', True),
        'positive': config.get('regression.positive', False),
        'solver': config.get('regression.solver', 'auto')
    }

    model = Ridge(**regression_params)
    model.fit(X_train_tfidf, y_train)
    
    # Save model
    with open(os.path.join(exp_dir, 'ridge_regression_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    # Make predictions
    print("Making predictions...")
    y_train_pred = model.predict(X_train_tfidf)
    
    # Clip predictions to valid range [0, 1] (assuming weights are in this range)
    y_train_pred = np.clip(y_train_pred, 0.01, 1.0)
    
    if train_split < 1.0:
        y_val_pred = model.predict(X_val_tfidf)
        y_val_pred = np.clip(y_val_pred, 0.01, 1.0)
    
    # Compute metrics with confidence intervals
    print("Computing metrics with bootstrap confidence intervals...")
    train_metrics = compute_metrics(y_train, y_train_pred)
    
    if train_split < 1.0:
        val_metrics = compute_metrics(y_val, y_val_pred)
    
    # Log metrics
    final_metrics = {
        'train_metrics': train_metrics,
        'data_stats': data_stats,
        'model_params': {
            'vectorizer_params': vectorizer_params,
            'regression_params': regression_params,
            'vocabulary_size': len(vectorizer.vocabulary_),
            'feature_count': X_train_tfidf.shape[1]
        }
    }
    
    if train_split < 1.0:
        final_metrics['val_metrics'] = val_metrics
    
    tracker.log_metrics(final_metrics)
    
    # Save predictions
    print("Saving predictions...")
    
    # Training predictions
    train_predictions_df = pd.DataFrame({
        'query_text': X_train_text,
        'actual': y_train,
        'predicted': y_train_pred,
        'absolute_error': np.abs(y_train - y_train_pred)
    })
    train_predictions_df.to_csv(os.path.join(exp_dir, 'train_predictions.csv'), index=False)
    
    # Validation predictions (if applicable)
    if train_split < 1.0:
        val_predictions_df = pd.DataFrame({
            'query_text': X_val_text,
            'actual_best_weight': y_val,
            'predicted_best_weight': y_val_pred,
            'absolute_error': np.abs(y_val - y_val_pred)
        })
        val_predictions_df.to_csv(os.path.join(exp_dir, 'val_predictions.csv'), index=False)
    
    # Feature analysis
    print("Performing feature analysis...")
    feature_names = vectorizer.get_feature_names_out()
    feature_importance = save_feature_analysis(model, feature_names, exp_dir, config)
    
    # Save results
    tracker.save_results()
    
    # Print summary with confidence intervals
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Experiment directory: {exp_dir}")
    print(f"Training samples: {len(X_train_text)}")
    print(f"Features used: {X_train_tfidf.shape[1]}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    print(f"\nTRAIN METRICS (with 95% CI):")
    for metric in ['mae', 'mse', 'rmse', 'r2']:
        value = train_metrics[metric]
        ci_lower = train_metrics[f'{metric}_ci_lower']
        ci_upper = train_metrics[f'{metric}_ci_upper']
        print(f"  {metric}: {value:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    for metric in ['pearson_correlation', 'spearman_correlation', 'mape']:
        value = train_metrics[metric]
        print(f"  {metric}: {value:.4f}")
    
    if train_split < 1.0:
        print(f"\nVALIDATION METRICS (with 95% CI):")
        for metric in ['mae', 'mse', 'rmse', 'r2']:
            value = val_metrics[metric]
            ci_lower = val_metrics[f'{metric}_ci_lower']
            ci_upper = val_metrics[f'{metric}_ci_upper']
            print(f"  {metric}: {value:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        for metric in ['pearson_correlation', 'spearman_correlation', 'mape']:
            value = val_metrics[metric]
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nModel and vectorizer saved to: {exp_dir}")

    # ADD THIS LINE AT THE END OF main()
    return exp_dir

if __name__ == "__main__":
    config_file = 'config.yaml'
    main(config_file)