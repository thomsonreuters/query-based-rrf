#!/bin/bash

# Run timing experiments on SageMaker Studio with 10 sample questions
# Make sure your data is in: /home/sagemaker-user/query-aware-rrf/query-based-rrf/data/

echo "=========================================="
echo "Timing Experiments with 10 Sample Questions"
echo "=========================================="

# Change to repo directory
cd /home/sagemaker-user/query-aware-rrf/query-based-rrf

# 1. Ridge Regression
echo -e "\n--- Experiment 1: Ridge Regression ---"
cd experiment/ridge-regression/ridge-regression-mean-best-weight
time python run.py 2>&1 | head -20

# 2. RoBERTa Regression
echo -e "\n--- Experiment 2: RoBERTa Regression ---"
cd ../../roberta-regression/roberta-experiment-mean-best-weight
time python run.py 2>&1 | head -20

# 3. RoBERTa Interval Weight
echo -e "\n--- Experiment 3: RoBERTa Interval Weight ---"
cd ../../roberta-interval-weight
time python run_mul.py 2>&1 | head -20

# 4. ModernBERT Regression
echo -e "\n--- Experiment 4: ModernBERT Regression ---"
cd ../../modern-bert-regression
time python run.py 2>&1 | head -20

# 5. ModernBERT Interval Weight
echo -e "\n--- Experiment 5: ModernBERT Interval Weight ---"
cd ../modern-bert-interval-weight
time python run_mul.py 2>&1 | head -20

# 6. Dynamic Alpha Tuning (requires AWS/Bedrock setup)
echo -e "\n--- Experiment 6: Dynamic Alpha Tuning ---"
cd ../dynamic-alpha-tuning
# Uncomment if AWS/Bedrock is configured:
# time python dynamic_alpha_tuning.py --dataset msmarco --num_queries 10

echo -e "\n=========================================="
echo "All experiments completed!"
echo "==========================================">