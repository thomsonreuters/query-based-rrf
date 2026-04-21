#!/bin/bash
# Script to reproduce the timing-samples dataset on another machine
#
# Usage:
#   export QBR_REPO_PATH=/path/to/query-based-rrf  # Optional: set repo path
#   ./reproduce_timing_samples.sh
#
# Or with explicit data directory:
#   ./reproduce_timing_samples.sh --data-dir /custom/path/to/data/input

# Set default repo path if not provided
if [ -z "$QBR_REPO_PATH" ]; then
    export QBR_REPO_PATH="/Users/a6128162/Repos/query-based-rrf"
    echo "QBR_REPO_PATH not set. Using default: $QBR_REPO_PATH"
else
    echo "Using QBR_REPO_PATH: $QBR_REPO_PATH"
fi

# Default data directory
DATA_DIR="$QBR_REPO_PATH/data/input"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--data-dir /path/to/data/input]"
            exit 1
            ;;
    esac
done

echo "=== Reproducing timing-samples dataset ==="
echo "Data directory: $DATA_DIR"

# Step 1: Create sampled TREC and CSV files
echo -e "\nStep 1: Running prepare_sample_data.py to create TREC and CSV files..."
python prepare_sample_data.py --data-dir "$DATA_DIR" --num-samples 100 --seed 42

# Check if step 1 succeeded
if [ $? -ne 0 ]; then
    echo "Error: prepare_sample_data.py failed"
    exit 1
fi

# Step 2: Create corpus.jsonl and queries.jsonl from the sampled data
echo -e "\nStep 2: Running create_timing_jsonl_files_verified.py to create JSONL files..."
python create_timing_jsonl_files_verified.py --data-dir "$DATA_DIR"

# Check if step 2 succeeded
if [ $? -ne 0 ]; then
    echo "Error: create_timing_jsonl_files_verified.py failed"
    exit 1
fi

echo -e "\n=== Done! ==="
echo "The timing-samples dataset has been created at: $DATA_DIR/timing-samples/"
echo "Contents:"
ls -la "$DATA_DIR/timing-samples/"