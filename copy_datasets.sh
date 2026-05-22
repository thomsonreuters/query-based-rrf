#!/bin/bash

# Define datasets and combinations
DATASETS=("nq" "msmarco" "nfcorpus" "acord-entire-corpus")
COMBINATIONS=(
    "bm25_vs_biencoder"
    "bm25_vs_qwen3"
    "rm3_vs_biencoder"
    "rm3_vs_qwen3"
)

# Define source and destination paths
SOURCE_BASE="/Users/a6128162/Downloads/extra/huaiyaom0/tr-intern/wrrf/experiment/modern-bert-interval-weight/modern-bert-interval-weight-experiments"
DEST_BASE="s3://a204383-ml-workspace-practicallawqw7t-use1/wrrf/models/modern-bert-interval-weight-experiments"

# Loop through each dataset and combination
for dataset in "${DATASETS[@]}"; do
    for combination in "${COMBINATIONS[@]}"; do
        # Refresh token at the beginning of each iteration
        echo "Refreshing token..."
        source ~/.zprofile
        # Use the full command instead of the alias
        cloud-tool --region us-east-1 --profile tr-labs-prod login --account-id 451191978663 --role human-role/a204383-DataScientist --username MGMT\\M6128162 --password=$VAULTPW

        # Find the folder that matches the pattern {dataset}-{combination}_*
        FOLDER_NAME=$(ls -d "${SOURCE_BASE}/${dataset}-${combination}_"* 2>/dev/null | head -1 | xargs basename)

        if [ -z "$FOLDER_NAME" ]; then
            echo "Warning: No folder found matching ${dataset}-${combination}_*"
            continue
        fi

        # Construct source and destination paths
        SOURCE_PATH="${SOURCE_BASE}/${FOLDER_NAME}/"
        DEST_PATH="${DEST_BASE}/${FOLDER_NAME}/"

        # Print what we're copying
        echo "Copying ${FOLDER_NAME}..."
        echo "From: ${SOURCE_PATH}"
        echo "To: ${DEST_PATH}"

        # Perform the copy - this copies the contents into the named folder
        aws s3 cp "${SOURCE_PATH}" "${DEST_PATH}" --recursive

        # Check if copy was successful
        if [ $? -eq 0 ]; then
            echo "Successfully copied ${FOLDER_NAME}"
        else
            echo "Failed to copy ${FOLDER_NAME}"
        fi

        echo "----------------------------------------"
    done
done

echo "All copy operations completed!"