#!/bin/bash

# Move to the script's root directory (ensure paths work correctly)
cd "$(dirname "$0")/.." || exit

# Define file paths
SOURCE_FILE="./data/selected_tokens_parsable.txt"
TRANSFORMER_MODEL="./models_final_200/segmenter_three"
OUTPUT_FILE="./results/transformer_three_results.json"

# Run the Transformer segmentation script
python3 scripts/transformer_segment.py \
    --source "$SOURCE_FILE" \
    --transformer_model "$TRANSFORMER_MODEL" \
    --output "$OUTPUT_FILE"

# Print message after completion
echo "Segmentation completed. Results saved in $OUTPUT_FILE."
