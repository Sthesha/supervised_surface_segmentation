#!/bin/bash

# Move to the script's root directory (ensure paths work correctly)
cd "$(dirname "$0")/.." || exit

# Define variables for paths
SOURCE_FILE="./data/selected_tokens_parsable.txt"
PARSE_GRAMMAR="./grammars/ZMLargeExtChunkParse.pgf"
LINEARIZE_GRAMMAR="./grammars/ZMLargeExtChunkLinC.pgf"
OUTPUT_FILE="./results/rule_based_results_three.json"

# Run the rule-based segmentation script
python3 scripts/rule_based_segment.py \
    --source "$SOURCE_FILE" \
    --parse_grammar "$PARSE_GRAMMAR" \
    --linearize_grammar "$LINEARIZE_GRAMMAR" \
    --output "$OUTPUT_FILE"

# Print message after completion
echo "Segmentation completed. Results saved in $OUTPUT_FILE."
