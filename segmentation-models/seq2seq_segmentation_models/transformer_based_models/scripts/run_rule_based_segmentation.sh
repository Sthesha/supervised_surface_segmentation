#!/bin/bash

cd "$(dirname "$0")/.." || exit

SOURCE_FILE="./data/selected_tokens_parsable.txt"
PARSE_GRAMMAR="./grammars/ZMLargeExtChunkParse.pgf"
LINEARIZE_GRAMMAR="./grammars/ZMLargeExtChunkLinC.pgf"
OUTPUT_FILE="./results/rule_based_results_three.json"

python3 scripts/rule_based_segment.py \
    --source "$SOURCE_FILE" \
    --parse_grammar "$PARSE_GRAMMAR" \
    --linearize_grammar "$LINEARIZE_GRAMMAR" \
    --output "$OUTPUT_FILE"

echo "Segmentation completed. Results saved in $OUTPUT_FILE."
