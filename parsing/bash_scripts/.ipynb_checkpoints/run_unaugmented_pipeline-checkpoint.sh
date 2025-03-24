#!/bin/bash
set -e

#####################################
# Script: run_unaugmented_pipeline.sh
# Purpose: Run unaugmented tree processing
#####################################

# ========== Configuration ==========
BASE_DIR="/root/cnlp/parsing/parsing"

DATA_FOLDER="$BASE_DIR/data_folder"
GRAMMAR_FOLDER="$BASE_DIR/grammars"
PYTHON_SCRIPTS_FOLDER="$BASE_DIR/python_scripts"
GENERATED_DATA_FOLDER="$BASE_DIR/parses/various_zulu_corpora_folder"

docker exec parallel_parsing mkdir -p "$GENERATED_DATA_FOLDER"

INPUT_SENTENCES_AND_TREES="$DATA_FOLDER/combined_unique.csv"
OUTPUT_FOUND_SUB_TREES="$DATA_FOLDER/found_sub_trees.txt"
EXTRACTION_GRAMMAR="$GRAMMAR_FOLDER/ZMLargeExtChunkParse.pgf"
ERROR_LOG="$DATA_FOLDER/errors.log"

EXTRACTION_CORES=50
EXTRACTION_CHUNK_SIZE=10
EXTRACTION_TIMEOUT=30
EXTRACTION_BOUNDARY="-"
EXTRACTION_TREE_COLUMN="tree_1"

LINEARISE_INPUT="$OUTPUT_FOUND_SUB_TREES"
LINEARISE_OUTPUT="$GENERATED_DATA_FOLDER/unaugmented_and_linearised.csv"
VALID_LINEARISE_OUTPUT="$GENERATED_DATA_FOLDER/valid_unaugmented_and_linearised.csv"
INVALID_LINEARISE_OUTPUT="$GENERATED_DATA_FOLDER/invalid_unaugmented_and_linearised.csv"

LINEARISE_CORES=50
LINEARISE_TIMEOUT=60
NUM_GRAMMARS=4
TREES_PER_CHUNK=2
LINEARISE_CHUNK_SIZE=$((NUM_GRAMMARS * TREES_PER_CHUNK))
LIMIT=100
DEBUG=1

GRAMMAR_A="$GRAMMAR_FOLDER/ZMLargeExtChunkParse.pgf"
GRAMMAR_B="$GRAMMAR_FOLDER/ZMLargeExtChunkA.pgf"
GRAMMAR_C="$GRAMMAR_FOLDER/ZMLargeExtChunkB.pgf"
GRAMMAR_D="$GRAMMAR_FOLDER/ZMLargeExtChunkC.pgf"

####################################
# Step 1: Sub-tree Extraction
####################################
echo "Extracting sub-trees..."
docker exec parallel_parsing python3 "$PYTHON_SCRIPTS_FOLDER/get_sub_trees.py" \
  --input "$INPUT_SENTENCES_AND_TREES" \
  --output "$OUTPUT_FOUND_SUB_TREES" \
  --grammar "$EXTRACTION_GRAMMAR" \
  --log "$ERROR_LOG" \
  --cores "$EXTRACTION_CORES" \
  --chunk-size "$EXTRACTION_CHUNK_SIZE" \
  --timeout "$EXTRACTION_TIMEOUT" \
  --boundary "$EXTRACTION_BOUNDARY" \
  --tree-column "$EXTRACTION_TREE_COLUMN"

echo "Sub-tree extraction complete."

####################################
# Step 2: Tree Linearisation (Unaugmented)
####################################
echo "Starting linearisation for unaugmented trees..."
echo "Using chunk size of $LINEARISE_CHUNK_SIZE ($TREES_PER_CHUNK trees × $NUM_GRAMMARS grammars)"

docker exec parallel_parsing python3 "$PYTHON_SCRIPTS_FOLDER/linearise_with_grammars.py" \
    "$LINEARISE_INPUT" \
    "$LINEARISE_OUTPUT" \
    --cores "$LINEARISE_CORES" \
    --timeout "$LINEARISE_TIMEOUT" \
    --chunk-size "$LINEARISE_CHUNK_SIZE" \
    --grammars "$GRAMMAR_A" "$GRAMMAR_B" "$GRAMMAR_C" "$GRAMMAR_D" \
    $( [ $LIMIT -gt 0 ] && echo "--limit $LIMIT" ) \
    $( [ $DEBUG -eq 1 ] && echo "--debug" ) \
    --log_dir "$GENERATED_DATA_FOLDER" \

NUM_LINEAR=$(docker exec parallel_parsing sh -c "wc -l < '$LINEARISE_OUTPUT'")
NUM_LINEAR=$((NUM_LINEAR - 1))
echo "Linearisation complete. $NUM_LINEAR trees processed."

####################################
# Step 3: Validate Linearised Trees
####################################
echo "Validating linearised results..."
docker exec parallel_parsing python3 "$PYTHON_SCRIPTS_FOLDER/validate_segmentations.py" \
    "$LINEARISE_OUTPUT" \
    "$VALID_LINEARISE_OUTPUT" \
    "$INVALID_LINEARISE_OUTPUT"

echo "🎉 All steps completed for unaugmented trees."
