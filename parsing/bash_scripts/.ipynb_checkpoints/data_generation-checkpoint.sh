#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

##############################
# Configuration Variables
##############################

# Base directory for relative paths (adjust if needed)
BASE_DIR="/root/cnlp/parsing/parsing"

# Directories
DATA_FOLDER="$BASE_DIR/data_folder"
GRAMMAR_FOLDER="$BASE_DIR/grammars"
PYTHON_SCRIPTS_FOLDER="$BASE_DIR/python_scripts"
BASH_SCRIPTS_FOLDER="$BASE_DIR/bash_scripts"
GENERATED_DATA_FOLDER="$BASE_DIR/parses/various_zulu_corpora_folder_one"  # Main output folder for parses

docker exec parallel_parsing mkdir -p $GENERATED_DATA_FOLDER


##############################
# Files for Sub-tree Extraction
##############################
INPUT_SENTENCES_AND_TREES="$DATA_FOLDER/sentences_and_trees_various.csv"
OUTPUT_FOUND_SUB_TREES="$GENERATED_DATA_FOLDER/found_sub_trees.csv"
EXTRACTION_GRAMMAR="$GRAMMAR_FOLDER/ZMLargeExtChunkParse.pgf"
ERROR_LOG="$GENERATED_DATA_FOLDER/errors.log"

# Parameters for sub-tree extraction
EXTRACTION_CORES=50
EXTRACTION_CHUNK_SIZE=10
EXTRACTION_TIMEOUT=30
EXTRACTION_BOUNDARY="-"
EXTRACTION_TREE_COLUMN="tree_1"

##############################
# Files for Tree Augmentation
##############################
AUGMENT_INPUT="$OUTPUT_FOUND_SUB_TREES"  # Use output from sub-tree extraction
AUGMENT_OUTPUT="$GENERATED_DATA_FOLDER/augmented_trees.txt"
TRACK_FILE="$GENERATED_DATA_FOLDER/transformation_track.json"
AUGMENT_TREE_COLUMN=0  # 0-based index for the tree column in CSV
AUGMENT_VARIATIONS=5

# Transformation toggles (set to 1 to disable, 0 to enable)
DISABLE_NOUNS=0
DISABLE_VERBS=0
DISABLE_STATIVE=0
DISABLE_PRODROPS=0
DISABLE_COMPLV2=0
DISABLE_POLARITY=0
DISABLE_NUMS=0
DISABLE_TENSES=0

##############################
# Files for Tree Linearisation
##############################
LINEARISE_INPUT="$AUGMENT_OUTPUT"  # Use output from augmentation
LINEARISE_OUTPUT="$GENERATED_DATA_FOLDER/augmented_and_linearised.csv"
VALID_LINEARISE_OUTPUT="$GENERATED_DATA_FOLDER/valid_augmented_and_linearised.csv"
INVALID_LINEARISE_OUTPUT="$GENERATED_DATA_FOLDER/invalid_augmented_and_linearised.csv"

LINEARISE_CORES=50
LINEARISE_TIMEOUT=3
NUM_GRAMMARS=4
TREES_PER_CHUNK=2
# Calculate chunk size automatically
LINEARISE_CHUNK_SIZE=$((NUM_GRAMMARS * TREES_PER_CHUNK))
LIMIT=100    # Set limit to 0 for no limit
DEBUG=1    # Set to 1 to enable debug

# Grammar files for linearisation
GRAMMAR_A="$GRAMMAR_FOLDER/ZMLargeExtChunkParse.pgf"
GRAMMAR_B="$GRAMMAR_FOLDER/ZMLargeExtChunkA.pgf"
GRAMMAR_C="$GRAMMAR_FOLDER/ZMLargeExtChunkB.pgf"
GRAMMAR_D="$GRAMMAR_FOLDER/ZMLargeExtChunkC.pgf"

##############################
# Step 1: Sub-tree Extraction
##############################
echo "Starting sub-tree extraction..."
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
echo "Sub-tree extraction completed."

##############################
# Step 2: Tree Augmentation
##############################
echo "Starting tree augmentation..."
docker exec parallel_parsing python3 "$PYTHON_SCRIPTS_FOLDER/linguistic_tree_augmenter.py" \
  "$AUGMENT_INPUT" \
  "$AUGMENT_OUTPUT" \
  --tree-column "$AUGMENT_TREE_COLUMN" \
  --variations "$AUGMENT_VARIATIONS" \
  --track-file "$TRACK_FILE" \
  $( [ $DISABLE_NOUNS -eq 1 ] && echo "--disable-nouns" ) \
  $( [ $DISABLE_VERBS -eq 1 ] && echo "--disable-verbs" ) \
  $( [ $DISABLE_STATIVE -eq 1 ] && echo "--disable-stative" ) \
  $( [ $DISABLE_PRODROPS -eq 1 ] && echo "--disable-prodrops" ) \
  $( [ $DISABLE_COMPLV2 -eq 1 ] && echo "--disable-complv2" ) \
  $( [ $DISABLE_POLARITY -eq 1 ] && echo "--disable-polarity" ) \
  $( [ $DISABLE_NUMS -eq 1 ] && echo "--disable-nums" ) \
  $( [ $DISABLE_TENSES -eq 1 ] && echo "--disable-tenses" )
  
# Check if augmentation was successful and count lines (excluding header)
NUM_AUGMENTED=$(docker exec parallel_parsing sh -c "wc -l < '$AUGMENT_OUTPUT'")
NUM_AUGMENTED=$((NUM_AUGMENTED - 1))
echo "Tree augmentation completed successfully."
echo "Number of augmented trees generated: $NUM_AUGMENTED"
echo "Output written to: $AUGMENT_OUTPUT"
echo "Transformation tracking data saved to: $TRACK_FILE"


##############################
# Step 3: Tree Linearisation
##############################
echo "Starting tree linearisation..."
echo "Using chunk size of $LINEARISE_CHUNK_SIZE (processing $TREES_PER_CHUNK trees at a time)"
docker exec parallel_parsing python3 "$PYTHON_SCRIPTS_FOLDER/linearise_with_grammars.py" \
    "$LINEARISE_INPUT" \
    "$LINEARISE_OUTPUT" \
    --cores "$LINEARISE_CORES" \
    --timeout "$LINEARISE_TIMEOUT" \
    --chunk-size "$LINEARISE_CHUNK_SIZE" \
    --grammars "$GRAMMAR_A" "$GRAMMAR_B" "$GRAMMAR_C" "$GRAMMAR_D" \
    $( [ $LIMIT -gt 0 ] && echo "--limit $LIMIT" ) \
    $( [ $DEBUG -eq 1 ] && echo "--debug" ) \
    --log_dir $GENERATED_DATA_FOLDER \
    
# Check if linearisation was successful and count processed trees (excluding header)
NUM_LINEAR=$(docker exec parallel_parsing sh -c "wc -l < '$LINEARISE_OUTPUT'")
NUM_LINEAR=$((NUM_LINEAR - 1))
echo "Tree linearisation completed successfully."

##############################
# Step 4: Validate Linearised Trees
##############################
echo "Validating linearised trees..."
docker exec parallel_parsing python3 "$PYTHON_SCRIPTS_FOLDER/validate_segmentations.py" "$LINEARISE_OUTPUT" "$VALID_LINEARISE_OUTPUT" "$INVALID_LINEARISE_OUTPUT"


echo "All steps completed successfully."
