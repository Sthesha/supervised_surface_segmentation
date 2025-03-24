#!/bin/bash

# Set variables
INPUT_FILE="../data_folder/found_sub_trees.csv"
OUTPUT_FILE="../data_folder/augmented_trees.txt"
TRACK_FILE="../data_folder/transformation_track.json"
TREE_COLUMN=0  # Column index (0-based) containing trees
VARIATIONS=5  # Number of variations per tree

# Disable transformations as needed (set to 1 to disable)
DISABLE_NOUNS=0
DISABLE_VERBS=0
DISABLE_STATIVE=0
DISABLE_PRODROPS=0
DISABLE_COMPLV2=0
DISABLE_POLARITY=0
DISABLE_NUMS=0
DISABLE_TENSES=0

# Run the Python script
python3 linguistic_tree_augmenter.py \
  "$INPUT_FILE" \
  "$OUTPUT_FILE" \
  --tree-column $TREE_COLUMN \
  --variations $VARIATIONS \
  --track-file "$TRACK_FILE" \
  $( [ $DISABLE_NOUNS -eq 1 ] && echo "--disable-nouns" ) \
  $( [ $DISABLE_VERBS -eq 1 ] && echo "--disable-verbs" ) \
  $( [ $DISABLE_STATIVE -eq 1 ] && echo "--disable-stative" ) \
  $( [ $DISABLE_PRODROPS -eq 1 ] && echo "--disable-prodrops" ) \
  $( [ $DISABLE_COMPLV2 -eq 1 ] && echo "--disable-complv2" ) \
  $( [ $DISABLE_POLARITY -eq 1 ] && echo "--disable-polarity" ) \
  $( [ $DISABLE_NUMS -eq 1 ] && echo "--disable-nums" ) \
  $( [ $DISABLE_TENSES -eq 1 ] && echo "--disable-tenses" )

# Check if the command was successful
# Check if the command was successful
if [ $? -eq 0 ]; then
    NUM_TREES=$(wc -l < "$OUTPUT_FILE")
    NUM_TREES=$((NUM_TREES - 1))  # Exclude header row
    echo "Tree augmentation completed successfully"
    echo "Number of augmented trees generated: $NUM_TREES"
    echo "Output written to: $OUTPUT_FILE"
    echo "Transformation tracking data saved to: $TRACK_FILE"
else
    echo "Error: Tree augmentation failed"
    exit 1
fi