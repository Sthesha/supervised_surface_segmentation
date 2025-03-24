#!/bin/bash

# Set variables
INPUT_FILE="../data_folder/augmented_trees.txt"
OUTPUT_FILE="../data_folder/augmented_and_linearised.csv"
NUM_CORES=50  # Can be any number now
TIMEOUT=30
NUM_GRAMMARS=4
TREES_PER_CHUNK=2  # Number of complete trees to process per chunk
CHUNK_SIZE=$((NUM_GRAMMARS * TREES_PER_CHUNK))  # Automatically calculate chunk size
LIMIT=0
DEBUG=1

# Grammar files
GRAMMAR_A="../grammars/ZMLargeExtChunkParse.pgf"
GRAMMAR_B="../grammars/ZMLargeExtChunkA.pgf"
GRAMMAR_C="../grammars/ZMLargeExtChunkB.pgf"
GRAMMAR_D="../grammars/ZMLargeExtChunkC.pgf"

echo "Using chunk size of $CHUNK_SIZE (processing $TREES_PER_CHUNK trees at a time)"

# Run the Python script with all grammars
python3 ../python_scripts/linearise_with_grammars.py \
    "$INPUT_FILE" \
    "$OUTPUT_FILE" \
    --cores "$NUM_CORES" \
    --timeout "$TIMEOUT" \
    --chunk-size "$CHUNK_SIZE" \
    --grammars "$GRAMMAR_A" "$GRAMMAR_B" "$GRAMMAR_C" "$GRAMMAR_D" \
    $( [ $LIMIT -gt 0 ] && echo "--limit $LIMIT" ) \
    $( [ $DEBUG -eq 1 ] && echo "--debug" )

# Check if the command was successful
if [ $? -eq 0 ]; then
    NUM_PROCESSED=$(wc -l < "$OUTPUT_FILE")
    NUM_PROCESSED=$((NUM_PROCESSED - 1))  # Exclude header row
    echo "Tree processing completed successfully"
    echo "Number of trees processed: $NUM_PROCESSED"
    echo "Output written to: $OUTPUT_FILE"
    echo "Detailed logs available in: logs/tree_processing_*.log"
else
    echo "Error: Tree processing failed"
    exit 1
fi