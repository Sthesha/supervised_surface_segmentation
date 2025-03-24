#!/bin/bash
FILTER="?"
# The number of parses to generate.
MAX_PARSES=2
# The number of cores you want to use.
MAX_PROCESSES=10

# Set the number of sentences to extract per category.
NUM_SENTENCES=10

OUTPUT_DATA_NAME=sentences_and_trees_various
# Required to define the following based on your system:
LOCAL_GRAMMAR_LOC=./grammars
PGF_BASENAME=ZMLargeExtChunkParse
CORPUS_BASENAME=various_zulu_corpora
RUN_LABEL=various_zulu_corpora_folder

DOCKER_BASE=/root/cnlp/parsing/parsing
DOCKER_GRAMMAR_LOC=$DOCKER_BASE/grammars
LOCAL_CORPUS_LOC=$DOCKER_BASE/corpora
DOCKER_OUTPUT_LOC=$DOCKER_BASE/parses

BATCHING_SCRIPT=batch_json_sentences.py
BATCH_RUN_SCRIPT=parallel_batch_parsing_two.sh
PARSE_SCRIPT_NAME=parallel_parser.py
CONSOL_SCRIPT_NAME=combine_batches.py
CONVERTION_SCRIPT_NAME=json_to_csv.py

batching_script=$DOCKER_BASE/python_scripts/$BATCHING_SCRIPT
parse_by_batch_script=$DOCKER_BASE/bash_scripts/$BATCH_RUN_SCRIPT
parse_script=$DOCKER_BASE/python_scripts/$PARSE_SCRIPT_NAME
consol_script=$DOCKER_BASE/python_scripts/$CONSOL_SCRIPT_NAME
convention_script=$DOCKER_BASE/python_scripts/$CONVERTION_SCRIPT_NAME

grammar=$DOCKER_GRAMMAR_LOC/$PGF_BASENAME.pgf
input=$LOCAL_CORPUS_LOC/$CORPUS_BASENAME.json
output_folder=$DOCKER_OUTPUT_LOC/$RUN_LABEL

echo "CONFIG variable is set to: $LOCAL_GRAMMAR_LOC"

echo "Prepare Batches: $LOCAL_GRAMMAR_LOC"
# Run the Python script with the main output folder.
# The script creates an aggregated file (aggregated.txt) in the main folder
# and a 'batches' subfolder with all batch files.
docker exec parallel_parsing python3 $batching_script $input $output_folder -s 5 -n $NUM_SENTENCES

echo "Perform parsing: $LOCAL_GRAMMAR_LOC"
# The parse script still operates on the batches in the subfolder.
echo "Running: docker exec parallel_parsing bash $parse_by_batch_script $parse_script $output_folder/batches $grammar $FILTER $MAX_PARSES $MAX_PROCESSES"
docker exec parallel_parsing bash $parse_by_batch_script $parse_script $output_folder/batches $grammar $FILTER $MAX_PARSES $MAX_PROCESSES 

echo "Consolidate parses"
# Consolidate parses using the batches folder.
docker exec parallel_parsing python3 $consol_script $OUTPUT_DATA_NAME $output_folder/batches $output_folder

echo "converting JSON parses to CSV parses "
# Converting the JSON to CSV

docker exec parallel_parsing python3 $convention_script $output_folder/$OUTPUT_DATA_NAME.json $output_folder/$OUTPUT_DATA_NAME.csv


