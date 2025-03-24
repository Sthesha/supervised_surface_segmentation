#!/bin/bash

PARSE_SCRIPT=$1
BATCH_FOLDER=$2
GRAMMAR=$3
FILTERS=$4
MAX_PARSES=$5
MAX_PARALLEL_PROCESSES=$6  # Use the provided value for MAX_PARALLEL_PROCESSES

# Create an array to store batch file paths
batch_files=()

# Collect batch file paths in the array
for batch_file in $BATCH_FOLDER/batch_*.txt; do
  batch_files+=($batch_file)
done

# Process batch files in parallel using xargs
printf "%s\n" "${batch_files[@]}" | xargs -n 1 -P $MAX_PARALLEL_PROCESSES -I {} bash -c "filename=\$(basename {}); filename_noext=\${filename%.*}; dirname=\$(dirname {}); python3 $PARSE_SCRIPT -g $GRAMMAR -f $FILTERS -m $MAX_PARSES -c Zul {} \$dirname/\$filename_noext.json \$dirname/\$filename_noext.log"

