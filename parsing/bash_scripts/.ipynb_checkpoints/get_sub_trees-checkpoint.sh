#!/bin/bash

BASE_DIR="/root/cnlp/parsing/parsing"
PYTHON_SCRIPTS_FOLDER="$BASE_DIR/python_scripts"
# Directories
DATA_FOLDER="$BASE_DIR/data_folder"
GRAMMAR_FOLDER="$BASE_DIR/grammars"
PYTHON_SCRIPTS_FOLDER="$BASE_DIR/python_scripts"
BASH_SCRIPTS_FOLDER="$BASE_DIR/bash_scripts"
GENERATED_DATA_FOLDER="$BASE_DIR/parses/various_zulu_corpora_folder"  # Main output folder for parses


docker exec parallel_parsing python3 $PYTHON_SCRIPTS_FOLDER/get_sub_trees.py \
  --input $DATA_FOLDER/combined_unique.csv \
  --output $DATA_FOLDER/found_sub_trees_11.csv \
  --grammar $GRAMMAR_FOLDER/ZMLargeExtChunkA.pgf \
  --log $GENERATED_DATA_FOLDER/errors.log \
  --cores 10 \
  --chunk-size 10 \
  --timeout 30 \
  --boundary "-" \
  --tree-column "tree_1"
