#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json

BATCH_SIZE_DEFAULT = 5

def extract_all_sentences_from_json(json_filepath, num_sentences=None):
    """
    Reads the JSON file and extracts sentences from all categories,
    ignoring category distinctions. If num_sentences is provided, only that many
    sentences per category are taken; otherwise all sentences are used.
    """
    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sentences = []
    for category, entries in data.items():
        count = 0
        for entry in entries:
            if 'sentences' in entry and entry['sentences']:
                sentences.append(entry['sentences'])
                if num_sentences is not None:
                    count += 1
                    if count >= num_sentences:
                        break
    return sentences

def write_aggregated_file(sentences, output_filepath):
    """
    Writes all aggregated sentences into one file.
    """
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + "\n")
    print("Created aggregated file:", output_filepath)

def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def load_data_into_batches(aggregated_filepath, batch_size):
    """
    Reads the aggregated file and splits its lines into batches.
    """
    with open(aggregated_filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return list(chunk_list(lines, batch_size))

def write_batched_input(batches, folder):
    """
    Writes each batch to a file in the given folder.
    Batch files are named 'batch_1.txt', 'batch_2.txt', etc.
    """
    counter = 0
    for batch in batches:
        counter += 1
        batch_filename = f"batch_{counter}.txt"
        batch_filepath = os.path.join(folder, batch_filename)
        with open(batch_filepath, 'w', encoding='utf-8') as f:
            for line in batch:
                f.write(line + "\n")
        print("Created batch file:", batch_filepath)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract all sentences from a JSON file into an aggregated file and split that file into batches."
    )
    parser.add_argument('input', help="Path to the JSON file.")
    parser.add_argument('output', help="Folder where the aggregated file and batches will be saved.")
    parser.add_argument('-s', '--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help="Number of lines per batch.")
    parser.add_argument('-n', '--num_sentences', type=int, default=None,
                        help="Optional: Maximum number of sentences to extract per category.")
    args = parser.parse_args()

    # Ensure output folder exists.
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Define aggregated file path.
    aggregated_filepath = os.path.join(args.output, "sentences_to_parse.txt")
    
    # Extract all sentences from the JSON.
    sentences = extract_all_sentences_from_json(args.input, args.num_sentences)
    
    # Write aggregated file (all sentences in one file).
    write_aggregated_file(sentences, aggregated_filepath)
    
    # Create a "batches" subfolder inside the output folder.
    batches_folder = os.path.join(args.output, "batches")
    if not os.path.exists(batches_folder):
        os.makedirs(batches_folder)
    
    # Load aggregated file into batches.
    batches = load_data_into_batches(aggregated_filepath, args.batch_size)
    
    # Write each batch file into the single batches folder.
    write_batched_input(batches, batches_folder)
