import multiprocessing
import pgf
import time
import csv
import os
import logging
import pandas as pd
from itertools import product
from typing import List, Tuple, Dict, Set
import argparse
import pgfaux.linearize as li
import pgfaux.analyze as an


# Constants
SEGMENT_BOUNDARY = "-"
LINEARIZATION_TIMEOUT = 30
NUM_CORES = 50
CHUNK_SIZE = 1000

def setup_logging(log_file):
    """Set up logging configuration."""
    logging.basicConfig(
        filename=log_file,
        level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def process_single_tree(args: Tuple[str, str, pgf.PGF, str, int]) -> Tuple[str, str, str]:
    """Process a single tree with grammar."""
    tree_str, grammar_path, grammar, seg_boundary, timeout = args
    pid = os.getpid()
    
    try:
        zul = grammar.languages['ZMLargeExtChunkZul']
        chunk_subtrees = an.subtrees_of_cat(tree_str, "Chunk", grammar)
        results = []
        
        for subtree in chunk_subtrees:
            try:
                start_time = time.time()
                chunk_lin = zul.linearize(subtree)
                
                if time.time() - start_time > timeout:
                    logging.warning(f"Process {pid}: Timeout for tree {subtree}")
                    continue
                
                if chunk_lin and "[" not in chunk_lin and len(chunk_lin.split()) > 1:
                    processed_lin = post_process(chunk_lin, seg_boundary)
                    results.append((str(subtree), processed_lin))
                    
            except Exception as e:
                logging.error(f"Process {pid} linearization error for subtree {subtree}: {str(e)}")
                continue
                
        return tree_str, results, None
        
    except Exception as e:
        logging.error(f"Process {pid} error processing tree {tree_str}: {str(e)}")
        return tree_str, [], str(e)

def post_process(line: str, seg_boundary: str = SEGMENT_BOUNDARY) -> str:
    """Post-process the linearized output."""
    line = line.replace('* : ','')
    line = line.replace("* | ", "")
    line = line.replace("* ", "")
    while " | | " in line:
        line = line.replace(" | | ", " | ")
    while " : : " in line:
        line = line.replace(" : : ", " : ")

    line = line.replace(" | : ", "")
    line = line.replace(" : | ", seg_boundary)
    line = line.replace(" : ", "")
    line = line.replace(" | ", seg_boundary)
    line = line.replace("| ", "")
    line = line.replace(": ", "")
    line = line.replace(":-", "")
    line = line.replace(" |", seg_boundary)
    line = line.replace(":", "")
    return line


def get_processed_trees(output_file: str) -> Set[str]:
    """Get set of already processed trees."""
    processed = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                processed.update(row[0] for row in reader if row)
            logging.info(f"Found {len(processed)} previously processed trees")
        except Exception as e:
            logging.error(f"Error reading processed trees: {e}")
    return processed




def save_results(results: List[Tuple[str, List[Tuple[str, str]], str]], 
                output_file: str,
                found_trees: Dict[str, str],
                is_first_write: bool) -> Tuple[int, int, int]:
    """Save results and return statistics."""
    successful = 0
    duplicates = 0
    failed = 0
    
    mode = 'w' if is_first_write else 'a'
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if is_first_write:
            writer.writerow(['trees', 'phrases'])
            
        for _, subtree_results, error in results:
            if error:
                failed += 1
                continue
                
            for subtree, processed_lin in subtree_results:
                if processed_lin not in found_trees.values():
                    writer.writerow([subtree, processed_lin])
                    found_trees[subtree] = processed_lin
                    successful += 1
                else:
                    duplicates += 1
                    
    return successful, duplicates, failed

def generate_tree_txt_from_csv(csv_file: str):
    """Extract tree strings (first column) from CSV and save to .txt"""
    base, _ = os.path.splitext(csv_file)
    txt_file = base + ".txt"

    with open(csv_file, 'r', encoding='utf-8') as f_csv:
        reader = csv.reader(f_csv)
        try:
            header = next(reader)  # Skip header safely
        except StopIteration:
            print(f"⚠️ File {csv_file} is empty — no trees to extract.")
            return

        with open(txt_file, 'w', encoding='utf-8') as f_txt:
            for row in reader:
                if row and row[0].strip():
                    f_txt.write(row[0].strip() + "\n")

    print(f"Extracted trees to: {txt_file}")


def process_chunk(args: Tuple[List[str], str, str, Dict[str, str], bool, str, int]) -> Tuple[int, int, int]:
    """Process a chunk of trees and save results."""
    chunk, grammar_path, output_file, found_trees, is_first_chunk, seg_boundary, timeout = args

    # Load grammar once per process
    grammar = pgf.readPGF(grammar_path)

    # Process trees in chunk
    tasks = [(tree, grammar_path, grammar, seg_boundary, timeout) for tree in chunk]
    results = [process_single_tree(task) for task in tasks]

    # Save results to CSV only
    success, duplicates, failed = save_results(results, output_file, found_trees, is_first_chunk)
    
    return success, duplicates, failed


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process and linearize trees using PGF grammar.")
    parser.add_argument('--input', type=str, required=True, help="Path to input CSV file containing trees.")
    parser.add_argument('--output', type=str, required=True, help="Path to output CSV file for results.")
    parser.add_argument('--grammar', type=str, required=True, help="Path to PGF grammar file.")
    parser.add_argument('--log', type=str, default='linearization_errors.log', help="Path to log file.")
    parser.add_argument('--cores', type=int, default=NUM_CORES, help="Number of cores to use for multiprocessing.")
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help="Size of each chunk for multiprocessing.")
    parser.add_argument('--timeout', type=int, default=LINEARIZATION_TIMEOUT, help="Timeout for linearizing a single tree.")
    parser.add_argument('--boundary', type=str, default=SEGMENT_BOUNDARY, help="Segment boundary character.")
    parser.add_argument('--tree-column', type=str, default='tree1', help="Column name in the input file containing trees.")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    setup_logging(args.log)
    
    input_file = args.input
    output_file = args.output
    grammar_path = args.grammar
    num_cores = args.cores
    chunk_size = args.chunk_size
    timeout = args.timeout
    seg_boundary = args.boundary
    tree_column = args.tree_column
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get already processed trees
    processed_trees = get_processed_trees(output_file)
    print(f"Found {len(processed_trees)} already processed trees")

    # Read input trees
    print("Reading input trees...")
    trees = pd.read_csv(input_file)[tree_column].tolist()
    trees = [t for t in trees if t not in processed_trees][:100]

    if not trees:
        print("No new trees to process!")
        return

    print(f"Found {len(trees)} new trees to process")

    # Split into chunks
    chunks = [trees[i:i + chunk_size] for i in range(0, len(trees), chunk_size)]
    
    # Shared dictionary for found trees
    manager = multiprocessing.Manager()
    found_trees = manager.dict()

    # Prepare arguments for process_chunk
    chunk_args = [(chunk, grammar_path, output_file, found_trees, i == 0 and not processed_trees, seg_boundary, timeout) 
                 for i, chunk in enumerate(chunks)]

    # Initialize multiprocessing
    multiprocessing.set_start_method('fork', force=True)
    
    # Process using Pool
    print(f"Starting processing with {num_cores} cores...")
    total_successful = 0
    total_duplicates = 0
    total_failed = 0
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        for successful, duplicates, failed in pool.imap_unordered(process_chunk, chunk_args):
            total_successful += successful
            total_duplicates += duplicates
            total_failed += failed
            print(f"Progress: Success={total_successful}, Duplicates={total_duplicates}, Failed={total_failed}")

    generate_tree_txt_from_csv(output_file)
    
    print("\nProcessing Summary:")
    print(f"Total trees processed: {len(trees)}")
    print(f"Successfully linearized (unique): {total_successful}")
    print(f"Duplicate linearizations: {total_duplicates}")
    print(f"Failed to linearize: {total_failed}")
    print(f"Unique trees written to file: {len(found_trees)}")

if __name__ == "__main__":
    main()
