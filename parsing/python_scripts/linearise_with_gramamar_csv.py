import argparse
import multiprocessing
import pgf
import time
import csv
import os
import logging
from itertools import product
from typing import List, Tuple, Set

# Constants
SEGMENT_BOUNDARY = "-"

def setup_logging(debug: bool = False):
    """Set up logging configuration."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f"tree_processing_{int(time.time())}.log")
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        filename=log_file,
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

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

def read_trees_from_csv(input_file: str, tree_column: str) -> List[str]:
    """Read trees from a specific column in a CSV file."""
    trees = []
    try:
        # Convert relative path to absolute path
        input_file = os.path.abspath(input_file)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if tree_column not in reader.fieldnames:
                raise ValueError(f"Column '{tree_column}' not found in CSV. Available columns: {', '.join(reader.fieldnames)}")
            
            trees = [row[tree_column].strip() for row in reader if row[tree_column].strip()]
            logging.info(f"Successfully read {len(trees)} trees from CSV column '{tree_column}'")
            
    except Exception as e:
        logging.error(f"Error reading input CSV: {e}")
        raise
        
    return trees

def process_single_tree(args: Tuple[str, str, int]) -> Tuple[str, str, str, str]:
    """Process a single tree with a single grammar."""
    tree_str, grammar_path, timeout = args
    pid = os.getpid()
    
    try:
        # Convert relative path to absolute path for grammar
        grammar_path = os.path.abspath(grammar_path)
        
        # Load grammar inside the process
        gr = pgf.readPGF(grammar_path)
        lang = next(iter(gr.languages.values()))
        
        # Parse and linearize
        expr = pgf.readExpr(tree_str)
        start_time = time.time()
        linearized_version = lang.linearize(expr)
        try:
            linearized_version = lang.linearize(expr)
            if time.time() - start_time > timeout:
                logging.warning(f"Process {pid}: Timeout on tree {tree_str}")
                return tree_str, grammar_path, None, "Timeout"
                
            if "[" in linearized_version:
                logging.warning(f"Process {pid}: Invalid output for tree {tree_str}")
                return tree_str, grammar_path, None, "Invalid output"
                
            result = post_process(linearized_version, SEGMENT_BOUNDARY)
            return tree_str, grammar_path, result, None
            
        except Exception as e:
            logging.error(f"Process {pid} linearization error: {str(e)}")
            return tree_str, grammar_path, None, f"Linearization error: {str(e)}"
            
    except Exception as e:
        logging.error(f"Process {pid} grammar loading error: {str(e)}")
        return tree_str, grammar_path, None, f"Grammar loading error: {str(e)}"

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

def chunk_tasks(tasks: List[Tuple[str, str, int]], chunk_size: int) -> List[List[Tuple[str, str, int]]]:
    """Split tasks into chunks for better load balancing."""
    return [tasks[i:i + chunk_size] for i in range(0, len(tasks), chunk_size)]

def process_chunk(chunk: List[Tuple[str, str, int]]) -> List[Tuple[str, str, str, str]]:
    """Process a chunk of tasks and return results."""
    return [process_single_tree(task) for task in chunk]

def save_chunk_results(results: List[Tuple[str, str, str, str]], 
                      output_file: str,
                      grammars: List[str],
                      is_first_chunk: bool) -> Tuple[int, int]:
    """Save chunk results incrementally and return statistics."""
    successful = 0
    failed = 0
    
    try:
        # Convert relative path to absolute path
        output_file = os.path.abspath(output_file)
        
        # Organize results by tree for this chunk
        tree_results = {}
        for tree_str, grammar_path, result, error in results:
            if tree_str not in tree_results:
                tree_results[tree_str] = [None] * len(grammars)
                
            grammar_index = grammars.index(os.path.abspath(grammar_path))
            tree_results[tree_str][grammar_index] = result if result else f"Error: {error}"

        # Filter results where all grammars succeeded
        filtered_results = {
            tree_str: linearizations 
            for tree_str, linearizations in tree_results.items()
            if all(not str(lin).startswith("Error") for lin in linearizations)
        }

        # Write results
        mode = 'w' if is_first_chunk else 'a'
        with open(output_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if is_first_chunk:
                # Create column headers starting with Parse, then A, B, C
                headers = ["Tree", "Parse"]
                headers.extend([f"Linearized{chr(65 + i)}" for i in range(len(grammars) - 1)])
                writer.writerow(headers)
            
            for tree_str, linearizations in filtered_results.items():
                writer.writerow([tree_str] + linearizations)
                successful += 1

        failed = len(tree_results) - successful
        logging.info(f"Chunk processed: {successful} successful, {failed} failed")
        
    except Exception as e:
        logging.error(f"Error saving chunk results: {e}")
        raise
        
    return successful, failed

def main():
    parser = argparse.ArgumentParser(description="Parallel processing of trees with grammars.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file containing trees")
    parser.add_argument("tree_column", type=str, help="Name of the column containing trees in the CSV")
    parser.add_argument("output_file", type=str, help="Path to the output CSV file")
    parser.add_argument("--cores", type=int, default=50, help="Number of CPU cores to use")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout for linearization in seconds")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of processing chunks")
    parser.add_argument("--limit", type=int, default=0, help="Limit on number of trees to process (0 for no limit)")
    parser.add_argument("--grammars", nargs="+", required=True, help="List of grammar file paths")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    try:
        args = parser.parse_args()
        setup_logging(args.debug)
        
        # Create output directory if needed
        output_dir = os.path.dirname(os.path.abspath(args.output_file))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Get already processed trees
        processed_trees = get_processed_trees(args.output_file)
        
        # Read input trees from CSV
        logging.info(f"Reading input trees from CSV column '{args.tree_column}'...")
        trees = read_trees_from_csv(args.input_file, args.tree_column)
        trees = [tree for tree in trees if tree not in processed_trees]
            
        if args.limit > 0:
            trees = trees[:args.limit]
            
        if not trees:
            print("No new trees to process!")
            return 0
            
        total_trees = len(trees)
        logging.info(f"Found {total_trees} new trees to process")
        
        # Convert grammar paths to absolute paths
        grammar_paths = [os.path.abspath(g) for g in args.grammars]
        
        # Create all combinations of trees and grammars
        tasks = list(product(trees, grammar_paths, [args.timeout]))
        chunks = chunk_tasks(tasks, args.chunk_size)
        
        # Initialize multiprocessing
        multiprocessing.set_start_method('fork', force=True)
        
        # Process using Pool
        total_successful = 0
        total_failed = 0
        processed_chunks = 0
        total_chunks = len(chunks)
        
        with multiprocessing.Pool(processes=args.cores) as pool:
            for chunk_result in pool.imap_unordered(process_chunk, chunks):
                successful, failed = save_chunk_results(
                    chunk_result, 
                    args.output_file,
                    grammar_paths,
                    processed_chunks == 0 and not processed_trees
                )
                
                total_successful += successful
                total_failed += failed
                processed_chunks += 1
                
                if args.debug or processed_chunks % max(1, total_chunks//4) == 0:
                    progress = (processed_chunks / total_chunks) * 100
                    logging.info(f"Progress: {progress:.1f}% - Success: {total_successful}, Failed: {total_failed}")
                    print(f"Progress: {progress:.1f}% - Success: {total_successful}, Failed: {total_failed}")
        
        print("\nProcessing Summary:")
        print(f"Total trees processed: {total_trees}")
        print(f"Successfully processed: {total_successful}")
        print(f"Failed to process: {total_failed}")
        print(f"Output written to: {args.output_file}")
        print("Detailed logs available in: logs/tree_processing_*.log")
        
        return 0
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        print(f"Error: Tree processing failed - {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())