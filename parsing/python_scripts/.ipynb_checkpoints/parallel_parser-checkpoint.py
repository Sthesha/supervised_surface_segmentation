import pgf
import argparse
import json
import os
import time
from multiprocessing import Pool, TimeoutError
import fcntl

MAX_PARSES = 2
MAX_PROCESSES = 20
TIMEOUT = 120  # Maximum time (in seconds) allowed for parsing a single sentence

conc_grammar = None
filters = None

def init_worker(grammar_path, lang_code, shared_filters):
    global conc_grammar, filters
    grammar = pgf.readPGF(grammar_path)
    conc_name = grammar.abstractName[:-3] + lang_code if grammar.abstractName[-3:] == 'Abs' else grammar.abstractName + lang_code
    conc_grammar = grammar.languages[conc_name]
    filters = shared_filters

def parse_sentence(input):
    global conc_grammar, filters
    parse = {input: []}
    try:
        p_iter = conc_grammar.parse(input)
        parses = 0
        while parses < MAX_PARSES:
            try:
                (p, e) = next(p_iter)
                to_filter = any(f in str(e) for f in filters)
                if not to_filter:
                    parse[input].append(str(e))
                    parses += 1
            except StopIteration:
                break
        return parse if parse[input] else (None, f"No valid parses found for sentence: {input}")
    except pgf.ParseError:
        return None, f"Parse error for sentence: {input}"
    except Exception as e:
        return None, f"Unexpected error for sentence: {input}. Error: {str(e)}"

def write_with_lock(file_path, content):
    with open(file_path, 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)

def process_results(result, input, output_file, log_file):
    if isinstance(result, dict):
        content = json.dumps(result) + '\n'
        write_with_lock(output_file, content)
    elif isinstance(result, tuple):
        write_with_lock(log_file, f"{result[1]}\n")
    else:
        write_with_lock(log_file, f"Unexpected result type for sentence: {input}\n")

def batch_parses_as_strings(inputs, grammar_path, lang_code, shared_filters, output_file, log_file):
    with Pool(processes=MAX_PROCESSES, initializer=init_worker, initargs=(grammar_path, lang_code, shared_filters)) as pool:
        results = []
        for input in inputs:
            results.append((input, pool.apply_async(parse_sentence, (input,))))
        
        for input, res in results:
            try:
                result = res.get(timeout=TIMEOUT)
                process_results(result, input, output_file, log_file)
            except TimeoutError:
                write_with_lock(log_file, f"Parsing took too long for sentence: {input}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", dest="grammar", help="PGF grammar file path")
    parser.add_argument("-c", dest="lang_code", help="3-letter ISO code, eg Zul")
    parser.add_argument("input", help="filename of line separated sentences")
    parser.add_argument("output", help="write trees to...")
    parser.add_argument("log", help="filename for log on failures and successes")
    parser.add_argument("-m", dest="max_parses", type=int, default=MAX_PARSES, help="maximum number of parses")
    parser.add_argument("-f", dest="filters", nargs='*', help="give strings to filter out")
    parser.add_argument("-p", dest="max_processes", type=int, default=MAX_PROCESSES, help="maximum number of parallel processes")
    parser.add_argument("-t", dest="timeout", type=int, default=TIMEOUT, help="timeout for parsing a single sentence")

    args = parser.parse_args()

    MAX_PARSES = args.max_parses
    MAX_PROCESSES = args.max_processes
    TIMEOUT = args.timeout

    shared_filters = args.filters if args.filters else []

    with open(args.input) as f:
        lines = [l.strip() for l in f.readlines()]

    # Clear output and log files
    open(args.output, 'w').close()
    open(args.log, 'w').close()

    start_time = time.time()
    batch_parses_as_strings(lines, args.grammar, args.lang_code, shared_filters, args.output, args.log)
    end_time = time.time()

    print(f"Total processing time: {end_time - start_time:.2f} seconds")
