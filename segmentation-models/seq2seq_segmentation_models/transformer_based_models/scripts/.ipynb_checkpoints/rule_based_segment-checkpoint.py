import os
import json
import time
import argparse
import pgf

# Segmentation boundary for rule-based method
SEGMENT_BOUNDARY = "-"

def post_process(line: str, seg_boundary: str = SEGMENT_BOUNDARY) -> str:
    """Post-process the linearized output to format segmentation."""
    line = line.replace('* : ', '')
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
    
def load_rule_based_models(parse_grammar_path: str, linearize_grammar_path: str):
    """Load Rule-based parsing and linearization grammars."""
    parse_grammar = pgf.readPGF(os.path.abspath(parse_grammar_path))
    linearize_grammar = pgf.readPGF(os.path.abspath(linearize_grammar_path))
    zul_parse = next(iter(parse_grammar.languages.values()))  # Parsing grammar
    zul_linearize = next(iter(linearize_grammar.languages.values()))  # Linearization grammar
    return zul_parse, zul_linearize

def rule_based_segment(word: str, zul_parse, zul_linearize) -> str:
    """Segment a word using rule-based parsing and linearization grammars."""
    try:
        parsed = zul_parse.parse(word)
        prob, expr = next(parsed)  # Get first parse (returns tuple)

        # Linearize using separate grammar
        segmented_output = zul_linearize.linearize(expr)

        # Post-process the segmented output
        return post_process(segmented_output, SEGMENT_BOUNDARY)

    except StopIteration:
        print(f"No valid parse found for '{word}'")
        return word  # Return original word if no parse is found

    except Exception as e:
        print(f"Error processing '{word}': {e}")
        return word  # Return original word if any error occurs

def load_words(file_path: str):
    """Load words from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f.readlines()]
    return words

def main():
    parser = argparse.ArgumentParser(description="Segment words using Rule-Based Model.")
    parser.add_argument("--source", type=str, required=True, help="Path to source text file with words to segment")
    parser.add_argument("--parse_grammar", type=str, required=True, help="Path to parsing grammar PGF file")
    parser.add_argument("--linearize_grammar", type=str, required=True, help="Path to linearization grammar PGF file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file to save segmentation results")

    args = parser.parse_args()

    # Load words
    words = load_words(args.source)
    print(f"Loaded {len(words)} words from {args.source}")

    # Load Rule-based grammars
    zul_parse, zul_linearize = load_rule_based_models(args.parse_grammar, args.linearize_grammar)

    # Measure segmentation time
    start_time = time.time()
    rule_based_results = [rule_based_segment(word, zul_parse, zul_linearize) for word in words]
    rule_based_time = time.time() - start_time
    print(f"Rule-Based Segmentation Time: {rule_based_time:.4f} seconds")

    # Save results
    results = {
        "Rule-Based": {
            "time": rule_based_time,
            "segmented_words": rule_based_results
        }
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Segmentation results saved to {args.output}")

if __name__ == "__main__":
    main()