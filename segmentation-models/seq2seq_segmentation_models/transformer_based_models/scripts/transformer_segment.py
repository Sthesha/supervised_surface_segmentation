import os
import json
import time
import argparse
import sys

sys.path.append("/workspace/parsing/pytorch/")

from seq2seq_train import Seq2SeqTrainer 

def load_transformer_model(model_path: str):
    """Load Transformer-based segmentation model."""
    config_path = os.path.join(model_path, "model_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        transformer_config = json.load(f)
    
    return Seq2SeqTrainer(transformer_config)

def load_words(file_path: str):
    """Load words from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f.readlines()]
    return words

def main():
    parser = argparse.ArgumentParser(description="Segment words using Transformer model.")
    parser.add_argument("--source", type=str, required=True, help="Path to source text file with words to segment")
    parser.add_argument("--transformer_model", type=str, required=True, help="Path to Transformer model directory")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file to save segmentation results")

    args = parser.parse_args()

    # Load words
    words = load_words(args.source)
    print(f"Loaded {len(words)} words from {args.source}")

    # Load Transformer model
    transformer_segmenter = load_transformer_model(args.transformer_model)

    # Measure segmentation time
    start_time = time.time()
    transformer_results = [transformer_segmenter.segment_sentence(word) for word in words]
    transformer_time = time.time() - start_time
    print(f"Transformer Segmentation Time: {transformer_time:.4f} seconds")

    # Save results
    results = {
        "Transformer": {
            "time": transformer_time,
            "segmented_words": transformer_results
        }
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Segmentation results saved to {args.output}")

if __name__ == "__main__":
    main()
