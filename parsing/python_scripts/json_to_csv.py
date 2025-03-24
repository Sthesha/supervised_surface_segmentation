import json
import csv
import argparse
import os

def json_to_csv(input_json_file, output_csv_path):
    with open(input_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_rows = []

    for batch_key, content in data.items():
        if isinstance(content, list):
            # Case: List of dictionaries [{sentence: [tree1, tree2]}, ...]
            for entry in content:
                for sentence, trees in entry.items():
                    row = [sentence] + trees
                    all_rows.append(row)
        elif isinstance(content, dict):
            # Case: Direct dictionary {sentence: [tree1, tree2], ...}
            for sentence, trees in content.items():
                row = [sentence] + trees
                all_rows.append(row)
        else:
            print(f"Skipping {batch_key}: unexpected format")

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Find the max number of trees to define headers dynamically
    max_trees = max(len(row) - 1 for row in all_rows)
    headers = ["sentence"] + [f"tree_{i+1}" for i in range(max_trees)]

    # Write to CSV
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for row in all_rows:
            # Pad rows with fewer trees than max
            row += [""] * (1 + max_trees - len(row))
            writer.writerow(row)

    print(f"✅ CSV saved to: {output_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Path to the input JSON file")
    parser.add_argument("csv_output", help="Path to the output CSV file")
    args = parser.parse_args()

    json_to_csv(args.json_file, args.csv_output)
