#!/usr/bin/env python3
import json
import glob
import os

def parse_json_file(file_path):
    """Parse a JSON file that might contain multiple JSON objects."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            
            # Skip empty files
            if not content:
                return None, "Empty file"
            
            # Try parsing as a single JSON object first.
            try:
                data = json.loads(content)
                return [data], None
            except json.JSONDecodeError:
                # Fall back to parsing line by line.
                valid_jsons = []
                for line in content.splitlines():
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            obj = json.loads(line)
                            valid_jsons.append(obj)
                        except json.JSONDecodeError:
                            continue
                
                if valid_jsons:
                    return valid_jsons, None
                return None, "No valid JSON objects found"
                
    except Exception as e:
        return None, str(e)

def combine_json_files(batches_folder, output_file):
    combined_data = {}
    json_files = sorted(glob.glob(os.path.join(batches_folder, '*.json')))  # Sort for consistent processing
    
    # Statistics
    total_files = len(json_files)
    processed_files = 0
    empty_files = 0
    error_files = 0
    
    print(f"Found {total_files} JSON files to process in '{batches_folder}'...")
    
    for file_name in json_files:
        try:
            data, error = parse_json_file(file_name)
            
            if error == "Empty file":
                print(f"Skipping empty file: {file_name}")
                empty_files += 1
                continue
                
            if data is None:
                print(f"Error processing {file_name}: {error}")
                error_files += 1
                continue
            
            # Use the file's basename (without extension) as key.
            key = os.path.splitext(os.path.basename(file_name))[0]
            if len(data) == 1:
                combined_data[key] = data[0]
            else:
                combined_data[key] = data
            
            processed_files += 1
            print(f"Successfully processed: {file_name}")
            
        except Exception as e:
            print(f"Unexpected error processing {file_name}: {str(e)}")
            error_files += 1
    
    # Write combined data to output file.
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(combined_data, outfile, ensure_ascii=False, indent=2)
        
        # Print summary.
        print("\nProcessing Summary:")
        print(f"Total files found: {total_files}")
        print(f"Successfully processed: {processed_files}")
        print(f"Empty files: {empty_files}")
        print(f"Files with errors: {error_files}")
        print(f"\nCombined data written to: {output_file}")
        
    except Exception as e:
        print(f"Error writing output file: {e}")

def write_consolidated_parses(args):
    # Construct the full path for the consolidated output file.
    output_file = os.path.join(args.write_folder, args.basename + '.json')
    combine_json_files(args.batches_folder, output_file)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('basename', help='Basename of the corpus file (e.g., "grouped_cleaned_output")')
    parser.add_argument('batches_folder', help='Path to the folder containing batched JSON files')
    parser.add_argument('write_folder', help='Path to the folder where the consolidated JSON file will be written')

    args = parser.parse_args()
    write_consolidated_parses(args)
