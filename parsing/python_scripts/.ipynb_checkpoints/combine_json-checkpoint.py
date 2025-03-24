import json
import glob
import os

def parse_json_file(file_path):
    """Parse JSON file that might contain multiple JSON objects"""
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            
            # Skip empty files
            if not content:
                return None, "Empty file"
            
            # Try parsing as single JSON object first
            try:
                data = json.loads(content)
                return [data], None
            except json.JSONDecodeError:
                # Try parsing line by line
                valid_jsons = []
                for line in content.splitlines():
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            obj = json.loads(line)
                            valid_jsons.append(obj)
                        except json.JSONDecodeError as e:
                            continue
                
                if valid_jsons:
                    return valid_jsons, None
                return None, "No valid JSON objects found"
                
    except Exception as e:
        return None, str(e)

def combine_json_files(output_file='combined_content.json'):
    combined_data = {}
    json_files = sorted(glob.glob('*.json'))  # Sort files for consistent processing
    
    # Statistics
    total_files = len(json_files)
    processed_files = 0
    empty_files = 0
    error_files = 0
    
    print(f"Found {total_files} JSON files to process...")
    
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
            
            # Store the data
            key = os.path.splitext(file_name)[0]
            if len(data) == 1:
                # If single object, store directly
                combined_data[key] = data[0]
            else:
                # If multiple objects, store as array
                combined_data[key] = data
            
            processed_files += 1
            print(f"Successfully processed: {file_name}")
            
        except Exception as e:
            print(f"Unexpected error processing {file_name}: {str(e)}")
            error_files += 1
    
    # Write combined data to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(combined_data, outfile, ensure_ascii=False, indent=2)
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Total files found: {total_files}")
        print(f"Successfully processed: {processed_files}")
        print(f"Empty files: {empty_files}")
        print(f"Files with errors: {error_files}")
        print(f"\nCombined data written to: {output_file}")
        
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    combine_json_files()
