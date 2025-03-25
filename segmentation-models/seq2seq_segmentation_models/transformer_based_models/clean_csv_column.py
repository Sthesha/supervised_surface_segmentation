import pandas as pd
import re
import argparse

def clean_text(text):
    """Remove non-alphabetic characters and normalize to lowercase."""
    # Convert text to lowercase
    text = text.lower()
    # Remove non-alphabetic characters, keep only words with alphabets
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = ' '.join(text.split())
    return text

def process_csv(input_csv, column_name, output_txt):
    """Process the CSV file to clean and normalize specified column, and save the result to a text file."""
    # Load CSV file
    df = pd.read_csv(input_csv)
    
    # Check if the specified column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the CSV file.")
    
    # Clean the specified column
    df[column_name] = df[column_name].astype(str).apply(clean_text)
    
    # Save cleaned data to a text file
    with open(output_txt, 'w', encoding='utf-8') as file:
        for line in df[column_name]:
            file.write(f"{line}\n")
    
    print(f"Cleaned data has been saved to '{output_txt}'")

def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(description="Clean and normalize a column in a CSV file and save to a text file.")
    parser.add_argument('input_csv', help="Path to the input CSV file.")
    parser.add_argument('column_name', help="Name of the column to clean and normalize.")
    parser.add_argument('output_txt', help="Path to the output text file.")

    args = parser.parse_args()
    
    process_csv(args.input_csv, args.column_name, args.output_txt)

if __name__ == "__main__":
    main()
