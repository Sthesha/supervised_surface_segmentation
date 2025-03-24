import csv
import argparse
from typing import List, Tuple
from collections import defaultdict

def has_invalid_vowel_sequence(word: str) -> bool:
    word = word.lower()
    for i in range(len(word) - 1):
        if word[i] == word[i + 1] and word[i:i + 2] != 'hh':
            return True
    return False

def clean_word(word: str) -> str:
    return word.replace('-', '').strip()

def extract_word_pairs(line: str) -> List[Tuple[str, List[str]]]:
    parts = line.strip().split(',')
    if len(parts) < 5:
        return []

    words_a = parts[1].strip().split()
    words_b = parts[2].strip().split()
    words_c = parts[3].strip().split()
    words_d = parts[4].strip().split()

    min_len = min(len(words_a), len(words_b), len(words_c), len(words_d))
    result = []

    for i in range(min_len):
        original = words_a[i]
        segments = [
            words_b[i] if i < len(words_b) else '',
            words_c[i] if i < len(words_c) else '',
            words_d[i] if i < len(words_d) else ''
        ]
        result.append((original, segments))

    return result

def validate_word_and_segments(word: str, segments: List[str]) -> bool:
    if has_invalid_vowel_sequence(word):
        return False

    for seg in segments:
        if seg and has_invalid_vowel_sequence(seg):
            return False

    word = clean_word(word)
    return all(clean_word(seg) == word for seg in segments if seg)

def process_file(input_file: str, valid_output: str, invalid_output: str):
    processed_words = set()
    valid_entries = set()
    invalid_entries = set()

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            word_pairs = extract_word_pairs(line)

            for word, segments in word_pairs:
                if word in processed_words:
                    continue

                entry = (word, *segments)

                if validate_word_and_segments(word, segments):
                    valid_entries.add(entry)
                else:
                    invalid_entries.add(entry)

                processed_words.add(word)

    # Write valid entries
    with open(valid_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Word', 'SegmentationA', 'SegmentationB', 'SegmentationC'])
        writer.writerows(sorted(valid_entries, key=lambda x: x[0].lower()))

    # Write invalid entries
    with open(invalid_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Word', 'SegmentationA', 'SegmentationB', 'SegmentationC'])
        writer.writerows(sorted(invalid_entries, key=lambda x: x[0].lower()))

    print(f"Total valid entries: {len(valid_entries)}")
    print(f"Total invalid entries: {len(invalid_entries)}")
    print(f"Valid entries written to: {valid_output}")
    print(f"Invalid entries written to: {invalid_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate segmentations in a linearized grammar output.")
    parser.add_argument("input_file", help="Path to the input CSV file")
    parser.add_argument("valid_output", help="Path to save valid segmentations")
    parser.add_argument("invalid_output", help="Path to save invalid segmentations")
    args = parser.parse_args()

    process_file(args.input_file, args.valid_output, args.invalid_output)