import json
import re
import random
import csv
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime

@dataclass
class TreeTransformation:
    """Records a transformation applied to a linguistic tree"""
    chunk_type: str
    element_type: str
    original_value: str
    new_value: str
    position: int
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class AugmentedLinguisticTree:
    """Represents an augmented linguistic tree with its transformation history"""
    original_tree: str
    current_tree: str
    tree_id: str = None
    transformations: List[TreeTransformation] = None

    def __post_init__(self):
        if self.tree_id is None:
            self.tree_id = f"tree_{random.randint(0, 999999):06d}"
        if self.transformations is None:
            self.transformations = []

    def to_dict(self) -> Dict:
        return {
            'tree_id': self.tree_id,
            'original_tree': self.original_tree,
            'current_tree': self.current_tree,
            'transformations': [vars(t) for t in self.transformations]
        }

class LinguisticTreeAugmenter:
    def __init__(self):
        self.patterns = {
            'UseN': re.compile(r'\(UseN ([^_]+)_\d+(?:_\d+)?_N\)'),
            'UseV': re.compile(r'\(UseV ([^_]+)_V\)'),
            'UseVStative': re.compile(r'\(UseVStative ([^_]+)_V\)'),
            'ComplV2': re.compile(r'ComplV2 ([^_\s]+)_V2'),
            'ProDrop': re.compile(r'\(ProDrop ([^_\s]+?)_Pron\)'),
            'PolarityPattern': re.compile(r'\b(PPos|PNeg)\b'),
            'Number': re.compile(r'\b(NumSg|NumPl)\b'),
            'Tense': re.compile(r'\b(TRemPastTemp|TPresTemp|TPastTemp|TFutTemp)\b')
        }
        
        self.element_cache = {
            'UseN': set(),
            'UseV': set(),
            'UseVStative': set(),
            'ComplV2': set(),
            'ProDrop': set(),
            'PolarityPattern': {'PPos', 'PNeg'},
            'Number': {'NumSg', 'NumPl'},
            'Tense': {'TRemPastTemp', 'TPresTemp', 'TPastTemp', 'TFutTemp'}
        }

    def extract_elements(self, tree: str):
        """Extract all linguistic elements from a tree"""
        for element_type, pattern in self.patterns.items():
            matches = pattern.findall(tree)
            if isinstance(matches, list):
                self.element_cache[element_type].update(matches)

    def transform_element(self, tree: str, element_type: str) -> Tuple[str, List[TreeTransformation]]:
        """Transform specific elements in the tree while maintaining structure"""
        transformations = []
        modified_tree = tree
        pattern = self.patterns[element_type]
        
        matches = pattern.finditer(tree)
        for match in matches:
            original = match.group(1)
            possible_replacements = self.element_cache[element_type] - {original}
            if possible_replacements:
                new_value = random.choice(list(possible_replacements))
                
                if element_type == 'UseN':
                    old_str = match.group(0)
                    number_class = old_str[old_str.index('_'):old_str.rindex('_')]
                    new_str = f'(UseN {new_value}{number_class}_N)'
                elif element_type == 'UseV':
                    old_str = match.group(0)
                    new_str = f'(UseV {new_value}_V)'
                elif element_type == 'UseVStative':
                    old_str = match.group(0)
                    new_str = f'(UseVStative {new_value}_V)'
                elif element_type == 'ProDrop':
                    old_str = match.group(0)
                    new_str = f'(ProDrop {new_value}_Pron)'
                elif element_type == 'ComplV2':
                    old_str = f'ComplV2 {original}_V2'
                    new_str = f'ComplV2 {new_value}_V2'
                else:
                    old_str = original
                    new_str = new_value
                
                modified_tree = modified_tree.replace(old_str, new_str, 1)
                
                transformations.append(TreeTransformation(
                    chunk_type=tree.split()[0] if tree.split() else "Unknown",
                    element_type=element_type,
                    original_value=original,
                    new_value=new_value,
                    position=match.start()
                ))
        
        return modified_tree, transformations

    def augment(self, trees: List[str], num_variations: int = 5, disabled_elements: Set[str] = None) -> List[AugmentedLinguisticTree]:
        """Generate augmented variations of the input trees"""
        if disabled_elements is None:
            disabled_elements = set()
            
        augmented_trees = []
        
        # First pass: extract all possible elements
        for tree in trees:
            self.extract_elements(tree)
        
        # Second pass: generate variations
        for original_tree in trees:
            # Create base tree object
            base_tree = AugmentedLinguisticTree(
                original_tree=original_tree,
                current_tree=original_tree
            )
            augmented_trees.append(base_tree)
            
            # Generate variations
            for _ in range(num_variations):
                current_tree = original_tree
                transformations = []
                
                available_transformations = [t for t in self.patterns.keys() 
                                          if t not in disabled_elements]
                
                # Apply random transformations
                for element_type in random.sample(available_transformations, 
                                                random.randint(1, len(available_transformations))):
                    modified_tree, new_transformations = self.transform_element(current_tree, element_type)
                    if modified_tree != current_tree:
                        current_tree = modified_tree
                        transformations.extend(new_transformations)
                
                if transformations:
                    augmented_trees.append(AugmentedLinguisticTree(
                        original_tree=original_tree,
                        current_tree=current_tree,
                        transformations=transformations
                    ))
        
        return augmented_trees

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Linguistic tree augmentation")
    parser.add_argument("input", help="Input CSV file with trees")
    parser.add_argument("output", help="Output CSV file for augmented trees")
    parser.add_argument("--tree-column", type=int, default=0, help="Column index containing trees (0-based)")
    parser.add_argument("--variations", type=int, default=5, help="Number of variations per tree")
    parser.add_argument("--track-file", help="JSON file to store transformation data")
    parser.add_argument("--disable-nouns", action="store_true", help="Disable shuffling nouns")
    parser.add_argument("--disable-verbs", action="store_true", help="Disable shuffling verbs")
    parser.add_argument("--disable-stative", action="store_true", help="Disable shuffling stative verbs")
    parser.add_argument("--disable-prodrops", action="store_true", help="Disable shuffling ProDrop elements")
    parser.add_argument("--disable-complv2", action="store_true", help="Disable shuffling ComplV2 elements")
    parser.add_argument("--disable-polarity", action="store_true", help="Disable shuffling PPos/PNeg patterns")
    parser.add_argument("--disable-nums", action="store_true", help="Disable shuffling NumSg/NumPl patterns")
    parser.add_argument("--disable-tenses", action="store_true", help="Disable shuffling tenses")
    
    args = parser.parse_args()
    
    disabled_elements = {
        'UseN': args.disable_nouns,
        'UseV': args.disable_verbs,
        'UseVStative': args.disable_stative,
        'ProDrop': args.disable_prodrops,
        'ComplV2': args.disable_complv2,
        'PolarityPattern': args.disable_polarity,
        'Number': args.disable_nums,
        'Tense': args.disable_tenses
    }
    
    disabled_set = {elem for elem, disabled in disabled_elements.items() if disabled}
    
    # Read trees from CSV
    try:
        trees = []
        with open(args.input, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip header row
            for row in csv_reader:
                if len(row) > args.tree_column:
                    trees.append(row[args.tree_column])
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return

    print(f"Read {len(trees)} trees from CSV")

    # Perform augmentation
    augmenter = LinguisticTreeAugmenter()
    augmented_trees = augmenter.augment(trees, args.variations, disabled_set)

    # Write augmented trees to file
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            for tree in augmented_trees:
                f.write(f"{tree.current_tree}\n")
        print(f"Successfully saved {len(augmented_trees)} augmented trees")
    except Exception as e:
        print(f"Error writing output file: {e}")

    # Save transformation tracking data if requested
    if args.track_file:
        try:
            tracking_data = [tree.to_dict() for tree in augmented_trees]
            with open(args.track_file, 'w', encoding='utf-8') as f:
                json.dump(tracking_data, f, indent=2, ensure_ascii=False)
            print(f"Saved transformation tracking data to {args.track_file}")
        except Exception as e:
            print(f"Error writing tracking file: {e}")

if __name__ == "__main__":
    main()