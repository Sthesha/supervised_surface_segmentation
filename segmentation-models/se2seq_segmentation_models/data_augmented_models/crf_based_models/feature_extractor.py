# feature_extractor.py
import warnings

class FeatureExtractor:
    """Handles feature extraction for morphological segmentation"""
    
    def __init__(self, config: dict):
        """
        Initialize feature extractor with configuration
        
        Args:
            config: Dictionary containing feature extraction parameters:
                - context_window: Size of context window (default: 3)
                - ngram_range: Tuple of (min_n, max_n) for n-grams (default: (1, 3))
                - use_position_features: Whether to use position features (default: True)
                - use_character_features: Whether to use character features (default: True)
                - use_pattern_features: Whether to use pattern features (default: True)
        """
        self.context_window = config.get('context_window', 3)
        self.min_n, self.max_n = config.get('ngram_range', (1, 3))
        self.use_position_features = config.get('use_position_features', True)
        self.use_character_features = config.get('use_character_features', True)
        self.use_pattern_features = config.get('use_pattern_features', True)
    
    def extract_character_features(self, word, i):
        """Extract features for a single character position"""
        features = {}
        
        if self.use_character_features:
            # Basic character features
            features["char"] = word[i]
            features["lower"] = word[i].lower()
            
            if self.use_position_features:
                # Position features
                features["start"] = i == 0
                features["end"] = i == len(word) - 1
                features["position"] = i
                features["word_length"] = len(word)
            
            # Context window features
            for j in range(-self.context_window, self.context_window + 1):
                if 0 <= i + j < len(word):
                    features[f"char_{j}"] = word[i + j]
                    features[f"is_vowel_{j}"] = word[i + j].lower() in 'aeiou'
            
            # N-gram features
            for n in range(self.min_n, self.max_n + 1):
                if i >= n:
                    features[f"prev_{n}gram"] = word[i-n:i]
                if i + n <= len(word):
                    features[f"next_{n}gram"] = word[i:i+n]
            
            # Character type features
            features["is_vowel"] = word[i].lower() in 'aeiou'
            features["is_consonant"] = word[i].lower() not in 'aeiou'
        
        return features
    
    def extract_pattern_features(self, word, i):
        """Extract pattern-based features"""
        features = {}
        
        if self.use_pattern_features:
            # Complex pattern features
            if i > 0:
                features["prev_is_vowel"] = word[i-1].lower() in 'aeiou'
                features["char_pair"] = word[i-1:i+1]
            if i < len(word) - 1:
                features["next_is_vowel"] = word[i+1].lower() in 'aeiou'
            
            # Syllable-like features
            if i > 0 and i < len(word) - 1:
                prev_char = word[i-1].lower()
                curr_char = word[i].lower()
                next_char = word[i+1].lower()
                features["syllable_pattern"] = (
                    ("V" if prev_char in 'aeiou' else "C") +
                    ("V" if curr_char in 'aeiou' else "C") +
                    ("V" if next_char in 'aeiou' else "C")
                )
        
        return features

    def extract_features(self, word_dictionary: dict):
        """Extract all features for the input data"""
        X = []  # Features for each word
        Y = []  # Labels for each word
        words = []  # Original words
        
        for word, label in word_dictionary.items():
            if len(word) != len(label):
                warnings.warn(f"Skipping word {word} due to length mismatch")
                continue
                
            word_features = []
            for i in range(len(word)):
                # Combine all features
                features = {
                    **self.extract_character_features(word, i),
                    **self.extract_pattern_features(word, i)
                }
                word_features.append(features)
            
            X.append(word_features)
            Y.append(list(label))
            words.append([char for char in word])
        
        return X, Y, words