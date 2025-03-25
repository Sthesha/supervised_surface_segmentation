import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split

class SegmentationDataProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.special_tokens = {
            'PAD': '[PAD]',
            'SOS': '[SOS]',
            'EOS': '[EOS]',
            'SEP': '[SEP]'  # For morpheme boundaries
        }
        self.input_texts_train: List[str] = []
        self.target_texts_train: List[str] = []
        self.input_chars: set = set()
        self.target_chars: set = set()
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        
    def load_data(self, data_path: str, data_length: Optional[int] = None) -> pd.DataFrame:
        """
        Load and optionally cut off data to specified length

        Args:
            data_path: Path to the data file
            data_length: Number of samples to use (if None, use full dataset)
        """
        try:
            data = pd.read_csv(data_path)
            print(f"Original dataset shape: {data.shape}")

            if data_length is not None:
                if data_length > len(data):
                    print(f"Warning: Requested sample size ({data_length}) exceeds available data ({len(data)})")
                    print("Using full dataset instead.")
                    return data
                else:
                    sampled_data = data.sample(n=data_length, random_state=self.config["random_seed"])
                    print(f"Cut-off dataset shape: {sampled_data.shape}")
                    return sampled_data

            return data

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}")
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        train_data, temp = train_test_split(
            data, 
            test_size=(1 - self.config["train_ratio"]), 
            random_state=self.config["random_seed"]
        )
        
        validation_data, test_data = train_test_split(
            temp,
            test_size=(self.config["test_ratio"] / (self.config["test_ratio"] + self.config["validation_ratio"])),
            random_state=self.config["random_seed"]
        )
        
        return train_data, validation_data, test_data
    
    def prepare_tokenization(self, train_data: pd.DataFrame):
        """Prepare character-level tokenization with special tokens"""
        self.input_texts_train = train_data['tokens'].tolist()
        self.target_texts_train = train_data[self.config['model_name']].tolist()
        
        # Build character vocabularies
        chars = set()
        for text in self.input_texts_train:
            chars.update(set(text))
        for text in self.target_texts_train:
            chars.update(set(text.replace('-', '')))
        
        # Add special tokens and create mappings
        special_tokens_list = list(self.special_tokens.values())
        all_chars = special_tokens_list + sorted(list(chars))
        self.char_to_idx = {char: idx for idx, char in enumerate(all_chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
    def encode_sequences(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode sequences with special tokens and character-level tokenization"""
        max_input_len = max(len(text) for text in self.input_texts_train) + 2  # +2 for SOS/EOS
        max_target_len = max(len(text) for text in self.target_texts_train) + 2
        vocab_size = len(self.char_to_idx)
        
        encoder_input_data = np.zeros((len(self.input_texts_train), max_input_len, vocab_size))
        decoder_input_data = np.zeros((len(self.input_texts_train), max_target_len, vocab_size))
        decoder_target_data = np.zeros((len(self.input_texts_train), max_target_len, vocab_size))
        
        for i, (input_text, target_text) in enumerate(zip(self.input_texts_train, self.target_texts_train)):
            # Encode input sequence
            input_chars = [self.special_tokens['SOS']] + list(input_text) + [self.special_tokens['EOS']]
            for t, char in enumerate(input_chars):
                encoder_input_data[i, t, self.char_to_idx[char]] = 1
            
            # Pad remaining positions
            for t in range(len(input_chars), max_input_len):
                encoder_input_data[i, t, self.char_to_idx[self.special_tokens['PAD']]] = 1
            
            # Encode target sequence
            target_chars = [self.special_tokens['SOS']]
            for segment in target_text.split('-'):
                target_chars.extend(list(segment) + [self.special_tokens['SEP']])
            target_chars[-1] = self.special_tokens['EOS']  # Replace last SEP with EOS
            
            # Decoder input (shifted right)
            for t, char in enumerate(target_chars):
                decoder_input_data[i, t, self.char_to_idx[char]] = 1
            
            # Decoder target (shifted left)
            for t, char in enumerate(target_chars[1:]):
                decoder_target_data[i, t, self.char_to_idx[char]] = 1
            
            # Pad remaining positions
            for t in range(len(target_chars), max_target_len):
                decoder_input_data[i, t, self.char_to_idx[self.special_tokens['PAD']]] = 1
                decoder_target_data[i, t, self.char_to_idx[self.special_tokens['PAD']]] = 1
                
        return encoder_input_data, decoder_input_data, decoder_target_data