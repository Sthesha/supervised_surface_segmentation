import json
import os
import re
from typing import Dict, List, Optional, Union
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.implementations import BaseTokenizer as TokenizersBaseTokenizer
from tokenizers.models import Model

class SyllableTokenizerModel(Model):
    def __init__(self):
        super().__init__()
        self.special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.next_id = len(self.special_tokens)
        self.word_freq = Counter()  # Track word frequencies
        self.tokenize_method = "syllable"  # Default tokenize method

    def set_tokenize_method(self, tokenize_method: str):
        self.tokenize_method = tokenize_method

    def tokenize(self, text: str) -> List[str]:
        if self.tokenize_method == "segment":
            return self.tokenize_segment(text)
        else:
            return self.tokenize_syllable(text)

    def tokenize_syllable(self, text: str) -> List[str]:
        tokens = []
        current_token = ""

        for char in text:
            if char.isspace():  # If the character is a space
                if current_token:  # If there's a current non-empty token, add it to tokens
                    tokens.append(current_token)
                    current_token = ""  # Reset current_token
                tokens.append(char)  # Add the space as a separate token
            else:
                current_token += char  # Append character to current_token

        # If there's any remaining non-empty token, add it to tokens
        if current_token:
            tokens.append(current_token)

        syllables = [self.syllabify(token) if not token.isspace() else [token] for token in tokens]
        flat_syllables = [syll for sublist in syllables for syll in sublist]
        self.word_freq.update(flat_syllables)  # Update word frequencies
        return flat_syllables


    def tokenize_segment(self, text: str) -> List[str]:
        tokens = re.split(r'(\s+|-)', text)  # Split on spaces or '-' and keep them as tokens
        tokens = [token for token in tokens if token]  # Remove empty tokens
        self.word_freq.update(tokens)  # Update word frequencies
        return tokens

    def add_tokens(self, tokens: List[str]):
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.next_id
                self.reverse_vocab[self.next_id] = token
                self.next_id += 1

    def syllabify(self, word: str) -> List[str]:
        # Updated syllabify function to split at the end of vowels and also handle spaces
        pattern = r'[^aeiou\s]*[aeiou]+(?:[^aeiou\s]*$)?|\s+'
        return re.findall(pattern, word, flags=re.IGNORECASE)

    def token_to_id(self, token: str) -> Optional[int]:
        return self.vocab.get(token, self.vocab["[UNK]"])

    def id_to_token(self, id: int) -> Optional[str]:
        return self.reverse_vocab.get(id, "[UNK]")

    def update_word_freq(self, tokens: List[str]):
        self.word_freq.update(tokens)

    def prune_vocab(self, max_vocab_size: Optional[int] = None, min_freq: Optional[int] = None):
        if max_vocab_size is None and min_freq is None:
            return

        if min_freq is None:
            min_freq = 1

        most_common = [token for token, freq in self.word_freq.items() if freq >= min_freq]
        
        if max_vocab_size is not None:
            most_common = sorted(most_common, key=lambda token: self.word_freq[token], reverse=True)[:max_vocab_size]

        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens + most_common)}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.next_id = len(self.vocab)

    def save_vocab(self, path: str, pretty: bool = True):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4 if pretty else None)

    def load_vocab(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.next_id = max(self.vocab.values()) + 1

    def save(self, folder: str, name: Optional[str] = None):
        vocab_path = os.path.join(folder, 'vocab.json')
        self.save_vocab(vocab_path)

    def train(self, files, trainer):
        pass

    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self, with_added_tokens: bool = True) -> dict:
        return self.vocab

    def __call__(self, text: str) -> List[str]:
        return self.tokenize(text)


class SyllableTokenizer(TokenizersBaseTokenizer):
    class Encoding:
        def __init__(self, ids: List[int]):
            self.ids = ids

    def __init__(self, count_frequency: bool = False,
                 max_vocab_size: Optional[int] = None,
                 min_freq: Optional[int] = None,
                 tokenize_method: str = "syllable"):
        model = SyllableTokenizerModel()
        model.set_tokenize_method(tokenize_method)
        tokenizer = Tokenizer(model)
        super().__init__(tokenizer)
        self._model = model 
        self.count_frequency = count_frequency
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq

    def encode(self, sequence: Union[str, List[str]], add_special_tokens: bool = False) -> 'SyllableTokenizer.Encoding':
        if isinstance(sequence, str):
            tokens = self._model.tokenize(sequence)
        else:
            tokens = sequence
        if add_special_tokens:
            tokens = ["[SOS]"] + tokens + ["[EOS]"]
        ids = [self._model.token_to_id(token) for token in tokens]
        return SyllableTokenizer.Encoding(ids)

    def decode(self, ids: List[int], skip_special_tokens: Optional[bool] = True) -> str:
        tokens = [self._model.id_to_token(id) for id in ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in tuple(self._model.special_tokens)]  
        decoded_text = "".join(tokens)
        return decoded_text

    def encode_batch(self, inputs: List[Union[str, List[str]]], add_special_tokens: bool = False) -> List['SyllableTokenizer.Encoding']:
        batch = []
        for input_ in inputs:
            if isinstance(input_, str):
                tokens = self._model.tokenize(input_)
            else:
                tokens = input_
            if add_special_tokens:
                tokens = ["[SOS]"] + tokens + ["[EOS]"]
            self._model.add_tokens(tokens)
            ids = [self._model.token_to_id(token) for token in tokens]
            batch.append(SyllableTokenizer.Encoding(ids))
        return batch

    def decode_batch(self, sequences: List[List[int]], skip_special_tokens: Optional[bool] = True) -> List[str]:
        batch = []
        for sequence in sequences:
            tokens = [self._model.id_to_token(id) for id in sequence]
            if skip_special_tokens:
                tokens = [token for token in tokens if token not in tuple(self._model.special_tokens)]  
            decoded_text = "".join(tokens)
            batch.append(decoded_text)
        return batch

    def token_to_id(self, token: str) -> Optional[int]:
        return self._model.token_to_id(token)
    
    def get_vocab_size(self) -> int:
        return self._model.vocab_size()
    
    def get_vocab(self, with_added_tokens: bool = True) -> dict:
        return self._model.get_vocab(with_added_tokens=with_added_tokens)

    def save(self, path: str, pretty: bool = True):
        self._model.save_vocab(path, pretty)

    def load(self, path: str):
        self._model.load_vocab(path)

    def prune_vocab(self):
        self._model.prune_vocab(self.max_vocab_size, self.min_freq)
    
    def get_special_tokens(self):
        return self._model.special_tokens

        
    @classmethod
    def from_file(cls, path: str) -> 'SyllableTokenizer':
        model = SyllableTokenizerModel()
        model.load_vocab(path)
        tokenizer = cls()
        tokenizer._model = model
        return tokenizer
    