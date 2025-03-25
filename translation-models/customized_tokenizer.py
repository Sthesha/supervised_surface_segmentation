import json
import os
import re
from typing import List, Optional, Union
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.implementations import BaseTokenizer as TokenizersBaseTokenizer


class SyllableTokenizerModel:
    def __init__(self, mode: str = "character"):
        """
        Initialize the tokenizer model.
        mode: str - Either "syllable", "character", or "segment".
        """
        if mode not in {"syllable", "character", "segment"}:
            raise ValueError("Invalid mode. Choose 'syllable', 'character', or 'segment'.")
        self.mode = mode
        self.special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.next_id = len(self.special_tokens)
        self.word_freq = Counter()  # Track word frequencies
        self.finalize_vocab = False  # Indicates whether the vocabulary is finalized

    def finalize(self):
        """Set the vocabulary as finalized, preventing further updates."""
        self.finalize_vocab = True

    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text based on the mode."""
        if self.mode == "character":
            tokens = list(text)  # Split into individual characters
        elif self.mode == "syllable":
            tokens = self.tokenize_syllables(text)
        elif self.mode == "segment":
            tokens = self.tokenize_segment(text)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
        
        # Update frequency counter
        self.word_freq.update(tokens)  # Automatically update word frequency
        return tokens


    def tokenize_syllables(self, text: str) -> List[str]:
        tokens = []
        current_token = ""
        for char in text:
            if char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
            else:
                current_token += char

        if current_token:
            tokens.append(current_token)

        syllables = [
            self.syllabify(token) if not token.isspace() else [token] for token in tokens
        ]
        return [syll for sublist in syllables for syll in sublist]

    def tokenize_segment(self, text: str) -> List[str]:
        """Tokenize the input text based on segments (e.g., split on spaces or '-')."""
        tokens = re.split(r'(\s+|-)', text)  # Split on spaces or '-' and keep them as tokens
        tokens = [token for token in tokens if token]  # Remove empty tokens
        return tokens

    def syllabify(self, word: str) -> List[str]:
        """Split a word into syllables."""
        pattern = r'[^aeiou\s]*[aeiou]+(?:[^aeiou\s]*$)?|\s+'
        return re.findall(pattern, word, flags=re.IGNORECASE)

    def add_tokens(self, tokens: List[str]):
        """Add tokens to the vocabulary."""
        if self.finalize_vocab:
            return  # Prevent adding new tokens

        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.next_id
                self.reverse_vocab[self.next_id] = token
                self.next_id += 1

    def token_to_id(self, token: str) -> Optional[int]:
        """Get the ID for a token, or return the ID for [UNK] if not found."""
        return self.vocab.get(token, self.vocab["[UNK]"])

    def id_to_token(self, id: int) -> Optional[str]:
        """Get the token corresponding to an ID."""
        return self.reverse_vocab.get(id, "[UNK]")

    def prune_vocab(self, max_vocab_size: Optional[int] = None, min_freq: Optional[int] = None):
        """Prune the vocabulary based on max_vocab_size and min_freq."""
        print(f"\nVocabulary Stats before pruning:")
        print(f"Total unique tokens: {len(self.word_freq)}")
        if max_vocab_size is None and min_freq is None:
            return

        if min_freq is None:
            min_freq = 1

        # Get tokens that meet frequency threshold
        most_common = [token for token, freq in self.word_freq.items() if freq >= min_freq]
        print(f"Tokens after frequency filtering (min_freq={min_freq}): {len(most_common)}")

        # Apply size limit if specified
        if max_vocab_size is not None:
            most_common = sorted(
                most_common, 
                key=lambda token: self.word_freq[token], 
                reverse=True
            )[:max_vocab_size - len(self.special_tokens)]  # Reserve space for special tokens
            print(f"Tokens after size limiting: {len(most_common)}")

        # Build new vocabulary with special tokens first
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        # Add remaining tokens
        for token in most_common:
            if token not in self.vocab:  # Avoid duplicating special tokens
                self.vocab[token] = len(self.vocab)

        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.next_id = len(self.vocab)
        print(f"Final vocabulary size (including special tokens): {len(self.vocab)}")

    def save_vocab(self, path: str, pretty: bool = True):
        """Save the vocabulary to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4 if pretty else None)

    def load_vocab(self, path: str):
        """Load the vocabulary from a file."""
        with open(path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.next_id = max(self.vocab.values()) + 1
        self.finalize_vocab = True

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

    def __init__(self, mode: str = "syllable", count_frequency: bool = False,
                 max_vocab_size: Optional[int] = None, min_freq: Optional[int] = None):
        """
        Initialize the tokenizer.
        mode: str - Either "syllable", "character", or "segment".
        """
        model = SyllableTokenizerModel(mode=mode)
        self._model = model  # Explicitly set the model
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
        self._model.add_tokens(tokens)  # Add tokens to vocab if not finalized
        ids = [self._model.token_to_id(token) for token in tokens]
        return SyllableTokenizer.Encoding(ids)

    def decode(self, ids: List[int], skip_special_tokens: Optional[bool] = True) -> str:
        tokens = [self._model.id_to_token(id) for id in ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in {"[SOS]", "[EOS]"}]
        return "".join(tokens)

    def encode_batch(self, inputs: List[Union[str, List[str]]], add_special_tokens: bool = True) -> List['SyllableTokenizer.Encoding']:
        batch = []
        for input_ in inputs:
            if isinstance(input_, str):
                tokens = self._model.tokenize(input_)
            else:
                tokens = input_
            if add_special_tokens:
                tokens = ["[SOS]"] + tokens + ["[EOS]"]
            self._model.add_tokens(tokens)  # Add tokens to vocab if not finalized
            ids = [self._model.token_to_id(token) for token in tokens]
            batch.append(SyllableTokenizer.Encoding(ids))
        return batch

    def decode_batch(self, sequences: List[List[int]], skip_special_tokens: Optional[bool] = True) -> List[str]:
        batch = []
        for sequence in sequences:
            tokens = [self._model.id_to_token(id) for id in sequence]
            if skip_special_tokens:
                tokens = [token for token in tokens if token not in {"[SOS]", "[EOS]"}]
            decoded_text = "".join(tokens)
            batch.append(decoded_text)
        return batch

    def save(self, path: str, pretty: bool = True):
        self._model.save_vocab(path, pretty)

    def load(self, path: str):
        self._model.load_vocab(path)

    def get_vocab_size(self) -> int:
        return self._model.vocab_size()

    def get_vocab(self) -> dict:
        return self._model.get_vocab()

    def token_to_id(self, token: str) -> Optional[int]:
        return self._model.token_to_id(token)

    def prune_vocab(self):
        self._model.prune_vocab(self.max_vocab_size, self.min_freq)

    def finalize_vocab(self):
        """Finalize the vocabulary to prevent new tokens from being added."""
        self._model.finalize()

    @classmethod
    def from_file(cls, path: str, mode: str = "syllable"):
        model = SyllableTokenizerModel(mode=mode)
        model.load_vocab(path)
        tokenizer = cls(mode=mode)  # Pass mode here
        tokenizer._model = model
        return tokenizer
