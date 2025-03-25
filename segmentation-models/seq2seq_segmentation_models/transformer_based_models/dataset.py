import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # Special tokens
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def truncate_and_pad(self, tokens, max_length, add_sos=True, add_eos=True):
        """Helper function to handle truncation and padding"""
        available_length = max_length - (1 if add_sos else 0) - (1 if add_eos else 0)
        
        if len(tokens) > available_length:
            tokens = tokens[:available_length]
        
        pad_length = max_length - len(tokens) - (1 if add_sos else 0) - (1 if add_eos else 0)
        
        sequence_parts = []
        if add_sos:
            sequence_parts.append(self.sos_token)
        
        sequence_parts.append(torch.tensor(tokens, dtype=torch.int64))
        
        if add_eos:
            sequence_parts.append(self.eos_token)
        
        if pad_length > 0:
            sequence_parts.append(torch.tensor([self.pad_token] * pad_length, dtype=torch.int64))
        
        return torch.cat(sequence_parts, dim=0)

    def __getitem__(self, idx):
        # Get the source-target pair
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform text to tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Create tensors with truncation/padding as needed
        encoder_input = self.truncate_and_pad(
            enc_input_tokens, 
            self.seq_len, 
            add_sos=True, 
            add_eos=True
        )
        
        decoder_input = self.truncate_and_pad(
            dec_input_tokens, 
            self.seq_len, 
            add_sos=True, 
            add_eos=False
        )
        
        label = self.truncate_and_pad(
            dec_input_tokens, 
            self.seq_len, 
            add_sos=False, 
            add_eos=True
        )

        # Create attention masks
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        decoder_mask = (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0))

        # Verify tensor sizes
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": encoder_mask,    # (1, 1, seq_len)
            "decoder_mask": decoder_mask,    # (1, seq_len, seq_len)
            "label": label,                  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size):
    """Create causal mask for decoder attention"""
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0