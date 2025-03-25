import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.chrf_score import corpus_chrf
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.translate.bleu_score import SmoothingFunction
import pandas as pd
from dataset import BilingualDataset, causal_mask
from model import build_transformer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
import warnings
from tqdm import tqdm
import torchmetrics
import matplotlib.pyplot as plt
import re
# from customized_tokenizer import SyllableTokenizer
from customized_tokenizer import SyllableTokenizer
from early_stopping import EarlyStopping
import random
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
import os
import torch.nn as nn
from ray import tune
from ray import train
from datetime import datetime
import json
from sacrebleu.metrics import BLEU, CHRF, TER


class Seq2SeqTrainer:
    def __init__(self, config):
        self.config = config
        self.eval_data_cut_off = config["data_length"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(self.config['experiment_name'])
        print("Using device:", self.device)
        with torch.no_grad():  # Avoid unnecessary memory allocation when checking for GPU
            if self.device.type == "cuda":
                print(f"Device name: {torch.cuda.get_device_name(0)}")
                print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3} GB")
            elif self.device.type == "cpu":
                print("Training on CPU. Consider using GPU for faster training.")

        try:
            data = pd.read_csv(self.config["datasource"])
            if self.eval_data_cut_off > len(data):
                print(f"Warning: Requested sample size ({self.eval_data_cut_off}) exceeds available data ({len(data)})")
                print(f"Using full dataset instead.")
                self.df = data
            else:
                self.df = data.sample(n=self.eval_data_cut_off, random_state=self.config["random_seed"])
            
            print(f"Original shape: {data.shape}")
            print(f"Final shape: {self.df.shape}")

            
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{self.config['datasource']}' not found. Please check the path.")
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}")

        self.save_config(self.config)
        self.load_tokenizers()
        self.get_ds()
        self.get_model()
        
    def load_tokenizers(self):
        self.tokenizer_src = self.get_or_build_tokenizer(self.config, self.config['lang_src'])
        self.tokenizer_tgt = self.get_or_build_tokenizer(self.config, self.config['lang_tgt'])
        
    def get_or_build_tokenizer(self, config, lang):
        tokenizer_path = Path(self.config['tokenizer_file'].format(lang))
        model_folder = Path(f"{self.config['file_path']}")
        tokenizer_file_path = model_folder / "tokenizers" / tokenizer_path
        tokenize_method = config['tokenize_method'] 
        print(f"Tokenizer path: {tokenizer_file_path}")

        if tokenizer_file_path.exists():
            print(f"Loading existing tokenizer from {tokenizer_file_path}")
            if self.config['tokenize_custom'][lang]:
                tokenizer = SyllableTokenizer.from_file(str(tokenizer_file_path), mode=tokenize_method)  # Pass the mode
            else:
                # Load default Tokenizer
                tokenizer = Tokenizer.from_file(str(tokenizer_file_path))
        else:
            print("Tokenizer not found. Creating a new one...")
            tokenizer_file_path.parent.mkdir(parents=True, exist_ok=True)
            # Determine the tokenization method
            if self.config['tokenize_custom'][lang]:
                if tokenize_method == 'character':
                    print("Creating custom Character-Level Tokenizer")
                    tokenizer = SyllableTokenizer(
                        mode="character",
                        count_frequency=True,
                        max_vocab_size=self.config['vocab_size'],
                        min_freq=1
                    )
                elif tokenize_method == 'syllable':
                    print("Creating custom Syllable Tokenizer")
                    tokenizer = SyllableTokenizer(
                        mode="syllable",
                        count_frequency=True,
                        max_vocab_size=self.config['vocab_size'],
                        min_freq=1
                    )
                elif tokenize_method == 'segment':
                    print("Creating custom Segment Tokenizer")
                    tokenizer = SyllableTokenizer(
                        mode="segment",
                        count_frequency=True,
                        max_vocab_size=self.config['vocab_size'],
                        min_freq=1
                    )
                else:
                    raise ValueError(f"Unsupported tokenization method: {tokenize_method}")
                    
                # Train tokenizer on the dataset
                tokenizer.encode_batch(self.get_all_sentences(lang), add_special_tokens=False)
                tokenizer.prune_vocab()
                print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
                # Save the tokenizer
                tokenizer.save(str(tokenizer_file_path), pretty=True)
            else:
                
                print("Creating default Tokenizer")
                tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
                tokenizer.pre_tokenizer = Whitespace()
                trainer = WordLevelTrainer(
                    special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
                    min_frequency=1,
                    vocab_size=self.config['vocab_size']
                )
                tokenizer.train_from_iterator(self.get_all_sentences(lang), trainer=trainer)
                tokenizer.save(str(tokenizer_file_path))

        return tokenizer
    
    
    def get_weights_file_path(self, config, epoch: str):
        model_folder = f"{self.config['file_path']}"
        model_filename = f"{self.config['model_basename']}{epoch}.pt"
        return str(Path('.') / model_folder / model_filename)

    # Find the latest weights file in the weights folder
    def latest_weights_file_path(self, config):
        model_folder = f"{self.config['file_path']}"
        model_filename = f"{self.config['model_basename']}*"
        weights_files = list(Path(model_folder).glob(model_filename))
        if len(weights_files) == 0:
            return None
        weights_files.sort()
        return str(weights_files[-1])

    def get_all_sentences(self, lang):
        if lang not in self.df.columns:
            raise ValueError(f"Column '{lang}' not found in the dataset.")
        else:
            for sentence in self.df[lang]:
                if not isinstance(sentence, str):
                    continue  # Skip non-string entries
                yield sentence

    def get_ds(self):
        
        ds_raw = [{"translation": {self.config['lang_src']: src, self.config['lang_tgt']: tgt}} 
                  for src, tgt in zip(self.df[self.config['lang_src']], self.df[self.config['lang_tgt']])]

                
        # Build tokenizers
        tokenizer_src = self.tokenizer_src
        tokenizer_tgt = self.tokenizer_tgt       


        dataset_length = len(ds_raw)
        print("the dataset length:", dataset_length)
        train_size = int(0.8 * dataset_length)
        val_size = int(0.1 * dataset_length)
        test_size = dataset_length - train_size - val_size

        generator = torch.Generator().manual_seed(self.config['random_seed'])
        train_ds_raw, val_ds_raw, test_ds_raw = random_split(
            ds_raw, [train_size, val_size, test_size], generator=generator
        )
        
        random_seed = self.config['random_seed']
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        train_ds = BilingualDataset(train_ds_raw,
                                    tokenizer_src,
                                    tokenizer_tgt,
                                    self.config['lang_src'],
                                    self.config['lang_tgt'],
                                    self.config['seq_len'])

        val_ds = BilingualDataset(val_ds_raw,
                                  tokenizer_src,
                                  tokenizer_tgt,
                                  self.config['lang_src'],
                                  self.config['lang_tgt'],
                                  self.config['seq_len'])

        test_ds = BilingualDataset(test_ds_raw,
                                   tokenizer_src,
                                   tokenizer_tgt,
                                   self.config['lang_src'],
                                   self.config['lang_tgt'],
                                   self.config['seq_len'])


        max_len_src = max([len(tokenizer_src.encode(item['translation'][self.config['lang_src']]).ids) for item in ds_raw])
        max_len_tgt = max([len(tokenizer_tgt.encode(item['translation'][self.config['lang_tgt']]).ids) for item in ds_raw])

        print(f'Max length of source sentence: {max_len_src}')
        print(f'Max length of target sentence: {max_len_tgt}')

        self.train_dataloader = DataLoader(train_ds, batch_size=self.config['batch_size'], shuffle=True)
        self.val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)
        self.test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)

    def get_model(self):
        self.model = build_transformer(
            self.tokenizer_src.get_vocab_size(),
            self.tokenizer_tgt.get_vocab_size(),
            self.config["seq_len"],
            self.config["seq_len"],
            d_model=self.config["d_model"],
            N=self.config["num_layers"],
            h=self.config["num_heads"],
            dropout=self.config["dropout"],
            d_ff=self.config["d_ff"]
        ).to(self.device)
        self.load_pretrained_model()

    def save_config(self, config, base_path=None):
        """
        Save model configuration to a JSON file.
        
        Args:
            config (dict): Configuration dictionary to save
            base_path (str, optional): Base path to save the config. If None, uses config's file_path
            
        Returns:
            str: Path where config was saved
        """
        # Use provided base_path or get from config
        save_path = base_path or self.config.get('file_path')
        if not save_path:
            raise ValueError("No save path provided and no 'file_path' in config")
            
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Add timestamp to config
        config_to_save = self.config.copy()
        config_to_save['saved_timestamp'] = datetime.now().isoformat()
        
        # Save config
        config_path = os.path.join(save_path, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=4)
            print(f"The config file has been saved on {save_path}")
            
        return config_path


        
    def load_pretrained_model(self, optimizer=None):
        """
        Load the best model with architecture compatibility check
        Args:
            optimizer: Optional optimizer to restore state
        Returns:
            tuple: (next_epoch, global_step)
        """
        model_filename = os.path.join(self.config['file_path'], 'best_model.pt')
        
        if os.path.exists(model_filename):
            state = torch.load(model_filename)
            
            try:
                self.model.load_state_dict(state['model_state_dict'])
                if optimizer:
                    optimizer.load_state_dict(state['optimizer_state_dict'])
                print(f"Loaded best model from epoch {state['epoch']}")
                
                # Only try to print config if it exists
                if 'model_config' in state:
                    print("Model configuration:")
                    for key, value in state['model_config'].items():
                        print(f"- {key}: {value}")
                
                return state['epoch'] + 1, state['global_step']
                
            except RuntimeError as e:
                print("Error loading model weights. Architecture mismatch:")
                print(str(e))
                return 0, 0
        else:
            print("No pretrained model found, starting from scratch.")
            return 0, 0

    def save_checkpoint(self, checkpoint_data, is_best=False):
        """
        Save model checkpoint and configuration
        Args:
            checkpoint_data (dict): Data to save in checkpoint
            is_best (bool): If True, this is the best model so far
        """
        epoch = checkpoint_data['epoch']
        
        # Add configuration to checkpoint data
        checkpoint_data['model_config'] = {
            "d_model": self.config["d_model"],
            "num_layers": self.config["num_layers"],
            "num_heads": self.config["num_heads"],
            "d_ff": self.config["d_ff"],
            "dropout": self.config["dropout"],
            "label_smoothing": self.config["label_smoothing"],
            "max_grad_norm": self.config["max_grad_norm"],
            "lr": self.config["lr"]
        }
        
        # Save regular epoch checkpoint
        checkpoints_dir = os.path.join(self.config['file_path'], 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        model_filename = os.path.join(checkpoints_dir, f"model_epoch_{epoch:02d}.pt")
        torch.save(checkpoint_data, model_filename)
        
        # If this is the best model, save it separately
        if is_best:
            best_model_path = os.path.join(self.config['file_path'], 'best_model.pt')
            torch.save(checkpoint_data, best_model_path)
            self.clean_up_checkpoints(epoch)
            if self.config.get('verbose', True):
                print(f'Saved best model from epoch {epoch} with validation loss: {checkpoint_data["val_loss"]:.6f}')
    
    def clean_up_checkpoints(self, current_epoch):
        """
        Clean up old checkpoints while keeping:
        1. The last 3 checkpoints
        2. The best model (stored separately)
        """
        if current_epoch < 3:  # Too early to clean up
            return
            
        checkpoints_dir = os.path.join(self.config['file_path'], 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            return
            
        # Get all checkpoint files
        checkpoints = [f for f in os.listdir(checkpoints_dir) if f.startswith('model_epoch_')]
        checkpoints.sort()  # Sort to ensure correct ordering
        
        # Keep only the last 3 checkpoints
        checkpoints_to_keep = checkpoints[-3:]
        
        # Remove old checkpoints
        for checkpoint in checkpoints:
            if checkpoint not in checkpoints_to_keep:
                checkpoint_path = os.path.join(checkpoints_dir, checkpoint)
                os.remove(checkpoint_path)

                
    def tune_hyperparameters(self, num_samples=10, num_epochs=None):
        # Define search space
        search_space = {
            "lr": tune.loguniform(1e-5, 1e-3),
            "d_model": tune.choice([128, 256, 512, 1024 ]),
            "num_layers": tune.choice([3, 4, 6]),
            "num_heads": tune.choice([4, 8, 16]),
            "d_ff": tune.choice([512, 1024, 2048]),
            "dropout": tune.uniform(0.1, 0.3),
            "label_smoothing": tune.uniform(0.0, 0.2),
            "max_grad_norm": tune.uniform(0.5, 2.0),
        }
    
        # Initialize Ray
        ray.init(ignore_reinit_error=True)
    
        # Configure search algorithm
        search_algo = OptunaSearch(
            metric="val_loss",
            mode="min",
        )
    
        # Configure scheduler
        scheduler = ASHAScheduler(
            time_attr='epoch',
            metric='val_loss',
            mode='min',
            max_t=num_epochs or self.config["num_epochs"],
            grace_period=2,
            reduction_factor=2,
            brackets=3
        )
        
        storage_path = os.path.abspath(os.path.join(self.config['file_path'], "ray_results"))
        
        print(storage_path)

        assert os.path.exists(self.config['file_path']), f"Path {self.config['file_path']} does not exist."

    
        # Run optimization
        
        analysis = tune.run(
            tune.with_parameters(
                training_function,
                device=self.device,
                tokenizer_src=self.tokenizer_src,
                tokenizer_tgt=self.tokenizer_tgt,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                seq_len=self.config["seq_len"],
                num_epochs=num_epochs or self.config["num_epochs"]
            ),
            config=search_space,
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=search_algo,
            resources_per_trial={
                "cpu": 32,
                "gpu": 1 if torch.cuda.is_available() else 0
            },
            storage_path=storage_path,
            name="transformer_tune",
            verbose=2,
        )
    
        best_config = analysis.get_best_config(metric="val_loss", mode="min")
        print("\nBest hyperparameters found:")
        print(best_config)
    
        # Update model config
        self.config.update({
            "lr": best_config["lr"],
            "d_model": best_config["d_model"],
            "num_layers": best_config["num_layers"],
            "num_heads": best_config["num_heads"],
            "d_ff": best_config["d_ff"],
            "dropout": best_config["dropout"],
            "label_smoothing": best_config["label_smoothing"],
            "max_grad_norm": best_config["max_grad_norm"]
        })

        self.get_model()
    
        return best_config, analysis
        
    def train(self):
        print("\nTraining with the following configuration:")
        print(f"Dataset size: {self.config['data_length']}")
        print(f"Learning rate: {self.config['lr']}")
        print(f"Model architecture:")
        print(f"- d_model: {self.config['d_model']}")
        print(f"- num_layers: {self.config['num_layers']}")
        print(f"- num_heads: {self.config['num_heads']}")
        print(f"- d_ff: {self.config['d_ff']}")
        print(f"- dropout: {self.config['dropout']}")
        print(f"Training parameters:")
        print(f"- label_smoothing: {self.config['label_smoothing']}")
        print(f"- max_grad_norm: {self.config['max_grad_norm']}\n")
        
        Path(f"{self.config['datasource']}_{self.config['model_folder']}").mkdir(parents=True, exist_ok=True)
    
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], betas=(0.9, 0.98), eps=1e-9)
        initial_epoch, global_step = self.load_pretrained_model(optimizer)
        
        # Initialize early stopping
        early_stopping = EarlyStopping(
            plateau_patience=self.config['plateau_patience'],
            worsen_patience=self.config['worsen_patience'],
            min_delta=self.config['min_delta'],
            verbose=self.config['verbose']
        )
    
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer_src.token_to_id('[PAD]'), 
            label_smoothing=self.config['label_smoothing']
        ).to(self.device)
    
        train_losses = []
        val_losses = []
    
        for epoch in range(initial_epoch, self.config['num_epochs']):
            torch.cuda.empty_cache()
            self.model.train()
            batch_iterator = tqdm(self.train_dataloader, desc=f"Processing Epoch {epoch:02d}")
            epoch_loss = 0
    
            for batch in batch_iterator:
                encoder_input = batch['encoder_input'].to(self.device)
                decoder_input = batch['decoder_input'].to(self.device)
                encoder_mask = batch['encoder_mask'].to(self.device)
                decoder_mask = batch['decoder_mask'].to(self.device)
    
                encoder_output = self.model.encode(encoder_input, encoder_mask)
                decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = self.model.project(decoder_output)
    
                label = batch['label'].to(self.device)
                loss = loss_fn(
                    proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), 
                    label.view(-1)
                )
                
                epoch_loss += loss.item()
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
    
                self.writer.add_scalar('train loss per step', loss.item(), global_step)
                self.writer.flush()
    
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["max_grad_norm"]
                )
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    
                global_step += 1
    
            epoch_loss /= len(self.train_dataloader)
            train_losses.append(epoch_loss)
            self.writer.add_scalar('train loss', epoch_loss, epoch)
    
            # Validate and get validation loss
            val_loss = self.validate(epoch, self.writer)
            val_losses.append(val_loss)
    
            # Prepare checkpoint data
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'val_loss': val_loss,
                'train_loss': epoch_loss
            }
    
            # Early stopping check
            print(f"Epoch {epoch} - Validation Loss: {val_loss:.4f}")
            early_stopping(val_loss, checkpoint_data, self.save_checkpoint)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
    
            # Clean up old checkpoints
            if epoch >= 3:
                old_model_filename = self.get_weights_file_path(self.config, f"{epoch - 3:02d}")
                if os.path.exists(old_model_filename):
                    os.remove(old_model_filename)
    
        self.writer.close()
        
        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(range(initial_epoch, len(train_losses) + initial_epoch), train_losses, label='Training Loss')
        plt.plot(range(initial_epoch, len(val_losses) + initial_epoch), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()
            
    def greedy_decode(self, model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
        sos_idx = tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = tokenizer_tgt.token_to_id('[EOS]')

        encoder_output = model.encode(source, source_mask)
        decoder_input = torch.full((1, 1), sos_idx, dtype=torch.int64).to(device)

        while decoder_input.size(1) < max_len:
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            prob = model.project(out[:, -1])
            next_word = prob.argmax(dim=1).item()
            decoder_input = torch.cat([decoder_input, torch.tensor([[next_word]], dtype=torch.int64).to(device)], dim=1)

            if next_word == eos_idx:
                break

        return decoder_input.squeeze(0)
    
    def beam_search_decode(self, model, beam_size, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
        sos_idx = tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = tokenizer_tgt.token_to_id('[EOS]')

        # Precompute the encoder output and reuse it for every step
        encoder_output = model.encode(source, source_mask)
        # Initialize the decoder input with the sos token
        decoder_initial_input = torch.full((1, 1), sos_idx, dtype=torch.int64).to(device)

        # Create a candidate list
        candidates = [(decoder_initial_input, 1)]

        while True:
            if any([cand.size(1) == max_len for cand, _ in candidates]):
                break

            # Create a new list of candidates
            new_candidates = []

            for candidate, score in candidates:

                # Do not expand candidates that have reached the eos token
                if candidate[0][-1].item() == eos_idx:
                    continue

                # Build the candidate's mask
                candidate_mask = causal_mask(candidate.size(1)).type_as(source_mask).to(device)
                # calculate output
                out = model.decode(encoder_output, source_mask, candidate, candidate_mask)
                # get next token probabilities
                prob = model.project(out[:, -1])
                # get the top k candidates
                topk_prob, topk_idx = torch.topk(prob, beam_size, dim=1)
                for i in range(beam_size):
                    # for each of the top k candidates, get the token and its probability
                    token = topk_idx[0][i].unsqueeze(0).unsqueeze(0)
                    token_prob = topk_prob[0][i].item()
                    # create a new candidate by appending the token to the current candidate
                    new_candidate = torch.cat([candidate, token], dim=1)
                    # We sum the log probabilities because the probabilities are in log space
                    new_candidates.append((new_candidate, score + token_prob))

            # Sort the new candidates by their score
            candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)
            # Keep only the top k candidates
            candidates = candidates[:beam_size]

            # If all the candidates have reached the eos token, stop
            if all([cand[0][-1].item() == eos_idx for cand, _ in candidates]):
                break

        # Return the best candidate
        return candidates[0][0].squeeze()
    
    def validate(self, epoch, writer):
        smoothie = SmoothingFunction().method2
        num_examples = 3
        self.model.eval()
        bleu_scores = []
        chrf_scores = []
        
        source_texts = []
        expected = []
        predicted = []
        count=0
        console_width=100
        val_loss = 0

        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(self.device)

        
        with torch.no_grad():
            for batch in self.val_dataloader:
                count += 1
                encoder_input = batch['encoder_input'].to(self.device)
                decoder_input = batch['decoder_input'].to(self.device)
                encoder_mask = batch['encoder_mask'].to(self.device)
                decoder_mask = batch['decoder_mask'].to(self.device)
                label = batch['label'].to(self.device)

                encoder_output = self.model.encode(encoder_input, encoder_mask)
                decoder_output = self.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = self.model.project(decoder_output)

                loss = loss_fn(proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))
                val_loss += loss.item()
                
                # check that the batch size is 1
                assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

                model_out = self.greedy_decode(self.model,
                                               encoder_input,
                                               encoder_mask,
                                               self.tokenizer_src,
                                               self.tokenizer_tgt,
                                               self.config['seq_len'],
                                               self.device)
                

                source_text = batch["src_text"][0]
                target_text = batch["tgt_text"][0]
                model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())

                source_texts.append(source_text)
                expected.append(target_text)
                predicted.append(model_out_text)

                # Print the source, target and model output
                print(f"{f'SOURCE: ':>12}{source_text}")
                print(f"{f'TARGET: ':>12}{target_text}")
                print(f"{f'PREDICTED: ':>12}{model_out_text}")

                if count == num_examples:
                    print('-'*console_width)
                    break
                    
                    
        val_loss /= len(self.val_dataloader)
        writer.add_scalar('val loss', val_loss, epoch)

        references = [[t.split("-")] for t in expected]
        hypotheses = [p.split("-") for p in predicted]
        
        bleu_score = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)


        references = [t for t in expected]
        hypotheses = [p for p in predicted]
        
        chrf_score = corpus_chrf(references, hypotheses)
        
        # Log the scores
        writer.add_scalar('val_bleu_val', bleu_score, epoch)
        writer.add_scalar('val_chrf_val', chrf_score, epoch)
        writer.flush()

        
        return val_loss

    
    
    def generate_predictions(self):
        """
        Generate predictions and save them to a CSV file along with source and target texts
        
        Returns:
            tuple: (predicted texts, target texts)
        """
        self.model.eval()
        test_tgt_texts = []
        test_pred_texts = []
        
        # Create a list to store all results
        all_results = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc=f"Evaluating {len(self.test_dataloader)} Samples"):
                encoder_input = batch["encoder_input"].to(self.device)
                encoder_mask = batch["encoder_mask"].to(self.device)
                model_out = self.greedy_decode(
                    self.model,
                    encoder_input,
                    encoder_mask,
                    self.tokenizer_src,
                    self.tokenizer_tgt,
                    self.config['seq_len'],
                    self.device
                )
    
                src_text = batch["src_text"][0]
                tgt_text = batch["tgt_text"][0]
                model_out_text = self.tokenizer_tgt.decode(model_out.detach().cpu().numpy(), skip_special_tokens=True)
                
                # Store the results
                all_results.append({
                    'source': src_text,
                    'target': tgt_text,
                    'predicted': model_out_text
                })
                
                # Keep track of predictions and targets for return value
                test_tgt_texts.append(tgt_text)
                test_pred_texts.append(model_out_text)
    
                if len(test_pred_texts) % (len(self.test_dataloader) // 4) == 0:
                    print(f"Source: {src_text}")
                    print(f"Target: {tgt_text}")
                    print(f"Predicted: {model_out_text}")
    
        save_dir = os.path.join(self.config['file_path'], 'predictions')
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(save_dir, f'predictions_{timestamp}.csv')
        
        df = pd.DataFrame(all_results)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"\nPredictions saved to: {csv_path}")
        
        return test_pred_texts, test_tgt_texts


    def get_predictions_from_file(self, file_path=None):
        """
        Read predictions from a CSV file and return them as lists
        
        Args:
            file_path (str, optional): Path to the CSV file. If None, will use the latest file in predictions directory
            
        Returns:
            tuple: (predicted texts, target texts, source texts)
            
        Raises:
            FileNotFoundError: If no prediction file is found
            ValueError: If the CSV file doesn't have the expected columns
        """

        if file_path is None:
            pred_dir = os.path.join(self.config['file_path'], 'predictions')
            if not os.path.exists(pred_dir):
                raise FileNotFoundError(f"Predictions directory not found at {pred_dir}")
                
            csv_files = [f for f in os.listdir(pred_dir) if f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError("No prediction CSV files found")
                
            # Sort by filename (which includes timestamp) and get the latest
            csv_files.sort()
            file_path = os.path.join(pred_dir, csv_files[-1])
            
        print(f"Reading predictions from: {file_path}")
        
        # Read the CSV file
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")
            
        # Check if required columns exist
        required_columns = ['source', 'target', 'predicted']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV file missing required columns: {missing_columns}")
            
        # Extract predictions, targets, and sources
        predictions = df['predicted'].tolist()
        targets = df['target'].tolist()
        sources = df['source'].tolist()
        
        print(f"Successfully loaded {len(predictions)} predictions")
        
        # Print a few examples
        num_examples = min(3, len(predictions))
        print("\nExample predictions:")
        for i in range(num_examples):
            print(f"\nExample {i+1}:")
            print(f"Source: {sources[i]}")
            print(f"Target: {targets[i]}")
            print(f"Predicted: {predictions[i]}")
            
        return predictions, targets, sources

    
    def eval_bleu_segment(self, predicted, targeted):
        
        # Input validation
        if not isinstance(predicted, list) or not isinstance(targeted, list):
            raise ValueError("Both predicted and targeted should be lists.")
            
        # Filter out nan values
        valid_pairs = [(p, t) for p, t in zip(predicted, targeted) 
                       if isinstance(p, str) and isinstance(t, str) and p.strip() and t.strip()]
        
        if not valid_pairs:
            print("Warning: No valid prediction-target pairs found")
            return {
                'unigram': 0.0,
                'bigram': 0.0,
                'equal': 0.0,
                'simple': 0.0
            }
        
        filtered_pred, filtered_targ = zip(*valid_pairs)
        
        # Print statistics about filtered data
        print(f"Original pairs: {len(predicted)}")
        print(f"Valid pairs after filtering: {len(valid_pairs)}")
        print(f"Filtered out {len(predicted) - len(valid_pairs)} pairs")
        
        references = [[t.split("-")] for t in filtered_targ]
        hypotheses = [p.split("-") for p in filtered_pred]
        
        smoothie = SmoothingFunction().method2
        
        bleu_scores = {
            'unigram': corpus_bleu(references, hypotheses, 
                                  weights=(1, 0, 0, 0), 
                                  smoothing_function=smoothie),
            'bigram': corpus_bleu(references, hypotheses, 
                                 weights=(0, 1, 0, 0), 
                                 smoothing_function=smoothie),
            'equal': corpus_bleu(references, hypotheses, 
                                weights=(0.5, 0.5, 0, 0), 
                                smoothing_function=smoothie),
            'simple': corpus_bleu(references, hypotheses)
        }
        
        return bleu_scores

    def eval_sacrebleu_segment(self, predicted, targeted):
        """
        Evaluate translations using SacreBLEU and return the BLEU score only.
        
        Args:
            predicted (list): List of predicted translations
            targeted (list): List of target translations
        
        Returns:
            float: The BLEU score
        """
        
        # Input validation
        if not isinstance(predicted, list) or not isinstance(targeted, list):
            raise ValueError("Both predicted and targeted should be lists.")
        
        # Filter out invalid values
        valid_pairs = [(p, t) for p, t in zip(predicted, targeted) 
                       if isinstance(p, str) and isinstance(t, str) and p.strip() and t.strip()]
        
        if not valid_pairs:
            print("Warning: No valid prediction-target pairs found")
            return {
                'sacre_bleu': 0.0,
                'sacre_chrf': 0.0
            }
        
        filtered_pred, filtered_targ = zip(*valid_pairs)
    
        # Print statistics about filtered data
        print(f"Original pairs: {len(predicted)}")
        print(f"Valid pairs after filtering: {len(valid_pairs)}")
        print(f"Filtered out {len(predicted) - len(valid_pairs)} pairs")
    
        
        references = [[" ".join(t.split("-"))] for t in filtered_targ]
        hypotheses = [" ".join(p.split("-")) for p in filtered_pred]

        bleu = BLEU()
        chrf = CHRF()
        
        # Calculate scores
        sacre_bleu = {
            'sacre_bleu': bleu.corpus_score(hypotheses, references).score,
            'sacre_chrf': chrf.corpus_score(hypotheses, references).score
        }
        
        return sacre_bleu

    
    def save_evaluation_results(self, model_path, position_scores, bleu_scores, chrf_score, sacre_bleu):
        # Create results dictionary
        results = {
            'Position Sensitive Metrics': {
                'Precision': position_scores['precision'],
                'Recall': position_scores['recall'],
                'F1': position_scores['f1']
            },
            'BLEU Scores': {
                'Unigram': bleu_scores['unigram'],
                'Bigram': bleu_scores['bigram'],
                'Equal Weights': bleu_scores['equal'],
                'Simple scores': bleu_scores['simple']
            },
            'sacreBLEU Scores': {
                'SACRE BLEU': sacre_bleu['sacre_bleu'],
                'SACRE chrF': sacre_bleu['sacre_chrf']
            },
            'chrF Score': chrf_score
        }
        
        # Save to text file
        txt_path = f"{model_path}/evaluation_results.txt"
        try:
            with open(txt_path, 'w') as f:
                f.write(f"=== Morphological Segmentation Evaluation Results TRANSFORMER V2 - {self.config['file_path'].split('/')[-1]} ===\n\n")
                
                # Position-sensitive scores
                f.write("Position-sensitive scores:\n")
                f.write(f"Precision: {results['Position Sensitive Metrics']['Precision']:.3f}\n")
                f.write(f"Recall: {results['Position Sensitive Metrics']['Recall']:.3f}\n")
                f.write(f"F1: {results['Position Sensitive Metrics']['F1']:.3f}\n\n")
                
                # BLEU scores
                f.write("BLEU Scores:\n")
                f.write(f"Unigram only: {results['BLEU Scores']['Unigram']:.4f}\n")
                f.write(f"Bigram only: {results['BLEU Scores']['Bigram']:.4f}\n")
                f.write(f"Equal weights: {results['BLEU Scores']['Equal Weights']:.4f}\n")
                f.write(f"Simple scores: {results['BLEU Scores']['Simple scores']:.4f}\n\n")
                
                # SACRE BLEU scores
                f.write(f"Sacre BLEU Score: {results['sacreBLEU Scores']['SACRE BLEU']:.4f}\n")
                f.write(f"Sacre chrF Score: {results['sacreBLEU Scores']['SACRE chrF']:.4f}\n\n")
                
                # chrF score
                f.write(f"chrF Score: {results['chrF Score']:.4f}\n")
            
            print(f"The evaluation results were successfully saved to:")
            print(f"- Text format: {txt_path}")
            
        except Exception as e:
            print(f"Error saving evaluation results: {e}")
            
        # Log the metrics to TensorBoard if writer is available
        if hasattr(self, 'writer') and self.writer is not None:
            # Log Position-sensitive metrics
            self.writer.add_scalar('final_precision', results['Position Sensitive Metrics']['Precision'])
            self.writer.add_scalar('final_recall', results['Position Sensitive Metrics']['Recall'])
            self.writer.add_scalar('final_f1', results['Position Sensitive Metrics']['F1'])
            
            # Log BLEU scores
            self.writer.add_scalar('final_bleu_unigram', results['BLEU Scores']['Unigram'])
            self.writer.add_scalar('final_bleu_bigram', results['BLEU Scores']['Bigram'])
            self.writer.add_scalar('final_bleu_equal', results['BLEU Scores']['Equal Weights'])
            self.writer.add_scalar('final_bleu_simple', results['BLEU Scores']['Simple scores'])
            
            # Log SACRE BLEU and chrF scores
            self.writer.add_scalar('final_sacre_bleu', results['sacreBLEU Scores']['SACRE BLEU'])
            self.writer.add_scalar('final_sacre_chrF', results['sacreBLEU Scores']['SACRE chrF'])
            
            # Log chrF Score
            self.writer.add_scalar('final_chrf', results['chrF Score'])
            
            self.writer.flush()
            
        
    def eval_morph_segments_position(self, predicted, target):
    
        predicted= [word.split('-') for word in predicted]
        target = [word.split('-') for word in target]
        
        correct = 0.0
        assert len(predicted)==len(target)
        
        # Iterate through predicted and target words
        for pred, targ in zip(predicted, target):
            # Create enumerated pairs to track position
            pred_with_pos = list(enumerate(pred))
            targ_with_pos = list(enumerate(targ))
            
            # Check matches at same positions
            for pos, p_morph in pred_with_pos:
                # Look for match at same position in target
                if pos < len(targ) and p_morph == targ[pos]:
                    correct += 1
        
        predicted_length = sum([len(pred) for pred in predicted])
        target_length = sum([len(targ) for targ in target])
        
        precision = correct/predicted_length
        recall = correct/target_length
        f_score = 2 * (precision * recall)/(precision + recall) if (precision + recall) > 0 else 0
        
        # Return scores as dictionary
        return {
            'precision': precision,
            'recall': recall,
            'f1': f_score
        }

    

    def eval_chrF_segment(self, predicted, targeted):

        # Input validation
        if not isinstance(predicted, list) or not isinstance(targeted, list):
            raise ValueError("Both predicted and targeted should be lists.")

        # Filter out nan values
        valid_pairs = [(p, t) for p, t in zip(predicted, targeted) 
                       if isinstance(p, str) and isinstance(t, str) and p.strip() and t.strip()]
        
        if not valid_pairs:
            print("Warning: No valid prediction-target pairs found")
            return 0.0
        
        filtered_pred, filtered_targ = zip(*valid_pairs)
        
        references = [t for t in filtered_targ]
        hypotheses = [p for p in filtered_pred]
        
        chrf_score = corpus_chrf(references, hypotheses)
    
        return chrf_score

    def translate(self, sentence: str):
        # Ensure model is in evaluation mode
        self.model.eval()
        seq_len = self.config['seq_len']

        # Tokenize the input sentence
        source = self.tokenizer_src.encode(sentence)
        source = torch.cat([
            torch.tensor([self.tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([self.tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
            torch.tensor([self.tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(self.device)
        
        source_mask = (source != self.tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(self.device)
        
        # Use the existing greedy_decode method to generate the translation
        model_out = self.greedy_decode(
            self.model,
            source.unsqueeze(0),  # Adding batch dimension
            source_mask,
            self.tokenizer_src,
            self.tokenizer_tgt,
            seq_len,
            self.device
        )

        # Decode the output tensor into text using the target tokenizer
        translated_sentence = self.tokenizer_tgt.decode(model_out.tolist())

        return translated_sentence
    
    def evaluate_on_test(self, external_test_data):
        test_data = [{"translation": {self.config['lang_src']: row[self.config['lang_src']],
                                      self.config['lang_tgt']: row[self.config['lang_tgt']]}}
                     for _, row in external_test_data.iterrows()]

        self.model.eval()
        test_tgt_texts = []
        test_pred_texts = []

        with torch.no_grad():
            for item in tqdm(test_data, desc="Evaluating on External Test Data"):
                src_text = item['translation'][self.config['lang_src']]
                tgt_text = item['translation'][self.config['lang_tgt']]

                encoder_input = self.tokenizer_src.encode(src_text)
                encoder_input = torch.cat([
                    torch.tensor([self.tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
                    torch.tensor(encoder_input.ids, dtype=torch.int64),
                    torch.tensor([self.tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
                    torch.tensor([self.tokenizer_src.token_to_id('[PAD]')] * (self.config['seq_len'] - len(encoder_input.ids) - 2), dtype=torch.int64)
                ], dim=0).to(self.device)

                encoder_mask = (encoder_input != self.tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(self.device)

#                 model_out = self.greedy_decode(
#                     self.model,
#                     encoder_input.unsqueeze(0),  # Adding batch dimension
#                     encoder_mask,
#                     self.tokenizer_src,
#                     self.tokenizer_tgt,
#                     self.config['seq_len'],
#                     self.device
#                 )
                
                model_out =  self.beam_search_decode(
                    self.model,
                    3,
                    encoder_input.unsqueeze(0),
                    encoder_mask,
                    self.tokenizer_src,
                    self.tokenizer_tgt,
                    self.config['seq_len'],
                    self.device
                )


                model_out_text = (self.tokenizer_tgt.decode(model_out.detach().cpu().numpy())).replace("-","")

                model_out_text=model_out_text.replace("-","")
                tgt_text=tgt_text.replace("-","")
                div = len(test_data)/4
#                 print("the div: ", div)

                test_tgt_texts.append(tgt_text)
                test_pred_texts.append(model_out_text)
#                 print("len src: ", len(test_tgt_texts), "divider : ", div , "trith: ", len(test_tgt_texts)%div==0)
                
                if (len(test_tgt_texts)%div==0):
                    print(f"{f'SOURCE: ':>12}{src_text}")
                    print(f"{f'TARGET: ':>12}{tgt_text}")
                    print(f"{f'PREDICTED: ':>12}{model_out_text}")

        # Compute BLEU and CHRf scores
        bleu_score = corpus_bleu([[text.split()] for text in test_tgt_texts], [text.split() for text in test_pred_texts])
        chrf_score = corpus_chrf(test_tgt_texts, test_pred_texts)

        print(f"BLEU Score on External Test Data: {bleu_score:.4f}")
        print(f"CHRf Score on External Test Data: {chrf_score:.4f}")
        self.writer.add_scalar('val_bleu_on_test', bleu_score)
        self.writer.add_scalar('val_chrf_on_test', chrf_score)

        return bleu_score, chrf_score, test_pred_texts


    def segment_sentence(self, sentence: str) -> str:
        """
        Segment a sentence by breaking it into words, translating each word, and joining with spaces.
        
        Args:
            sentence (str): Input sentence to segment
            
        Returns:
            str: Segmented sentence with morphemes joined by hyphens
        """
        words = sentence.strip().split()
        segmented_words = []
        
        for word in words:
            segmented = self.translate(word).strip()
            # Clean up any special tokens that might appear
            segmented_words.append(segmented)
        
        return " ".join(segmented_words)

def training_function(config, device, tokenizer_src, tokenizer_tgt, train_dataloader, val_dataloader, seq_len, num_epochs):
    # Build model with config parameters
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        seq_len,
        seq_len,
        d_model=config["d_model"],
        N=config["num_layers"],
        h=config["num_heads"],
        dropout=config["dropout"],
        d_ff=config["d_ff"]
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # Initialize loss function
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id('[PAD]'),
        label_smoothing=config["label_smoothing"]
    ).to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            # Calculate loss
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1)
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["max_grad_norm"]
            )
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                encoder_input = batch['encoder_input'].to(device)
                decoder_input = batch['decoder_input'].to(device)
                encoder_mask = batch['encoder_mask'].to(device)
                decoder_mask = batch['decoder_mask'].to(device)
                label = batch['label'].to(device)

                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.project(decoder_output)

                loss = loss_fn(
                    proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                    label.view(-1)
                )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)

        # Report metrics using train.report
        ray.train.report(
            {"train_loss": avg_train_loss,
             "val_loss": avg_val_loss,
             "epoch": epoch}
        )


