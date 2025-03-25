import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.chrf_score import corpus_chrf
from nltk.metrics import precision, recall, f_measure
from sacrebleu.metrics import BLEU, CHRF, TER
from typing import Dict, List, Tuple, Any
import json
import os
from pathlib import Path
from tqdm import tqdm as tqdm_bar
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

class LSTMEvaluator:
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.dimensions = None
        self.lstm_model = None
        self.encoder_model = None
        self.decoder_model = None
        self.test_data = None
        self.load_model_and_config()

    def load_model_and_config(self):
        try:
            # Load model config
            dimension_path = os.path.join(self.model_path, self.model_name, 'model_config.json')
            with open(dimension_path) as f:
                self.dimensions = json.load(f)

            # Load test data and model
            self.test_data = pd.read_csv(os.path.join(self.model_path, self.model_name, "test_data.csv"))
            self.lstm_model = load_model(os.path.join(self.model_path, self.model_name, 'lstm_model.h5'))
            
            self._setup_inference_models()
            
        except Exception as e:
            print(f"Error loading model and configuration: {str(e)}")
            raise

    def _setup_inference_models(self):
        """Setup encoder and decoder models for inference"""
        # Create reverse character mappings
        self.idx_to_char = self.dimensions['idx_to_char']
        self.special_tokens = self.dimensions['special_tokens']
        
        # Setup encoder
        encoder_inputs = self.lstm_model.input[0]
        _, state_h, state_c = self.lstm_model.layers[2].output
        self.encoder_model = Model(encoder_inputs, [state_h, state_c])

        # Setup decoder
        decoder_state_input_h = Input(shape=(self.dimensions['latent_dim'],), name="decoder_state_input_h")
        decoder_state_input_c = Input(shape=(self.dimensions['latent_dim'],), name="decoder_state_input_c")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        decoder_inputs = self.lstm_model.input[1]
        decoder_lstm = self.lstm_model.layers[3]
        decoder_dense = self.lstm_model.layers[4]

        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

    def decode_sequence(self, input_seq: np.ndarray) -> str:
        """Decode a single input sequence"""
        # Initial encoder states
        states_value = self.encoder_model.predict(input_seq, verbose=0)

        # Initialize target sequence with SOS token
        target_seq = np.zeros((1, 1, len(self.dimensions['char_to_idx'])))
        target_seq[0, 0, self.dimensions['char_to_idx'][self.special_tokens['SOS']]] = 1.

        decoded_tokens = []
        max_length = 100  # Safety limit

        while True:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value, verbose=0
            )
            
            # Sample token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.idx_to_char[str(sampled_token_index)]
            
            # Break conditions
            if sampled_char == self.special_tokens['EOS'] or len(decoded_tokens) > max_length:
                break
                
            if sampled_char != self.special_tokens['PAD']:
                if sampled_char == self.special_tokens['SEP']:
                    decoded_tokens.append('-')
                else:
                    decoded_tokens.append(sampled_char)

            # Update target sequence
            target_seq = np.zeros((1, 1, len(self.dimensions['char_to_idx'])))
            target_seq[0, 0, sampled_token_index] = 1.
            states_value = [h, c]

        return ''.join(decoded_tokens)

    def test_input(self, word: str) -> str:
        """Test a single input word"""
        # Prepare input sequence
        input_chars = [self.special_tokens['SOS']] + list(word) + [self.special_tokens['EOS']]
        max_length = max(len(text) for text in self.test_data['tokens']) + 2
        
        input_data = np.zeros((1, max_length, len(self.dimensions['char_to_idx'])))
        
        for t, char in enumerate(input_chars):
            input_data[0, t, self.dimensions['char_to_idx'][char]] = 1
        
        # Pad remaining positions
        for t in range(len(input_chars), max_length):
            input_data[0, t, self.dimensions['char_to_idx'][self.special_tokens['PAD']]] = 1
            
        return self.decode_sequence(input_data)

    def get_predictions(self) -> List[Tuple[str, str]]:
        """Get predictions for all test data using parallel processing while maintaining order"""
        print("\nGenerating predictions...")

        def process_row(idx, row_data):
            """Helper function to process a single row"""
            try:
                predicted = self.test_input(row_data['tokens'])
                return idx, (row_data[self.model_name], predicted)
            except Exception as e:
                print(f"\nError processing row {idx}: {str(e)}")
                return idx, None

        with ThreadPoolExecutor() as executor:
            # Submit all tasks and map futures to indices
            future_to_idx = {
                executor.submit(process_row, idx, row): idx
                for idx, row in self.test_data.iterrows()
            }

            # Create progress bar for completed tasks
            pbar = tqdm_bar(total=len(future_to_idx), desc="Evaluating", ncols=80)

            # Collect results while maintaining order
            ordered_results = [None] * len(self.test_data)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    if result and result[1] is not None:  # Ensure result is valid
                        ordered_results[idx] = result[1]
                except Exception as e:
                    print(f"Error in future {idx}: {str(e)}")
                finally:
                    pbar.update(1)

            pbar.close()

        # Remove any None results from failed predictions
        results = [r for r in ordered_results if r is not None]
        return results

    def _evaluate_position_sensitive(self, predicted, target):
        # Input validation 
        if not isinstance(predicted, list) or not isinstance(target, list):
            raise ValueError("Both predicted and target should be lists.")
        
        # Filter out invalid values
        valid_pairs = [(p, t) for p, t in zip(predicted, target) 
                       if isinstance(p, str) and isinstance(t, str) and p.strip() and t.strip()]
        
        if not valid_pairs:
            print("Warning: No valid prediction-target pairs found")
            return {
                'precision': 0.0,
                'recall': 0.0, 
                'f1': 0.0,
            }

        filtered_pred, filtered_targ = zip(*valid_pairs)
        
        print(f"Original pairs: {len(predicted)}")
        print(f"Valid pairs after filtering: {len(valid_pairs)}")
        print(f"Filtered out {len(predicted) - len(valid_pairs)} pairs")
        
        predicted_morphs = [p.split('-') for p in filtered_pred]
        target_morphs = [t.split('-') for t in filtered_targ]
        
        correct = sum(1 for pred, targ in zip(predicted_morphs, target_morphs)
                     for pos, p_morph in enumerate(pred)
                     if pos < len(targ) and p_morph == targ[pos])
        
        predicted_length = sum(len(pred) for pred in predicted_morphs)
        target_length = sum(len(targ) for targ in target_morphs)
        
        precision = correct/predicted_length if predicted_length > 0 else 0
        recall = correct/target_length if target_length > 0 else 0
        f1 = 2 * (precision * recall)/(precision + recall) if (precision + recall) > 0 else 0

        return {'precision': precision, 'recall': recall, 'f1': f1}

    def _evaluate_bleu(self, predicted, targeted):
        
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


    def _evaluate_chrf(self, predicted, targeted):
        """Calculate chrF score"""
        
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
        
        # Print statistics about filtered data
        print(f"Original pairs: {len(predicted)}")
        print(f"Valid pairs after filtering: {len(valid_pairs)}")
        print(f"Filtered out {len(predicted) - len(valid_pairs)} pairs")
        
        references = [t for t in filtered_targ]
        hypotheses = [p for p in filtered_pred]
        
        chrf_score = corpus_chrf(references, hypotheses)
    
        return chrf_score

    def _eval_sacrebleu_segment(self, predicted, targeted):
        
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
    
        bleu = BLEU()
        chrf = CHRF()
        
        references = [[" ".join(t.split("-"))] for t in filtered_targ]
        hypotheses = [" ".join(p.split("-")) for p in filtered_pred]
        
        return {
            'SACRE BLEU': bleu.corpus_score(hypotheses, references).score,
            'SACRE chrF': chrf.corpus_score(hypotheses, references).score
        }
    

    def evaluate(self) -> Dict[str, Any]:
        """Run full evaluation suite"""
        print(f"\nEvaluating model: {self.model_name}")
        results = self.get_predictions()
        
        predicted = [item[1] for item in results]
        target = [item[0] for item in results]          
                     
        # Calculate all metrics
        evaluation_results = {
            'position_scores': self._evaluate_position_sensitive(predicted, target),
            'bleu_scores': self._evaluate_bleu(predicted, target),
            'sacreBLEU_scores': self._eval_sacrebleu_segment(predicted, target),
            'chrf_score': self._evaluate_chrf(predicted, target)
        }
        
        return evaluation_results, predicted, target

    def save_results(self, results, predicted, targeted):
        
        """Save evaluation results and predictions"""
        # Save evaluation metrics
        output_path = os.path.join(self.model_path, self.model_name, 'evaluation_results.txt')
        
        with open(output_path, 'w') as f:
            f.write("=== Morphological Segmentation Evaluation Results ===\n\n")
            
            # Position-sensitive scores
            f.write("Position-sensitive scores:\n")
            f.write(f"Precision: {results['position_scores']['precision']:.3f}\n")
            f.write(f"Recall: {results['position_scores']['recall']:.3f}\n")
            f.write(f"F1: {results['position_scores']['f1']:.3f}\n\n")
            
            # BLEU scores
            f.write("BLEU Scores:\n")
            f.write(f"Unigram only: {results['bleu_scores']['unigram']:.4f}\n")
            f.write(f"Bigram only: {results['bleu_scores']['bigram']:.4f}\n")
            f.write(f"Equal weights: {results['bleu_scores']['equal']:.4f}\n\n")
            
            # SACREBLEU scores
            f.write("SACREBLEU Scores:\n")
            f.write(f"Sacre BLEU: {results['sacreBLEU_scores']['SACRE BLEU']:.4f}\n")
            f.write(f"Sacre chrF: {results['sacreBLEU_scores']['SACRE chrF']:.4f}\n\n")
            
            # chrF score
            f.write(f"chrF Score: {results['chrf_score']:.4f}\n")

        print(f"\nSaved evaluation metrics to {output_path}")

        # Save predictions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_dir = os.path.join(self.model_path, self.model_name, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        
        predictions_path = os.path.join(predictions_dir, f'predictions_{timestamp}.csv')
        
        
        # Create DataFrame with predictions and targets
        df = pd.DataFrame({
            'targeted': targeted,
            'predicted': predicted
        })
        
        # Save to CSV with UTF-8 encoding
        df.to_csv(predictions_path, index=False, encoding='utf-8')
        print(f"Saved predictions and targets to {predictions_path}")
        
        
    def get_predictions_from_file(self, file_path=None):
        """
        Read predictions from a CSV file and return them as lists

        Args:
            file_path (str, optional): Path to the CSV file. If None, will use the latest file in predictions directory

        Returns:
            tuple: (evaluation_results, predicted texts, target texts)
        """
        if file_path is None:
            pred_dir = os.path.join(self.model_path, self.model_name, 'predictions')
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
        required_columns = ['targeted', 'predicted']  # Changed from 'target' to 'targeted'

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV file missing required columns: {missing_columns}")

        # Extract predictions and targets
        predicted = df['predicted'].tolist()
        targeted = df['targeted'].tolist()  # Changed from 'target' to 'targeted'

        # Calculate all metrics
        evaluation_results = {
            'position_scores': self._evaluate_position_sensitive(predicted, targeted),
            'bleu_scores': self._evaluate_bleu(predicted, targeted),
            'sacreBLEU_scores': self._eval_sacrebleu_segment(predicted, targeted),
            'chrf_score': self._evaluate_chrf(predicted, targeted)
        }

        print(f"Successfully loaded {len(predicted)} predictions")

        # Print a few examples
        num_examples = min(3, len(predicted))
        print("\nExample predictions:")
        for i in range(num_examples):
            print(f"\nExample {i+1}:")
            print(f"Target: {targeted[i]}")
            print(f"Predicted: {predicted[i]}")

        return evaluation_results, predicted, targeted

    def format_time(self, seconds):
        """Format time in human-readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {int(seconds)}s"
        