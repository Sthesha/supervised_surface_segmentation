# evaluator.py
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.chrf_score import corpus_chrf, sentence_chrf
from sacrebleu.metrics import BLEU, CHRF, TER
import os
from datetime import datetime
import pandas as pd


class Evaluator:
    """Handles evaluation of morphological segmentation models"""
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
    
    def evaluate(self, model, test_data, test_segments):
        """Run full evaluation suite"""
        X_test, Y_test, test_tokens = self.feature_extractor.extract_features(test_data)
        Y_pred = model.predict(X_test)
        
        # Convert predictions to segmented format
        predicted, targeted = self._convert_predictions(Y_pred, test_tokens, test_segments)
        
        # Calculate all metrics
        results = {
            'position': self._evaluate_position_sensitive(predicted, targeted),
            'bleu_scores': self._evaluate_bleu(predicted, targeted),
            'sacre_bleu_scores': self._eval_sacrebleu_segment(predicted, targeted),
            'chrf': self._evaluate_chrf(predicted, targeted)
        }
        
        return results, predicted, targeted
    
    def _convert_predictions(self, Y_pred, test_tokens, test_segments):
        """Convert BMES predictions to segmented format"""
        predictions = []
        for token, pred_labels in zip(test_tokens, Y_pred):
            segmented = []
            for char, label in zip(token, pred_labels):
                segmented.append(char)
                if label in ['E', 'S']:
                    segmented.append('-')
            pred_segmented = ''.join(segmented).rstrip('-')
            predictions.append(pred_segmented)
        
        return predictions, list(test_segments.values())
    
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

    
    def save_results(self, results, model_path, predicted, targeted):
        """Save evaluation results and outputs to files"""
        os.makedirs(model_path, exist_ok=True)
        
        # Save evaluation metrics
        with open(f'{model_path}/evaluation_results.txt', 'w') as f:
            f.write("=== Morphological Segmentation Evaluation Results ===\n\n")
            
            # Position-sensitive scores
            f.write("Position-sensitive scores:\n")
            f.write(f"Precision: {results['position']['precision']:.3f}\n")
            f.write(f"Recall: {results['position']['recall']:.3f}\n")
            f.write(f"F1: {results['position']['f1']:.3f}\n\n")
            
            # BLEU scores
            f.write("BLEU Scores:\n")
            f.write(f"Unigram only: {results['bleu_scores']['unigram']:.4f}\n")
            f.write(f"Bigram only: {results['bleu_scores']['bigram']:.4f}\n")
            f.write(f"Equal weights: {results['bleu_scores']['equal']:.4f}\n\n")


            # SACRE BLEU scores
            f.write(f"Sacre BLEU: {results['sacre_bleu_scores']['sacre_bleu']:.4f}\n")
            f.write(f"Sacre chrF: {results['sacre_bleu_scores']['sacre_chrf']:.4f}\n\n")

            # chrF score
            f.write(f"chrF Score: {results['chrf']:.4f}\n")

        
        print(f"Saved evaluation metrics to {model_path}/evaluation_results.txt")

        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_dir = os.path.join(model_path, 'predictions')
        os.makedirs(predictions_dir, exist_ok=True)
        
        predictions_path = os.path.join(predictions_dir, f'predictions_{timestamp}.csv')
        
        # Create DataFrame with predictions and targets
        df = pd.DataFrame({
            'predicted': predicted,
            'target': targeted
        })
        
        # Save to CSV with UTF-8 encoding
        df.to_csv(predictions_path, index=False, encoding='utf-8')
        print(f"Saved predictions and targets to {predictions_path}")

        
    def get_predictions_from_file(self, file_path=None, predictions_dir=None):
        """
        Read predictions from a CSV file and return them as lists along with evaluation results
        
        Args:
            file_path (str, optional): Path to the CSV file. If None, will use the latest file in predictions directory
            predictions_dir (str, optional): Directory containing prediction files. Required if file_path is None
                
        Returns:
            tuple: (evaluation_results, predicted texts, target texts)
            
        Raises:
            FileNotFoundError: If no prediction file is found
            ValueError: If the CSV file doesn't have the expected columns
        """
        if file_path is None:
            if predictions_dir is None:
                raise ValueError("Either file_path or predictions_dir must be provided")
                
            if not os.path.exists(predictions_dir):
                raise FileNotFoundError(f"Predictions directory not found at {predictions_dir}")
                    
            csv_files = [f for f in os.listdir(predictions_dir) if f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError("No prediction CSV files found")
                    
            # Sort by filename (which includes timestamp) and get the latest
            csv_files.sort()
            file_path = os.path.join(predictions_dir, csv_files[-1])
                
        print(f"Reading predictions from: {file_path}")
        
        # Read the CSV file
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")
            
        # Check if required columns exist
        required_columns = ['predicted', 'target']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV file missing required columns: {missing_columns}")
                
        # Extract predictions and targets
        predicted = df['predicted'].tolist()
        target = df['target'].tolist()
        
        # Calculate all metrics
        results = {
            'position': self._evaluate_position_sensitive(predicted, target),
            'bleu_scores': self._evaluate_bleu(predicted, target),
            'sacre_bleu_scores': self._eval_sacrebleu_segment(predicted, target),
            'chrf': self._evaluate_chrf(predicted, target)
        }
        
        print(f"\nSuccessfully loaded {len(predicted)} predictions")
        
        # Print a few examples
        num_examples = min(3, len(predicted))
        print("\nExample predictions:")
        for i in range(num_examples):
            print(f"\nExample {i+1}:")
            print(f"Target: {target[i]}")
            print(f"Predicted: {predicted[i]}")
            
        return results, predicted, target