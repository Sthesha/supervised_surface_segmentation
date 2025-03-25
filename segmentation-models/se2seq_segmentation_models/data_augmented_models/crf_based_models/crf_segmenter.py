# main.py
import os
import json
from datetime import datetime
from data_loader import DataLoader
from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer
from evaluator import Evaluator

class MorphologicalSegmentation:
    """Main class that orchestrates the morphological segmentation pipeline"""
    
    def __init__(self, config):
        if 'model_name' not in config:
            raise ValueError("Config must include 'model_name'")
            
        self.config = config.copy()
        self.model_name = config['model_name']
        self.model_path = os.path.join(os.getcwd(), config['save_path'], self.model_name)
        os.makedirs(self.model_path, exist_ok=True)

        self.config["file_path"] = self.config.get("file_path", self.model_path)
        
        # Save configuration
        self._save_config()
        
        # Initialize components
        self.data_loader = DataLoader(config)
        self.feature_extractor = FeatureExtractor(config['feature_params'])
        self.trainer = ModelTrainer(self.feature_extractor, config['crf_params'], self.model_path)
        self.evaluator = Evaluator(self.feature_extractor)
        
        # Data storage
        self.data = None
        
        print(f"Initialized Morphological Segmentation with model name: {self.model_name}")

    def _save_config(self):
        """Save configuration to a JSON file"""
        config_path = os.path.join(self.model_path, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"Configuration saved to {config_path}")
    
    def load_data(self):
        """Load and preprocess the data"""
        print(f"Loading data from {self.config['data_path']}")
        try:
            self.data = self.data_loader.load_data()
            print("Data loaded successfully")
            return self.data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def train_model(self, force_retrain=False):
        """Train the morphological segmentation model or load if exists"""
        if self.data is None:
            print("No data available. Please load data first.")
            raise ValueError("Please load data first using load_data()")
        
        # Check for existing model
        model_file = os.path.join(self.model_path, f"{self.model_name}.pkl")
        if os.path.exists(model_file) and not force_retrain:
            print(f"Found existing model at {model_file}")
            try:
                model = self.trainer.load_model(model_file)
                print("Successfully loaded existing model")
                return model
            except Exception as e:
                print(f"Error loading existing model: {str(e)}")
                print("Will train a new model instead")
        
        if force_retrain:
            print("Force retrain enabled, training new model...")
            
        print("Starting model training")
        try:
            model = self.trainer.train(
                self.data['train']['tokens_labels'],
                self.data['dev']['tokens_labels'],
                use_grid_search=self.config['use_grid_search'],
                grid_params=self.config['grid_search_params']
            )
            
            # Save the trained model
            self.trainer.save_model(model_file)
            print(f"Model saved to {model_file}")
            
            return model
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def evaluate_model(self, model=None):
        """Evaluate the model performance."""
        if model is None and hasattr(self.trainer, 'model'):
            model = self.trainer.model
    
        if model is None:
            raise ValueError("No model available for evaluation.")
    
        if self.data is None or 'test' not in self.data:
            raise ValueError("Test data not loaded. Please ensure 'test' data is available.")
    
        test_data = self.data['test']['tokens_labels']
        test_segments = self.data['test'].get('segments', {})
        
        print("Starting model evaluation...")
        results, predictions, targeted = self.evaluator.evaluate(
            model,
            test_data,
            test_segments
        )
        self.evaluator.save_results(results, self.model_path, predictions, targeted)
        
        return results


    def find_best_parameters(self):
        """Run grid search to find best parameters using a subset of data"""
        print(f"Starting grid search for model: {self.model_name}")
        
        # Create a copy of the config for tuning
        tuning_config = self.config.copy()
        tuning_config['data_length'] = self.config.get('tuning_length', int(self.config['data_length'] * 0.2))
        
        # Create a new instance for tuning
        tuning_instance = MorphologicalSegmentation(tuning_config)
        tuning_instance.load_data()
        model = tuning_instance.train_model(force_retrain=True)
        
        # Get the best parameters
        best_params = model.get_params()
        print(f"\nBest parameters found for {self.model_name}:")        
        return best_params

    def train_with_parameters(self, best_params):
        """Train model with found parameters using full dataset"""
        # Use current config, just update the parameters
        self.config['use_grid_search'] = False
        self.config['data_length'] = self.config.get('data_length', 200000)
        
        # Only update parameters that were originally defined in the config
        original_params = set(self.config['crf_params'].keys())
        filtered_params = {k: v for k, v in best_params.items() if k in original_params}
        print(filtered_params)
        self.config['crf_params'].update(filtered_params)
        
        print(f"\nTraining final model for {self.model_name} with best parameters...")
        print("CRF parameters:")
        print(json.dumps(self.config['crf_params'], indent=2))
        
        # Save updated configuration
        self._save_config()
        
        # Load data and train with full dataset
        self.load_data()
        model = self.train_model(force_retrain=True)
        
        return self