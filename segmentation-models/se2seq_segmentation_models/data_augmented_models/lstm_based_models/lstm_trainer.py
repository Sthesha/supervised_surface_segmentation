import json
import os
from data_processor import SegmentationDataProcessor
from lstm_model import SegmentationModel


class SegmentationTrainer:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)
            
    def train(self):
        # Initialize data processor

        # Print configuration
        print("\nTraining Configuration:")
        for category, params in {
            "Dataset": {
                "Data path": self.config['data_path'],
                "Dataset size": self.config.get('data_length', 'Full dataset')
            },
            "Model": {
                "Latent dim": self.config['latent_dim'],
                "Label smoothing": self.config.get('label_smoothing', 0.1),
                "Gradient clip norm": self.config.get('clip_norm', 1.0)
            },
            "Training": {
                "Initial learning rate": self.config['initial_lr'],
                "Batch size": self.config['batch_size'],
                "Epochs": self.config['num_epochs'],
                "Dropout rate": self.config['dropout_rate'],
                "Decay steps": self.config['decay_steps'],
                "Decay rate": self.config['decay_rate'],
                "Early stopping patience": self.config['patience'],
                "Validation split": self.config['validation_split']
            }
        }.items():
            print(f"\n{category}:")
            for key, value in params.items():
                print(f"- {key}: {value}")
            
        data_processor = SegmentationDataProcessor(self.config)
        
        # Load and process data
        data = data_processor.load_data(
            self.config["data_path"],
            self.config.get("data_length")
        )
        train_data, val_data, test_data = data_processor.split_data(data)
        
        # Prepare tokenization
        data_processor.prepare_tokenization(train_data)
        
        # Encode sequences
        encoder_input_data, decoder_input_data, decoder_target_data = data_processor.encode_sequences()
        
        # Initialize and train model
        model = SegmentationModel(self.config, data_processor)
        model.build_model(len(data_processor.char_to_idx))
        
        # Train model
        model.train(encoder_input_data, decoder_input_data, decoder_target_data)
        
        # Plot training history
        model.plot_training_history()
        
        # Save model
        model.save_model(os.path.join(self.config["save_path"], self.config["model_name"]))
        
        # Save test data
        test_data.to_csv(
            os.path.join(self.config["save_path"], self.config["model_name"], 'test_data.csv'),
            index=False
        )
        
        return model, data_processor