# lstm_tuner.py
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import os
import tensorflow as tf
from data_processor import SegmentationDataProcessor
from lstm_model import SegmentationModel
import numpy as np
from functools import partial
from ray import train

class LSTMTuner:
    def __init__(self, config: dict):
        self.base_config = config

    @staticmethod
    def _training_function(config):
        """Static training function that Ray can pickle"""
        import tensorflow as tf  # Import inside function for Ray
        from data_processor import SegmentationDataProcessor
        from lstm_model import SegmentationModel
        
        # Initialize data processor with config
        data_processor = SegmentationDataProcessor(config)
                
        # Load and process data
        data = data_processor.load_data(
            config["data_path"],
            config.get("data_length")
        )
        train_data, val_data, _ = data_processor.split_data(data)
        data_processor.prepare_tokenization(train_data)
        
        # Create model
        model = SegmentationModel(config, data_processor)
        
        model.build_model(len(data_processor.char_to_idx))
        
        # Get data
        encoder_input_data, decoder_input_data, decoder_target_data = data_processor.encode_sequences()
        
        # Setup early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.get("patience", 5),
            restore_best_weights=True
        )
        
        # Train
        history = model.model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=int(config["batch_size"]),
            epochs=config["tuning_epochs"],
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Report metrics
        train.report({
            "loss": history.history['loss'][-1],
            "val_loss": history.history['val_loss'][-1],
            "accuracy": history.history['accuracy'][-1],
            "val_accuracy": history.history['val_accuracy'][-1]
        })
        
    def run_hyperparameter_search(self, num_samples=10, num_epochs=5):
        """Run hyperparameter optimization"""
        
        # Add tuning epochs to config
        config = self.base_config.copy()
        config["tuning_epochs"] = num_epochs
        
        # Define search space
        search_space = {
            **config,
            "latent_dim": tune.choice([128, 256, 512, 1024]),
            "dropout_rate": tune.uniform(0.1, 0.3),
            "initial_lr": tune.loguniform(1e-5, 1e-3),
            "decay_rate": tune.uniform(0.8, 0.99),
            "label_smoothing": tune.uniform(0.00, 0.2),
            "clip_norm": tune.uniform(0.5, 2.0)
        }
        
        # Initialize Ray
        if ray.is_initialized():
            ray.shutdown()
        ray.init(
            runtime_env={
                "working_dir": ".",  # Include current directory
                "pip": ["tensorflow", "pandas", "numpy"]  # Include required packages
            }
        )
        
        # Setup Optuna search algorithm
        search_algo = OptunaSearch(
            metric="val_loss",
            mode="min",
        )
        
        # Setup ASHA scheduler
        scheduler = ASHAScheduler(
            time_attr='training_iteration',
            metric='val_loss',
            mode='min',
            max_t=num_epochs,
            grace_period=2,
            reduction_factor=2
        )
        
        # Create results directory
        storage_path = os.path.abspath(os.path.join(
            os.getcwd(),
            self.base_config['save_path'], 
            config["model_name"],
            "ray_results"
        ))
        
        
        data_path = os.path.abspath(os.path.join(os.getcwd(), config["data_path"]))
        
        # Run optimization
        try:
            analysis = tune.run(
                self._training_function,  # Use static class method
                config=search_space,
                num_samples=num_samples,
                scheduler=scheduler,
                search_alg=search_algo,
                resources_per_trial={
                    "cpu": 16,  # Reduced for testing
                    "gpu": 1 if tf.config.list_physical_devices('GPU') else 0
                },
                storage_path=storage_path,
                name="lstm_tune",
                verbose=2,
                log_to_file=True,
                local_dir=data_path
            )
            
            # Get best config
            best_config = analysis.get_best_config(metric="val_loss", mode="min")
            best_trial = analysis.get_best_trial(metric="val_loss", mode="min")
            
            print("\nBest trial config:", best_config)
            print(f"Best trial final validation loss: {best_trial.last_result['val_loss']:.4f}")
            print(f"Best trial final validation accuracy: {best_trial.last_result['val_accuracy']:.4f}")
            
            return best_config, analysis
            
        except Exception as e:
            print(f"Error during tuning: {str(e)}")
            raise
        finally:
            ray.shutdown()

    def apply_best_config(self, best_config):
        """Apply best configuration to base config"""
        final_config = self.base_config.copy()
        final_config.update(best_config)
        return final_config