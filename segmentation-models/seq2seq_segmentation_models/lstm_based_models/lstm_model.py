import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense
from keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import json
from pathlib import Path
import numpy as np

class SegmentationModel:
    def __init__(self, config: dict, data_processor=None):
        self.config = config
        self.data_processor = data_processor
        self.model = None
        self.history = None

    def build_model(self, vocab_size: int):
        
        encoder_inputs = Input(shape=(None, vocab_size))
        encoder = LSTM(
            self.config["latent_dim"], 
            return_state=True, 
            recurrent_dropout=self.config.get("dropout_rate", 0.5)
        )
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        decoder_inputs = Input(shape=(None, vocab_size))
        decoder_lstm = LSTM(
            self.config["latent_dim"], 
            return_sequences=True, 
            return_state=True, 
            recurrent_dropout=self.config.get("dropout_rate", 0.5)
        )
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.config.get("initial_lr", 0.0001),
            decay_steps=self.config.get("decay_steps", 10000),
            decay_rate=self.config.get("decay_rate", 0.9)
        )
        optimizer = Adam(learning_rate=lr_schedule)
        
        
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=self.config.get("label_smoothing", 0.1)),
            metrics=['accuracy']
        )

    def train(self, encoder_input_data: np.ndarray, decoder_input_data: np.ndarray, 
              decoder_target_data: np.ndarray):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.get("patience", 5),
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=self.config["batch_size"],
            epochs=self.config["num_epochs"],
            validation_split=self.config.get("validation_split", 0.1),
            callbacks=[early_stopping]
        )

    def plot_training_history(self):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'])
        
        plt.subplot(1, 3, 2)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'])
        
        plt.subplot(1, 3, 3)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['accuracy'])
        plt.title('Loss vs Accuracy')
        plt.ylabel('Value')
        plt.xlabel('Epoch')
        plt.legend(['Loss', 'Accuracy'])
        
        plt.tight_layout()
        plt.show()

    def save_model(self, save_path: str):
        model_path = Path(save_path)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save(model_path / 'lstm_model.h5')

        # Save history
        if self.history:
            np.save(model_path / 'training_history.npy', self.history.history)

        # Save config
        dimensions = {
            'latent_dim': self.config["latent_dim"],
            'vocab_size': len(self.data_processor.char_to_idx),
            'char_to_idx': self.data_processor.char_to_idx,
            'idx_to_char': self.data_processor.idx_to_char,
            'special_tokens': self.data_processor.special_tokens,
            'model_config': self.config
        }

        with open(model_path / 'model_config.json', 'w') as f:
            json.dump(dimensions, f, indent=2)

    @classmethod
    def load_model(cls, model_path: str) -> 'SegmentationModel':
        model_path = Path(model_path)
        with open(model_path / 'model_config.json', 'r') as f:
            config = json.load(f)
            
        model = cls(config, None)
        model.model = load_model(model_path / 'lstm_model.h5')
        return model