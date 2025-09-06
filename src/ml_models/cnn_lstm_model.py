"""
CNN+LSTM hybrid model for time series prediction.
Combines convolutional layers for pattern recognition with LSTM layers for sequence modeling.
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dense, Dropout, 
    BatchNormalization, Input, Concatenate, Attention,
    GlobalAveragePooling1D, Flatten, Reshape
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
from datetime import datetime

from config.model_config import CNNLSTMConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNNLSTMModel:
    """CNN+LSTM hybrid model for financial time series prediction."""
    
    def __init__(self, config: CNNLSTMConfig):
        self.config = config
        self.model = None
        self.history = None
        self.is_trained = False
        
    def build_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build the CNN+LSTM model architecture."""
        try:
            # Input layer
            inputs = Input(shape=input_shape, name='input_layer')
            
            # CNN layers for pattern recognition
            cnn_output = self._build_cnn_layers(inputs)
            
            # LSTM layers for sequence modeling
            lstm_output = self._build_lstm_layers(cnn_output)
            
            # Dense layers for final prediction
            predictions = self._build_dense_layers(lstm_output)
            
            # Create model
            model = Model(inputs=inputs, outputs=predictions, name='CNN_LSTM_Model')
            
            # Compile model
            optimizer = self._get_optimizer()
            model.compile(
                optimizer=optimizer,
                loss=self.config.loss_function,
                metrics=self.config.metrics
            )
            
            self.model = model
            logger.info(f"CNN+LSTM model built successfully. Parameters: {model.count_params()}")
            return model
            
        except Exception as e:
            logger.error(f"Error building CNN+LSTM model: {e}")
            return None
    
    def _build_cnn_layers(self, inputs: Input) -> Any:
        """Build CNN layers for pattern recognition."""
        x = inputs
        
        # Add channel dimension for Conv1D
        x = Reshape((inputs.shape[1], inputs.shape[2], 1))(x)
        x = Reshape((inputs.shape[1], inputs.shape[2]))(x)  # Remove extra dimension
        
        # CNN layers
        for i, (filters, kernel_size) in enumerate(zip(self.config.cnn_filters, self.config.cnn_kernel_sizes)):
            x = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation=self.config.cnn_activation,
                padding='same',
                name=f'conv1d_{i+1}'
            )(x)
            
            x = BatchNormalization(name=f'batch_norm_conv_{i+1}')(x)
            x = Dropout(self.config.cnn_dropout, name=f'dropout_conv_{i+1}')(x)
            
            # Add pooling every other layer
            if i % 2 == 1:
                x = MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}')(x)
        
        return x
    
    def _build_lstm_layers(self, inputs: Any) -> Any:
        """Build LSTM layers for sequence modeling."""
        x = inputs
        
        # LSTM layers
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1 or self.config.lstm_return_sequences
            
            x = LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.config.lstm_dropout,
                recurrent_dropout=self.config.lstm_recurrent_dropout,
                name=f'lstm_{i+1}'
            )(x)
            
            x = BatchNormalization(name=f'batch_norm_lstm_{i+1}')(x)
            x = Dropout(self.config.lstm_dropout, name=f'dropout_lstm_{i+1}')(x)
        
        return x
    
    def _build_dense_layers(self, inputs: Any) -> Any:
        """Build dense layers for final prediction."""
        x = inputs
        
        # Flatten if needed
        if len(x.shape) > 2:
            x = Flatten()(x)
        
        # Dense layers
        for i, units in enumerate(self.config.dense_units):
            x = Dense(
                units=units,
                activation=self.config.dense_activation,
                name=f'dense_{i+1}'
            )(x)
            
            x = BatchNormalization(name=f'batch_norm_dense_{i+1}')(x)
            x = Dropout(self.config.dense_dropout, name=f'dropout_dense_{i+1}')(x)
        
        # Output layer
        output = Dense(1, activation='linear', name='output')(x)
        
        return output
    
    def _get_optimizer(self):
        """Get optimizer based on configuration."""
        if self.config.optimizer.lower() == 'adam':
            return Adam(learning_rate=self.config.learning_rate)
        elif self.config.optimizer.lower() == 'rmsprop':
            return RMSprop(learning_rate=self.config.learning_rate)
        else:
            return Adam(learning_rate=self.config.learning_rate)
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              callbacks: Optional[List] = None) -> Dict[str, Any]:
        """Train the CNN+LSTM model."""
        try:
            if self.model is None:
                raise ValueError("Model must be built before training")
            
            # Prepare callbacks
            if callbacks is None:
                callbacks = self._get_default_callbacks()
            
            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            
            # Train model
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )
            
            self.is_trained = True
            logger.info("CNN+LSTM model training completed")
            
            return {
                'history': self.history.history,
                'final_loss': self.history.history['loss'][-1],
                'final_val_loss': self.history.history.get('val_loss', [None])[-1]
            }
            
        except Exception as e:
            logger.error(f"Error training CNN+LSTM model: {e}")
            return {}
    
    def _get_default_callbacks(self) -> List:
        """Get default callbacks for training."""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        lr_reduction = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.config.patience // 2,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_reduction)
        
        # Model checkpoint
        checkpoint_path = f"models/cnn_lstm_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(model_checkpoint)
        
        return callbacks
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            predictions = self.model.predict(X, verbose=0)
            return predictions.flatten()
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.array([])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions (for classification tasks)."""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # For regression, we can convert to probabilities using sigmoid
            predictions = self.model.predict(X, verbose=0)
            probabilities = tf.nn.sigmoid(predictions).numpy().flatten()
            return probabilities
            
        except Exception as e:
            logger.error(f"Error making probability predictions: {e}")
            return np.array([])
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data."""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            # Make predictions
            y_pred = self.predict(X_test)
            
            # Calculate metrics
            mse = np.mean((y_test - y_pred) ** 2)
            mae = np.mean(np.abs(y_test - y_pred))
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
            
            # Directional accuracy
            direction_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred))
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'mape': mape,
                'r2': r2,
                'direction_accuracy': direction_accuracy
            }
            
            logger.info(f"Model evaluation completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model."""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before saving")
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a pre-trained model."""
        try:
            self.model = tf.keras.models.load_model(filepath)
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built yet"
        
        return self.model.summary()
    
    def get_feature_importance(self, X_sample: np.ndarray) -> np.ndarray:
        """Get feature importance using gradient-based methods."""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before getting feature importance")
            
            # Convert to tensor
            X_tensor = tf.Variable(X_sample, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                predictions = self.model(X_tensor)
            
            # Calculate gradients
            gradients = tape.gradient(predictions, X_tensor)
            
            # Calculate importance as absolute gradient values
            importance = tf.reduce_mean(tf.abs(gradients), axis=0).numpy()
            
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return np.array([])


class CNNLSTMEnsemble:
    """Ensemble of CNN+LSTM models for improved predictions."""
    
    def __init__(self, models: List[CNNLSTMModel]):
        self.models = models
        self.weights = None
        
    def set_weights(self, weights: List[float]):
        """Set weights for ensemble voting."""
        if len(weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalize weights
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        try:
            predictions = []
            
            for i, model in enumerate(self.models):
                if not model.is_trained:
                    logger.warning(f"Model {i} is not trained, skipping")
                    continue
                
                pred = model.predict(X)
                predictions.append(pred)
            
            if not predictions:
                raise ValueError("No trained models available for prediction")
            
            # Weighted average
            if self.weights:
                weighted_predictions = []
                for i, pred in enumerate(predictions):
                    weighted_predictions.append(pred * self.weights[i])
                ensemble_pred = np.sum(weighted_predictions, axis=0)
            else:
                # Simple average
                ensemble_pred = np.mean(predictions, axis=0)
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {e}")
            return np.array([])
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        try:
            y_pred = self.predict(X_test)
            
            # Calculate metrics
            mse = np.mean((y_test - y_pred) ** 2)
            mae = np.mean(np.abs(y_test - y_pred))
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
            direction_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred))
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'mape': mape,
                'r2': r2,
                'direction_accuracy': direction_accuracy
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {e}")
            return {}


# Utility functions
def create_cnn_lstm_model(config: CNNLSTMConfig, input_shape: Tuple[int, int]) -> CNNLSTMModel:
    """Create and build a CNN+LSTM model."""
    model = CNNLSTMModel(config)
    model.build_model(input_shape)
    return model


def train_cnn_lstm_model(model: CNNLSTMModel, 
                        X_train: np.ndarray, 
                        y_train: np.ndarray,
                        X_val: np.ndarray = None,
                        y_val: np.ndarray = None) -> Dict[str, Any]:
    """Train a CNN+LSTM model."""
    return model.train(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    # Example usage
    from config.model_config import CNNLSTMConfig
    
    # Create configuration
    config = CNNLSTMConfig()
    
    # Create model
    model = CNNLSTMModel(config)
    
    # Build model with sample input shape
    input_shape = (60, 20)  # 60 time steps, 20 features
    model.build_model(input_shape)
    
    # Print model summary
    print(model.get_model_summary())
    
    # Example training data
    X_train = np.random.randn(1000, 60, 20)
    y_train = np.random.randn(1000)
    X_val = np.random.randn(200, 60, 20)
    y_val = np.random.randn(200)
    
    # Train model
    history = model.train(X_train, y_train, X_val, y_val)
    print("Training completed!")
    
    # Make predictions
    X_test = np.random.randn(100, 60, 20)
    predictions = model.predict(X_test)
    print(f"Predictions shape: {predictions.shape}")
    
    # Evaluate model
    y_test = np.random.randn(100)
    metrics = model.evaluate(X_test, y_test)
    print(f"Evaluation metrics: {metrics}")
