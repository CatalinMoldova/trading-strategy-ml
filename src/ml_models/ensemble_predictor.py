"""
Ensemble predictor that combines CNN+LSTM and Random Forest models.
Implements weighted voting and confidence scoring for robust predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib
import os
from datetime import datetime

from .cnn_lstm_model import CNNLSTMModel
from .random_forest_model import RandomForestModel
from config.model_config import EnsembleConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Ensemble predictor combining CNN+LSTM and Random Forest models."""
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.cnn_lstm_model = None
        self.random_forest_model = None
        self.is_trained = False
        self.model_weights = None
        self.performance_history = []
        
    def add_models(self, cnn_lstm_model: CNNLSTMModel, random_forest_model: RandomForestModel):
        """Add CNN+LSTM and Random Forest models to the ensemble."""
        self.cnn_lstm_model = cnn_lstm_model
        self.random_forest_model = random_forest_model
        
        logger.info("Models added to ensemble")
    
    def train_ensemble(self, 
                      X_train: np.ndarray, 
                      y_train: np.ndarray,
                      X_val: Optional[np.ndarray] = None,
                      y_val: Optional[np.ndarray] = None,
                      feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train the ensemble models."""
        try:
            if self.cnn_lstm_model is None or self.random_forest_model is None:
                raise ValueError("Both CNN+LSTM and Random Forest models must be added before training")
            
            results = {}
            
            # Train CNN+LSTM model
            logger.info("Training CNN+LSTM model...")
            cnn_lstm_results = self.cnn_lstm_model.train(X_train, y_train, X_val, y_val)
            results['cnn_lstm'] = cnn_lstm_results
            
            # Prepare data for Random Forest (flatten sequences)
            X_train_rf = X_train.reshape(X_train.shape[0], -1)
            X_val_rf = X_val.reshape(X_val.shape[0], -1) if X_val is not None else None
            
            # Train Random Forest model
            logger.info("Training Random Forest model...")
            rf_results = self.random_forest_model.train(X_train_rf, y_train, X_val_rf, y_val, feature_names)
            results['random_forest'] = rf_results
            
            # Calculate initial weights based on validation performance
            if X_val is not None and y_val is not None:
                self._calculate_initial_weights(X_val, y_val)
            
            self.is_trained = True
            
            logger.info("Ensemble training completed")
            return results
            
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            return {}
    
    def _calculate_initial_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """Calculate initial model weights based on validation performance."""
        try:
            # Get predictions from both models
            cnn_lstm_pred = self.cnn_lstm_model.predict(X_val)
            rf_pred = self.random_forest_model.predict(X_val.reshape(X_val.shape[0], -1))
            
            # Calculate MSE for each model
            cnn_lstm_mse = mean_squared_error(y_val, cnn_lstm_pred)
            rf_mse = mean_squared_error(y_val, rf_pred)
            
            # Calculate weights (inverse of MSE)
            total_inverse_mse = (1/cnn_lstm_mse) + (1/rf_mse)
            cnn_lstm_weight = (1/cnn_lstm_mse) / total_inverse_mse
            rf_weight = (1/rf_mse) / total_inverse_mse
            
            # Apply configuration weights
            self.model_weights = {
                'cnn_lstm': cnn_lstm_weight * self.config.cnn_lstm_weight,
                'random_forest': rf_weight * self.config.random_forest_weight
            }
            
            # Normalize weights
            total_weight = sum(self.model_weights.values())
            self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}
            
            logger.info(f"Initial weights calculated: {self.model_weights}")
            
        except Exception as e:
            logger.error(f"Error calculating initial weights: {e}")
            # Use default weights
            self.model_weights = {
                'cnn_lstm': self.config.cnn_lstm_weight,
                'random_forest': self.config.random_forest_weight
            }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        try:
            if not self.is_trained:
                raise ValueError("Ensemble must be trained before making predictions")
            
            # Get predictions from both models
            cnn_lstm_pred = self.cnn_lstm_model.predict(X)
            rf_pred = self.random_forest_model.predict(X.reshape(X.shape[0], -1))
            
            # Calculate ensemble prediction based on voting method
            if self.config.voting_method == 'weighted':
                ensemble_pred = (
                    cnn_lstm_pred * self.model_weights['cnn_lstm'] +
                    rf_pred * self.model_weights['random_forest']
                )
            elif self.config.voting_method == 'soft':
                # Soft voting using probability-like scores
                ensemble_pred = (cnn_lstm_pred + rf_pred) / 2
            else:  # hard voting
                ensemble_pred = (cnn_lstm_pred + rf_pred) / 2
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {e}")
            return np.array([])
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence scores."""
        try:
            if not self.is_trained:
                raise ValueError("Ensemble must be trained before making predictions")
            
            # Get predictions from both models
            cnn_lstm_pred = self.cnn_lstm_model.predict(X)
            rf_pred = self.random_forest_model.predict(X.reshape(X.shape[0], -1))
            
            # Calculate ensemble prediction
            ensemble_pred = self.predict(X)
            
            # Calculate confidence based on agreement between models
            prediction_diff = np.abs(cnn_lstm_pred - rf_pred)
            max_diff = np.max(prediction_diff)
            
            if max_diff > 0:
                confidence = 1 - (prediction_diff / max_diff)
            else:
                confidence = np.ones_like(prediction_diff)
            
            # Apply confidence thresholds
            confidence = np.clip(confidence, self.config.min_confidence, self.config.max_confidence)
            
            return ensemble_pred, confidence
            
        except Exception as e:
            logger.error(f"Error making predictions with confidence: {e}")
            return np.array([]), np.array([])
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate ensemble performance."""
        try:
            if not self.is_trained:
                raise ValueError("Ensemble must be trained before evaluation")
            
            # Get ensemble predictions
            ensemble_pred = self.predict(X_test)
            
            # Get individual model predictions
            cnn_lstm_pred = self.cnn_lstm_model.predict(X_test)
            rf_pred = self.random_forest_model.predict(X_test.reshape(X_test.shape[0], -1))
            
            # Calculate metrics for ensemble
            ensemble_metrics = self._calculate_metrics(y_test, ensemble_pred, 'ensemble')
            
            # Calculate metrics for individual models
            cnn_lstm_metrics = self._calculate_metrics(y_test, cnn_lstm_pred, 'cnn_lstm')
            rf_metrics = self._calculate_metrics(y_test, rf_pred, 'random_forest')
            
            # Calculate confidence metrics
            _, confidence = self.predict_with_confidence(X_test)
            confidence_metrics = {
                'mean_confidence': np.mean(confidence),
                'std_confidence': np.std(confidence),
                'min_confidence': np.min(confidence),
                'max_confidence': np.max(confidence)
            }
            
            results = {
                'ensemble_metrics': ensemble_metrics,
                'cnn_lstm_metrics': cnn_lstm_metrics,
                'random_forest_metrics': rf_metrics,
                'confidence_metrics': confidence_metrics,
                'model_weights': self.model_weights
            }
            
            logger.info(f"Ensemble evaluation completed: {ensemble_metrics}")
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating ensemble: {e}")
            return {}
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        try:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))
            
            metrics = {
                f'{model_name}_mse': mse,
                f'{model_name}_mae': mae,
                f'{model_name}_r2': r2,
                f'{model_name}_direction_accuracy': direction_accuracy
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def update_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """Update model weights based on recent performance."""
        try:
            if not self.is_trained:
                raise ValueError("Ensemble must be trained before updating weights")
            
            # Get predictions
            cnn_lstm_pred = self.cnn_lstm_model.predict(X_val)
            rf_pred = self.random_forest_model.predict(X_val.reshape(X_val.shape[0], -1))
            
            # Calculate recent performance
            cnn_lstm_mse = mean_squared_error(y_val, cnn_lstm_pred)
            rf_mse = mean_squared_error(y_val, rf_pred)
            
            # Update weights based on performance
            total_inverse_mse = (1/cnn_lstm_mse) + (1/rf_mse)
            new_cnn_lstm_weight = (1/cnn_lstm_mse) / total_inverse_mse
            new_rf_weight = (1/rf_mse) / total_inverse_mse
            
            # Apply configuration weights
            self.model_weights = {
                'cnn_lstm': new_cnn_lstm_weight * self.config.cnn_lstm_weight,
                'random_forest': new_rf_weight * self.config.random_forest_weight
            }
            
            # Normalize weights
            total_weight = sum(self.model_weights.values())
            self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}
            
            # Store performance history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'cnn_lstm_mse': cnn_lstm_mse,
                'rf_mse': rf_mse,
                'weights': self.model_weights.copy()
            })
            
            logger.info(f"Weights updated: {self.model_weights}")
            
        except Exception as e:
            logger.error(f"Error updating weights: {e}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get ensemble feature importance from Random Forest model."""
        try:
            if not self.is_trained:
                raise ValueError("Ensemble must be trained before getting feature importance")
            
            return self.random_forest_model.get_feature_importance()
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame()
    
    def get_model_performance_history(self) -> pd.DataFrame:
        """Get model performance history."""
        try:
            if not self.performance_history:
                return pd.DataFrame()
            
            history_df = pd.DataFrame(self.performance_history)
            return history_df
            
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return pd.DataFrame()
    
    def save_ensemble(self, filepath: str) -> bool:
        """Save the ensemble model."""
        try:
            if not self.is_trained:
                raise ValueError("Ensemble must be trained before saving")
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save ensemble information
            ensemble_info = {
                'config': self.config.__dict__,
                'model_weights': self.model_weights,
                'performance_history': self.performance_history,
                'is_trained': self.is_trained
            }
            
            joblib.dump(ensemble_info, filepath)
            
            # Save individual models
            cnn_lstm_path = filepath.replace('.pkl', '_cnn_lstm.h5')
            rf_path = filepath.replace('.pkl', '_random_forest.pkl')
            
            self.cnn_lstm_model.save_model(cnn_lstm_path)
            self.random_forest_model.save_model(rf_path)
            
            logger.info(f"Ensemble saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving ensemble: {e}")
            return False
    
    def load_ensemble(self, filepath: str) -> bool:
        """Load a pre-trained ensemble."""
        try:
            # Load ensemble information
            ensemble_info = joblib.load(filepath)
            
            self.config = EnsembleConfig(**ensemble_info['config'])
            self.model_weights = ensemble_info['model_weights']
            self.performance_history = ensemble_info['performance_history']
            self.is_trained = ensemble_info['is_trained']
            
            # Load individual models
            cnn_lstm_path = filepath.replace('.pkl', '_cnn_lstm.h5')
            rf_path = filepath.replace('.pkl', '_random_forest.pkl')
            
            if os.path.exists(cnn_lstm_path):
                self.cnn_lstm_model = CNNLSTMModel(self.config)
                self.cnn_lstm_model.load_model(cnn_lstm_path)
            
            if os.path.exists(rf_path):
                from config.model_config import RandomForestConfig
                rf_config = RandomForestConfig()
                self.random_forest_model = RandomForestModel(rf_config)
                self.random_forest_model.load_model(rf_path)
            
            logger.info(f"Ensemble loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ensemble: {e}")
            return False
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get ensemble information."""
        info = {
            'is_trained': self.is_trained,
            'voting_method': self.config.voting_method,
            'model_weights': self.model_weights,
            'performance_history_length': len(self.performance_history),
            'cnn_lstm_trained': self.cnn_lstm_model.is_trained if self.cnn_lstm_model else False,
            'random_forest_trained': self.random_forest_model.is_trained if self.random_forest_model else False
        }
        
        return info


class DynamicEnsemblePredictor(EnsemblePredictor):
    """Dynamic ensemble that adapts weights based on recent performance."""
    
    def __init__(self, config: EnsembleConfig, adaptation_window: int = 30):
        super().__init__(config)
        self.adaptation_window = adaptation_window
        self.recent_performance = []
    
    def predict_with_adaptive_weights(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with adaptively updated weights."""
        try:
            if not self.is_trained:
                raise ValueError("Ensemble must be trained before making predictions")
            
            # Get predictions from both models
            cnn_lstm_pred = self.cnn_lstm_model.predict(X)
            rf_pred = self.random_forest_model.predict(X.reshape(X.shape[0], -1))
            
            # Calculate adaptive weights based on recent performance
            adaptive_weights = self._calculate_adaptive_weights()
            
            # Calculate ensemble prediction
            ensemble_pred = (
                cnn_lstm_pred * adaptive_weights['cnn_lstm'] +
                rf_pred * adaptive_weights['random_forest']
            )
            
            # Calculate confidence
            prediction_diff = np.abs(cnn_lstm_pred - rf_pred)
            max_diff = np.max(prediction_diff) if np.max(prediction_diff) > 0 else 1
            confidence = 1 - (prediction_diff / max_diff)
            confidence = np.clip(confidence, self.config.min_confidence, self.config.max_confidence)
            
            return ensemble_pred, confidence
            
        except Exception as e:
            logger.error(f"Error making adaptive predictions: {e}")
            return np.array([]), np.array([])
    
    def _calculate_adaptive_weights(self) -> Dict[str, float]:
        """Calculate adaptive weights based on recent performance."""
        try:
            if len(self.recent_performance) < 2:
                return self.model_weights
            
            # Get recent performance
            recent_perf = self.recent_performance[-self.adaptation_window:]
            
            # Calculate average performance for each model
            cnn_lstm_performance = np.mean([p['cnn_lstm_mse'] for p in recent_perf])
            rf_performance = np.mean([p['rf_mse'] for p in recent_perf])
            
            # Calculate adaptive weights
            total_inverse_perf = (1/cnn_lstm_performance) + (1/rf_performance)
            adaptive_weights = {
                'cnn_lstm': (1/cnn_lstm_performance) / total_inverse_perf,
                'random_forest': (1/rf_performance) / total_inverse_perf
            }
            
            return adaptive_weights
            
        except Exception as e:
            logger.error(f"Error calculating adaptive weights: {e}")
            return self.model_weights
    
    def update_performance(self, X_val: np.ndarray, y_val: np.ndarray):
        """Update recent performance for adaptive weighting."""
        try:
            # Get predictions
            cnn_lstm_pred = self.cnn_lstm_model.predict(X_val)
            rf_pred = self.random_forest_model.predict(X_val.reshape(X_val.shape[0], -1))
            
            # Calculate performance
            cnn_lstm_mse = mean_squared_error(y_val, cnn_lstm_pred)
            rf_mse = mean_squared_error(y_val, rf_pred)
            
            # Store recent performance
            self.recent_performance.append({
                'timestamp': datetime.now(),
                'cnn_lstm_mse': cnn_lstm_mse,
                'rf_mse': rf_mse
            })
            
            # Keep only recent performance
            if len(self.recent_performance) > self.adaptation_window * 2:
                self.recent_performance = self.recent_performance[-self.adaptation_window:]
            
            logger.info(f"Performance updated: CNN+LSTM MSE: {cnn_lstm_mse:.6f}, RF MSE: {rf_mse:.6f}")
            
        except Exception as e:
            logger.error(f"Error updating performance: {e}")


# Utility functions
def create_ensemble_predictor(config: EnsembleConfig) -> EnsemblePredictor:
    """Create an ensemble predictor."""
    return EnsemblePredictor(config)


def create_dynamic_ensemble_predictor(config: EnsembleConfig, adaptation_window: int = 30) -> DynamicEnsemblePredictor:
    """Create a dynamic ensemble predictor."""
    return DynamicEnsemblePredictor(config, adaptation_window)


if __name__ == "__main__":
    # Example usage
    from config.model_config import EnsembleConfig, CNNLSTMConfig, RandomForestConfig
    
    # Create configurations
    ensemble_config = EnsembleConfig()
    cnn_lstm_config = CNNLSTMConfig()
    rf_config = RandomForestConfig()
    
    # Create models
    cnn_lstm_model = CNNLSTMModel(cnn_lstm_config)
    rf_model = RandomForestModel(rf_config)
    
    # Create ensemble
    ensemble = EnsemblePredictor(ensemble_config)
    ensemble.add_models(cnn_lstm_model, rf_model)
    
    # Example training data
    X_train = np.random.randn(1000, 60, 20)
    y_train = np.random.randn(1000)
    X_val = np.random.randn(200, 60, 20)
    y_val = np.random.randn(200)
    
    # Train ensemble
    results = ensemble.train_ensemble(X_train, y_train, X_val, y_val)
    print("Ensemble training completed!")
    
    # Make predictions
    X_test = np.random.randn(100, 60, 20)
    predictions = ensemble.predict(X_test)
    predictions_with_conf, confidence = ensemble.predict_with_confidence(X_test)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Confidence shape: {confidence.shape}")
    
    # Evaluate ensemble
    y_test = np.random.randn(100)
    metrics = ensemble.evaluate(X_test, y_test)
    print(f"Ensemble metrics: {metrics['ensemble_metrics']}")
    
    # Get ensemble info
    info = ensemble.get_ensemble_info()
    print(f"Ensemble info: {info}")
