"""
Model trainer for orchestrating the training of all ML models.
Handles hyperparameter optimization, cross-validation, and model selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import os
from datetime import datetime
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from .cnn_lstm_model import CNNLSTMModel
from .random_forest_model import RandomForestModel
from .ensemble_predictor import EnsemblePredictor, DynamicEnsemblePredictor
from .data_preprocessor import TimeSeriesPreprocessor
from config.model_config import ModelConfig, CNNLSTMConfig, RandomForestConfig, EnsembleConfig, TrainingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Main trainer class for orchestrating model training and optimization."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.training_config = config.training
        self.models = {}
        self.training_results = {}
        self.best_models = {}
        
    def prepare_training_data(self, 
                             features: pd.DataFrame, 
                             target: pd.Series) -> Dict[str, Any]:
        """Prepare data for model training."""
        try:
            preprocessor = TimeSeriesPreprocessor(
                sequence_length=self.config.cnn_lstm.sequence_length,
                prediction_horizon=self.config.cnn_lstm.prediction_horizon,
                test_size=self.training_config.test_split,
                validation_size=self.training_config.validation_split,
                scaler_type=self.config.feature_engineering.normalization_method
            )
            
            data = preprocessor.prepare_data(features, target)
            
            if not data:
                raise ValueError("Failed to prepare training data")
            
            logger.info("Training data prepared successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return {}
    
    def train_cnn_lstm_model(self, 
                           X_train: np.ndarray, 
                           y_train: np.ndarray,
                           X_val: np.ndarray, 
                           y_val: np.ndarray,
                           optimize_hyperparams: bool = True) -> Dict[str, Any]:
        """Train CNN+LSTM model with optional hyperparameter optimization."""
        try:
            logger.info("Starting CNN+LSTM model training")
            
            if optimize_hyperparams and self.training_config.use_optuna:
                # Hyperparameter optimization
                best_params = self._optimize_cnn_lstm_hyperparams(X_train, y_train, X_val, y_val)
                
                # Update config with best parameters
                for param, value in best_params.items():
                    if hasattr(self.config.cnn_lstm, param):
                        setattr(self.config.cnn_lstm, param, value)
            
            # Create and train model
            model = CNNLSTMModel(self.config.cnn_lstm)
            model.build_model((X_train.shape[1], X_train.shape[2]))
            
            # Train model
            training_results = model.train(X_train, y_train, X_val, y_val)
            
            # Store model and results
            self.models['cnn_lstm'] = model
            self.training_results['cnn_lstm'] = training_results
            
            logger.info("CNN+LSTM model training completed")
            return training_results
            
        except Exception as e:
            logger.error(f"Error training CNN+LSTM model: {e}")
            return {}
    
    def train_random_forest_model(self, 
                                X_train: np.ndarray, 
                                y_train: np.ndarray,
                                X_val: np.ndarray, 
                                y_val: np.ndarray,
                                feature_names: List[str],
                                optimize_hyperparams: bool = True) -> Dict[str, Any]:
        """Train Random Forest model with optional hyperparameter optimization."""
        try:
            logger.info("Starting Random Forest model training")
            
            # Flatten sequences for Random Forest
            X_train_rf = X_train.reshape(X_train.shape[0], -1)
            X_val_rf = X_val.reshape(X_val.shape[0], -1)
            
            if optimize_hyperparams and self.training_config.use_optuna:
                # Hyperparameter optimization
                best_params = self._optimize_random_forest_hyperparams(X_train_rf, y_train, X_val_rf, y_val)
                
                # Update config with best parameters
                for param, value in best_params.items():
                    if hasattr(self.config.random_forest, param):
                        setattr(self.config.random_forest, param, value)
            
            # Create and train model
            model = RandomForestModel(self.config.random_forest)
            training_results = model.train(X_train_rf, y_train, X_val_rf, y_val, feature_names)
            
            # Store model and results
            self.models['random_forest'] = model
            self.training_results['random_forest'] = training_results
            
            logger.info("Random Forest model training completed")
            return training_results
            
        except Exception as e:
            logger.error(f"Error training Random Forest model: {e}")
            return {}
    
    def train_ensemble_model(self, 
                            X_train: np.ndarray, 
                            y_train: np.ndarray,
                            X_val: np.ndarray, 
                            y_val: np.ndarray,
                            feature_names: List[str]) -> Dict[str, Any]:
        """Train ensemble model."""
        try:
            logger.info("Starting ensemble model training")
            
            # Ensure individual models are trained
            if 'cnn_lstm' not in self.models or 'random_forest' not in self.models:
                raise ValueError("Both CNN+LSTM and Random Forest models must be trained first")
            
            # Create ensemble
            ensemble = EnsemblePredictor(self.config.ensemble)
            ensemble.add_models(self.models['cnn_lstm'], self.models['random_forest'])
            
            # Train ensemble
            training_results = ensemble.train_ensemble(X_train, y_train, X_val, y_val, feature_names)
            
            # Store model and results
            self.models['ensemble'] = ensemble
            self.training_results['ensemble'] = training_results
            
            logger.info("Ensemble model training completed")
            return training_results
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            return {}
    
    def train_all_models(self, 
                        features: pd.DataFrame, 
                        target: pd.Series,
                        optimize_hyperparams: bool = True) -> Dict[str, Any]:
        """Train all models in sequence."""
        try:
            logger.info("Starting comprehensive model training")
            
            # Prepare data
            data = self.prepare_training_data(features, target)
            if not data:
                raise ValueError("Failed to prepare training data")
            
            # Extract data
            X_train = data['X_train']
            X_val = data['X_val']
            X_test = data['X_test']
            y_train = data['y_train']
            y_val = data['y_val']
            y_test = data['y_test']
            feature_names = data['feature_names']
            
            # Train individual models
            cnn_lstm_results = self.train_cnn_lstm_model(X_train, y_train, X_val, y_val, optimize_hyperparams)
            rf_results = self.train_random_forest_model(X_train, y_train, X_val, y_val, feature_names, optimize_hyperparams)
            
            # Train ensemble
            ensemble_results = self.train_ensemble_model(X_train, y_train, X_val, y_val, feature_names)
            
            # Evaluate all models on test set
            test_results = self.evaluate_all_models(X_test, y_test)
            
            # Compile results
            all_results = {
                'cnn_lstm': cnn_lstm_results,
                'random_forest': rf_results,
                'ensemble': ensemble_results,
                'test_results': test_results,
                'data_info': {
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test),
                    'features': len(feature_names),
                    'sequence_length': data['sequence_length']
                }
            }
            
            logger.info("All models trained successfully")
            return all_results
            
        except Exception as e:
            logger.error(f"Error training all models: {e}")
            return {}
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate all trained models on test data."""
        try:
            test_results = {}
            
            # Evaluate CNN+LSTM
            if 'cnn_lstm' in self.models:
                cnn_lstm_metrics = self.models['cnn_lstm'].evaluate(X_test, y_test)
                test_results['cnn_lstm'] = cnn_lstm_metrics
            
            # Evaluate Random Forest
            if 'random_forest' in self.models:
                X_test_rf = X_test.reshape(X_test.shape[0], -1)
                rf_metrics = self.models['random_forest'].evaluate(X_test_rf, y_test)
                test_results['random_forest'] = rf_metrics
            
            # Evaluate Ensemble
            if 'ensemble' in self.models:
                ensemble_metrics = self.models['ensemble'].evaluate(X_test, y_test)
                test_results['ensemble'] = ensemble_metrics
            
            logger.info("All models evaluated on test data")
            return test_results
            
        except Exception as e:
            logger.error(f"Error evaluating models: {e}")
            return {}
    
    def _optimize_cnn_lstm_hyperparams(self, 
                                      X_train: np.ndarray, 
                                      y_train: np.ndarray,
                                      X_val: np.ndarray, 
                                      y_val: np.ndarray) -> Dict[str, Any]:
        """Optimize CNN+LSTM hyperparameters using Optuna."""
        try:
            def objective(trial):
                # Suggest hyperparameters
                lstm_units_1 = trial.suggest_int('lstm_units_1', 32, 256)
                lstm_units_2 = trial.suggest_int('lstm_units_2', 16, 128)
                dense_units = trial.suggest_int('dense_units', 16, 128)
                learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
                dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
                
                # Create config with suggested parameters
                config = CNNLSTMConfig(
                    lstm_units=[lstm_units_1, lstm_units_2],
                    dense_units=[dense_units],
                    learning_rate=learning_rate,
                    lstm_dropout=dropout_rate,
                    dense_dropout=dropout_rate,
                    epochs=50,  # Reduced for optimization
                    patience=5
                )
                
                # Create and train model
                model = CNNLSTMModel(config)
                model.build_model((X_train.shape[1], X_train.shape[2]))
                
                # Train with early stopping
                model.train(X_train, y_train, X_val, y_val)
                
                # Evaluate on validation set
                val_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, val_pred)
                
                return mse
            
            # Create study
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=self.training_config.n_trials)
            
            best_params = study.best_params
            logger.info(f"CNN+LSTM hyperparameter optimization completed. Best MSE: {study.best_value:.6f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error optimizing CNN+LSTM hyperparameters: {e}")
            return {}
    
    def _optimize_random_forest_hyperparams(self, 
                                         X_train: np.ndarray, 
                                         y_train: np.ndarray,
                                         X_val: np.ndarray, 
                                         y_val: np.ndarray) -> Dict[str, Any]:
        """Optimize Random Forest hyperparameters using Optuna."""
        try:
            def objective(trial):
                # Suggest hyperparameters
                n_estimators = trial.suggest_int('n_estimators', 50, 500)
                max_depth = trial.suggest_int('max_depth', 5, 30)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
                max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                
                # Create config with suggested parameters
                config = RandomForestConfig(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    cv_folds=3  # Reduced for optimization
                )
                
                # Create and train model
                model = RandomForestModel(config)
                model.train(X_train, y_train, X_val, y_val)
                
                # Evaluate on validation set
                val_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, val_pred)
                
                return mse
            
            # Create study
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=self.training_config.n_trials)
            
            best_params = study.best_params
            logger.info(f"Random Forest hyperparameter optimization completed. Best MSE: {study.best_value:.6f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error optimizing Random Forest hyperparameters: {e}")
            return {}
    
    def cross_validate_model(self, 
                           model_name: str, 
                           X: np.ndarray, 
                           y: np.ndarray,
                           cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation for a specific model."""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            # Prepare data for cross-validation
            if model_name == 'random_forest':
                X_cv = X.reshape(X.shape[0], -1)
            else:
                X_cv = X
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_cv):
                X_train_cv = X_cv[train_idx]
                X_val_cv = X_cv[val_idx]
                y_train_cv = y[train_idx]
                y_val_cv = y[val_idx]
                
                # Train model on fold
                if model_name == 'cnn_lstm':
                    temp_model = CNNLSTMModel(self.config.cnn_lstm)
                    temp_model.build_model((X_train_cv.shape[1], X_train_cv.shape[2]))
                    temp_model.train(X_train_cv, y_train_cv, X_val_cv, y_val_cv)
                    val_pred = temp_model.predict(X_val_cv)
                elif model_name == 'random_forest':
                    temp_model = RandomForestModel(self.config.random_forest)
                    temp_model.train(X_train_cv, y_train_cv, X_val_cv, y_val_cv)
                    val_pred = temp_model.predict(X_val_cv)
                else:  # ensemble
                    val_pred = model.predict(X_val_cv)
                
                # Calculate score
                mse = mean_squared_error(y_val_cv, val_pred)
                cv_scores.append(mse)
            
            cv_results = {
                'cv_scores': cv_scores,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'cv_min': np.min(cv_scores),
                'cv_max': np.max(cv_scores)
            }
            
            logger.info(f"Cross-validation completed for {model_name}: {cv_results['cv_mean']:.6f} ± {cv_results['cv_std']:.6f}")
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation for {model_name}: {e}")
            return {}
    
    def save_models(self, base_path: str) -> bool:
        """Save all trained models."""
        try:
            os.makedirs(base_path, exist_ok=True)
            
            for model_name, model in self.models.items():
                if model_name == 'cnn_lstm':
                    model_path = os.path.join(base_path, f"{model_name}.h5")
                    model.save_model(model_path)
                elif model_name == 'random_forest':
                    model_path = os.path.join(base_path, f"{model_name}.pkl")
                    model.save_model(model_path)
                elif model_name == 'ensemble':
                    model_path = os.path.join(base_path, f"{model_name}.pkl")
                    model.save_ensemble(model_path)
            
            # Save training results
            results_path = os.path.join(base_path, "training_results.pkl")
            joblib.dump(self.training_results, results_path)
            
            logger.info(f"All models saved to {base_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self, base_path: str) -> bool:
        """Load pre-trained models."""
        try:
            # Load CNN+LSTM model
            cnn_lstm_path = os.path.join(base_path, "cnn_lstm.h5")
            if os.path.exists(cnn_lstm_path):
                self.models['cnn_lstm'] = CNNLSTMModel(self.config.cnn_lstm)
                self.models['cnn_lstm'].load_model(cnn_lstm_path)
            
            # Load Random Forest model
            rf_path = os.path.join(base_path, "random_forest.pkl")
            if os.path.exists(rf_path):
                self.models['random_forest'] = RandomForestModel(self.config.random_forest)
                self.models['random_forest'].load_model(rf_path)
            
            # Load Ensemble model
            ensemble_path = os.path.join(base_path, "ensemble.pkl")
            if os.path.exists(ensemble_path):
                self.models['ensemble'] = EnsemblePredictor(self.config.ensemble)
                self.models['ensemble'].load_ensemble(ensemble_path)
            
            # Load training results
            results_path = os.path.join(base_path, "training_results.pkl")
            if os.path.exists(results_path):
                self.training_results = joblib.load(results_path)
            
            logger.info(f"Models loaded from {base_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_best_model(self, metric: str = 'direction_accuracy') -> Tuple[str, Any]:
        """Get the best performing model based on specified metric."""
        try:
            best_model_name = None
            best_score = -np.inf if metric in ['direction_accuracy', 'r2'] else np.inf
            
            for model_name, results in self.training_results.items():
                if 'test_results' in results:
                    test_metrics = results['test_results']
                    if metric in test_metrics:
                        score = test_metrics[metric]
                        
                        if metric in ['direction_accuracy', 'r2']:
                            if score > best_score:
                                best_score = score
                                best_model_name = model_name
                        else:  # MSE, MAE
                            if score < best_score:
                                best_score = score
                                best_model_name = model_name
            
            if best_model_name:
                return best_model_name, self.models[best_model_name]
            else:
                logger.warning(f"No model found with metric {metric}")
                return None, None
                
        except Exception as e:
            logger.error(f"Error getting best model: {e}")
            return None, None
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training results."""
        try:
            summary = {
                'models_trained': list(self.models.keys()),
                'training_config': self.training_config.__dict__,
                'model_configs': {
                    'cnn_lstm': self.config.cnn_lstm.__dict__,
                    'random_forest': self.config.random_forest.__dict__,
                    'ensemble': self.config.ensemble.__dict__
                }
            }
            
            # Add performance metrics if available
            if self.training_results:
                summary['performance_summary'] = {}
                for model_name, results in self.training_results.items():
                    if 'test_results' in results:
                        summary['performance_summary'][model_name] = results['test_results']
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting training summary: {e}")
            return {}


# Utility functions
def train_trading_models(features: pd.DataFrame, 
                        target: pd.Series,
                        config: ModelConfig,
                        optimize_hyperparams: bool = True) -> ModelTrainer:
    """Quick function to train all trading models."""
    trainer = ModelTrainer(config)
    trainer.train_all_models(features, target, optimize_hyperparams)
    return trainer


def load_trained_models(base_path: str, config: ModelConfig) -> ModelTrainer:
    """Load pre-trained models."""
    trainer = ModelTrainer(config)
    trainer.load_models(base_path)
    return trainer


if __name__ == "__main__":
    # Example usage
    from config.model_config import ModelConfig
    import yfinance as yf
    
    # Create configuration
    config = ModelConfig()
    
    # Download sample data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="2y")
    
    # Create features
    features = pd.DataFrame({
        'close': df['Close'],
        'volume': df['Volume'],
        'sma_20': df['Close'].rolling(20).mean(),
        'rsi': df['Close'].rolling(14).apply(lambda x: 100 - (100 / (1 + x.pct_change().mean())))
    }).dropna()
    
    # Create target
    target = features['close'].pct_change().shift(-1)
    
    # Train models
    trainer = train_trading_models(features, target, config, optimize_hyperparams=False)
    
    print("Training completed!")
    print(f"Models trained: {trainer.get_training_summary()['models_trained']}")
    
    # Get best model
    best_model_name, best_model = trainer.get_best_model('direction_accuracy')
    print(f"Best model: {best_model_name}")
    
    # Save models
    trainer.save_models("models/trained_models")
    print("Models saved successfully!")
