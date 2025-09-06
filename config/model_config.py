"""
Model configuration for machine learning components.
Defines hyperparameters, model architectures, and training settings.
"""

import os
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class CNNLSTMConfig:
    """Configuration for CNN+LSTM hybrid model."""
    
    # Data parameters
    sequence_length: int = 60  # Number of time steps to look back
    feature_dim: int = 20  # Number of features per time step
    prediction_horizon: int = 1  # Number of steps to predict ahead
    
    # CNN parameters
    cnn_filters: List[int] = None
    cnn_kernel_sizes: List[int] = None
    cnn_activation: str = 'relu'
    cnn_dropout: float = 0.2
    
    # LSTM parameters
    lstm_units: List[int] = None
    lstm_dropout: float = 0.2
    lstm_recurrent_dropout: float = 0.2
    lstm_return_sequences: bool = True
    
    # Dense layers
    dense_units: List[int] = None
    dense_activation: str = 'relu'
    dense_dropout: float = 0.3
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = 'adam'
    loss_function: str = 'mse'
    metrics: List[str] = None
    
    # Early stopping
    patience: int = 10
    min_delta: float = 0.001
    
    def __post_init__(self):
        if self.cnn_filters is None:
            self.cnn_filters = [32, 64, 128]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 3, 3]
        if self.lstm_units is None:
            self.lstm_units = [128, 64]
        if self.dense_units is None:
            self.dense_units = [64, 32]
        if self.metrics is None:
            self.metrics = ['mae', 'mape']


@dataclass
class RandomForestConfig:
    """Configuration for Random Forest model."""
    
    # Model parameters
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    max_features: str = 'sqrt'
    bootstrap: bool = True
    random_state: int = 42
    
    # Feature selection
    feature_importance_threshold: float = 0.01
    max_features_selected: int = 50
    
    # Cross-validation
    cv_folds: int = 5
    scoring: str = 'neg_mean_squared_error'


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""
    
    # Model weights
    cnn_lstm_weight: float = 0.6
    random_forest_weight: float = 0.4
    
    # Confidence thresholds
    min_confidence: float = 0.6
    max_confidence: float = 0.95
    
    # Voting method
    voting_method: str = 'weighted'  # 'weighted', 'soft', 'hard'
    
    # Performance tracking
    performance_window: int = 30  # Days to track performance
    rebalance_frequency: int = 7  # Days between rebalancing


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering."""
    
    # Technical indicators
    momentum_indicators: List[str] = None
    volatility_indicators: List[str] = None
    volume_indicators: List[str] = None
    trend_indicators: List[str] = None
    
    # Lookback periods
    short_period: int = 14
    medium_period: int = 30
    long_period: int = 60
    
    # Normalization
    normalize_features: bool = True
    normalization_method: str = 'standard'  # 'standard', 'minmax', 'robust'
    
    # Feature selection
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    
    def __post_init__(self):
        if self.momentum_indicators is None:
            self.momentum_indicators = ['RSI', 'MACD', 'ROC', 'STOCH', 'WILLR']
        if self.volatility_indicators is None:
            self.volatility_indicators = ['BBANDS', 'ATR', 'NATR', 'TRANGE']
        if self.volume_indicators is None:
            self.volume_indicators = ['OBV', 'AD', 'ADOSC', 'MFI']
        if self.trend_indicators is None:
            self.trend_indicators = ['SMA', 'EMA', 'MACD', 'ADX', 'AROON']


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Data splitting
    train_split: float = 0.7
    validation_split: float = 0.15
    test_split: float = 0.15
    
    # Cross-validation
    cv_folds: int = 5
    time_series_cv: bool = True
    
    # Hyperparameter optimization
    use_optuna: bool = True
    n_trials: int = 100
    optimization_direction: str = 'minimize'
    
    # Model persistence
    save_models: bool = True
    model_save_path: str = 'models/'
    best_model_criteria: str = 'validation_loss'
    
    # Logging
    log_level: str = 'INFO'
    log_file: str = 'training.log'
    tensorboard_logs: str = 'logs/tensorboard/'


@dataclass
class PredictionConfig:
    """Configuration for model predictions."""
    
    # Prediction settings
    batch_size: int = 32
    confidence_threshold: float = 0.6
    
    # Real-time settings
    prediction_frequency: int = 300  # seconds
    max_prediction_age: int = 600  # seconds
    
    # Output settings
    output_format: str = 'json'  # 'json', 'csv', 'parquet'
    include_confidence: bool = True
    include_feature_importance: bool = True


class ModelConfig:
    """Main configuration class for all ML models."""
    
    def __init__(self):
        self.cnn_lstm = CNNLSTMConfig()
        self.random_forest = RandomForestConfig()
        self.ensemble = EnsembleConfig()
        self.feature_engineering = FeatureEngineeringConfig()
        self.training = TrainingConfig()
        self.prediction = PredictionConfig()
        
        # Environment-specific settings
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
        
    def get_model_config(self, model_name: str) -> Any:
        """Get configuration for a specific model."""
        config_map = {
            'cnn_lstm': self.cnn_lstm,
            'random_forest': self.random_forest,
            'ensemble': self.ensemble,
            'feature_engineering': self.feature_engineering,
            'training': self.training,
            'prediction': self.prediction
        }
        return config_map.get(model_name)
    
    def update_config(self, model_name: str, **kwargs):
        """Update configuration parameters."""
        config = self.get_model_config(model_name)
        if config:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'cnn_lstm': self.cnn_lstm.__dict__,
            'random_forest': self.random_forest.__dict__,
            'ensemble': self.ensemble.__dict__,
            'feature_engineering': self.feature_engineering.__dict__,
            'training': self.training.__dict__,
            'prediction': self.prediction.__dict__,
            'environment': self.environment,
            'debug_mode': self.debug_mode
        }


# Default model configurations for different strategies
STRATEGY_CONFIGS = {
    'momentum': {
        'cnn_lstm': {
            'sequence_length': 30,
            'lstm_units': [64, 32],
            'learning_rate': 0.001
        },
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 15
        }
    },
    'mean_reversion': {
        'cnn_lstm': {
            'sequence_length': 20,
            'lstm_units': [32, 16],
            'learning_rate': 0.0005
        },
        'random_forest': {
            'n_estimators': 150,
            'max_depth': 12
        }
    },
    'volatility': {
        'cnn_lstm': {
            'sequence_length': 45,
            'lstm_units': [128, 64, 32],
            'learning_rate': 0.002
        },
        'random_forest': {
            'n_estimators': 300,
            'max_depth': 20
        }
    }
}


def get_strategy_config(strategy_name: str) -> ModelConfig:
    """Get configuration for a specific trading strategy."""
    config = ModelConfig()
    
    if strategy_name in STRATEGY_CONFIGS:
        strategy_params = STRATEGY_CONFIGS[strategy_name]
        
        for model_name, params in strategy_params.items():
            config.update_config(model_name, **params)
    
    return config


# Model performance thresholds
PERFORMANCE_THRESHOLDS = {
    'min_accuracy': 0.55,
    'min_sharpe_ratio': 1.0,
    'max_drawdown': 0.15,
    'min_win_rate': 0.52,
    'max_correlation': 0.7
}


if __name__ == "__main__":
    # Example usage
    config = ModelConfig()
    print("CNN+LSTM Configuration:")
    print(config.cnn_lstm)
    
    print("\nStrategy-specific configuration:")
    momentum_config = get_strategy_config('momentum')
    print(f"Momentum strategy sequence length: {momentum_config.cnn_lstm.sequence_length}")
