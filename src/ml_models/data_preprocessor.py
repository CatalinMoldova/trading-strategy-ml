"""
Data preprocessor for machine learning models.
Handles time series data preparation, feature scaling, and sequence generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """Preprocessor for time series data used in ML models."""
    
    def __init__(self, 
                 sequence_length: int = 60,
                 prediction_horizon: int = 1,
                 test_size: float = 0.2,
                 validation_size: float = 0.1,
                 scaler_type: str = 'standard'):
        
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.test_size = test_size
        self.validation_size = validation_size
        self.scaler_type = scaler_type
        
        # Initialize scalers
        self.feature_scaler = self._get_scaler(scaler_type)
        self.target_scaler = self._get_scaler(scaler_type)
        
        # Store fitted scalers
        self.is_fitted = False
        
    def _get_scaler(self, scaler_type: str):
        """Get appropriate scaler based on type."""
        if scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'minmax':
            return MinMaxScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the input data."""
        try:
            df_clean = df.copy()
            
            # Remove rows with all NaN values
            df_clean = df_clean.dropna(how='all')
            
            # Handle infinite values
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill missing values
            df_clean = df_clean.fillna(method='ffill')
            
            # Backward fill any remaining NaN values
            df_clean = df_clean.fillna(method='bfill')
            
            # Remove any remaining rows with NaN values
            df_clean = df_clean.dropna()
            
            logger.info(f"Data cleaned: {len(df)} -> {len(df_clean)} rows")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return df
    
    def create_sequences(self, 
                        features: pd.DataFrame, 
                        target: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        try:
            # Ensure data is aligned
            common_index = features.index.intersection(target.index)
            features_aligned = features.loc[common_index]
            target_aligned = target.loc[common_index]
            
            # Convert to numpy arrays
            feature_values = features_aligned.values
            target_values = target_aligned.values
            
            # Create sequences
            X, y = [], []
            
            for i in range(self.sequence_length, len(feature_values) - self.prediction_horizon + 1):
                X.append(feature_values[i-self.sequence_length:i])
                y.append(target_values[i + self.prediction_horizon - 1])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Created sequences: X shape {X.shape}, y shape {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return np.array([]), np.array([])
    
    def scale_features(self, 
                      X_train: np.ndarray, 
                      X_val: Optional[np.ndarray] = None,
                      X_test: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
        """Scale features using the fitted scaler."""
        try:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before scaling")
            
            # Scale training data
            X_train_scaled = self.feature_scaler.transform(
                X_train.reshape(-1, X_train.shape[-1])
            ).reshape(X_train.shape)
            
            results = [X_train_scaled]
            
            # Scale validation data if provided
            if X_val is not None:
                X_val_scaled = self.feature_scaler.transform(
                    X_val.reshape(-1, X_val.shape[-1])
                ).reshape(X_val.shape)
                results.append(X_val_scaled)
            
            # Scale test data if provided
            if X_test is not None:
                X_test_scaled = self.feature_scaler.transform(
                    X_test.reshape(-1, X_test.shape[-1])
                ).reshape(X_test.shape)
                results.append(X_test_scaled)
            
            logger.info("Features scaled successfully")
            return tuple(results)
            
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            return (X_train,)
    
    def scale_targets(self, 
                     y_train: np.ndarray, 
                     y_val: Optional[np.ndarray] = None,
                     y_test: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ...]:
        """Scale targets using the fitted scaler."""
        try:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before scaling")
            
            # Scale training targets
            y_train_scaled = self.target_scaler.transform(y_train.reshape(-1, 1)).flatten()
            
            results = [y_train_scaled]
            
            # Scale validation targets if provided
            if y_val is not None:
                y_val_scaled = self.target_scaler.transform(y_val.reshape(-1, 1)).flatten()
                results.append(y_val_scaled)
            
            # Scale test targets if provided
            if y_test is not None:
                y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, 1)).flatten()
                results.append(y_test_scaled)
            
            logger.info("Targets scaled successfully")
            return tuple(results)
            
        except Exception as e:
            logger.error(f"Error scaling targets: {e}")
            return (y_train,)
    
    def inverse_scale_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform scaled targets back to original scale."""
        try:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted before inverse scaling")
            
            y_original = self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
            return y_original
            
        except Exception as e:
            logger.error(f"Error inverse scaling targets: {e}")
            return y_scaled
    
    def fit_scalers(self, 
                   X_train: np.ndarray, 
                   y_train: np.ndarray) -> None:
        """Fit scalers on training data."""
        try:
            # Fit feature scaler
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            self.feature_scaler.fit(X_train_reshaped)
            
            # Fit target scaler
            y_train_reshaped = y_train.reshape(-1, 1)
            self.target_scaler.fit(y_train_reshaped)
            
            self.is_fitted = True
            logger.info("Scalers fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting scalers: {e}")
    
    def prepare_data(self, 
                    features: pd.DataFrame, 
                    target: pd.Series,
                    split_by_date: bool = True,
                    split_date: Optional[str] = None) -> Dict[str, Any]:
        """Prepare data for model training and evaluation."""
        try:
            # Clean data
            features_clean = self.clean_data(features)
            target_clean = self.clean_data(target.to_frame()).iloc[:, 0]
            
            # Align features and target
            common_index = features_clean.index.intersection(target_clean.index)
            features_aligned = features_clean.loc[common_index]
            target_aligned = target_clean.loc[common_index]
            
            # Create sequences
            X, y = self.create_sequences(features_aligned, target_aligned)
            
            if len(X) == 0:
                logger.error("No sequences created")
                return {}
            
            # Split data
            if split_by_date and split_date:
                # Split by date
                split_idx = int(len(X) * (1 - self.test_size))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Further split training data for validation
                val_split_idx = int(len(X_train) * (1 - self.validation_size))
                X_train, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
                y_train, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
                
            else:
                # Random split
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=(self.test_size + self.validation_size), 
                    random_state=42, shuffle=False
                )
                
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=self.test_size/(self.test_size + self.validation_size),
                    random_state=42, shuffle=False
                )
            
            # Fit scalers
            self.fit_scalers(X_train, y_train)
            
            # Scale data
            X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(X_train, X_val, X_test)
            y_train_scaled, y_val_scaled, y_test_scaled = self.scale_targets(y_train, y_val, y_test)
            
            # Prepare result dictionary
            result = {
                'X_train': X_train_scaled,
                'X_val': X_val_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train_scaled,
                'y_val': y_val_scaled,
                'y_test': y_test_scaled,
                'y_train_original': y_train,
                'y_val_original': y_val,
                'y_test_original': y_test,
                'feature_names': list(features_aligned.columns),
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon
            }
            
            logger.info(f"Data prepared successfully: Train {X_train_scaled.shape}, Val {X_val_scaled.shape}, Test {X_test_scaled.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return {}
    
    def create_rolling_windows(self, 
                              features: pd.DataFrame, 
                              target: pd.Series,
                              window_size: int = 252,
                              step_size: int = 21) -> List[Dict[str, Any]]:
        """Create rolling windows for walk-forward analysis."""
        try:
            windows = []
            
            for start_idx in range(0, len(features) - window_size, step_size):
                end_idx = start_idx + window_size
                
                window_features = features.iloc[start_idx:end_idx]
                window_target = target.iloc[start_idx:end_idx]
                
                # Create sequences for this window
                X, y = self.create_sequences(window_features, window_target)
                
                if len(X) > 0:
                    windows.append({
                        'X': X,
                        'y': y,
                        'start_date': window_features.index[0],
                        'end_date': window_features.index[-1],
                        'features': window_features,
                        'target': window_target
                    })
            
            logger.info(f"Created {len(windows)} rolling windows")
            return windows
            
        except Exception as e:
            logger.error(f"Error creating rolling windows: {e}")
            return []
    
    def get_feature_importance_data(self, 
                                   features: pd.DataFrame, 
                                   target: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for feature importance analysis."""
        try:
            # Clean and align data
            features_clean = self.clean_data(features)
            target_clean = self.clean_data(target.to_frame()).iloc[:, 0]
            
            common_index = features_clean.index.intersection(target_clean.index)
            features_aligned = features_clean.loc[common_index]
            target_aligned = target_clean.loc[common_index]
            
            # Handle missing values
            imputer = SimpleImputer(strategy='mean')
            features_imputed = imputer.fit_transform(features_aligned)
            
            # Scale features
            scaler = self._get_scaler(self.scaler_type)
            features_scaled = scaler.fit_transform(features_imputed)
            
            logger.info(f"Feature importance data prepared: {features_scaled.shape}")
            return features_scaled, target_aligned.values
            
        except Exception as e:
            logger.error(f"Error preparing feature importance data: {e}")
            return np.array([]), np.array([])


class DataAugmentation:
    """Data augmentation techniques for time series data."""
    
    @staticmethod
    def add_noise(data: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to the data."""
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise
    
    @staticmethod
    def time_warp(data: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """Apply time warping to the data."""
        # Simple time warping implementation
        warped_data = data.copy()
        for i in range(len(data)):
            warp_factor = np.random.normal(1, sigma)
            warp_factor = max(0.5, min(2.0, warp_factor))  # Clamp between 0.5 and 2.0
            warped_data[i] = data[i] * warp_factor
        return warped_data
    
    @staticmethod
    def magnitude_warp(data: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """Apply magnitude warping to the data."""
        warp_factor = np.random.normal(1, sigma, data.shape)
        warp_factor = np.clip(warp_factor, 0.5, 2.0)
        return data * warp_factor
    
    @staticmethod
    def augment_dataset(X: np.ndarray, y: np.ndarray, augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Augment the dataset with various techniques."""
        augmented_X = [X]
        augmented_y = [y]
        
        for _ in range(augmentation_factor):
            # Add noise
            X_noise = DataAugmentation.add_noise(X)
            augmented_X.append(X_noise)
            augmented_y.append(y)
            
            # Time warp
            X_warp = DataAugmentation.time_warp(X)
            augmented_X.append(X_warp)
            augmented_y.append(y)
            
            # Magnitude warp
            X_mag = DataAugmentation.magnitude_warp(X)
            augmented_X.append(X_mag)
            augmented_y.append(y)
        
        return np.concatenate(augmented_X), np.concatenate(augmented_y)


# Utility functions
def prepare_training_data(features: pd.DataFrame, 
                         target: pd.Series,
                         sequence_length: int = 60,
                         test_size: float = 0.2) -> Dict[str, Any]:
    """Quick function to prepare training data."""
    preprocessor = TimeSeriesPreprocessor(
        sequence_length=sequence_length,
        test_size=test_size
    )
    
    return preprocessor.prepare_data(features, target)


def create_walk_forward_data(features: pd.DataFrame,
                           target: pd.Series,
                           window_size: int = 252,
                           step_size: int = 21) -> List[Dict[str, Any]]:
    """Create walk-forward analysis data."""
    preprocessor = TimeSeriesPreprocessor()
    return preprocessor.create_rolling_windows(features, target, window_size, step_size)


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="2y")
    
    # Create simple features
    features = pd.DataFrame({
        'close': df['Close'],
        'volume': df['Volume'],
        'sma_20': df['Close'].rolling(20).mean(),
        'rsi': df['Close'].rolling(14).apply(lambda x: 100 - (100 / (1 + x.pct_change().mean())))
    }).dropna()
    
    # Create target (next day return)
    target = features['close'].pct_change().shift(-1)
    
    # Prepare data
    preprocessor = TimeSeriesPreprocessor(sequence_length=30)
    data = preprocessor.prepare_data(features, target)
    
    if data:
        print(f"Training data shape: {data['X_train'].shape}")
        print(f"Validation data shape: {data['X_val'].shape}")
        print(f"Test data shape: {data['X_test'].shape}")
        print(f"Feature names: {data['feature_names']}")
