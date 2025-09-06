"""
Random Forest model for feature importance analysis and trading signal generation.
Provides interpretable predictions and feature rankings.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
import joblib
import os
from datetime import datetime

from config.model_config import RandomForestConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestModel:
    """Random Forest model for financial prediction and feature analysis."""
    
    def __init__(self, config: RandomForestConfig, task_type: str = 'regression'):
        self.config = config
        self.task_type = task_type
        self.model = None
        self.feature_selector = None
        self.feature_importance_ = None
        self.is_trained = False
        self.feature_names_ = None
        
    def build_model(self) -> Any:
        """Build the Random Forest model."""
        try:
            if self.task_type == 'regression':
                self.model = RandomForestRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    min_samples_split=self.config.min_samples_split,
                    min_samples_leaf=self.config.min_samples_leaf,
                    max_features=self.config.max_features,
                    bootstrap=self.config.bootstrap,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    verbose=1
                )
            elif self.task_type == 'classification':
                self.model = RandomForestClassifier(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    min_samples_split=self.config.min_samples_split,
                    min_samples_leaf=self.config.min_samples_leaf,
                    max_features=self.config.max_features,
                    bootstrap=self.config.bootstrap,
                    random_state=self.config.random_state,
                    n_jobs=-1,
                    verbose=1
                )
            else:
                raise ValueError(f"Unknown task type: {self.task_type}")
            
            logger.info(f"Random Forest model built for {self.task_type}")
            return self.model
            
        except Exception as e:
            logger.error(f"Error building Random Forest model: {e}")
            return None
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Train the Random Forest model."""
        try:
            if self.model is None:
                self.build_model()
            
            # Store feature names
            self.feature_names_ = feature_names
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Get feature importance
            self.feature_importance_ = self.model.feature_importances_
            
            # Create feature selector
            self.feature_selector = SelectFromModel(
                self.model, 
                threshold=self.config.feature_importance_threshold
            )
            
            # Fit feature selector
            self.feature_selector.fit(X_train, y_train)
            
            # Calculate training metrics
            train_pred = self.predict(X_train)
            train_metrics = self._calculate_metrics(y_train, train_pred, 'train')
            
            # Calculate validation metrics if provided
            val_metrics = {}
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_metrics = self._calculate_metrics(y_val, val_pred, 'val')
            
            # Cross-validation scores
            cv_scores = self._cross_validate(X_train, y_train)
            
            self.is_trained = True
            
            results = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'cv_scores': cv_scores,
                'feature_importance': self.get_feature_importance(),
                'selected_features': self.get_selected_features()
            }
            
            logger.info("Random Forest model training completed")
            return results
            
        except Exception as e:
            logger.error(f"Error training Random Forest model: {e}")
            return {}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            predictions = self.model.predict(X)
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.array([])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions (for classification tasks)."""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            if self.task_type != 'classification':
                raise ValueError("Probability predictions only available for classification tasks")
            
            probabilities = self.model.predict_proba(X)
            return probabilities
            
        except Exception as e:
            logger.error(f"Error making probability predictions: {e}")
            return np.array([])
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data."""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before evaluation")
            
            predictions = self.predict(X_test)
            metrics = self._calculate_metrics(y_test, predictions, 'test')
            
            logger.info(f"Model evaluation completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        try:
            if self.task_type == 'regression':
                metrics = {
                    f'{prefix}_mse': mean_squared_error(y_true, y_pred),
                    f'{prefix}_mae': mean_absolute_error(y_true, y_pred),
                    f'{prefix}_r2': r2_score(y_true, y_pred),
                    f'{prefix}_direction_accuracy': np.mean(np.sign(y_true) == np.sign(y_pred))
                }
            else:  # classification
                metrics = {
                    f'{prefix}_accuracy': accuracy_score(y_true, y_pred),
                    f'{prefix}_direction_accuracy': accuracy_score(y_true, y_pred)
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def _cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform cross-validation."""
        try:
            cv_scores = cross_val_score(
                self.model, X, y, 
                cv=self.config.cv_folds, 
                scoring=self.config.scoring,
                n_jobs=-1
            )
            
            return {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {}
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance rankings."""
        try:
            if self.feature_importance_ is None:
                logger.warning("Model not trained yet, no feature importance available")
                return pd.DataFrame()
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names_ or [f'feature_{i}' for i in range(len(self.feature_importance_))],
                'importance': self.feature_importance_
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return pd.DataFrame()
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        try:
            if self.feature_selector is None:
                return []
            
            selected_indices = self.feature_selector.get_support(indices=True)
            selected_features = [self.feature_names_[i] for i in selected_indices] if self.feature_names_ else []
            
            return selected_features
            
        except Exception as e:
            logger.error(f"Error getting selected features: {e}")
            return []
    
    def get_top_features(self, n: int = 20) -> List[str]:
        """Get top N most important features."""
        try:
            importance_df = self.get_feature_importance()
            if importance_df.empty:
                return []
            
            return importance_df.head(n)['feature'].tolist()
            
        except Exception as e:
            logger.error(f"Error getting top features: {e}")
            return []
    
    def plot_feature_importance(self, n: int = 20, save_path: Optional[str] = None):
        """Plot feature importance."""
        try:
            import matplotlib.pyplot as plt
            
            importance_df = self.get_feature_importance()
            if importance_df.empty:
                logger.warning("No feature importance data available")
                return
            
            # Get top N features
            top_features = importance_df.head(n)
            
            # Create plot
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {n} Feature Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
    
    def hyperparameter_tuning(self, 
                             X_train: np.ndarray, 
                             y_train: np.ndarray,
                             method: str = 'random',
                             n_iter: int = 100) -> Dict[str, Any]:
        """Perform hyperparameter tuning."""
        try:
            if self.model is None:
                self.build_model()
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Choose search method
            if method == 'grid':
                search = GridSearchCV(
                    self.model, param_grid, 
                    cv=self.config.cv_folds, 
                    scoring=self.config.scoring,
                    n_jobs=-1, verbose=1
                )
            else:  # random
                search = RandomizedSearchCV(
                    self.model, param_grid, 
                    n_iter=n_iter,
                    cv=self.config.cv_folds, 
                    scoring=self.config.scoring,
                    n_jobs=-1, verbose=1, random_state=42
                )
            
            # Perform search
            search.fit(X_train, y_train)
            
            # Update model with best parameters
            self.model = search.best_estimator_
            
            results = {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'cv_results': search.cv_results_
            }
            
            logger.info(f"Hyperparameter tuning completed. Best score: {search.best_score_}")
            return results
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {e}")
            return {}
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model."""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before saving")
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            joblib.dump(self.model, filepath)
            
            # Save additional information
            model_info = {
                'feature_names': self.feature_names_,
                'feature_importance': self.feature_importance_,
                'selected_features': self.get_selected_features(),
                'config': self.config.__dict__,
                'task_type': self.task_type
            }
            
            info_path = filepath.replace('.pkl', '_info.pkl')
            joblib.dump(model_info, info_path)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a pre-trained model."""
        try:
            # Load model
            self.model = joblib.load(filepath)
            
            # Load additional information
            info_path = filepath.replace('.pkl', '_info.pkl')
            if os.path.exists(info_path):
                model_info = joblib.load(info_path)
                self.feature_names_ = model_info.get('feature_names')
                self.feature_importance_ = model_info.get('feature_importance')
                self.task_type = model_info.get('task_type', 'regression')
            
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.is_trained:
            return {"status": "Model not trained"}
        
        info = {
            'task_type': self.task_type,
            'n_features': len(self.feature_importance_) if self.feature_importance_ is not None else 0,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'selected_features_count': len(self.get_selected_features()),
            'top_features': self.get_top_features(10)
        }
        
        return info


class RandomForestEnsemble:
    """Ensemble of Random Forest models for improved predictions."""
    
    def __init__(self, models: List[RandomForestModel]):
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
    
    def get_ensemble_feature_importance(self) -> pd.DataFrame:
        """Get ensemble feature importance."""
        try:
            all_importance = []
            
            for model in self.models:
                if model.is_trained and model.feature_importance_ is not None:
                    all_importance.append(model.feature_importance_)
            
            if not all_importance:
                return pd.DataFrame()
            
            # Average feature importance across models
            avg_importance = np.mean(all_importance, axis=0)
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': self.models[0].feature_names_ or [f'feature_{i}' for i in range(len(avg_importance))],
                'importance': avg_importance
            })
            
            return importance_df.sort_values('importance', ascending=False)
            
        except Exception as e:
            logger.error(f"Error getting ensemble feature importance: {e}")
            return pd.DataFrame()


# Utility functions
def create_random_forest_model(config: RandomForestConfig, task_type: str = 'regression') -> RandomForestModel:
    """Create a Random Forest model."""
    model = RandomForestModel(config, task_type)
    model.build_model()
    return model


def train_random_forest_model(model: RandomForestModel,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: np.ndarray = None,
                            y_val: np.ndarray = None,
                            feature_names: List[str] = None) -> Dict[str, Any]:
    """Train a Random Forest model."""
    return model.train(X_train, y_train, X_val, y_val, feature_names)


if __name__ == "__main__":
    # Example usage
    from config.model_config import RandomForestConfig
    
    # Create configuration
    config = RandomForestConfig()
    
    # Create model
    model = RandomForestModel(config, 'regression')
    
    # Example training data
    X_train = np.random.randn(1000, 50)
    y_train = np.random.randn(1000)
    X_val = np.random.randn(200, 50)
    y_val = np.random.randn(200)
    feature_names = [f'feature_{i}' for i in range(50)]
    
    # Train model
    results = model.train(X_train, y_train, X_val, y_val, feature_names)
    print("Training completed!")
    print(f"Training metrics: {results['train_metrics']}")
    print(f"Validation metrics: {results['val_metrics']}")
    
    # Get feature importance
    importance_df = model.get_feature_importance()
    print(f"Top 10 features: {importance_df.head(10)}")
    
    # Make predictions
    X_test = np.random.randn(100, 50)
    predictions = model.predict(X_test)
    print(f"Predictions shape: {predictions.shape}")
    
    # Evaluate model
    y_test = np.random.randn(100)
    metrics = model.evaluate(X_test, y_test)
    print(f"Test metrics: {metrics}")
    
    # Get model info
    info = model.get_model_info()
    print(f"Model info: {info}")
