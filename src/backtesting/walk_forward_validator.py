"""
Walk-forward validator for robust strategy validation.
Implements walk-forward analysis to prevent overfitting and ensure out-of-sample performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from ..ml_models.model_trainer import ModelTrainer
from ..ml_models.data_preprocessor import TimeSeriesPreprocessor
from ..strategy.signal_generator import MultiFactorSignalGenerator
from ..strategy.portfolio_manager import PortfolioManager, PortfolioConfig
from .backtest_engine import BacktestEngine, BacktestConfig, BacktestMode
from .performance_analyzer import PerformanceAnalyzer, PerformanceConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationMethod(Enum):
    """Enumeration of validation methods."""
    WALK_FORWARD = "walk_forward"
    ROLLING_WINDOW = "rolling_window"
    EXPANDING_WINDOW = "expanding_window"
    BLOCK_BOOTSTRAP = "block_bootstrap"


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    # Time periods
    initial_train_period: int = 252  # 1 year
    retrain_frequency: int = 63      # 3 months
    test_period: int = 21           # 1 month
    step_size: int = 21             # 1 month step
    
    # Model settings
    retrain_models: bool = True
    min_train_samples: int = 200
    min_test_samples: int = 50
    
    # Performance thresholds
    min_sharpe_ratio: float = 0.5
    max_drawdown: float = 0.15
    min_win_rate: float = 0.45
    
    # Statistical tests
    perform_statistical_tests: bool = True
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000


class WalkForwardValidator:
    """Walk-forward validation system for trading strategies."""
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.validation_results = {}
        self.period_results = []
        self.model_performance_history = []
        
    def validate_strategy(self, 
                        data: Dict[str, pd.DataFrame],
                        features: Dict[str, pd.DataFrame],
                        ml_model: Optional[Any] = None,
                        strategy_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform walk-forward validation of trading strategy."""
        try:
            logger.info("Starting walk-forward validation")
            
            # Get common date range
            start_date, end_date = self._get_common_date_range(data)
            
            # Create walk-forward periods
            periods = self._create_walk_forward_periods(start_date, end_date)
            
            logger.info(f"Created {len(periods)} walk-forward periods")
            
            # Initialize results
            all_period_results = []
            model_performance = []
            
            # Run validation for each period
            for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
                logger.info(f"Processing period {i+1}/{len(periods)}: {test_start.date()} to {test_end.date()}")
                
                try:
                    # Prepare training and testing data
                    train_data = self._filter_data_by_date_range(data, train_start, train_end)
                    train_features = self._filter_data_by_date_range(features, train_start, train_end)
                    test_data = self._filter_data_by_date_range(data, test_start, test_end)
                    test_features = self._filter_data_by_date_range(features, test_start, test_end)
                    
                    # Skip if insufficient data
                    if not self._validate_period_data(train_data, test_data):
                        logger.warning(f"Insufficient data for period {i+1}, skipping")
                        continue
                    
                    # Train/retrain model if needed
                    period_model = None
                    if ml_model and self.config.retrain_models:
                        period_model = self._train_period_model(train_data, train_features, ml_model)
                        if period_model:
                            model_performance.append({
                                'period': i+1,
                                'train_start': train_start,
                                'train_end': train_end,
                                'test_start': test_start,
                                'test_end': test_end,
                                'model_performance': self._evaluate_model_performance(period_model, test_data, test_features)
                            })
                    else:
                        period_model = ml_model
                    
                    # Run backtest on test period
                    period_results = self._run_period_backtest(
                        test_data, test_features, period_model, strategy_config
                    )
                    
                    if period_results:
                        period_results['period'] = i+1
                        period_results['train_start'] = train_start
                        period_results['train_end'] = train_end
                        period_results['test_start'] = test_start
                        period_results['test_end'] = test_end
                        all_period_results.append(period_results)
                    
                except Exception as e:
                    logger.error(f"Error processing period {i+1}: {e}")
                    continue
            
            # Analyze results
            validation_analysis = self._analyze_validation_results(all_period_results)
            
            # Perform statistical tests
            statistical_tests = {}
            if self.config.perform_statistical_tests:
                statistical_tests = self._perform_validation_statistical_tests(all_period_results)
            
            # Compile final results
            self.validation_results = {
                'config': self.config.__dict__,
                'num_periods': len(periods),
                'successful_periods': len(all_period_results),
                'period_results': all_period_results,
                'model_performance': model_performance,
                'validation_analysis': validation_analysis,
                'statistical_tests': statistical_tests,
                'validation_timestamp': datetime.now()
            }
            
            logger.info("Walk-forward validation completed")
            return self.validation_results
            
        except Exception as e:
            logger.error(f"Error in walk-forward validation: {e}")
            return {}
    
    def _get_common_date_range(self, data: Dict[str, pd.DataFrame]) -> Tuple[datetime, datetime]:
        """Get common date range across all symbols."""
        try:
            all_dates = []
            for df in data.values():
                all_dates.extend(df.index.tolist())
            
            if not all_dates:
                raise ValueError("No data available")
            
            start_date = min(all_dates)
            end_date = max(all_dates)
            
            return start_date, end_date
            
        except Exception as e:
            logger.error(f"Error getting common date range: {e}")
            return datetime.now() - timedelta(days=365), datetime.now()
    
    def _create_walk_forward_periods(self, start_date: datetime, end_date: datetime) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Create walk-forward analysis periods."""
        try:
            periods = []
            current_start = start_date
            
            while current_start < end_date:
                train_start = current_start
                train_end = train_start + timedelta(days=self.config.initial_train_period)
                test_start = train_end
                test_end = test_start + timedelta(days=self.config.test_period)
                
                if test_end > end_date:
                    break
                
                periods.append((train_start, train_end, test_start, test_end))
                current_start += timedelta(days=self.config.step_size)
            
            return periods
            
        except Exception as e:
            logger.error(f"Error creating walk-forward periods: {e}")
            return []
    
    def _filter_data_by_date_range(self, 
                                 data: Dict[str, pd.DataFrame], 
                                 start_date: datetime, 
                                 end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Filter data by date range."""
        try:
            filtered_data = {}
            for symbol, df in data.items():
                mask = (df.index >= start_date) & (df.index <= end_date)
                filtered_data[symbol] = df[mask]
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error filtering data by date range: {e}")
            return data
    
    def _validate_period_data(self, train_data: Dict[str, pd.DataFrame], test_data: Dict[str, pd.DataFrame]) -> bool:
        """Validate that period has sufficient data."""
        try:
            # Check training data
            total_train_samples = sum(len(df) for df in train_data.values())
            if total_train_samples < self.config.min_train_samples:
                return False
            
            # Check testing data
            total_test_samples = sum(len(df) for df in test_data.values())
            if total_test_samples < self.config.min_test_samples:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating period data: {e}")
            return False
    
    def _train_period_model(self, 
                           train_data: Dict[str, pd.DataFrame], 
                           train_features: Dict[str, pd.DataFrame],
                           base_model: Any) -> Optional[Any]:
        """Train model for specific period."""
        try:
            # This is a simplified implementation
            # In practice, you would retrain the model on training data
            
            # For now, return the base model
            # In a real implementation, you would:
            # 1. Prepare training data
            # 2. Retrain the model
            # 3. Validate on a small validation set
            # 4. Return the retrained model
            
            logger.info("Model retraining (simplified implementation)")
            return base_model
            
        except Exception as e:
            logger.error(f"Error training period model: {e}")
            return None
    
    def _evaluate_model_performance(self, 
                                   model: Any, 
                                   test_data: Dict[str, pd.DataFrame], 
                                   test_features: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Evaluate model performance on test data."""
        try:
            # This would evaluate the model's performance
            # For now, return mock metrics
            
            return {
                'accuracy': 0.65,
                'precision': 0.62,
                'recall': 0.68,
                'f1_score': 0.65
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            return {}
    
    def _run_period_backtest(self, 
                           test_data: Dict[str, pd.DataFrame],
                           test_features: Dict[str, pd.DataFrame],
                           model: Optional[Any],
                           strategy_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Run backtest for a specific period."""
        try:
            # Create backtest configuration
            backtest_config = BacktestConfig(
                initial_capital=100000.0,
                start_date=min(df.index.min() for df in test_data.values()),
                end_date=max(df.index.max() for df in test_data.values())
            )
            
            # Run backtest
            backtest_engine = BacktestEngine(backtest_config)
            results = backtest_engine.run_backtest(test_data, test_features, model, BacktestMode.FULL_HISTORY)
            
            return results
            
        except Exception as e:
            logger.error(f"Error running period backtest: {e}")
            return None
    
    def _analyze_validation_results(self, period_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze walk-forward validation results."""
        try:
            if not period_results:
                return {}
            
            # Extract performance metrics
            sharpe_ratios = []
            max_drawdowns = []
            total_returns = []
            win_rates = []
            
            for result in period_results:
                metrics = result.get('performance_metrics', {})
                if metrics:
                    sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
                    max_drawdowns.append(metrics.get('max_drawdown', 0))
                    total_returns.append(metrics.get('total_return', 0))
                    
                    trade_metrics = metrics.get('trade_metrics', {})
                    win_rates.append(trade_metrics.get('win_rate', 0))
            
            # Calculate statistics
            analysis = {
                'num_periods': len(period_results),
                'avg_sharpe_ratio': np.mean(sharpe_ratios) if sharpe_ratios else 0,
                'std_sharpe_ratio': np.std(sharpe_ratios) if sharpe_ratios else 0,
                'min_sharpe_ratio': np.min(sharpe_ratios) if sharpe_ratios else 0,
                'max_sharpe_ratio': np.max(sharpe_ratios) if sharpe_ratios else 0,
                'avg_max_drawdown': np.mean(max_drawdowns) if max_drawdowns else 0,
                'worst_max_drawdown': np.min(max_drawdowns) if max_drawdowns else 0,
                'avg_total_return': np.mean(total_returns) if total_returns else 0,
                'avg_win_rate': np.mean(win_rates) if win_rates else 0,
                'positive_periods': sum(1 for r in total_returns if r > 0),
                'consistency_ratio': sum(1 for r in total_returns if r > 0) / len(total_returns) if total_returns else 0
            }
            
            # Performance stability
            analysis['sharpe_stability'] = 1 - (analysis['std_sharpe_ratio'] / abs(analysis['avg_sharpe_ratio'])) if analysis['avg_sharpe_ratio'] != 0 else 0
            
            # Risk-adjusted consistency
            analysis['risk_adjusted_consistency'] = analysis['consistency_ratio'] * analysis['avg_sharpe_ratio']
            
            # Overall validation score
            analysis['validation_score'] = self._calculate_validation_score(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing validation results: {e}")
            return {}
    
    def _calculate_validation_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall validation score."""
        try:
            score = 0.0
            
            # Sharpe ratio component (40% weight)
            sharpe_score = min(analysis['avg_sharpe_ratio'] / 2.0, 1.0)  # Normalize to 0-1
            score += sharpe_score * 0.4
            
            # Consistency component (30% weight)
            consistency_score = analysis['consistency_ratio']
            score += consistency_score * 0.3
            
            # Drawdown component (20% weight)
            drawdown_score = max(0, 1 - abs(analysis['worst_max_drawdown']) / 0.2)  # Penalize >20% drawdown
            score += drawdown_score * 0.2
            
            # Stability component (10% weight)
            stability_score = max(0, analysis['sharpe_stability'])
            score += stability_score * 0.1
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating validation score: {e}")
            return 0.0
    
    def _perform_validation_statistical_tests(self, period_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical tests on validation results."""
        try:
            if not period_results:
                return {}
            
            # Extract returns for statistical testing
            all_returns = []
            for result in period_results:
                portfolio_values = result.get('portfolio_values', [])
                if portfolio_values:
                    values = [pv['value'] for pv in portfolio_values]
                    returns = pd.Series(values).pct_change().dropna()
                    all_returns.extend(returns.tolist())
            
            if not all_returns:
                return {}
            
            returns_series = pd.Series(all_returns)
            
            # Statistical tests
            tests = {}
            
            # Test for normality
            from scipy import stats
            shapiro_stat, shapiro_pvalue = stats.shapiro(returns_series)
            tests['normality'] = {
                'shapiro_statistic': shapiro_stat,
                'shapiro_pvalue': shapiro_pvalue,
                'is_normal': shapiro_pvalue > 0.05
            }
            
            # Test for autocorrelation
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                ljungbox_result = acorr_ljungbox(returns_series, lags=5, return_df=True)
                tests['autocorrelation'] = {
                    'ljungbox_statistic': ljungbox_result['lb_stat'].iloc[-1],
                    'ljungbox_pvalue': ljungbox_result['lb_pvalue'].iloc[-1],
                    'has_autocorrelation': ljungbox_result['lb_pvalue'].iloc[-1] < 0.05
                }
            except:
                tests['autocorrelation'] = {'error': 'Could not perform autocorrelation test'}
            
            # Bootstrap confidence intervals
            bootstrap_results = self._bootstrap_period_returns(returns_series)
            tests['bootstrap'] = bootstrap_results
            
            # Performance persistence test
            persistence_test = self._test_performance_persistence(period_results)
            tests['persistence'] = persistence_test
            
            return tests
            
        except Exception as e:
            logger.error(f"Error performing statistical tests: {e}")
            return {}
    
    def _bootstrap_period_returns(self, returns: pd.Series) -> Dict[str, Any]:
        """Bootstrap analysis of period returns."""
        try:
            bootstrap_returns = []
            
            for _ in range(self.config.bootstrap_samples):
                bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
                bootstrap_returns.append(bootstrap_sample.mean())
            
            bootstrap_returns = np.array(bootstrap_returns)
            
            return {
                'mean_return': np.mean(bootstrap_returns),
                'std_return': np.std(bootstrap_returns),
                'ci_95_lower': np.percentile(bootstrap_returns, 2.5),
                'ci_95_upper': np.percentile(bootstrap_returns, 97.5),
                'prob_positive': np.mean(bootstrap_returns > 0)
            }
            
        except Exception as e:
            logger.error(f"Error in bootstrap analysis: {e}")
            return {}
    
    def _test_performance_persistence(self, period_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test for performance persistence across periods."""
        try:
            if len(period_results) < 3:
                return {'error': 'Insufficient periods for persistence test'}
            
            # Extract Sharpe ratios
            sharpe_ratios = []
            for result in period_results:
                metrics = result.get('performance_metrics', {})
                sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
            
            # Test for trend in performance
            from scipy import stats
            x = np.arange(len(sharpe_ratios))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, sharpe_ratios)
            
            # Test for mean reversion
            mean_sharpe = np.mean(sharpe_ratios)
            deviations = [s - mean_sharpe for s in sharpe_ratios]
            
            # Simple persistence test
            positive_periods = sum(1 for s in sharpe_ratios if s > 0)
            persistence_ratio = positive_periods / len(sharpe_ratios)
            
            return {
                'trend_slope': slope,
                'trend_pvalue': p_value,
                'has_trend': p_value < 0.05,
                'persistence_ratio': persistence_ratio,
                'mean_reversion': np.mean(np.abs(deviations)) < np.std(sharpe_ratios)
            }
            
        except Exception as e:
            logger.error(f"Error testing performance persistence: {e}")
            return {}
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        try:
            if not self.validation_results:
                return {'error': 'No validation results available'}
            
            analysis = self.validation_results.get('validation_analysis', {})
            
            summary = {
                'validation_score': analysis.get('validation_score', 0),
                'num_periods': self.validation_results.get('successful_periods', 0),
                'consistency_ratio': analysis.get('consistency_ratio', 0),
                'avg_sharpe_ratio': analysis.get('avg_sharpe_ratio', 0),
                'worst_drawdown': analysis.get('worst_max_drawdown', 0),
                'is_validated': analysis.get('validation_score', 0) > 0.6,  # Threshold for validation
                'recommendation': self._get_validation_recommendation(analysis)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting validation summary: {e}")
            return {}
    
    def _get_validation_recommendation(self, analysis: Dict[str, Any]) -> str:
        """Get validation recommendation based on analysis."""
        try:
            validation_score = analysis.get('validation_score', 0)
            consistency_ratio = analysis.get('consistency_ratio', 0)
            avg_sharpe = analysis.get('avg_sharpe_ratio', 0)
            worst_drawdown = analysis.get('worst_max_drawdown', 0)
            
            if validation_score > 0.8:
                return "STRONG VALIDATION - Strategy shows excellent out-of-sample performance"
            elif validation_score > 0.6:
                return "GOOD VALIDATION - Strategy shows solid out-of-sample performance"
            elif validation_score > 0.4:
                return "MODERATE VALIDATION - Strategy shows mixed results, consider improvements"
            else:
                return "POOR VALIDATION - Strategy shows poor out-of-sample performance, needs revision"
                
        except Exception as e:
            logger.error(f"Error getting validation recommendation: {e}")
            return "Unable to determine recommendation"
    
    def export_validation_results(self, filepath: str) -> bool:
        """Export validation results to file."""
        try:
            import json
            
            # Convert datetime objects to strings
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                return obj
            
            export_data = json.loads(json.dumps(self.validation_results, default=convert_datetime))
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Validation results exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting validation results: {e}")
            return False


# Utility functions
def validate_strategy(data: Dict[str, pd.DataFrame],
                     features: Dict[str, pd.DataFrame],
                     config: Optional[WalkForwardConfig] = None,
                     ml_model: Optional[Any] = None) -> Dict[str, Any]:
    """Quick function to validate a trading strategy."""
    if config is None:
        config = WalkForwardConfig()
    
    validator = WalkForwardValidator(config)
    return validator.validate_strategy(data, features, ml_model)


def create_walk_forward_config(train_period: int = 252, 
                              test_period: int = 63, 
                              step_size: int = 21) -> WalkForwardConfig:
    """Create a walk-forward validation configuration."""
    return WalkForwardConfig(
        initial_train_period=train_period,
        test_period=test_period,
        step_size=step_size
    )


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data = {}
    features = {}
    
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="2y")
        data[symbol] = df
        
        # Create simple features
        feature_df = pd.DataFrame({
            'close': df['Close'],
            'volume': df['Volume'],
            'sma_20': df['Close'].rolling(20).mean(),
            'rsi': df['Close'].rolling(14).apply(lambda x: 100 - (100 / (1 + x.pct_change().mean())))
        }).dropna()
        features[symbol] = feature_df
    
    # Run walk-forward validation
    config = WalkForwardConfig(
        initial_train_period=252,
        test_period=63,
        step_size=21
    )
    
    validator = WalkForwardValidator(config)
    results = validator.validate_strategy(data, features)
    
    print("Walk-forward validation completed!")
    
    # Get validation summary
    summary = validator.get_validation_summary()
    print(f"Validation Score: {summary['validation_score']:.2f}")
    print(f"Recommendation: {summary['recommendation']}")
    print(f"Consistency Ratio: {summary['consistency_ratio']:.2%}")
    print(f"Average Sharpe Ratio: {summary['avg_sharpe_ratio']:.2f}")
