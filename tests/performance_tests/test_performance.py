"""
Performance tests for the trading strategy ML system.
Tests system performance, scalability, and resource usage.
"""

import unittest
import pandas as pd
import numpy as np
import time
import psutil
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import components to test
from src.data_pipeline.market_data_collector import MarketDataCollector
from src.data_pipeline.indicator_engine import IndicatorEngine
from src.data_pipeline.feature_engineer import FeatureEngineer
from src.data_pipeline.data_storage import DataStorage

from src.ml_models.data_preprocessor import DataPreprocessor
from src.ml_models.cnn_lstm_model import CNNLSTMModel
from src.ml_models.random_forest_model import RandomForestModel
from src.ml_models.ensemble_predictor import EnsemblePredictor

from src.strategy.signal_generator import SignalGenerator
from src.strategy.position_sizer import PositionSizer
from src.strategy.risk_manager import RiskManager

from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.performance_analyzer import PerformanceAnalyzer
from src.backtesting.statistical_validator import StatisticalValidator


class PerformanceTestBase(unittest.TestCase):
    """Base class for performance tests with common utilities."""
    
    def setUp(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    def tearDown(self):
        self.end_time = time.time()
        self.end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time = self.end_time - self.start_time
        memory_usage = self.end_memory - self.start_memory
        
        print(f"\nTest: {self._testMethodName}")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Memory usage: {memory_usage:.2f} MB")
    
    def generate_large_dataset(self, rows: int, symbols: list) -> dict:
        """Generate large dataset for performance testing."""
        data = {}
        
        for symbol in symbols:
            np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
            
            df = pd.DataFrame({
                'open': np.random.rand(rows) * 100 + 100,
                'high': np.random.rand(rows) * 100 + 105,
                'low': np.random.rand(rows) * 100 + 95,
                'close': np.random.rand(rows) * 100 + 100,
                'volume': np.random.randint(1000, 2000, rows)
            }, index=pd.date_range('2020-01-01', periods=rows))
            
            data[symbol] = df
        
        return data


class TestDataPipelinePerformance(PerformanceTestBase):
    """Performance tests for data pipeline components."""
    
    def test_indicator_engine_performance(self):
        """Test IndicatorEngine performance with large datasets."""
        engine = IndicatorEngine()
        
        # Generate large dataset
        large_data = self.generate_large_dataset(10000, ['AAPL'])
        df = large_data['AAPL']
        
        # Test performance
        start_time = time.time()
        result = engine.calculate_all_indicators(df.copy())
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        self.assertLess(execution_time, 10.0, "Indicator calculation should complete within 10 seconds")
        self.assertFalse(result.empty)
        self.assertIn('RSI', result.columns)
        self.assertIn('MACD', result.columns)
        
        print(f"Indicator calculation time: {execution_time:.2f} seconds for {len(df)} rows")
    
    def test_feature_engineer_performance(self):
        """Test FeatureEngineer performance with large datasets."""
        engineer = FeatureEngineer()
        
        # Generate large dataset
        large_data = self.generate_large_dataset(10000, ['AAPL'])
        df = large_data['AAPL']
        
        # Apply indicators first
        df_with_indicators = IndicatorEngine().calculate_all_indicators(df.copy())
        
        # Test basic feature generation
        start_time = time.time()
        result = engineer.generate_basic_features(df_with_indicators.copy())
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        self.assertLess(execution_time, 5.0, "Basic feature generation should complete within 5 seconds")
        self.assertIn('daily_return', result.columns)
        self.assertIn('log_return', result.columns)
        
        print(f"Basic feature generation time: {execution_time:.2f} seconds for {len(df)} rows")
    
    def test_cross_sectional_normalization_performance(self):
        """Test cross-sectional normalization performance."""
        engineer = FeatureEngineer()
        
        # Generate large dataset for multiple symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        large_data = self.generate_large_dataset(5000, symbols)
        
        # Apply indicators
        for symbol in symbols:
            large_data[symbol] = IndicatorEngine().calculate_all_indicators(large_data[symbol])
            large_data[symbol] = engineer.generate_basic_features(large_data[symbol])
        
        # Test cross-sectional normalization
        feature_cols = ['close', 'RSI', 'MACD']
        
        start_time = time.time()
        normalized_data = engineer.cross_sectional_normalize(large_data, feature_cols)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        self.assertLess(execution_time, 15.0, "Cross-sectional normalization should complete within 15 seconds")
        self.assertEqual(len(normalized_data), len(symbols))
        
        print(f"Cross-sectional normalization time: {execution_time:.2f} seconds for {len(symbols)} symbols")
    
    def test_ml_data_preparation_performance(self):
        """Test ML data preparation performance."""
        engineer = FeatureEngineer()
        
        # Generate large dataset
        large_data = self.generate_large_dataset(10000, ['AAPL'])
        df = large_data['AAPL']
        
        # Apply full pipeline
        df_processed = IndicatorEngine().calculate_all_indicators(df.copy())
        df_processed = engineer.generate_basic_features(df_processed)
        df_processed = engineer.generate_advanced_features(df_processed)
        df_processed['next_day_return'] = df_processed['close'].pct_change().shift(-1)
        
        # Test ML data preparation
        start_time = time.time()
        ml_data = engineer.prepare_ml_data(df_processed.dropna(), target_column='next_day_return', time_steps=60)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        self.assertLess(execution_time, 8.0, "ML data preparation should complete within 8 seconds")
        self.assertIn('X', ml_data)
        self.assertIn('y', ml_data)
        
        print(f"ML data preparation time: {execution_time:.2f} seconds for {len(df)} rows")


class TestMLModelsPerformance(PerformanceTestBase):
    """Performance tests for ML models."""
    
    def test_cnn_lstm_model_performance(self):
        """Test CNN-LSTM model performance."""
        # Generate sample data
        time_steps = 60
        n_features = 20
        num_samples = 1000
        
        X_dummy = np.random.rand(num_samples, time_steps, n_features)
        y_dummy = np.random.rand(num_samples, 1) * 0.1
        
        # Split data
        split_idx = int(num_samples * 0.8)
        X_train, X_val = X_dummy[:split_idx], X_dummy[split_idx:]
        y_train, y_val = y_dummy[:split_idx], y_dummy[split_idx:]
        
        # Test model creation and training
        start_time = time.time()
        
        model = CNNLSTMModel(time_steps=time_steps, n_features=n_features)
        
        # Mock training to avoid actual training time
        with patch.object(model.model, 'fit') as mock_fit:
            mock_fit.return_value = Mock()
            history = model.train(X_train, y_train, X_val, y_val)
        
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        self.assertLess(execution_time, 5.0, "CNN-LSTM model creation should complete within 5 seconds")
        self.assertIsNotNone(model.model)
        
        print(f"CNN-LSTM model creation time: {execution_time:.2f} seconds")
    
    def test_random_forest_model_performance(self):
        """Test Random Forest model performance."""
        # Generate sample data
        n_features = 100
        num_samples = 10000
        
        X_dummy = np.random.rand(num_samples, n_features)
        y_dummy = np.random.rand(num_samples) * 0.1
        
        # Test model creation and training
        start_time = time.time()
        
        model = RandomForestModel()
        model.train(X_dummy, y_dummy)
        
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        self.assertLess(execution_time, 30.0, "Random Forest training should complete within 30 seconds")
        self.assertIsNotNone(model.model)
        
        print(f"Random Forest training time: {execution_time:.2f} seconds for {num_samples} samples")
    
    def test_ensemble_predictor_performance(self):
        """Test EnsemblePredictor performance."""
        predictor = EnsemblePredictor()
        
        # Generate large predictions
        num_predictions = 10000
        predictions = {
            'cnn_lstm': np.random.randn(num_predictions, 1) * 0.02,
            'random_forest': np.random.randn(num_predictions, 1) * 0.015
        }
        
        # Test prediction combination
        start_time = time.time()
        combined = predictor.combine_predictions(predictions)
        signals, confidence = predictor.generate_signal(combined)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        self.assertLess(execution_time, 2.0, "Ensemble prediction should complete within 2 seconds")
        self.assertEqual(len(combined), num_predictions)
        self.assertEqual(len(signals), num_predictions)
        
        print(f"Ensemble prediction time: {execution_time:.2f} seconds for {num_predictions} predictions")


class TestStrategyPerformance(PerformanceTestBase):
    """Performance tests for strategy components."""
    
    def test_signal_generation_performance(self):
        """Test signal generation performance."""
        generator = SignalGenerator()
        
        # Generate large number of signals
        num_signals = 10000
        
        # Test ML signal generation
        start_time = time.time()
        
        for i in range(num_signals):
            ml_prediction = np.random.randn() * 0.02
            ml_confidence = abs(np.random.randn() * 0.01)
            signal, confidence = generator.generate_ml_signal(ml_prediction, ml_confidence)
        
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        self.assertLess(execution_time, 5.0, "Signal generation should complete within 5 seconds")
        
        print(f"Signal generation time: {execution_time:.2f} seconds for {num_signals} signals")
    
    def test_position_sizing_performance(self):
        """Test position sizing performance."""
        sizer = PositionSizer()
        
        # Generate large number of position size calculations
        num_calculations = 10000
        
        # Test position sizing
        start_time = time.time()
        
        for i in range(num_calculations):
            signal_confidence = np.random.rand()
            asset_volatility = np.random.rand() * 0.5 + 0.1
            account_balance = np.random.rand() * 1000000 + 100000
            price = np.random.rand() * 500 + 50
            
            position_units, capital_allocated = sizer.get_position_size(
                signal_confidence=signal_confidence,
                asset_volatility=asset_volatility,
                account_balance=account_balance,
                price=price
            )
        
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        self.assertLess(execution_time, 10.0, "Position sizing should complete within 10 seconds")
        
        print(f"Position sizing time: {execution_time:.2f} seconds for {num_calculations} calculations")
    
    def test_risk_management_performance(self):
        """Test risk management performance."""
        risk_manager = RiskManager()
        
        # Generate large number of risk checks
        num_checks = 10000
        
        # Test risk management
        start_time = time.time()
        
        for i in range(num_checks):
            # Test trade validation
            is_valid, reason = risk_manager.validate_new_trade(
                symbol='AAPL',
                trade_type='BUY',
                quantity=np.random.randint(1, 1000),
                price=np.random.rand() * 500 + 50,
                current_portfolio_value=np.random.rand() * 1000000 + 100000,
                open_positions_count=np.random.randint(0, 20)
            )
            
            # Test exit conditions
            should_exit, exit_reason = risk_manager.check_position_exit_conditions(
                current_price=np.random.rand() * 500 + 50,
                position_entry_price=np.random.rand() * 500 + 50,
                trade_type='BUY'
            )
        
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        self.assertLess(execution_time, 8.0, "Risk management should complete within 8 seconds")
        
        print(f"Risk management time: {execution_time:.2f} seconds for {num_checks} checks")


class TestBacktestingPerformance(PerformanceTestBase):
    """Performance tests for backtesting components."""
    
    def test_backtest_engine_performance(self):
        """Test BacktestEngine performance."""
        engine = BacktestEngine()
        
        # Generate large dataset
        large_data = self.generate_large_dataset(5000, ['AAPL', 'MSFT', 'GOOGL'])
        
        # Mock components
        mock_signal_generator = Mock()
        mock_position_sizer = Mock()
        mock_risk_manager = Mock()
        mock_ensemble_predictor = Mock()
        
        # Mock returns
        mock_signal_generator.generate_signal.return_value = (1, 0.8)
        mock_position_sizer.get_position_size.return_value = (100, 15000)
        mock_risk_manager.validate_new_trade.return_value = (True, "Valid")
        mock_ensemble_predictor.combine_predictions.return_value = np.array([0.02])
        
        # Test backtest performance
        start_time = time.time()
        
        result = engine.run_backtest(
            data=large_data,
            signal_generator=mock_signal_generator,
            position_sizer=mock_position_sizer,
            risk_manager=mock_risk_manager,
            ensemble_predictor=mock_ensemble_predictor
        )
        
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        self.assertLess(execution_time, 30.0, "Backtest should complete within 30 seconds")
        self.assertIsInstance(result, dict)
        self.assertIn('performance_metrics', result)
        
        print(f"Backtest time: {execution_time:.2f} seconds for {len(large_data)} symbols")
    
    def test_performance_analyzer_performance(self):
        """Test PerformanceAnalyzer performance."""
        analyzer = PerformanceAnalyzer()
        
        # Generate large backtest results
        num_days = 10000
        portfolio_values = []
        
        for i in range(num_days):
            portfolio_values.append({
                'date': f'2023-01-{i+1:02d}',
                'value': 100000 + i * 10 + np.random.randn() * 1000,
                'cash': 10000,
                'positions_value': 90000 + i * 10,
                'unrealized_pnl': np.random.randn() * 1000
            })
        
        mock_results = {
            'portfolio_values': portfolio_values,
            'trades': []
        }
        
        # Test performance analysis
        start_time = time.time()
        
        analysis = analyzer.analyze_performance(mock_results)
        
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        self.assertLess(execution_time, 15.0, "Performance analysis should complete within 15 seconds")
        self.assertIsInstance(analysis, dict)
        self.assertIn('performance_metrics', analysis)
        
        print(f"Performance analysis time: {execution_time:.2f} seconds for {num_days} days")
    
    def test_statistical_validator_performance(self):
        """Test StatisticalValidator performance."""
        validator = StatisticalValidator()
        
        # Generate large backtest results
        num_days = 5000
        portfolio_values = []
        
        for i in range(num_days):
            portfolio_values.append({
                'date': f'2023-01-{i+1:02d}',
                'value': 100000 + i * 10 + np.random.randn() * 1000
            })
        
        mock_results = {
            'portfolio_values': portfolio_values
        }
        
        # Test statistical validation
        start_time = time.time()
        
        validation = validator.validate_performance(mock_results)
        
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertions
        self.assertLess(execution_time, 20.0, "Statistical validation should complete within 20 seconds")
        self.assertIsInstance(validation, dict)
        self.assertIn('performance_significance', validation)
        
        print(f"Statistical validation time: {execution_time:.2f} seconds for {num_days} days")


class TestMemoryUsage(PerformanceTestBase):
    """Tests for memory usage and leaks."""
    
    def test_data_pipeline_memory_usage(self):
        """Test memory usage of data pipeline components."""
        # Generate large dataset
        large_data = self.generate_large_dataset(20000, ['AAPL'])
        df = large_data['AAPL']
        
        # Test memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Apply indicators
        engine = IndicatorEngine()
        df_with_indicators = engine.calculate_all_indicators(df.copy())
        
        after_indicators = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Apply feature engineering
        engineer = FeatureEngineer()
        df_with_features = engineer.generate_basic_features(df_with_indicators.copy())
        df_with_features = engineer.generate_advanced_features(df_with_features.copy())
        
        after_features = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Memory usage assertions
        indicators_memory = after_indicators - initial_memory
        features_memory = after_features - after_indicators
        
        self.assertLess(indicators_memory, 500, "Indicator calculation should use less than 500MB")
        self.assertLess(features_memory, 200, "Feature engineering should use less than 200MB")
        
        print(f"Memory usage - Indicators: {indicators_memory:.2f} MB")
        print(f"Memory usage - Features: {features_memory:.2f} MB")
    
    def test_ml_models_memory_usage(self):
        """Test memory usage of ML models."""
        # Generate large dataset
        time_steps = 60
        n_features = 50
        num_samples = 5000
        
        X_dummy = np.random.rand(num_samples, time_steps, n_features)
        y_dummy = np.random.rand(num_samples, 1) * 0.1
        
        # Test CNN-LSTM memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        model = CNNLSTMModel(time_steps=time_steps, n_features=n_features)
        
        after_model_creation = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Test Random Forest memory usage
        rf_model = RandomForestModel()
        rf_model.train(X_dummy.reshape(num_samples, -1), y_dummy.flatten())
        
        after_rf_training = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Memory usage assertions
        cnn_lstm_memory = after_model_creation - initial_memory
        rf_memory = after_rf_training - after_model_creation
        
        self.assertLess(cnn_lstm_memory, 300, "CNN-LSTM model should use less than 300MB")
        self.assertLess(rf_memory, 400, "Random Forest model should use less than 400MB")
        
        print(f"Memory usage - CNN-LSTM: {cnn_lstm_memory:.2f} MB")
        print(f"Memory usage - Random Forest: {rf_memory:.2f} MB")


class TestScalability(PerformanceTestBase):
    """Tests for system scalability."""
    
    def test_data_scalability(self):
        """Test data processing scalability."""
        sizes = [1000, 5000, 10000, 20000]
        execution_times = []
        
        for size in sizes:
            # Generate data
            data = self.generate_large_dataset(size, ['AAPL'])
            df = data['AAPL']
            
            # Test performance
            start_time = time.time()
            
            engine = IndicatorEngine()
            result = engine.calculate_all_indicators(df.copy())
            
            end_time = time.time()
            execution_times.append(end_time - start_time)
        
        # Check scalability (should be roughly linear)
        for i in range(1, len(execution_times)):
            ratio = execution_times[i] / execution_times[i-1]
            size_ratio = sizes[i] / sizes[i-1]
            
            # Allow some tolerance for non-linear behavior
            self.assertLess(ratio, size_ratio * 2, f"Scalability issue: {ratio:.2f} vs {size_ratio:.2f}")
        
        print(f"Scalability test - Sizes: {sizes}")
        print(f"Scalability test - Times: {execution_times}")
    
    def test_symbol_scalability(self):
        """Test scalability with number of symbols."""
        symbol_counts = [1, 5, 10, 20]
        execution_times = []
        
        for count in symbol_counts:
            symbols = [f'SYMBOL_{i}' for i in range(count)]
            data = self.generate_large_dataset(1000, symbols)
            
            # Test performance
            start_time = time.time()
            
            engineer = FeatureEngineer()
            for symbol in symbols:
                df_with_indicators = IndicatorEngine().calculate_all_indicators(data[symbol].copy())
                df_with_features = engineer.generate_basic_features(df_with_indicators.copy())
            
            end_time = time.time()
            execution_times.append(end_time - start_time)
        
        # Check scalability
        for i in range(1, len(execution_times)):
            ratio = execution_times[i] / execution_times[i-1]
            symbol_ratio = symbol_counts[i] / symbol_counts[i-1]
            
            self.assertLess(ratio, symbol_ratio * 2, f"Symbol scalability issue: {ratio:.2f} vs {symbol_ratio:.2f}")
        
        print(f"Symbol scalability test - Counts: {symbol_counts}")
        print(f"Symbol scalability test - Times: {execution_times}")


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDataPipelinePerformance,
        TestMLModelsPerformance,
        TestStrategyPerformance,
        TestBacktestingPerformance,
        TestMemoryUsage,
        TestScalability
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nPerformance Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
