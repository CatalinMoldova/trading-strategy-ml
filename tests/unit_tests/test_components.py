"""
Unit tests for the trading strategy ML system.
Tests individual components in isolation to ensure correctness.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime, timedelta

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


class TestMarketDataCollector(unittest.TestCase):
    """Test cases for MarketDataCollector."""
    
    def setUp(self):
        self.collector = MarketDataCollector()
    
    def test_init(self):
        """Test MarketDataCollector initialization."""
        self.assertIsNotNone(self.collector)
        self.assertIsNone(self.collector.ts)  # No API key provided
    
    @patch('yfinance.download')
    def test_get_yahoo_finance_data(self, mock_download):
        """Test Yahoo Finance data collection."""
        # Mock data
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [103, 102, 104],
            'Low': [99, 100, 101],
            'Close': [101, 102, 103],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        mock_download.return_value = mock_data
        
        result = self.collector.get_yahoo_finance_data("AAPL", period="1y")
        
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)
        self.assertIn('close', result.columns)
        self.assertIn('volume', result.columns)
    
    def test_get_multiple_symbols_data(self):
        """Test multiple symbols data collection."""
        symbols = ["AAPL", "MSFT"]
        
        with patch.object(self.collector, 'get_yahoo_finance_data') as mock_get:
            mock_get.return_value = pd.DataFrame({
                'close': [100, 101, 102],
                'volume': [1000, 1100, 1200]
            })
            
            result = self.collector.get_multiple_symbols_data(symbols, source="yahoo")
            
            self.assertEqual(len(result), 2)
            self.assertIn("AAPL", result)
            self.assertIn("MSFT", result)


class TestIndicatorEngine(unittest.TestCase):
    """Test cases for IndicatorEngine."""
    
    def setUp(self):
        self.engine = IndicatorEngine()
        
        # Sample data
        self.sample_data = pd.DataFrame({
            'open': [100, 102, 101, 105, 103, 106, 108, 107, 110, 109],
            'high': [103, 104, 105, 107, 106, 109, 110, 109, 112, 111],
            'low': [99, 101, 100, 103, 102, 104, 106, 105, 108, 107],
            'close': [102, 103, 104, 106, 105, 108, 109, 108, 111, 110],
            'volume': [1000, 1200, 1100, 1500, 1300, 1600, 1700, 1400, 1800, 1500]
        })
    
    def test_calculate_momentum_indicators(self):
        """Test momentum indicators calculation."""
        result = self.engine.calculate_momentum_indicators(self.sample_data.copy())
        
        self.assertIn('RSI', result.columns)
        self.assertIn('MACD', result.columns)
        self.assertIn('ROC', result.columns)
        self.assertIn('STOCH_K', result.columns)
        self.assertIn('ADX', result.columns)
    
    def test_calculate_volatility_indicators(self):
        """Test volatility indicators calculation."""
        result = self.engine.calculate_volatility_indicators(self.sample_data.copy())
        
        self.assertIn('BB_UPPER', result.columns)
        self.assertIn('BB_MIDDLE', result.columns)
        self.assertIn('BB_LOWER', result.columns)
        self.assertIn('ATR', result.columns)
    
    def test_calculate_volume_indicators(self):
        """Test volume indicators calculation."""
        result = self.engine.calculate_volume_indicators(self.sample_data.copy())
        
        self.assertIn('OBV', result.columns)
        self.assertIn('MFI', result.columns)
    
    def test_calculate_all_indicators(self):
        """Test all indicators calculation."""
        result = self.engine.calculate_all_indicators(self.sample_data.copy())
        
        # Check that all indicator types are present
        momentum_cols = ['RSI', 'MACD', 'ROC', 'STOCH_K', 'ADX']
        volatility_cols = ['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'ATR']
        volume_cols = ['OBV', 'MFI']
        
        for col in momentum_cols + volatility_cols + volume_cols:
            self.assertIn(col, result.columns)


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer."""
    
    def setUp(self):
        self.engineer = FeatureEngineer()
        
        # Sample data with indicators
        self.sample_data = pd.DataFrame({
            'open': [100, 102, 101, 105, 103],
            'high': [103, 104, 105, 107, 106],
            'low': [99, 101, 100, 103, 102],
            'close': [102, 103, 104, 106, 105],
            'volume': [1000, 1200, 1100, 1500, 1300],
            'RSI': [50, 55, 60, 65, 70],
            'MACD': [1.0, 1.5, 2.0, 2.5, 3.0],
            'ATR': [0.5, 0.6, 0.7, 0.8, 0.9]
        })
    
    def test_generate_basic_features(self):
        """Test basic feature generation."""
        result = self.engineer.generate_basic_features(self.sample_data.copy())
        
        self.assertIn('daily_return', result.columns)
        self.assertIn('log_return', result.columns)
        self.assertIn('high_low_range', result.columns)
        self.assertIn('open_close_range', result.columns)
    
    def test_generate_advanced_features(self):
        """Test advanced feature generation."""
        result = self.engineer.generate_advanced_features(self.sample_data.copy())
        
        self.assertIn('volatility_ratio', result.columns)
        self.assertIn('RSI_MACD_interaction', result.columns)
    
    def test_cross_sectional_normalize(self):
        """Test cross-sectional normalization."""
        data_dict = {
            'AAPL': self.sample_data.copy(),
            'MSFT': self.sample_data.copy() * 1.5
        }
        
        feature_cols = ['close', 'RSI', 'MACD']
        result = self.engineer.cross_sectional_normalize(data_dict, feature_cols)
        
        self.assertEqual(len(result), 2)
        self.assertIn('AAPL', result)
        self.assertIn('MSFT', result)
        
        # Check normalized columns exist
        for symbol in result:
            for col in feature_cols:
                self.assertIn(f'{col}_cs_norm', result[symbol].columns)
    
    def test_time_series_standardize(self):
        """Test time-series standardization."""
        feature_cols = ['RSI', 'MACD', 'ATR']
        result = self.engineer.time_series_standardize(
            self.sample_data.copy(), feature_cols, 'TEST'
        )
        
        # Check scaler is stored
        self.assertIn('TEST_standard_scaler', self.engineer.scalers)
        
        # Check features are scaled
        for col in feature_cols:
            self.assertIn(col, result.columns)
    
    def test_prepare_ml_data(self):
        """Test ML data preparation."""
        result = self.engineer.prepare_ml_data(
            self.sample_data.copy(), 
            target_column='close', 
            time_steps=3
        )
        
        self.assertIn('X', result)
        self.assertIn('y', result)
        self.assertIsInstance(result['X'], np.ndarray)
        self.assertIsInstance(result['y'], np.ndarray)


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor."""
    
    def setUp(self):
        self.preprocessor = DataPreprocessor()
        
        # Sample data
        self.sample_data = np.random.rand(100, 5)
        self.sample_features = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100)
        })
        self.sample_target = pd.Series(np.random.rand(100))
    
    def test_create_sequences(self):
        """Test sequence creation."""
        time_steps = 10
        X, y = self.preprocessor.create_sequences(self.sample_data, time_steps)
        
        self.assertEqual(X.shape[1], time_steps)
        self.assertEqual(X.shape[2], self.sample_data.shape[1])
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len(self.sample_data) - time_steps)
    
    def test_create_sequences_with_target(self):
        """Test sequence creation with target."""
        time_steps = 5
        X, y = self.preprocessor.create_sequences_with_target(
            self.sample_features, self.sample_target, time_steps
        )
        
        self.assertEqual(X.shape[1], time_steps)
        self.assertEqual(X.shape[2], len(self.sample_features.columns))
        self.assertEqual(len(X), len(y))
    
    def test_scale_features(self):
        """Test feature scaling."""
        feature_cols = ['feature1', 'feature2']
        result = self.preprocessor.scale_features(
            self.sample_features.copy(), feature_cols, 'standard', 'TEST', fit=True
        )
        
        # Check scaler is stored
        self.assertIn('TEST_standard_scaler', self.preprocessor.scalers)
        
        # Check features are scaled
        for col in feature_cols:
            self.assertIn(col, result.columns)
    
    def test_inverse_scale_target(self):
        """Test inverse scaling."""
        # First scale the target
        target_scaled = self.preprocessor.scale_features(
            self.sample_target.to_frame(), ['target'], 'standard', 'TEST', fit=True
        )['target']
        
        # Then inverse scale
        original = self.preprocessor.inverse_scale_target(
            target_scaled.values, 'TEST', 'standard'
        )
        
        # Check that inverse scaling works
        np.testing.assert_array_almost_equal(original, self.sample_target.values, decimal=5)


class TestSignalGenerator(unittest.TestCase):
    """Test cases for SignalGenerator."""
    
    def setUp(self):
        self.generator = SignalGenerator()
    
    def test_generate_ml_signal(self):
        """Test ML signal generation."""
        # Test buy signal
        signal, confidence = self.generator.generate_ml_signal(0.03, 0.035)
        self.assertEqual(signal, 1)
        self.assertEqual(confidence, 0.035)
        
        # Test sell signal
        signal, confidence = self.generator.generate_ml_signal(-0.025, 0.028)
        self.assertEqual(signal, -1)
        self.assertEqual(confidence, 0.028)
        
        # Test hold signal (low confidence)
        signal, confidence = self.generator.generate_ml_signal(0.01, 0.015)
        self.assertEqual(signal, 0)
        self.assertEqual(confidence, 0.015)
    
    def test_generate_multi_factor_score(self):
        """Test multi-factor score generation."""
        features = pd.Series({
            'RSI': 0.7,
            'MACD_HIST': 0.05,
            'ATR_NORM': 0.01,
            'VOL_RATIO': -0.02
        })
        
        factor_weights = {
            'RSI': 0.4,
            'MACD_HIST': 0.3,
            'ATR_NORM': -0.1,
            'VOL_RATIO': 0.2
        }
        
        score = self.generator.generate_multi_factor_score(features, factor_weights)
        self.assertIsInstance(score, float)
    
    def test_combine_signals(self):
        """Test signal combination."""
        # Test buy signal combination
        signal, confidence = self.generator.combine_signals(
            ml_signal=1, ml_confidence=0.035, factor_score=0.5,
            ml_weight=0.6, factor_weight=0.4, score_threshold=0.01
        )
        
        self.assertIsInstance(signal, int)
        self.assertIsInstance(confidence, float)
        self.assertIn(signal, [-1, 0, 1])


class TestPositionSizer(unittest.TestCase):
    """Test cases for PositionSizer."""
    
    def setUp(self):
        self.sizer = PositionSizer()
    
    def test_calculate_garch_volatility(self):
        """Test GARCH volatility calculation."""
        # Generate sample returns
        returns = np.random.normal(0.0005, 0.01, 500)
        
        volatility = self.sizer.calculate_garch_volatility(returns)
        
        self.assertIsInstance(volatility, float)
        self.assertGreater(volatility, 0)
    
    def test_calculate_kelly_criterion(self):
        """Test Kelly Criterion calculation."""
        # Test valid inputs
        kelly_f = self.sizer.calculate_kelly_criterion(0.6, 1.5)
        self.assertIsInstance(kelly_f, float)
        self.assertGreaterEqual(kelly_f, 0)
        
        # Test invalid inputs
        kelly_f_invalid = self.sizer.calculate_kelly_criterion(0.5, 0)
        self.assertEqual(kelly_f_invalid, 0)
    
    def test_get_position_size(self):
        """Test position size calculation."""
        position_units, capital_allocated = self.sizer.get_position_size(
            signal_confidence=0.75,
            asset_volatility=0.20,
            account_balance=100000.0,
            price=150.0,
            win_rate=0.58,
            avg_win_loss_ratio=1.3
        )
        
        self.assertIsInstance(position_units, float)
        self.assertIsInstance(capital_allocated, float)
        self.assertGreaterEqual(position_units, 0)
        self.assertGreaterEqual(capital_allocated, 0)


class TestRiskManager(unittest.TestCase):
    """Test cases for RiskManager."""
    
    def setUp(self):
        self.risk_manager = RiskManager()
    
    def test_calculate_stop_loss_price(self):
        """Test stop-loss price calculation."""
        entry_price = 100.0
        
        # Test buy position
        stop_loss_buy = self.risk_manager.calculate_stop_loss_price(entry_price, 'BUY')
        self.assertEqual(stop_loss_buy, entry_price * (1 - self.risk_manager.stop_loss_pct))
        
        # Test sell position
        stop_loss_sell = self.risk_manager.calculate_stop_loss_price(entry_price, 'SELL')
        self.assertEqual(stop_loss_sell, entry_price * (1 + self.risk_manager.stop_loss_pct))
    
    def test_calculate_take_profit_price(self):
        """Test take-profit price calculation."""
        entry_price = 100.0
        
        # Test buy position
        take_profit_buy = self.risk_manager.calculate_take_profit_price(entry_price, 'BUY')
        self.assertEqual(take_profit_buy, entry_price * (1 + self.risk_manager.take_profit_pct))
        
        # Test sell position
        take_profit_sell = self.risk_manager.calculate_take_profit_price(entry_price, 'SELL')
        self.assertEqual(take_profit_sell, entry_price * (1 - self.risk_manager.take_profit_pct))
    
    def test_validate_new_trade(self):
        """Test new trade validation."""
        # Test valid trade
        is_valid, reason = self.risk_manager.validate_new_trade(
            symbol='AAPL',
            trade_type='BUY',
            quantity=100,
            price=150.0,
            current_portfolio_value=100000.0,
            open_positions_count=5
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(reason, "Trade valid.")
        
        # Test invalid trade (too many positions)
        is_valid, reason = self.risk_manager.validate_new_trade(
            symbol='AAPL',
            trade_type='BUY',
            quantity=100,
            price=150.0,
            current_portfolio_value=100000.0,
            open_positions_count=15  # Exceeds max_open_positions
        )
        
        self.assertFalse(is_valid)
        self.assertIn("Max open positions", reason)
    
    def test_check_position_exit_conditions(self):
        """Test position exit conditions."""
        entry_price = 100.0
        
        # Test stop-loss hit
        should_exit, reason = self.risk_manager.check_position_exit_conditions(
            current_price=95.0,  # Below stop-loss
            position_entry_price=entry_price,
            trade_type='BUY'
        )
        
        self.assertTrue(should_exit)
        self.assertIn("Stop-loss", reason)
        
        # Test take-profit hit
        should_exit, reason = self.risk_manager.check_position_exit_conditions(
            current_price=105.0,  # Above take-profit
            position_entry_price=entry_price,
            trade_type='BUY'
        )
        
        self.assertTrue(should_exit)
        self.assertIn("Take-profit", reason)


class TestEnsemblePredictor(unittest.TestCase):
    """Test cases for EnsemblePredictor."""
    
    def setUp(self):
        self.predictor = EnsemblePredictor()
    
    def test_combine_predictions(self):
        """Test prediction combination."""
        predictions = {
            'cnn_lstm': np.array([0.02, 0.03, -0.01]),
            'random_forest': np.array([0.015, 0.025, -0.005])
        }
        
        combined = self.predictor.combine_predictions(predictions)
        
        self.assertIsInstance(combined, np.ndarray)
        self.assertEqual(len(combined), 3)
    
    def test_generate_signal(self):
        """Test signal generation."""
        predictions = np.array([0.03, -0.025, 0.01])
        
        signals, confidence = self.predictor.generate_signal(predictions)
        
        self.assertIsInstance(signals, np.ndarray)
        self.assertIsInstance(confidence, np.ndarray)
        self.assertEqual(len(signals), len(predictions))
        self.assertEqual(len(confidence), len(predictions))


class TestBacktestEngine(unittest.TestCase):
    """Test cases for BacktestEngine."""
    
    def setUp(self):
        self.engine = BacktestEngine()
        
        # Mock data
        self.mock_data = {
            'AAPL': pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=10),
                'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
                'high': [103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
                'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
                'close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
                'volume': [1000] * 10,
                'RSI': [50] * 10,
                'MACD': [1.0] * 10
            })
        }
    
    def test_init(self):
        """Test BacktestEngine initialization."""
        self.assertEqual(self.engine.initial_capital, 100000.0)
        self.assertEqual(self.engine.commission, 0.001)
        self.assertEqual(self.engine.slippage, 0.0005)
    
    @patch('src.backtesting.backtest_engine.SignalGenerator')
    @patch('src.backtesting.backtest_engine.PositionSizer')
    @patch('src.backtesting.backtest_engine.RiskManager')
    @patch('src.backtesting.backtest_engine.EnsemblePredictor')
    def test_run_backtest(self, mock_ensemble, mock_risk, mock_position, mock_signal):
        """Test backtest execution."""
        # Mock the components
        mock_signal.return_value.generate_signal.return_value = (1, 0.8)
        mock_position.return_value.get_position_size.return_value = (100, 15000)
        mock_risk.return_value.validate_new_trade.return_value = (True, "Valid")
        mock_ensemble.return_value.combine_predictions.return_value = np.array([0.02])
        
        result = self.engine.run_backtest(
            data=self.mock_data,
            signal_generator=mock_signal.return_value,
            position_sizer=mock_position.return_value,
            risk_manager=mock_risk.return_value,
            ensemble_predictor=mock_ensemble.return_value
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('performance_metrics', result)
        self.assertIn('portfolio_values', result)
        self.assertIn('trades', result)


class TestPerformanceAnalyzer(unittest.TestCase):
    """Test cases for PerformanceAnalyzer."""
    
    def setUp(self):
        self.analyzer = PerformanceAnalyzer()
        
        # Mock backtest results
        self.mock_results = {
            'portfolio_values': [
                {'date': '2023-01-01', 'value': 100000, 'cash': 10000, 'positions_value': 90000, 'unrealized_pnl': 0},
                {'date': '2023-01-02', 'value': 101000, 'cash': 10000, 'positions_value': 91000, 'unrealized_pnl': 1000},
                {'date': '2023-01-03', 'value': 102000, 'cash': 10000, 'positions_value': 92000, 'unrealized_pnl': 2000}
            ],
            'trades': [
                {'date': '2023-01-01', 'symbol': 'AAPL', 'trade_type': 'BUY', 'quantity': 100, 'price': 150, 'fee': 15},
                {'date': '2023-01-02', 'symbol': 'AAPL', 'trade_type': 'SELL', 'quantity': 100, 'price': 155, 'fee': 15.5}
            ]
        }
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        metrics = self.analyzer.calculate_performance_metrics(self.mock_results)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_return', metrics)
        self.assertIn('annualized_return', metrics)
        self.assertIn('volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
    
    def test_calculate_risk_metrics(self):
        """Test risk metrics calculation."""
        risk_metrics = self.analyzer.calculate_risk_metrics(self.mock_results)
        
        self.assertIsInstance(risk_metrics, dict)
        self.assertIn('var_95', risk_metrics)
        self.assertIn('cvar_95', risk_metrics)
        self.assertIn('max_drawdown', risk_metrics)
        self.assertIn('downside_deviation', risk_metrics)
    
    def test_analyze_performance(self):
        """Test overall performance analysis."""
        analysis = self.analyzer.analyze_performance(self.mock_results)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('performance_metrics', analysis)
        self.assertIn('risk_metrics', analysis)
        self.assertIn('trade_analysis', analysis)


class TestStatisticalValidator(unittest.TestCase):
    """Test cases for StatisticalValidator."""
    
    def setUp(self):
        self.validator = StatisticalValidator()
        
        # Mock backtest results
        self.mock_results = {
            'portfolio_values': [
                {'date': '2023-01-01', 'value': 100000},
                {'date': '2023-01-02', 'value': 101000},
                {'date': '2023-01-03', 'value': 102000}
            ]
        }
    
    def test_validate_performance(self):
        """Test performance validation."""
        validation = self.validator.validate_performance(self.mock_results)
        
        self.assertIsInstance(validation, dict)
        self.assertIn('performance_significance', validation)
        self.assertIn('overall_assessment', validation)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestMarketDataCollector,
        TestIndicatorEngine,
        TestFeatureEngineer,
        TestDataPreprocessor,
        TestSignalGenerator,
        TestPositionSizer,
        TestRiskManager,
        TestEnsemblePredictor,
        TestBacktestEngine,
        TestPerformanceAnalyzer,
        TestStatisticalValidator
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
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
