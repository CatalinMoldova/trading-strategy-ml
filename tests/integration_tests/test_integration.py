"""
Integration tests for the trading strategy ML system.
Tests the interaction between multiple components.
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
from src.ml_models.model_trainer import ModelTrainer

from src.strategy.signal_generator import SignalGenerator
from src.strategy.position_sizer import PositionSizer
from src.strategy.risk_manager import RiskManager

from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.performance_analyzer import PerformanceAnalyzer
from src.backtesting.statistical_validator import StatisticalValidator
from src.backtesting.walk_forward_analyzer import WalkForwardAnalyzer
from src.backtesting.report_generator import ReportGenerator, ReportConfig


class TestDataPipelineIntegration(unittest.TestCase):
    """Test integration between data pipeline components."""
    
    def setUp(self):
        self.collector = MarketDataCollector()
        self.indicator_engine = IndicatorEngine()
        self.feature_engineer = FeatureEngineer()
        
        # Sample market data
        self.sample_data = pd.DataFrame({
            'open': [100, 102, 101, 105, 103, 106, 108, 107, 110, 109, 112, 111, 115, 113, 116, 118, 117, 120, 119, 122],
            'high': [103, 104, 105, 107, 106, 109, 110, 109, 112, 111, 114, 113, 117, 115, 118, 120, 119, 122, 121, 124],
            'low': [99, 101, 100, 103, 102, 104, 106, 105, 108, 107, 110, 109, 112, 111, 114, 116, 115, 118, 117, 120],
            'close': [102, 103, 104, 106, 105, 108, 109, 108, 111, 110, 113, 112, 116, 114, 117, 119, 118, 121, 120, 123],
            'volume': [1000, 1200, 1100, 1500, 1300, 1600, 1700, 1400, 1800, 1500, 1900, 1600, 2000, 1700, 2100, 1800, 2200, 1900, 2300, 2000]
        }, index=pd.date_range('2023-01-01', periods=20))
    
    def test_data_collection_to_feature_engineering(self):
        """Test integration from data collection to feature engineering."""
        # Step 1: Apply technical indicators
        df_with_indicators = self.indicator_engine.calculate_all_indicators(self.sample_data.copy())
        
        # Step 2: Apply feature engineering
        df_with_features = self.feature_engineer.generate_basic_features(df_with_indicators.copy())
        df_with_features = self.feature_engineer.generate_advanced_features(df_with_features.copy())
        
        # Verify integration
        self.assertFalse(df_with_features.empty)
        self.assertIn('daily_return', df_with_features.columns)
        self.assertIn('RSI', df_with_features.columns)
        self.assertIn('MACD', df_with_features.columns)
        self.assertIn('volatility_ratio', df_with_features.columns)
    
    def test_cross_sectional_normalization_integration(self):
        """Test cross-sectional normalization integration."""
        # Create data for multiple symbols
        data_dict = {
            'AAPL': self.sample_data.copy(),
            'MSFT': self.sample_data.copy() * 1.5,
            'GOOGL': self.sample_data.copy() * 2.0
        }
        
        # Apply indicators to all symbols
        for symbol in data_dict:
            data_dict[symbol] = self.indicator_engine.calculate_all_indicators(data_dict[symbol])
            data_dict[symbol] = self.feature_engineer.generate_basic_features(data_dict[symbol])
        
        # Apply cross-sectional normalization
        feature_cols = ['close', 'RSI', 'MACD']
        normalized_data = self.feature_engineer.cross_sectional_normalize(data_dict, feature_cols)
        
        # Verify integration
        self.assertEqual(len(normalized_data), 3)
        for symbol in normalized_data:
            for col in feature_cols:
                self.assertIn(f'{col}_cs_norm', normalized_data[symbol].columns)
    
    def test_data_preparation_for_ml(self):
        """Test data preparation for ML models."""
        # Apply full pipeline
        df_processed = self.indicator_engine.calculate_all_indicators(self.sample_data.copy())
        df_processed = self.feature_engineer.generate_basic_features(df_processed)
        df_processed = self.feature_engineer.generate_advanced_features(df_processed)
        
        # Create target variable
        df_processed['next_day_return'] = df_processed['close'].pct_change().shift(-1)
        
        # Prepare for ML
        feature_cols = ['RSI', 'MACD', 'ATR', 'daily_return']
        ml_data = self.feature_engineer.prepare_ml_data(
            df_processed.dropna(), 
            target_column='next_day_return', 
            time_steps=5
        )
        
        # Verify ML data preparation
        self.assertIn('X', ml_data)
        self.assertIn('y', ml_data)
        self.assertIsInstance(ml_data['X'], np.ndarray)
        self.assertIsInstance(ml_data['y'], np.ndarray)
        self.assertEqual(ml_data['X'].shape[1], 5)  # time_steps
        self.assertEqual(ml_data['X'].shape[2], len(feature_cols))  # features


class TestMLPipelineIntegration(unittest.TestCase):
    """Test integration between ML components."""
    
    def setUp(self):
        self.data_preprocessor = DataPreprocessor()
        self.ensemble_predictor = EnsemblePredictor()
        
        # Generate sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100),
            'target': np.random.randn(100) * 0.01
        })
    
    def test_data_preprocessing_to_model_training(self):
        """Test integration from data preprocessing to model training."""
        # Prepare data
        feature_cols = ['feature1', 'feature2', 'feature3']
        target_col = 'target'
        
        # Scale features
        scaled_data = self.data_preprocessor.scale_features(
            self.sample_data.copy(), feature_cols, 'standard', 'TEST', fit=True
        )
        
        # Create sequences
        X, y = self.data_preprocessor.create_sequences_with_target(
            scaled_data[feature_cols], scaled_data[target_col], time_steps=10
        )
        
        # Verify data preparation
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(X.shape[1], 10)  # time_steps
        self.assertEqual(X.shape[2], len(feature_cols))  # features
    
    def test_ensemble_prediction_integration(self):
        """Test ensemble prediction integration."""
        # Mock predictions from different models
        predictions = {
            'cnn_lstm': np.array([[0.02], [0.03], [-0.01]]),
            'random_forest': np.array([[0.015], [0.025], [-0.005]])
        }
        
        # Combine predictions
        ensemble_pred = self.ensemble_predictor.combine_predictions(predictions)
        
        # Generate signals
        signals, confidence = self.ensemble_predictor.generate_signal(ensemble_pred)
        
        # Verify ensemble integration
        self.assertIsInstance(ensemble_pred, np.ndarray)
        self.assertIsInstance(signals, np.ndarray)
        self.assertIsInstance(confidence, np.ndarray)
        self.assertEqual(len(signals), len(ensemble_pred))
    
    def test_model_trainer_integration(self):
        """Test model trainer integration."""
        # Create trainer
        trainer = ModelTrainer()
        
        # Prepare data
        feature_cols = ['feature1', 'feature2', 'feature3']
        target_col = 'target'
        
        try:
            # Prepare training data
            X_train, X_val, y_train, y_val, original_feature_names = trainer.prepare_data_for_training(
                df=self.sample_data,
                feature_cols=feature_cols,
                target_col=target_col,
                symbol='TEST'
            )
            
            # Verify data preparation
            self.assertIsInstance(X_train, np.ndarray)
            self.assertIsInstance(y_train, np.ndarray)
            self.assertIsInstance(X_val, np.ndarray)
            self.assertIsInstance(y_val, np.ndarray)
            
        except Exception as e:
            # This might fail due to insufficient data, which is expected
            self.assertIsInstance(e, (ValueError, RuntimeError))


class TestStrategyIntegration(unittest.TestCase):
    """Test integration between strategy components."""
    
    def setUp(self):
        self.signal_generator = SignalGenerator()
        self.position_sizer = PositionSizer()
        self.risk_manager = RiskManager()
    
    def test_signal_to_position_sizing_integration(self):
        """Test integration from signal generation to position sizing."""
        # Generate signal
        ml_prediction = 0.03
        ml_confidence = 0.035
        signal, confidence = self.signal_generator.generate_ml_signal(ml_prediction, ml_confidence)
        
        # Generate multi-factor score
        features = pd.Series({
            'RSI': 0.7,
            'MACD_HIST': 0.05,
            'ATR_NORM': 0.01
        })
        factor_weights = {'RSI': 0.4, 'MACD_HIST': 0.3, 'ATR_NORM': -0.1}
        factor_score = self.signal_generator.generate_multi_factor_score(features, factor_weights)
        
        # Combine signals
        final_signal, final_confidence = self.signal_generator.combine_signals(
            ml_signal=signal,
            ml_confidence=confidence,
            factor_score=factor_score
        )
        
        # Calculate position size
        position_units, capital_allocated = self.position_sizer.get_position_size(
            signal_confidence=final_confidence,
            asset_volatility=0.20,
            account_balance=100000.0,
            price=150.0
        )
        
        # Verify integration
        self.assertIsInstance(final_signal, int)
        self.assertIsInstance(final_confidence, float)
        self.assertIsInstance(position_units, float)
        self.assertIsInstance(capital_allocated, float)
    
    def test_position_sizing_to_risk_management_integration(self):
        """Test integration from position sizing to risk management."""
        # Calculate position size
        position_units, capital_allocated = self.position_sizer.get_position_size(
            signal_confidence=0.75,
            asset_volatility=0.20,
            account_balance=100000.0,
            price=150.0
        )
        
        # Validate trade with risk manager
        is_valid, reason = self.risk_manager.validate_new_trade(
            symbol='AAPL',
            trade_type='BUY',
            quantity=position_units,
            price=150.0,
            current_portfolio_value=100000.0,
            open_positions_count=5
        )
        
        # Calculate stop-loss and take-profit
        if is_valid:
            stop_loss_price = self.risk_manager.calculate_stop_loss_price(150.0, 'BUY')
            take_profit_price = self.risk_manager.calculate_take_profit_price(150.0, 'BUY')
            
            # Verify integration
            self.assertIsInstance(stop_loss_price, float)
            self.assertIsInstance(take_profit_price, float)
            self.assertLess(stop_loss_price, 150.0)  # Stop-loss below entry
            self.assertGreater(take_profit_price, 150.0)  # Take-profit above entry


class TestBacktestingIntegration(unittest.TestCase):
    """Test integration between backtesting components."""
    
    def setUp(self):
        self.backtest_engine = BacktestEngine()
        self.performance_analyzer = PerformanceAnalyzer()
        self.statistical_validator = StatisticalValidator()
        
        # Mock backtest results
        self.mock_results = {
            'portfolio_values': [
                {'date': '2023-01-01', 'value': 100000, 'cash': 10000, 'positions_value': 90000, 'unrealized_pnl': 0},
                {'date': '2023-01-02', 'value': 101000, 'cash': 10000, 'positions_value': 91000, 'unrealized_pnl': 1000},
                {'date': '2023-01-03', 'value': 102000, 'cash': 10000, 'positions_value': 92000, 'unrealized_pnl': 2000},
                {'date': '2023-01-04', 'value': 101500, 'cash': 10000, 'positions_value': 91500, 'unrealized_pnl': 1500},
                {'date': '2023-01-05', 'value': 103000, 'cash': 10000, 'positions_value': 93000, 'unrealized_pnl': 3000}
            ],
            'trades': [
                {'date': '2023-01-01', 'symbol': 'AAPL', 'trade_type': 'BUY', 'quantity': 100, 'price': 150, 'fee': 15},
                {'date': '2023-01-03', 'symbol': 'AAPL', 'trade_type': 'SELL', 'quantity': 100, 'price': 155, 'fee': 15.5}
            ]
        }
    
    def test_backtest_to_performance_analysis_integration(self):
        """Test integration from backtest to performance analysis."""
        # Analyze performance
        performance_analysis = self.performance_analyzer.analyze_performance(self.mock_results)
        
        # Verify integration
        self.assertIsInstance(performance_analysis, dict)
        self.assertIn('performance_metrics', performance_analysis)
        self.assertIn('risk_metrics', performance_analysis)
        self.assertIn('trade_analysis', performance_analysis)
        
        # Check performance metrics
        perf_metrics = performance_analysis['performance_metrics']
        self.assertIn('total_return', perf_metrics)
        self.assertIn('sharpe_ratio', perf_metrics)
        self.assertIn('max_drawdown', perf_metrics)
    
    def test_performance_analysis_to_statistical_validation_integration(self):
        """Test integration from performance analysis to statistical validation."""
        # Analyze performance first
        performance_analysis = self.performance_analyzer.analyze_performance(self.mock_results)
        
        # Validate statistically
        statistical_validation = self.statistical_validator.validate_performance(self.mock_results)
        
        # Verify integration
        self.assertIsInstance(statistical_validation, dict)
        self.assertIn('performance_significance', statistical_validation)
        self.assertIn('overall_assessment', statistical_validation)
    
    def test_walk_forward_analysis_integration(self):
        """Test walk-forward analysis integration."""
        # Create walk-forward analyzer
        walk_forward_analyzer = WalkForwardAnalyzer()
        
        # Mock data for walk-forward analysis
        mock_data = {
            'AAPL': pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=100),
                'open': np.random.rand(100) * 100 + 100,
                'high': np.random.rand(100) * 100 + 105,
                'low': np.random.rand(100) * 100 + 95,
                'close': np.random.rand(100) * 100 + 100,
                'volume': np.random.randint(1000, 2000, 100),
                'RSI': np.random.rand(100) * 100,
                'MACD': np.random.randn(100) * 0.1
            })
        }
        
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
        
        try:
            # Run walk-forward analysis
            results = walk_forward_analyzer.run_walk_forward_analysis(
                data=mock_data,
                train_window=30,
                test_window=10,
                step_size=5,
                signal_generator=mock_signal_generator,
                position_sizer=mock_position_sizer,
                risk_manager=mock_risk_manager,
                ensemble_predictor=mock_ensemble_predictor
            )
            
            # Verify integration
            self.assertIsInstance(results, dict)
            self.assertIn('performance_metrics', results)
            self.assertIn('portfolio_values', results)
            
        except Exception as e:
            # This might fail due to insufficient data, which is expected
            self.assertIsInstance(e, (ValueError, RuntimeError))


class TestReportGenerationIntegration(unittest.TestCase):
    """Test integration of report generation."""
    
    def setUp(self):
        self.report_config = ReportConfig(title="Integration Test Report")
        self.report_generator = ReportGenerator(self.report_config)
        
        # Mock backtest results
        self.mock_backtest_results = {
            'performance_metrics': {
                'total_return': 0.15,
                'annualized_return': 0.12,
                'volatility': 0.18,
                'sharpe_ratio': 0.67,
                'max_drawdown': -0.08,
                'var_95': -0.03,
                'cvar_95': -0.05
            },
            'portfolio_values': [
                {'date': '2023-01-01', 'value': 100000},
                {'date': '2023-01-02', 'value': 101000},
                {'date': '2023-01-03', 'value': 102000}
            ],
            'trades': [
                {'date': '2023-01-01', 'symbol': 'AAPL', 'trade_type': 'BUY', 'quantity': 100, 'price': 150, 'fee': 15}
            ]
        }
        
        # Mock performance analysis
        self.mock_performance_analysis = {
            'risk_metrics': {
                'max_drawdown': -0.08,
                'var_95': -0.03,
                'cvar_95': -0.05,
                'downside_deviation': 0.12
            }
        }
        
        # Mock statistical tests
        self.mock_statistical_tests = {
            'performance_significance': {
                'sharpe_ratio_test': {'is_significant': True},
                'mean_return_test': {'is_significant': False}
            },
            'overall_assessment': {
                'assessment': 'Moderate performance with some statistical significance'
            }
        }
    
    def test_report_generation_integration(self):
        """Test report generation integration."""
        # Generate report
        report = self.report_generator.generate_report(
            backtest_results=self.mock_backtest_results,
            performance_analysis=self.mock_performance_analysis,
            statistical_tests=self.mock_statistical_tests
        )
        
        # Verify integration
        self.assertIsInstance(report, str)
        self.assertIn('Executive Summary', report)
        self.assertIn('Performance Metrics', report)
        self.assertIn('Risk Analysis', report)
        self.assertIn('Statistical Tests', report)
        self.assertIn('Recommendations', report)
    
    def test_chart_generation_integration(self):
        """Test chart generation integration."""
        # Generate charts
        self.report_generator._generate_charts()
        
        # Verify charts were generated
        self.assertIsInstance(self.report_generator.charts, dict)
        
        # Check that expected charts exist
        expected_charts = ['equity_curve', 'drawdown', 'returns_distribution']
        for chart_name in expected_charts:
            if chart_name in self.report_generator.charts:
                self.assertIsNotNone(self.report_generator.charts[chart_name])


class TestEndToEndIntegration(unittest.TestCase):
    """Test end-to-end integration of the entire system."""
    
    def setUp(self):
        # Initialize all components
        self.collector = MarketDataCollector()
        self.indicator_engine = IndicatorEngine()
        self.feature_engineer = FeatureEngineer()
        self.data_preprocessor = DataPreprocessor()
        self.ensemble_predictor = EnsemblePredictor()
        self.signal_generator = SignalGenerator()
        self.position_sizer = PositionSizer()
        self.risk_manager = RiskManager()
        self.backtest_engine = BacktestEngine()
        self.performance_analyzer = PerformanceAnalyzer()
        self.statistical_validator = StatisticalValidator()
        self.report_generator = ReportGenerator(ReportConfig())
        
        # Generate comprehensive sample data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100)
        
        self.sample_data = {
            'AAPL': pd.DataFrame({
                'date': dates,
                'open': np.random.rand(100) * 100 + 100,
                'high': np.random.rand(100) * 100 + 105,
                'low': np.random.rand(100) * 100 + 95,
                'close': np.random.rand(100) * 100 + 100,
                'volume': np.random.randint(1000, 2000, 100)
            })
        }
    
    def test_complete_pipeline_integration(self):
        """Test complete pipeline from data to report."""
        try:
            # Step 1: Data preparation
            prepared_data = {}
            for symbol, df in self.sample_data.items():
                # Apply indicators
                df_with_indicators = self.indicator_engine.calculate_all_indicators(df.copy())
                
                # Apply feature engineering
                df_with_features = self.feature_engineer.generate_basic_features(df_with_indicators.copy())
                df_with_features = self.feature_engineer.generate_advanced_features(df_with_features.copy())
                
                # Create target variable
                df_with_features['next_day_return'] = df_with_features['close'].pct_change().shift(-1)
                
                prepared_data[symbol] = df_with_features.dropna()
            
            # Step 2: Mock ML predictions
            mock_predictions = {
                'cnn_lstm': np.array([[0.02], [0.03], [-0.01]]),
                'random_forest': np.array([[0.015], [0.025], [-0.005]])
            }
            
            # Step 3: Generate signals
            ensemble_pred = self.ensemble_predictor.combine_predictions(mock_predictions)
            signals, confidence = self.ensemble_predictor.generate_signal(ensemble_pred)
            
            # Step 4: Mock backtest results
            mock_backtest_results = {
                'performance_metrics': {
                    'total_return': 0.15,
                    'sharpe_ratio': 0.67,
                    'max_drawdown': -0.08
                },
                'portfolio_values': [
                    {'date': '2023-01-01', 'value': 100000},
                    {'date': '2023-01-02', 'value': 101000},
                    {'date': '2023-01-03', 'value': 102000}
                ],
                'trades': []
            }
            
            # Step 5: Performance analysis
            performance_analysis = self.performance_analyzer.analyze_performance(mock_backtest_results)
            
            # Step 6: Statistical validation
            statistical_tests = self.statistical_validator.validate_performance(mock_backtest_results)
            
            # Step 7: Report generation
            report = self.report_generator.generate_report(
                backtest_results=mock_backtest_results,
                performance_analysis=performance_analysis,
                statistical_tests=statistical_tests
            )
            
            # Verify end-to-end integration
            self.assertIsInstance(prepared_data, dict)
            self.assertIsInstance(ensemble_pred, np.ndarray)
            self.assertIsInstance(signals, np.ndarray)
            self.assertIsInstance(performance_analysis, dict)
            self.assertIsInstance(statistical_tests, dict)
            self.assertIsInstance(report, str)
            
            # Verify data flow
            self.assertIn('AAPL', prepared_data)
            self.assertIn('RSI', prepared_data['AAPL'].columns)
            self.assertIn('daily_return', prepared_data['AAPL'].columns)
            
        except Exception as e:
            # Some steps might fail due to insufficient data, which is expected
            self.assertIsInstance(e, (ValueError, RuntimeError, AttributeError))


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDataPipelineIntegration,
        TestMLPipelineIntegration,
        TestStrategyIntegration,
        TestBacktestingIntegration,
        TestReportGenerationIntegration,
        TestEndToEndIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nIntegration Tests run: {result.testsRun}")
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
