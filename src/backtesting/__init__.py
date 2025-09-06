"""
Main backtesting orchestrator that coordinates all backtesting components.
Provides a unified interface for running comprehensive backtesting analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import backtesting components
from .backtest_engine import BacktestEngine
from .performance_analyzer import PerformanceAnalyzer
from .statistical_validator import StatisticalValidator
from .walk_forward_analyzer import WalkForwardAnalyzer
from .report_generator import ReportGenerator, ReportConfig

# Import strategy components
from ..strategy.signal_generator import SignalGenerator
from ..strategy.position_sizer import PositionSizer
from ..strategy.risk_manager import RiskManager

# Import ML components
from ..ml_models.ensemble_predictor import EnsemblePredictor
from ..ml_models.data_preprocessor import DataPreprocessor

# Import data pipeline
from ..data_pipeline.feature_engineer import FeatureEngineer
from ..data_pipeline.indicator_engine import IndicatorEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Enumeration of backtesting modes."""
    SINGLE_RUN = "single_run"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    STRESS_TEST = "stress_test"


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    # Data settings
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    symbols: List[str] = None
    benchmark_symbol: str = "SPY"
    
    # Strategy settings
    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.0005
    
    # ML settings
    ml_models: List[str] = None
    feature_columns: List[str] = None
    target_column: str = "next_day_return"
    
    # Risk settings
    max_position_size: float = 0.05
    max_portfolio_risk: float = 0.20
    stop_loss: float = 0.02
    take_profit: float = 0.04
    
    # Backtesting settings
    mode: BacktestMode = BacktestMode.SINGLE_RUN
    train_window: int = 252
    test_window: int = 63
    step_size: int = 21
    
    # Output settings
    generate_report: bool = True
    save_results: bool = True
    results_directory: str = "results"
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        if self.ml_models is None:
            self.ml_models = ["cnn_lstm", "random_forest"]
        if self.feature_columns is None:
            self.feature_columns = ["RSI", "MACD", "ATR", "OBV", "daily_return"]


class BacktestOrchestrator:
    """Main orchestrator for comprehensive backtesting analysis."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results = {}
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all backtesting components."""
        try:
            # Strategy components
            self.signal_generator = SignalGenerator()
            self.position_sizer = PositionSizer()
            self.risk_manager = RiskManager(
                max_position_size_pct=self.config.max_position_size,
                max_portfolio_risk_pct=self.config.max_portfolio_risk,
                stop_loss_pct=self.config.stop_loss,
                take_profit_pct=self.config.take_profit
            )
            
            # ML components
            self.ensemble_predictor = EnsemblePredictor()
            self.data_preprocessor = DataPreprocessor()
            
            # Data pipeline components
            self.feature_engineer = FeatureEngineer()
            self.indicator_engine = IndicatorEngine()
            
            # Backtesting components
            self.backtest_engine = BacktestEngine(
                initial_capital=self.config.initial_capital,
                commission=self.config.commission,
                slippage=self.config.slippage
            )
            
            self.performance_analyzer = PerformanceAnalyzer()
            self.statistical_validator = StatisticalValidator()
            self.walk_forward_analyzer = WalkForwardAnalyzer()
            
            # Report generator
            self.report_generator = ReportGenerator(
                ReportConfig(title=f"Trading Strategy Backtest Report - {self.config.mode.value}")
            )
            
            logger.info("All backtesting components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def run_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run comprehensive backtesting analysis."""
        try:
            logger.info(f"Starting {self.config.mode.value} backtesting")
            
            if self.config.mode == BacktestMode.SINGLE_RUN:
                results = self._run_single_backtest(data)
            elif self.config.mode == BacktestMode.WALK_FORWARD:
                results = self._run_walk_forward_backtest(data)
            elif self.config.mode == BacktestMode.MONTE_CARLO:
                results = self._run_monte_carlo_backtest(data)
            elif self.config.mode == BacktestMode.STRESS_TEST:
                results = self._run_stress_test_backtest(data)
            else:
                raise ValueError(f"Unsupported backtest mode: {self.config.mode}")
            
            # Generate comprehensive report
            if self.config.generate_report:
                report = self._generate_comprehensive_report(results)
                results['report'] = report
            
            # Save results
            if self.config.save_results:
                self._save_results(results)
            
            logger.info("Backtesting completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            raise
    
    def _run_single_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run single backtest."""
        try:
            logger.info("Running single backtest")
            
            # Prepare data
            prepared_data = self._prepare_data(data)
            
            # Run backtest
            backtest_results = self.backtest_engine.run_backtest(
                data=prepared_data,
                signal_generator=self.signal_generator,
                position_sizer=self.position_sizer,
                risk_manager=self.risk_manager,
                ensemble_predictor=self.ensemble_predictor
            )
            
            # Analyze performance
            performance_analysis = self.performance_analyzer.analyze_performance(backtest_results)
            
            # Statistical validation
            statistical_tests = self.statistical_validator.validate_performance(backtest_results)
            
            # Compile results
            results = {
                'backtest_results': backtest_results,
                'performance_analysis': performance_analysis,
                'statistical_tests': statistical_tests,
                'config': self.config
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in single backtest: {e}")
            raise
    
    def _run_walk_forward_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run walk-forward backtest."""
        try:
            logger.info("Running walk-forward backtest")
            
            # Prepare data
            prepared_data = self._prepare_data(data)
            
            # Run walk-forward analysis
            walk_forward_results = self.walk_forward_analyzer.run_walk_forward_analysis(
                data=prepared_data,
                train_window=self.config.train_window,
                test_window=self.config.test_window,
                step_size=self.config.step_size,
                signal_generator=self.signal_generator,
                position_sizer=self.position_sizer,
                risk_manager=self.risk_manager,
                ensemble_predictor=self.ensemble_predictor
            )
            
            # Analyze performance
            performance_analysis = self.performance_analyzer.analyze_performance(walk_forward_results)
            
            # Statistical validation
            statistical_tests = self.statistical_validator.validate_performance(walk_forward_results)
            
            # Compile results
            results = {
                'walk_forward_results': walk_forward_results,
                'performance_analysis': performance_analysis,
                'statistical_tests': statistical_tests,
                'config': self.config
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in walk-forward backtest: {e}")
            raise
    
    def _run_monte_carlo_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run Monte Carlo backtest."""
        try:
            logger.info("Running Monte Carlo backtest")
            
            # Prepare data
            prepared_data = self._prepare_data(data)
            
            # Run multiple backtests with different random seeds
            monte_carlo_results = []
            num_simulations = 100
            
            for i in range(num_simulations):
                logger.info(f"Running simulation {i+1}/{num_simulations}")
                
                # Set random seed for reproducibility
                np.random.seed(i)
                
                # Run backtest
                backtest_results = self.backtest_engine.run_backtest(
                    data=prepared_data,
                    signal_generator=self.signal_generator,
                    position_sizer=self.position_sizer,
                    risk_manager=self.risk_manager,
                    ensemble_predictor=self.ensemble_predictor
                )
                
                monte_carlo_results.append(backtest_results)
            
            # Analyze Monte Carlo results
            performance_analysis = self._analyze_monte_carlo_results(monte_carlo_results)
            
            # Compile results
            results = {
                'monte_carlo_results': monte_carlo_results,
                'performance_analysis': performance_analysis,
                'config': self.config
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo backtest: {e}")
            raise
    
    def _run_stress_test_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run stress test backtest."""
        try:
            logger.info("Running stress test backtest")
            
            # Prepare data
            prepared_data = self._prepare_data(data)
            
            # Run stress tests
            stress_test_results = []
            
            # Test 1: High volatility scenario
            high_vol_data = self._simulate_high_volatility_scenario(prepared_data)
            high_vol_results = self.backtest_engine.run_backtest(
                data=high_vol_data,
                signal_generator=self.signal_generator,
                position_sizer=self.position_sizer,
                risk_manager=self.risk_manager,
                ensemble_predictor=self.ensemble_predictor
            )
            stress_test_results.append({
                'scenario': 'high_volatility',
                'results': high_vol_results
            })
            
            # Test 2: Market crash scenario
            crash_data = self._simulate_market_crash_scenario(prepared_data)
            crash_results = self.backtest_engine.run_backtest(
                data=crash_data,
                signal_generator=self.signal_generator,
                position_sizer=self.position_sizer,
                risk_manager=self.risk_manager,
                ensemble_predictor=self.ensemble_predictor
            )
            stress_test_results.append({
                'scenario': 'market_crash',
                'results': crash_results
            })
            
            # Test 3: Low liquidity scenario
            low_liquidity_data = self._simulate_low_liquidity_scenario(prepared_data)
            low_liquidity_results = self.backtest_engine.run_backtest(
                data=low_liquidity_data,
                signal_generator=self.signal_generator,
                position_sizer=self.position_sizer,
                risk_manager=self.risk_manager,
                ensemble_predictor=self.ensemble_predictor
            )
            stress_test_results.append({
                'scenario': 'low_liquidity',
                'results': low_liquidity_results
            })
            
            # Analyze stress test results
            performance_analysis = self._analyze_stress_test_results(stress_test_results)
            
            # Compile results
            results = {
                'stress_test_results': stress_test_results,
                'performance_analysis': performance_analysis,
                'config': self.config
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in stress test backtest: {e}")
            raise
    
    def _prepare_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare data for backtesting."""
        try:
            prepared_data = {}
            
            for symbol, df in data.items():
                if symbol not in self.config.symbols:
                    continue
                
                # Apply technical indicators
                df_with_indicators = self.indicator_engine.calculate_all_indicators(df.copy())
                
                # Apply feature engineering
                df_with_features = self.feature_engineer.generate_basic_features(df_with_indicators.copy())
                df_with_features = self.feature_engineer.generate_advanced_features(df_with_features.copy())
                
                # Create target variable
                df_with_features[self.config.target_column] = df_with_features['close'].pct_change().shift(-1)
                
                # Store prepared data
                prepared_data[symbol] = df_with_features.dropna()
            
            return prepared_data
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def _analyze_monte_carlo_results(self, monte_carlo_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze Monte Carlo backtest results."""
        try:
            # Extract performance metrics
            total_returns = []
            sharpe_ratios = []
            max_drawdowns = []
            
            for results in monte_carlo_results:
                performance_metrics = results.get('performance_metrics', {})
                total_returns.append(performance_metrics.get('total_return', 0))
                sharpe_ratios.append(performance_metrics.get('sharpe_ratio', 0))
                max_drawdowns.append(performance_metrics.get('max_drawdown', 0))
            
            # Calculate statistics
            analysis = {
                'total_return_stats': {
                    'mean': np.mean(total_returns),
                    'std': np.std(total_returns),
                    'min': np.min(total_returns),
                    'max': np.max(total_returns),
                    'percentile_5': np.percentile(total_returns, 5),
                    'percentile_95': np.percentile(total_returns, 95)
                },
                'sharpe_ratio_stats': {
                    'mean': np.mean(sharpe_ratios),
                    'std': np.std(sharpe_ratios),
                    'min': np.min(sharpe_ratios),
                    'max': np.max(sharpe_ratios),
                    'percentile_5': np.percentile(sharpe_ratios, 5),
                    'percentile_95': np.percentile(sharpe_ratios, 95)
                },
                'max_drawdown_stats': {
                    'mean': np.mean(max_drawdowns),
                    'std': np.std(max_drawdowns),
                    'min': np.min(max_drawdowns),
                    'max': np.max(max_drawdowns),
                    'percentile_5': np.percentile(max_drawdowns, 5),
                    'percentile_95': np.percentile(max_drawdowns, 95)
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing Monte Carlo results: {e}")
            return {}
    
    def _analyze_stress_test_results(self, stress_test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze stress test results."""
        try:
            analysis = {}
            
            for test in stress_test_results:
                scenario = test['scenario']
                results = test['results']
                
                performance_metrics = results.get('performance_metrics', {})
                
                analysis[scenario] = {
                    'total_return': performance_metrics.get('total_return', 0),
                    'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
                    'max_drawdown': performance_metrics.get('max_drawdown', 0),
                    'volatility': performance_metrics.get('volatility', 0)
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing stress test results: {e}")
            return {}
    
    def _simulate_high_volatility_scenario(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Simulate high volatility scenario."""
        try:
            high_vol_data = {}
            
            for symbol, df in data.items():
                df_copy = df.copy()
                
                # Increase volatility by 50%
                df_copy['high'] = df_copy['high'] * (1 + np.random.normal(0, 0.01, len(df_copy)))
                df_copy['low'] = df_copy['low'] * (1 + np.random.normal(0, 0.01, len(df_copy)))
                df_copy['close'] = df_copy['close'] * (1 + np.random.normal(0, 0.01, len(df_copy)))
                
                high_vol_data[symbol] = df_copy
            
            return high_vol_data
            
        except Exception as e:
            logger.error(f"Error simulating high volatility scenario: {e}")
            return data
    
    def _simulate_market_crash_scenario(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Simulate market crash scenario."""
        try:
            crash_data = {}
            
            for symbol, df in data.items():
                df_copy = df.copy()
                
                # Simulate 20% crash in the middle of the period
                crash_start = len(df_copy) // 2
                crash_end = crash_start + 30  # 30-day crash
                
                # Apply crash
                df_copy.loc[crash_start:crash_end, 'close'] *= 0.8
                df_copy.loc[crash_start:crash_end, 'high'] *= 0.8
                df_copy.loc[crash_start:crash_end, 'low'] *= 0.8
                
                crash_data[symbol] = df_copy
            
            return crash_data
            
        except Exception as e:
            logger.error(f"Error simulating market crash scenario: {e}")
            return data
    
    def _simulate_low_liquidity_scenario(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Simulate low liquidity scenario."""
        try:
            low_liquidity_data = {}
            
            for symbol, df in data.items():
                df_copy = df.copy()
                
                # Reduce volume by 70%
                df_copy['volume'] = df_copy['volume'] * 0.3
                
                # Increase bid-ask spread (simulated through higher volatility)
                df_copy['high'] = df_copy['high'] * (1 + np.random.normal(0, 0.005, len(df_copy)))
                df_copy['low'] = df_copy['low'] * (1 + np.random.normal(0, 0.005, len(df_copy)))
                
                low_liquidity_data[symbol] = df_copy
            
            return low_liquidity_data
            
        except Exception as e:
            logger.error(f"Error simulating low liquidity scenario: {e}")
            return data
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive report."""
        try:
            # Extract results based on mode
            if self.config.mode == BacktestMode.SINGLE_RUN:
                backtest_results = results.get('backtest_results', {})
                performance_analysis = results.get('performance_analysis', {})
                statistical_tests = results.get('statistical_tests', {})
                
                report = self.report_generator.generate_report(
                    backtest_results=backtest_results,
                    performance_analysis=performance_analysis,
                    statistical_tests=statistical_tests
                )
                
            elif self.config.mode == BacktestMode.WALK_FORWARD:
                walk_forward_results = results.get('walk_forward_results', {})
                performance_analysis = results.get('performance_analysis', {})
                statistical_tests = results.get('statistical_tests', {})
                
                report = self.report_generator.generate_report(
                    backtest_results=walk_forward_results,
                    performance_analysis=performance_analysis,
                    statistical_tests=statistical_tests
                )
                
            else:
                # For Monte Carlo and stress tests, create a summary report
                report = self._generate_summary_report(results)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return f"Error generating report: {e}"
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate summary report for Monte Carlo and stress tests."""
        try:
            report = f"""
# Trading Strategy Backtest Report - {self.config.mode.value.title()}

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

This report summarizes the results of {self.config.mode.value} backtesting analysis.

"""
            
            if self.config.mode == BacktestMode.MONTE_CARLO:
                performance_analysis = results.get('performance_analysis', {})
                
                report += "## Monte Carlo Analysis\n\n"
                
                # Total return statistics
                total_return_stats = performance_analysis.get('total_return_stats', {})
                report += f"**Total Return Statistics:**\n"
                report += f"- Mean: {total_return_stats.get('mean', 0):.2%}\n"
                report += f"- Std Dev: {total_return_stats.get('std', 0):.2%}\n"
                report += f"- 5th Percentile: {total_return_stats.get('percentile_5', 0):.2%}\n"
                report += f"- 95th Percentile: {total_return_stats.get('percentile_95', 0):.2%}\n\n"
                
                # Sharpe ratio statistics
                sharpe_stats = performance_analysis.get('sharpe_ratio_stats', {})
                report += f"**Sharpe Ratio Statistics:**\n"
                report += f"- Mean: {sharpe_stats.get('mean', 0):.2f}\n"
                report += f"- Std Dev: {sharpe_stats.get('std', 0):.2f}\n"
                report += f"- 5th Percentile: {sharpe_stats.get('percentile_5', 0):.2f}\n"
                report += f"- 95th Percentile: {sharpe_stats.get('percentile_95', 0):.2f}\n\n"
                
            elif self.config.mode == BacktestMode.STRESS_TEST:
                performance_analysis = results.get('performance_analysis', {})
                
                report += "## Stress Test Analysis\n\n"
                
                for scenario, metrics in performance_analysis.items():
                    report += f"**{scenario.replace('_', ' ').title()} Scenario:**\n"
                    report += f"- Total Return: {metrics.get('total_return', 0):.2%}\n"
                    report += f"- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}\n"
                    report += f"- Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n"
                    report += f"- Volatility: {metrics.get('volatility', 0):.2%}\n\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return f"Error generating summary report: {e}"
    
    def _save_results(self, results: Dict[str, Any]):
        """Save backtest results."""
        try:
            import os
            import json
            
            # Create results directory
            os.makedirs(self.config.results_directory, exist_ok=True)
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_filename = f"backtest_results_{self.config.mode.value}_{timestamp}.json"
            results_path = os.path.join(self.config.results_directory, results_filename)
            
            # Convert results to JSON-serializable format
            json_results = self._convert_results_to_json(results)
            
            with open(results_path, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _convert_results_to_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert results to JSON-serializable format."""
        try:
            json_results = {}
            
            for key, value in results.items():
                if key == 'config':
                    # Convert config to dict
                    json_results[key] = value.__dict__
                elif isinstance(value, dict):
                    json_results[key] = value
                elif isinstance(value, list):
                    json_results[key] = value
                else:
                    json_results[key] = str(value)
            
            return json_results
            
        except Exception as e:
            logger.error(f"Error converting results to JSON: {e}")
            return results


# Utility functions
def run_backtest(data: Dict[str, pd.DataFrame], 
                config: Optional[BacktestConfig] = None) -> Dict[str, Any]:
    """Quick function to run a backtest."""
    if config is None:
        config = BacktestConfig()
    
    orchestrator = BacktestOrchestrator(config)
    return orchestrator.run_backtest(data)


def create_backtest_config(mode: BacktestMode = BacktestMode.SINGLE_RUN) -> BacktestConfig:
    """Create a backtest configuration."""
    return BacktestConfig(mode=mode)


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    symbols = ["AAPL", "MSFT", "GOOGL"]
    data = {}
    
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y")
        data[symbol] = df
    
    # Create backtest configuration
    config = BacktestConfig(
        mode=BacktestMode.SINGLE_RUN,
        symbols=symbols,
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    # Run backtest
    orchestrator = BacktestOrchestrator(config)
    results = orchestrator.run_backtest(data)
    
    print("Backtest completed successfully!")
    print(f"Results keys: {list(results.keys())}")
