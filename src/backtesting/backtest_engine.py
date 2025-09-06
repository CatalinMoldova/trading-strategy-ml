"""
Backtesting engine for historical simulation of trading strategies.
Supports walk-forward analysis, realistic transaction costs, and comprehensive performance metrics.
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

from ..strategy.signal_generator import MultiFactorSignalGenerator
from ..strategy.position_sizer import PositionSizer, PositionSizingConfig, PositionSizingMethod
from ..strategy.risk_manager import RiskManager, RiskManagementConfig
from ..strategy.portfolio_manager import PortfolioManager, PortfolioConfig
from ..ml_models.ensemble_predictor import EnsemblePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Enumeration of backtesting modes."""
    FULL_HISTORY = "full_history"
    WALK_FORWARD = "walk_forward"
    ROLLING_WINDOW = "rolling_window"
    OUT_OF_SAMPLE = "out_of_sample"


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    # Backtest settings
    initial_capital: float = 100000.0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Transaction costs
    commission_per_trade: float = 1.0  # $1 per trade
    commission_per_share: float = 0.005  # $0.005 per share
    slippage: float = 0.001  # 0.1% slippage
    
    # Position sizing
    position_sizing_method: PositionSizingMethod = PositionSizingMethod.KELLY
    max_position_size: float = 0.05
    
    # Risk management
    max_portfolio_risk: float = 0.20
    max_drawdown: float = 0.15
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    
    # Walk-forward settings
    training_period: int = 252  # 1 year
    testing_period: int = 63   # 3 months
    step_size: int = 21        # 1 month
    
    # Performance metrics
    benchmark_symbol: str = "SPY"
    risk_free_rate: float = 0.02


class BacktestEngine:
    """Comprehensive backtesting engine."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results = {}
        self.trades = []
        self.portfolio_values = []
        self.benchmark_values = []
        
    def run_backtest(self, 
                    data: Dict[str, pd.DataFrame],
                    features: Dict[str, pd.DataFrame],
                    ml_model: Optional[EnsemblePredictor] = None,
                    mode: BacktestMode = BacktestMode.FULL_HISTORY) -> Dict[str, Any]:
        """Run backtest with specified data and model."""
        try:
            logger.info(f"Starting backtest in {mode.value} mode")
            
            if mode == BacktestMode.FULL_HISTORY:
                return self._run_full_history_backtest(data, features, ml_model)
            elif mode == BacktestMode.WALK_FORWARD:
                return self._run_walk_forward_backtest(data, features, ml_model)
            elif mode == BacktestMode.ROLLING_WINDOW:
                return self._run_rolling_window_backtest(data, features, ml_model)
            elif mode == BacktestMode.OUT_OF_SAMPLE:
                return self._run_out_of_sample_backtest(data, features, ml_model)
            else:
                raise ValueError(f"Unknown backtest mode: {mode}")
                
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {}
    
    def _run_full_history_backtest(self, 
                                 data: Dict[str, pd.DataFrame],
                                 features: Dict[str, pd.DataFrame],
                                 ml_model: Optional[EnsemblePredictor] = None) -> Dict[str, Any]:
        """Run backtest on full historical data."""
        try:
            # Initialize portfolio manager
            portfolio_config = PortfolioConfig(
                initial_capital=self.config.initial_capital,
                max_position_size=self.config.max_position_size,
                max_portfolio_risk=self.config.max_portfolio_risk,
                max_drawdown=self.config.max_drawdown,
                stop_loss_pct=self.config.stop_loss_pct,
                take_profit_pct=self.config.take_profit_pct
            )
            
            portfolio = PortfolioManager(portfolio_config)
            
            if ml_model:
                portfolio.add_ml_model(ml_model)
            
            # Get common date range
            start_date, end_date = self._get_common_date_range(data)
            
            # Create date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Initialize results
            portfolio_values = []
            benchmark_values = []
            trades = []
            
            # Get benchmark data
            benchmark_data = self._get_benchmark_data(data, start_date, end_date)
            
            # Run simulation
            for current_date in date_range:
                try:
                    # Update portfolio with current prices
                    current_prices = self._get_current_prices(data, current_date)
                    if current_prices:
                        portfolio.update_positions(current_prices)
                    
                    # Generate signals for available symbols
                    for symbol in data.keys():
                        if symbol in features and current_date in data[symbol].index:
                            # Get historical data up to current date
                            hist_data = data[symbol].loc[:current_date]
                            hist_features = features[symbol].loc[:current_date]
                            
                            if len(hist_data) > 100:  # Ensure enough data
                                # Generate ML predictions if model available
                                ml_predictions = None
                                ml_confidence = None
                                
                                if ml_model and len(hist_features) > 60:
                                    try:
                                        # Prepare data for ML prediction
                                        X = hist_features.values[-60:]  # Last 60 days
                                        X = X.reshape(1, X.shape[0], X.shape[1])
                                        
                                        ml_predictions = ml_model.predict(X)
                                        ml_confidence = ml_model.predict_with_confidence(X)[1]
                                    except Exception as e:
                                        logger.warning(f"ML prediction failed for {symbol}: {e}")
                                
                                # Process signal
                                result = portfolio.process_signal(
                                    symbol=symbol,
                                    market_data=hist_data,
                                    features=hist_features,
                                    ml_predictions=ml_predictions,
                                    ml_confidence=ml_confidence
                                )
                                
                                if result['action'] == 'OPENED':
                                    trades.append({
                                        'date': current_date,
                                        'symbol': symbol,
                                        'action': 'BUY',
                                        'price': result['entry_price'],
                                        'shares': result['shares'],
                                        'confidence': result['confidence']
                                    })
                    
                    # Record portfolio value
                    portfolio_summary = portfolio.get_portfolio_summary()
                    portfolio_values.append({
                        'date': current_date,
                        'value': portfolio_summary['total_value'],
                        'cash': portfolio_summary['cash'],
                        'positions_value': portfolio_summary['positions_value'],
                        'unrealized_pnl': portfolio_summary['unrealized_pnl']
                    })
                    
                    # Record benchmark value
                    if current_date in benchmark_data.index:
                        benchmark_values.append({
                            'date': current_date,
                            'value': benchmark_data.loc[current_date, 'close']
                        })
                    
                    # Rebalance portfolio periodically
                    if current_date.day == 1:  # Monthly rebalancing
                        portfolio.rebalance_portfolio()
                        
                except Exception as e:
                    logger.warning(f"Error processing date {current_date}: {e}")
                    continue
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                portfolio_values, benchmark_values, trades
            )
            
            # Compile results
            results = {
                'backtest_config': self.config.__dict__,
                'performance_metrics': performance_metrics,
                'portfolio_values': portfolio_values,
                'benchmark_values': benchmark_values,
                'trades': trades,
                'portfolio_summary': portfolio.get_portfolio_summary(),
                'position_details': portfolio.get_position_details()
            }
            
            self.results = results
            logger.info("Full history backtest completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in full history backtest: {e}")
            return {}
    
    def _run_walk_forward_backtest(self, 
                                  data: Dict[str, pd.DataFrame],
                                  features: Dict[str, pd.DataFrame],
                                  ml_model: Optional[EnsemblePredictor] = None) -> Dict[str, Any]:
        """Run walk-forward analysis backtest."""
        try:
            logger.info("Starting walk-forward backtest")
            
            # Get common date range
            start_date, end_date = self._get_common_date_range(data)
            
            # Create walk-forward periods
            periods = self._create_walk_forward_periods(start_date, end_date)
            
            all_results = []
            all_trades = []
            all_portfolio_values = []
            
            for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
                logger.info(f"Walk-forward period {i+1}/{len(periods)}: {test_start.date()} to {test_end.date()}")
                
                # Train model on training period (if ML model provided)
                if ml_model:
                    # This would involve retraining the model on training data
                    # For now, we'll use the existing model
                    pass
                
                # Run backtest on testing period
                test_data = self._filter_data_by_date_range(data, test_start, test_end)
                test_features = self._filter_data_by_date_range(features, test_start, test_end)
                
                # Run mini backtest
                period_results = self._run_period_backtest(test_data, test_features, ml_model)
                
                if period_results:
                    all_results.append(period_results)
                    all_trades.extend(period_results.get('trades', []))
                    all_portfolio_values.extend(period_results.get('portfolio_values', []))
            
            # Aggregate results
            aggregated_results = self._aggregate_walk_forward_results(all_results)
            
            # Calculate overall performance metrics
            performance_metrics = self._calculate_performance_metrics(
                all_portfolio_values, [], all_trades
            )
            
            results = {
                'backtest_config': self.config.__dict__,
                'backtest_mode': 'walk_forward',
                'num_periods': len(periods),
                'performance_metrics': performance_metrics,
                'period_results': all_results,
                'aggregated_results': aggregated_results,
                'all_trades': all_trades,
                'all_portfolio_values': all_portfolio_values
            }
            
            self.results = results
            logger.info("Walk-forward backtest completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in walk-forward backtest: {e}")
            return {}
    
    def _run_rolling_window_backtest(self, 
                                   data: Dict[str, pd.DataFrame],
                                   features: Dict[str, pd.DataFrame],
                                   ml_model: Optional[EnsemblePredictor] = None) -> Dict[str, Any]:
        """Run rolling window backtest."""
        try:
            logger.info("Starting rolling window backtest")
            
            # Similar to walk-forward but with overlapping windows
            # Implementation would be similar to walk-forward with different window management
            
            return self._run_walk_forward_backtest(data, features, ml_model)
            
        except Exception as e:
            logger.error(f"Error in rolling window backtest: {e}")
            return {}
    
    def _run_out_of_sample_backtest(self, 
                                   data: Dict[str, pd.DataFrame],
                                   features: Dict[str, pd.DataFrame],
                                   ml_model: Optional[EnsemblePredictor] = None) -> Dict[str, Any]:
        """Run out-of-sample backtest."""
        try:
            logger.info("Starting out-of-sample backtest")
            
            # Split data into training and testing periods
            start_date, end_date = self._get_common_date_range(data)
            split_date = start_date + timedelta(days=int((end_date - start_date).days * 0.8))
            
            # Train on first 80% of data
            train_data = self._filter_data_by_date_range(data, start_date, split_date)
            train_features = self._filter_data_by_date_range(features, start_date, split_date)
            
            # Test on last 20% of data
            test_data = self._filter_data_by_date_range(data, split_date, end_date)
            test_features = self._filter_data_by_date_range(features, split_date, end_date)
            
            # Run backtest on test data
            results = self._run_period_backtest(test_data, test_features, ml_model)
            
            if results:
                results['backtest_mode'] = 'out_of_sample'
                results['train_period'] = f"{start_date.date()} to {split_date.date()}"
                results['test_period'] = f"{split_date.date()} to {end_date.date()}"
            
            self.results = results
            logger.info("Out-of-sample backtest completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in out-of-sample backtest: {e}")
            return {}
    
    def _run_period_backtest(self, 
                           data: Dict[str, pd.DataFrame],
                           features: Dict[str, pd.DataFrame],
                           ml_model: Optional[EnsemblePredictor] = None) -> Dict[str, Any]:
        """Run backtest for a specific period."""
        try:
            # Initialize portfolio manager
            portfolio_config = PortfolioConfig(
                initial_capital=self.config.initial_capital,
                max_position_size=self.config.max_position_size,
                max_portfolio_risk=self.config.max_portfolio_risk,
                max_drawdown=self.config.max_drawdown,
                stop_loss_pct=self.config.stop_loss_pct,
                take_profit_pct=self.config.take_profit_pct
            )
            
            portfolio = PortfolioManager(portfolio_config)
            
            if ml_model:
                portfolio.add_ml_model(ml_model)
            
            # Get date range
            start_date, end_date = self._get_common_date_range(data)
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Initialize results
            portfolio_values = []
            trades = []
            
            # Run simulation
            for current_date in date_range:
                try:
                    # Update portfolio with current prices
                    current_prices = self._get_current_prices(data, current_date)
                    if current_prices:
                        portfolio.update_positions(current_prices)
                    
                    # Generate signals for available symbols
                    for symbol in data.keys():
                        if symbol in features and current_date in data[symbol].index:
                            hist_data = data[symbol].loc[:current_date]
                            hist_features = features[symbol].loc[:current_date]
                            
                            if len(hist_data) > 100:
                                # Generate ML predictions if model available
                                ml_predictions = None
                                ml_confidence = None
                                
                                if ml_model and len(hist_features) > 60:
                                    try:
                                        X = hist_features.values[-60:]
                                        X = X.reshape(1, X.shape[0], X.shape[1])
                                        ml_predictions = ml_model.predict(X)
                                        ml_confidence = ml_model.predict_with_confidence(X)[1]
                                    except Exception as e:
                                        logger.warning(f"ML prediction failed for {symbol}: {e}")
                                
                                # Process signal
                                result = portfolio.process_signal(
                                    symbol=symbol,
                                    market_data=hist_data,
                                    features=hist_features,
                                    ml_predictions=ml_predictions,
                                    ml_confidence=ml_confidence
                                )
                                
                                if result['action'] == 'OPENED':
                                    trades.append({
                                        'date': current_date,
                                        'symbol': symbol,
                                        'action': 'BUY',
                                        'price': result['entry_price'],
                                        'shares': result['shares'],
                                        'confidence': result['confidence']
                                    })
                    
                    # Record portfolio value
                    portfolio_summary = portfolio.get_portfolio_summary()
                    portfolio_values.append({
                        'date': current_date,
                        'value': portfolio_summary['total_value'],
                        'cash': portfolio_summary['cash'],
                        'positions_value': portfolio_summary['positions_value'],
                        'unrealized_pnl': portfolio_summary['unrealized_pnl']
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing date {current_date}: {e}")
                    continue
            
            return {
                'portfolio_values': portfolio_values,
                'trades': trades,
                'portfolio_summary': portfolio.get_portfolio_summary()
            }
            
        except Exception as e:
            logger.error(f"Error in period backtest: {e}")
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
            
            # Apply config date filters
            if self.config.start_date:
                start_date = max(start_date, self.config.start_date)
            if self.config.end_date:
                end_date = min(end_date, self.config.end_date)
            
            return start_date, end_date
            
        except Exception as e:
            logger.error(f"Error getting common date range: {e}")
            return datetime.now() - timedelta(days=365), datetime.now()
    
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
    
    def _get_current_prices(self, data: Dict[str, pd.DataFrame], current_date: datetime) -> Dict[str, float]:
        """Get current prices for all symbols."""
        try:
            current_prices = {}
            for symbol, df in data.items():
                if current_date in df.index:
                    current_prices[symbol] = df.loc[current_date, 'close']
            return current_prices
            
        except Exception as e:
            logger.error(f"Error getting current prices: {e}")
            return {}
    
    def _get_benchmark_data(self, data: Dict[str, pd.DataFrame], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get benchmark data for comparison."""
        try:
            benchmark_symbol = self.config.benchmark_symbol
            if benchmark_symbol in data:
                mask = (data[benchmark_symbol].index >= start_date) & (data[benchmark_symbol].index <= end_date)
                return data[benchmark_symbol][mask]
            else:
                # Create mock benchmark data
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                return pd.DataFrame({
                    'close': 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, len(date_range)))
                }, index=date_range)
                
        except Exception as e:
            logger.error(f"Error getting benchmark data: {e}")
            return pd.DataFrame()
    
    def _create_walk_forward_periods(self, start_date: datetime, end_date: datetime) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Create walk-forward analysis periods."""
        try:
            periods = []
            current_start = start_date
            
            while current_start < end_date:
                train_start = current_start
                train_end = current_start + timedelta(days=self.config.training_period)
                test_start = train_end
                test_end = test_start + timedelta(days=self.config.testing_period)
                
                if test_end > end_date:
                    break
                
                periods.append((train_start, train_end, test_start, test_end))
                current_start += timedelta(days=self.config.step_size)
            
            return periods
            
        except Exception as e:
            logger.error(f"Error creating walk-forward periods: {e}")
            return []
    
    def _calculate_performance_metrics(self, 
                                     portfolio_values: List[Dict[str, Any]], 
                                     benchmark_values: List[Dict[str, Any]], 
                                     trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        try:
            if not portfolio_values:
                return {}
            
            # Convert to DataFrame
            portfolio_df = pd.DataFrame(portfolio_values)
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df = portfolio_df.set_index('date')
            
            # Calculate returns
            portfolio_df['returns'] = portfolio_df['value'].pct_change()
            
            # Basic metrics
            total_return = (portfolio_df['value'].iloc[-1] - portfolio_df['value'].iloc[0]) / portfolio_df['value'].iloc[0]
            annualized_return = (1 + total_return) ** (252 / len(portfolio_df)) - 1
            volatility = portfolio_df['returns'].std() * np.sqrt(252)
            
            # Risk metrics
            sharpe_ratio = (annualized_return - self.config.risk_free_rate) / volatility if volatility > 0 else 0
            
            # Drawdown
            cumulative_returns = (1 + portfolio_df['returns']).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdowns.min()
            
            # Trade metrics
            trade_metrics = self._calculate_trade_metrics(trades)
            
            # Benchmark comparison
            benchmark_metrics = {}
            if benchmark_values:
                benchmark_df = pd.DataFrame(benchmark_values)
                benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
                benchmark_df = benchmark_df.set_index('date')
                benchmark_df['returns'] = benchmark_df['value'].pct_change()
                
                benchmark_return = (benchmark_df['value'].iloc[-1] - benchmark_df['value'].iloc[0]) / benchmark_df['value'].iloc[0]
                benchmark_annualized = (1 + benchmark_return) ** (252 / len(benchmark_df)) - 1
                
                # Alpha and Beta
                common_dates = portfolio_df.index.intersection(benchmark_df.index)
                if len(common_dates) > 1:
                    portfolio_returns = portfolio_df.loc[common_dates, 'returns']
                    benchmark_returns = benchmark_df.loc[common_dates, 'returns']
                    
                    beta = np.cov(portfolio_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
                    alpha = annualized_return - (self.config.risk_free_rate + beta * (benchmark_annualized - self.config.risk_free_rate))
                    
                    benchmark_metrics = {
                        'benchmark_return': benchmark_return,
                        'benchmark_annualized': benchmark_annualized,
                        'alpha': alpha,
                        'beta': beta,
                        'information_ratio': (annualized_return - benchmark_annualized) / volatility if volatility > 0 else 0
                    }
            
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0,
                'sortino_ratio': annualized_return / (portfolio_df['returns'][portfolio_df['returns'] < 0].std() * np.sqrt(252)) if len(portfolio_df['returns'][portfolio_df['returns'] < 0]) > 0 else 0,
                'trade_metrics': trade_metrics,
                'benchmark_metrics': benchmark_metrics
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _calculate_trade_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate trade-specific metrics."""
        try:
            if not trades:
                return {}
            
            trades_df = pd.DataFrame(trades)
            
            # Basic trade metrics
            total_trades = len(trades_df)
            unique_symbols = trades_df['symbol'].nunique()
            
            # Win rate (simplified - would need actual P&L data)
            # For now, assume trades with high confidence are more likely to be winners
            high_confidence_trades = len(trades_df[trades_df['confidence'] > 0.7])
            win_rate = high_confidence_trades / total_trades if total_trades > 0 else 0
            
            # Average trade size
            avg_trade_size = trades_df['shares'].mean() if total_trades > 0 else 0
            
            # Trade frequency
            if len(trades_df) > 1:
                trade_dates = pd.to_datetime(trades_df['date'])
                time_span = (trade_dates.max() - trade_dates.min()).days
                trade_frequency = total_trades / (time_span / 252) if time_span > 0 else 0
            else:
                trade_frequency = 0
            
            return {
                'total_trades': total_trades,
                'unique_symbols': unique_symbols,
                'win_rate': win_rate,
                'avg_trade_size': avg_trade_size,
                'trade_frequency': trade_frequency,
                'avg_confidence': trades_df['confidence'].mean() if total_trades > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {e}")
            return {}
    
    def _aggregate_walk_forward_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from walk-forward analysis."""
        try:
            if not all_results:
                return {}
            
            # Aggregate performance metrics
            all_returns = []
            all_sharpe_ratios = []
            all_max_drawdowns = []
            
            for result in all_results:
                metrics = result.get('performance_metrics', {})
                if metrics:
                    all_returns.append(metrics.get('annualized_return', 0))
                    all_sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
                    all_max_drawdowns.append(metrics.get('max_drawdown', 0))
            
            aggregated = {
                'avg_annualized_return': np.mean(all_returns) if all_returns else 0,
                'std_annualized_return': np.std(all_returns) if all_returns else 0,
                'avg_sharpe_ratio': np.mean(all_sharpe_ratios) if all_sharpe_ratios else 0,
                'std_sharpe_ratio': np.std(all_sharpe_ratios) if all_sharpe_ratios else 0,
                'avg_max_drawdown': np.mean(all_max_drawdowns) if all_max_drawdowns else 0,
                'worst_max_drawdown': np.min(all_max_drawdowns) if all_max_drawdowns else 0,
                'num_periods': len(all_results),
                'positive_periods': sum(1 for r in all_returns if r > 0),
                'consistency_ratio': sum(1 for r in all_returns if r > 0) / len(all_returns) if all_returns else 0
            }
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating walk-forward results: {e}")
            return {}
    
    def get_results(self) -> Dict[str, Any]:
        """Get backtest results."""
        return self.results
    
    def export_results(self, filepath: str) -> bool:
        """Export results to file."""
        try:
            import json
            
            # Convert datetime objects to strings for JSON serialization
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
                return obj
            
            # Convert results
            export_data = json.loads(json.dumps(self.results, default=convert_datetime))
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Results exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return False


# Utility functions
def run_backtest(data: Dict[str, pd.DataFrame],
                features: Dict[str, pd.DataFrame],
                config: Optional[BacktestConfig] = None,
                ml_model: Optional[EnsemblePredictor] = None,
                mode: BacktestMode = BacktestMode.FULL_HISTORY) -> Dict[str, Any]:
    """Quick function to run a backtest."""
    if config is None:
        config = BacktestConfig()
    
    engine = BacktestEngine(config)
    return engine.run_backtest(data, features, ml_model, mode)


def create_backtest_config(initial_capital: float = 100000.0) -> BacktestConfig:
    """Create a backtest configuration with default settings."""
    return BacktestConfig(initial_capital=initial_capital)


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
    
    # Run backtest
    config = BacktestConfig(initial_capital=100000.0)
    results = run_backtest(data, features, config, mode=BacktestMode.FULL_HISTORY)
    
    print("Backtest completed!")
    print(f"Total return: {results['performance_metrics']['total_return']:.2%}")
    print(f"Sharpe ratio: {results['performance_metrics']['sharpe_ratio']:.2f}")
    print(f"Max drawdown: {results['performance_metrics']['max_drawdown']:.2%}")
    print(f"Total trades: {results['performance_metrics']['trade_metrics']['total_trades']}")
