"""
Performance analyzer for comprehensive evaluation of trading strategy results.
Calculates advanced metrics, risk-adjusted returns, and statistical significance tests.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Enumeration of performance metrics."""
    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VAR = "var"
    CVAR = "cvar"
    ALPHA = "alpha"
    BETA = "beta"
    INFORMATION_RATIO = "information_ratio"
    TREYNOR_RATIO = "treynor_ratio"


@dataclass
class PerformanceConfig:
    """Configuration for performance analysis."""
    risk_free_rate: float = 0.02
    benchmark_symbol: str = "SPY"
    confidence_level: float = 0.95
    lookback_period: int = 252
    bootstrap_samples: int = 1000
    monte_carlo_simulations: int = 10000


class PerformanceAnalyzer:
    """Comprehensive performance analysis system."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.results = {}
        
    def analyze_performance(self, 
                          portfolio_values: List[Dict[str, Any]], 
                          benchmark_values: Optional[List[Dict[str, Any]]] = None,
                          trades: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Perform comprehensive performance analysis."""
        try:
            logger.info("Starting performance analysis")
            
            # Convert to DataFrames
            portfolio_df = self._prepare_portfolio_data(portfolio_values)
            benchmark_df = self._prepare_benchmark_data(benchmark_values) if benchmark_values else None
            
            # Calculate returns
            portfolio_df = self._calculate_returns(portfolio_df)
            if benchmark_df is not None:
                benchmark_df = self._calculate_returns(benchmark_df)
            
            # Basic performance metrics
            basic_metrics = self._calculate_basic_metrics(portfolio_df)
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(portfolio_df)
            
            # Risk-adjusted metrics
            risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(portfolio_df, benchmark_df)
            
            # Drawdown analysis
            drawdown_analysis = self._analyze_drawdowns(portfolio_df)
            
            # Trade analysis
            trade_analysis = self._analyze_trades(trades) if trades else {}
            
            # Benchmark comparison
            benchmark_comparison = self._compare_with_benchmark(portfolio_df, benchmark_df) if benchmark_df is not None else {}
            
            # Statistical tests
            statistical_tests = self._perform_statistical_tests(portfolio_df, benchmark_df)
            
            # Monte Carlo analysis
            monte_carlo_analysis = self._perform_monte_carlo_analysis(portfolio_df)
            
            # Compile results
            results = {
                'basic_metrics': basic_metrics,
                'risk_metrics': risk_metrics,
                'risk_adjusted_metrics': risk_adjusted_metrics,
                'drawdown_analysis': drawdown_analysis,
                'trade_analysis': trade_analysis,
                'benchmark_comparison': benchmark_comparison,
                'statistical_tests': statistical_tests,
                'monte_carlo_analysis': monte_carlo_analysis,
                'analysis_timestamp': datetime.now()
            }
            
            self.results = results
            logger.info("Performance analysis completed")
            return results
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {e}")
            return {}
    
    def _prepare_portfolio_data(self, portfolio_values: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare portfolio data for analysis."""
        try:
            df = pd.DataFrame(portfolio_values)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.sort_index()
            return df
            
        except Exception as e:
            logger.error(f"Error preparing portfolio data: {e}")
            return pd.DataFrame()
    
    def _prepare_benchmark_data(self, benchmark_values: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare benchmark data for analysis."""
        try:
            df = pd.DataFrame(benchmark_values)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.sort_index()
            return df
            
        except Exception as e:
            logger.error(f"Error preparing benchmark data: {e}")
            return pd.DataFrame()
    
    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns from price data."""
        try:
            if 'value' in df.columns:
                df['returns'] = df['value'].pct_change()
            elif 'close' in df.columns:
                df['returns'] = df['close'].pct_change()
            else:
                logger.warning("No price column found for returns calculation")
                return df
            
            # Calculate cumulative returns
            df['cumulative_returns'] = (1 + df['returns']).cumprod()
            
            # Calculate log returns
            df['log_returns'] = np.log(df['cumulative_returns'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return df
    
    def _calculate_basic_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        try:
            if df.empty or 'returns' not in df.columns:
                return {}
            
            returns = df['returns'].dropna()
            
            # Total return
            total_return = (df['value'].iloc[-1] - df['value'].iloc[0]) / df['value'].iloc[0] if 'value' in df.columns else 0
            
            # Annualized return
            num_periods = len(returns)
            annualized_return = (1 + total_return) ** (252 / num_periods) - 1 if num_periods > 0 else 0
            
            # Volatility
            volatility = returns.std() * np.sqrt(252)
            
            # Mean return
            mean_return = returns.mean() * 252
            
            # Skewness and Kurtosis
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Best and worst days
            best_day = returns.max()
            worst_day = returns.min()
            
            # Positive and negative days
            positive_days = (returns > 0).sum()
            negative_days = (returns < 0).sum()
            win_rate = positive_days / len(returns) if len(returns) > 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'mean_return': mean_return,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'best_day': best_day,
                'worst_day': worst_day,
                'positive_days': positive_days,
                'negative_days': negative_days,
                'win_rate': win_rate,
                'total_days': len(returns)
            }
            
        except Exception as e:
            logger.error(f"Error calculating basic metrics: {e}")
            return {}
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk-related metrics."""
        try:
            if df.empty or 'returns' not in df.columns:
                return {}
            
            returns = df['returns'].dropna()
            
            # Value at Risk (VaR)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Conditional Value at Risk (CVaR)
            cvar_95 = returns[returns <= var_95].mean()
            cvar_99 = returns[returns <= var_99].mean()
            
            # Expected Shortfall
            expected_shortfall = cvar_95
            
            # Downside deviation
            downside_returns = returns[returns < 0]
            downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252) if len(downside_returns) > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdowns.min()
            
            # Drawdown duration
            drawdown_periods = self._calculate_drawdown_periods(drawdowns)
            max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
            avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
            
            # Tail ratio
            tail_ratio = abs(var_95) / abs(var_99) if var_99 != 0 else 0
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'expected_shortfall': expected_shortfall,
                'downside_deviation': downside_deviation,
                'max_drawdown': max_drawdown,
                'max_drawdown_duration': max_drawdown_duration,
                'avg_drawdown_duration': avg_drawdown_duration,
                'tail_ratio': tail_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_risk_adjusted_metrics(self, 
                                       portfolio_df: pd.DataFrame, 
                                       benchmark_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        try:
            if portfolio_df.empty or 'returns' not in portfolio_df.columns:
                return {}
            
            portfolio_returns = portfolio_df['returns'].dropna()
            
            # Sharpe ratio
            excess_returns = portfolio_returns - self.config.risk_free_rate / 252
            sharpe_ratio = excess_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
            
            # Sortino ratio
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
            
            # Calmar ratio
            annualized_return = (1 + portfolio_returns.mean()) ** 252 - 1
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Treynor ratio (requires beta)
            treynor_ratio = 0
            if benchmark_df is not None and 'returns' in benchmark_df.columns:
                benchmark_returns = benchmark_df['returns'].dropna()
                beta = self._calculate_beta(portfolio_returns, benchmark_returns)
                treynor_ratio = excess_returns.mean() / beta * np.sqrt(252) if beta != 0 else 0
            
            # Information ratio
            information_ratio = 0
            if benchmark_df is not None and 'returns' in benchmark_df.columns:
                benchmark_returns = benchmark_df['returns'].dropna()
                common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
                if len(common_dates) > 1:
                    portfolio_aligned = portfolio_returns.loc[common_dates]
                    benchmark_aligned = benchmark_returns.loc[common_dates]
                    active_returns = portfolio_aligned - benchmark_aligned
                    tracking_error = active_returns.std() * np.sqrt(252)
                    information_ratio = active_returns.mean() / tracking_error * np.sqrt(252) if tracking_error > 0 else 0
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'treynor_ratio': treynor_ratio,
                'information_ratio': information_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted metrics: {e}")
            return {}
    
    def _analyze_drawdowns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze drawdown patterns."""
        try:
            if df.empty or 'returns' not in df.columns:
                return {}
            
            returns = df['returns'].dropna()
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            
            # Find drawdown periods
            drawdown_periods = []
            in_drawdown = False
            start_date = None
            
            for date, dd in drawdowns.items():
                if dd < -0.001 and not in_drawdown:  # Start of drawdown
                    in_drawdown = True
                    start_date = date
                elif dd >= -0.001 and in_drawdown:  # End of drawdown
                    in_drawdown = False
                    if start_date:
                        period_length = (date - start_date).days
                        max_dd_in_period = drawdowns.loc[start_date:date].min()
                        drawdown_periods.append({
                            'start_date': start_date,
                            'end_date': date,
                            'duration_days': period_length,
                            'max_drawdown': max_dd_in_period
                        })
            
            # Calculate statistics
            if drawdown_periods:
                durations = [p['duration_days'] for p in drawdown_periods]
                max_drawdowns = [p['max_drawdown'] for p in drawdown_periods]
                
                analysis = {
                    'num_drawdowns': len(drawdown_periods),
                    'max_drawdown': min(max_drawdowns),
                    'avg_drawdown': np.mean(max_drawdowns),
                    'max_duration': max(durations),
                    'avg_duration': np.mean(durations),
                    'drawdown_periods': drawdown_periods
                }
            else:
                analysis = {
                    'num_drawdowns': 0,
                    'max_drawdown': 0,
                    'avg_drawdown': 0,
                    'max_duration': 0,
                    'avg_duration': 0,
                    'drawdown_periods': []
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing drawdowns: {e}")
            return {}
    
    def _analyze_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trading activity and performance."""
        try:
            if not trades:
                return {}
            
            trades_df = pd.DataFrame(trades)
            
            # Basic trade statistics
            total_trades = len(trades_df)
            unique_symbols = trades_df['symbol'].nunique()
            
            # Trade frequency
            if len(trades_df) > 1:
                trade_dates = pd.to_datetime(trades_df['date'])
                time_span = (trade_dates.max() - trade_dates.min()).days
                trades_per_year = total_trades / (time_span / 252) if time_span > 0 else 0
            else:
                trades_per_year = 0
            
            # Symbol concentration
            symbol_counts = trades_df['symbol'].value_counts()
            top_symbol = symbol_counts.index[0] if len(symbol_counts) > 0 else None
            top_symbol_pct = symbol_counts.iloc[0] / total_trades if total_trades > 0 else 0
            
            # Confidence analysis
            if 'confidence' in trades_df.columns:
                avg_confidence = trades_df['confidence'].mean()
                high_confidence_trades = len(trades_df[trades_df['confidence'] > 0.7])
                high_confidence_pct = high_confidence_trades / total_trades if total_trades > 0 else 0
            else:
                avg_confidence = 0
                high_confidence_trades = 0
                high_confidence_pct = 0
            
            # Trade size analysis
            if 'shares' in trades_df.columns:
                avg_trade_size = trades_df['shares'].mean()
                max_trade_size = trades_df['shares'].max()
                min_trade_size = trades_df['shares'].min()
            else:
                avg_trade_size = 0
                max_trade_size = 0
                min_trade_size = 0
            
            return {
                'total_trades': total_trades,
                'unique_symbols': unique_symbols,
                'trades_per_year': trades_per_year,
                'top_symbol': top_symbol,
                'top_symbol_pct': top_symbol_pct,
                'avg_confidence': avg_confidence,
                'high_confidence_trades': high_confidence_trades,
                'high_confidence_pct': high_confidence_pct,
                'avg_trade_size': avg_trade_size,
                'max_trade_size': max_trade_size,
                'min_trade_size': min_trade_size
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trades: {e}")
            return {}
    
    def _compare_with_benchmark(self, 
                              portfolio_df: pd.DataFrame, 
                              benchmark_df: pd.DataFrame) -> Dict[str, Any]:
        """Compare portfolio performance with benchmark."""
        try:
            if portfolio_df.empty or benchmark_df.empty:
                return {}
            
            # Align data
            common_dates = portfolio_df.index.intersection(benchmark_df.index)
            if len(common_dates) < 2:
                return {}
            
            portfolio_aligned = portfolio_df.loc[common_dates]
            benchmark_aligned = benchmark_df.loc[common_dates]
            
            # Calculate returns
            portfolio_returns = portfolio_aligned['returns']
            benchmark_returns = benchmark_aligned['returns']
            
            # Basic comparison
            portfolio_total_return = (portfolio_aligned['value'].iloc[-1] - portfolio_aligned['value'].iloc[0]) / portfolio_aligned['value'].iloc[0]
            benchmark_total_return = (benchmark_aligned['value'].iloc[-1] - benchmark_aligned['value'].iloc[0]) / benchmark_aligned['value'].iloc[0]
            
            # Alpha and Beta
            beta = self._calculate_beta(portfolio_returns, benchmark_returns)
            alpha = self._calculate_alpha(portfolio_returns, benchmark_returns, beta)
            
            # Correlation
            correlation = portfolio_returns.corr(benchmark_returns)
            
            # Tracking error
            active_returns = portfolio_returns - benchmark_returns
            tracking_error = active_returns.std() * np.sqrt(252)
            
            # Information ratio
            information_ratio = active_returns.mean() / tracking_error * np.sqrt(252) if tracking_error > 0 else 0
            
            # Up and down capture
            up_capture, down_capture = self._calculate_capture_ratios(portfolio_returns, benchmark_returns)
            
            return {
                'portfolio_total_return': portfolio_total_return,
                'benchmark_total_return': benchmark_total_return,
                'excess_return': portfolio_total_return - benchmark_total_return,
                'alpha': alpha,
                'beta': beta,
                'correlation': correlation,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'up_capture': up_capture,
                'down_capture': down_capture
            }
            
        except Exception as e:
            logger.error(f"Error comparing with benchmark: {e}")
            return {}
    
    def _perform_statistical_tests(self, 
                                 portfolio_df: pd.DataFrame, 
                                 benchmark_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        try:
            if portfolio_df.empty or 'returns' not in portfolio_df.columns:
                return {}
            
            portfolio_returns = portfolio_df['returns'].dropna()
            
            tests = {}
            
            # Test for normality
            shapiro_stat, shapiro_pvalue = stats.shapiro(portfolio_returns)
            tests['normality_test'] = {
                'shapiro_statistic': shapiro_stat,
                'shapiro_pvalue': shapiro_pvalue,
                'is_normal': shapiro_pvalue > 0.05
            }
            
            # Test for autocorrelation
            from statsmodels.stats.diagnostic import acorr_ljungbox
            try:
                ljungbox_result = acorr_ljungbox(portfolio_returns, lags=10, return_df=True)
                tests['autocorrelation_test'] = {
                    'ljungbox_statistic': ljungbox_result['lb_stat'].iloc[-1],
                    'ljungbox_pvalue': ljungbox_result['lb_pvalue'].iloc[-1],
                    'has_autocorrelation': ljungbox_result['lb_pvalue'].iloc[-1] < 0.05
                }
            except:
                tests['autocorrelation_test'] = {'error': 'Could not perform autocorrelation test'}
            
            # Test for heteroscedasticity
            try:
                from statsmodels.stats.diagnostic import het_arch
                arch_stat, arch_pvalue, _, _ = het_arch(portfolio_returns)
                tests['heteroscedasticity_test'] = {
                    'arch_statistic': arch_stat,
                    'arch_pvalue': arch_pvalue,
                    'has_heteroscedasticity': arch_pvalue < 0.05
                }
            except:
                tests['heteroscedasticity_test'] = {'error': 'Could not perform heteroscedasticity test'}
            
            # Bootstrap confidence intervals
            bootstrap_results = self._bootstrap_returns(portfolio_returns)
            tests['bootstrap_analysis'] = bootstrap_results
            
            return tests
            
        except Exception as e:
            logger.error(f"Error performing statistical tests: {e}")
            return {}
    
    def _perform_monte_carlo_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform Monte Carlo simulation analysis."""
        try:
            if df.empty or 'returns' not in df.columns:
                return {}
            
            returns = df['returns'].dropna()
            
            # Monte Carlo simulation
            simulations = []
            for _ in range(self.config.monte_carlo_simulations):
                # Bootstrap returns
                simulated_returns = np.random.choice(returns, size=len(returns), replace=True)
                simulated_cumulative = (1 + simulated_returns).cumprod()
                simulations.append(simulated_cumulative.iloc[-1])
            
            simulations = np.array(simulations)
            
            # Calculate statistics
            mean_simulation = np.mean(simulations)
            std_simulation = np.std(simulations)
            var_95_simulation = np.percentile(simulations, 5)
            var_99_simulation = np.percentile(simulations, 1)
            
            # Probability of loss
            prob_loss = np.mean(simulations < 1)
            
            # Probability of beating benchmark
            prob_beat_benchmark = 0
            if 'benchmark_values' in self.results:
                # This would require benchmark data
                pass
            
            return {
                'mean_simulation': mean_simulation,
                'std_simulation': std_simulation,
                'var_95_simulation': var_95_simulation,
                'var_99_simulation': var_99_simulation,
                'prob_loss': prob_loss,
                'prob_beat_benchmark': prob_beat_benchmark,
                'num_simulations': self.config.monte_carlo_simulations
            }
            
        except Exception as e:
            logger.error(f"Error performing Monte Carlo analysis: {e}")
            return {}
    
    def _calculate_beta(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta coefficient."""
        try:
            if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
                return 0
            
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            
            return covariance / benchmark_variance if benchmark_variance > 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 0
    
    def _calculate_alpha(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series, beta: float) -> float:
        """Calculate alpha coefficient."""
        try:
            if len(portfolio_returns) < 2:
                return 0
            
            portfolio_mean = portfolio_returns.mean() * 252
            benchmark_mean = benchmark_returns.mean() * 252
            
            alpha = portfolio_mean - (self.config.risk_free_rate + beta * (benchmark_mean - self.config.risk_free_rate))
            return alpha
            
        except Exception as e:
            logger.error(f"Error calculating alpha: {e}")
            return 0
    
    def _calculate_capture_ratios(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[float, float]:
        """Calculate up and down capture ratios."""
        try:
            # Up periods
            up_periods = benchmark_returns > 0
            if up_periods.sum() > 0:
                portfolio_up = portfolio_returns[up_periods].mean()
                benchmark_up = benchmark_returns[up_periods].mean()
                up_capture = portfolio_up / benchmark_up if benchmark_up != 0 else 0
            else:
                up_capture = 0
            
            # Down periods
            down_periods = benchmark_returns < 0
            if down_periods.sum() > 0:
                portfolio_down = portfolio_returns[down_periods].mean()
                benchmark_down = benchmark_returns[down_periods].mean()
                down_capture = portfolio_down / benchmark_down if benchmark_down != 0 else 0
            else:
                down_capture = 0
            
            return up_capture, down_capture
            
        except Exception as e:
            logger.error(f"Error calculating capture ratios: {e}")
            return 0, 0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            return drawdowns.min()
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0
    
    def _calculate_drawdown_periods(self, drawdowns: pd.Series) -> List[int]:
        """Calculate drawdown period durations."""
        try:
            periods = []
            in_drawdown = False
            start_idx = None
            
            for i, dd in enumerate(drawdowns):
                if dd < -0.001 and not in_drawdown:
                    in_drawdown = True
                    start_idx = i
                elif dd >= -0.001 and in_drawdown:
                    in_drawdown = False
                    if start_idx is not None:
                        periods.append(i - start_idx)
            
            return periods
            
        except Exception as e:
            logger.error(f"Error calculating drawdown periods: {e}")
            return []
    
    def _bootstrap_returns(self, returns: pd.Series) -> Dict[str, Any]:
        """Perform bootstrap analysis on returns."""
        try:
            bootstrap_returns = []
            
            for _ in range(self.config.bootstrap_samples):
                bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
                bootstrap_returns.append(bootstrap_sample.mean() * 252)
            
            bootstrap_returns = np.array(bootstrap_returns)
            
            return {
                'mean_return': np.mean(bootstrap_returns),
                'std_return': np.std(bootstrap_returns),
                'ci_95_lower': np.percentile(bootstrap_returns, 2.5),
                'ci_95_upper': np.percentile(bootstrap_returns, 97.5),
                'ci_99_lower': np.percentile(bootstrap_returns, 0.5),
                'ci_99_upper': np.percentile(bootstrap_returns, 99.5)
            }
            
        except Exception as e:
            logger.error(f"Error in bootstrap analysis: {e}")
            return {}
    
    def get_results(self) -> Dict[str, Any]:
        """Get performance analysis results."""
        return self.results
    
    def generate_report(self) -> str:
        """Generate a comprehensive performance report."""
        try:
            if not self.results:
                return "No analysis results available"
            
            report = []
            report.append("=" * 60)
            report.append("PERFORMANCE ANALYSIS REPORT")
            report.append("=" * 60)
            report.append(f"Analysis Date: {self.results.get('analysis_timestamp', 'N/A')}")
            report.append("")
            
            # Basic metrics
            basic_metrics = self.results.get('basic_metrics', {})
            if basic_metrics:
                report.append("BASIC PERFORMANCE METRICS")
                report.append("-" * 30)
                report.append(f"Total Return: {basic_metrics.get('total_return', 0):.2%}")
                report.append(f"Annualized Return: {basic_metrics.get('annualized_return', 0):.2%}")
                report.append(f"Volatility: {basic_metrics.get('volatility', 0):.2%}")
                report.append(f"Win Rate: {basic_metrics.get('win_rate', 0):.2%}")
                report.append("")
            
            # Risk metrics
            risk_metrics = self.results.get('risk_metrics', {})
            if risk_metrics:
                report.append("RISK METRICS")
                report.append("-" * 30)
                report.append(f"Max Drawdown: {risk_metrics.get('max_drawdown', 0):.2%}")
                report.append(f"VaR (95%): {risk_metrics.get('var_95', 0):.2%}")
                report.append(f"CVaR (95%): {risk_metrics.get('cvar_95', 0):.2%}")
                report.append("")
            
            # Risk-adjusted metrics
            risk_adjusted = self.results.get('risk_adjusted_metrics', {})
            if risk_adjusted:
                report.append("RISK-ADJUSTED METRICS")
                report.append("-" * 30)
                report.append(f"Sharpe Ratio: {risk_adjusted.get('sharpe_ratio', 0):.2f}")
                report.append(f"Sortino Ratio: {risk_adjusted.get('sortino_ratio', 0):.2f}")
                report.append(f"Calmar Ratio: {risk_adjusted.get('calmar_ratio', 0):.2f}")
                report.append("")
            
            # Benchmark comparison
            benchmark_comparison = self.results.get('benchmark_comparison', {})
            if benchmark_comparison:
                report.append("BENCHMARK COMPARISON")
                report.append("-" * 30)
                report.append(f"Alpha: {benchmark_comparison.get('alpha', 0):.2%}")
                report.append(f"Beta: {benchmark_comparison.get('beta', 0):.2f}")
                report.append(f"Information Ratio: {benchmark_comparison.get('information_ratio', 0):.2f}")
                report.append("")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {e}"


# Utility functions
def analyze_performance(portfolio_values: List[Dict[str, Any]], 
                       benchmark_values: Optional[List[Dict[str, Any]]] = None,
                       trades: Optional[List[Dict[str, Any]]] = None,
                       config: Optional[PerformanceConfig] = None) -> Dict[str, Any]:
    """Quick function to analyze performance."""
    if config is None:
        config = PerformanceConfig()
    
    analyzer = PerformanceAnalyzer(config)
    return analyzer.analyze_performance(portfolio_values, benchmark_values, trades)


def create_performance_config(risk_free_rate: float = 0.02) -> PerformanceConfig:
    """Create a performance analysis configuration."""
    return PerformanceConfig(risk_free_rate=risk_free_rate)


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="1y")
    
    # Create mock portfolio values
    portfolio_values = []
    initial_value = 100000
    current_value = initial_value
    
    for i, (date, row) in enumerate(df.iterrows()):
        # Simulate some returns
        daily_return = np.random.normal(0.0005, 0.02)
        current_value *= (1 + daily_return)
        
        portfolio_values.append({
            'date': date,
            'value': current_value,
            'cash': current_value * 0.1,
            'positions_value': current_value * 0.9,
            'unrealized_pnl': current_value * 0.01
        })
    
    # Create mock benchmark values
    benchmark_values = []
    benchmark_value = 100
    
    for i, (date, row) in enumerate(df.iterrows()):
        benchmark_return = np.random.normal(0.0003, 0.015)
        benchmark_value *= (1 + benchmark_return)
        
        benchmark_values.append({
            'date': date,
            'value': benchmark_value
        })
    
    # Analyze performance
    config = PerformanceConfig()
    analyzer = PerformanceAnalyzer(config)
    results = analyzer.analyze_performance(portfolio_values, benchmark_values)
    
    print("Performance analysis completed!")
    print(analyzer.generate_report())
    
    # Get specific metrics
    print(f"\nSharpe Ratio: {results['risk_adjusted_metrics']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['risk_metrics']['max_drawdown']:.2%}")
    print(f"Alpha: {results['benchmark_comparison']['alpha']:.2%}")
