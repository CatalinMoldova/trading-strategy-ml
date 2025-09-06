"""
Report generator for creating comprehensive trading strategy reports.
Generates detailed performance reports, visualizations, and analysis summaries.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Enumeration of report formats."""
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"


class ChartType(Enum):
    """Enumeration of chart types."""
    EQUITY_CURVE = "equity_curve"
    DRAWDOWN = "drawdown"
    RETURNS_DISTRIBUTION = "returns_distribution"
    ROLLING_METRICS = "rolling_metrics"
    TRADE_ANALYSIS = "trade_analysis"
    RISK_METRICS = "risk_metrics"
    BENCHMARK_COMPARISON = "benchmark_comparison"


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    # Report settings
    title: str = "Trading Strategy Performance Report"
    author: str = "Trading Strategy ML"
    date: Optional[datetime] = None
    
    # Chart settings
    chart_style: str = "plotly"  # "matplotlib" or "plotly"
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    
    # Report sections
    include_executive_summary: bool = True
    include_performance_metrics: bool = True
    include_risk_analysis: bool = True
    include_trade_analysis: bool = True
    include_statistical_tests: bool = True
    include_charts: bool = True
    include_recommendations: bool = True
    
    # Output settings
    save_charts: bool = True
    charts_directory: str = "charts"
    report_directory: str = "reports"


class ReportGenerator:
    """Comprehensive report generation system."""
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.report_data = {}
        self.charts = {}
        
    def generate_report(self, 
                       backtest_results: Dict[str, Any],
                       performance_analysis: Optional[Dict[str, Any]] = None,
                       statistical_tests: Optional[Dict[str, Any]] = None,
                       walk_forward_results: Optional[Dict[str, Any]] = None) -> str:
        """Generate comprehensive trading strategy report."""
        try:
            logger.info("Starting report generation")
            
            # Store report data
            self.report_data = {
                'backtest_results': backtest_results,
                'performance_analysis': performance_analysis or {},
                'statistical_tests': statistical_tests or {},
                'walk_forward_results': walk_forward_results or {},
                'generation_timestamp': datetime.now()
            }
            
            # Generate charts
            if self.config.include_charts:
                self._generate_charts()
            
            # Generate report sections
            report_sections = []
            
            if self.config.include_executive_summary:
                report_sections.append(self._generate_executive_summary())
            
            if self.config.include_performance_metrics:
                report_sections.append(self._generate_performance_metrics_section())
            
            if self.config.include_risk_analysis:
                report_sections.append(self._generate_risk_analysis_section())
            
            if self.config.include_trade_analysis:
                report_sections.append(self._generate_trade_analysis_section())
            
            if self.config.include_statistical_tests:
                report_sections.append(self._generate_statistical_tests_section())
            
            if self.config.include_charts:
                report_sections.append(self._generate_charts_section())
            
            if self.config.include_recommendations:
                report_sections.append(self._generate_recommendations_section())
            
            # Compile final report
            report = self._compile_report(report_sections)
            
            # Save report
            if self.config.save_charts:
                self._save_report(report)
            
            logger.info("Report generation completed")
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating report: {e}"
    
    def _generate_charts(self):
        """Generate all charts for the report."""
        try:
            backtest_results = self.report_data['backtest_results']
            performance_analysis = self.report_data['performance_analysis']
            
            # Equity curve chart
            if 'portfolio_values' in backtest_results:
                self.charts['equity_curve'] = self._create_equity_curve_chart(backtest_results)
            
            # Drawdown chart
            if 'portfolio_values' in backtest_results:
                self.charts['drawdown'] = self._create_drawdown_chart(backtest_results)
            
            # Returns distribution chart
            if 'portfolio_values' in backtest_results:
                self.charts['returns_distribution'] = self._create_returns_distribution_chart(backtest_results)
            
            # Rolling metrics chart
            if 'portfolio_values' in backtest_results:
                self.charts['rolling_metrics'] = self._create_rolling_metrics_chart(backtest_results)
            
            # Trade analysis chart
            if 'trades' in backtest_results:
                self.charts['trade_analysis'] = self._create_trade_analysis_chart(backtest_results)
            
            # Risk metrics chart
            if performance_analysis:
                self.charts['risk_metrics'] = self._create_risk_metrics_chart(performance_analysis)
            
            # Benchmark comparison chart
            if 'benchmark_values' in backtest_results:
                self.charts['benchmark_comparison'] = self._create_benchmark_comparison_chart(backtest_results)
            
            logger.info(f"Generated {len(self.charts)} charts")
            
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
    
    def _create_equity_curve_chart(self, backtest_results: Dict[str, Any]) -> Any:
        """Create equity curve chart."""
        try:
            portfolio_values = backtest_results['portfolio_values']
            benchmark_values = backtest_results.get('benchmark_values', [])
            
            # Convert to DataFrame
            portfolio_df = pd.DataFrame(portfolio_values)
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            
            if self.config.chart_style == 'plotly':
                fig = go.Figure()
                
                # Portfolio equity curve
                fig.add_trace(go.Scatter(
                    x=portfolio_df['date'],
                    y=portfolio_df['value'],
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='blue', width=2)
                ))
                
                # Benchmark equity curve
                if benchmark_values:
                    benchmark_df = pd.DataFrame(benchmark_values)
                    benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
                    fig.add_trace(go.Scatter(
                        x=benchmark_df['date'],
                        y=benchmark_df['value'],
                        mode='lines',
                        name='Benchmark',
                        line=dict(color='red', width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    title='Portfolio Equity Curve',
                    xaxis_title='Date',
                    yaxis_title='Portfolio Value ($)',
                    hovermode='x unified'
                )
                
                return fig
            
            else:  # matplotlib
                fig, ax = plt.subplots(figsize=self.config.figure_size)
                
                ax.plot(portfolio_df['date'], portfolio_df['value'], label='Portfolio', linewidth=2)
                
                if benchmark_values:
                    benchmark_df = pd.DataFrame(benchmark_values)
                    benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
                    ax.plot(benchmark_df['date'], benchmark_df['value'], label='Benchmark', linewidth=2, linestyle='--')
                
                ax.set_title('Portfolio Equity Curve')
                ax.set_xlabel('Date')
                ax.set_ylabel('Portfolio Value ($)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                return fig
                
        except Exception as e:
            logger.error(f"Error creating equity curve chart: {e}")
            return None
    
    def _create_drawdown_chart(self, backtest_results: Dict[str, Any]) -> Any:
        """Create drawdown chart."""
        try:
            portfolio_values = backtest_results['portfolio_values']
            portfolio_df = pd.DataFrame(portfolio_values)
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            
            # Calculate drawdown
            portfolio_df['cumulative'] = portfolio_df['value'] / portfolio_df['value'].iloc[0]
            portfolio_df['running_max'] = portfolio_df['cumulative'].expanding().max()
            portfolio_df['drawdown'] = (portfolio_df['cumulative'] - portfolio_df['running_max']) / portfolio_df['running_max']
            
            if self.config.chart_style == 'plotly':
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=portfolio_df['date'],
                    y=portfolio_df['drawdown'] * 100,
                    mode='lines',
                    name='Drawdown',
                    fill='tonexty',
                    line=dict(color='red', width=1)
                ))
                
                fig.update_layout(
                    title='Portfolio Drawdown',
                    xaxis_title='Date',
                    yaxis_title='Drawdown (%)',
                    hovermode='x unified'
                )
                
                return fig
            
            else:  # matplotlib
                fig, ax = plt.subplots(figsize=self.config.figure_size)
                
                ax.fill_between(portfolio_df['date'], portfolio_df['drawdown'] * 100, 0, 
                               color='red', alpha=0.3, label='Drawdown')
                ax.plot(portfolio_df['date'], portfolio_df['drawdown'] * 100, color='red', linewidth=1)
                
                ax.set_title('Portfolio Drawdown')
                ax.set_xlabel('Date')
                ax.set_ylabel('Drawdown (%)')
                ax.grid(True, alpha=0.3)
                
                return fig
                
        except Exception as e:
            logger.error(f"Error creating drawdown chart: {e}")
            return None
    
    def _create_returns_distribution_chart(self, backtest_results: Dict[str, Any]) -> Any:
        """Create returns distribution chart."""
        try:
            portfolio_values = backtest_results['portfolio_values']
            portfolio_df = pd.DataFrame(portfolio_values)
            
            # Calculate returns
            returns = portfolio_df['value'].pct_change().dropna()
            
            if self.config.chart_style == 'plotly':
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=returns * 100,
                    nbinsx=50,
                    name='Returns Distribution',
                    opacity=0.7
                ))
                
                # Add normal distribution overlay
                x_range = np.linspace(returns.min() * 100, returns.max() * 100, 100)
                normal_dist = stats.norm.pdf(x_range, returns.mean() * 100, returns.std() * 100)
                normal_dist = normal_dist * len(returns) * (returns.max() - returns.min()) / 50 * 100
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=normal_dist,
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title='Returns Distribution',
                    xaxis_title='Daily Returns (%)',
                    yaxis_title='Frequency',
                    hovermode='x unified'
                )
                
                return fig
            
            else:  # matplotlib
                fig, ax = plt.subplots(figsize=self.config.figure_size)
                
                ax.hist(returns * 100, bins=50, alpha=0.7, label='Returns Distribution')
                
                # Add normal distribution overlay
                x_range = np.linspace(returns.min() * 100, returns.max() * 100, 100)
                normal_dist = stats.norm.pdf(x_range, returns.mean() * 100, returns.std() * 100)
                ax.plot(x_range, normal_dist * len(returns) * (returns.max() - returns.min()) / 50 * 100, 
                       'r--', label='Normal Distribution')
                
                ax.set_title('Returns Distribution')
                ax.set_xlabel('Daily Returns (%)')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                return fig
                
        except Exception as e:
            logger.error(f"Error creating returns distribution chart: {e}")
            return None
    
    def _create_rolling_metrics_chart(self, backtest_results: Dict[str, Any]) -> Any:
        """Create rolling metrics chart."""
        try:
            portfolio_values = backtest_results['portfolio_values']
            portfolio_df = pd.DataFrame(portfolio_values)
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            
            # Calculate rolling metrics
            portfolio_df['returns'] = portfolio_df['value'].pct_change()
            portfolio_df['rolling_sharpe'] = portfolio_df['returns'].rolling(252).mean() / portfolio_df['returns'].rolling(252).std() * np.sqrt(252)
            portfolio_df['rolling_volatility'] = portfolio_df['returns'].rolling(252).std() * np.sqrt(252)
            
            if self.config.chart_style == 'plotly':
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Rolling Sharpe Ratio', 'Rolling Volatility'),
                    vertical_spacing=0.1
                )
                
                fig.add_trace(go.Scatter(
                    x=portfolio_df['date'],
                    y=portfolio_df['rolling_sharpe'],
                    mode='lines',
                    name='Rolling Sharpe Ratio',
                    line=dict(color='blue')
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=portfolio_df['date'],
                    y=portfolio_df['rolling_volatility'],
                    mode='lines',
                    name='Rolling Volatility',
                    line=dict(color='green')
                ), row=2, col=1)
                
                fig.update_layout(
                    title='Rolling Performance Metrics',
                    height=600,
                    showlegend=False
                )
                
                return fig
            
            else:  # matplotlib
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figure_size)
                
                ax1.plot(portfolio_df['date'], portfolio_df['rolling_sharpe'], color='blue')
                ax1.set_title('Rolling Sharpe Ratio')
                ax1.set_ylabel('Sharpe Ratio')
                ax1.grid(True, alpha=0.3)
                
                ax2.plot(portfolio_df['date'], portfolio_df['rolling_volatility'], color='green')
                ax2.set_title('Rolling Volatility')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Volatility')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                return fig
                
        except Exception as e:
            logger.error(f"Error creating rolling metrics chart: {e}")
            return None
    
    def _create_trade_analysis_chart(self, backtest_results: Dict[str, Any]) -> Any:
        """Create trade analysis chart."""
        try:
            trades = backtest_results.get('trades', [])
            if not trades:
                return None
            
            trades_df = pd.DataFrame(trades)
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            
            # Count trades by symbol
            symbol_counts = trades_df['symbol'].value_counts()
            
            if self.config.chart_style == 'plotly':
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=symbol_counts.index,
                    y=symbol_counts.values,
                    name='Trades by Symbol'
                ))
                
                fig.update_layout(
                    title='Trades by Symbol',
                    xaxis_title='Symbol',
                    yaxis_title='Number of Trades'
                )
                
                return fig
            
            else:  # matplotlib
                fig, ax = plt.subplots(figsize=self.config.figure_size)
                
                ax.bar(symbol_counts.index, symbol_counts.values)
                ax.set_title('Trades by Symbol')
                ax.set_xlabel('Symbol')
                ax.set_ylabel('Number of Trades')
                ax.tick_params(axis='x', rotation=45)
                
                return fig
                
        except Exception as e:
            logger.error(f"Error creating trade analysis chart: {e}")
            return None
    
    def _create_risk_metrics_chart(self, performance_analysis: Dict[str, Any]) -> Any:
        """Create risk metrics chart."""
        try:
            risk_metrics = performance_analysis.get('risk_metrics', {})
            if not risk_metrics:
                return None
            
            # Extract key risk metrics
            metrics = ['var_95', 'cvar_95', 'max_drawdown', 'downside_deviation']
            values = [abs(risk_metrics.get(metric, 0)) for metric in metrics]
            labels = ['VaR (95%)', 'CVaR (95%)', 'Max Drawdown', 'Downside Dev']
            
            if self.config.chart_style == 'plotly':
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=labels,
                    y=values,
                    name='Risk Metrics'
                ))
                
                fig.update_layout(
                    title='Risk Metrics',
                    xaxis_title='Metric',
                    yaxis_title='Value'
                )
                
                return fig
            
            else:  # matplotlib
                fig, ax = plt.subplots(figsize=self.config.figure_size)
                
                ax.bar(labels, values)
                ax.set_title('Risk Metrics')
                ax.set_ylabel('Value')
                ax.tick_params(axis='x', rotation=45)
                
                return fig
                
        except Exception as e:
            logger.error(f"Error creating risk metrics chart: {e}")
            return None
    
    def _create_benchmark_comparison_chart(self, backtest_results: Dict[str, Any]) -> Any:
        """Create benchmark comparison chart."""
        try:
            portfolio_values = backtest_results['portfolio_values']
            benchmark_values = backtest_results.get('benchmark_values', [])
            
            if not benchmark_values:
                return None
            
            portfolio_df = pd.DataFrame(portfolio_values)
            benchmark_df = pd.DataFrame(benchmark_values)
            
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            benchmark_df['date'] = pd.to_datetime(benchmark_df['date'])
            
            # Calculate cumulative returns
            portfolio_df['cumulative'] = portfolio_df['value'] / portfolio_df['value'].iloc[0]
            benchmark_df['cumulative'] = benchmark_df['value'] / benchmark_df['value'].iloc[0]
            
            if self.config.chart_style == 'plotly':
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=portfolio_df['date'],
                    y=portfolio_df['cumulative'],
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=benchmark_df['date'],
                    y=benchmark_df['cumulative'],
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title='Portfolio vs Benchmark',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Returns',
                    hovermode='x unified'
                )
                
                return fig
            
            else:  # matplotlib
                fig, ax = plt.subplots(figsize=self.config.figure_size)
                
                ax.plot(portfolio_df['date'], portfolio_df['cumulative'], label='Portfolio', linewidth=2)
                ax.plot(benchmark_df['date'], benchmark_df['cumulative'], label='Benchmark', linewidth=2, linestyle='--')
                
                ax.set_title('Portfolio vs Benchmark')
                ax.set_xlabel('Date')
                ax.set_ylabel('Cumulative Returns')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                return fig
                
        except Exception as e:
            logger.error(f"Error creating benchmark comparison chart: {e}")
            return None
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary section."""
        try:
            backtest_results = self.report_data['backtest_results']
            performance_analysis = self.report_data['performance_analysis']
            
            # Extract key metrics
            performance_metrics = backtest_results.get('performance_metrics', {})
            
            total_return = performance_metrics.get('total_return', 0)
            sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
            max_drawdown = performance_metrics.get('max_drawdown', 0)
            
            # Generate summary
            summary = f"""
## Executive Summary

The trading strategy has been evaluated using comprehensive backtesting and statistical analysis. 

**Key Performance Metrics:**
- **Total Return**: {total_return:.2%}
- **Sharpe Ratio**: {sharpe_ratio:.2f}
- **Maximum Drawdown**: {max_drawdown:.2%}

**Overall Assessment:**
"""
            
            if sharpe_ratio > 1.5:
                summary += "The strategy demonstrates excellent risk-adjusted returns with a Sharpe ratio above 1.5."
            elif sharpe_ratio > 1.0:
                summary += "The strategy shows good risk-adjusted returns with a Sharpe ratio above 1.0."
            elif sharpe_ratio > 0.5:
                summary += "The strategy shows moderate risk-adjusted returns."
            else:
                summary += "The strategy shows poor risk-adjusted returns and requires improvement."
            
            if abs(max_drawdown) < 0.10:
                summary += " Risk management is effective with maximum drawdown below 10%."
            elif abs(max_drawdown) < 0.20:
                summary += " Risk management is acceptable with maximum drawdown below 20%."
            else:
                summary += " Risk management needs improvement with maximum drawdown above 20%."
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return "## Executive Summary\n\nError generating executive summary."
    
    def _generate_performance_metrics_section(self) -> str:
        """Generate performance metrics section."""
        try:
            backtest_results = self.report_data['backtest_results']
            performance_metrics = backtest_results.get('performance_metrics', {})
            
            section = """
## Performance Metrics

### Basic Performance
"""
            
            # Basic metrics
            basic_metrics = [
                ('Total Return', performance_metrics.get('total_return', 0), 'percentage'),
                ('Annualized Return', performance_metrics.get('annualized_return', 0), 'percentage'),
                ('Volatility', performance_metrics.get('volatility', 0), 'percentage'),
                ('Sharpe Ratio', performance_metrics.get('sharpe_ratio', 0), 'decimal'),
                ('Sortino Ratio', performance_metrics.get('sortino_ratio', 0), 'decimal'),
                ('Calmar Ratio', performance_metrics.get('calmar_ratio', 0), 'decimal')
            ]
            
            for metric_name, value, format_type in basic_metrics:
                if format_type == 'percentage':
                    section += f"- **{metric_name}**: {value:.2%}\n"
                else:
                    section += f"- **{metric_name}**: {value:.2f}\n"
            
            # Risk metrics
            section += "\n### Risk Metrics\n"
            risk_metrics = [
                ('Maximum Drawdown', performance_metrics.get('max_drawdown', 0), 'percentage'),
                ('VaR (95%)', performance_metrics.get('var_95', 0), 'percentage'),
                ('CVaR (95%)', performance_metrics.get('cvar_95', 0), 'percentage')
            ]
            
            for metric_name, value, format_type in risk_metrics:
                if format_type == 'percentage':
                    section += f"- **{metric_name}**: {value:.2%}\n"
                else:
                    section += f"- **{metric_name}**: {value:.2f}\n"
            
            return section
            
        except Exception as e:
            logger.error(f"Error generating performance metrics section: {e}")
            return "## Performance Metrics\n\nError generating performance metrics."
    
    def _generate_risk_analysis_section(self) -> str:
        """Generate risk analysis section."""
        try:
            performance_analysis = self.report_data['performance_analysis']
            risk_metrics = performance_analysis.get('risk_metrics', {})
            
            section = """
## Risk Analysis

### Risk Metrics
"""
            
            # Risk metrics
            risk_metrics_list = [
                ('Maximum Drawdown', risk_metrics.get('max_drawdown', 0)),
                ('VaR (95%)', risk_metrics.get('var_95', 0)),
                ('CVaR (95%)', risk_metrics.get('cvar_95', 0)),
                ('Downside Deviation', risk_metrics.get('downside_deviation', 0))
            ]
            
            for metric_name, value in risk_metrics_list:
                section += f"- **{metric_name}**: {value:.2%}\n"
            
            # Risk assessment
            section += "\n### Risk Assessment\n"
            
            max_drawdown = abs(risk_metrics.get('max_drawdown', 0))
            if max_drawdown < 0.05:
                section += "**Low Risk**: Maximum drawdown is below 5%, indicating excellent risk control.\n"
            elif max_drawdown < 0.10:
                section += "**Moderate Risk**: Maximum drawdown is below 10%, indicating good risk control.\n"
            elif max_drawdown < 0.20:
                section += "**High Risk**: Maximum drawdown is below 20%, indicating acceptable risk control.\n"
            else:
                section += "**Very High Risk**: Maximum drawdown exceeds 20%, indicating poor risk control.\n"
            
            return section
            
        except Exception as e:
            logger.error(f"Error generating risk analysis section: {e}")
            return "## Risk Analysis\n\nError generating risk analysis."
    
    def _generate_trade_analysis_section(self) -> str:
        """Generate trade analysis section."""
        try:
            backtest_results = self.report_data['backtest_results']
            trades = backtest_results.get('trades', [])
            
            section = """
## Trade Analysis

### Trading Activity
"""
            
            if trades:
                trades_df = pd.DataFrame(trades)
                
                total_trades = len(trades_df)
                unique_symbols = trades_df['symbol'].nunique()
                avg_confidence = trades_df['confidence'].mean() if 'confidence' in trades_df.columns else 0
                
                section += f"- **Total Trades**: {total_trades}\n"
                section += f"- **Unique Symbols**: {unique_symbols}\n"
                section += f"- **Average Confidence**: {avg_confidence:.2%}\n"
                
                # Top symbols
                symbol_counts = trades_df['symbol'].value_counts()
                section += f"\n### Top Trading Symbols\n"
                for symbol, count in symbol_counts.head(5).items():
                    section += f"- **{symbol}**: {count} trades\n"
                
            else:
                section += "No trades recorded during the backtest period.\n"
            
            return section
            
        except Exception as e:
            logger.error(f"Error generating trade analysis section: {e}")
            return "## Trade Analysis\n\nError generating trade analysis."
    
    def _generate_statistical_tests_section(self) -> str:
        """Generate statistical tests section."""
        try:
            statistical_tests = self.report_data['statistical_tests']
            
            section = """
## Statistical Tests

### Significance Tests
"""
            
            if statistical_tests:
                # Performance significance
                perf_tests = statistical_tests.get('performance_significance', {})
                if perf_tests:
                    section += "**Performance Significance:**\n"
                    
                    sharpe_test = perf_tests.get('sharpe_ratio_test', {})
                    if sharpe_test:
                        is_significant = sharpe_test.get('is_significant', False)
                        section += f"- Sharpe Ratio Test: {'Significant' if is_significant else 'Not Significant'}\n"
                    
                    mean_test = perf_tests.get('mean_return_test', {})
                    if mean_test:
                        is_significant = mean_test.get('is_significant', False)
                        section += f"- Mean Return Test: {'Significant' if is_significant else 'Not Significant'}\n"
                
                # Overall assessment
                overall = statistical_tests.get('overall_assessment', {})
                if overall:
                    assessment = overall.get('assessment', 'Unknown')
                    section += f"\n**Overall Assessment**: {assessment}\n"
                
            else:
                section += "No statistical tests performed.\n"
            
            return section
            
        except Exception as e:
            logger.error(f"Error generating statistical tests section: {e}")
            return "## Statistical Tests\n\nError generating statistical tests."
    
    def _generate_charts_section(self) -> str:
        """Generate charts section."""
        try:
            section = """
## Charts and Visualizations

The following charts provide visual analysis of the strategy performance:

"""
            
            chart_descriptions = {
                'equity_curve': 'Portfolio equity curve showing the growth of the portfolio value over time',
                'drawdown': 'Drawdown chart showing the maximum peak-to-trough decline',
                'returns_distribution': 'Distribution of daily returns compared to normal distribution',
                'rolling_metrics': 'Rolling Sharpe ratio and volatility over time',
                'trade_analysis': 'Analysis of trading activity by symbol',
                'risk_metrics': 'Key risk metrics visualization',
                'benchmark_comparison': 'Comparison of portfolio performance against benchmark'
            }
            
            for chart_name, chart in self.charts.items():
                if chart is not None:
                    description = chart_descriptions.get(chart_name, f'{chart_name} chart')
                    section += f"- **{chart_name.replace('_', ' ').title()}**: {description}\n"
            
            return section
            
        except Exception as e:
            logger.error(f"Error generating charts section: {e}")
            return "## Charts\n\nError generating charts section."
    
    def _generate_recommendations_section(self) -> str:
        """Generate recommendations section."""
        try:
            backtest_results = self.report_data['backtest_results']
            performance_metrics = backtest_results.get('performance_metrics', {})
            statistical_tests = self.report_data['statistical_tests']
            
            section = """
## Recommendations

Based on the comprehensive analysis, the following recommendations are made:

"""
            
            # Performance-based recommendations
            sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
            max_drawdown = performance_metrics.get('max_drawdown', 0)
            
            if sharpe_ratio < 0.5:
                section += "- **Improve Strategy**: The Sharpe ratio is below 0.5, indicating poor risk-adjusted returns. Consider revising the strategy logic.\n"
            elif sharpe_ratio < 1.0:
                section += "- **Enhance Strategy**: The Sharpe ratio is below 1.0. Consider optimizing parameters or improving signal quality.\n"
            else:
                section += "- **Strategy Performance**: The Sharpe ratio is above 1.0, indicating good risk-adjusted returns.\n"
            
            if abs(max_drawdown) > 0.20:
                section += "- **Risk Management**: Maximum drawdown exceeds 20%. Implement stricter risk controls and position sizing.\n"
            elif abs(max_drawdown) > 0.15:
                section += "- **Risk Management**: Maximum drawdown exceeds 15%. Consider tightening risk controls.\n"
            else:
                section += "- **Risk Management**: Drawdown levels are acceptable.\n"
            
            # Statistical significance recommendations
            overall_assessment = statistical_tests.get('overall_assessment', {})
            if overall_assessment.get('is_overall_significant', False):
                section += "- **Statistical Significance**: Strategy performance is statistically significant.\n"
            else:
                section += "- **Statistical Significance**: Strategy performance is not statistically significant. Consider increasing sample size or improving strategy.\n"
            
            # General recommendations
            section += """
### General Recommendations

1. **Continue Monitoring**: Regularly monitor strategy performance and market conditions.
2. **Risk Management**: Maintain strict risk controls and position sizing rules.
3. **Diversification**: Consider diversifying across different asset classes or strategies.
4. **Backtesting**: Continue to validate strategy performance using walk-forward analysis.
5. **Documentation**: Maintain detailed records of all trades and performance metrics.
"""
            
            return section
            
        except Exception as e:
            logger.error(f"Error generating recommendations section: {e}")
            return "## Recommendations\n\nError generating recommendations."
    
    def _compile_report(self, sections: List[str]) -> str:
        """Compile the final report."""
        try:
            # Header
            report_date = self.config.date or datetime.now()
            header = f"""
# {self.config.title}

**Author**: {self.config.author}  
**Date**: {report_date.strftime('%Y-%m-%d %H:%M:%S')}  
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

"""
            
            # Compile sections
            report = header + "\n".join(sections)
            
            return report
            
        except Exception as e:
            logger.error(f"Error compiling report: {e}")
            return f"Error compiling report: {e}"
    
    def _save_report(self, report: str):
        """Save the report to file."""
        try:
            import os
            
            # Create directories
            os.makedirs(self.config.report_directory, exist_ok=True)
            os.makedirs(self.config.charts_directory, exist_ok=True)
            
            # Save report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_filename = f"trading_strategy_report_{timestamp}.md"
            report_path = os.path.join(self.config.report_directory, report_filename)
            
            with open(report_path, 'w') as f:
                f.write(report)
            
            # Save charts
            for chart_name, chart in self.charts.items():
                if chart is not None:
                    chart_filename = f"{chart_name}_{timestamp}.html"
                    chart_path = os.path.join(self.config.charts_directory, chart_filename)
                    
                    if self.config.chart_style == 'plotly':
                        chart.write_html(chart_path)
                    else:  # matplotlib
                        chart.savefig(chart_path.replace('.html', '.png'), dpi=self.config.dpi, bbox_inches='tight')
            
            logger.info(f"Report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")


# Utility functions
def generate_trading_report(backtest_results: Dict[str, Any],
                          performance_analysis: Optional[Dict[str, Any]] = None,
                          statistical_tests: Optional[Dict[str, Any]] = None,
                          config: Optional[ReportConfig] = None) -> str:
    """Quick function to generate a trading strategy report."""
    if config is None:
        config = ReportConfig()
    
    generator = ReportGenerator(config)
    return generator.generate_report(backtest_results, performance_analysis, statistical_tests)


def create_report_config(title: str = "Trading Strategy Performance Report") -> ReportConfig:
    """Create a report configuration."""
    return ReportConfig(title=title)


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="1y")
    
    # Create mock backtest results
    portfolio_values = []
    initial_value = 100000
    current_value = initial_value
    
    for i, (date, row) in enumerate(df.iterrows()):
        daily_return = np.random.normal(0.0005, 0.02)
        current_value *= (1 + daily_return)
        
        portfolio_values.append({
            'date': date,
            'value': current_value,
            'cash': current_value * 0.1,
            'positions_value': current_value * 0.9,
            'unrealized_pnl': current_value * 0.01
        })
    
    # Mock performance metrics
    performance_metrics = {
        'total_return': 0.15,
        'annualized_return': 0.12,
        'volatility': 0.18,
        'sharpe_ratio': 0.67,
        'max_drawdown': -0.08,
        'var_95': -0.03,
        'cvar_95': -0.05
    }
    
    backtest_results = {
        'performance_metrics': performance_metrics,
        'portfolio_values': portfolio_values,
        'trades': []
    }
    
    # Generate report
    config = ReportConfig(title="Sample Trading Strategy Report")
    generator = ReportGenerator(config)
    report = generator.generate_report(backtest_results)
    
    print("Report generated successfully!")
    print(report[:500] + "...")  # Show first 500 characters
