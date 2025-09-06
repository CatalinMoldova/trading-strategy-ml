"""
Statistical tests module for validating trading strategy performance.
Implements bootstrap tests, significance testing, and robustness checks.
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


class TestType(Enum):
    """Enumeration of statistical test types."""
    BOOTSTRAP = "bootstrap"
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    JARQUE_BERA = "jarque_bera"
    LJUNG_BOX = "ljung_box"
    ARCH = "arch"
    DICKEY_FULLER = "dickey_fuller"


@dataclass
class StatisticalTestConfig:
    """Configuration for statistical tests."""
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    block_size: int = 5
    random_seed: int = 42
    alternative: str = "two-sided"  # "two-sided", "greater", "less"


class StatisticalTester:
    """Comprehensive statistical testing system for trading strategies."""
    
    def __init__(self, config: StatisticalTestConfig):
        self.config = config
        self.test_results = {}
        np.random.seed(config.random_seed)
        
    def run_comprehensive_tests(self, 
                              strategy_returns: pd.Series,
                              benchmark_returns: Optional[pd.Series] = None,
                              risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """Run comprehensive statistical tests on strategy returns."""
        try:
            logger.info("Starting comprehensive statistical tests")
            
            # Clean data
            strategy_returns = strategy_returns.dropna()
            if benchmark_returns is not None:
                benchmark_returns = benchmark_returns.dropna()
            
            # Align data if benchmark provided
            if benchmark_returns is not None:
                common_dates = strategy_returns.index.intersection(benchmark_returns.index)
                strategy_returns = strategy_returns.loc[common_dates]
                benchmark_returns = benchmark_returns.loc[common_dates]
            
            # Run individual tests
            tests = {}
            
            # Basic statistical tests
            tests['normality'] = self._test_normality(strategy_returns)
            tests['autocorrelation'] = self._test_autocorrelation(strategy_returns)
            tests['heteroscedasticity'] = self._test_heteroscedasticity(strategy_returns)
            tests['stationarity'] = self._test_stationarity(strategy_returns)
            
            # Performance tests
            tests['performance_significance'] = self._test_performance_significance(strategy_returns, risk_free_rate)
            
            # Bootstrap tests
            tests['bootstrap_sharpe'] = self._bootstrap_sharpe_ratio(strategy_returns, risk_free_rate)
            tests['bootstrap_max_drawdown'] = self._bootstrap_max_drawdown(strategy_returns)
            tests['bootstrap_returns'] = self._bootstrap_returns(strategy_returns)
            
            # Benchmark comparison tests
            if benchmark_returns is not None:
                tests['benchmark_comparison'] = self._test_benchmark_comparison(strategy_returns, benchmark_returns)
                tests['alpha_significance'] = self._test_alpha_significance(strategy_returns, benchmark_returns, risk_free_rate)
            
            # Robustness tests
            tests['robustness'] = self._test_robustness(strategy_returns)
            
            # Multiple testing correction
            tests['multiple_testing'] = self._apply_multiple_testing_correction(tests)
            
            # Overall assessment
            tests['overall_assessment'] = self._assess_overall_significance(tests)
            
            self.test_results = tests
            logger.info("Comprehensive statistical tests completed")
            return tests
            
        except Exception as e:
            logger.error(f"Error in comprehensive statistical tests: {e}")
            return {}
    
    def _test_normality(self, returns: pd.Series) -> Dict[str, Any]:
        """Test for normality of returns."""
        try:
            # Shapiro-Wilk test
            shapiro_stat, shapiro_pvalue = stats.shapiro(returns)
            
            # Jarque-Bera test
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.kstest(returns, 'norm', args=(returns.mean(), returns.std()))
            
            # Skewness and Kurtosis
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            return {
                'shapiro_wilk': {
                    'statistic': shapiro_stat,
                    'pvalue': shapiro_pvalue,
                    'is_normal': shapiro_pvalue > 0.05
                },
                'jarque_bera': {
                    'statistic': jb_stat,
                    'pvalue': jb_pvalue,
                    'is_normal': jb_pvalue > 0.05
                },
                'kolmogorov_smirnov': {
                    'statistic': ks_stat,
                    'pvalue': ks_pvalue,
                    'is_normal': ks_pvalue > 0.05
                },
                'skewness': skewness,
                'kurtosis': kurtosis,
                'overall_normal': all([shapiro_pvalue > 0.05, jb_pvalue > 0.05])
            }
            
        except Exception as e:
            logger.error(f"Error testing normality: {e}")
            return {}
    
    def _test_autocorrelation(self, returns: pd.Series) -> Dict[str, Any]:
        """Test for autocorrelation in returns."""
        try:
            # Ljung-Box test
            from statsmodels.stats.diagnostic import acorr_ljungbox
            ljungbox_result = acorr_ljungbox(returns, lags=10, return_df=True)
            
            # Durbin-Watson test
            from statsmodels.stats.diagnostic import durbin_watson
            dw_stat = durbin_watson(returns)
            
            # Autocorrelation coefficients
            autocorr_coeffs = [returns.autocorr(lag=i) for i in range(1, 11)]
            
            return {
                'ljung_box': {
                    'statistic': ljungbox_result['lb_stat'].iloc[-1],
                    'pvalue': ljungbox_result['lb_pvalue'].iloc[-1],
                    'has_autocorrelation': ljungbox_result['lb_pvalue'].iloc[-1] < 0.05
                },
                'durbin_watson': {
                    'statistic': dw_stat,
                    'has_autocorrelation': dw_stat < 1.5 or dw_stat > 2.5
                },
                'autocorr_coefficients': autocorr_coeffs,
                'max_autocorr': max(abs(ac) for ac in autocorr_coeffs)
            }
            
        except Exception as e:
            logger.error(f"Error testing autocorrelation: {e}")
            return {}
    
    def _test_heteroscedasticity(self, returns: pd.Series) -> Dict[str, Any]:
        """Test for heteroscedasticity in returns."""
        try:
            # ARCH test
            from statsmodels.stats.diagnostic import het_arch
            arch_stat, arch_pvalue, _, _ = het_arch(returns)
            
            # Breusch-Pagan test
            from statsmodels.stats.diagnostic import het_breuschpagan
            try:
                # Create a simple regression model
                X = np.column_stack([np.ones(len(returns)), np.arange(len(returns))])
                bp_stat, bp_pvalue, _, _ = het_breuschpagan(returns, X)
            except:
                bp_stat, bp_pvalue = np.nan, np.nan
            
            # Rolling volatility
            rolling_vol = returns.rolling(20).std()
            vol_volatility = rolling_vol.std()
            
            return {
                'arch_test': {
                    'statistic': arch_stat,
                    'pvalue': arch_pvalue,
                    'has_heteroscedasticity': arch_pvalue < 0.05
                },
                'breusch_pagan': {
                    'statistic': bp_stat,
                    'pvalue': bp_pvalue,
                    'has_heteroscedasticity': bp_pvalue < 0.05 if not np.isnan(bp_pvalue) else False
                },
                'volatility_of_volatility': vol_volatility,
                'overall_heteroscedastic': arch_pvalue < 0.05
            }
            
        except Exception as e:
            logger.error(f"Error testing heteroscedasticity: {e}")
            return {}
    
    def _test_stationarity(self, returns: pd.Series) -> Dict[str, Any]:
        """Test for stationarity of returns."""
        try:
            # Augmented Dickey-Fuller test
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(returns)
            
            # KPSS test
            from statsmodels.tsa.stattools import kpss
            try:
                kpss_result = kpss(returns, regression='c')
            except:
                kpss_result = (np.nan, np.nan, np.nan, {})
            
            return {
                'adf_test': {
                    'statistic': adf_result[0],
                    'pvalue': adf_result[1],
                    'critical_values': adf_result[4],
                    'is_stationary': adf_result[1] < 0.05
                },
                'kpss_test': {
                    'statistic': kpss_result[0],
                    'pvalue': kpss_result[1],
                    'critical_values': kpss_result[3],
                    'is_stationary': kpss_result[1] > 0.05
                },
                'overall_stationary': adf_result[1] < 0.05
            }
            
        except Exception as e:
            logger.error(f"Error testing stationarity: {e}")
            return {}
    
    def _test_performance_significance(self, returns: pd.Series, risk_free_rate: float) -> Dict[str, Any]:
        """Test significance of strategy performance."""
        try:
            # Test if mean return is significantly different from risk-free rate
            excess_returns = returns - risk_free_rate / 252
            t_stat, t_pvalue = stats.ttest_1samp(excess_returns, 0)
            
            # Test if Sharpe ratio is significantly positive
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252)
            sharpe_se = np.sqrt((1 + 0.5 * sharpe_ratio**2) / len(returns))
            sharpe_t_stat = sharpe_ratio / sharpe_se
            sharpe_pvalue = 2 * (1 - stats.norm.cdf(abs(sharpe_t_stat)))
            
            # Test if returns are significantly positive
            positive_returns = (returns > 0).sum()
            total_returns = len(returns)
            win_rate = positive_returns / total_returns
            
            # Binomial test for win rate
            binom_stat, binom_pvalue = stats.binom_test(positive_returns, total_returns, 0.5, alternative='greater')
            
            return {
                'mean_return_test': {
                    't_statistic': t_stat,
                    'pvalue': t_pvalue,
                    'is_significant': t_pvalue < 0.05
                },
                'sharpe_ratio_test': {
                    'sharpe_ratio': sharpe_ratio,
                    't_statistic': sharpe_t_stat,
                    'pvalue': sharpe_pvalue,
                    'is_significant': sharpe_pvalue < 0.05
                },
                'win_rate_test': {
                    'win_rate': win_rate,
                    'pvalue': binom_pvalue,
                    'is_significant': binom_pvalue < 0.05
                }
            }
            
        except Exception as e:
            logger.error(f"Error testing performance significance: {e}")
            return {}
    
    def _bootstrap_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float) -> Dict[str, Any]:
        """Bootstrap test for Sharpe ratio."""
        try:
            bootstrap_sharpes = []
            
            for _ in range(self.config.bootstrap_samples):
                bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
                excess_returns = bootstrap_sample - risk_free_rate / 252
                sharpe = excess_returns.mean() / bootstrap_sample.std() * np.sqrt(252)
                bootstrap_sharpes.append(sharpe)
            
            bootstrap_sharpes = np.array(bootstrap_sharpes)
            
            # Calculate confidence intervals
            ci_lower = np.percentile(bootstrap_sharpes, (1 - self.config.confidence_level) / 2 * 100)
            ci_upper = np.percentile(bootstrap_sharpes, (1 + self.config.confidence_level) / 2 * 100)
            
            # Test if Sharpe ratio is significantly positive
            prob_positive = np.mean(bootstrap_sharpes > 0)
            
            return {
                'mean_sharpe': np.mean(bootstrap_sharpes),
                'std_sharpe': np.std(bootstrap_sharpes),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'prob_positive': prob_positive,
                'is_significant': ci_lower > 0
            }
            
        except Exception as e:
            logger.error(f"Error in bootstrap Sharpe ratio test: {e}")
            return {}
    
    def _bootstrap_max_drawdown(self, returns: pd.Series) -> Dict[str, Any]:
        """Bootstrap test for maximum drawdown."""
        try:
            bootstrap_drawdowns = []
            
            for _ in range(self.config.bootstrap_samples):
                bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
                cumulative_returns = (1 + bootstrap_sample).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdowns = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdowns.min()
                bootstrap_drawdowns.append(max_drawdown)
            
            bootstrap_drawdowns = np.array(bootstrap_drawdowns)
            
            # Calculate confidence intervals
            ci_lower = np.percentile(bootstrap_drawdowns, (1 - self.config.confidence_level) / 2 * 100)
            ci_upper = np.percentile(bootstrap_drawdowns, (1 + self.config.confidence_level) / 2 * 100)
            
            return {
                'mean_drawdown': np.mean(bootstrap_drawdowns),
                'std_drawdown': np.std(bootstrap_drawdowns),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'prob_acceptable': np.mean(bootstrap_drawdowns > -0.15)  # 15% threshold
            }
            
        except Exception as e:
            logger.error(f"Error in bootstrap max drawdown test: {e}")
            return {}
    
    def _bootstrap_returns(self, returns: pd.Series) -> Dict[str, Any]:
        """Bootstrap analysis of returns."""
        try:
            bootstrap_returns = []
            
            for _ in range(self.config.bootstrap_samples):
                bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
                bootstrap_returns.append(bootstrap_sample.mean())
            
            bootstrap_returns = np.array(bootstrap_returns)
            
            # Calculate confidence intervals
            ci_lower = np.percentile(bootstrap_returns, (1 - self.config.confidence_level) / 2 * 100)
            ci_upper = np.percentile(bootstrap_returns, (1 + self.config.confidence_level) / 2 * 100)
            
            return {
                'mean_return': np.mean(bootstrap_returns),
                'std_return': np.std(bootstrap_returns),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'prob_positive': np.mean(bootstrap_returns > 0)
            }
            
        except Exception as e:
            logger.error(f"Error in bootstrap returns test: {e}")
            return {}
    
    def _test_benchmark_comparison(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, Any]:
        """Test strategy performance against benchmark."""
        try:
            # Paired t-test
            excess_returns = strategy_returns - benchmark_returns
            t_stat, t_pvalue = stats.ttest_1samp(excess_returns, 0)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_pvalue = stats.mannwhitneyu(strategy_returns, benchmark_returns, alternative='two-sided')
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(strategy_returns, benchmark_returns)
            
            # Correlation test
            correlation, corr_pvalue = stats.pearsonr(strategy_returns, benchmark_returns)
            
            return {
                'paired_t_test': {
                    'statistic': t_stat,
                    'pvalue': t_pvalue,
                    'is_significant': t_pvalue < 0.05
                },
                'mann_whitney': {
                    'statistic': u_stat,
                    'pvalue': u_pvalue,
                    'is_significant': u_pvalue < 0.05
                },
                'kolmogorov_smirnov': {
                    'statistic': ks_stat,
                    'pvalue': ks_pvalue,
                    'is_significant': ks_pvalue < 0.05
                },
                'correlation': {
                    'coefficient': correlation,
                    'pvalue': corr_pvalue,
                    'is_significant': corr_pvalue < 0.05
                }
            }
            
        except Exception as e:
            logger.error(f"Error testing benchmark comparison: {e}")
            return {}
    
    def _test_alpha_significance(self, strategy_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float) -> Dict[str, Any]:
        """Test significance of alpha."""
        try:
            # Calculate alpha and beta
            excess_strategy = strategy_returns - risk_free_rate / 252
            excess_benchmark = benchmark_returns - risk_free_rate / 252
            
            # Regression to get alpha and beta
            from sklearn.linear_model import LinearRegression
            X = excess_benchmark.values.reshape(-1, 1)
            y = excess_strategy.values
            
            reg = LinearRegression().fit(X, y)
            alpha = reg.intercept_ * 252  # Annualized
            beta = reg.coef_[0]
            
            # Calculate standard errors
            residuals = y - reg.predict(X)
            mse = np.mean(residuals**2)
            n = len(y)
            
            # Standard error of alpha
            alpha_se = np.sqrt(mse * (1/n + np.mean(X)**2 / np.sum((X - np.mean(X))**2))) * np.sqrt(252)
            
            # t-test for alpha
            alpha_t_stat = alpha / alpha_se
            alpha_pvalue = 2 * (1 - stats.norm.cdf(abs(alpha_t_stat)))
            
            return {
                'alpha': alpha,
                'beta': beta,
                'alpha_se': alpha_se,
                't_statistic': alpha_t_stat,
                'pvalue': alpha_pvalue,
                'is_significant': alpha_pvalue < 0.05
            }
            
        except Exception as e:
            logger.error(f"Error testing alpha significance: {e}")
            return {}
    
    def _test_robustness(self, returns: pd.Series) -> Dict[str, Any]:
        """Test robustness of strategy performance."""
        try:
            # Test with different subsamples
            n = len(returns)
            subsample_sizes = [int(n * 0.5), int(n * 0.75), int(n * 0.9)]
            
            subsample_results = {}
            for size in subsample_sizes:
                if size < 50:  # Minimum sample size
                    continue
                
                subsample_returns = returns.sample(n=size, random_state=self.config.random_seed)
                subsample_sharpe = subsample_returns.mean() / subsample_returns.std() * np.sqrt(252)
                subsample_results[f'size_{size}'] = {
                    'sharpe_ratio': subsample_sharpe,
                    'mean_return': subsample_returns.mean() * 252,
                    'volatility': subsample_returns.std() * np.sqrt(252)
                }
            
            # Test with different time periods
            period_results = {}
            if n > 252:  # At least 1 year of data
                # Split into quarters
                quarter_size = n // 4
                for i in range(4):
                    start_idx = i * quarter_size
                    end_idx = (i + 1) * quarter_size if i < 3 else n
                    quarter_returns = returns.iloc[start_idx:end_idx]
                    
                    if len(quarter_returns) > 20:  # Minimum sample size
                        quarter_sharpe = quarter_returns.mean() / quarter_returns.std() * np.sqrt(252)
                        period_results[f'quarter_{i+1}'] = {
                            'sharpe_ratio': quarter_sharpe,
                            'mean_return': quarter_returns.mean() * 252,
                            'volatility': quarter_returns.std() * np.sqrt(252)
                        }
            
            # Calculate robustness metrics
            sharpe_ratios = [result['sharpe_ratio'] for result in subsample_results.values()]
            sharpe_ratios.extend([result['sharpe_ratio'] for result in period_results.values()])
            
            if sharpe_ratios:
                sharpe_std = np.std(sharpe_ratios)
                sharpe_cv = sharpe_std / np.mean(sharpe_ratios) if np.mean(sharpe_ratios) != 0 else np.inf
                robustness_score = 1 / (1 + sharpe_cv)  # Higher is more robust
            else:
                sharpe_std = 0
                sharpe_cv = 0
                robustness_score = 0
            
            return {
                'subsample_results': subsample_results,
                'period_results': period_results,
                'sharpe_std': sharpe_std,
                'sharpe_cv': sharpe_cv,
                'robustness_score': robustness_score,
                'is_robust': robustness_score > 0.5
            }
            
        except Exception as e:
            logger.error(f"Error testing robustness: {e}")
            return {}
    
    def _apply_multiple_testing_correction(self, tests: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multiple testing correction."""
        try:
            # Collect all p-values
            pvalues = []
            test_names = []
            
            for test_name, test_result in tests.items():
                if isinstance(test_result, dict):
                    for subtest_name, subtest_result in test_result.items():
                        if isinstance(subtest_result, dict) and 'pvalue' in subtest_result:
                            pvalues.append(subtest_result['pvalue'])
                            test_names.append(f"{test_name}_{subtest_name}")
            
            if not pvalues:
                return {'corrected_tests': {}, 'correction_method': 'none'}
            
            # Apply Bonferroni correction
            bonferroni_corrected = [p * len(pvalues) for p in pvalues]
            bonferroni_corrected = [min(p, 1.0) for p in bonferroni_corrected]
            
            # Apply FDR correction (Benjamini-Hochberg)
            from statsmodels.stats.multitest import multipletests
            fdr_corrected = multipletests(pvalues, method='fdr_bh')[1]
            
            # Create corrected results
            corrected_results = {}
            for i, test_name in enumerate(test_names):
                corrected_results[test_name] = {
                    'original_pvalue': pvalues[i],
                    'bonferroni_corrected': bonferroni_corrected[i],
                    'fdr_corrected': fdr_corrected[i],
                    'is_significant_bonferroni': bonferroni_corrected[i] < 0.05,
                    'is_significant_fdr': fdr_corrected[i] < 0.05
                }
            
            return {
                'corrected_tests': corrected_results,
                'correction_method': 'bonferroni_fdr',
                'num_tests': len(pvalues)
            }
            
        except Exception as e:
            logger.error(f"Error applying multiple testing correction: {e}")
            return {}
    
    def _assess_overall_significance(self, tests: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall significance of strategy performance."""
        try:
            # Count significant tests
            significant_tests = 0
            total_tests = 0
            
            # Key tests to consider
            key_tests = [
                'performance_significance.mean_return_test',
                'performance_significance.sharpe_ratio_test',
                'performance_significance.win_rate_test',
                'bootstrap_sharpe.is_significant',
                'alpha_significance.is_significant'
            ]
            
            for test_path in key_tests:
                test_parts = test_path.split('.')
                test_result = tests
                
                try:
                    for part in test_parts:
                        test_result = test_result[part]
                    
                    if isinstance(test_result, dict) and 'is_significant' in test_result:
                        total_tests += 1
                        if test_result['is_significant']:
                            significant_tests += 1
                    elif isinstance(test_result, bool):
                        total_tests += 1
                        if test_result:
                            significant_tests += 1
                except:
                    continue
            
            # Calculate overall significance
            significance_ratio = significant_tests / total_tests if total_tests > 0 else 0
            
            # Determine overall assessment
            if significance_ratio >= 0.8:
                assessment = "HIGHLY_SIGNIFICANT"
            elif significance_ratio >= 0.6:
                assessment = "SIGNIFICANT"
            elif significance_ratio >= 0.4:
                assessment = "MODERATELY_SIGNIFICANT"
            else:
                assessment = "NOT_SIGNIFICANT"
            
            return {
                'significant_tests': significant_tests,
                'total_tests': total_tests,
                'significance_ratio': significance_ratio,
                'assessment': assessment,
                'is_overall_significant': significance_ratio >= 0.6
            }
            
        except Exception as e:
            logger.error(f"Error assessing overall significance: {e}")
            return {}
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of statistical test results."""
        try:
            if not self.test_results:
                return {'error': 'No test results available'}
            
            summary = {
                'overall_assessment': self.test_results.get('overall_assessment', {}),
                'key_findings': [],
                'recommendations': []
            }
            
            # Extract key findings
            overall = self.test_results.get('overall_assessment', {})
            if overall.get('is_overall_significant', False):
                summary['key_findings'].append("Strategy shows statistically significant performance")
            else:
                summary['key_findings'].append("Strategy performance is not statistically significant")
            
            # Add specific findings
            if 'normality' in self.test_results:
                norm_test = self.test_results['normality']
                if norm_test.get('overall_normal', False):
                    summary['key_findings'].append("Returns are normally distributed")
                else:
                    summary['key_findings'].append("Returns are not normally distributed")
            
            if 'autocorrelation' in self.test_results:
                autocorr_test = self.test_results['autocorrelation']
                if autocorr_test.get('ljung_box', {}).get('has_autocorrelation', False):
                    summary['key_findings'].append("Returns show autocorrelation")
                else:
                    summary['key_findings'].append("Returns show no significant autocorrelation")
            
            # Generate recommendations
            if not overall.get('is_overall_significant', False):
                summary['recommendations'].append("Consider improving strategy or increasing sample size")
            
            if 'robustness' in self.test_results:
                robustness = self.test_results['robustness']
                if not robustness.get('is_robust', False):
                    summary['recommendations'].append("Strategy performance is not robust across different periods")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting test summary: {e}")
            return {}


# Utility functions
def run_statistical_tests(strategy_returns: pd.Series,
                         benchmark_returns: Optional[pd.Series] = None,
                         config: Optional[StatisticalTestConfig] = None) -> Dict[str, Any]:
    """Quick function to run statistical tests."""
    if config is None:
        config = StatisticalTestConfig()
    
    tester = StatisticalTester(config)
    return tester.run_comprehensive_tests(strategy_returns, benchmark_returns)


def create_statistical_test_config(confidence_level: float = 0.95, 
                                  bootstrap_samples: int = 1000) -> StatisticalTestConfig:
    """Create a statistical test configuration."""
    return StatisticalTestConfig(
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples
    )


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="2y")
    
    # Create mock strategy returns
    strategy_returns = df['Close'].pct_change().dropna()
    
    # Create mock benchmark returns
    benchmark_ticker = yf.Ticker("SPY")
    benchmark_df = benchmark_ticker.history(period="2y")
    benchmark_returns = benchmark_df['Close'].pct_change().dropna()
    
    # Align data
    common_dates = strategy_returns.index.intersection(benchmark_returns.index)
    strategy_returns = strategy_returns.loc[common_dates]
    benchmark_returns = benchmark_returns.loc[common_dates]
    
    # Run statistical tests
    config = StatisticalTestConfig()
    tester = StatisticalTester(config)
    results = tester.run_comprehensive_tests(strategy_returns, benchmark_returns)
    
    print("Statistical tests completed!")
    
    # Get test summary
    summary = tester.get_test_summary()
    print(f"Overall Assessment: {summary['overall_assessment']['assessment']}")
    print(f"Key Findings: {summary['key_findings']}")
    print(f"Recommendations: {summary['recommendations']}")
    
    # Show specific results
    if 'performance_significance' in results:
        perf_test = results['performance_significance']
        print(f"\nPerformance Significance:")
        print(f"Sharpe Ratio Test: {perf_test['sharpe_ratio_test']['is_significant']}")
        print(f"Mean Return Test: {perf_test['mean_return_test']['is_significant']}")
