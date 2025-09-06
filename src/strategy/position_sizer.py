"""
Position sizing module implementing Kelly Criterion and volatility targeting.
Provides dynamic position sizing based on signal confidence and market conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats
from scipy.optimize import minimize_scalar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """Enumeration of position sizing methods."""
    KELLY = "kelly"
    VOLATILITY_TARGET = "volatility_target"
    FIXED_FRACTIONAL = "fixed_fractional"
    RISK_PARITY = "risk_parity"
    ADAPTIVE = "adaptive"


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing."""
    method: PositionSizingMethod = PositionSizingMethod.KELLY
    max_position_size: float = 0.1  # Maximum 10% of portfolio per position
    min_position_size: float = 0.01  # Minimum 1% of portfolio per position
    volatility_target: float = 0.15  # Target 15% annual volatility
    kelly_fraction: float = 0.25  # Use 25% of Kelly optimal size
    lookback_period: int = 252  # Lookback period for volatility calculation
    confidence_threshold: float = 0.6  # Minimum confidence for position sizing
    risk_free_rate: float = 0.02  # Risk-free rate for Sharpe calculations


class PositionSizer:
    """Position sizing calculator using various methods."""
    
    def __init__(self, config: PositionSizingConfig):
        self.config = config
        self.position_history = []
        self.volatility_history = []
        
    def calculate_position_size(self, 
                              signal_confidence: float,
                              expected_return: float,
                              volatility: float,
                              current_price: float,
                              portfolio_value: float,
                              signal_strength: str = 'MODERATE') -> Dict[str, Any]:
        """Calculate optimal position size based on signal and market conditions."""
        try:
            # Validate inputs
            if signal_confidence < self.config.confidence_threshold:
                return self._create_position_result(0, 'LOW_CONFIDENCE', 0)
            
            # Calculate position size based on method
            if self.config.method == PositionSizingMethod.KELLY:
                position_size = self._calculate_kelly_position(
                    expected_return, volatility, signal_confidence
                )
            elif self.config.method == PositionSizingMethod.VOLATILITY_TARGET:
                position_size = self._calculate_volatility_target_position(
                    volatility, signal_confidence
                )
            elif self.config.method == PositionSizingMethod.FIXED_FRACTIONAL:
                position_size = self._calculate_fixed_fractional_position(
                    signal_confidence, signal_strength
                )
            elif self.config.method == PositionSizingMethod.RISK_PARITY:
                position_size = self._calculate_risk_parity_position(
                    volatility, signal_confidence
                )
            elif self.config.method == PositionSizingMethod.ADAPTIVE:
                position_size = self._calculate_adaptive_position(
                    expected_return, volatility, signal_confidence, signal_strength
                )
            else:
                raise ValueError(f"Unknown position sizing method: {self.config.method}")
            
            # Apply position limits
            position_size = self._apply_position_limits(position_size, signal_confidence)
            
            # Calculate position value and shares
            position_value = position_size * portfolio_value
            shares = int(position_value / current_price) if current_price > 0 else 0
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                position_size, expected_return, volatility, portfolio_value
            )
            
            result = self._create_position_result(
                position_size, 'SUCCESS', shares, position_value, risk_metrics
            )
            
            # Store position history
            self._store_position_history(result, expected_return, volatility)
            
            logger.info(f"Position size calculated: {position_size:.3f} ({shares} shares)")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self._create_position_result(0, 'ERROR', 0)
    
    def _calculate_kelly_position(self, 
                                expected_return: float, 
                                volatility: float, 
                                confidence: float) -> float:
        """Calculate position size using Kelly Criterion."""
        try:
            # Kelly formula: f = (bp - q) / b
            # where b = odds received, p = probability of win, q = probability of loss
            
            # Estimate win probability from confidence
            win_prob = confidence
            loss_prob = 1 - win_prob
            
            # Estimate odds from expected return
            if expected_return > 0:
                odds = expected_return / volatility  # Risk-adjusted return
            else:
                odds = 0
            
            # Kelly fraction
            if odds > 0:
                kelly_fraction = (odds * win_prob - loss_prob) / odds
            else:
                kelly_fraction = 0
            
            # Apply Kelly fraction scaling
            position_size = kelly_fraction * self.config.kelly_fraction
            
            # Ensure non-negative
            position_size = max(0, position_size)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating Kelly position: {e}")
            return 0
    
    def _calculate_volatility_target_position(self, 
                                            volatility: float, 
                                            confidence: float) -> float:
        """Calculate position size using volatility targeting."""
        try:
            # Target volatility scaling
            volatility_ratio = self.config.volatility_target / volatility
            
            # Scale by confidence
            confidence_scaling = confidence ** 2  # Square confidence for conservative scaling
            
            # Calculate position size
            position_size = volatility_ratio * confidence_scaling
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating volatility target position: {e}")
            return 0
    
    def _calculate_fixed_fractional_position(self, 
                                           confidence: float, 
                                           signal_strength: str) -> float:
        """Calculate position size using fixed fractional method."""
        try:
            # Base position size
            base_size = 0.05  # 5% base position
            
            # Confidence scaling
            confidence_scaling = confidence
            
            # Signal strength scaling
            strength_scaling = {
                'WEAK': 0.5,
                'MODERATE': 1.0,
                'STRONG': 1.5,
                'VERY_STRONG': 2.0
            }.get(signal_strength, 1.0)
            
            position_size = base_size * confidence_scaling * strength_scaling
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating fixed fractional position: {e}")
            return 0
    
    def _calculate_risk_parity_position(self, 
                                       volatility: float, 
                                       confidence: float) -> float:
        """Calculate position size using risk parity approach."""
        try:
            # Risk parity: equal risk contribution
            # Position size inversely proportional to volatility
            
            if volatility > 0:
                risk_parity_size = 1.0 / volatility
            else:
                risk_parity_size = 0
            
            # Normalize and scale by confidence
            position_size = (risk_parity_size / 10) * confidence  # Normalize to reasonable range
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating risk parity position: {e}")
            return 0
    
    def _calculate_adaptive_position(self, 
                                   expected_return: float, 
                                   volatility: float, 
                                   confidence: float,
                                   signal_strength: str) -> float:
        """Calculate position size using adaptive method combining multiple approaches."""
        try:
            # Get individual position sizes
            kelly_size = self._calculate_kelly_position(expected_return, volatility, confidence)
            vol_target_size = self._calculate_volatility_target_position(volatility, confidence)
            fixed_size = self._calculate_fixed_fractional_position(confidence, signal_strength)
            
            # Weighted combination
            weights = {
                'kelly': 0.4,
                'volatility_target': 0.3,
                'fixed_fractional': 0.3
            }
            
            adaptive_size = (
                kelly_size * weights['kelly'] +
                vol_target_size * weights['volatility_target'] +
                fixed_size * weights['fixed_fractional']
            )
            
            return adaptive_size
            
        except Exception as e:
            logger.error(f"Error calculating adaptive position: {e}")
            return 0
    
    def _apply_position_limits(self, position_size: float, confidence: float) -> float:
        """Apply position size limits based on confidence and configuration."""
        try:
            # Scale limits by confidence
            max_size = self.config.max_position_size * confidence
            min_size = self.config.min_position_size
            
            # Apply limits
            position_size = max(min_size, min(max_size, position_size))
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error applying position limits: {e}")
            return 0
    
    def _calculate_risk_metrics(self, 
                              position_size: float, 
                              expected_return: float, 
                              volatility: float,
                              portfolio_value: float) -> Dict[str, float]:
        """Calculate risk metrics for the position."""
        try:
            position_value = position_size * portfolio_value
            
            # Value at Risk (VaR) - 95% confidence
            var_95 = position_value * volatility * 1.645
            
            # Expected Shortfall (ES) - 95% confidence
            es_95 = position_value * volatility * 2.06
            
            # Maximum Drawdown estimate
            max_dd_estimate = position_value * volatility * 2.0
            
            # Sharpe ratio estimate
            sharpe_estimate = expected_return / volatility if volatility > 0 else 0
            
            # Risk-adjusted return
            risk_adj_return = expected_return / volatility if volatility > 0 else 0
            
            metrics = {
                'position_value': position_value,
                'var_95': var_95,
                'es_95': es_95,
                'max_dd_estimate': max_dd_estimate,
                'sharpe_estimate': sharpe_estimate,
                'risk_adj_return': risk_adj_return,
                'volatility': volatility,
                'expected_return': expected_return
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _create_position_result(self, 
                              position_size: float, 
                              status: str, 
                              shares: int = 0,
                              position_value: float = 0,
                              risk_metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """Create standardized position result."""
        return {
            'position_size': position_size,
            'shares': shares,
            'position_value': position_value,
            'status': status,
            'timestamp': datetime.now(),
            'risk_metrics': risk_metrics or {}
        }
    
    def _store_position_history(self, 
                              result: Dict[str, Any], 
                              expected_return: float, 
                              volatility: float):
        """Store position sizing history."""
        try:
            history_entry = {
                'timestamp': result['timestamp'],
                'position_size': result['position_size'],
                'shares': result['shares'],
                'position_value': result['position_value'],
                'expected_return': expected_return,
                'volatility': volatility,
                'status': result['status']
            }
            
            self.position_history.append(history_entry)
            
            # Keep only recent history
            if len(self.position_history) > 1000:
                self.position_history = self.position_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error storing position history: {e}")
    
    def calculate_portfolio_position_sizes(self, 
                                         signals: List[Dict[str, Any]], 
                                         portfolio_value: float,
                                         correlation_matrix: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Calculate position sizes for multiple assets considering portfolio effects."""
        try:
            if not signals:
                return []
            
            # Calculate individual position sizes
            individual_sizes = []
            for signal in signals:
                position_result = self.calculate_position_size(
                    signal_confidence=signal.get('confidence', 0.5),
                    expected_return=signal.get('expected_return', 0.02),
                    volatility=signal.get('volatility', 0.2),
                    current_price=signal.get('current_price', 100),
                    portfolio_value=portfolio_value,
                    signal_strength=signal.get('signal_strength', 'MODERATE')
                )
                individual_sizes.append(position_result)
            
            # Apply portfolio-level constraints
            total_position_size = sum(pos['position_size'] for pos in individual_sizes)
            
            # Scale down if total exceeds maximum portfolio allocation
            max_total_allocation = 0.8  # Maximum 80% of portfolio in positions
            if total_position_size > max_total_allocation:
                scaling_factor = max_total_allocation / total_position_size
                for pos in individual_sizes:
                    pos['position_size'] *= scaling_factor
                    pos['shares'] = int(pos['position_size'] * portfolio_value / signal.get('current_price', 100))
                    pos['position_value'] = pos['position_size'] * portfolio_value
            
            # Apply correlation-based adjustments if correlation matrix provided
            if correlation_matrix is not None and len(signals) > 1:
                individual_sizes = self._apply_correlation_adjustments(
                    individual_sizes, correlation_matrix
                )
            
            logger.info(f"Calculated portfolio position sizes for {len(signals)} assets")
            return individual_sizes
            
        except Exception as e:
            logger.error(f"Error calculating portfolio position sizes: {e}")
            return []
    
    def _apply_correlation_adjustments(self, 
                                     position_sizes: List[Dict[str, Any]], 
                                     correlation_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """Apply correlation-based adjustments to position sizes."""
        try:
            adjusted_sizes = position_sizes.copy()
            
            # Reduce position sizes for highly correlated assets
            for i in range(len(adjusted_sizes)):
                for j in range(i + 1, len(adjusted_sizes)):
                    correlation = correlation_matrix[i, j]
                    
                    # If correlation is high (>0.7), reduce both positions
                    if correlation > 0.7:
                        reduction_factor = 1 - (correlation - 0.7) * 0.5  # Reduce by up to 15%
                        adjusted_sizes[i]['position_size'] *= reduction_factor
                        adjusted_sizes[j]['position_size'] *= reduction_factor
            
            return adjusted_sizes
            
        except Exception as e:
            logger.error(f"Error applying correlation adjustments: {e}")
            return position_sizes
    
    def optimize_position_sizes(self, 
                               signals: List[Dict[str, Any]], 
                               portfolio_value: float,
                               objective: str = 'sharpe') -> List[Dict[str, Any]]:
        """Optimize position sizes using portfolio optimization."""
        try:
            if len(signals) < 2:
                return self.calculate_portfolio_position_sizes(signals, portfolio_value)
            
            # Extract expected returns and volatilities
            expected_returns = np.array([s.get('expected_return', 0.02) for s in signals])
            volatilities = np.array([s.get('volatility', 0.2) for s in signals])
            confidences = np.array([s.get('confidence', 0.5) for s in signals])
            
            # Create covariance matrix (simplified - assumes independence for now)
            cov_matrix = np.diag(volatilities ** 2)
            
            # Define objective function
            def objective_function(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                if objective == 'sharpe':
                    return -(portfolio_return - self.config.risk_free_rate) / portfolio_volatility
                elif objective == 'volatility':
                    return portfolio_volatility
                else:
                    return -portfolio_return
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
            ]
            
            # Bounds (0 to max position size)
            bounds = [(0, self.config.max_position_size) for _ in range(len(signals))]
            
            # Initial guess
            x0 = np.ones(len(signals)) / len(signals)
            
            # Optimize
            from scipy.optimize import minimize
            result = minimize(
                objective_function, x0, method='SLSQP',
                bounds=bounds, constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                
                # Scale by confidence
                optimal_weights = optimal_weights * confidences
                
                # Create position results
                optimized_positions = []
                for i, signal in enumerate(signals):
                    position_size = optimal_weights[i]
                    shares = int(position_size * portfolio_value / signal.get('current_price', 100))
                    
                    position_result = self._create_position_result(
                        position_size, 'OPTIMIZED', shares,
                        position_size * portfolio_value,
                        self._calculate_risk_metrics(
                            position_size, 
                            signal.get('expected_return', 0.02),
                            signal.get('volatility', 0.2),
                            portfolio_value
                        )
                    )
                    optimized_positions.append(position_result)
                
                logger.info("Position sizes optimized successfully")
                return optimized_positions
            else:
                logger.warning("Optimization failed, using individual calculations")
                return self.calculate_portfolio_position_sizes(signals, portfolio_value)
                
        except Exception as e:
            logger.error(f"Error optimizing position sizes: {e}")
            return self.calculate_portfolio_position_sizes(signals, portfolio_value)
    
    def get_position_history(self) -> pd.DataFrame:
        """Get position sizing history."""
        try:
            if not self.position_history:
                return pd.DataFrame()
            
            return pd.DataFrame(self.position_history)
            
        except Exception as e:
            logger.error(f"Error getting position history: {e}")
            return pd.DataFrame()
    
    def get_position_statistics(self) -> Dict[str, Any]:
        """Get position sizing statistics."""
        try:
            if not self.position_history:
                return {'message': 'No position history available'}
            
            history_df = pd.DataFrame(self.position_history)
            
            stats = {
                'total_positions': len(history_df),
                'avg_position_size': history_df['position_size'].mean(),
                'max_position_size': history_df['position_size'].max(),
                'min_position_size': history_df['position_size'].min(),
                'avg_volatility': history_df['volatility'].mean(),
                'avg_expected_return': history_df['expected_return'].mean(),
                'success_rate': len(history_df[history_df['status'] == 'SUCCESS']) / len(history_df)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting position statistics: {e}")
            return {}


# Utility functions
def calculate_position_size(signal_confidence: float,
                          expected_return: float,
                          volatility: float,
                          current_price: float,
                          portfolio_value: float,
                          method: PositionSizingMethod = PositionSizingMethod.KELLY) -> Dict[str, Any]:
    """Quick function to calculate position size."""
    config = PositionSizingConfig(method=method)
    sizer = PositionSizer(config)
    return sizer.calculate_position_size(
        signal_confidence, expected_return, volatility, current_price, portfolio_value
    )


def create_position_sizer(method: PositionSizingMethod = PositionSizingMethod.KELLY) -> PositionSizer:
    """Create a position sizer with specified method."""
    config = PositionSizingConfig(method=method)
    return PositionSizer(config)


if __name__ == "__main__":
    # Example usage
    config = PositionSizingConfig(method=PositionSizingMethod.KELLY)
    sizer = PositionSizer(config)
    
    # Calculate position size
    result = sizer.calculate_position_size(
        signal_confidence=0.75,
        expected_return=0.05,
        volatility=0.20,
        current_price=150.0,
        portfolio_value=100000,
        signal_strength='STRONG'
    )
    
    print("Position sizing result:")
    print(f"Position size: {result['position_size']:.3f}")
    print(f"Shares: {result['shares']}")
    print(f"Position value: ${result['position_value']:,.2f}")
    print(f"Status: {result['status']}")
    
    # Portfolio example
    signals = [
        {
            'confidence': 0.8,
            'expected_return': 0.06,
            'volatility': 0.18,
            'current_price': 150.0,
            'signal_strength': 'STRONG'
        },
        {
            'confidence': 0.6,
            'expected_return': 0.04,
            'volatility': 0.15,
            'current_price': 200.0,
            'signal_strength': 'MODERATE'
        }
    ]
    
    portfolio_positions = sizer.calculate_portfolio_position_sizes(signals, 100000)
    print(f"\nPortfolio positions calculated for {len(portfolio_positions)} assets")
    
    # Get statistics
    stats = sizer.get_position_statistics()
    print(f"\nPosition statistics: {stats}")
