"""
Risk management module implementing stop-loss, take-profit, and portfolio-level controls.
Provides comprehensive risk management for trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import talib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Enumeration of risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAction(Enum):
    """Enumeration of risk management actions."""
    HOLD = "hold"
    REDUCE = "reduce"
    CLOSE = "close"
    EMERGENCY_EXIT = "emergency_exit"


@dataclass
class RiskManagementConfig:
    """Configuration for risk management."""
    # Position-level risk controls
    max_position_size: float = 0.05  # Maximum 5% per position
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit
    trailing_stop_pct: float = 0.015  # 1.5% trailing stop
    
    # Portfolio-level risk controls
    max_portfolio_risk: float = 0.20  # Maximum 20% portfolio risk
    max_drawdown: float = 0.15  # Maximum 15% drawdown
    max_correlation: float = 0.7  # Maximum correlation between positions
    max_sector_exposure: float = 0.30  # Maximum 30% sector exposure
    
    # Risk monitoring
    var_confidence: float = 0.95  # VaR confidence level
    stress_test_scenarios: int = 1000  # Number of stress test scenarios
    risk_check_frequency: int = 300  # Risk check frequency in seconds
    
    # Emergency controls
    emergency_stop_loss: float = 0.05  # 5% emergency stop loss
    max_daily_loss: float = 0.03  # Maximum 3% daily loss
    max_consecutive_losses: int = 5  # Maximum consecutive losses


class RiskManager:
    """Comprehensive risk management system."""
    
    def __init__(self, config: RiskManagementConfig):
        self.config = config
        self.positions = {}
        self.portfolio_history = []
        self.risk_alerts = []
        self.emergency_mode = False
        
    def validate_position(self, 
                         symbol: str,
                         position_size: float,
                         entry_price: float,
                         signal_confidence: float,
                         current_portfolio_value: float) -> Dict[str, Any]:
        """Validate a new position against risk parameters."""
        try:
            validation_result = {
                'approved': True,
                'warnings': [],
                'adjustments': {},
                'risk_level': RiskLevel.LOW
            }
            
            # Check position size limits
            position_value = position_size * current_portfolio_value
            position_pct = position_value / current_portfolio_value
            
            if position_pct > self.config.max_position_size:
                validation_result['approved'] = False
                validation_result['warnings'].append(f"Position size {position_pct:.2%} exceeds maximum {self.config.max_position_size:.2%}")
                validation_result['adjustments']['position_size'] = self.config.max_position_size
                validation_result['risk_level'] = RiskLevel.HIGH
            
            # Check signal confidence
            if signal_confidence < 0.6:
                validation_result['warnings'].append(f"Low signal confidence: {signal_confidence:.2%}")
                validation_result['risk_level'] = RiskLevel.MEDIUM
            
            # Check portfolio concentration
            total_exposure = sum(pos['position_size'] for pos in self.positions.values())
            if total_exposure + position_pct > 0.8:  # 80% maximum total exposure
                validation_result['warnings'].append("Portfolio concentration too high")
                validation_result['risk_level'] = RiskLevel.HIGH
            
            # Check correlation with existing positions
            correlation_warnings = self._check_correlation_risk(symbol, position_pct)
            validation_result['warnings'].extend(correlation_warnings)
            
            logger.info(f"Position validation for {symbol}: {'Approved' if validation_result['approved'] else 'Rejected'}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating position: {e}")
            return {'approved': False, 'warnings': [f'Validation error: {e}'], 'risk_level': RiskLevel.CRITICAL}
    
    def add_position(self, 
                    symbol: str,
                    position_size: float,
                    entry_price: float,
                    entry_time: datetime,
                    signal_confidence: float) -> bool:
        """Add a new position to risk management."""
        try:
            position = {
                'symbol': symbol,
                'position_size': position_size,
                'entry_price': entry_price,
                'entry_time': entry_time,
                'current_price': entry_price,
                'signal_confidence': signal_confidence,
                'stop_loss': entry_price * (1 - self.config.stop_loss_pct),
                'take_profit': entry_price * (1 + self.config.take_profit_pct),
                'trailing_stop': entry_price * (1 - self.config.trailing_stop_pct),
                'max_price': entry_price,
                'unrealized_pnl': 0.0,
                'status': 'OPEN'
            }
            
            self.positions[symbol] = position
            logger.info(f"Added position for {symbol}: {position_size:.2%} at ${entry_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return False
    
    def update_position(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Update position with current price and check risk triggers."""
        try:
            if symbol not in self.positions:
                return {'action': RiskAction.HOLD, 'reason': 'Position not found'}
            
            position = self.positions[symbol]
            position['current_price'] = current_price
            
            # Calculate unrealized P&L
            price_change = (current_price - position['entry_price']) / position['entry_price']
            position['unrealized_pnl'] = price_change * position['position_size']
            
            # Update trailing stop
            if current_price > position['max_price']:
                position['max_price'] = current_price
                new_trailing_stop = current_price * (1 - self.config.trailing_stop_pct)
                position['trailing_stop'] = max(position['trailing_stop'], new_trailing_stop)
            
            # Check risk triggers
            risk_action = self._check_position_risk_triggers(position)
            
            if risk_action['action'] != RiskAction.HOLD:
                logger.info(f"Risk trigger for {symbol}: {risk_action['action'].value} - {risk_action['reason']}")
            
            return risk_action
            
        except Exception as e:
            logger.error(f"Error updating position: {e}")
            return {'action': RiskAction.HOLD, 'reason': f'Update error: {e}'}
    
    def _check_position_risk_triggers(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Check if position triggers any risk management actions."""
        try:
            current_price = position['current_price']
            entry_price = position['entry_price']
            
            # Check stop loss
            if current_price <= position['stop_loss']:
                return {
                    'action': RiskAction.CLOSE,
                    'reason': f'Stop loss triggered at ${current_price:.2f}',
                    'trigger_price': current_price
                }
            
            # Check trailing stop
            if current_price <= position['trailing_stop']:
                return {
                    'action': RiskAction.CLOSE,
                    'reason': f'Trailing stop triggered at ${current_price:.2f}',
                    'trigger_price': current_price
                }
            
            # Check take profit
            if current_price >= position['take_profit']:
                return {
                    'action': RiskAction.CLOSE,
                    'reason': f'Take profit triggered at ${current_price:.2f}',
                    'trigger_price': current_price
                }
            
            # Check emergency stop loss
            emergency_stop = entry_price * (1 - self.config.emergency_stop_loss)
            if current_price <= emergency_stop:
                return {
                    'action': RiskAction.EMERGENCY_EXIT,
                    'reason': f'Emergency stop loss triggered at ${current_price:.2f}',
                    'trigger_price': current_price
                }
            
            return {'action': RiskAction.HOLD, 'reason': 'No triggers'}
            
        except Exception as e:
            logger.error(f"Error checking risk triggers: {e}")
            return {'action': RiskAction.HOLD, 'reason': f'Check error: {e}'}
    
    def check_portfolio_risk(self, 
                           portfolio_value: float,
                           portfolio_positions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Check portfolio-level risk metrics."""
        try:
            risk_metrics = {
                'total_exposure': 0.0,
                'unrealized_pnl': 0.0,
                'var_95': 0.0,
                'max_drawdown': 0.0,
                'correlation_risk': 0.0,
                'sector_concentration': {},
                'risk_level': RiskLevel.LOW,
                'actions': []
            }
            
            # Calculate total exposure and P&L
            for symbol, position in portfolio_positions.items():
                position_value = position['position_size'] * portfolio_value
                risk_metrics['total_exposure'] += position['position_size']
                risk_metrics['unrealized_pnl'] += position.get('unrealized_pnl', 0) * portfolio_value
            
            # Calculate Value at Risk
            risk_metrics['var_95'] = self._calculate_portfolio_var(portfolio_positions, portfolio_value)
            
            # Calculate maximum drawdown
            risk_metrics['max_drawdown'] = self._calculate_max_drawdown()
            
            # Check correlation risk
            risk_metrics['correlation_risk'] = self._calculate_correlation_risk(portfolio_positions)
            
            # Check sector concentration
            risk_metrics['sector_concentration'] = self._calculate_sector_concentration(portfolio_positions)
            
            # Determine overall risk level
            risk_metrics['risk_level'] = self._determine_risk_level(risk_metrics)
            
            # Generate risk actions
            risk_metrics['actions'] = self._generate_risk_actions(risk_metrics)
            
            # Store portfolio history
            self._store_portfolio_history(risk_metrics, portfolio_value)
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error checking portfolio risk: {e}")
            return {'risk_level': RiskLevel.CRITICAL, 'actions': ['Error in risk calculation']}
    
    def _calculate_portfolio_var(self, 
                                portfolio_positions: Dict[str, Dict[str, Any]], 
                                portfolio_value: float) -> float:
        """Calculate portfolio Value at Risk."""
        try:
            if not portfolio_positions:
                return 0.0
            
            # Simplified VaR calculation
            total_volatility = 0.0
            for position in portfolio_positions.values():
                # Assume 20% volatility for each position (simplified)
                position_volatility = position['position_size'] * 0.20
                total_volatility += position_volatility ** 2
            
            portfolio_volatility = np.sqrt(total_volatility)
            
            # VaR at specified confidence level
            confidence_factor = 1.645 if self.config.var_confidence == 0.95 else 2.326  # 99%
            var = portfolio_value * portfolio_volatility * confidence_factor
            
            return var
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from portfolio history."""
        try:
            if len(self.portfolio_history) < 2:
                return 0.0
            
            portfolio_values = [entry['portfolio_value'] for entry in self.portfolio_history]
            peak = portfolio_values[0]
            max_dd = 0.0
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
            
            return max_dd
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_correlation_risk(self, portfolio_positions: Dict[str, Dict[str, Any]]) -> float:
        """Calculate correlation risk in portfolio."""
        try:
            if len(portfolio_positions) < 2:
                return 0.0
            
            # Simplified correlation risk calculation
            symbols = list(portfolio_positions.keys())
            max_correlation = 0.0
            
            # Check pairwise correlations (simplified)
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    # Assume correlation based on sector similarity (simplified)
                    correlation = self._estimate_correlation(symbol1, symbol2)
                    max_correlation = max(max_correlation, correlation)
            
            return max_correlation
            
        except Exception as e:
            logger.error(f"Error calculating correlation risk: {e}")
            return 0.0
    
    def _estimate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Estimate correlation between two symbols (simplified)."""
        # This is a simplified implementation
        # In practice, you would use historical correlation data
        
        # Tech stocks correlation
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']
        if symbol1 in tech_stocks and symbol2 in tech_stocks:
            return 0.7
        
        # Financial stocks correlation
        financial_stocks = ['JPM', 'BAC', 'WFC', 'GS', 'MS']
        if symbol1 in financial_stocks and symbol2 in financial_stocks:
            return 0.6
        
        # Default correlation
        return 0.3
    
    def _calculate_sector_concentration(self, portfolio_positions: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate sector concentration risk."""
        try:
            sector_exposure = {}
            
            for symbol, position in portfolio_positions.items():
                sector = self._get_sector(symbol)
                if sector not in sector_exposure:
                    sector_exposure[sector] = 0.0
                sector_exposure[sector] += position['position_size']
            
            return sector_exposure
            
        except Exception as e:
            logger.error(f"Error calculating sector concentration: {e}")
            return {}
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol (simplified)."""
        # Simplified sector mapping
        sector_map = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'AMZN': 'Technology', 'META': 'Technology', 'NVDA': 'Technology',
            'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
            'GS': 'Financials', 'MS': 'Financials',
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy'
        }
        return sector_map.get(symbol, 'Other')
    
    def _determine_risk_level(self, risk_metrics: Dict[str, Any]) -> RiskLevel:
        """Determine overall portfolio risk level."""
        try:
            risk_score = 0
            
            # Total exposure risk
            if risk_metrics['total_exposure'] > 0.8:
                risk_score += 3
            elif risk_metrics['total_exposure'] > 0.6:
                risk_score += 2
            elif risk_metrics['total_exposure'] > 0.4:
                risk_score += 1
            
            # Drawdown risk
            if risk_metrics['max_drawdown'] > self.config.max_drawdown:
                risk_score += 3
            elif risk_metrics['max_drawdown'] > self.config.max_drawdown * 0.8:
                risk_score += 2
            elif risk_metrics['max_drawdown'] > self.config.max_drawdown * 0.6:
                risk_score += 1
            
            # Correlation risk
            if risk_metrics['correlation_risk'] > self.config.max_correlation:
                risk_score += 2
            elif risk_metrics['correlation_risk'] > self.config.max_correlation * 0.8:
                risk_score += 1
            
            # Determine risk level
            if risk_score >= 6:
                return RiskLevel.CRITICAL
            elif risk_score >= 4:
                return RiskLevel.HIGH
            elif risk_score >= 2:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"Error determining risk level: {e}")
            return RiskLevel.CRITICAL
    
    def _generate_risk_actions(self, risk_metrics: Dict[str, Any]) -> List[str]:
        """Generate risk management actions based on metrics."""
        try:
            actions = []
            
            # High exposure actions
            if risk_metrics['total_exposure'] > 0.8:
                actions.append("Reduce total portfolio exposure")
            
            # High drawdown actions
            if risk_metrics['max_drawdown'] > self.config.max_drawdown:
                actions.append("Implement drawdown protection")
            
            # High correlation actions
            if risk_metrics['correlation_risk'] > self.config.max_correlation:
                actions.append("Reduce correlated positions")
            
            # Sector concentration actions
            for sector, exposure in risk_metrics['sector_concentration'].items():
                if exposure > self.config.max_sector_exposure:
                    actions.append(f"Reduce {sector} sector exposure")
            
            return actions
            
        except Exception as e:
            logger.error(f"Error generating risk actions: {e}")
            return ["Error in risk action generation"]
    
    def _check_correlation_risk(self, symbol: str, position_size: float) -> List[str]:
        """Check correlation risk for a new position."""
        try:
            warnings = []
            
            for existing_symbol, existing_position in self.positions.items():
                correlation = self._estimate_correlation(symbol, existing_symbol)
                
                if correlation > self.config.max_correlation:
                    warnings.append(f"High correlation ({correlation:.2f}) with {existing_symbol}")
                
                # Check combined exposure
                combined_exposure = position_size + existing_position['position_size']
                if combined_exposure > self.config.max_position_size * 1.5:
                    warnings.append(f"High combined exposure with {existing_symbol}")
            
            return warnings
            
        except Exception as e:
            logger.error(f"Error checking correlation risk: {e}")
            return [f"Correlation check error: {e}"]
    
    def _store_portfolio_history(self, risk_metrics: Dict[str, Any], portfolio_value: float):
        """Store portfolio risk history."""
        try:
            history_entry = {
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_value,
                'total_exposure': risk_metrics['total_exposure'],
                'unrealized_pnl': risk_metrics['unrealized_pnl'],
                'var_95': risk_metrics['var_95'],
                'max_drawdown': risk_metrics['max_drawdown'],
                'risk_level': risk_metrics['risk_level'].value
            }
            
            self.portfolio_history.append(history_entry)
            
            # Keep only recent history
            if len(self.portfolio_history) > 1000:
                self.portfolio_history = self.portfolio_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error storing portfolio history: {e}")
    
    def close_position(self, symbol: str, reason: str = "Manual close") -> bool:
        """Close a position."""
        try:
            if symbol not in self.positions:
                logger.warning(f"Position {symbol} not found for closing")
                return False
            
            position = self.positions[symbol]
            position['status'] = 'CLOSED'
            position['close_reason'] = reason
            position['close_time'] = datetime.now()
            
            logger.info(f"Closed position {symbol}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        try:
            summary = {
                'active_positions': len([p for p in self.positions.values() if p['status'] == 'OPEN']),
                'total_positions': len(self.positions),
                'emergency_mode': self.emergency_mode,
                'risk_alerts': len(self.risk_alerts),
                'portfolio_history_length': len(self.portfolio_history)
            }
            
            if self.portfolio_history:
                latest_risk = self.portfolio_history[-1]
                summary.update({
                    'latest_portfolio_value': latest_risk['portfolio_value'],
                    'latest_exposure': latest_risk['total_exposure'],
                    'latest_risk_level': latest_risk['risk_level'],
                    'latest_max_drawdown': latest_risk['max_drawdown']
                })
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {}


# Utility functions
def create_risk_manager(config: Optional[RiskManagementConfig] = None) -> RiskManager:
    """Create a risk manager with default or custom configuration."""
    if config is None:
        config = RiskManagementConfig()
    return RiskManager(config)


def validate_trading_signal(symbol: str,
                          position_size: float,
                          entry_price: float,
                          signal_confidence: float,
                          portfolio_value: float,
                          risk_manager: RiskManager) -> Dict[str, Any]:
    """Quick function to validate a trading signal."""
    return risk_manager.validate_position(
        symbol, position_size, entry_price, signal_confidence, portfolio_value
    )


if __name__ == "__main__":
    # Example usage
    config = RiskManagementConfig()
    risk_manager = RiskManager(config)
    
    # Validate a position
    validation = risk_manager.validate_position(
        symbol="AAPL",
        position_size=0.03,  # 3% position
        entry_price=150.0,
        signal_confidence=0.75,
        current_portfolio_value=100000
    )
    
    print("Position validation:")
    print(f"Approved: {validation['approved']}")
    print(f"Warnings: {validation['warnings']}")
    print(f"Risk level: {validation['risk_level'].value}")
    
    # Add position
    risk_manager.add_position(
        symbol="AAPL",
        position_size=0.03,
        entry_price=150.0,
        entry_time=datetime.now(),
        signal_confidence=0.75
    )
    
    # Update position
    risk_action = risk_manager.update_position("AAPL", 145.0)  # Price dropped
    print(f"\nRisk action: {risk_action['action'].value} - {risk_action['reason']}")
    
    # Check portfolio risk
    portfolio_positions = {"AAPL": {"position_size": 0.03, "unrealized_pnl": -0.02}}
    risk_metrics = risk_manager.check_portfolio_risk(100000, portfolio_positions)
    
    print(f"\nPortfolio risk level: {risk_metrics['risk_level'].value}")
    print(f"Risk actions: {risk_metrics['actions']}")
    
    # Get risk summary
    summary = risk_manager.get_risk_summary()
    print(f"\nRisk summary: {summary}")
