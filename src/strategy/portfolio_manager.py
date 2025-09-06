"""
Portfolio manager for handling multiple positions and portfolio-level operations.
Coordinates between signal generation, position sizing, and risk management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

from .signal_generator import MultiFactorSignalGenerator, RealTimeSignalGenerator
from .position_sizer import PositionSizer, PositionSizingConfig, PositionSizingMethod
from .risk_manager import RiskManager, RiskManagementConfig, RiskAction
from ..ml_models.ensemble_predictor import EnsemblePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioStatus(Enum):
    """Enumeration of portfolio statuses."""
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class PortfolioConfig:
    """Configuration for portfolio management."""
    # Portfolio settings
    initial_capital: float = 100000.0
    max_positions: int = 10
    rebalance_frequency: int = 24  # hours
    position_review_frequency: int = 4  # hours
    
    # Signal settings
    min_signal_confidence: float = 0.6
    signal_cooldown_minutes: int = 30
    
    # Position sizing
    position_sizing_method: PositionSizingMethod = PositionSizingMethod.KELLY
    max_position_size: float = 0.05
    
    # Risk management
    max_portfolio_risk: float = 0.20
    max_drawdown: float = 0.15
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04


class PortfolioManager:
    """Main portfolio management system."""
    
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.status = PortfolioStatus.ACTIVE
        
        # Initialize components
        self.signal_generator = MultiFactorSignalGenerator()
        self.position_sizer = PositionSizer(
            PositionSizingConfig(method=config.position_sizing_method)
        )
        self.risk_manager = RiskManager(
            RiskManagementConfig(
                max_position_size=config.max_position_size,
                max_portfolio_risk=config.max_portfolio_risk,
                max_drawdown=config.max_drawdown,
                stop_loss_pct=config.stop_loss_pct,
                take_profit_pct=config.take_profit_pct
            )
        )
        
        # Portfolio state
        self.positions = {}
        self.cash = config.initial_capital
        self.total_value = config.initial_capital
        self.portfolio_history = []
        self.trade_history = []
        
        # Performance tracking
        self.daily_returns = []
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        
        # Last update times
        self.last_rebalance = datetime.now()
        self.last_position_review = datetime.now()
        
    def add_ml_model(self, ensemble_model: EnsemblePredictor):
        """Add ML model for signal generation."""
        self.signal_generator.ensemble_model = ensemble_model
        logger.info("ML model added to portfolio manager")
    
    def process_signal(self, 
                      symbol: str,
                      market_data: pd.DataFrame,
                      features: pd.DataFrame,
                      ml_predictions: Optional[np.ndarray] = None,
                      ml_confidence: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Process a trading signal and potentially open a position."""
        try:
            if self.status != PortfolioStatus.ACTIVE:
                return {'action': 'REJECTED', 'reason': f'Portfolio status: {self.status.value}'}
            
            # Check if we can add more positions
            if len(self.positions) >= self.config.max_positions:
                return {'action': 'REJECTED', 'reason': 'Maximum positions reached'}
            
            # Check if position already exists
            if symbol in self.positions:
                return {'action': 'REJECTED', 'reason': 'Position already exists'}
            
            # Generate signals
            signals_df = self.signal_generator.generate_signals(
                market_data, features, ml_predictions, ml_confidence
            )
            
            if signals_df.empty:
                return {'action': 'REJECTED', 'reason': 'No signals generated'}
            
            latest_signal = signals_df.iloc[-1]
            
            # Check signal confidence
            if latest_signal['signal_confidence'] < self.config.min_signal_confidence:
                return {'action': 'REJECTED', 'reason': f'Low confidence: {latest_signal["signal_confidence"]:.3f}'}
            
            # Get current price
            current_price = market_data['close'].iloc[-1]
            
            # Calculate position size
            position_result = self.position_sizer.calculate_position_size(
                signal_confidence=latest_signal['signal_confidence'],
                expected_return=0.05,  # Assume 5% expected return
                volatility=0.20,  # Assume 20% volatility
                current_price=current_price,
                portfolio_value=self.total_value,
                signal_strength=latest_signal['signal_strength']
            )
            
            if position_result['status'] != 'SUCCESS':
                return {'action': 'REJECTED', 'reason': f'Position sizing failed: {position_result["status"]}'}
            
            # Validate position with risk manager
            validation = self.risk_manager.validate_position(
                symbol=symbol,
                position_size=position_result['position_size'],
                entry_price=current_price,
                signal_confidence=latest_signal['signal_confidence'],
                current_portfolio_value=self.total_value
            )
            
            if not validation['approved']:
                return {'action': 'REJECTED', 'reason': f'Risk validation failed: {validation["warnings"]}'}
            
            # Open position
            success = self._open_position(
                symbol=symbol,
                position_size=position_result['position_size'],
                entry_price=current_price,
                signal_confidence=latest_signal['signal_confidence'],
                signal_data=latest_signal
            )
            
            if success:
                return {
                    'action': 'OPENED',
                    'symbol': symbol,
                    'position_size': position_result['position_size'],
                    'shares': position_result['shares'],
                    'entry_price': current_price,
                    'confidence': latest_signal['signal_confidence']
                }
            else:
                return {'action': 'REJECTED', 'reason': 'Failed to open position'}
                
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return {'action': 'ERROR', 'reason': f'Processing error: {e}'}
    
    def _open_position(self, 
                      symbol: str,
                      position_size: float,
                      entry_price: float,
                      signal_confidence: float,
                      signal_data: pd.Series) -> bool:
        """Open a new position."""
        try:
            # Calculate position value and shares
            position_value = position_size * self.total_value
            shares = int(position_value / entry_price)
            actual_position_value = shares * entry_price
            
            # Update cash
            self.cash -= actual_position_value
            
            # Create position
            position = {
                'symbol': symbol,
                'shares': shares,
                'entry_price': entry_price,
                'entry_time': datetime.now(),
                'position_size': position_size,
                'position_value': actual_position_value,
                'signal_confidence': signal_confidence,
                'signal_data': signal_data.to_dict(),
                'status': 'OPEN',
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0
            }
            
            # Add to positions
            self.positions[symbol] = position
            
            # Add to risk manager
            self.risk_manager.add_position(
                symbol=symbol,
                position_size=position_size,
                entry_price=entry_price,
                entry_time=datetime.now(),
                signal_confidence=signal_confidence
            )
            
            # Record trade
            self._record_trade('OPEN', symbol, shares, entry_price, actual_position_value)
            
            logger.info(f"Opened position: {symbol} - {shares} shares at ${entry_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return False
    
    def update_positions(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Update all positions with current prices."""
        try:
            update_results = {}
            positions_to_close = []
            
            for symbol, position in self.positions.items():
                if position['status'] != 'OPEN':
                    continue
                
                if symbol not in current_prices:
                    logger.warning(f"No current price for {symbol}")
                    continue
                
                current_price = current_prices[symbol]
                
                # Update position
                position['current_price'] = current_price
                position['unrealized_pnl'] = (current_price - position['entry_price']) * position['shares']
                position['current_value'] = current_price * position['shares']
                
                # Check risk triggers
                risk_action = self.risk_manager.update_position(symbol, current_price)
                
                if risk_action['action'] == RiskAction.CLOSE:
                    positions_to_close.append((symbol, risk_action['reason']))
                elif risk_action['action'] == RiskAction.EMERGENCY_EXIT:
                    positions_to_close.append((symbol, risk_action['reason']))
                    self.status = PortfolioStatus.EMERGENCY_STOP
                
                update_results[symbol] = {
                    'current_price': current_price,
                    'unrealized_pnl': position['unrealized_pnl'],
                    'current_value': position['current_value'],
                    'risk_action': risk_action
                }
            
            # Close positions that triggered risk management
            for symbol, reason in positions_to_close:
                self.close_position(symbol, reason)
            
            # Update portfolio value
            self._update_portfolio_value()
            
            return update_results
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
            return {}
    
    def close_position(self, symbol: str, reason: str = "Manual close") -> bool:
        """Close a position."""
        try:
            if symbol not in self.positions:
                logger.warning(f"Position {symbol} not found")
                return False
            
            position = self.positions[symbol]
            
            if position['status'] != 'OPEN':
                logger.warning(f"Position {symbol} is not open")
                return False
            
            # Get current price (assume it's available)
            current_price = position.get('current_price', position['entry_price'])
            
            # Calculate realized P&L
            realized_pnl = (current_price - position['entry_price']) * position['shares']
            position_value = current_price * position['shares']
            
            # Update cash
            self.cash += position_value
            
            # Update position
            position['status'] = 'CLOSED'
            position['exit_price'] = current_price
            position['exit_time'] = datetime.now()
            position['realized_pnl'] = realized_pnl
            position['close_reason'] = reason
            
            # Remove from risk manager
            self.risk_manager.close_position(symbol, reason)
            
            # Record trade
            self._record_trade('CLOSE', symbol, -position['shares'], current_price, position_value)
            
            logger.info(f"Closed position: {symbol} - P&L: ${realized_pnl:.2f} - Reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def rebalance_portfolio(self) -> Dict[str, Any]:
        """Rebalance the portfolio based on current conditions."""
        try:
            if self.status != PortfolioStatus.ACTIVE:
                return {'action': 'SKIPPED', 'reason': f'Portfolio status: {self.status.value}'}
            
            # Check if rebalancing is needed
            time_since_rebalance = datetime.now() - self.last_rebalance
            if time_since_rebalance.total_seconds() < self.config.rebalance_frequency * 3600:
                return {'action': 'SKIPPED', 'reason': 'Too soon for rebalancing'}
            
            rebalance_results = {
                'action': 'REBALANCED',
                'timestamp': datetime.now(),
                'positions_reviewed': len(self.positions),
                'positions_closed': 0,
                'positions_adjusted': 0
            }
            
            # Review each position
            positions_to_close = []
            
            for symbol, position in self.positions.items():
                if position['status'] != 'OPEN':
                    continue
                
                # Check if position should be closed based on time or performance
                time_held = datetime.now() - position['entry_time']
                
                # Close positions held too long (e.g., 30 days)
                if time_held.days > 30:
                    positions_to_close.append((symbol, "Held too long"))
                    continue
                
                # Close positions with large losses
                if position['unrealized_pnl'] < -position['position_value'] * 0.1:  # 10% loss
                    positions_to_close.append((symbol, "Large loss"))
                    continue
            
            # Close positions
            for symbol, reason in positions_to_close:
                self.close_position(symbol, reason)
                rebalance_results['positions_closed'] += 1
            
            # Update last rebalance time
            self.last_rebalance = datetime.now()
            
            logger.info(f"Portfolio rebalanced: {rebalance_results['positions_closed']} positions closed")
            return rebalance_results
            
        except Exception as e:
            logger.error(f"Error rebalancing portfolio: {e}")
            return {'action': 'ERROR', 'reason': f'Rebalancing error: {e}'}
    
    def _update_portfolio_value(self):
        """Update total portfolio value."""
        try:
            # Calculate total value from positions
            positions_value = sum(
                pos.get('current_value', pos['position_value']) 
                for pos in self.positions.values() 
                if pos['status'] == 'OPEN'
            )
            
            # Update total value
            self.total_value = self.cash + positions_value
            
            # Calculate daily return
            if self.portfolio_history:
                previous_value = self.portfolio_history[-1]['total_value']
                daily_return = (self.total_value - previous_value) / previous_value
                self.daily_returns.append(daily_return)
            
            # Store portfolio history
            self._store_portfolio_history()
            
        except Exception as e:
            logger.error(f"Error updating portfolio value: {e}")
    
    def _store_portfolio_history(self):
        """Store portfolio state in history."""
        try:
            history_entry = {
                'timestamp': datetime.now(),
                'total_value': self.total_value,
                'cash': self.cash,
                'positions_value': self.total_value - self.cash,
                'num_positions': len([p for p in self.positions.values() if p['status'] == 'OPEN']),
                'unrealized_pnl': sum(p.get('unrealized_pnl', 0) for p in self.positions.values()),
                'status': self.status.value
            }
            
            self.portfolio_history.append(history_entry)
            
            # Keep only recent history
            if len(self.portfolio_history) > 1000:
                self.portfolio_history = self.portfolio_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error storing portfolio history: {e}")
    
    def _record_trade(self, action: str, symbol: str, shares: int, price: float, value: float):
        """Record a trade in history."""
        try:
            trade = {
                'timestamp': datetime.now(),
                'action': action,
                'symbol': symbol,
                'shares': shares,
                'price': price,
                'value': value
            }
            
            self.trade_history.append(trade)
            
            # Keep only recent trades
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        try:
            # Calculate performance metrics
            if len(self.daily_returns) > 1:
                total_return = (self.total_value - self.config.initial_capital) / self.config.initial_capital
                volatility = np.std(self.daily_returns) * np.sqrt(252) if len(self.daily_returns) > 1 else 0
                sharpe_ratio = np.mean(self.daily_returns) / np.std(self.daily_returns) * np.sqrt(252) if np.std(self.daily_returns) > 0 else 0
                
                # Calculate max drawdown
                cumulative_returns = np.cumprod([1 + r for r in self.daily_returns])
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
            else:
                total_return = 0
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
            
            # Get risk metrics
            portfolio_positions = {
                symbol: {
                    'position_size': pos['position_size'],
                    'unrealized_pnl': pos.get('unrealized_pnl', 0)
                }
                for symbol, pos in self.positions.items()
                if pos['status'] == 'OPEN'
            }
            
            risk_metrics = self.risk_manager.check_portfolio_risk(self.total_value, portfolio_positions)
            
            summary = {
                'timestamp': datetime.now(),
                'status': self.status.value,
                'total_value': self.total_value,
                'cash': self.cash,
                'positions_value': self.total_value - self.cash,
                'initial_capital': self.config.initial_capital,
                'total_return': total_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'num_positions': len([p for p in self.positions.values() if p['status'] == 'OPEN']),
                'risk_level': risk_metrics.get('risk_level', 'unknown'),
                'risk_actions': risk_metrics.get('actions', []),
                'unrealized_pnl': sum(p.get('unrealized_pnl', 0) for p in self.positions.values()),
                'realized_pnl': sum(p.get('realized_pnl', 0) for p in self.positions.values())
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def get_position_details(self) -> Dict[str, Any]:
        """Get detailed information about all positions."""
        try:
            position_details = {}
            
            for symbol, position in self.positions.items():
                details = {
                    'symbol': symbol,
                    'shares': position['shares'],
                    'entry_price': position['entry_price'],
                    'current_price': position.get('current_price', position['entry_price']),
                    'entry_time': position['entry_time'],
                    'position_size': position['position_size'],
                    'position_value': position['position_value'],
                    'current_value': position.get('current_value', position['position_value']),
                    'unrealized_pnl': position.get('unrealized_pnl', 0),
                    'realized_pnl': position.get('realized_pnl', 0),
                    'status': position['status'],
                    'signal_confidence': position['signal_confidence']
                }
                
                if position['status'] == 'CLOSED':
                    details.update({
                        'exit_price': position.get('exit_price'),
                        'exit_time': position.get('exit_time'),
                        'close_reason': position.get('close_reason')
                    })
                
                position_details[symbol] = details
            
            return position_details
            
        except Exception as e:
            logger.error(f"Error getting position details: {e}")
            return {}
    
    def export_portfolio_data(self) -> Dict[str, Any]:
        """Export portfolio data for analysis."""
        try:
            export_data = {
                'config': self.config.__dict__,
                'portfolio_history': self.portfolio_history,
                'trade_history': self.trade_history,
                'position_details': self.get_position_details(),
                'daily_returns': self.daily_returns,
                'risk_summary': self.risk_manager.get_risk_summary()
            }
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting portfolio data: {e}")
            return {}


# Utility functions
def create_portfolio_manager(initial_capital: float = 100000.0) -> PortfolioManager:
    """Create a portfolio manager with default configuration."""
    config = PortfolioConfig(initial_capital=initial_capital)
    return PortfolioManager(config)


def create_portfolio_manager_with_config(config: PortfolioConfig) -> PortfolioManager:
    """Create a portfolio manager with custom configuration."""
    return PortfolioManager(config)


if __name__ == "__main__":
    # Example usage
    config = PortfolioConfig(initial_capital=100000.0)
    portfolio = PortfolioManager(config)
    
    # Process a signal
    import yfinance as yf
    
    # Download sample data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="1y")
    
    # Create features
    features = pd.DataFrame({
        'close': df['Close'],
        'volume': df['Volume'],
        'sma_20': df['Close'].rolling(20).mean(),
        'rsi': df['Close'].rolling(14).apply(lambda x: 100 - (100 / (1 + x.pct_change().mean())))
    }).dropna()
    
    # Process signal
    result = portfolio.process_signal("AAPL", df, features)
    print(f"Signal processing result: {result}")
    
    # Update positions
    current_prices = {"AAPL": 155.0}
    update_results = portfolio.update_positions(current_prices)
    print(f"Position updates: {update_results}")
    
    # Get portfolio summary
    summary = portfolio.get_portfolio_summary()
    print(f"Portfolio summary: {summary}")
    
    # Rebalance portfolio
    rebalance_result = portfolio.rebalance_portfolio()
    print(f"Rebalance result: {rebalance_result}")
    
    # Get position details
    positions = portfolio.get_position_details()
    print(f"Position details: {positions}")
