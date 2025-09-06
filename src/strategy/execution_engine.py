"""
Execution engine for managing order execution and trade management.
Handles order routing, execution quality, and transaction cost analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Enumeration of order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Enumeration of order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Enumeration of order statuses."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order data structure."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    status: OrderStatus = OrderStatus.PENDING
    created_time: datetime = None
    filled_time: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: int = 0
    commission: float = 0.0
    slippage: float = 0.0
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now()


@dataclass
class ExecutionConfig:
    """Configuration for execution engine."""
    # Execution settings
    max_order_size: int = 10000
    min_order_size: int = 1
    default_commission: float = 0.005  # $0.005 per share
    max_slippage: float = 0.001  # 0.1% max slippage
    
    # Order management
    order_timeout: int = 300  # 5 minutes
    retry_attempts: int = 3
    retry_delay: int = 5  # seconds
    
    # Market simulation
    simulate_execution: bool = True
    execution_delay: float = 0.1  # seconds
    fill_probability: float = 0.95  # 95% fill probability
    
    # Transaction costs
    bid_ask_spread: float = 0.001  # 0.1% spread
    market_impact_factor: float = 0.0001  # Market impact per share


class ExecutionEngine:
    """Order execution engine with market simulation capabilities."""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.orders = {}
        self.order_queue = queue.Queue()
        self.execution_thread = None
        self.is_running = False
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'total_commission': 0.0,
            'total_slippage': 0.0
        }
        
    def start(self):
        """Start the execution engine."""
        try:
            if self.is_running:
                logger.warning("Execution engine already running")
                return
            
            self.is_running = True
            self.execution_thread = threading.Thread(target=self._execution_loop)
            self.execution_thread.daemon = True
            self.execution_thread.start()
            
            logger.info("Execution engine started")
            
        except Exception as e:
            logger.error(f"Error starting execution engine: {e}")
    
    def stop(self):
        """Stop the execution engine."""
        try:
            self.is_running = False
            if self.execution_thread:
                self.execution_thread.join(timeout=5)
            
            logger.info("Execution engine stopped")
            
        except Exception as e:
            logger.error(f"Error stopping execution engine: {e}")
    
    def submit_order(self, order: Order) -> bool:
        """Submit an order for execution."""
        try:
            # Validate order
            if not self._validate_order(order):
                return False
            
            # Add to orders dictionary
            self.orders[order.order_id] = order
            
            # Add to execution queue
            self.order_queue.put(order)
            
            # Update stats
            self.execution_stats['total_orders'] += 1
            
            logger.info(f"Order submitted: {order.order_id} - {order.side.value} {order.quantity} {order.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        try:
            if order_id not in self.orders:
                logger.warning(f"Order {order_id} not found")
                return False
            
            order = self.orders[order_id]
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                logger.warning(f"Cannot cancel order {order_id} with status {order.status.value}")
                return False
            
            order.status = OrderStatus.CANCELLED
            self.execution_stats['cancelled_orders'] += 1
            
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get the status of an order."""
        try:
            if order_id in self.orders:
                return self.orders[order_id].status
            return None
            
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None
    
    def get_order_details(self, order_id: str) -> Optional[Order]:
        """Get detailed information about an order."""
        try:
            return self.orders.get(order_id)
            
        except Exception as e:
            logger.error(f"Error getting order details: {e}")
            return None
    
    def _validate_order(self, order: Order) -> bool:
        """Validate an order before submission."""
        try:
            # Check order size
            if order.quantity < self.config.min_order_size:
                logger.warning(f"Order quantity {order.quantity} below minimum {self.config.min_order_size}")
                return False
            
            if order.quantity > self.config.max_order_size:
                logger.warning(f"Order quantity {order.quantity} above maximum {self.config.max_order_size}")
                return False
            
            # Check price for limit orders
            if order.order_type == OrderType.LIMIT and order.price is None:
                logger.warning("Limit order must have a price")
                return False
            
            # Check stop price for stop orders
            if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
                logger.warning("Stop order must have a stop price")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating order: {e}")
            return False
    
    def _execution_loop(self):
        """Main execution loop running in separate thread."""
        try:
            while self.is_running:
                try:
                    # Get next order from queue
                    order = self.order_queue.get(timeout=1)
                    
                    # Process order
                    self._process_order(order)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in execution loop: {e}")
                    
        except Exception as e:
            logger.error(f"Execution loop error: {e}")
    
    def _process_order(self, order: Order):
        """Process a single order."""
        try:
            order.status = OrderStatus.SUBMITTED
            
            if self.config.simulate_execution:
                # Simulate execution
                self._simulate_execution(order)
            else:
                # Real execution (placeholder for actual broker integration)
                self._execute_real_order(order)
                
        except Exception as e:
            logger.error(f"Error processing order: {e}")
            order.status = OrderStatus.REJECTED
            self.execution_stats['rejected_orders'] += 1
    
    def _simulate_execution(self, order: Order):
        """Simulate order execution with realistic delays and fills."""
        try:
            # Simulate execution delay
            if self.config.execution_delay > 0:
                time.sleep(self.config.execution_delay)
            
            # Check if order should be filled
            if np.random.random() > self.config.fill_probability:
                order.status = OrderStatus.REJECTED
                self.execution_stats['rejected_orders'] += 1
                logger.warning(f"Order {order.order_id} rejected (simulation)")
                return
            
            # Simulate fill
            self._simulate_fill(order)
            
        except Exception as e:
            logger.error(f"Error simulating execution: {e}")
            order.status = OrderStatus.REJECTED
            self.execution_stats['rejected_orders'] += 1
    
    def _simulate_fill(self, order: Order):
        """Simulate order fill with realistic pricing."""
        try:
            # Get current market price (simplified)
            current_price = self._get_current_price(order.symbol)
            
            # Calculate fill price based on order type
            if order.order_type == OrderType.MARKET:
                fill_price = self._calculate_market_fill_price(order, current_price)
            elif order.order_type == OrderType.LIMIT:
                fill_price = self._calculate_limit_fill_price(order, current_price)
            elif order.order_type == OrderType.STOP:
                fill_price = self._calculate_stop_fill_price(order, current_price)
            else:
                fill_price = current_price
            
            # Calculate slippage
            slippage = abs(fill_price - current_price) / current_price
            slippage = min(slippage, self.config.max_slippage)
            
            # Calculate commission
            commission = order.quantity * self.config.default_commission
            
            # Update order
            order.status = OrderStatus.FILLED
            order.filled_time = datetime.now()
            order.filled_price = fill_price
            order.filled_quantity = order.quantity
            order.commission = commission
            order.slippage = slippage
            
            # Update stats
            self.execution_stats['filled_orders'] += 1
            self.execution_stats['total_commission'] += commission
            self.execution_stats['total_slippage'] += slippage
            
            logger.info(f"Order filled: {order.order_id} - {order.quantity} shares at ${fill_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error simulating fill: {e}")
            order.status = OrderStatus.REJECTED
            self.execution_stats['rejected_orders'] += 1
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current market price for a symbol (simplified)."""
        # This is a simplified implementation
        # In practice, you would get real-time prices from a data provider
        
        # Mock prices for common symbols
        mock_prices = {
            'AAPL': 150.0,
            'MSFT': 300.0,
            'GOOGL': 2500.0,
            'AMZN': 3200.0,
            'META': 200.0,
            'TSLA': 800.0,
            'SPY': 400.0,
            'QQQ': 350.0
        }
        
        return mock_prices.get(symbol, 100.0)
    
    def _calculate_market_fill_price(self, order: Order, current_price: float) -> float:
        """Calculate fill price for market orders."""
        try:
            # Add bid-ask spread
            spread = current_price * self.config.bid_ask_spread
            
            if order.side == OrderSide.BUY:
                # Buy at ask price
                fill_price = current_price + spread / 2
            else:
                # Sell at bid price
                fill_price = current_price - spread / 2
            
            # Add market impact
            market_impact = order.quantity * self.config.market_impact_factor
            if order.side == OrderSide.BUY:
                fill_price += market_impact
            else:
                fill_price -= market_impact
            
            return fill_price
            
        except Exception as e:
            logger.error(f"Error calculating market fill price: {e}")
            return current_price
    
    def _calculate_limit_fill_price(self, order: Order, current_price: float) -> float:
        """Calculate fill price for limit orders."""
        try:
            # Check if limit order can be filled
            if order.side == OrderSide.BUY and order.price >= current_price:
                # Buy limit can be filled
                return min(order.price, current_price)
            elif order.side == OrderSide.SELL and order.price <= current_price:
                # Sell limit can be filled
                return max(order.price, current_price)
            else:
                # Limit order cannot be filled
                raise ValueError("Limit order cannot be filled at current price")
                
        except Exception as e:
            logger.error(f"Error calculating limit fill price: {e}")
            raise
    
    def _calculate_stop_fill_price(self, order: Order, current_price: float) -> float:
        """Calculate fill price for stop orders."""
        try:
            # Check if stop order is triggered
            if order.side == OrderSide.BUY and current_price >= order.stop_price:
                # Buy stop triggered
                return current_price
            elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                # Sell stop triggered
                return current_price
            else:
                # Stop order not triggered
                raise ValueError("Stop order not triggered")
                
        except Exception as e:
            logger.error(f"Error calculating stop fill price: {e}")
            raise
    
    def _execute_real_order(self, order: Order):
        """Execute real order through broker API (placeholder)."""
        try:
            # This would integrate with actual broker APIs
            # For now, just simulate execution
            logger.info(f"Executing real order: {order.order_id}")
            self._simulate_execution(order)
            
        except Exception as e:
            logger.error(f"Error executing real order: {e}")
            order.status = OrderStatus.REJECTED
            self.execution_stats['rejected_orders'] += 1
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution engine statistics."""
        try:
            stats = self.execution_stats.copy()
            
            # Calculate additional metrics
            if stats['total_orders'] > 0:
                stats['fill_rate'] = stats['filled_orders'] / stats['total_orders']
                stats['avg_commission'] = stats['total_commission'] / stats['filled_orders'] if stats['filled_orders'] > 0 else 0
                stats['avg_slippage'] = stats['total_slippage'] / stats['filled_orders'] if stats['filled_orders'] > 0 else 0
            else:
                stats['fill_rate'] = 0
                stats['avg_commission'] = 0
                stats['avg_slippage'] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting execution stats: {e}")
            return {}
    
    def get_order_history(self) -> List[Order]:
        """Get all orders in execution history."""
        try:
            return list(self.orders.values())
            
        except Exception as e:
            logger.error(f"Error getting order history: {e}")
            return []
    
    def get_filled_orders(self) -> List[Order]:
        """Get all filled orders."""
        try:
            return [order for order in self.orders.values() if order.status == OrderStatus.FILLED]
            
        except Exception as e:
            logger.error(f"Error getting filled orders: {e}")
            return []
    
    def calculate_transaction_costs(self, orders: List[Order]) -> Dict[str, float]:
        """Calculate total transaction costs for a list of orders."""
        try:
            total_commission = sum(order.commission for order in orders)
            total_slippage = sum(order.slippage * order.filled_price * order.filled_quantity for order in orders)
            total_costs = total_commission + total_slippage
            
            return {
                'total_commission': total_commission,
                'total_slippage': total_slippage,
                'total_costs': total_costs,
                'avg_commission_per_order': total_commission / len(orders) if orders else 0,
                'avg_slippage_per_order': total_slippage / len(orders) if orders else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating transaction costs: {e}")
            return {}


# Utility functions
def create_execution_engine(config: Optional[ExecutionConfig] = None) -> ExecutionEngine:
    """Create an execution engine with default or custom configuration."""
    if config is None:
        config = ExecutionConfig()
    return ExecutionEngine(config)


def create_order(symbol: str, 
                side: OrderSide, 
                order_type: OrderType, 
                quantity: int,
                price: Optional[float] = None,
                stop_price: Optional[float] = None) -> Order:
    """Create a new order."""
    order_id = f"{symbol}_{side.value}_{order_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return Order(
        order_id=order_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        price=price,
        stop_price=stop_price
    )


if __name__ == "__main__":
    # Example usage
    config = ExecutionConfig(simulate_execution=True)
    engine = ExecutionEngine(config)
    
    # Start engine
    engine.start()
    
    # Create and submit orders
    order1 = create_order("AAPL", OrderSide.BUY, OrderType.MARKET, 100)
    order2 = create_order("MSFT", OrderSide.SELL, OrderType.LIMIT, 50, price=300.0)
    
    # Submit orders
    engine.submit_order(order1)
    engine.submit_order(order2)
    
    # Wait for execution
    time.sleep(2)
    
    # Check order status
    print(f"Order 1 status: {engine.get_order_status(order1.order_id)}")
    print(f"Order 2 status: {engine.get_order_status(order2.order_id)}")
    
    # Get execution stats
    stats = engine.get_execution_stats()
    print(f"Execution stats: {stats}")
    
    # Get filled orders
    filled_orders = engine.get_filled_orders()
    print(f"Filled orders: {len(filled_orders)}")
    
    # Calculate transaction costs
    costs = engine.calculate_transaction_costs(filled_orders)
    print(f"Transaction costs: {costs}")
    
    # Stop engine
    engine.stop()
