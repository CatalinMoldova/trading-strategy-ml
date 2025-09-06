"""
Signal generator for creating trading signals based on ML predictions and technical analysis.
Implements multi-factor scoring and confidence-based signal generation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
from enum import Enum
import talib

from ..ml_models.ensemble_predictor import EnsemblePredictor
from ..data_pipeline.indicator_engine import TechnicalIndicatorEngine
from config.model_config import EnsembleConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Enumeration of signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStrength(Enum):
    """Enumeration of signal strengths."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@pd.api.extensions.register_dataframe_accessor("signal")
class SignalAccessor:
    """Pandas accessor for signal-related operations."""
    
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
    
    def add_signal_metadata(self, signal_type: str, confidence: float, strength: str):
        """Add signal metadata to DataFrame."""
        self._obj[f'{signal_type}_signal'] = 1
        self._obj[f'{signal_type}_confidence'] = confidence
        self._obj[f'{signal_type}_strength'] = strength
        return self._obj


class MultiFactorSignalGenerator:
    """Multi-factor signal generator combining ML predictions with technical analysis."""
    
    def __init__(self, 
                 ensemble_model: Optional[EnsemblePredictor] = None,
                 config: Optional[EnsembleConfig] = None):
        self.ensemble_model = ensemble_model
        self.config = config or EnsembleConfig()
        self.indicator_engine = TechnicalIndicatorEngine()
        self.signal_history = []
        
    def generate_signals(self, 
                        data: pd.DataFrame, 
                        features: pd.DataFrame,
                        ml_predictions: Optional[np.ndarray] = None,
                        ml_confidence: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Generate trading signals based on multiple factors."""
        try:
            signals_df = data.copy()
            
            # Initialize signal columns
            signals_df['ml_signal'] = 0
            signals_df['technical_signal'] = 0
            signals_df['momentum_signal'] = 0
            signals_df['volatility_signal'] = 0
            signals_df['volume_signal'] = 0
            signals_df['trend_signal'] = 0
            signals_df['composite_signal'] = 0
            signals_df['signal_confidence'] = 0.0
            signals_df['signal_strength'] = 'HOLD'
            
            # Generate ML-based signals
            if ml_predictions is not None:
                ml_signals = self._generate_ml_signals(ml_predictions, ml_confidence)
                signals_df['ml_signal'] = ml_signals['signal']
                signals_df['ml_confidence'] = ml_signals['confidence']
            
            # Generate technical signals
            technical_signals = self._generate_technical_signals(data)
            signals_df['technical_signal'] = technical_signals['signal']
            
            # Generate momentum signals
            momentum_signals = self._generate_momentum_signals(data)
            signals_df['momentum_signal'] = momentum_signals['signal']
            
            # Generate volatility signals
            volatility_signals = self._generate_volatility_signals(data)
            signals_df['volatility_signal'] = volatility_signals['signal']
            
            # Generate volume signals
            volume_signals = self._generate_volume_signals(data)
            signals_df['volume_signal'] = volume_signals['signal']
            
            # Generate trend signals
            trend_signals = self._generate_trend_signals(data)
            signals_df['trend_signal'] = trend_signals['signal']
            
            # Generate composite signal
            composite_signals = self._generate_composite_signal(signals_df)
            signals_df['composite_signal'] = composite_signals['signal']
            signals_df['signal_confidence'] = composite_signals['confidence']
            signals_df['signal_strength'] = composite_signals['strength']
            
            # Store signal history
            self._store_signal_history(signals_df)
            
            logger.info(f"Generated signals for {len(signals_df)} data points")
            return signals_df
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return data
    
    def _generate_ml_signals(self, 
                            predictions: np.ndarray, 
                            confidence: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Generate signals based on ML predictions."""
        try:
            signals = np.zeros_like(predictions)
            
            # Convert predictions to signals
            # Assuming predictions are returns or price changes
            signals[predictions > 0.01] = 1  # Buy signal for positive predictions > 1%
            signals[predictions < -0.01] = -1  # Sell signal for negative predictions < -1%
            
            # Apply confidence threshold
            if confidence is not None:
                low_confidence_mask = confidence < self.config.min_confidence
                signals[low_confidence_mask] = 0  # No signal for low confidence
            
            return {
                'signal': signals,
                'confidence': confidence if confidence is not None else np.ones_like(predictions)
            }
            
        except Exception as e:
            logger.error(f"Error generating ML signals: {e}")
            return {'signal': np.zeros_like(predictions), 'confidence': np.zeros_like(predictions)}
    
    def _generate_technical_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate signals based on technical indicators."""
        try:
            signals = np.zeros(len(data))
            
            # RSI signals
            rsi = talib.RSI(data['close'].values)
            signals[rsi < 30] += 1  # Oversold - buy signal
            signals[rsi > 70] -= 1  # Overbought - sell signal
            
            # MACD signals
            macd, signal, hist = talib.MACD(data['close'].values)
            signals[macd > signal] += 1  # MACD above signal - buy
            signals[macd < signal] -= 1  # MACD below signal - sell
            
            # Bollinger Bands signals
            upper, middle, lower = talib.BBANDS(data['close'].values)
            signals[data['close'].values < lower] += 1  # Price below lower band - buy
            signals[data['close'].values > upper] -= 1  # Price above upper band - sell
            
            # Normalize signals
            signals = np.clip(signals, -1, 1)
            
            return {'signal': signals}
            
        except Exception as e:
            logger.error(f"Error generating technical signals: {e}")
            return {'signal': np.zeros(len(data))}
    
    def _generate_momentum_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate signals based on momentum indicators."""
        try:
            signals = np.zeros(len(data))
            
            # Rate of Change
            roc = talib.ROC(data['close'].values, timeperiod=10)
            signals[roc > 5] += 1  # Strong positive momentum
            signals[roc < -5] -= 1  # Strong negative momentum
            
            # Stochastic Oscillator
            slowk, slowd = talib.STOCH(data['high'].values, data['low'].values, data['close'].values)
            signals[(slowk < 20) & (slowd < 20)] += 1  # Oversold
            signals[(slowk > 80) & (slowd > 80)] -= 1  # Overbought
            
            # Williams %R
            willr = talib.WILLR(data['high'].values, data['low'].values, data['close'].values)
            signals[willr < -80] += 1  # Oversold
            signals[willr > -20] -= 1  # Overbought
            
            # Normalize signals
            signals = np.clip(signals, -1, 1)
            
            return {'signal': signals}
            
        except Exception as e:
            logger.error(f"Error generating momentum signals: {e}")
            return {'signal': np.zeros(len(data))}
    
    def _generate_volatility_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate signals based on volatility indicators."""
        try:
            signals = np.zeros(len(data))
            
            # Average True Range
            atr = talib.ATR(data['high'].values, data['low'].values, data['close'].values)
            atr_sma = pd.Series(atr).rolling(20).mean()
            
            # High volatility - reduce position size
            signals[atr > atr_sma * 1.5] -= 0.5
            # Low volatility - increase position size
            signals[atr < atr_sma * 0.5] += 0.5
            
            # Bollinger Band Width
            upper, middle, lower = talib.BBANDS(data['close'].values)
            bb_width = (upper - lower) / middle
            bb_width_sma = pd.Series(bb_width).rolling(20).mean()
            
            # Volatility expansion - breakout potential
            signals[bb_width > bb_width_sma * 1.2] += 0.5
            # Volatility contraction - consolidation
            signals[bb_width < bb_width_sma * 0.8] -= 0.5
            
            # Normalize signals
            signals = np.clip(signals, -1, 1)
            
            return {'signal': signals}
            
        except Exception as e:
            logger.error(f"Error generating volatility signals: {e}")
            return {'signal': np.zeros(len(data))}
    
    def _generate_volume_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate signals based on volume indicators."""
        try:
            signals = np.zeros(len(data))
            
            if 'volume' not in data.columns:
                return {'signal': signals}
            
            # Volume moving average
            volume_sma = data['volume'].rolling(20).mean()
            volume_ratio = data['volume'] / volume_sma
            
            # High volume with price increase - bullish
            price_change = data['close'].pct_change()
            signals[(volume_ratio > 1.5) & (price_change > 0)] += 1
            # High volume with price decrease - bearish
            signals[(volume_ratio > 1.5) & (price_change < 0)] -= 1
            
            # On Balance Volume
            obv = talib.OBV(data['close'].values, data['volume'].values)
            obv_sma = pd.Series(obv).rolling(20).mean()
            signals[obv > obv_sma] += 0.5
            signals[obv < obv_sma] -= 0.5
            
            # Money Flow Index
            mfi = talib.MFI(data['high'].values, data['low'].values, data['close'].values, data['volume'].values)
            signals[mfi < 20] += 1  # Oversold
            signals[mfi > 80] -= 1  # Overbought
            
            # Normalize signals
            signals = np.clip(signals, -1, 1)
            
            return {'signal': signals}
            
        except Exception as e:
            logger.error(f"Error generating volume signals: {e}")
            return {'signal': np.zeros(len(data))}
    
    def _generate_trend_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate signals based on trend indicators."""
        try:
            signals = np.zeros(len(data))
            
            # Moving Average Crossovers
            sma_5 = talib.SMA(data['close'].values, timeperiod=5)
            sma_20 = talib.SMA(data['close'].values, timeperiod=20)
            sma_50 = talib.SMA(data['close'].values, timeperiod=50)
            
            # Short-term trend
            signals[sma_5 > sma_20] += 0.5
            signals[sma_5 < sma_20] -= 0.5
            
            # Medium-term trend
            signals[sma_20 > sma_50] += 0.5
            signals[sma_20 < sma_50] -= 0.5
            
            # ADX trend strength
            adx = talib.ADX(data['high'].values, data['low'].values, data['close'].values)
            signals[adx > 25] += 0.5  # Strong trend
            signals[adx < 20] -= 0.5  # Weak trend
            
            # Aroon Oscillator
            aroon_osc = talib.AROONOSC(data['high'].values, data['low'].values)
            signals[aroon_osc > 50] += 0.5  # Uptrend
            signals[aroon_osc < -50] -= 0.5  # Downtrend
            
            # Normalize signals
            signals = np.clip(signals, -1, 1)
            
            return {'signal': signals}
            
        except Exception as e:
            logger.error(f"Error generating trend signals: {e}")
            return {'signal': np.zeros(len(data))}
    
    def _generate_composite_signal(self, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate composite signal from all individual signals."""
        try:
            # Define weights for different signal types
            weights = {
                'ml_signal': 0.4,  # ML predictions get highest weight
                'technical_signal': 0.2,
                'momentum_signal': 0.15,
                'volatility_signal': 0.1,
                'volume_signal': 0.1,
                'trend_signal': 0.05
            }
            
            # Calculate weighted composite signal
            composite_signal = np.zeros(len(signals_df))
            total_weight = 0
            
            for signal_type, weight in weights.items():
                if signal_type in signals_df.columns:
                    composite_signal += signals_df[signal_type] * weight
                    total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                composite_signal = composite_signal / total_weight
            
            # Calculate confidence based on signal agreement
            signal_columns = [col for col in signals_df.columns if col.endswith('_signal')]
            signal_values = signals_df[signal_columns].values
            
            # Calculate agreement (how many signals agree on direction)
            positive_signals = np.sum(signal_values > 0, axis=1)
            negative_signals = np.sum(signal_values < 0, axis=1)
            total_signals = len(signal_columns)
            
            agreement = np.maximum(positive_signals, negative_signals) / total_signals
            confidence = agreement * 0.8 + 0.2  # Scale to 0.2-1.0 range
            
            # Determine signal strength
            strength = np.where(
                np.abs(composite_signal) > 0.7, 'VERY_STRONG',
                np.where(np.abs(composite_signal) > 0.5, 'STRONG',
                np.where(np.abs(composite_signal) > 0.3, 'MODERATE', 'WEAK'))
            )
            
            return {
                'signal': composite_signal,
                'confidence': confidence,
                'strength': strength
            }
            
        except Exception as e:
            logger.error(f"Error generating composite signal: {e}")
            return {'signal': np.zeros(len(signals_df)), 'confidence': np.zeros(len(signals_df)), 'strength': ['WEAK'] * len(signals_df)}
    
    def _store_signal_history(self, signals_df: pd.DataFrame):
        """Store signal history for analysis."""
        try:
            latest_signals = signals_df.iloc[-1:].copy()
            latest_signals['timestamp'] = datetime.now()
            self.signal_history.append(latest_signals)
            
            # Keep only recent history (last 1000 signals)
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
                
        except Exception as e:
            logger.error(f"Error storing signal history: {e}")
    
    def get_signal_summary(self, signals_df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of generated signals."""
        try:
            summary = {
                'total_signals': len(signals_df),
                'buy_signals': len(signals_df[signals_df['composite_signal'] > 0.1]),
                'sell_signals': len(signals_df[signals_df['composite_signal'] < -0.1]),
                'hold_signals': len(signals_df[np.abs(signals_df['composite_signal']) <= 0.1]),
                'avg_confidence': signals_df['signal_confidence'].mean(),
                'signal_distribution': signals_df['signal_strength'].value_counts().to_dict()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting signal summary: {e}")
            return {}
    
    def filter_signals_by_confidence(self, 
                                   signals_df: pd.DataFrame, 
                                   min_confidence: float = 0.6) -> pd.DataFrame:
        """Filter signals by minimum confidence threshold."""
        try:
            filtered_df = signals_df[signals_df['signal_confidence'] >= min_confidence].copy()
            logger.info(f"Filtered signals: {len(signals_df)} -> {len(filtered_df)} (confidence >= {min_confidence})")
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error filtering signals: {e}")
            return signals_df
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get signal generation statistics."""
        try:
            if not self.signal_history:
                return {'message': 'No signal history available'}
            
            history_df = pd.concat(self.signal_history, ignore_index=True)
            
            stats = {
                'total_signals_generated': len(history_df),
                'signal_types': history_df['signal_strength'].value_counts().to_dict(),
                'avg_confidence': history_df['signal_confidence'].mean(),
                'confidence_std': history_df['signal_confidence'].std(),
                'recent_signals': history_df.tail(10).to_dict('records')
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting signal statistics: {e}")
            return {}


class RealTimeSignalGenerator(MultiFactorSignalGenerator):
    """Real-time signal generator for live trading."""
    
    def __init__(self, 
                 ensemble_model: Optional[EnsemblePredictor] = None,
                 config: Optional[EnsembleConfig] = None):
        super().__init__(ensemble_model, config)
        self.last_signal_time = None
        self.signal_cooldown = timedelta(minutes=5)  # Minimum time between signals
    
    def generate_real_time_signal(self, 
                                 current_data: pd.DataFrame,
                                 features: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate a real-time trading signal."""
        try:
            current_time = datetime.now()
            
            # Check cooldown period
            if (self.last_signal_time and 
                current_time - self.last_signal_time < self.signal_cooldown):
                return None
            
            # Generate signals for current data
            signals_df = self.generate_signals(current_data, features)
            
            if signals_df.empty:
                return None
            
            latest_signal = signals_df.iloc[-1]
            
            # Only return signal if confidence is high enough
            if latest_signal['signal_confidence'] < self.config.min_confidence:
                return None
            
            # Create signal dictionary
            signal = {
                'timestamp': current_time,
                'signal_type': 'BUY' if latest_signal['composite_signal'] > 0.1 else 'SELL' if latest_signal['composite_signal'] < -0.1 else 'HOLD',
                'signal_strength': latest_signal['signal_strength'],
                'confidence': latest_signal['signal_confidence'],
                'composite_score': latest_signal['composite_signal'],
                'ml_signal': latest_signal.get('ml_signal', 0),
                'technical_signal': latest_signal.get('technical_signal', 0),
                'momentum_signal': latest_signal.get('momentum_signal', 0),
                'volatility_signal': latest_signal.get('volatility_signal', 0),
                'volume_signal': latest_signal.get('volume_signal', 0),
                'trend_signal': latest_signal.get('trend_signal', 0)
            }
            
            # Update last signal time
            self.last_signal_time = current_time
            
            logger.info(f"Generated real-time signal: {signal['signal_type']} with confidence {signal['confidence']:.3f}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating real-time signal: {e}")
            return None


# Utility functions
def generate_trading_signals(data: pd.DataFrame, 
                           features: pd.DataFrame,
                           ensemble_model: Optional[EnsemblePredictor] = None) -> pd.DataFrame:
    """Quick function to generate trading signals."""
    generator = MultiFactorSignalGenerator(ensemble_model)
    return generator.generate_signals(data, features)


def create_real_time_generator(ensemble_model: Optional[EnsemblePredictor] = None) -> RealTimeSignalGenerator:
    """Create a real-time signal generator."""
    return RealTimeSignalGenerator(ensemble_model)


if __name__ == "__main__":
    # Example usage
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
    
    # Generate signals
    generator = MultiFactorSignalGenerator()
    signals_df = generator.generate_signals(df, features)
    
    print("Signals generated successfully!")
    print(f"Signal summary: {generator.get_signal_summary(signals_df)}")
    
    # Show recent signals
    recent_signals = signals_df.tail(10)[['composite_signal', 'signal_confidence', 'signal_strength']]
    print("\nRecent signals:")
    print(recent_signals)
