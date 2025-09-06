"""
Technical indicator engine using TA-Lib for calculating various technical indicators.
Supports momentum, volatility, volume, and trend indicators.
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndicatorType(Enum):
    """Enumeration of indicator types."""
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TREND = "trend"
    PATTERN = "pattern"


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    name: str
    indicator_type: IndicatorType
    parameters: Dict[str, Any]
    description: str = ""


class TechnicalIndicatorEngine:
    """Engine for calculating technical indicators using TA-Lib."""
    
    def __init__(self):
        self.indicators_config = self._initialize_indicators()
    
    def _initialize_indicators(self) -> Dict[str, IndicatorConfig]:
        """Initialize indicator configurations."""
        return {
            # Momentum Indicators
            'RSI': IndicatorConfig(
                name='RSI',
                indicator_type=IndicatorType.MOMENTUM,
                parameters={'timeperiod': 14},
                description='Relative Strength Index'
            ),
            'MACD': IndicatorConfig(
                name='MACD',
                indicator_type=IndicatorType.MOMENTUM,
                parameters={'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
                description='Moving Average Convergence Divergence'
            ),
            'ROC': IndicatorConfig(
                name='ROC',
                indicator_type=IndicatorType.MOMENTUM,
                parameters={'timeperiod': 10},
                description='Rate of Change'
            ),
            'STOCH': IndicatorConfig(
                name='STOCH',
                indicator_type=IndicatorType.MOMENTUM,
                parameters={'fastk_period': 5, 'slowk_period': 3, 'slowd_period': 3},
                description='Stochastic Oscillator'
            ),
            'WILLR': IndicatorConfig(
                name='WILLR',
                indicator_type=IndicatorType.MOMENTUM,
                parameters={'timeperiod': 14},
                description='Williams %R'
            ),
            'CCI': IndicatorConfig(
                name='CCI',
                indicator_type=IndicatorType.MOMENTUM,
                parameters={'timeperiod': 14},
                description='Commodity Channel Index'
            ),
            
            # Volatility Indicators
            'BBANDS': IndicatorConfig(
                name='BBANDS',
                indicator_type=IndicatorType.VOLATILITY,
                parameters={'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
                description='Bollinger Bands'
            ),
            'ATR': IndicatorConfig(
                name='ATR',
                indicator_type=IndicatorType.VOLATILITY,
                parameters={'timeperiod': 14},
                description='Average True Range'
            ),
            'NATR': IndicatorConfig(
                name='NATR',
                indicator_type=IndicatorType.VOLATILITY,
                parameters={'timeperiod': 14},
                description='Normalized Average True Range'
            ),
            'TRANGE': IndicatorConfig(
                name='TRANGE',
                indicator_type=IndicatorType.VOLATILITY,
                parameters={},
                description='True Range'
            ),
            
            # Volume Indicators
            'OBV': IndicatorConfig(
                name='OBV',
                indicator_type=IndicatorType.VOLUME,
                parameters={},
                description='On Balance Volume'
            ),
            'AD': IndicatorConfig(
                name='AD',
                indicator_type=IndicatorType.VOLUME,
                parameters={},
                description='Accumulation/Distribution Line'
            ),
            'ADOSC': IndicatorConfig(
                name='ADOSC',
                indicator_type=IndicatorType.VOLUME,
                parameters={'fastperiod': 3, 'slowperiod': 10},
                description='Accumulation/Distribution Oscillator'
            ),
            'MFI': IndicatorConfig(
                name='MFI',
                indicator_type=IndicatorType.VOLUME,
                parameters={'timeperiod': 14},
                description='Money Flow Index'
            ),
            
            # Trend Indicators
            'SMA': IndicatorConfig(
                name='SMA',
                indicator_type=IndicatorType.TREND,
                parameters={'timeperiod': 20},
                description='Simple Moving Average'
            ),
            'EMA': IndicatorConfig(
                name='EMA',
                indicator_type=IndicatorType.TREND,
                parameters={'timeperiod': 20},
                description='Exponential Moving Average'
            ),
            'ADX': IndicatorConfig(
                name='ADX',
                indicator_type=IndicatorType.TREND,
                parameters={'timeperiod': 14},
                description='Average Directional Movement Index'
            ),
            'AROON': IndicatorConfig(
                name='AROON',
                indicator_type=IndicatorType.TREND,
                parameters={'timeperiod': 14},
                description='Aroon Oscillator'
            ),
            'AROONOSC': IndicatorConfig(
                name='AROONOSC',
                indicator_type=IndicatorType.TREND,
                parameters={'timeperiod': 14},
                description='Aroon Oscillator'
            ),
            
            # Pattern Recognition
            'DOJI': IndicatorConfig(
                name='DOJI',
                indicator_type=IndicatorType.PATTERN,
                parameters={},
                description='Doji Pattern'
            ),
            'HAMMER': IndicatorConfig(
                name='HAMMER',
                indicator_type=IndicatorType.PATTERN,
                parameters={},
                description='Hammer Pattern'
            ),
            'ENGULFING': IndicatorConfig(
                name='ENGULFING',
                indicator_type=IndicatorType.PATTERN,
                parameters={},
                description='Engulfing Pattern'
            ),
        }
    
    def calculate_indicator(
        self, 
        df: pd.DataFrame, 
        indicator_name: str,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> pd.Series:
        """Calculate a single technical indicator."""
        
        if indicator_name not in self.indicators_config:
            raise ValueError(f"Unknown indicator: {indicator_name}")
        
        config = self.indicators_config[indicator_name]
        params = custom_params or config.parameters
        
        try:
            # Get OHLCV data
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            open_price = df['open'].values
            volume = df['volume'].values if 'volume' in df.columns else None
            
            # Calculate indicator based on type
            if indicator_name == 'RSI':
                result = talib.RSI(close, **params)
            elif indicator_name == 'MACD':
                macd, signal, hist = talib.MACD(close, **params)
                result = macd  # Return MACD line
            elif indicator_name == 'ROC':
                result = talib.ROC(close, **params)
            elif indicator_name == 'STOCH':
                slowk, slowd = talib.STOCH(high, low, close, **params)
                result = slowk  # Return %K
            elif indicator_name == 'WILLR':
                result = talib.WILLR(high, low, close, **params)
            elif indicator_name == 'CCI':
                result = talib.CCI(high, low, close, **params)
            elif indicator_name == 'BBANDS':
                upper, middle, lower = talib.BBANDS(close, **params)
                result = (close - lower) / (upper - lower)  # Bollinger Band Position
            elif indicator_name == 'ATR':
                result = talib.ATR(high, low, close, **params)
            elif indicator_name == 'NATR':
                result = talib.NATR(high, low, close, **params)
            elif indicator_name == 'TRANGE':
                result = talib.TRANGE(high, low, close)
            elif indicator_name == 'OBV':
                result = talib.OBV(close, volume)
            elif indicator_name == 'AD':
                result = talib.AD(high, low, close, volume)
            elif indicator_name == 'ADOSC':
                result = talib.ADOSC(high, low, close, volume, **params)
            elif indicator_name == 'MFI':
                result = talib.MFI(high, low, close, volume, **params)
            elif indicator_name == 'SMA':
                result = talib.SMA(close, **params)
            elif indicator_name == 'EMA':
                result = talib.EMA(close, **params)
            elif indicator_name == 'ADX':
                result = talib.ADX(high, low, close, **params)
            elif indicator_name == 'AROON':
                aroondown, aroonup = talib.AROON(high, low, **params)
                result = aroonup - aroondown  # Aroon Oscillator
            elif indicator_name == 'AROONOSC':
                result = talib.AROONOSC(high, low, **params)
            elif indicator_name == 'DOJI':
                result = talib.CDLDOJI(open_price, high, low, close)
            elif indicator_name == 'HAMMER':
                result = talib.CDLHAMMER(open_price, high, low, close)
            elif indicator_name == 'ENGULFING':
                result = talib.CDLENGULFING(open_price, high, low, close)
            else:
                raise ValueError(f"Calculation not implemented for {indicator_name}")
            
            return pd.Series(result, index=df.index, name=indicator_name)
            
        except Exception as e:
            logger.error(f"Error calculating {indicator_name}: {e}")
            return pd.Series(np.nan, index=df.index, name=indicator_name)
    
    def calculate_multiple_indicators(
        self, 
        df: pd.DataFrame, 
        indicator_names: List[str],
        custom_params: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """Calculate multiple technical indicators."""
        
        # Start with original data
        results = df.copy()
        custom_params = custom_params or {}
        
        for indicator_name in indicator_names:
            try:
                params = custom_params.get(indicator_name)
                result = self.calculate_indicator(df, indicator_name, params)
                results[indicator_name] = result
                
            except Exception as e:
                logger.error(f"Error calculating {indicator_name}: {e}")
                results[indicator_name] = np.nan
        
        return results
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all available technical indicators."""
        indicator_names = list(self.indicators_config.keys())
        return self.calculate_multiple_indicators(df, indicator_names)
    
    def calculate_indicators_by_type(
        self, 
        df: pd.DataFrame, 
        indicator_type: IndicatorType
    ) -> pd.DataFrame:
        """Calculate indicators of a specific type."""
        
        indicator_names = [
            name for name, config in self.indicators_config.items()
            if config.indicator_type == indicator_type
        ]
        
        return self.calculate_multiple_indicators(df, indicator_names)
    
    def get_indicator_info(self, indicator_name: str) -> Optional[IndicatorConfig]:
        """Get information about a specific indicator."""
        return self.indicators_config.get(indicator_name)
    
    def get_indicators_by_type(self, indicator_type: IndicatorType) -> List[str]:
        """Get list of indicators by type."""
        return [
            name for name, config in self.indicators_config.items()
            if config.indicator_type == indicator_type
        ]
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that DataFrame has required OHLCV columns."""
        required_columns = ['open', 'high', 'low', 'close']
        
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns. Required: {required_columns}")
            return False
        
        if df.empty:
            logger.error("DataFrame is empty")
            return False
        
        if df.isnull().all().any():
            logger.warning("DataFrame contains all-null columns")
        
        return True
    
    def calculate_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate custom indicators not in TA-Lib."""
        
        results = pd.DataFrame(index=df.index)
        
        try:
            # Price-based indicators
            results['PRICE_CHANGE'] = df['close'].pct_change()
            results['PRICE_CHANGE_ABS'] = df['close'].diff()
            results['LOG_RETURN'] = np.log(df['close'] / df['close'].shift(1))
            
            # Volatility indicators
            results['VOLATILITY_20'] = df['close'].rolling(20).std()
            results['VOLATILITY_RATIO'] = (
                df['close'].rolling(5).std() / df['close'].rolling(20).std()
            )
            
            # Volume indicators
            if 'volume' in df.columns:
                results['VOLUME_SMA'] = df['volume'].rolling(20).mean()
                results['VOLUME_RATIO'] = df['volume'] / results['VOLUME_SMA']
                results['PRICE_VOLUME'] = df['close'] * df['volume']
            
            # Moving average crossovers
            results['SMA_5'] = df['close'].rolling(5).mean()
            results['SMA_10'] = df['close'].rolling(10).mean()
            results['SMA_20'] = df['close'].rolling(20).mean()
            results['SMA_50'] = df['close'].rolling(50).mean()
            
            results['MA_CROSS_5_10'] = results['SMA_5'] - results['SMA_10']
            results['MA_CROSS_10_20'] = results['SMA_10'] - results['SMA_20']
            results['MA_CROSS_20_50'] = results['SMA_20'] - results['SMA_50']
            
            # Support and resistance levels
            results['HIGH_20'] = df['high'].rolling(20).max()
            results['LOW_20'] = df['low'].rolling(20).min()
            results['SUPPORT_RESISTANCE'] = (df['close'] - results['LOW_20']) / (results['HIGH_20'] - results['LOW_20'])
            
            # Gap analysis
            results['GAP'] = df['open'] - df['close'].shift(1)
            results['GAP_PCT'] = results['GAP'] / df['close'].shift(1)
            
            logger.info("Custom indicators calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating custom indicators: {e}")
        
        return results
    
    def calculate_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive set of features including TA-Lib and custom indicators."""
        
        if not self.validate_data(df):
            return pd.DataFrame()
        
        # Calculate TA-Lib indicators
        ta_indicators = self.calculate_all_indicators(df)
        
        # Calculate custom indicators
        custom_indicators = self.calculate_custom_indicators(df)
        
        # Combine all indicators
        all_features = pd.concat([ta_indicators, custom_indicators], axis=1)
        
        # Remove columns with all NaN values
        all_features = all_features.dropna(axis=1, how='all')
        
        logger.info(f"Calculated {len(all_features.columns)} features")
        
        return all_features


# Utility functions
def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate momentum-based features."""
    engine = TechnicalIndicatorEngine()
    return engine.calculate_indicators_by_type(df, IndicatorType.MOMENTUM)


def calculate_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volatility-based features."""
    engine = TechnicalIndicatorEngine()
    return engine.calculate_indicators_by_type(df, IndicatorType.VOLATILITY)


def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate volume-based features."""
    engine = TechnicalIndicatorEngine()
    return engine.calculate_indicators_by_type(df, IndicatorType.VOLUME)


def calculate_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trend-based features."""
    engine = TechnicalIndicatorEngine()
    return engine.calculate_indicators_by_type(df, IndicatorType.TREND)


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="1y")
    
    # Initialize engine
    engine = TechnicalIndicatorEngine()
    
    # Calculate comprehensive features
    features = engine.calculate_comprehensive_features(df)
    
    print(f"Calculated {len(features.columns)} features")
    print("Feature names:", list(features.columns))
    
    # Calculate specific indicator types
    momentum_features = calculate_momentum_features(df)
    print(f"Momentum features: {list(momentum_features.columns)}")
