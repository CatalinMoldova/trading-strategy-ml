"""
Feature engineering pipeline for creating advanced trading features.
Combines technical indicators with statistical and machine learning features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from scipy import stats
from scipy.signal import find_peaks
import talib

from .indicator_engine import TechnicalIndicatorEngine, IndicatorType
from config.model_config import FeatureEngineeringConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering for trading strategies."""
    
    def __init__(self, config: Optional[FeatureEngineeringConfig] = None):
        self.config = config or FeatureEngineeringConfig()
        self.indicator_engine = TechnicalIndicatorEngine()
        self.scalers = {}
        self.feature_selectors = {}
        
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features."""
        features = pd.DataFrame(index=df.index)
        
        try:
            # Basic price features
            features['PRICE_CHANGE'] = df['close'].pct_change()
            features['PRICE_CHANGE_ABS'] = df['close'].diff()
            features['LOG_RETURN'] = np.log(df['close'] / df['close'].shift(1))
            
            # Price ratios
            features['HIGH_LOW_RATIO'] = df['high'] / df['low']
            features['CLOSE_OPEN_RATIO'] = df['close'] / df['open']
            features['HIGH_CLOSE_RATIO'] = df['high'] / df['close']
            features['LOW_CLOSE_RATIO'] = df['low'] / df['close']
            
            # Price position within daily range
            features['PRICE_POSITION'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            features['PRICE_POSITION_SMA'] = features['PRICE_POSITION'].rolling(20).mean()
            
            # Gap features
            features['GAP'] = df['open'] - df['close'].shift(1)
            features['GAP_PCT'] = features['GAP'] / df['close'].shift(1)
            features['GAP_FILLED'] = np.where(
                (features['GAP'] > 0) & (df['low'] <= df['close'].shift(1)), 1,
                np.where((features['GAP'] < 0) & (df['high'] >= df['close'].shift(1)), 1, 0)
            )
            
            # Intraday volatility
            features['INTRADAY_VOLATILITY'] = (df['high'] - df['low']) / df['close']
            features['INTRADAY_VOLATILITY_SMA'] = features['INTRADAY_VOLATILITY'].rolling(20).mean()
            
            logger.info("Price features created successfully")
            
        except Exception as e:
            logger.error(f"Error creating price features: {e}")
        
        return features
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features."""
        features = pd.DataFrame(index=df.index)
        
        if 'volume' not in df.columns:
            logger.warning("Volume data not available")
            return features
        
        try:
            # Basic volume features
            features['VOLUME_CHANGE'] = df['volume'].pct_change()
            features['VOLUME_SMA'] = df['volume'].rolling(20).mean()
            features['VOLUME_RATIO'] = df['volume'] / features['VOLUME_SMA']
            
            # Volume-price relationship
            features['PRICE_VOLUME'] = df['close'] * df['volume']
            features['VOLUME_PRICE_TREND'] = features['PRICE_VOLUME'].pct_change()
            
            # Volume moving averages
            features['VOLUME_SMA_5'] = df['volume'].rolling(5).mean()
            features['VOLUME_SMA_10'] = df['volume'].rolling(10).mean()
            features['VOLUME_SMA_20'] = df['volume'].rolling(20).mean()
            
            # Volume ratios
            features['VOLUME_RATIO_5_20'] = features['VOLUME_SMA_5'] / features['VOLUME_SMA_20']
            features['VOLUME_RATIO_10_20'] = features['VOLUME_SMA_10'] / features['VOLUME_SMA_20']
            
            # Volume spikes
            features['VOLUME_SPIKE'] = np.where(
                df['volume'] > features['VOLUME_SMA'] * 2, 1, 0
            )
            
            # On Balance Volume features
            obv = talib.OBV(df['close'], df['volume'])
            features['OBV'] = obv
            features['OBV_SMA'] = obv.rolling(20).mean()
            features['OBV_RATIO'] = obv / features['OBV_SMA']
            
            logger.info("Volume features created successfully")
            
        except Exception as e:
            logger.error(f"Error creating volume features: {e}")
        
        return features
    
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum-based features."""
        features = pd.DataFrame(index=df.index)
        
        try:
            # RSI features
            rsi = talib.RSI(df['close'], timeperiod=14)
            features['RSI'] = rsi
            features['RSI_OVERBOUGHT'] = np.where(rsi > 70, 1, 0)
            features['RSI_OVERSOLD'] = np.where(rsi < 30, 1, 0)
            features['RSI_DIVERGENCE'] = self._calculate_divergence(df['close'], rsi)
            
            # MACD features
            macd, signal, hist = talib.MACD(df['close'])
            features['MACD'] = macd
            features['MACD_SIGNAL'] = signal
            features['MACD_HISTOGRAM'] = hist
            features['MACD_CROSSOVER'] = np.where(
                (macd > signal) & (macd.shift(1) <= signal.shift(1)), 1,
                np.where((macd < signal) & (macd.shift(1) >= signal.shift(1)), -1, 0)
            )
            
            # Rate of Change features
            roc_5 = talib.ROC(df['close'], timeperiod=5)
            roc_10 = talib.ROC(df['close'], timeperiod=10)
            roc_20 = talib.ROC(df['close'], timeperiod=20)
            
            features['ROC_5'] = roc_5
            features['ROC_10'] = roc_10
            features['ROC_20'] = roc_20
            features['ROC_MOMENTUM'] = roc_5 - roc_20
            
            # Stochastic features
            slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
            features['STOCH_K'] = slowk
            features['STOCH_D'] = slowd
            features['STOCH_OVERBOUGHT'] = np.where(slowk > 80, 1, 0)
            features['STOCH_OVERSOLD'] = np.where(slowk < 20, 1, 0)
            
            # Williams %R
            willr = talib.WILLR(df['high'], df['low'], df['close'])
            features['WILLR'] = willr
            features['WILLR_OVERBOUGHT'] = np.where(willr > -20, 1, 0)
            features['WILLR_OVERSOLD'] = np.where(willr < -80, 1, 0)
            
            logger.info("Momentum features created successfully")
            
        except Exception as e:
            logger.error(f"Error creating momentum features: {e}")
        
        return features
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features."""
        features = pd.DataFrame(index=df.index)
        
        try:
            # Bollinger Bands features
            upper, middle, lower = talib.BBANDS(df['close'])
            features['BB_UPPER'] = upper
            features['BB_MIDDLE'] = middle
            features['BB_LOWER'] = lower
            features['BB_POSITION'] = (df['close'] - lower) / (upper - lower)
            features['BB_WIDTH'] = (upper - lower) / middle
            features['BB_SQUEEZE'] = np.where(features['BB_WIDTH'] < features['BB_WIDTH'].rolling(20).mean() * 0.8, 1, 0)
            
            # ATR features
            atr = talib.ATR(df['high'], df['low'], df['close'])
            features['ATR'] = atr
            features['ATR_PCT'] = atr / df['close']
            features['ATR_SMA'] = atr.rolling(20).mean()
            features['ATR_RATIO'] = atr / features['ATR_SMA']
            
            # Volatility ratios
            features['VOLATILITY_5'] = df['close'].rolling(5).std()
            features['VOLATILITY_20'] = df['close'].rolling(20).std()
            features['VOLATILITY_RATIO'] = features['VOLATILITY_5'] / features['VOLATILITY_20']
            
            # Historical volatility
            features['HV_10'] = df['close'].pct_change().rolling(10).std() * np.sqrt(252)
            features['HV_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
            features['HV_30'] = df['close'].pct_change().rolling(30).std() * np.sqrt(252)
            
            # Volatility clustering
            returns = df['close'].pct_change()
            features['VOLATILITY_CLUSTERING'] = returns.rolling(5).std() / returns.rolling(20).std()
            
            logger.info("Volatility features created successfully")
            
        except Exception as e:
            logger.error(f"Error creating volatility features: {e}")
        
        return features
    
    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create trend-based features."""
        features = pd.DataFrame(index=df.index)
        
        try:
            # Moving averages
            sma_5 = talib.SMA(df['close'], timeperiod=5)
            sma_10 = talib.SMA(df['close'], timeperiod=10)
            sma_20 = talib.SMA(df['close'], timeperiod=20)
            sma_50 = talib.SMA(df['close'], timeperiod=50)
            sma_200 = talib.SMA(df['close'], timeperiod=200)
            
            features['SMA_5'] = sma_5
            features['SMA_10'] = sma_10
            features['SMA_20'] = sma_20
            features['SMA_50'] = sma_50
            features['SMA_200'] = sma_200
            
            # Price relative to moving averages
            features['PRICE_SMA_5_RATIO'] = df['close'] / sma_5
            features['PRICE_SMA_20_RATIO'] = df['close'] / sma_20
            features['PRICE_SMA_50_RATIO'] = df['close'] / sma_50
            
            # Moving average crossovers
            features['MA_CROSS_5_10'] = sma_5 - sma_10
            features['MA_CROSS_10_20'] = sma_10 - sma_20
            features['MA_CROSS_20_50'] = sma_20 - sma_50
            features['MA_CROSS_50_200'] = sma_50 - sma_200
            
            # Trend strength
            features['TREND_STRENGTH'] = (
                (sma_5 > sma_10).astype(int) +
                (sma_10 > sma_20).astype(int) +
                (sma_20 > sma_50).astype(int) +
                (sma_50 > sma_200).astype(int)
            )
            
            # ADX features
            adx = talib.ADX(df['high'], df['low'], df['close'])
            features['ADX'] = adx
            features['ADX_TREND'] = np.where(adx > 25, 1, np.where(adx < 20, -1, 0))
            
            # Aroon features
            aroondown, aroonup = talib.AROON(df['high'], df['low'])
            features['AROON_UP'] = aroonup
            features['AROON_DOWN'] = aroondown
            features['AROON_OSC'] = aroonup - aroondown
            
            logger.info("Trend features created successfully")
            
        except Exception as e:
            logger.error(f"Error creating trend features: {e}")
        
        return features
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        features = pd.DataFrame(index=df.index)
        
        try:
            # Rolling statistics
            for window in [5, 10, 20, 50]:
                features[f'MEAN_{window}'] = df['close'].rolling(window).mean()
                features[f'STD_{window}'] = df['close'].rolling(window).std()
                features[f'MIN_{window}'] = df['close'].rolling(window).min()
                features[f'MAX_{window}'] = df['close'].rolling(window).max()
                features[f'MEDIAN_{window}'] = df['close'].rolling(window).median()
                features[f'SKEW_{window}'] = df['close'].rolling(window).skew()
                features[f'KURT_{window}'] = df['close'].rolling(window).kurt()
            
            # Percentile features
            for window in [20, 50]:
                features[f'PERCENTILE_25_{window}'] = df['close'].rolling(window).quantile(0.25)
                features[f'PERCENTILE_75_{window}'] = df['close'].rolling(window).quantile(0.75)
                features[f'PERCENTILE_RANK_{window}'] = df['close'].rolling(window).rank(pct=True)
            
            # Correlation features
            returns = df['close'].pct_change()
            features['AUTOCORR_1'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=1))
            features['AUTOCORR_5'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=5))
            
            # Hurst exponent (trend persistence)
            features['HURST_EXPONENT'] = self._calculate_hurst_exponent(df['close'])
            
            logger.info("Statistical features created successfully")
            
        except Exception as e:
            logger.error(f"Error creating statistical features: {e}")
        
        return features
    
    def create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create pattern recognition features."""
        features = pd.DataFrame(index=df.index)
        
        try:
            # Candlestick patterns
            features['DOJI'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            features['HAMMER'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            features['ENGULFING'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            features['MORNING_STAR'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
            features['EVENING_STAR'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
            
            # Support and resistance levels
            features['SUPPORT_LEVEL'] = df['low'].rolling(20).min()
            features['RESISTANCE_LEVEL'] = df['high'].rolling(20).max()
            features['SUPPORT_DISTANCE'] = (df['close'] - features['SUPPORT_LEVEL']) / df['close']
            features['RESISTANCE_DISTANCE'] = (features['RESISTANCE_LEVEL'] - df['close']) / df['close']
            
            # Breakout patterns
            features['BREAKOUT_UP'] = np.where(
                df['close'] > features['RESISTANCE_LEVEL'].shift(1), 1, 0
            )
            features['BREAKOUT_DOWN'] = np.where(
                df['close'] < features['SUPPORT_LEVEL'].shift(1), 1, 0
            )
            
            logger.info("Pattern features created successfully")
            
        except Exception as e:
            logger.error(f"Error creating pattern features: {e}")
        
        return features
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        features = pd.DataFrame(index=df.index)
        
        try:
            # Extract time components
            features['HOUR'] = df.index.hour
            features['DAY_OF_WEEK'] = df.index.dayofweek
            features['DAY_OF_MONTH'] = df.index.day
            features['MONTH'] = df.index.month
            features['QUARTER'] = df.index.quarter
            features['YEAR'] = df.index.year
            
            # Cyclical encoding
            features['HOUR_SIN'] = np.sin(2 * np.pi * features['HOUR'] / 24)
            features['HOUR_COS'] = np.cos(2 * np.pi * features['HOUR'] / 24)
            features['DAY_SIN'] = np.sin(2 * np.pi * features['DAY_OF_WEEK'] / 7)
            features['DAY_COS'] = np.cos(2 * np.pi * features['DAY_OF_WEEK'] / 7)
            features['MONTH_SIN'] = np.sin(2 * np.pi * features['MONTH'] / 12)
            features['MONTH_COS'] = np.cos(2 * np.pi * features['MONTH'] / 12)
            
            # Market session features
            features['IS_MARKET_OPEN'] = np.where(
                (features['HOUR'] >= 9) & (features['HOUR'] < 16), 1, 0
            )
            features['IS_EARLY_MORNING'] = np.where(
                (features['HOUR'] >= 9) & (features['HOUR'] < 11), 1, 0
            )
            features['IS_LATE_AFTERNOON'] = np.where(
                (features['HOUR'] >= 14) & (features['HOUR'] < 16), 1, 0
            )
            
            logger.info("Time features created successfully")
            
        except Exception as e:
            logger.error(f"Error creating time features: {e}")
        
        return features
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'close', lags: List[int] = None) -> pd.DataFrame:
        """Create lagged features."""
        if lags is None:
            lags = [1, 2, 3, 5, 10, 20]
        
        features = pd.DataFrame(index=df.index)
        
        try:
            for lag in lags:
                features[f'{target_col}_LAG_{lag}'] = df[target_col].shift(lag)
                features[f'{target_col}_LAG_{lag}_PCT'] = df[target_col].pct_change(lag)
            
            # Rolling window features
            for window in [5, 10, 20]:
                features[f'{target_col}_ROLLING_MEAN_{window}'] = df[target_col].rolling(window).mean()
                features[f'{target_col}_ROLLING_STD_{window}'] = df[target_col].rolling(window).std()
                features[f'{target_col}_ROLLING_MIN_{window}'] = df[target_col].rolling(window).min()
                features[f'{target_col}_ROLLING_MAX_{window}'] = df[target_col].rolling(window).max()
            
            logger.info(f"Lag features created for {len(lags)} lags")
            
        except Exception as e:
            logger.error(f"Error creating lag features: {e}")
        
        return features
    
    def _calculate_divergence(self, price: pd.Series, indicator: pd.Series, window: int = 20) -> pd.Series:
        """Calculate divergence between price and indicator."""
        price_trend = price.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        indicator_trend = indicator.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        divergence = np.where(
            (price_trend > 0) & (indicator_trend < 0), -1,  # Bearish divergence
            np.where((price_trend < 0) & (indicator_trend > 0), 1, 0)  # Bullish divergence
        )
        
        return pd.Series(divergence, index=price.index)
    
    def _calculate_hurst_exponent(self, series: pd.Series, window: int = 50) -> pd.Series:
        """Calculate Hurst exponent for trend persistence."""
        def hurst(x):
            if len(x) < 10:
                return np.nan
            
            lags = range(2, min(20, len(x) // 2))
            tau = [np.sqrt(np.std(np.subtract(x[lag:], x[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        
        return series.rolling(window).apply(hurst)
    
    def normalize_features(self, features: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """Normalize features using specified method."""
        if method == 'zscore':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Fit scaler on non-null data
        non_null_data = features.dropna()
        if non_null_data.empty:
            return features
        
        scaler.fit(non_null_data)
        
        # Transform all data
        normalized_features = pd.DataFrame(
            scaler.transform(features.fillna(features.mean())),
            index=features.index,
            columns=features.columns
        )
        
        # Store scaler for later use
        self.scalers[method] = scaler
        
        logger.info(f"Features normalized using {method}")
        return normalized_features
    
    def select_features(
        self, 
        features: pd.DataFrame, 
        target: pd.Series, 
        method: str = 'mutual_info',
        k: int = 50
    ) -> pd.DataFrame:
        """Select most relevant features."""
        
        # Remove features with too many NaN values
        features_clean = features.dropna(axis=1, thresh=len(features) * 0.5)
        
        if features_clean.empty:
            logger.warning("No features remaining after cleaning")
            return features
        
        # Align features and target
        common_index = features_clean.index.intersection(target.index)
        features_aligned = features_clean.loc[common_index]
        target_aligned = target.loc[common_index]
        
        # Remove any remaining NaN values
        mask = ~(features_aligned.isnull().any(axis=1) | target_aligned.isnull())
        features_final = features_aligned[mask]
        target_final = target_aligned[mask]
        
        if len(features_final) < 10:
            logger.warning("Insufficient data for feature selection")
            return features
        
        # Select features
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=min(k, len(features_final.columns)))
        elif method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=min(k, len(features_final.columns)))
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        try:
            selector.fit(features_final, target_final)
            selected_features = features_final.iloc[:, selector.get_support()]
            
            # Store selector for later use
            self.feature_selectors[method] = selector
            
            logger.info(f"Selected {len(selected_features.columns)} features using {method}")
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return features
    
    def create_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set."""
        logger.info("Creating comprehensive feature set")
        
        # Start with original data
        result_df = df.copy()
        all_features = []
        
        # Create different feature types
        feature_types = [
            ('price', self.create_price_features),
            ('volume', self.create_volume_features),
            ('momentum', self.create_momentum_features),
            ('volatility', self.create_volatility_features),
            ('trend', self.create_trend_features),
            ('statistical', self.create_statistical_features),
            ('pattern', self.create_pattern_features),
            ('time', self.create_time_features),
            ('lag', lambda x: self.create_lag_features(x))
        ]
        
        for feature_type, feature_func in feature_types:
            try:
                features = feature_func(df)
                if not features.empty:
                    all_features.append(features)
                    logger.info(f"Created {len(features.columns)} {feature_type} features")
            except Exception as e:
                logger.error(f"Error creating {feature_type} features: {e}")
        
        if not all_features:
            logger.error("No features created")
            return result_df
        
        # Combine all features
        combined_features = pd.concat(all_features, axis=1)
        
        # Add features to original data
        result_df = pd.concat([result_df, combined_features], axis=1)
        
        # Remove highly correlated features
        result_df = self._remove_correlated_features(result_df)
        
        logger.info(f"Total features created: {len(result_df.columns) - len(df.columns)}")
        return result_df
    
    def _remove_correlated_features(self, features: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features."""
        # Essential columns to preserve
        essential_cols = ['open', 'high', 'low', 'close', 'volume']
        
        corr_matrix = features.corr().abs()
        
        # Find pairs of highly correlated features
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop (excluding essential columns)
        to_drop = []
        for column in upper_tri.columns:
            if column not in essential_cols and any(upper_tri[column] > threshold):
                to_drop.append(column)
        
        # Drop highly correlated features
        features_cleaned = features.drop(columns=to_drop)
        
        logger.info(f"Removed {len(to_drop)} highly correlated features")
        return features_cleaned


# Utility functions
def create_features_for_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Create comprehensive features for a single symbol."""
    engineer = FeatureEngineer()
    return engineer.create_comprehensive_features(df)


def create_features_batch(symbol_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Create features for multiple symbols."""
    engineer = FeatureEngineer()
    features = {}
    
    for symbol, df in symbol_data.items():
        try:
            features[symbol] = engineer.create_comprehensive_features(df)
            logger.info(f"Created features for {symbol}")
        except Exception as e:
            logger.error(f"Error creating features for {symbol}: {e}")
    
    return features


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download sample data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="1y")
    
    # Create features
    engineer = FeatureEngineer()
    features = engineer.create_comprehensive_features(df)
    
    print(f"Created {len(features.columns)} features")
    print("Feature categories:", list(features.columns))
    
    # Normalize features
    normalized_features = engineer.normalize_features(features)
    print("Features normalized successfully")
