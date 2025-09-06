"""
Market data collector for fetching real-time and historical data from various APIs.
Supports Alpha Vantage, Yahoo Finance, and other financial data sources.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import time
import logging
from dataclasses import dataclass

from config.api_config import APIConfig, DataSourceConfig, SymbolConfig, TimeframeConfig
from config.database_config import DatabaseConfig
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Data class for market data points."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None


class RateLimiter:
    """Rate limiter to handle API rate limits."""
    
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        
        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.calls.append(now)


class MarketDataCollector:
    """Main class for collecting market data from various sources."""
    
    def __init__(self):
        self.api_config = APIConfig()
        self.db_config = DatabaseConfig()
        from config.database_config import RedisConfig
        self.redis_client = redis.Redis(**RedisConfig().connection_params)
        self.rate_limiter = RateLimiter()
        
        # Initialize API clients
        self.yf_client = yf
        if self.api_config.alpha_vantage_key:
            self.av_client = TimeSeries(key=self.api_config.alpha_vantage_key)
        else:
            self.av_client = None
            logger.warning("Alpha Vantage API key not found. Some features may be limited.")
    
    def get_yahoo_finance_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Synchronous wrapper for Yahoo Finance data collection."""
        try:
            ticker = self.yf_client.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            # Convert column names to lowercase for consistency
            data.columns = [col.lower() for col in data.columns]
            return data
        except Exception as e:
            logger.error(f"Error collecting Yahoo Finance data for {symbol}: {e}")
            return pd.DataFrame()

    async def collect_yahoo_data(
        self, 
        symbols: List[str], 
        timeframe: str = '1d',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """Collect data from Yahoo Finance."""
        try:
            await self.rate_limiter.wait_if_needed()
            
            if start_date is None:
                start_date = datetime.now() - timedelta(days=365)
            if end_date is None:
                end_date = datetime.now()
            
            data = {}
            for symbol in symbols:
                try:
                    ticker = self.yf_client.Ticker(symbol)
                    df = ticker.history(
                        start=start_date,
                        end=end_date,
                        interval=timeframe
                    )
                    
                    if not df.empty:
                        # Standardize column names
                        df.columns = df.columns.str.lower()
                        df = df.rename(columns={'adj close': 'adjusted_close'})
                        df['symbol'] = symbol
                        df['timestamp'] = df.index
                        data[symbol] = df
                        logger.info(f"Collected {len(df)} records for {symbol}")
                    else:
                        logger.warning(f"No data found for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error collecting data for {symbol}: {e}")
                    continue
            
            return data
            
        except Exception as e:
            logger.error(f"Error in Yahoo Finance data collection: {e}")
            return {}
    
    async def collect_alpha_vantage_data(
        self,
        symbols: List[str],
        timeframe: str = 'daily'
    ) -> Dict[str, pd.DataFrame]:
        """Collect data from Alpha Vantage."""
        if not self.av_client:
            logger.error("Alpha Vantage client not initialized")
            return {}
        
        data = {}
        for symbol in symbols:
            try:
                await self.rate_limiter.wait_if_needed()
                
                if timeframe == 'daily':
                    raw_data, _ = self.av_client.get_daily_adjusted(symbol, outputsize='full')
                elif timeframe == 'intraday':
                    raw_data, _ = self.av_client.get_intraday(symbol, interval='5min', outputsize='full')
                else:
                    logger.warning(f"Unsupported timeframe: {timeframe}")
                    continue
                
                if raw_data:
                    df = pd.DataFrame.from_dict(raw_data, orient='index')
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                    
                    # Standardize column names
                    df.columns = df.columns.str.lower()
                    df = df.rename(columns={
                        '1. open': 'open',
                        '2. high': 'high',
                        '3. low': 'low',
                        '4. close': 'close',
                        '5. adjusted close': 'adjusted_close',
                        '6. volume': 'volume'
                    })
                    
                    df['symbol'] = symbol
                    df['timestamp'] = df.index
                    data[symbol] = df
                    logger.info(f"Collected {len(df)} records for {symbol} from Alpha Vantage")
                
            except Exception as e:
                logger.error(f"Error collecting Alpha Vantage data for {symbol}: {e}")
                continue
        
        return data
    
    def store_market_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Store market data in PostgreSQL database."""
        try:
            engine = self.db_config.create_engine()
            
            for symbol, df in data.items():
                if df.empty:
                    continue
                
                # Prepare data for database
                df_to_store = df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close']].copy()
                df_to_store = df_to_store.reset_index(drop=True)
                
                # Store in database
                df_to_store.to_sql(
                    'market_data',
                    engine,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                
                logger.info(f"Stored {len(df_to_store)} records for {symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
            return False
    
    def cache_market_data(self, data: Dict[str, pd.DataFrame], ttl: int = 3600):
        """Cache market data in Redis."""
        try:
            for symbol, df in data.items():
                if df.empty:
                    continue
                
                # Convert DataFrame to JSON for caching
                df_json = df.to_json(orient='records', date_format='iso')
                
                # Cache with TTL
                cache_key = f"market_data:{symbol}:{datetime.now().strftime('%Y%m%d')}"
                self.redis_client.setex(cache_key, ttl, df_json)
                
            logger.info(f"Cached data for {len(data)} symbols")
            
        except Exception as e:
            logger.error(f"Error caching market data: {e}")
    
    def get_cached_data(self, symbol: str, date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Retrieve cached market data from Redis."""
        try:
            if date is None:
                date = datetime.now().strftime('%Y%m%d')
            
            cache_key = f"market_data:{symbol}:{date}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                df = pd.read_json(cached_data, orient='records')
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cached data for {symbol}: {e}")
            return None
    
    async def collect_real_time_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Collect real-time market data."""
        real_time_data = {}
        
        for symbol in symbols:
            try:
                await self.rate_limiter.wait_if_needed()
                
                ticker = self.yf_client.Ticker(symbol)
                info = ticker.info
                
                # Extract relevant real-time data
                real_time_data[symbol] = {
                    'current_price': info.get('currentPrice', info.get('regularMarketPrice')),
                    'previous_close': info.get('previousClose'),
                    'day_high': info.get('dayHigh'),
                    'day_low': info.get('dayLow'),
                    'volume': info.get('volume'),
                    'market_cap': info.get('marketCap'),
                    'timestamp': datetime.now()
                }
                
            except Exception as e:
                logger.error(f"Error collecting real-time data for {symbol}: {e}")
                continue
        
        return real_time_data
    
    async def collect_historical_data(
        self,
        symbols: List[str],
        timeframe: str = '1d',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: str = 'yahoo'
    ) -> Dict[str, pd.DataFrame]:
        """Collect historical data from specified source."""
        
        if source == 'yahoo':
            return await self.collect_yahoo_data(symbols, timeframe, start_date, end_date)
        elif source == 'alpha_vantage':
            return await self.collect_alpha_vantage_data(symbols, timeframe)
        else:
            logger.error(f"Unsupported data source: {source}")
            return {}
    
    def get_available_symbols(self, category: str = 'all') -> List[str]:
        """Get available symbols by category."""
        symbol_config = SymbolConfig()
        
        if category == 'all':
            return list(symbol_config.get_all_symbols().keys())
        else:
            return list(symbol_config.get_symbols_by_category(category).keys())
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists and is tradeable."""
        try:
            ticker = self.yf_client.Ticker(symbol)
            info = ticker.info
            
            # Check if symbol has valid data
            return info.get('regularMarketPrice') is not None
            
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {e}")
            return False
    
    async def collect_data_batch(
        self,
        symbols: List[str],
        timeframe: str = '1d',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: str = 'yahoo',
        store_data: bool = True,
        cache_data: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Collect data for multiple symbols in batch."""
        
        logger.info(f"Starting batch data collection for {len(symbols)} symbols")
        
        # Collect data
        data = await self.collect_historical_data(
            symbols, timeframe, start_date, end_date, source
        )
        
        # Store data if requested
        if store_data and data:
            self.store_market_data(data)
        
        # Cache data if requested
        if cache_data and data:
            self.cache_market_data(data)
        
        logger.info(f"Batch collection completed. Collected data for {len(data)} symbols")
        return data


# Utility functions
async def collect_spy_data(days_back: int = 365) -> pd.DataFrame:
    """Quick function to collect SPY data."""
    collector = MarketDataCollector()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    data = await collector.collect_historical_data(['SPY'], '1d', start_date, end_date)
    return data.get('SPY', pd.DataFrame())


async def collect_multiple_timeframes(symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
    """Collect data for multiple timeframes for a single symbol."""
    collector = MarketDataCollector()
    data = {}
    
    for tf in timeframes:
        tf_data = await collector.collect_historical_data([symbol], tf)
        if tf_data:
            data[tf] = tf_data[symbol]
    
    return data


if __name__ == "__main__":
    # Example usage
    async def main():
        collector = MarketDataCollector()
        
        # Collect data for major indices
        symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT']
        data = await collector.collect_data_batch(symbols)
        
        print(f"Collected data for {len(data)} symbols")
        for symbol, df in data.items():
            print(f"{symbol}: {len(df)} records from {df.index.min()} to {df.index.max()}")
    
    asyncio.run(main())
