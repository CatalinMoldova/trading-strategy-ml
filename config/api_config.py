"""
API configuration for external data sources.
Supports Alpha Vantage, Yahoo Finance, and other financial APIs.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class APIConfig:
    """Configuration for external API integrations."""
    
    def __init__(self):
        # Alpha Vantage API
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.alpha_vantage_base_url = 'https://www.alphavantage.co/query'
        
        # Yahoo Finance (no API key required)
        self.yahoo_finance_base_url = 'https://query1.finance.yahoo.com/v8/finance/chart'
        
        # IEX Cloud API
        self.iex_cloud_key = os.getenv('IEX_CLOUD_API_KEY')
        self.iex_cloud_base_url = 'https://cloud.iexapis.com/stable'
        
        # Polygon.io API
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        self.polygon_base_url = 'https://api.polygon.io'
        
        # Rate limiting settings
        self.rate_limit_per_minute = 60
        self.rate_limit_per_hour = 1000
        
    def get_alpha_vantage_params(self) -> Dict[str, str]:
        """Get Alpha Vantage API parameters."""
        if not self.alpha_vantage_key:
            raise ValueError("Alpha Vantage API key not found in environment variables")
        
        return {
            'apikey': self.alpha_vantage_key,
            'outputsize': 'full',
            'datatype': 'json'
        }
    
    def get_iex_cloud_params(self) -> Dict[str, str]:
        """Get IEX Cloud API parameters."""
        if not self.iex_cloud_key:
            raise ValueError("IEX Cloud API key not found in environment variables")
        
        return {
            'token': self.iex_cloud_key
        }
    
    def get_polygon_params(self) -> Dict[str, str]:
        """Get Polygon.io API parameters."""
        if not self.polygon_key:
            raise ValueError("Polygon.io API key not found in environment variables")
        
        return {
            'apikey': self.polygon_key
        }


class DataSourceConfig:
    """Configuration for different data sources and their capabilities."""
    
    DATA_SOURCES = {
        'alpha_vantage': {
            'name': 'Alpha Vantage',
            'supports': ['daily', 'intraday', 'fundamentals'],
            'rate_limit': 5,  # calls per minute for free tier
            'historical_limit': '20 years',
            'real_time': False
        },
        'yahoo_finance': {
            'name': 'Yahoo Finance',
            'supports': ['daily', 'intraday', 'fundamentals'],
            'rate_limit': 2000,  # calls per hour
            'historical_limit': 'No limit',
            'real_time': True
        },
        'iex_cloud': {
            'name': 'IEX Cloud',
            'supports': ['daily', 'intraday', 'fundamentals', 'news'],
            'rate_limit': 100000,  # calls per month
            'historical_limit': '15 years',
            'real_time': True
        },
        'polygon': {
            'name': 'Polygon.io',
            'supports': ['daily', 'intraday', 'fundamentals', 'news', 'options'],
            'rate_limit': 1000,  # calls per minute
            'historical_limit': '20+ years',
            'real_time': True
        }
    }
    
    @classmethod
    def get_source_info(cls, source: str) -> Dict[str, Any]:
        """Get information about a specific data source."""
        return cls.DATA_SOURCES.get(source, {})
    
    @classmethod
    def get_available_sources(cls) -> list:
        """Get list of available data sources."""
        return list(cls.DATA_SOURCES.keys())


class SymbolConfig:
    """Configuration for trading symbols and asset classes."""
    
    # Major stock indices
    INDICES = {
        'SPY': 'S&P 500 ETF',
        'QQQ': 'NASDAQ 100 ETF',
        'IWM': 'Russell 2000 ETF',
        'VTI': 'Total Stock Market ETF',
        'DIA': 'Dow Jones ETF'
    }
    
    # Sector ETFs
    SECTORS = {
        'XLK': 'Technology',
        'XLF': 'Financials',
        'XLE': 'Energy',
        'XLV': 'Healthcare',
        'XLI': 'Industrials',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLU': 'Utilities',
        'XLB': 'Materials',
        'XLRE': 'Real Estate',
        'XLC': 'Communication Services'
    }
    
    # Individual stocks (FAANG + others)
    INDIVIDUAL_STOCKS = {
        'AAPL': 'Apple Inc.',
        'MSFT': 'Microsoft Corporation',
        'GOOGL': 'Alphabet Inc.',
        'AMZN': 'Amazon.com Inc.',
        'META': 'Meta Platforms Inc.',
        'TSLA': 'Tesla Inc.',
        'NVDA': 'NVIDIA Corporation',
        'BRK.B': 'Berkshire Hathaway Inc.',
        'JPM': 'JPMorgan Chase & Co.',
        'JNJ': 'Johnson & Johnson'
    }
    
    # Cryptocurrencies
    CRYPTO = {
        'BTC-USD': 'Bitcoin',
        'ETH-USD': 'Ethereum',
        'ADA-USD': 'Cardano',
        'SOL-USD': 'Solana',
        'DOT-USD': 'Polkadot'
    }
    
    # Forex pairs
    FOREX = {
        'EURUSD=X': 'Euro/US Dollar',
        'GBPUSD=X': 'British Pound/US Dollar',
        'USDJPY=X': 'US Dollar/Japanese Yen',
        'USDCHF=X': 'US Dollar/Swiss Franc',
        'AUDUSD=X': 'Australian Dollar/US Dollar'
    }
    
    @classmethod
    def get_all_symbols(cls) -> Dict[str, str]:
        """Get all available symbols."""
        all_symbols = {}
        all_symbols.update(cls.INDICES)
        all_symbols.update(cls.SECTORS)
        all_symbols.update(cls.INDIVIDUAL_STOCKS)
        all_symbols.update(cls.CRYPTO)
        all_symbols.update(cls.FOREX)
        return all_symbols
    
    @classmethod
    def get_symbols_by_category(cls, category: str) -> Dict[str, str]:
        """Get symbols by category."""
        category_map = {
            'indices': cls.INDICES,
            'sectors': cls.SECTORS,
            'stocks': cls.INDIVIDUAL_STOCKS,
            'crypto': cls.CRYPTO,
            'forex': cls.FOREX
        }
        return category_map.get(category, {})


class TimeframeConfig:
    """Configuration for different timeframes."""
    
    TIMEFRAMES = {
        '1m': {'name': '1 Minute', 'seconds': 60},
        '5m': {'name': '5 Minutes', 'seconds': 300},
        '15m': {'name': '15 Minutes', 'seconds': 900},
        '30m': {'name': '30 Minutes', 'seconds': 1800},
        '1h': {'name': '1 Hour', 'seconds': 3600},
        '4h': {'name': '4 Hours', 'seconds': 14400},
        '1d': {'name': '1 Day', 'seconds': 86400},
        '1wk': {'name': '1 Week', 'seconds': 604800},
        '1mo': {'name': '1 Month', 'seconds': 2592000}
    }
    
    @classmethod
    def get_timeframe_info(cls, timeframe: str) -> Dict[str, Any]:
        """Get information about a specific timeframe."""
        return cls.TIMEFRAMES.get(timeframe, {})
    
    @classmethod
    def get_available_timeframes(cls) -> list:
        """Get list of available timeframes."""
        return list(cls.TIMEFRAMES.keys())


# Environment variable template
ENV_TEMPLATE = """
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_strategy
DB_USER=postgres
DB_PASSWORD=your_password_here

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
IEX_CLOUD_API_KEY=your_iex_cloud_key_here
POLYGON_API_KEY=your_polygon_key_here

# Trading Configuration
DEFAULT_SYMBOLS=AAPL,MSFT,GOOGL,AMZN,META,TSLA,SPY,QQQ
DEFAULT_TIMEFRAME=1d
MAX_POSITION_SIZE=0.05
MAX_PORTFOLIO_RISK=0.20
STOP_LOSS_PCT=0.02
TAKE_PROFIT_PCT=0.04
"""


def create_env_template():
    """Create .env template file."""
    with open('.env.template', 'w') as f:
        f.write(ENV_TEMPLATE)
    print("Created .env.template file. Copy to .env and fill in your API keys.")


if __name__ == "__main__":
    create_env_template()
