"""
Database configuration for the trading strategy system.
Supports PostgreSQL with TimescaleDB for time series data.
"""

import os
from typing import Dict, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()


class DatabaseConfig:
    """Database configuration and connection management."""
    
    def __init__(self):
        self.host = os.getenv('DB_HOST', 'localhost')
        self.port = os.getenv('DB_PORT', '5432')
        self.database = os.getenv('DB_NAME', 'trading_strategy')
        self.username = os.getenv('DB_USER', 'postgres')
        self.password = os.getenv('DB_PASSWORD', 'password')
        
    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    def create_engine(self) -> Any:
        """Create SQLAlchemy engine with optimized settings."""
        return create_engine(
            self.connection_string,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
    
    def get_session(self):
        """Create database session."""
        engine = self.create_engine()
        Session = sessionmaker(bind=engine)
        return Session()


class RedisConfig:
    """Redis configuration for caching and real-time data."""
    
    def __init__(self):
        self.host = os.getenv('REDIS_HOST', 'localhost')
        self.port = int(os.getenv('REDIS_PORT', '6379'))
        self.password = os.getenv('REDIS_PASSWORD', None)
        self.db = int(os.getenv('REDIS_DB', '0'))
        
    @property
    def connection_params(self) -> Dict[str, Any]:
        """Get Redis connection parameters."""
        params = {
            'host': self.host,
            'port': self.port,
            'db': self.db,
            'decode_responses': True
        }
        if self.password:
            params['password'] = self.password
        return params


# Database table schemas
MARKET_DATA_SCHEMA = """
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(12, 4),
    high DECIMAL(12, 4),
    low DECIMAL(12, 4),
    close DECIMAL(12, 4),
    volume BIGINT,
    adjusted_close DECIMAL(12, 4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp 
ON market_data (symbol, timestamp DESC);
"""

FEATURES_SCHEMA = """
CREATE TABLE IF NOT EXISTS features (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value DECIMAL(20, 8),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timestamp, feature_name)
);

CREATE INDEX IF NOT EXISTS idx_features_symbol_timestamp 
ON features (symbol, timestamp DESC);
"""

SIGNALS_SCHEMA = """
CREATE TABLE IF NOT EXISTS trading_signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    signal_type VARCHAR(20) NOT NULL,
    signal_value DECIMAL(10, 4),
    confidence DECIMAL(5, 4),
    model_name VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp 
ON trading_signals (symbol, timestamp DESC);
"""

PERFORMANCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(50) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(20, 8),
    period_start TIMESTAMPTZ,
    period_end TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_performance_strategy_timestamp 
ON performance_metrics (strategy_name, timestamp DESC);
"""


def initialize_database():
    """Initialize database with required tables and TimescaleDB extensions."""
    db_config = DatabaseConfig()
    engine = db_config.create_engine()
    
    with engine.connect() as conn:
        # Enable TimescaleDB extension
        conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
        
        # Create tables
        conn.execute(MARKET_DATA_SCHEMA)
        conn.execute(FEATURES_SCHEMA)
        conn.execute(SIGNALS_SCHEMA)
        conn.execute(PERFORMANCE_SCHEMA)
        
        # Convert to hypertables for time series optimization
        try:
            conn.execute("SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);")
            conn.execute("SELECT create_hypertable('features', 'timestamp', if_not_exists => TRUE);")
            conn.execute("SELECT create_hypertable('trading_signals', 'timestamp', if_not_exists => TRUE);")
            conn.execute("SELECT create_hypertable('performance_metrics', 'timestamp', if_not_exists => TRUE);")
        except Exception as e:
            print(f"TimescaleDB hypertable creation warning: {e}")
        
        conn.commit()
    
    print("Database initialized successfully!")


if __name__ == "__main__":
    initialize_database()
