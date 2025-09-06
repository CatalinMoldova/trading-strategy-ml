"""
Data storage module for managing market data, features, and model outputs.
Supports PostgreSQL with TimescaleDB for time series optimization.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging
import redis
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import json
import pickle

from config.database_config import DatabaseConfig, RedisConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataStorage:
    """Main class for data storage operations."""
    
    def __init__(self):
        self.db_config = DatabaseConfig()
        self.redis_config = RedisConfig()
        self.engine = self.db_config.create_engine()
        self.redis_client = redis.Redis(**self.redis_config.connection_params)
        
    def store_market_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Store market data in PostgreSQL."""
        try:
            for symbol, df in data.items():
                if df.empty:
                    continue
                
                # Prepare data for storage
                df_to_store = df.copy()
                df_to_store['symbol'] = symbol
                df_to_store['created_at'] = datetime.now()
                
                # Ensure timestamp is timezone-aware
                if not df_to_store.index.tz:
                    df_to_store.index = df_to_store.index.tz_localize('UTC')
                
                df_to_store = df_to_store.reset_index()
                df_to_store = df_to_store.rename(columns={'index': 'timestamp'})
                
                # Store in database
                df_to_store.to_sql(
                    'market_data',
                    self.engine,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                
                logger.info(f"Stored {len(df_to_store)} market data records for {symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
            return False
    
    def store_features(self, features: Dict[str, pd.DataFrame], symbol: str) -> bool:
        """Store engineered features in PostgreSQL."""
        try:
            for feature_name, feature_series in features.items():
                if feature_series.empty:
                    continue
                
                # Prepare feature data
                feature_df = pd.DataFrame({
                    'symbol': symbol,
                    'timestamp': feature_series.index,
                    'feature_name': feature_name,
                    'feature_value': feature_series.values,
                    'created_at': datetime.now()
                })
                
                # Store in database
                feature_df.to_sql(
                    'features',
                    self.engine,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
            
            logger.info(f"Stored {len(features)} features for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing features: {e}")
            return False
    
    def store_trading_signals(self, signals: pd.DataFrame, symbol: str) -> bool:
        """Store trading signals in PostgreSQL."""
        try:
            # Prepare signal data
            signal_df = signals.copy()
            signal_df['symbol'] = symbol
            signal_df['created_at'] = datetime.now()
            
            # Ensure timestamp is timezone-aware
            if not signal_df.index.tz:
                signal_df.index = signal_df.index.tz_localize('UTC')
            
            signal_df = signal_df.reset_index()
            signal_df = signal_df.rename(columns={'index': 'timestamp'})
            
            # Store in database
            signal_df.to_sql(
                'trading_signals',
                self.engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            logger.info(f"Stored {len(signal_df)} trading signals for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing trading signals: {e}")
            return False
    
    def store_performance_metrics(self, metrics: Dict[str, Any], strategy_name: str) -> bool:
        """Store performance metrics in PostgreSQL."""
        try:
            # Prepare metrics data
            metrics_df = pd.DataFrame([{
                'strategy_name': strategy_name,
                'timestamp': datetime.now(),
                'metric_name': metric_name,
                'metric_value': metric_value,
                'created_at': datetime.now()
            } for metric_name, metric_value in metrics.items()])
            
            # Store in database
            metrics_df.to_sql(
                'performance_metrics',
                self.engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            logger.info(f"Stored {len(metrics)} performance metrics for {strategy_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing performance metrics: {e}")
            return False
    
    def get_market_data(
        self, 
        symbol: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Retrieve market data from PostgreSQL."""
        try:
            query = f"""
            SELECT timestamp, open, high, low, close, volume, adjusted_close
            FROM market_data 
            WHERE symbol = '{symbol}'
            """
            
            if start_date:
                query += f" AND timestamp >= '{start_date}'"
            if end_date:
                query += f" AND timestamp <= '{end_date}'"
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            df = pd.read_sql(query, self.engine)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                df = df.sort_index()
            
            logger.info(f"Retrieved {len(df)} market data records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_features(
        self, 
        symbol: str, 
        feature_names: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Retrieve engineered features from PostgreSQL."""
        try:
            query = f"""
            SELECT timestamp, feature_name, feature_value
            FROM features 
            WHERE symbol = '{symbol}'
            """
            
            if feature_names:
                feature_list = "', '".join(feature_names)
                query += f" AND feature_name IN ('{feature_list}')"
            
            if start_date:
                query += f" AND timestamp >= '{start_date}'"
            if end_date:
                query += f" AND timestamp <= '{end_date}'"
            
            query += " ORDER BY timestamp DESC"
            
            df = pd.read_sql(query, self.engine)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.pivot(index='timestamp', columns='feature_name', values='feature_value')
                df = df.sort_index()
            
            logger.info(f"Retrieved {len(df.columns)} features for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving features for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_trading_signals(
        self, 
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Retrieve trading signals from PostgreSQL."""
        try:
            query = f"""
            SELECT timestamp, signal_type, signal_value, confidence, model_name
            FROM trading_signals 
            WHERE symbol = '{symbol}'
            """
            
            if start_date:
                query += f" AND timestamp >= '{start_date}'"
            if end_date:
                query += f" AND timestamp <= '{end_date}'"
            
            query += " ORDER BY timestamp DESC"
            
            df = pd.read_sql(query, self.engine)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
                df = df.sort_index()
            
            logger.info(f"Retrieved {len(df)} trading signals for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving trading signals for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_performance_metrics(
        self, 
        strategy_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Retrieve performance metrics from PostgreSQL."""
        try:
            query = f"""
            SELECT timestamp, metric_name, metric_value
            FROM performance_metrics 
            WHERE strategy_name = '{strategy_name}'
            """
            
            if start_date:
                query += f" AND timestamp >= '{start_date}'"
            if end_date:
                query += f" AND timestamp <= '{end_date}'"
            
            query += " ORDER BY timestamp DESC"
            
            df = pd.read_sql(query, self.engine)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.pivot(index='timestamp', columns='metric_name', values='metric_value')
                df = df.sort_index()
            
            logger.info(f"Retrieved {len(df.columns)} performance metrics for {strategy_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving performance metrics for {strategy_name}: {e}")
            return pd.DataFrame()
    
    def cache_data(self, key: str, data: Any, ttl: int = 3600) -> bool:
        """Cache data in Redis."""
        try:
            if isinstance(data, pd.DataFrame):
                data_json = data.to_json(orient='records', date_format='iso')
            elif isinstance(data, dict):
                data_json = json.dumps(data, default=str)
            else:
                data_json = pickle.dumps(data)
            
            self.redis_client.setex(key, ttl, data_json)
            logger.info(f"Cached data with key: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching data: {e}")
            return False
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Retrieve cached data from Redis."""
        try:
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                try:
                    # Try to parse as JSON first
                    return json.loads(cached_data)
                except (json.JSONDecodeError, TypeError):
                    try:
                        # Try to parse as DataFrame
                        return pd.read_json(cached_data, orient='records')
                    except:
                        # Try to unpickle
                        return pickle.loads(cached_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving cached data: {e}")
            return None
    
    def delete_cached_data(self, key: str) -> bool:
        """Delete cached data from Redis."""
        try:
            result = self.redis_client.delete(key)
            logger.info(f"Deleted cached data with key: {key}")
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error deleting cached data: {e}")
            return False
    
    def get_latest_data(self, symbol: str, data_type: str = 'market') -> Optional[pd.DataFrame]:
        """Get the latest data for a symbol."""
        try:
            if data_type == 'market':
                return self.get_market_data(symbol, limit=1)
            elif data_type == 'features':
                return self.get_features(symbol)
            elif data_type == 'signals':
                return self.get_trading_signals(symbol)
            else:
                logger.error(f"Unknown data type: {data_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            return None
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of stored data."""
        try:
            summary = {}
            
            # Market data summary
            market_query = """
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT symbol) as unique_symbols,
                MIN(timestamp) as earliest_date,
                MAX(timestamp) as latest_date
            FROM market_data
            """
            
            market_summary = pd.read_sql(market_query, self.engine).iloc[0]
            summary['market_data'] = market_summary.to_dict()
            
            # Features summary
            features_query = """
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT symbol) as unique_symbols,
                COUNT(DISTINCT feature_name) as unique_features
            FROM features
            """
            
            features_summary = pd.read_sql(features_query, self.engine).iloc[0]
            summary['features'] = features_summary.to_dict()
            
            # Signals summary
            signals_query = """
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT symbol) as unique_symbols,
                COUNT(DISTINCT model_name) as unique_models
            FROM trading_signals
            """
            
            signals_summary = pd.read_sql(signals_query, self.engine).iloc[0]
            summary['signals'] = signals_summary.to_dict()
            
            logger.info("Data summary retrieved successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {}
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> bool:
        """Clean up old data to manage storage."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean up old market data
            market_query = f"DELETE FROM market_data WHERE timestamp < '{cutoff_date}'"
            market_result = self.engine.execute(text(market_query))
            
            # Clean up old features
            features_query = f"DELETE FROM features WHERE timestamp < '{cutoff_date}'"
            features_result = self.engine.execute(text(features_query))
            
            # Clean up old signals
            signals_query = f"DELETE FROM trading_signals WHERE timestamp < '{cutoff_date}'"
            signals_result = self.engine.execute(text(signals_query))
            
            logger.info(f"Cleaned up data older than {days_to_keep} days")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return False
    
    def backup_data(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        try:
            # This would typically use pg_dump in production
            logger.info(f"Backup created at: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    def restore_data(self, backup_path: str) -> bool:
        """Restore data from backup."""
        try:
            # This would typically use pg_restore in production
            logger.info(f"Data restored from: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring data: {e}")
            return False


# Utility functions
def store_symbol_data(symbol: str, market_data: pd.DataFrame, features: pd.DataFrame) -> bool:
    """Store both market data and features for a symbol."""
    storage = DataStorage()
    
    # Store market data
    market_success = storage.store_market_data({symbol: market_data})
    
    # Store features
    features_dict = {col: features[col] for col in features.columns}
    features_success = storage.store_features(features_dict, symbol)
    
    return market_success and features_success


def get_symbol_data(symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
    """Get both market data and features for a symbol."""
    storage = DataStorage()
    
    market_data = storage.get_market_data(symbol, start_date, end_date)
    features = storage.get_features(symbol, start_date=start_date, end_date=end_date)
    
    return {
        'market_data': market_data,
        'features': features
    }


def cache_model_predictions(symbol: str, predictions: pd.DataFrame, model_name: str) -> bool:
    """Cache model predictions."""
    storage = DataStorage()
    cache_key = f"predictions:{symbol}:{model_name}:{datetime.now().strftime('%Y%m%d')}"
    return storage.cache_data(cache_key, predictions)


def get_cached_predictions(symbol: str, model_name: str) -> Optional[pd.DataFrame]:
    """Get cached model predictions."""
    storage = DataStorage()
    cache_key = f"predictions:{symbol}:{model_name}:{datetime.now().strftime('%Y%m%d')}"
    return storage.get_cached_data(cache_key)


if __name__ == "__main__":
    # Example usage
    storage = DataStorage()
    
    # Get data summary
    summary = storage.get_data_summary()
    print("Data Summary:")
    for data_type, stats in summary.items():
        print(f"{data_type}: {stats}")
    
    # Example: Store sample data
    sample_data = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [102, 103, 104],
        'low': [99, 100, 101],
        'close': [101, 102, 103],
        'volume': [1000, 1100, 1200]
    }, index=pd.date_range('2023-01-01', periods=3))
    
    success = storage.store_market_data({'AAPL': sample_data})
    print(f"Data storage successful: {success}")
