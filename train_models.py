"""
Model Training Script for Trading Strategy ML System
This script demonstrates how to train the CNN+LSTM and Random Forest models.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import components
from src.data_pipeline.market_data_collector import MarketDataCollector
from src.data_pipeline.indicator_engine import TechnicalIndicatorEngine
from src.data_pipeline.feature_engineer import FeatureEngineer
from src.ml_models.model_trainer import ModelTrainer
from src.ml_models.data_preprocessor import DataPreprocessor
from config.model_config import ModelConfig


def collect_training_data(symbols, period="2y", interval="1d"):
    """
    Collect training data for multiple symbols.
    
    Args:
        symbols (list): List of stock symbols to collect data for
        period (str): Data period (e.g., "1y", "2y", "5y")
        interval (str): Data interval (e.g., "1d", "1h", "5m")
    
    Returns:
        dict: Dictionary with symbol as key and DataFrame as value
    """
    print("🔄 Collecting training data...")
    
    collector = MarketDataCollector()
    data_dict = {}
    
    for i, symbol in enumerate(symbols):
        print(f"  📊 Collecting data for {symbol} ({i+1}/{len(symbols)})")
        
        try:
            # Collect data
            data = collector.get_yahoo_finance_data(symbol, period=period, interval=interval)
            
            if data.empty:
                print(f"  ⚠️  No data found for {symbol}, skipping...")
                continue
            
            # Store data
            data_dict[symbol] = data
            print(f"  ✅ Collected {len(data)} days of data for {symbol}")
            
        except Exception as e:
            print(f"  ❌ Error collecting data for {symbol}: {e}")
            continue
    
    print(f"✅ Data collection complete! Collected data for {len(data_dict)} symbols")
    return data_dict


def prepare_features(data_dict):
    """
    Prepare features for all symbols.
    
    Args:
        data_dict (dict): Dictionary with symbol as key and DataFrame as value
    
    Returns:
        dict: Dictionary with processed data for each symbol
    """
    print("🔄 Preparing features...")
    
    indicator_engine = TechnicalIndicatorEngine()
    feature_engineer = FeatureEngineer()
    processed_data = {}
    
    for symbol, data in data_dict.items():
        print(f"  🔧 Processing features for {symbol}")
        
        try:
            # Calculate technical indicators
            data_with_indicators = indicator_engine.calculate_all_indicators(data.copy())
            
            # Generate basic features
            data_with_features = feature_engineer.generate_basic_features(data_with_indicators.copy())
            
            # Generate advanced features
            data_with_features = feature_engineer.generate_advanced_features(data_with_features.copy())
            
            # Create target variable (next day's return)
            data_with_features['next_day_return'] = data_with_features['close'].pct_change().shift(-1)
            
            # Store processed data
            processed_data[symbol] = data_with_features.dropna()
            
            print(f"  ✅ Processed {len(processed_data[symbol])} samples for {symbol}")
            
        except Exception as e:
            print(f"  ❌ Error processing features for {symbol}: {e}")
            continue
    
    print(f"✅ Feature preparation complete! Processed {len(processed_data)} symbols")
    return processed_data


def train_models_for_symbol(symbol, data, feature_cols, target_col):
    """
    Train models for a single symbol.
    
    Args:
        symbol (str): Stock symbol
        data (pd.DataFrame): Processed data with features
        feature_cols (list): List of feature column names
        target_col (str): Target column name
    
    Returns:
        dict: Training results
    """
    print(f"🤖 Training models for {symbol}...")
    
    try:
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Prepare training data
        X_train, X_val, y_train, y_val, feature_names = trainer.prepare_data_for_training(
            df=data,
            feature_cols=feature_cols,
            target_col=target_col,
            symbol=symbol
        )
        
        print(f"  📊 Training data shape: {X_train.shape}")
        print(f"  📊 Validation data shape: {X_val.shape}")
        
        # Train all models
        trainer.train_all_models(X_train, X_val, y_train, y_val, feature_names)
        
        # Evaluate models
        evaluation_results = trainer.evaluate_models(X_val, y_val, symbol)
        
        print(f"  ✅ Training complete for {symbol}")
        print(f"  📈 CNN+LSTM R²: {evaluation_results['cnn_lstm_r2']:.3f}")
        print(f"  📈 Random Forest R²: {evaluation_results['random_forest_r2']:.3f}")
        print(f"  📈 Ensemble R²: {evaluation_results['ensemble_r2']:.3f}")
        
        return {
            'symbol': symbol,
            'trainer': trainer,
            'evaluation_results': evaluation_results,
            'feature_names': feature_names
        }
        
    except Exception as e:
        print(f"  ❌ Error training models for {symbol}: {e}")
        return None


def train_models_batch(processed_data, feature_cols, target_col):
    """
    Train models for multiple symbols in batch.
    
    Args:
        processed_data (dict): Processed data for all symbols
        feature_cols (list): List of feature column names
        target_col (str): Target column name
    
    Returns:
        dict: Training results for all symbols
    """
    print("🤖 Training models for all symbols...")
    
    training_results = {}
    
    for symbol, data in processed_data.items():
        result = train_models_for_symbol(symbol, data, feature_cols, target_col)
        if result:
            training_results[symbol] = result
    
    print(f"✅ Batch training complete! Trained models for {len(training_results)} symbols")
    return training_results


def train_universal_model(processed_data, feature_cols, target_col):
    """
    Train a universal model using data from all symbols.
    
    Args:
        processed_data (dict): Processed data for all symbols
        feature_cols (list): List of feature column names
        target_col (str): Target column name
    
    Returns:
        dict: Universal model training results
    """
    print("🌍 Training universal model...")
    
    try:
        # Combine data from all symbols
        combined_data = []
        for symbol, data in processed_data.items():
            data_copy = data.copy()
            data_copy['symbol'] = symbol
            combined_data.append(data_copy)
        
        # Concatenate all data
        universal_data = pd.concat(combined_data, ignore_index=True)
        
        print(f"  📊 Universal dataset size: {len(universal_data)} samples")
        print(f"  📊 Symbols included: {universal_data['symbol'].nunique()}")
        
        # Train universal model
        trainer = ModelTrainer()
        
        X_train, X_val, y_train, y_val, feature_names = trainer.prepare_data_for_training(
            df=universal_data,
            feature_cols=feature_cols,
            target_col=target_col,
            symbol='UNIVERSAL'
        )
        
        print(f"  📊 Training data shape: {X_train.shape}")
        print(f"  📊 Validation data shape: {X_val.shape}")
        
        # Train all models
        trainer.train_all_models(X_train, X_val, y_train, y_val, feature_names)
        
        # Evaluate models
        evaluation_results = trainer.evaluate_models(X_val, y_val, 'UNIVERSAL')
        
        print(f"  ✅ Universal model training complete")
        print(f"  📈 CNN+LSTM R²: {evaluation_results['cnn_lstm_r2']:.3f}")
        print(f"  📈 Random Forest R²: {evaluation_results['random_forest_r2']:.3f}")
        print(f"  📈 Ensemble R²: {evaluation_results['ensemble_r2']:.3f}")
        
        return {
            'trainer': trainer,
            'evaluation_results': evaluation_results,
            'feature_names': feature_names,
            'universal_data': universal_data
        }
        
    except Exception as e:
        print(f"  ❌ Error training universal model: {e}")
        return None


def save_training_results(training_results, universal_results, output_dir="models"):
    """
    Save training results and models.
    
    Args:
        training_results (dict): Individual symbol training results
        universal_results (dict): Universal model training results
        output_dir (str): Output directory for models
    """
    print("💾 Saving training results...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual models
    for symbol, result in training_results.items():
        if result and 'trainer' in result:
            trainer = result['trainer']
            
            # Save CNN+LSTM model
            if trainer.cnn_lstm_model:
                trainer.cnn_lstm_model.save_model(
                    path=output_dir,
                    filename=f"cnn_lstm_{symbol}.h5"
                )
            
            # Save Random Forest model
            if trainer.random_forest_model:
                trainer.random_forest_model.save_model(
                    path=output_dir,
                    filename=f"random_forest_{symbol}.pkl"
                )
    
    # Save universal model
    if universal_results and 'trainer' in universal_results:
        trainer = universal_results['trainer']
        
        # Save universal models
        if trainer.cnn_lstm_model:
            trainer.cnn_lstm_model.save_model(
                path=output_dir,
                filename="cnn_lstm_universal.h5"
            )
        
        if trainer.random_forest_model:
            trainer.random_forest_model.save_model(
                path=output_dir,
                filename="random_forest_universal.pkl"
            )
    
    # Save training summary
    summary = {
        'training_date': datetime.now().isoformat(),
        'individual_models': len(training_results),
        'universal_model': universal_results is not None,
        'symbols_trained': list(training_results.keys())
    }
    
    import json
    with open(os.path.join(output_dir, "training_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Training results saved to {output_dir}")


def main():
    """
    Main training function.
    """
    print("🚀 Starting Model Training Pipeline")
    print("=" * 50)
    
    # Configuration
    SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
    PERIOD = "2y"  # 2 years of data
    INTERVAL = "1d"  # Daily data
    
    # Feature columns to use
    FEATURE_COLS = [
        'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'ROC', 'STOCH_K', 'STOCH_D',
        'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'ATR', 'NATR', 'OBV', 'MFI',
        'daily_return', 'log_return', 'high_low_range', 'open_close_range',
        'volatility_ratio', 'RSI_MACD_interaction'
    ]
    
    TARGET_COL = 'next_day_return'
    
    try:
        # Step 1: Collect data
        data_dict = collect_training_data(SYMBOLS, PERIOD, INTERVAL)
        
        if not data_dict:
            print("❌ No data collected. Exiting...")
            return
        
        # Step 2: Prepare features
        processed_data = prepare_features(data_dict)
        
        if not processed_data:
            print("❌ No data processed. Exiting...")
            return
        
        # Step 3: Train individual models
        print("\n" + "=" * 50)
        training_results = train_models_batch(processed_data, FEATURE_COLS, TARGET_COL)
        
        # Step 4: Train universal model
        print("\n" + "=" * 50)
        universal_results = train_universal_model(processed_data, FEATURE_COLS, TARGET_COL)
        
        # Step 5: Save results
        print("\n" + "=" * 50)
        save_training_results(training_results, universal_results)
        
        # Step 6: Print summary
        print("\n" + "=" * 50)
        print("🎉 TRAINING COMPLETE!")
        print(f"📊 Symbols trained: {len(training_results)}")
        print(f"🌍 Universal model: {'✅' if universal_results else '❌'}")
        print(f"💾 Models saved to: models/")
        
        # Print performance summary
        if training_results:
            print("\n📈 PERFORMANCE SUMMARY:")
            for symbol, result in training_results.items():
                if result and 'evaluation_results' in result:
                    eval_results = result['evaluation_results']
                    print(f"  {symbol}:")
                    print(f"    CNN+LSTM R²: {eval_results['cnn_lstm_r2']:.3f}")
                    print(f"    Random Forest R²: {eval_results['random_forest_r2']:.3f}")
                    print(f"    Ensemble R²: {eval_results['ensemble_r2']:.3f}")
        
        if universal_results and 'evaluation_results' in universal_results:
            eval_results = universal_results['evaluation_results']
            print(f"\n🌍 UNIVERSAL MODEL:")
            print(f"  CNN+LSTM R²: {eval_results['cnn_lstm_r2']:.3f}")
            print(f"  Random Forest R²: {eval_results['random_forest_r2']:.3f}")
            print(f"  Ensemble R²: {eval_results['ensemble_r2']:.3f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
