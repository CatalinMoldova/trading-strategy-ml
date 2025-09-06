"""
Quick Model Training Script - Simple Example
This script shows how to train models with minimal configuration.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import components
from src.data_pipeline.market_data_collector import MarketDataCollector
from src.data_pipeline.indicator_engine import TechnicalIndicatorEngine
from src.data_pipeline.feature_engineer import FeatureEngineer
from src.ml_models.model_trainer import ModelTrainer
from config.model_config import ModelConfig


def quick_train_example():
    """
    Quick training example with a single symbol.
    """
    print("🚀 Quick Model Training Example")
    print("=" * 40)
    
    # Step 1: Collect data for a single symbol
    print("📊 Collecting data for AAPL...")
    collector = MarketDataCollector()
    data = collector.get_yahoo_finance_data("AAPL", period="2y", interval="1d")
    
    if data.empty:
        print("❌ No data collected. Please check your internet connection.")
        return
    
    print(f"✅ Collected {len(data)} days of data")
    
    # Step 2: Calculate technical indicators
    print("🔧 Calculating technical indicators...")
    engine = TechnicalIndicatorEngine()
    data = engine.calculate_all_indicators(data)
    print(f"✅ Added {len([col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume']])} technical indicators")
    
    # Step 3: Generate features
    print("🔧 Generating features...")
    fe = FeatureEngineer()
    data = fe.create_comprehensive_features(data)
    
    # Create target variable
    data['next_day_return'] = data['close'].pct_change().shift(-1)
    
    # Remove rows with NaN values
    print(f"Data shape before cleaning: {data.shape}")
    
    # Remove columns that are completely NaN
    data = data.dropna(axis=1, how='all')
    print(f"After removing all-NaN columns: {data.shape}")
    
    # Fill remaining NaN values with forward fill, then backward fill
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Remove any remaining rows with NaN values
    data = data.dropna()
    print(f"✅ Generated features. Dataset size: {len(data)} samples")
    
    # Step 4: Prepare training data
    print("🤖 Preparing training data...")
    config = ModelConfig()
    trainer = ModelTrainer(config)
    
    # Define features to use
    feature_cols = ['RSI_OVERBOUGHT', 'RSI_OVERSOLD', 'MACD_SIGNAL', 'MACD_HISTOGRAM', 'ATR_SMA', 'ATR_RATIO', 'VOLATILITY_20', 'VOLATILITY_RATIO']
    
    try:
        # Prepare features and target
        features = data[feature_cols]
        target = data['next_day_return']
        
        # Prepare training data
        training_data = trainer.prepare_training_data(features, target)
        
        if not training_data:
            print("❌ Failed to prepare training data")
            return
        
        X_train = training_data['X_train']
        X_val = training_data['X_val']
        y_train = training_data['y_train']
        y_val = training_data['y_val']
        feature_names = feature_cols
        
        print(f"✅ Training data prepared:")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Validation samples: {X_val.shape[0]}")
        print(f"   Features: {X_train.shape[2]}")
        print(f"   Time steps: {X_train.shape[1]}")
        
    except Exception as e:
        print(f"❌ Error preparing training data: {e}")
        print("💡 Try using fewer features or more data")
        return
    
    # Step 5: Train models
    print("🤖 Training models...")
    try:
        trainer.train_all_models(features, target)
        print("✅ Models trained successfully!")
        
    except Exception as e:
        print(f"❌ Error training models: {e}")
        print("💡 Check that you have sufficient data and correct feature names")
        return
    
    # Step 6: Evaluate models
    print("📈 Evaluating models...")
    try:
        # Get test data from the prepared training data
        test_data = trainer.prepare_training_data(features, target)
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        evaluation_results = trainer.evaluate_all_models(X_test, y_test)
        
        print("✅ Model Evaluation Results:")
        if 'cnn_lstm' in evaluation_results:
            print(f"   CNN+LSTM R²: {evaluation_results['cnn_lstm'].get('r2_score', 'N/A')}")
            print(f"   CNN+LSTM MSE: {evaluation_results['cnn_lstm'].get('mse', 'N/A')}")
        
        if 'random_forest' in evaluation_results:
            print(f"   Random Forest R²: {evaluation_results['random_forest'].get('r2_score', 'N/A')}")
            print(f"   Random Forest MSE: {evaluation_results['random_forest'].get('mse', 'N/A')}")
        
        if 'ensemble' in evaluation_results:
            print(f"   Ensemble R²: {evaluation_results['ensemble'].get('r2_score', 'N/A')}")
            print(f"   Ensemble MSE: {evaluation_results['ensemble'].get('mse', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error evaluating models: {e}")
        return
    
    # Step 7: Save models
    print("💾 Saving models...")
    try:
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        # Save models
        if trainer.cnn_lstm_model:
            trainer.cnn_lstm_model.save_model(path="models", filename="cnn_lstm_aapl.h5")
            print("✅ CNN+LSTM model saved")
        
        if trainer.random_forest_model:
            trainer.random_forest_model.save_model(path="models", filename="random_forest_aapl.pkl")
            print("✅ Random Forest model saved")
        
        print("💾 All models saved to models/ directory")
        
    except Exception as e:
        print(f"❌ Error saving models: {e}")
        return
    
    print("\n🎉 TRAINING COMPLETE!")
    print("=" * 40)
    print("📁 Models saved to: models/")
    print("🔧 Next steps:")
    print("   1. Run backtest: python run_backtest.py")
    print("   2. Check model performance in models/ directory")
    print("   3. Use trained models for live trading")


if __name__ == "__main__":
    quick_train_example()
