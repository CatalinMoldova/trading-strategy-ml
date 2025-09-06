# How to Train the Trading Strategy ML Models

## Quick Start (Recommended for Beginners)

### 1. Run the Quick Training Script

```bash
cd trading_strategy_ml
python quick_train.py
```

This will:
- Collect 1 year of AAPL data
- Calculate technical indicators
- Generate features
- Train CNN+LSTM and Random Forest models
- Save models to `models/` directory

### 2. Check Results

After training completes, you'll see:
```
🎉 TRAINING COMPLETE!
📁 Models saved to: models/
📈 Model Evaluation Results:
   CNN+LSTM R²: 0.234
   Random Forest R²: 0.189
   Ensemble R²: 0.245
```

## Advanced Training (Multiple Symbols)

### 1. Run the Full Training Pipeline

```bash
python train_models.py
```

This will:
- Collect data for 8 symbols (AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, NFLX)
- Train individual models for each symbol
- Train a universal model using all data
- Save all models with performance metrics

### 2. Customize Training

Edit `train_models.py` to modify:

```python
# Change symbols to train
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

# Change data period
PERIOD = "3y"  # 3 years of data

# Change data frequency
INTERVAL = "1h"  # Hourly data

# Modify features
FEATURE_COLS = [
    'RSI', 'MACD', 'ATR', 'daily_return',  # Add/remove features
    'volatility_ratio', 'RSI_MACD_interaction'
]
```

## Step-by-Step Manual Training

### 1. Collect Data

```python
from src.data_pipeline.market_data_collector import MarketDataCollector

collector = MarketDataCollector()
data = collector.get_yahoo_finance_data("AAPL", period="2y", interval="1d")
print(f"Collected {len(data)} days of data")
```

### 2. Calculate Technical Indicators

```python
from src.data_pipeline.indicator_engine import IndicatorEngine

engine = IndicatorEngine()
data = engine.calculate_all_indicators(data)
print("Added technical indicators:", [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume']])
```

### 3. Generate Features

```python
from src.data_pipeline.feature_engineer import FeatureEngineer

fe = FeatureEngineer()
data = fe.generate_basic_features(data)
data = fe.generate_advanced_features(data)

# Create target variable
data['next_day_return'] = data['close'].pct_change().shift(-1)
data = data.dropna()
print(f"Dataset size: {len(data)} samples")
```

### 4. Train Models

```python
from src.ml_models.model_trainer import ModelTrainer

trainer = ModelTrainer()

# Define features
feature_cols = ['RSI', 'MACD', 'ATR', 'daily_return', 'volatility_ratio']

# Prepare training data
X_train, X_val, y_train, y_val, feature_names = trainer.prepare_data_for_training(
    df=data,
    feature_cols=feature_cols,
    target_col='next_day_return',
    symbol='AAPL'
)

# Train models
trainer.train_all_models(X_train, X_val, y_train, y_val, feature_names)

# Evaluate models
evaluation_results = trainer.evaluate_models(X_val, y_val, 'AAPL')
print("Evaluation results:", evaluation_results)
```

### 5. Save Models

```python
import os
os.makedirs("models", exist_ok=True)

# Save CNN+LSTM model
trainer.cnn_lstm_model.save_model(path="models", filename="cnn_lstm_aapl.h5")

# Save Random Forest model
trainer.random_forest_model.save_model(path="models", filename="random_forest_aapl.pkl")
```

## Training Configuration

### Model Parameters

Edit `config/model_config.py` to modify:

```python
class ModelConfig:
    # CNN+LSTM Parameters
    CNN_FILTERS = 64          # Number of CNN filters
    LSTM_UNITS = 100         # Number of LSTM units
    DROPOUT_RATE = 0.3       # Dropout rate
    TIME_STEPS = 60          # Number of time steps
    N_FEATURES = 20          # Number of features
    
    # Training Parameters
    BATCH_SIZE = 32          # Training batch size
    EPOCHS = 50              # Number of training epochs
    LEARNING_RATE = 0.001    # Learning rate
    VALIDATION_SPLIT = 0.2   # Validation split ratio
    
    # Random Forest Parameters
    RF_N_ESTIMATORS = 100    # Number of trees
    RF_MAX_DEPTH = 10        # Maximum tree depth
```

### Data Requirements

**Minimum Requirements:**
- **Data Points**: At least 1000 samples (about 4 years of daily data)
- **Features**: At least 5 technical indicators
- **Symbols**: 1+ symbols for training

**Recommended:**
- **Data Points**: 2000+ samples (8+ years of daily data)
- **Features**: 10+ technical indicators
- **Symbols**: 5+ symbols for robust training

## Troubleshooting

### Common Issues

#### 1. "No data collected"
```bash
# Check internet connection
ping google.com

# Try different symbol
python -c "from src.data_pipeline.market_data_collector import MarketDataCollector; print(MarketDataCollector().get_yahoo_finance_data('AAPL', '1mo', '1d').shape)"
```

#### 2. "Insufficient data for training"
```python
# Use longer period
data = collector.get_yahoo_finance_data("AAPL", period="5y", interval="1d")

# Or use more symbols
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
```

#### 3. "Model training failed"
```python
# Reduce model complexity
CNN_FILTERS = 32
LSTM_UNITS = 50
EPOCHS = 20

# Or use fewer features
feature_cols = ['RSI', 'MACD', 'ATR']  # Only 3 features
```

#### 4. "Memory error"
```python
# Reduce batch size
BATCH_SIZE = 16

# Process data in chunks
def process_large_dataset(df, chunk_size=1000):
    for chunk in pd.read_csv('data.csv', chunksize=chunk_size):
        yield process_chunk(chunk)
```

### Performance Tips

#### 1. Use GPU for Training
```python
# Install TensorFlow GPU
pip install tensorflow-gpu

# Verify GPU usage
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
```

#### 2. Optimize Data Processing
```python
# Use vectorized operations
data['RSI'] = ta.RSI(data['close'], timeperiod=14)

# Cache processed data
import pickle
with open('processed_data.pkl', 'wb') as f:
    pickle.dump(data, f)
```

#### 3. Parallel Processing
```python
# Train multiple symbols in parallel
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(train_symbol, symbol) for symbol in symbols]
    results = [future.result() for future in futures]
```

## Model Performance

### Expected Results

**Good Performance:**
- R² > 0.15 (15% variance explained)
- MSE < 0.001 (low prediction error)
- Directional accuracy > 55%

**Excellent Performance:**
- R² > 0.25 (25% variance explained)
- MSE < 0.0005 (very low prediction error)
- Directional accuracy > 60%

### Model Comparison

| Model | R² | MSE | Training Time | Use Case |
|-------|----|----|----|----|
| CNN+LSTM | 0.20-0.30 | 0.0005-0.001 | 10-30 min | Time series patterns |
| Random Forest | 0.15-0.25 | 0.0008-0.0015 | 2-5 min | Feature importance |
| Ensemble | 0.25-0.35 | 0.0004-0.0008 | 12-35 min | Best overall |

## Next Steps

After training models:

1. **Run Backtest**: `python run_backtest.py`
2. **Analyze Performance**: Check evaluation results
3. **Optimize Parameters**: Tune hyperparameters
4. **Deploy Models**: Use for live trading
5. **Monitor Performance**: Retrain periodically

## Support

If you encounter issues:

1. **Check Logs**: Look for error messages in console output
2. **Verify Data**: Ensure data collection is working
3. **Test Components**: Run individual components separately
4. **Check Dependencies**: Verify all packages are installed
5. **Read Documentation**: Check `docs/user_guide.md` for detailed help
