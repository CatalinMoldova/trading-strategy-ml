# Trading Strategy ML - User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Quick Start](#quick-start)
5. [Data Pipeline](#data-pipeline)
6. [Machine Learning](#machine-learning)
7. [Trading Strategy](#trading-strategy)
8. [Backtesting](#backtesting)
9. [API Reference](#api-reference)
10. [Examples](#examples)
11. [Troubleshooting](#troubleshooting)
12. [FAQ](#faq)

## Getting Started

The Trading Strategy ML system is a comprehensive quantitative trading platform that combines traditional technical analysis with advanced machine learning techniques. This guide will help you get started with the system.

### What You'll Learn

- How to install and configure the system
- How to collect and process market data
- How to train machine learning models
- How to implement trading strategies
- How to run backtests and analyze performance
- How to use the API for custom implementations

### Prerequisites

- Python 3.9 or higher
- Basic knowledge of Python programming
- Understanding of financial markets and trading concepts
- Familiarity with machine learning concepts (helpful but not required)

## Installation

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10+
- **Python**: 3.9 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space minimum
- **Internet**: Stable internet connection for data collection

### Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/trading-strategy-ml.git
cd trading-strategy-ml
```

#### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n trading-strategy python=3.9
conda activate trading-strategy
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Install TA-Lib (Required for Technical Indicators)

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
pip install TA-Lib
```

**On macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**On Windows:**
```bash
# Download pre-compiled wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.19-cp39-cp39-win_amd64.whl
```

#### 5. Verify Installation

```bash
python -c "import pandas, numpy, sklearn, tensorflow, talib; print('Installation successful!')"
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_db
DB_USER=user
DB_PASSWORD=password

# API Keys
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Model Configuration
MODEL_SAVE_PATH=models/
```

### Configuration Files

#### Database Configuration (`config/database_config.py`)

```python
import os

class DatabaseConfig:
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "trading_db")
    DB_USER = os.getenv("DB_USER", "user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    TIMESCALEDB_ENABLED = os.getenv("TIMESCALEDB_ENABLED", "True").lower() == "true"
```

#### Model Configuration (`config/model_config.py`)

```python
class ModelConfig:
    # CNN-LSTM Model Parameters
    CNN_FILTERS = 64
    CNN_KERNEL_SIZE = 2
    LSTM_UNITS = 100
    DROPOUT_RATE = 0.3
    TIME_STEPS = 60
    N_FEATURES = 20
    
    # Training Parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10
    
    # Ensemble Parameters
    ENSEMBLE_WEIGHTS = {"cnn_lstm": 0.6, "random_forest": 0.4}
    CONFIDENCE_THRESHOLD = 0.55
```

## Quick Start

### 1. Basic Data Collection

```python
from src.data_pipeline.market_data_collector import MarketDataCollector

# Initialize collector
collector = MarketDataCollector()

# Collect data for a single symbol
data = collector.get_yahoo_finance_data("AAPL", period="1y", interval="1d")
print(data.head())
```

### 2. Calculate Technical Indicators

```python
from src.data_pipeline.indicator_engine import IndicatorEngine

# Initialize indicator engine
engine = IndicatorEngine()

# Calculate all indicators
data_with_indicators = engine.calculate_all_indicators(data)
print(data_with_indicators.columns)
```

### 3. Generate Features

```python
from src.data_pipeline.feature_engineer import FeatureEngineer

# Initialize feature engineer
fe = FeatureEngineer()

# Generate basic features
data_with_features = fe.generate_basic_features(data_with_indicators)
print(data_with_features.columns)
```

### 4. Train ML Models

```python
from src.ml_models.model_trainer import ModelTrainer

# Initialize trainer
trainer = ModelTrainer()

# Prepare data
feature_cols = ['RSI', 'MACD', 'ATR', 'daily_return']
target_col = 'next_day_return'

# Create target variable
data_with_features[target_col] = data_with_features['close'].pct_change().shift(-1)

# Train models
X_train, X_val, y_train, y_val, feature_names = trainer.prepare_data_for_training(
    df=data_with_features.dropna(),
    feature_cols=feature_cols,
    target_col=target_col,
    symbol='AAPL'
)

trainer.train_all_models(X_train, X_val, y_train, y_val, feature_names)
```

### 5. Run Backtest

```python
from src.backtesting.backtest_engine import BacktestEngine
from src.strategy.signal_generator import SignalGenerator
from src.strategy.position_sizer import PositionSizer
from src.strategy.risk_manager import RiskManager
from src.ml_models.ensemble_predictor import EnsemblePredictor

# Initialize components
backtest_engine = BacktestEngine()
signal_generator = SignalGenerator()
position_sizer = PositionSizer()
risk_manager = RiskManager()
ensemble_predictor = EnsemblePredictor()

# Prepare data
data_dict = {'AAPL': data_with_features}

# Run backtest
results = backtest_engine.run_backtest(
    data=data_dict,
    signal_generator=signal_generator,
    position_sizer=position_sizer,
    risk_manager=risk_manager,
    ensemble_predictor=ensemble_predictor
)

print(f"Total Return: {results['performance_metrics']['total_return']:.2%}")
print(f"Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.2f}")
```

## Data Pipeline

### Market Data Collection

#### Yahoo Finance Data

```python
from src.data_pipeline.market_data_collector import MarketDataCollector

collector = MarketDataCollector()

# Daily data
daily_data = collector.get_yahoo_finance_data(
    ticker="AAPL",
    period="1y",  # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    interval="1d"  # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
)

# Intraday data
intraday_data = collector.get_yahoo_finance_data(
    ticker="AAPL",
    period="5d",
    interval="5m"
)

# Multiple symbols
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
multi_data = collector.get_multiple_symbols_data(
    symbols=symbols,
    source="yahoo",
    period="6mo",
    interval="1d"
)
```

#### Alpha Vantage Data

```python
# Daily data
daily_data = collector.get_alpha_vantage_daily_data(
    ticker="AAPL",
    outputsize="full"  # 'compact' or 'full'
)

# Intraday data
intraday_data = collector.get_alpha_vantage_intraday_data(
    ticker="AAPL",
    interval="5min",  # 1min, 5min, 15min, 30min, 60min
    outputsize="full"
)
```

### Technical Indicators

#### Momentum Indicators

```python
from src.data_pipeline.indicator_engine import IndicatorEngine

engine = IndicatorEngine()

# Calculate momentum indicators
data = engine.calculate_momentum_indicators(data)
# Adds: RSI, MACD, MACD_SIGNAL, MACD_HIST, ROC, STOCH_K, STOCH_D, ADX
```

#### Volatility Indicators

```python
# Calculate volatility indicators
data = engine.calculate_volatility_indicators(data)
# Adds: BB_UPPER, BB_MIDDLE, BB_LOWER, ATR, NATR, TRANGE
```

#### Volume Indicators

```python
# Calculate volume indicators
data = engine.calculate_volume_indicators(data)
# Adds: OBV, MFI
```

#### All Indicators

```python
# Calculate all indicators at once
data = engine.calculate_all_indicators(data)
```

### Feature Engineering

#### Basic Features

```python
from src.data_pipeline.feature_engineer import FeatureEngineer

fe = FeatureEngineer()

# Generate basic features
data = fe.generate_basic_features(data)
# Adds: daily_return, log_return, high_low_range, open_close_range
```

#### Advanced Features

```python
# Generate advanced features
data = fe.generate_advanced_features(data)
# Adds: volatility_ratio, RSI_MACD_interaction
```

#### Cross-Sectional Normalization

```python
# Normalize features across multiple symbols
data_dict = {
    'AAPL': data_aapl,
    'MSFT': data_msft,
    'GOOGL': data_googl
}

feature_cols = ['close', 'RSI', 'MACD']
normalized_data = fe.cross_sectional_normalize(data_dict, feature_cols)
```

#### Time Series Standardization

```python
# Standardize features for a single symbol
feature_cols = ['RSI', 'MACD', 'ATR']
standardized_data = fe.time_series_standardize(
    data, feature_cols, symbol='AAPL'
)
```

#### ML Data Preparation

```python
# Prepare data for ML models
ml_data = fe.prepare_ml_data(
    df=data,
    target_column='next_day_return',
    time_steps=60
)

print(f"X shape: {ml_data['X'].shape}")
print(f"y shape: {ml_data['y'].shape}")
```

## Machine Learning

### Data Preprocessing

```python
from src.ml_models.data_preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()

# Scale features
scaled_data = preprocessor.scale_features(
    df=data,
    feature_cols=['RSI', 'MACD', 'ATR'],
    scaler_type='standard',
    symbol='AAPL',
    fit=True
)

# Create sequences for time series models
X, y = preprocessor.create_sequences_with_target(
    features_df=scaled_data[['RSI', 'MACD', 'ATR']],
    target_series=data['next_day_return'],
    time_steps=60
)
```

### CNN+LSTM Model

```python
from src.ml_models.cnn_lstm_model import CNNLSTMModel

# Initialize model
model = CNNLSTMModel(
    time_steps=60,
    n_features=20,
    output_dim=1
)

# Train model
history = model.train(X_train, y_train, X_val, y_val)

# Make predictions
predictions = model.predict(X_test)

# Save model
model.save_model()
```

### Random Forest Model

```python
from src.ml_models.random_forest_model import RandomForestModel

# Initialize model
rf_model = RandomForestModel()

# Train model
rf_model.train(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)

# Get feature importance
feature_importance = rf_model.get_feature_importance(['RSI', 'MACD', 'ATR'])
print(feature_importance)

# Save model
rf_model.save_model()
```

### Ensemble Predictor

```python
from src.ml_models.ensemble_predictor import EnsemblePredictor

# Initialize ensemble
ensemble = EnsemblePredictor()

# Combine predictions
predictions = {
    'cnn_lstm': cnn_lstm_predictions,
    'random_forest': rf_predictions
}

combined_predictions = ensemble.combine_predictions(predictions)

# Generate signals
signals, confidence = ensemble.generate_signal(combined_predictions)
```

### Model Training

```python
from src.ml_models.model_trainer import ModelTrainer

# Initialize trainer
trainer = ModelTrainer()

# Prepare data
X_train, X_val, y_train, y_val, feature_names = trainer.prepare_data_for_training(
    df=data,
    feature_cols=['RSI', 'MACD', 'ATR'],
    target_col='next_day_return',
    symbol='AAPL'
)

# Train all models
trainer.train_all_models(X_train, X_val, y_train, y_val, feature_names)

# Evaluate models
evaluation_results = trainer.evaluate_models(X_val, y_val, 'AAPL')
print(evaluation_results)
```

## Trading Strategy

### Signal Generation

```python
from src.strategy.signal_generator import SignalGenerator

# Initialize signal generator
sg = SignalGenerator(confidence_threshold=0.02)

# Generate ML signal
ml_signal, ml_confidence = sg.generate_ml_signal(
    ml_prediction=0.03,
    ml_confidence=0.035
)

# Generate multi-factor score
features = pd.Series({
    'RSI': 0.7,
    'MACD_HIST': 0.05,
    'ATR_NORM': 0.01
})

factor_weights = {
    'RSI': 0.4,
    'MACD_HIST': 0.3,
    'ATR_NORM': -0.1
}

factor_score = sg.generate_multi_factor_score(features, factor_weights)

# Combine signals
final_signal, final_confidence = sg.combine_signals(
    ml_signal=ml_signal,
    ml_confidence=ml_confidence,
    factor_score=factor_score,
    ml_weight=0.6,
    factor_weight=0.4
)
```

### Position Sizing

```python
from src.strategy.position_sizer import PositionSizer

# Initialize position sizer
ps = PositionSizer(kelly_fraction=0.5)

# Calculate GARCH volatility
returns = data['close'].pct_change().dropna()
volatility = ps.calculate_garch_volatility(returns)

# Calculate Kelly Criterion
kelly_fraction = ps.calculate_kelly_criterion(
    win_rate=0.58,
    avg_win_loss_ratio=1.3
)

# Get position size
position_units, capital_allocated = ps.get_position_size(
    signal_confidence=0.75,
    asset_volatility=volatility,
    account_balance=100000.0,
    price=150.0,
    win_rate=0.58,
    avg_win_loss_ratio=1.3
)
```

### Risk Management

```python
from src.strategy.risk_manager import RiskManager

# Initialize risk manager
rm = RiskManager(
    max_position_size_pct=0.05,
    max_portfolio_risk_pct=0.20,
    stop_loss_pct=0.02,
    take_profit_pct=0.04
)

# Validate new trade
is_valid, reason = rm.validate_new_trade(
    symbol='AAPL',
    trade_type='BUY',
    quantity=100,
    price=150.0,
    current_portfolio_value=100000.0,
    open_positions_count=5
)

# Calculate stop-loss and take-profit
stop_loss_price = rm.calculate_stop_loss_price(150.0, 'BUY')
take_profit_price = rm.calculate_take_profit_price(150.0, 'BUY')

# Check exit conditions
should_exit, exit_reason = rm.check_position_exit_conditions(
    current_price=145.0,
    position_entry_price=150.0,
    trade_type='BUY'
)
```

## Backtesting

### Basic Backtest

```python
from src.backtesting.backtest_engine import BacktestEngine

# Initialize backtest engine
backtest_engine = BacktestEngine(
    initial_capital=100000.0,
    commission=0.001,
    slippage=0.0005
)

# Run backtest
results = backtest_engine.run_backtest(
    data=data_dict,
    signal_generator=signal_generator,
    position_sizer=position_sizer,
    risk_manager=risk_manager,
    ensemble_predictor=ensemble_predictor
)

# Print results
print(f"Total Return: {results['performance_metrics']['total_return']:.2%}")
print(f"Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['performance_metrics']['max_drawdown']:.2%}")
```

### Performance Analysis

```python
from src.backtesting.performance_analyzer import PerformanceAnalyzer

# Initialize analyzer
analyzer = PerformanceAnalyzer()

# Analyze performance
analysis = analyzer.analyze_performance(results)

# Print analysis
print("Performance Metrics:")
for metric, value in analysis['performance_metrics'].items():
    print(f"{metric}: {value:.4f}")

print("\nRisk Metrics:")
for metric, value in analysis['risk_metrics'].items():
    print(f"{metric}: {value:.4f}")
```

### Statistical Validation

```python
from src.backtesting.statistical_validator import StatisticalValidator

# Initialize validator
validator = StatisticalValidator()

# Validate performance
validation = validator.validate_performance(results)

# Print validation results
print("Statistical Tests:")
for test, result in validation['performance_significance'].items():
    print(f"{test}: {result}")
```

### Walk-Forward Analysis

```python
from src.backtesting.walk_forward_analyzer import WalkForwardAnalyzer

# Initialize analyzer
wf_analyzer = WalkForwardAnalyzer()

# Run walk-forward analysis
wf_results = wf_analyzer.run_walk_forward_analysis(
    data=data_dict,
    train_window=252,  # 1 year
    test_window=63,    # 3 months
    step_size=21,      # 1 month
    signal_generator=signal_generator,
    position_sizer=position_sizer,
    risk_manager=risk_manager,
    ensemble_predictor=ensemble_predictor
)

# Print results
print(f"Average Sharpe: {wf_results['performance_metrics']['sharpe_ratio']:.2f}")
print(f"Consistency: {wf_results['consistency_metrics']['positive_periods']:.2%}")
```

### Report Generation

```python
from src.backtesting.report_generator import ReportGenerator, ReportConfig

# Initialize report generator
config = ReportConfig(title="My Trading Strategy Report")
report_generator = ReportGenerator(config)

# Generate report
report = report_generator.generate_report(
    backtest_results=results,
    performance_analysis=analysis,
    statistical_tests=validation
)

# Save report
with open("trading_report.md", "w") as f:
    f.write(report)
```

## API Reference

### Data Pipeline APIs

#### MarketDataCollector

```python
class MarketDataCollector:
    def __init__(self, alpha_vantage_api_key: str = None)
    def get_yahoo_finance_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame
    def get_alpha_vantage_daily_data(ticker: str, outputsize: str = 'full') -> pd.DataFrame
    def get_alpha_vantage_intraday_data(ticker: str, interval: str = '5min', outputsize: str = 'full') -> pd.DataFrame
    def get_multiple_symbols_data(symbols: List[str], source: str = "yahoo", **kwargs) -> Dict[str, pd.DataFrame]
```

#### IndicatorEngine

```python
class IndicatorEngine:
    def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame
    def calculate_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame
    def calculate_volume_indicators(df: pd.DataFrame) -> pd.DataFrame
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame
```

#### FeatureEngineer

```python
class FeatureEngineer:
    def generate_basic_features(df: pd.DataFrame) -> pd.DataFrame
    def generate_advanced_features(df: pd.DataFrame) -> pd.DataFrame
    def cross_sectional_normalize(df_dict: Dict[str, pd.DataFrame], feature_cols: List[str]) -> Dict[str, pd.DataFrame]
    def time_series_standardize(df: pd.DataFrame, feature_cols: List[str], symbol: str) -> pd.DataFrame
    def prepare_ml_data(df: pd.DataFrame, target_column: str, time_steps: int) -> Dict[str, np.ndarray]
```

### ML Model APIs

#### CNNLSTMModel

```python
class CNNLSTMModel:
    def __init__(self, time_steps: int, n_features: int, output_dim: int = 1)
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Any
    def predict(self, X: np.ndarray) -> np.ndarray
    def save_model(self, path: str = None, filename: str = None)
    def load_model(self, path: str = None, filename: str = None)
```

#### RandomForestModel

```python
class RandomForestModel:
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, random_state: int = 42)
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None
    def predict(self, X: np.ndarray) -> np.ndarray
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]
    def save_model(self, path: str = None, filename: str = None)
    def load_model(self, path: str = None, filename: str = None)
```

#### EnsemblePredictor

```python
class EnsemblePredictor:
    def __init__(self, weights: Dict[str, float] = None)
    def combine_predictions(self, predictions: Dict[str, np.ndarray]) -> np.ndarray
    def generate_signal(self, ensemble_prediction: np.ndarray, threshold: float = 0.55) -> Tuple[np.ndarray, np.ndarray]
```

### Strategy APIs

#### SignalGenerator

```python
class SignalGenerator:
    def __init__(self, confidence_threshold: float = 0.55)
    def generate_ml_signal(self, ml_prediction: float, ml_confidence: float) -> Tuple[int, float]
    def generate_multi_factor_score(self, features: pd.Series, factor_weights: Dict[str, float]) -> float
    def combine_signals(self, ml_signal: int, ml_confidence: float, factor_score: float, ml_weight: float = 0.7, factor_weight: float = 0.3, score_threshold: float = 0.0) -> Tuple[int, float]
```

#### PositionSizer

```python
class PositionSizer:
    def __init__(self, risk_free_rate: float = 0.02, kelly_fraction: float = 0.5)
    def calculate_garch_volatility(self, returns: np.ndarray) -> float
    def calculate_kelly_criterion(self, win_rate: float, avg_win_loss_ratio: float) -> float
    def get_position_size(self, signal_confidence: float, asset_volatility: float, account_balance: float, price: float, risk_per_trade_pct: float = 0.01, win_rate: float = 0.55, avg_win_loss_ratio: float = 1.2) -> Tuple[float, float]
```

#### RiskManager

```python
class RiskManager:
    def __init__(self, max_position_size_pct: float = 0.05, max_portfolio_risk_pct: float = 0.20, stop_loss_pct: float = 0.02, take_profit_pct: float = 0.04, max_daily_loss_pct: float = 0.05, max_open_positions: int = 10)
    def calculate_stop_loss_price(self, entry_price: float, trade_type: str) -> float
    def calculate_take_profit_price(self, entry_price: float, trade_type: str) -> float
    def validate_new_trade(self, symbol: str, trade_type: str, quantity: float, price: float, current_portfolio_value: float, open_positions_count: int) -> Tuple[bool, str]
    def check_position_exit_conditions(self, current_price: float, position_entry_price: float, trade_type: str) -> Tuple[bool, str]
```

### Backtesting APIs

#### BacktestEngine

```python
class BacktestEngine:
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.001, slippage: float = 0.0005)
    def run_backtest(self, data: Dict[str, pd.DataFrame], signal_generator: SignalGenerator, position_sizer: PositionSizer, risk_manager: RiskManager, ensemble_predictor: EnsemblePredictor) -> Dict[str, Any]
```

#### PerformanceAnalyzer

```python
class PerformanceAnalyzer:
    def calculate_performance_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, float]
    def calculate_risk_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, float]
    def analyze_performance(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]
```

#### StatisticalValidator

```python
class StatisticalValidator:
    def validate_performance(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]
    def test_sharpe_ratio_significance(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> Dict[str, Any]
    def test_mean_return_significance(self, returns: np.ndarray) -> Dict[str, Any]
    def bootstrap_analysis(self, returns: np.ndarray, n_bootstrap: int = 1000) -> Dict[str, Any]
```

## Examples

### Example 1: Complete Trading Pipeline

```python
import pandas as pd
import numpy as np
from src.data_pipeline.market_data_collector import MarketDataCollector
from src.data_pipeline.indicator_engine import IndicatorEngine
from src.data_pipeline.feature_engineer import FeatureEngineer
from src.ml_models.model_trainer import ModelTrainer
from src.strategy.signal_generator import SignalGenerator
from src.strategy.position_sizer import PositionSizer
from src.strategy.risk_manager import RiskManager
from src.backtesting.backtest_engine import BacktestEngine
from src.backtesting.performance_analyzer import PerformanceAnalyzer

def run_complete_pipeline():
    # 1. Data Collection
    collector = MarketDataCollector()
    data = collector.get_yahoo_finance_data("AAPL", period="2y", interval="1d")
    
    # 2. Technical Indicators
    engine = IndicatorEngine()
    data = engine.calculate_all_indicators(data)
    
    # 3. Feature Engineering
    fe = FeatureEngineer()
    data = fe.generate_basic_features(data)
    data = fe.generate_advanced_features(data)
    
    # 4. Create Target Variable
    data['next_day_return'] = data['close'].pct_change().shift(-1)
    
    # 5. Train ML Models
    trainer = ModelTrainer()
    feature_cols = ['RSI', 'MACD', 'ATR', 'daily_return']
    X_train, X_val, y_train, y_val, feature_names = trainer.prepare_data_for_training(
        df=data.dropna(),
        feature_cols=feature_cols,
        target_col='next_day_return',
        symbol='AAPL'
    )
    
    trainer.train_all_models(X_train, X_val, y_train, y_val, feature_names)
    
    # 6. Initialize Strategy Components
    signal_generator = SignalGenerator()
    position_sizer = PositionSizer()
    risk_manager = RiskManager()
    
    # 7. Run Backtest
    backtest_engine = BacktestEngine()
    data_dict = {'AAPL': data}
    
    results = backtest_engine.run_backtest(
        data=data_dict,
        signal_generator=signal_generator,
        position_sizer=position_sizer,
        risk_manager=risk_manager,
        ensemble_predictor=trainer.ensemble_predictor
    )
    
    # 8. Analyze Performance
    analyzer = PerformanceAnalyzer()
    analysis = analyzer.analyze_performance(results)
    
    # 9. Print Results
    print("=== TRADING STRATEGY RESULTS ===")
    print(f"Total Return: {results['performance_metrics']['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['performance_metrics']['max_drawdown']:.2%}")
    print(f"Win Rate: {analysis['trade_analysis']['win_rate']:.2%}")
    
    return results, analysis

# Run the complete pipeline
results, analysis = run_complete_pipeline()
```

### Example 2: Multi-Symbol Strategy

```python
def run_multi_symbol_strategy():
    # Symbols to trade
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    # Initialize components
    collector = MarketDataCollector()
    engine = IndicatorEngine()
    fe = FeatureEngineer()
    
    # Collect data for all symbols
    data_dict = {}
    for symbol in symbols:
        print(f"Processing {symbol}...")
        
        # Collect data
        data = collector.get_yahoo_finance_data(symbol, period="1y", interval="1d")
        
        # Calculate indicators
        data = engine.calculate_all_indicators(data)
        
        # Generate features
        data = fe.generate_basic_features(data)
        data = fe.generate_advanced_features(data)
        
        # Create target
        data['next_day_return'] = data['close'].pct_change().shift(-1)
        
        data_dict[symbol] = data.dropna()
    
    # Cross-sectional normalization
    feature_cols = ['RSI', 'MACD', 'ATR']
    normalized_data = fe.cross_sectional_normalize(data_dict, feature_cols)
    
    # Train models for each symbol
    trainer = ModelTrainer()
    models = {}
    
    for symbol in symbols:
        print(f"Training models for {symbol}...")
        
        X_train, X_val, y_train, y_val, feature_names = trainer.prepare_data_for_training(
            df=normalized_data[symbol],
            feature_cols=feature_cols,
            target_col='next_day_return',
            symbol=symbol
        )
        
        trainer.train_all_models(X_train, X_val, y_train, y_val, feature_names)
        models[symbol] = trainer
    
    # Run backtest
    backtest_engine = BacktestEngine()
    signal_generator = SignalGenerator()
    position_sizer = PositionSizer()
    risk_manager = RiskManager()
    
    results = backtest_engine.run_backtest(
        data=normalized_data,
        signal_generator=signal_generator,
        position_sizer=position_sizer,
        risk_manager=risk_manager,
        ensemble_predictor=models[symbols[0]].ensemble_predictor
    )
    
    return results

# Run multi-symbol strategy
multi_results = run_multi_symbol_strategy()
```

### Example 3: Walk-Forward Analysis

```python
def run_walk_forward_analysis():
    from src.backtesting.walk_forward_analyzer import WalkForwardAnalyzer
    
    # Initialize components
    collector = MarketDataCollector()
    engine = IndicatorEngine()
    fe = FeatureEngineer()
    trainer = ModelTrainer()
    
    # Collect data
    data = collector.get_yahoo_finance_data("AAPL", period="3y", interval="1d")
    data = engine.calculate_all_indicators(data)
    data = fe.generate_basic_features(data)
    data = fe.generate_advanced_features(data)
    data['next_day_return'] = data['close'].pct_change().shift(-1)
    
    # Initialize walk-forward analyzer
    wf_analyzer = WalkForwardAnalyzer()
    
    # Run walk-forward analysis
    wf_results = wf_analyzer.run_walk_forward_analysis(
        data={'AAPL': data.dropna()},
        train_window=252,  # 1 year
        test_window=63,    # 3 months
        step_size=21,      # 1 month
        signal_generator=SignalGenerator(),
        position_sizer=PositionSizer(),
        risk_manager=RiskManager(),
        ensemble_predictor=EnsemblePredictor()
    )
    
    # Print results
    print("=== WALK-FORWARD ANALYSIS RESULTS ===")
    print(f"Average Sharpe: {wf_results['performance_metrics']['sharpe_ratio']:.2f}")
    print(f"Average Return: {wf_results['performance_metrics']['annualized_return']:.2%}")
    print(f"Max Drawdown: {wf_results['performance_metrics']['max_drawdown']:.2%}")
    print(f"Consistency: {wf_results['consistency_metrics']['positive_periods']:.2%}")
    
    return wf_results

# Run walk-forward analysis
wf_results = run_walk_forward_analysis()
```

## Troubleshooting

### Common Issues

#### 1. TA-Lib Installation Issues

**Problem**: `ImportError: No module named 'talib'`

**Solution**:
```bash
# On Ubuntu/Debian
sudo apt-get install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib

# On macOS
brew install ta-lib
pip install TA-Lib

# On Windows
# Download pre-compiled wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.19-cp39-cp39-win_amd64.whl
```

#### 2. Data Collection Issues

**Problem**: Yahoo Finance API rate limiting

**Solution**:
```python
import time
import random

def fetch_with_backoff(func, *args, **kwargs):
    for attempt in range(3):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < 2:
                time.sleep(random.uniform(1, 3) * (2 ** attempt))
            else:
                raise e

# Use with data collection
data = fetch_with_backoff(collector.get_yahoo_finance_data, "AAPL", "1y", "1d")
```

#### 3. Memory Issues

**Problem**: Out of memory during model training

**Solution**:
```python
# Reduce batch size
BATCH_SIZE = 16

# Reduce model complexity
LSTM_UNITS = 50
CNN_FILTERS = 32

# Process data in chunks
def process_large_dataset(df, chunk_size=1000):
    for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
        processed_chunk = process_chunk(chunk)
        yield processed_chunk
        del processed_chunk
```

#### 4. Model Training Issues

**Problem**: Model not converging

**Solution**:
```python
# Adjust learning rate
LEARNING_RATE = 0.0001

# Increase early stopping patience
EARLY_STOPPING_PATIENCE = 20

# Add more data
# Ensure sufficient training data (at least 1000 samples)
```

#### 5. Database Connection Issues

**Problem**: Database connection timeout

**Solution**:
```python
import psycopg2
from psycopg2 import OperationalError

def connect_with_retry(max_retries=3):
    for attempt in range(max_retries):
        try:
            return psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                connect_timeout=30
            )
        except OperationalError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e
```

### Debugging Tips

#### 1. Enable Debug Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

#### 2. Use Performance Profiling

```python
import cProfile
import pstats

def profile_function(func, *args, **kwargs):
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
    
    return result

# Profile your function
results = profile_function(run_backtest, data, signal_generator, position_sizer, risk_manager, ensemble_predictor)
```

#### 3. Memory Profiling

```python
import tracemalloc

def profile_memory(func, *args, **kwargs):
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    
    return result

# Profile memory usage
results = profile_memory(run_backtest, data, signal_generator, position_sizer, risk_manager, ensemble_predictor)
```

## FAQ

### Q: What is the minimum amount of data needed to train the models?

A: The models require at least 1000 samples for training. For the CNN+LSTM model, this means at least 1000 days of data (about 4 years of trading data). More data generally leads to better model performance.

### Q: Can I use this system for live trading?

A: The system is designed for backtesting and research. For live trading, you would need to implement additional components such as:
- Real-time data feeds
- Order execution system
- Risk monitoring
- Compliance reporting

### Q: How often should I retrain the models?

A: It depends on market conditions and model performance. Generally, retraining every 3-6 months is recommended, or when model performance degrades significantly.

### Q: What are the system requirements for running this?

A: Minimum requirements:
- Python 3.9+
- 8GB RAM
- 10GB storage
- Stable internet connection

Recommended requirements:
- Python 3.9+
- 16GB RAM
- 50GB SSD storage
- High-speed internet connection

### Q: Can I add custom indicators?

A: Yes, you can extend the `IndicatorEngine` class to add custom indicators. See the technical documentation for details.

### Q: How do I optimize the model parameters?

A: You can use the `ModelTrainer` class with different parameter configurations, or implement hyperparameter optimization using libraries like Optuna or Hyperopt.

### Q: What data sources are supported?

A: Currently supported:
- Yahoo Finance (free)
- Alpha Vantage (free tier available)
- Custom CSV files

You can extend the `MarketDataCollector` class to add additional data sources.

### Q: How do I handle missing data?

A: The system automatically handles missing data by:
- Forward-filling price data
- Interpolating technical indicators
- Dropping rows with insufficient data for ML models

### Q: Can I run this on a cloud platform?

A: Yes, the system can be deployed on cloud platforms like AWS, Google Cloud, or Azure. Use Docker for containerized deployment.

### Q: How do I contribute to the project?

A: Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

For more information, see the contributing guidelines in the repository.
