# Multi-Factor Momentum Trading Strategy with ML Enhancement - Technical Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Data Pipeline](#data-pipeline)
4. [Machine Learning Models](#machine-learning-models)
5. [Trading Strategy](#trading-strategy)
6. [Backtesting Framework](#backtesting-framework)
7. [Risk Management](#risk-management)
8. [Performance Analysis](#performance-analysis)
9. [Deployment](#deployment)
10. [API Reference](#api-reference)
11. [Configuration](#configuration)
12. [Testing](#testing)
13. [Troubleshooting](#troubleshooting)

## System Overview

The Multi-Factor Momentum Trading Strategy with ML Enhancement is a comprehensive quantitative trading system that combines traditional technical analysis with advanced machine learning techniques to generate trading signals and manage risk.

### Key Features

- **Multi-Factor Analysis**: Combines momentum, volatility, and mean-reversion factors
- **Machine Learning Integration**: CNN+LSTM hybrid model and Random Forest ensemble
- **Real-time Data Processing**: Supports multiple data sources and timeframes
- **Advanced Risk Management**: Dynamic position sizing and portfolio-level controls
- **Comprehensive Backtesting**: Walk-forward analysis and statistical validation
- **Scalable Architecture**: Microservices-based design with Docker support

### Performance Targets

- **Sharpe Ratio**: > 1.5
- **Maximum Drawdown**: < 15%
- **Annual Return**: > 20%
- **Win Rate**: > 55%

## Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Data Pipeline │    │   ML Pipeline   │
│                 │    │                 │    │                 │
│ • Yahoo Finance │───▶│ • Collectors    │───▶│ • Preprocessing │
│ • Alpha Vantage │    │ • Indicators     │    │ • Models        │
│ • Custom APIs   │    │ • Features      │    │ • Ensemble      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Trading       │    │   Strategy      │    │   Risk          │
│   Engine        │◀───│   Engine        │◀───│   Management    │
│                 │    │                 │    │                 │
│ • Execution     │    │ • Signal Gen    │    │ • Position Size │
│ • Order Mgmt    │    │ • Factor Score  │    │ • Stop Loss     │
│ • Portfolio     │    │ • Combination   │    │ • Take Profit   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Backtesting   │    │   Performance   │    │   Reporting     │
│   Framework     │    │   Analysis      │    │   System        │
│                 │    │                 │    │                 │
│ • Simulation    │───▶│ • Metrics       │───▶│ • Reports      │
│ • Walk Forward  │    │ • Risk Analysis  │    │ • Charts       │
│ • Monte Carlo   │    │ • Validation    │    │ • Alerts       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Architecture

#### Data Pipeline Components

- **MarketDataCollector**: Fetches data from various sources
- **IndicatorEngine**: Calculates technical indicators using TA-Lib
- **FeatureEngineer**: Creates advanced features and normalizations
- **DataStorage**: Manages data persistence and caching

#### ML Pipeline Components

- **DataPreprocessor**: Prepares data for ML models
- **CNNLSTMModel**: Hybrid CNN+LSTM for time series prediction
- **RandomForestModel**: Random Forest for feature importance
- **EnsemblePredictor**: Combines model predictions
- **ModelTrainer**: Orchestrates model training

#### Strategy Components

- **SignalGenerator**: Generates trading signals
- **PositionSizer**: Calculates optimal position sizes
- **RiskManager**: Implements risk controls

#### Backtesting Components

- **BacktestEngine**: Executes backtesting simulations
- **PerformanceAnalyzer**: Calculates performance metrics
- **StatisticalValidator**: Validates statistical significance
- **WalkForwardAnalyzer**: Implements walk-forward analysis
- **ReportGenerator**: Generates comprehensive reports

## Data Pipeline

### Data Sources

#### Yahoo Finance
- **Coverage**: Global equities, ETFs, indices
- **Timeframes**: 1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
- **Data Points**: OHLCV, dividends, splits
- **Rate Limits**: None (unofficial API)

#### Alpha Vantage
- **Coverage**: US equities, forex, crypto
- **Timeframes**: 1min, 5min, 15min, 30min, 60min, daily
- **Data Points**: OHLCV, adjusted prices
- **Rate Limits**: 5 calls/min, 500 calls/day (free tier)

### Technical Indicators

#### Momentum Indicators
- **RSI (Relative Strength Index)**: 14-period default
- **MACD (Moving Average Convergence Divergence)**: 12,26,9 periods
- **ROC (Rate of Change)**: 10-period default
- **Stochastic Oscillator**: 14,3,3 periods
- **ADX (Average Directional Index)**: 14-period default

#### Volatility Indicators
- **Bollinger Bands**: 20-period, 2 standard deviations
- **ATR (Average True Range)**: 14-period default
- **NATR (Normalized ATR)**: 14-period default
- **True Range**: Daily range calculation

#### Volume Indicators
- **OBV (On-Balance Volume)**: Cumulative volume
- **MFI (Money Flow Index)**: 14-period default
- **Volume Rate**: Volume change rate

### Feature Engineering

#### Basic Features
- **Daily Return**: Percentage change in close price
- **Log Return**: Natural logarithm of price ratio
- **High-Low Range**: Normalized daily range
- **Open-Close Range**: Normalized daily movement

#### Advanced Features
- **Volatility Ratio**: ATR normalized by price
- **RSI-MACD Interaction**: Combined momentum signal
- **Cross-Sectional Normalization**: Relative to universe
- **Time Series Standardization**: Z-score normalization

#### ML Features
- **Lagged Features**: Historical values with various lags
- **Rolling Statistics**: Moving averages and standard deviations
- **Technical Pattern Features**: Support/resistance levels
- **Market Regime Features**: Volatility and trend indicators

## Machine Learning Models

### CNN+LSTM Hybrid Model

#### Architecture
```
Input Layer (60 timesteps × 20 features)
    ↓
Conv1D Layer (64 filters, kernel_size=2)
    ↓
MaxPooling1D (pool_size=2)
    ↓
Dropout (0.3)
    ↓
Reshape for LSTM
    ↓
Bidirectional LSTM (100 units)
    ↓
Dropout (0.3)
    ↓
Dense Layer (1 output)
```

#### Hyperparameters
- **Time Steps**: 60 (configurable)
- **Features**: 20 (configurable)
- **CNN Filters**: 64
- **LSTM Units**: 100
- **Dropout Rate**: 0.3
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)

#### Training Process
1. **Data Preparation**: Create sequences with sliding windows
2. **Feature Scaling**: StandardScaler normalization
3. **Train/Validation Split**: 80/20 split
4. **Model Training**: Adam optimizer with MSE loss
5. **Early Stopping**: Patience of 10 epochs
6. **Model Saving**: HDF5 format with weights

### Random Forest Model

#### Configuration
- **N Estimators**: 100
- **Max Depth**: 10
- **Random State**: 42
- **Features**: Flattened time series data

#### Feature Importance
- **Calculation**: Mean decrease in impurity
- **Ranking**: Top features by importance
- **Usage**: Feature selection and interpretation

### Ensemble Predictor

#### Weighted Voting
- **CNN+LSTM Weight**: 0.6
- **Random Forest Weight**: 0.4
- **Confidence Threshold**: 0.55

#### Signal Generation
- **Buy Signal**: Prediction > threshold
- **Sell Signal**: Prediction < -threshold
- **Hold Signal**: Prediction within threshold

## Trading Strategy

### Multi-Factor Scoring

#### Momentum Factors
- **RSI Score**: Normalized RSI (0-1 scale)
- **MACD Score**: MACD histogram normalized
- **ROC Score**: Rate of change normalized
- **Price Momentum**: Short-term price trend

#### Volatility Factors
- **ATR Score**: Normalized ATR
- **Bollinger Position**: Price position within bands
- **Volatility Regime**: High/low volatility detection

#### Mean Reversion Factors
- **RSI Divergence**: RSI vs price divergence
- **Bollinger Reversion**: Distance from bands
- **Oversold/Overbought**: Extreme RSI levels

### Signal Generation

#### ML Signal
```python
def generate_ml_signal(prediction, confidence):
    if confidence >= threshold:
        if prediction > 0:
            return 1  # Buy
        elif prediction < 0:
            return -1  # Sell
    return 0  # Hold
```

#### Factor Signal
```python
def generate_factor_score(features, weights):
    score = 0.0
    for factor, weight in weights.items():
        score += features[factor] * weight
    return score
```

#### Combined Signal
```python
def combine_signals(ml_signal, ml_confidence, factor_score):
    ml_component = ml_signal * ml_confidence * ml_weight
    factor_component = factor_score * factor_weight
    return ml_component + factor_component
```

### Position Sizing

#### Kelly Criterion
```python
def calculate_kelly_fraction(win_rate, avg_win_loss_ratio):
    return win_rate - (1 - win_rate) / avg_win_loss_ratio
```

#### GARCH Volatility
- **Model**: GARCH(1,1)
- **Forecast**: 1-step ahead conditional variance
- **Annualization**: √252 scaling

#### Position Size Calculation
```python
def get_position_size(signal_confidence, asset_volatility, account_balance):
    kelly_fraction = calculate_kelly_criterion(win_rate, win_loss_ratio)
    optimal_fraction = kelly_fraction * kelly_multiplier * signal_confidence
    volatility_scalar = target_volatility / asset_volatility
    capital_allocated = account_balance * optimal_fraction * volatility_scalar
    return capital_allocated / current_price
```

## Backtesting Framework

### Backtest Engine

#### Simulation Process
1. **Data Loading**: Load historical data for test period
2. **Signal Generation**: Generate signals for each timestamp
3. **Position Sizing**: Calculate position sizes
4. **Risk Validation**: Check risk constraints
5. **Trade Execution**: Execute trades with costs
6. **Portfolio Update**: Update portfolio values
7. **Performance Tracking**: Record metrics

#### Transaction Costs
- **Commission**: 0.1% per trade (configurable)
- **Slippage**: 0.05% per trade (configurable)
- **Bid-Ask Spread**: Market-dependent

#### Portfolio Tracking
- **Cash Balance**: Available cash
- **Positions**: Current holdings
- **Unrealized PnL**: Mark-to-market gains/losses
- **Total Value**: Portfolio net worth

### Walk-Forward Analysis

#### Process
1. **Training Window**: 252 days (1 year)
2. **Testing Window**: 63 days (3 months)
3. **Step Size**: 21 days (1 month)
4. **Rolling Window**: Move forward by step size

#### Validation
- **Out-of-Sample Testing**: Test on unseen data
- **Parameter Stability**: Check parameter consistency
- **Performance Decay**: Monitor performance over time

### Monte Carlo Simulation

#### Process
1. **Bootstrap Sampling**: Random sampling with replacement
2. **Multiple Runs**: 100+ simulations
3. **Statistical Analysis**: Mean, std, percentiles
4. **Confidence Intervals**: 5th and 95th percentiles

### Statistical Validation

#### Performance Tests
- **Sharpe Ratio Test**: t-test for significance
- **Mean Return Test**: t-test against zero
- **Jarque-Bera Test**: Normality of returns
- **Ljung-Box Test**: Serial correlation

#### Risk Tests
- **VaR Backtesting**: Kupiec test
- **CVaR Validation**: Expected shortfall test
- **Drawdown Analysis**: Maximum drawdown test

## Risk Management

### Position-Level Controls

#### Stop Loss
- **Percentage**: 2% from entry price
- **ATR-Based**: 2x ATR from entry
- **Trailing**: Dynamic stop loss

#### Take Profit
- **Percentage**: 4% from entry price
- **Risk-Reward**: 2:1 ratio
- **Trailing**: Dynamic take profit

### Portfolio-Level Controls

#### Position Limits
- **Max Position Size**: 5% of portfolio
- **Max Open Positions**: 10 positions
- **Sector Limits**: 20% per sector

#### Risk Limits
- **Max Drawdown**: 20% of portfolio
- **Daily Loss Limit**: 5% of portfolio
- **VaR Limit**: 2% daily VaR

### Dynamic Risk Management

#### Volatility Targeting
- **Target Volatility**: 15% annualized
- **Position Scaling**: Inverse volatility scaling
- **Rebalancing**: Monthly rebalancing

#### Correlation Management
- **Correlation Limit**: 0.7 between positions
- **Diversification**: Minimum 5 uncorrelated positions
- **Sector Rotation**: Dynamic sector allocation

## Performance Analysis

### Key Metrics

#### Return Metrics
- **Total Return**: Cumulative return over period
- **Annualized Return**: Annualized return rate
- **Excess Return**: Return above benchmark
- **Alpha**: Risk-adjusted excess return

#### Risk Metrics
- **Volatility**: Standard deviation of returns
- **Sharpe Ratio**: Return per unit of risk
- **Sortino Ratio**: Return per unit of downside risk
- **Calmar Ratio**: Return per unit of max drawdown

#### Risk-Adjusted Metrics
- **VaR (95%)**: 95% Value at Risk
- **CVaR (95%)**: 95% Conditional Value at Risk
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Downside Deviation**: Standard deviation of negative returns

### Performance Attribution

#### Factor Attribution
- **Momentum Contribution**: Return from momentum factors
- **Volatility Contribution**: Return from volatility factors
- **Mean Reversion Contribution**: Return from mean reversion factors
- **ML Contribution**: Return from ML predictions

#### Risk Attribution
- **Systematic Risk**: Market risk exposure
- **Idiosyncratic Risk**: Stock-specific risk
- **Factor Risk**: Factor exposure risk
- **Model Risk**: ML model risk

### Benchmark Comparison

#### Benchmarks
- **S&P 500**: Market benchmark
- **Risk-Free Rate**: Treasury bill rate
- **Custom Benchmark**: Sector-specific benchmark

#### Comparison Metrics
- **Information Ratio**: Active return per unit of tracking error
- **Tracking Error**: Standard deviation of excess returns
- **Beta**: Market sensitivity
- **Correlation**: Return correlation with benchmark

## Deployment

### Docker Deployment

#### Services
- **Database**: PostgreSQL with TimescaleDB
- **Cache**: Redis for data caching
- **Application**: Main trading application
- **Jupyter**: Data analysis environment

#### Configuration
```yaml
version: '3.8'
services:
  db:
    image: timescale/timescaledb-ha:pg14-latest
    environment:
      POSTGRES_DB: trading_db
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
  
  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
  
  app:
    build: .
    environment:
      DB_HOST: db
      REDIS_HOST: redis
    ports:
      - "8000:8000"
```

### Environment Setup

#### Required Environment Variables
```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_db
DB_USER=user
DB_PASSWORD=password

# APIs
ALPHA_VANTAGE_API_KEY=your_api_key

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
```

#### Installation Steps
1. **Clone Repository**: `git clone <repository_url>`
2. **Install Dependencies**: `pip install -r requirements.txt`
3. **Setup Database**: `docker-compose up -d db redis`
4. **Initialize Database**: `python src/data_pipeline/data_storage.py`
5. **Run Tests**: `python tests/test_runner.py --category all`
6. **Start Application**: `docker-compose up`

### Monitoring

#### Health Checks
- **Database Connection**: PostgreSQL connectivity
- **Redis Connection**: Redis connectivity
- **API Endpoints**: External API availability
- **Model Performance**: ML model accuracy

#### Logging
- **Application Logs**: Trading decisions and errors
- **Performance Logs**: Execution times and metrics
- **Error Logs**: Exception handling and debugging
- **Audit Logs**: Trade execution and modifications

## API Reference

### Data Pipeline APIs

#### MarketDataCollector
```python
class MarketDataCollector:
    def get_yahoo_finance_data(ticker: str, period: str, interval: str) -> pd.DataFrame
    def get_alpha_vantage_daily_data(ticker: str, outputsize: str) -> pd.DataFrame
    def get_alpha_vantage_intraday_data(ticker: str, interval: str, outputsize: str) -> pd.DataFrame
    def get_multiple_symbols_data(symbols: List[str], source: str, **kwargs) -> Dict[str, pd.DataFrame]
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
    def __init__(time_steps: int, n_features: int, output_dim: int = 1)
    def train(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Any
    def predict(X: np.ndarray) -> np.ndarray
    def save_model(path: str, filename: str)
    def load_model(path: str, filename: str)
```

#### RandomForestModel
```python
class RandomForestModel:
    def __init__(n_estimators: int = 100, max_depth: int = 10, random_state: int = 42)
    def train(X_train: np.ndarray, y_train: np.ndarray) -> None
    def predict(X: np.ndarray) -> np.ndarray
    def get_feature_importance(feature_names: List[str]) -> Dict[str, float]
    def save_model(path: str, filename: str)
    def load_model(path: str, filename: str)
```

#### EnsemblePredictor
```python
class EnsemblePredictor:
    def __init__(weights: Dict[str, float] = None)
    def combine_predictions(predictions: Dict[str, np.ndarray]) -> np.ndarray
    def generate_signal(ensemble_prediction: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]
```

### Strategy APIs

#### SignalGenerator
```python
class SignalGenerator:
    def __init__(confidence_threshold: float = 0.55)
    def generate_ml_signal(ml_prediction: float, ml_confidence: float) -> Tuple[int, float]
    def generate_multi_factor_score(features: pd.Series, factor_weights: Dict[str, float]) -> float
    def combine_signals(ml_signal: int, ml_confidence: float, factor_score: float) -> Tuple[int, float]
```

#### PositionSizer
```python
class PositionSizer:
    def __init__(risk_free_rate: float = 0.02, kelly_fraction: float = 0.5)
    def calculate_garch_volatility(returns: np.ndarray) -> float
    def calculate_kelly_criterion(win_rate: float, avg_win_loss_ratio: float) -> float
    def get_position_size(signal_confidence: float, asset_volatility: float, account_balance: float, price: float) -> Tuple[float, float]
```

#### RiskManager
```python
class RiskManager:
    def __init__(max_position_size_pct: float = 0.05, max_portfolio_risk_pct: float = 0.20, stop_loss_pct: float = 0.02, take_profit_pct: float = 0.04)
    def calculate_stop_loss_price(entry_price: float, trade_type: str) -> float
    def calculate_take_profit_price(entry_price: float, trade_type: str) -> float
    def validate_new_trade(symbol: str, trade_type: str, quantity: float, price: float, current_portfolio_value: float, open_positions_count: int) -> Tuple[bool, str]
    def check_position_exit_conditions(current_price: float, position_entry_price: float, trade_type: str) -> Tuple[bool, str]
```

### Backtesting APIs

#### BacktestEngine
```python
class BacktestEngine:
    def __init__(initial_capital: float = 100000.0, commission: float = 0.001, slippage: float = 0.0005)
    def run_backtest(data: Dict[str, pd.DataFrame], signal_generator: SignalGenerator, position_sizer: PositionSizer, risk_manager: RiskManager, ensemble_predictor: EnsemblePredictor) -> Dict[str, Any]
```

#### PerformanceAnalyzer
```python
class PerformanceAnalyzer:
    def calculate_performance_metrics(backtest_results: Dict[str, Any]) -> Dict[str, float]
    def calculate_risk_metrics(backtest_results: Dict[str, Any]) -> Dict[str, float]
    def analyze_performance(backtest_results: Dict[str, Any]) -> Dict[str, Any]
```

#### StatisticalValidator
```python
class StatisticalValidator:
    def validate_performance(backtest_results: Dict[str, Any]) -> Dict[str, Any]
    def test_sharpe_ratio_significance(returns: np.ndarray, risk_free_rate: float = 0.02) -> Dict[str, Any]
    def test_mean_return_significance(returns: np.ndarray) -> Dict[str, Any]
    def bootstrap_analysis(returns: np.ndarray, n_bootstrap: int = 1000) -> Dict[str, Any]
```

## Configuration

### Model Configuration

#### CNN+LSTM Parameters
```python
class ModelConfig:
    CNN_FILTERS = 64
    CNN_KERNEL_SIZE = 2
    LSTM_UNITS = 100
    DROPOUT_RATE = 0.3
    TIME_STEPS = 60
    N_FEATURES = 20
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10
```

#### Random Forest Parameters
```python
class ModelConfig:
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10
    RF_RANDOM_STATE = 42
```

#### Ensemble Parameters
```python
class ModelConfig:
    ENSEMBLE_WEIGHTS = {"cnn_lstm": 0.6, "random_forest": 0.4}
    CONFIDENCE_THRESHOLD = 0.55
```

### Database Configuration

#### PostgreSQL/TimescaleDB
```python
class DatabaseConfig:
    DB_HOST = "localhost"
    DB_PORT = 5432
    DB_NAME = "trading_db"
    DB_USER = "user"
    DB_PASSWORD = "password"
    TIMESCALEDB_ENABLED = True
```

#### Redis Configuration
```python
class RedisConfig:
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    REDIS_PASSWORD = None
    REDIS_EXPIRY = 3600
```

### API Configuration

#### Alpha Vantage
```python
class APIConfig:
    ALPHA_VANTAGE_API_KEY = "your_api_key"
    ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
    ALPHA_VANTAGE_RATE_LIMIT = 5  # calls per minute
```

#### Yahoo Finance
```python
class APIConfig:
    YAHOO_FINANCE_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
    YAHOO_FINANCE_RATE_LIMIT = None  # No official limit
```

## Testing

### Test Structure

#### Unit Tests
- **Location**: `tests/unit_tests/`
- **Coverage**: Individual component testing
- **Mocking**: External dependencies mocked
- **Assertions**: Component behavior validation

#### Integration Tests
- **Location**: `tests/integration_tests/`
- **Coverage**: Component interaction testing
- **Data Flow**: End-to-end data flow validation
- **Dependencies**: Real component interactions

#### Performance Tests
- **Location**: `tests/performance_tests/`
- **Coverage**: Performance and scalability testing
- **Metrics**: Execution time and memory usage
- **Thresholds**: Performance benchmarks

### Running Tests

#### Command Line
```bash
# Run all tests
python tests/test_runner.py --category all

# Run specific category
python tests/test_runner.py --category unit
python tests/test_runner.py --category integration
python tests/test_runner.py --category performance

# Save results
python tests/test_runner.py --category all --save
```

#### Test Configuration
```python
class TestConfig:
    PERFORMANCE_THRESHOLDS = {
        "indicator_calculation": 10.0,
        "feature_generation": 5.0,
        "ml_training": 30.0,
        "backtest_execution": 30.0
    }
    
    MEMORY_THRESHOLDS = {
        "indicator_calculation": 500,
        "feature_generation": 200,
        "ml_model_creation": 300
    }
```

### Test Coverage

#### Target Coverage
- **Unit Tests**: > 90% line coverage
- **Integration Tests**: > 80% integration coverage
- **Performance Tests**: All critical paths tested

#### Coverage Reporting
```bash
# Generate coverage report
coverage run -m pytest tests/
coverage report
coverage html
```

## Troubleshooting

### Common Issues

#### Data Collection Issues

**Problem**: Yahoo Finance API rate limiting
**Solution**: Implement exponential backoff and caching
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
```

**Problem**: Alpha Vantage API key issues
**Solution**: Validate API key and handle errors
```python
def validate_api_key(api_key):
    if not api_key or api_key == "YOUR_ALPHA_VANTAGE_API_KEY":
        raise ValueError("Invalid Alpha Vantage API key")
    return True
```

#### ML Model Issues

**Problem**: Model training convergence issues
**Solution**: Adjust learning rate and early stopping
```python
# Reduce learning rate
LEARNING_RATE = 0.0001

# Increase early stopping patience
EARLY_STOPPING_PATIENCE = 20
```

**Problem**: Memory issues during training
**Solution**: Reduce batch size and model complexity
```python
# Smaller batch size
BATCH_SIZE = 16

# Fewer LSTM units
LSTM_UNITS = 50
```

#### Database Issues

**Problem**: Connection timeout
**Solution**: Increase connection timeout and retry logic
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

**Problem**: TimescaleDB extension not available
**Solution**: Check TimescaleDB installation
```sql
-- Check if TimescaleDB is installed
SELECT * FROM pg_extension WHERE extname = 'timescaledb';

-- Install TimescaleDB if not available
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
```

#### Performance Issues

**Problem**: Slow indicator calculations
**Solution**: Use vectorized operations and caching
```python
# Use pandas vectorized operations
df['RSI'] = ta.RSI(df['close'], timeperiod=14)

# Cache calculated indicators
@lru_cache(maxsize=1000)
def cached_rsi(prices_tuple, period):
    return ta.RSI(np.array(prices_tuple), timeperiod=period)
```

**Problem**: Memory usage too high
**Solution**: Process data in chunks and clean up
```python
def process_large_dataset(df, chunk_size=1000):
    for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
        processed_chunk = process_chunk(chunk)
        yield processed_chunk
        del processed_chunk  # Clean up memory
```

### Debugging

#### Logging Configuration
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_strategy.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

#### Performance Profiling
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
```

#### Memory Profiling
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
```

### Support

#### Documentation
- **README**: Project overview and setup
- **API Reference**: Detailed API documentation
- **Examples**: Code examples and tutorials
- **Troubleshooting**: Common issues and solutions

#### Community
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Questions and community support
- **Wiki**: Additional documentation and guides

#### Professional Support
- **Consulting**: Custom implementation and optimization
- **Training**: Team training and knowledge transfer
- **Maintenance**: Ongoing support and updates
