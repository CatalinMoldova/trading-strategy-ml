# GitHub and Google Colab Setup Guide

## Overview
This guide will help you set up your Trading Strategy ML project on GitHub and run it on Google Colab with GPU support.

## Prerequisites
- GitHub account
- Google account (for Colab)
- Git installed on your local machine

## Step 1: Set Up GitHub Repository

### 1.1 Create a New Repository on GitHub
1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Name it `trading-strategy-ml`
5. Add a description: "Multi-Factor Momentum Trading Strategy with ML Enhancement"
6. Make it **Public** (required for free Colab access)
7. Don't initialize with README (we already have one)
8. Click "Create repository"

### 1.2 Push Your Local Project to GitHub

Open terminal/command prompt and run these commands:

```bash
# Navigate to your project directory
cd /Users/catalinbotezat/Documents/Projects/Trading/trading_strategy_ml

# Initialize git repository (if not already done)
git init

# Add all files to staging
git add .

# Create initial commit
git commit -m "Initial commit: Trading Strategy ML project"

# Add GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/trading-strategy-ml.git

# Push to GitHub
git push -u origin main
```

### 1.3 Verify Upload
- Go to your GitHub repository
- Verify all files are uploaded correctly
- Check that the `.gitignore` file is working (large model files should be excluded)

## Step 2: Set Up Google Colab

### 2.1 Access Your Repository in Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "File" → "Open notebook"
3. Select "GitHub" tab
4. Enter your repository URL: `https://github.com/YOUR_USERNAME/trading-strategy-ml`
5. Select the `Trading_Strategy_ML_Colab.ipynb` notebook

### 2.2 Enable GPU Runtime
1. In Colab, click "Runtime" → "Change runtime type"
2. Set "Hardware accelerator" to **GPU**
3. Choose "T4" or "V100" (T4 is usually available for free)
4. Click "Save"

### 2.3 Run the Setup Cells
1. Execute the first cell to check GPU availability
2. Run the installation cell to install required packages
3. Execute the repository cloning cell (update the URL with your username)

## Step 3: Configure API Keys (Optional)

### 3.1 Alpha Vantage API Key
If you want to use Alpha Vantage for additional data:

1. Go to [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Get a free API key
3. In Colab, add this cell:

```python
# Set your API key
import os
os.environ['ALPHA_VANTAGE_API_KEY'] = 'YOUR_API_KEY_HERE'
```

### 3.2 Other API Keys
Add any other API keys you need in the same way.

## Step 4: Run the Complete Pipeline

### 4.1 Execute All Cells
1. Run all cells in sequence
2. Monitor the output for any errors
3. The process will:
   - Collect market data
   - Train ML models with GPU acceleration
   - Generate trading signals
   - Run backtesting
   - Analyze performance

### 4.2 Expected Runtime
- **Data Collection**: 2-5 minutes
- **Model Training**: 10-30 minutes (depending on GPU)
- **Backtesting**: 5-10 minutes
- **Total**: 20-45 minutes

## Step 5: Save Your Work

### 5.1 Save Models
Add this cell to save your trained models:

```python
# Save models
cnn_lstm.save_model('cnn_lstm_model.h5')
rf_model.save_model('random_forest_model.pkl')

# Save results
import pickle
with open('backtest_results.pkl', 'wb') as f:
    pickle.dump(backtest_results, f)

print("Models and results saved!")
```

### 5.2 Download Results
1. Right-click on saved files in Colab's file browser
2. Select "Download"
3. Or use this code to download:

```python
from google.colab import files
files.download('cnn_lstm_model.h5')
files.download('random_forest_model.pkl')
files.download('backtest_results.pkl')
```

## Step 6: Push Updates Back to GitHub

### 6.1 Commit Changes
If you make improvements in Colab:

```bash
# In your local repository
git add .
git commit -m "Updated models and results"
git push origin main
```

### 6.2 Sync with Colab
In Colab, pull the latest changes:

```python
!git pull origin main
```

## Troubleshooting

### Common Issues

#### 1. GPU Not Available
- Check if you're using the free tier (GPU access is limited)
- Try refreshing the page and requesting GPU again
- Consider using Colab Pro for more reliable GPU access

#### 2. Package Installation Errors
- Some packages might conflict with Colab's environment
- Try installing packages one by one
- Use `%pip install` instead of `!pip install`

#### 3. Memory Issues
- Reduce batch size in model training
- Use smaller datasets for testing
- Clear variables: `%reset -f`

#### 4. Repository Access Issues
- Ensure your repository is public
- Check the repository URL is correct
- Verify you have push access

### Performance Tips

#### 1. Optimize for Colab
- Use smaller datasets for initial testing
- Implement early stopping in model training
- Use mixed precision training

#### 2. Memory Management
```python
# Clear memory
import gc
gc.collect()

# Clear TensorFlow session
tf.keras.backend.clear_session()
```

#### 3. Faster Training
```python
# Use mixed precision
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')
```

## Advanced Usage

### 1. Custom Datasets
- Upload your own data files to Colab
- Modify the data collection code
- Use different symbols or timeframes

### 2. Model Improvements
- Experiment with different architectures
- Try different hyperparameters
- Implement additional features

### 3. Strategy Enhancements
- Add more technical indicators
- Implement different risk management rules
- Test on different asset classes

## Next Steps

1. **Experiment**: Try different models and parameters
2. **Optimize**: Fine-tune hyperparameters
3. **Scale**: Test on more symbols and longer timeframes
4. **Deploy**: Consider deploying to cloud platforms
5. **Monitor**: Set up real-time monitoring

## Resources

- [Google Colab Documentation](https://colab.research.google.com/notebooks/intro.ipynb)
- [TensorFlow GPU Guide](https://www.tensorflow.org/guide/gpu)
- [GitHub Documentation](https://docs.github.com/)
- [Trading Strategy ML Documentation](./docs/)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the error messages carefully
3. Search for solutions online
4. Consider asking on relevant forums

---

**Happy Trading! 🚀📈**
