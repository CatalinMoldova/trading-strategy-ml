#!/bin/bash

# GitHub Setup Script for Trading Strategy ML
# This script helps you push your project to GitHub

echo "🚀 Setting up Trading Strategy ML on GitHub..."

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
fi

# Add all files
echo "📝 Adding files to git..."
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "✅ No changes to commit"
else
    echo "💾 Committing changes..."
    git commit -m "Update trading strategy ML project"
fi

# Check if remote origin exists
if git remote get-url origin >/dev/null 2>&1; then
    echo "🔄 Pushing to existing repository..."
    git push origin main
else
    echo "⚠️  No remote repository configured."
    echo "Please run these commands manually:"
    echo ""
    echo "git remote add origin https://github.com/YOUR_USERNAME/trading-strategy-ml.git"
    echo "git push -u origin main"
    echo ""
    echo "Replace YOUR_USERNAME with your actual GitHub username"
fi

echo "✅ GitHub setup complete!"
echo ""
echo "Next steps:"
echo "1. Go to https://github.com/YOUR_USERNAME/trading-strategy-ml"
echo "2. Copy the repository URL"
echo "3. Open Google Colab"
echo "4. Use the Trading_Strategy_ML_Colab.ipynb notebook"
echo "5. Update the repository URL in the notebook"
echo "6. Run all cells to start training with GPU!"
