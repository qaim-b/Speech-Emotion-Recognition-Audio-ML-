#!/bin/bash

echo "===================================================================="
echo "Speech Emotion Recognition - Quick Setup"
echo "===================================================================="

# Check Python version
echo -e "\n1. Checking Python version..."
python3 --version

# Create virtual environment
echo -e "\n2. Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo -e "\n3. Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo -e "\n4. Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo -e "\n5. Installing dependencies..."
pip install -r requirements.txt

# Run test
echo -e "\n6. Running installation test..."
python test_setup.py

echo -e "\n===================================================================="
echo "Setup complete! 🎉"
echo "===================================================================="
echo ""
echo "Next steps:"
echo "1. Activate environment: source .venv/bin/activate"
echo "2. Download dataset: https://zenodo.org/record/1188976"
echo "3. Extract to: data/RAVDESS/"
echo "4. Train model: python -m src.train --data_dir data/RAVDESS"
echo ""
echo "===================================================================="
