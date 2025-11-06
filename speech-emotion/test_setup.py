#!/usr/bin/env python3
"""Quick test to verify installation and generate sample data"""

import os
import sys
import numpy as np
import soundfile as sf

def generate_test_audio():
    """Generate a simple test audio file"""
    print("Generating test audio file...")
    
    # Create test data directory
    os.makedirs('data/test_audio', exist_ok=True)
    
    # Generate 3 seconds of random audio (simulating speech)
    sr = 16000
    duration = 3
    samples = int(sr * duration)
    
    # Create a simple tone with some noise (not real speech, just for testing)
    t = np.linspace(0, duration, samples)
    frequency = 220  # A3 note
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    audio += 0.1 * np.random.randn(samples)  # Add noise
    
    # Save test audio
    test_path = 'data/test_audio/test_sample.wav'
    sf.write(test_path, audio, sr)
    print(f"✓ Test audio saved to: {test_path}")
    
    return test_path

def test_imports():
    """Test if all required packages are installed"""
    print("\nTesting package imports...")
    
    packages = {
        'numpy': 'numpy',
        'scipy': 'scipy', 
        'pandas': 'pandas',
        'librosa': 'librosa',
        'torch': 'torch',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'soundfile': 'soundfile'
    }
    
    failed = []
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All packages installed successfully!")
    return True

def test_project_structure():
    """Verify project structure"""
    print("\nChecking project structure...")
    
    required_dirs = ['data', 'features', 'models', 'results', 'src']
    required_files = [
        'requirements.txt',
        'README.md',
        'app.py',
        'Dockerfile',
        'src/__init__.py',
        'src/features.py',
        'src/dataset.py',
        'src/models.py',
        'src/train.py',
        'src/eval.py',
        'src/utils.py'
    ]
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✓ {directory}/")
        else:
            print(f"✗ {directory}/ - MISSING")
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")

def test_feature_extraction():
    """Test feature extraction on generated audio"""
    print("\nTesting feature extraction...")
    
    try:
        from src.features import FeatureConfig, extract_from_path
        
        # Generate test audio
        test_path = generate_test_audio()
        
        # Extract features
        cfg = FeatureConfig()
        features = extract_from_path(test_path, cfg)
        
        print(f"✓ MFCC shape: {features['mfcc'].shape}")
        print(f"✓ LogMel shape: {features['logmel'].shape}")
        print("✓ Feature extraction working!")
        
        return True
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        return False

def main():
    print("="*60)
    print("Speech Emotion Recognition - Installation Test")
    print("="*60)
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    # Test project structure
    test_project_structure()
    
    # Test feature extraction
    test_feature_extraction()
    
    print("\n" + "="*60)
    print("✓ Installation test complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Download RAVDESS dataset: https://zenodo.org/record/1188976")
    print("2. Extract to: data/RAVDESS/")
    print("3. Train model: python -m src.train --data_dir data/RAVDESS")
    print("4. Run API: uvicorn app:app --host 0.0.0.0 --port 8080")
    print("="*60)

if __name__ == '__main__':
    main()
