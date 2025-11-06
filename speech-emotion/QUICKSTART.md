# 🚀 QUICKSTART - Get Running in 5 Minutes

## Option 1: Automated Setup (Linux/Mac)

```bash
# 1. Run setup script
bash setup.sh

# 2. Download dataset
# Go to: https://zenodo.org/record/1188976
# Download "Audio_Speech_Actors_01-24.zip"
# Extract to: data/RAVDESS/

# 3. Train model (15-30 min on CPU, 5-10 min on GPU)
source .venv/bin/activate
python -m src.train --data_dir data/RAVDESS --epochs 10

# 4. Test the model
python -m src.eval --data_dir data/RAVDESS

# 5. Run API
uvicorn app:app --reload
# Visit: http://localhost:8000
```

## Option 2: Manual Setup (Windows/All)

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate (Windows)
.venv\Scripts\activate
# Or Mac/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test installation
python test_setup.py

# 5. Download dataset (see Option 1, step 2)

# 6. Train model
python -m src.train --data_dir data/RAVDESS --epochs 10

# 7. Run API
uvicorn app:app --reload
```

## 🎯 Quick Commands

```bash
# Fast training (5 epochs)
python -m src.train --epochs 5 --batch_size 64

# Evaluate
python -m src.eval

# Run API
uvicorn app:app --host 0.0.0.0 --port 8080

# Test API
curl -X POST "http://localhost:8080/predict" -F "file=@audio.wav"

# Docker
docker build -t emotion-api .
docker run -p 8080:8080 emotion-api
```

## 📁 Where is Everything?

- `models/best_model.pth` - Your trained model
- `results/` - Training curves, confusion matrix
- `data/RAVDESS/` - Put your dataset here
- `app.py` - FastAPI server
- `src/train.py` - Training script
- `src/eval.py` - Evaluation script

## ❓ Troubleshooting

**No dataset?**
```bash
# Download from: https://zenodo.org/record/1188976
# Should see files like: data/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav
```

**Training too slow?**
```bash
# Reduce epochs and batch size
python -m src.train --epochs 5 --batch_size 16
```

**API says "Model not loaded"?**
```bash
# Train first!
python -m src.train --epochs 5
```

**Import errors?**
```bash
pip install -r requirements.txt
```

## 🎉 You're Ready!

After training, you have a working Speech Emotion Recognition system that can:
- ✅ Classify 8 emotions from audio
- ✅ Run as a REST API
- ✅ Deploy via Docker
- ✅ Show in your portfolio

**Add to your resume:**
- Built end-to-end ML pipeline with PyTorch
- Deployed FastAPI inference server
- Containerized with Docker
- Audio feature engineering with librosa
- Achieved XX% accuracy on RAVDESS dataset
