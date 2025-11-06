# Speech Emotion Recognition (SER) 🎤😊

Deep learning system to detect emotions (happy, sad, angry, neutral, calm, fearful, disgust, surprised) from speech audio clips.

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd speech-emotion
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset

**Option A: RAVDESS (Recommended)**
- Download from: https://zenodo.org/record/1188976
- Extract to: `data/RAVDESS/`

**Option B: TESS (Alternative)**
- Download from: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
- Extract to: `data/TESS/`

### 3. Train the Model

```bash
python -m src.train --data_dir data/RAVDESS --epochs 25 --batch_size 32
```

**Training arguments:**
- `--data_dir`: Path to dataset (default: `data/RAVDESS`)
- `--epochs`: Number of training epochs (default: 25)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--feature_type`: Feature type - `mfcc` or `logmel` (default: `mfcc`)

**Output:**
- Model checkpoints saved to `models/`
- Training curves saved to `results/training_curves.png`
- Metrics saved to `results/training_metrics.json`

### 4. Evaluate the Model

```bash
python -m src.eval --data_dir data/RAVDESS --checkpoint models/best_model.pth
```

**Evaluation arguments:**
- `--checkpoint`: Path to model checkpoint (default: `models/best_model.pth`)
- `--split`: Dataset split to evaluate - `train`, `val`, or `test` (default: `test`)

**Output:**
- Confusion matrix saved to `results/test_confusion_matrix.png`
- Classification report saved to `results/test_classification_report.txt`
- Metrics saved to `results/test_metrics.json`

### 5. Run API Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

Access the API at: http://localhost:8080

**API Endpoints:**
- `GET /` - API info and endpoints
- `GET /health` - Health check and model status
- `POST /predict` - Upload audio file for emotion prediction

**Test the API with curl:**
```bash
curl -X POST "http://localhost:8080/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_audio.wav"
```

### 6. Docker Deployment

```bash
# Build image
docker build -t emotion-api .

# Run container
docker run -p 8080:8080 emotion-api
```

## 📊 Architecture

**Pipeline:**
```
Audio Input → librosa → MFCC/Mel Features → PyTorch CNN → Softmax → Emotion Label
```

**Model:**
- 3 Convolutional blocks with BatchNorm + MaxPool + Dropout
- Adaptive pooling for variable-length inputs
- 3 Fully connected layers with dropout
- ~2-3M trainable parameters

**Features:**
- MFCC: 40 coefficients
- Mel Spectrogram: 64 mel bins
- Sample rate: 16kHz
- Audio duration: 3 seconds (padded/trimmed)

## 📁 Project Structure

```
speech-emotion/
├── data/               # Datasets (RAVDESS, TESS)
├── features/           # Cached feature files
├── models/             # Model checkpoints and artifacts
├── results/            # Training curves, confusion matrices, metrics
├── src/
│   ├── __init__.py
│   ├── dataset.py      # PyTorch dataset
│   ├── features.py     # Audio feature extraction
│   ├── models.py       # CNN architecture
│   ├── train.py        # Training script
│   ├── eval.py         # Evaluation script
│   └── utils.py        # Helper functions
├── app.py              # FastAPI inference API
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker configuration
└── README.md
```

## 🎯 Expected Performance

On RAVDESS dataset:
- **Training accuracy**: 85-95%
- **Validation accuracy**: 60-75%
- **Test accuracy**: 55-70%

Performance depends on:
- Dataset size and quality
- Number of training epochs
- Hyperparameter tuning
- Feature type (MFCC vs Mel)

## 🔧 Advanced Usage

**Train with custom parameters:**
```bash
python -m src.train \
  --data_dir data/RAVDESS \
  --epochs 50 \
  --batch_size 64 \
  --lr 0.0005 \
  --feature_type logmel
```

**Evaluate on validation set:**
```bash
python -m src.eval \
  --checkpoint models/best_model.pth \
  --split val
```

**Resume training from checkpoint:**
```python
# Modify src/train.py to load checkpoint in optimizer
```

## 📦 Dependencies

- Python 3.10+
- PyTorch 2.4.0
- librosa 0.10.2
- FastAPI 0.111.0
- scikit-learn 1.5.0
- numpy, pandas, matplotlib

See `requirements.txt` for full list.

## 🎓 Use Cases

- **Call centers**: Analyze customer emotion in real-time
- **Mental health**: Monitor emotional state in therapy sessions
- **Virtual assistants**: Respond appropriately to user emotions
- **Gaming**: Adaptive gameplay based on player emotion
- **Market research**: Analyze emotional responses in focus groups

## 🚨 Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'src'`
- **Fix**: Run commands from the `speech-emotion` directory

**Issue**: `FileNotFoundError: data/RAVDESS`
- **Fix**: Download and extract dataset to correct location

**Issue**: API returns "Model not loaded"
- **Fix**: Train model first: `python -m src.train`

**Issue**: CUDA out of memory
- **Fix**: Reduce batch size: `--batch_size 16`

**Issue**: Low accuracy
- **Fix**: Train for more epochs, use data augmentation, tune hyperparameters

## 📝 License

Educational project - use responsibly with proper attribution.

## 🤝 Contributing

This is a learning project. Fork it, experiment, and build on it!

## 🎉 Next Steps

1. ✅ Train baseline model
2. ⬜ Add data augmentation (pitch shift, time stretch, noise)
3. ⬜ Try different architectures (LSTM, Transformer)
4. ⬜ Implement real-time streaming inference
5. ⬜ Deploy to cloud (AWS/GCP/Azure)
6. ⬜ Build web frontend
7. ⬜ Add multilingual support

---

**Built for AI Engineer Readiness Projects** 🚀
