**🎤 Speech Emotion Recognition System**
Deep learning system that classifies emotions from speech audio with 83.7% accuracy using PyTorch and FastAPI.

🎯 Project Overview
This project implements an end-to-end machine learning pipeline for detecting emotions (angry, calm, disgust, fearful, happy, neutral, sad, surprised) from speech audio clips using convolutional neural networks.
Key Achievements

✅ 83.7% test accuracy on 8-class emotion classification
✅ 2.6M parameter CNN architecture
✅ 2,880 audio samples processed from RAVDESS dataset
✅ Production-ready REST API with FastAPI
✅ Docker containerization for deployment

📊 Model Performance
Accuracy by Emotion Class
EmotionPrecisionRecallF1-ScoreAngry100.0%87.5%93.3%Calm84.6%68.8%75.9%Disgust92.9%83.9%88.1%Fearful87.2%82.9%85.0%Happy78.0%92.9%84.8%Neutral100.0%66.7%80.0%Sad60.4%88.9%71.9%Surprised95.1%90.7%92.9%
Overall Test Accuracy: 83.68%
Training Progress
The model shows consistent learning across 10 epochs:

Training accuracy: 31.7% → 79.8%
Validation accuracy: 27.4% → 78.8%
Training loss: 1.83 → 0.56

Note: Validation curves show the model generalizes well with minimal overfitting.
🏗️ Architecture
Pipeline Overview
Audio Input → librosa → Feature Extraction → CNN → Softmax → Emotion Label
              (MFCC/Mel Spectrograms)
CNN Architecture

Input: MFCC features (40 coefficients, 3-second clips @ 16kHz)
Conv Block 1: 64 filters, BatchNorm, ReLU, MaxPool, Dropout(0.3)
Conv Block 2: 128 filters, BatchNorm, ReLU, MaxPool, Dropout(0.3)
Conv Block 3: 256 filters, BatchNorm, ReLU, MaxPool, Dropout(0.4)
FC Layers: 25644 → 512 → 256 → 8 classes
Total Parameters: 2,603,144

Feature Engineering

Sample rate: 16,000 Hz
Audio duration: 3 seconds (padded/trimmed)
MFCC coefficients: 40
Mel bins: 64
Hop length: 512
FFT window: 1024

🚀 Quick Start
Prerequisites

Python 3.10+
pip
Virtual environment (recommended)

Installation
bash# Clone repository
git clone https://github.com/YOUR_USERNAME/speech-emotion-recognition.git
cd speech-emotion-recognition

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Download Dataset
Download RAVDESS dataset from Zenodo:

File: Audio_Speech_Actors_01-24.zip (208 MB)
Extract to: data/RAVDESS/

Train Model
bashpython -m src.train --data_dir data/RAVDESS --epochs 10 --batch_size 32
Training time:

CPU: ~15-20 minutes
GPU: ~3-5 minutes

Evaluate Model
bashpython -m src.eval --data_dir data/RAVDESS --checkpoint models/best_model.pth
Output:

Confusion matrix: results/test_confusion_matrix.png
Classification report: results/test_classification_report.txt
Metrics: results/test_metrics.json

Run API Server
bashuvicorn app:app --host 0.0.0.0 --port 8000 --reload
Access API:

Interactive docs: http://localhost:8000/docs
Health check: http://localhost:8000/health

Test API
bashcurl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_audio.wav"
Example Response:
json{
  "emotion": "happy",
  "confidence": 0.929,
  "all_emotions": {
    "angry": 0.015,
    "calm": 0.012,
    "disgust": 0.008,
    "fearful": 0.023,
    "happy": 0.929,
    "neutral": 0.006,
    "sad": 0.004,
    "surprised": 0.003
  },
  "filename": "03-01-03-01-01-01-01.wav"
}
🐳 Docker Deployment
bash# Build image
docker build -t emotion-api .

# Run container
docker run -p 8080:8080 emotion-api

# Test
curl http://localhost:8080/health
📁 Project Structure
speech-emotion/
├── data/                   # Datasets (RAVDESS)
├── features/               # Cached feature files (.npy)
├── models/                 # Model checkpoints
│   ├── best_model.pth     # Best performing model
│   ├── scaler.pkl         # Feature scaler
│   └── label_encoder.pkl  # Label encoder
├── results/                # Training outputs
│   ├── training_curves.png
│   ├── test_confusion_matrix.png
│   └── metrics files
├── src/
│   ├── dataset.py         # PyTorch dataset loader
│   ├── features.py        # Audio feature extraction
│   ├── models.py          # CNN architecture
│   ├── train.py           # Training script
│   ├── eval.py            # Evaluation script
│   └── utils.py           # Helper functions
├── app.py                  # FastAPI inference API
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
└── README.md
🧪 Testing
Run installation test:
bashpython test_setup.py
📈 Results Analysis
Key Insights from Confusion Matrix
Strongest Performance:

Happy (92.9% recall) - model excels at detecting positive emotions
Surprised (90.7% recall) - distinctive acoustic patterns well-captured
Angry (87.5% recall) - strong negative emotion recognition

Common Confusions:

Calm ↔ Sad (14 misclassifications) - acoustically similar low-energy emotions
Model correctly identifies high-arousal emotions better than low-arousal ones

Business Implications:

Suitable for call center sentiment analysis
Can detect customer frustration/anger reliably
May need fine-tuning for subtle emotional distinctions

🛠️ Technologies Used
Core ML Stack

PyTorch 2.4.0 - Deep learning framework
librosa 0.10.2 - Audio processing and feature extraction
scikit-learn 1.5.0 - Data preprocessing and metrics
NumPy, Pandas - Data manipulation

Production Stack

FastAPI 0.111.0 - REST API framework
uvicorn - ASGI server
Docker - Containerization

Visualization

matplotlib 3.9.0 - Training curves
seaborn 0.13.0 - Confusion matrices

💡 Use Cases

Call Centers: Real-time customer emotion monitoring
Mental Health: Emotional state tracking in therapy sessions
Virtual Assistants: Context-aware response generation
Market Research: Emotion analysis in focus groups
Gaming: Adaptive gameplay based on player emotion
Education: Student engagement monitoring

🔮 Future Enhancements

 Real-time streaming audio inference
 Multi-language support (currently English only)
 Data augmentation (pitch shift, time stretch, noise injection)
 Attention mechanisms for improved accuracy
 Transformer-based architecture (e.g., Wav2Vec 2.0)
 Mobile deployment (TensorFlow Lite, ONNX)
 Web frontend with React
 Cloud deployment (AWS Lambda, GCP Cloud Run)
 A/B testing framework for model updates

📊 Dataset Information
RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

24 professional actors (12 male, 12 female)
2,880 audio files
8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
2 intensity levels (normal, strong)
Sample rate: 48 kHz (resampled to 16 kHz)

Citation:
Livingstone SR, Russo FA (2018) 
The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): 
A dynamic, multimodal set of facial and vocal expressions in North American English. 
PLoS ONE 13(5): e0196391.
🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
👤 Author
Your Name

GitHub: @your-username
LinkedIn: Your Name
Portfolio: yourwebsite.com

🙏 Acknowledgments

RAVDESS dataset creators for providing high-quality emotional speech data
PyTorch and FastAPI communities for excellent documentation
librosa maintainers for powerful audio processing tools

📞 Contact
For questions or collaboration opportunities, reach out via:

Email: your.email@example.com
LinkedIn: Your Profile


⭐ If you find this project helpful, please give it a star!
