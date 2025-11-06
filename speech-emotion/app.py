from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import numpy as np
import tempfile
import os
from src.features import FeatureConfig, extract_from_path
from src.models import EmotionCNN
from src.utils import load_scaler_and_encoder

app = FastAPI(title="Speech Emotion Recognition API", version="1.0")

# Global variables for model, scaler, and encoder
model = None
scaler = None
label_encoder = None
device = None
cfg = None

@app.on_event("startup")
async def load_model():
    global model, scaler, label_encoder, device, cfg
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load scaler and encoder
    model_dir = 'models'
    if not os.path.exists(model_dir):
        print("WARNING: models directory not found. Please train the model first.")
        return
    
    try:
        scaler, label_encoder = load_scaler_and_encoder(model_dir)
        print(f"Loaded scaler and label encoder. Classes: {label_encoder.classes_}")
    except Exception as e:
        print(f"WARNING: Could not load scaler/encoder: {e}")
        return
    
    # Load model
    checkpoint_path = os.path.join(model_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")
        return
    
    try:
        cfg = FeatureConfig()
        num_classes = len(label_encoder.classes_)
        model = EmotionCNN(num_classes=num_classes, feature_dim=cfg.n_mfcc).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded successfully from {checkpoint_path}")
        print(f"Checkpoint - Epoch: {checkpoint['epoch']}, Accuracy: {checkpoint['accuracy']:.2f}%")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        model = None

@app.get("/")
async def root():
    return {
        "message": "Speech Emotion Recognition API",
        "status": "online" if model is not None else "model not loaded",
        "endpoints": {
            "/predict": "POST - Upload audio file for emotion prediction",
            "/health": "GET - Check API health status"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model is not None else "model not loaded",
        "device": str(device),
        "classes": label_encoder.classes_.tolist() if label_encoder is not None else []
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or scaler is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    # Validate file type
    if not file.filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload audio file (.wav, .mp3, .ogg, .flac)")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Extract features
        features = extract_from_path(tmp_path, cfg)
        X = features['mfcc']  # (time, features)
        
        # Scale features
        X = scaler.transform(X)
        
        # Convert to tensor
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, time, features)
        
        # Predict
        with torch.no_grad():
            outputs = model(X)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get emotion label
        emotion_idx = predicted.item()
        emotion = label_encoder.inverse_transform([emotion_idx])[0]
        confidence_score = confidence.item()
        
        # Get all class probabilities
        all_probs = probabilities[0].cpu().numpy()
        emotion_scores = {
            label_encoder.classes_[i]: float(all_probs[i])
            for i in range(len(label_encoder.classes_))
        }
        
        # Cleanup temp file
        os.unlink(tmp_path)
        
        return JSONResponse(content={
            "emotion": emotion,
            "confidence": float(confidence_score),
            "all_emotions": emotion_scores,
            "filename": file.filename
        })
    
    except Exception as e:
        # Cleanup temp file if it exists
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
