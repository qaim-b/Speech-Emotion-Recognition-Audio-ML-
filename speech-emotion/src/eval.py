import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .features import FeatureConfig
from .dataset import SERDataset
from .models import EmotionCNN
from .utils import (load_checkpoint, load_scaler_and_encoder, 
                   plot_confusion_matrix, save_metrics, get_classification_report)

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc='Evaluating'):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = 100 * correct / total
    return np.array(all_labels), np.array(all_preds), accuracy

def main():
    parser = argparse.ArgumentParser(description='Evaluate Speech Emotion Recognition Model')
    parser.add_argument('--data_dir', type=str, default='data/RAVDESS', help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, default='models/best_model.pth', help='Path to checkpoint')
    parser.add_argument('--feature_type', type=str, default='mfcc', choices=['mfcc', 'logmel'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cache_dir', type=str, default='features')
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load scaler and encoder
    print("Loading scaler and label encoder...")
    scaler, label_encoder = load_scaler_and_encoder(args.save_dir)
    
    # Feature config
    cfg = FeatureConfig()
    feature_dim = cfg.n_mfcc if args.feature_type == 'mfcc' else cfg.n_mels
    
    # Create test dataset
    print(f"Loading {args.split} dataset...")
    test_dataset = SERDataset(
        args.data_dir, split=args.split, cfg=cfg,
        scaler=scaler,
        label_encoder=label_encoder,
        cache_dir=args.cache_dir, feature_type=args.feature_type
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"\nDataset Info:")
    print(f"  {args.split.capitalize()} samples: {len(test_dataset)}")
    print(f"  Num classes: {len(label_encoder.classes_)}")
    print(f"  Classes: {label_encoder.classes_}")
    
    # Load model
    num_classes = len(label_encoder.classes_)
    model = EmotionCNN(num_classes=num_classes, feature_dim=feature_dim).to(device)
    
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    epoch, loss, acc = load_checkpoint(model, None, args.checkpoint)
    print(f"Checkpoint: Epoch {epoch}, Val Loss: {loss:.4f}, Val Acc: {acc:.2f}%")
    
    # Evaluate
    print("\nEvaluating model...")
    y_true, y_pred, accuracy = evaluate_model(model, test_loader, device)
    
    print(f"\n{args.split.capitalize()} Accuracy: {accuracy:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    report = get_classification_report(y_true, y_pred, label_encoder.classes_)
    print(report)
    
    # Save classification report
    with open(os.path.join(args.results_dir, f'{args.split}_classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, label_encoder.classes_,
        os.path.join(args.results_dir, f'{args.split}_confusion_matrix.png')
    )
    
    # Save metrics
    metrics = {
        'split': args.split,
        'accuracy': accuracy,
        'checkpoint_epoch': epoch,
        'num_samples': len(test_dataset)
    }
    save_metrics(metrics, os.path.join(args.results_dir, f'{args.split}_metrics.json'))
    
    print(f"\nEvaluation complete! Results saved to {args.results_dir}")

if __name__ == '__main__':
    main()
