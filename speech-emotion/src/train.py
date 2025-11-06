import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .features import FeatureConfig
from .dataset import SERDataset
from .models import EmotionCNN, count_parameters
from .utils import save_checkpoint, save_scaler_and_encoder, plot_training_curves, save_metrics

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in tqdm(dataloader, desc='Validation'):
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def main():
    parser = argparse.ArgumentParser(description='Train Speech Emotion Recognition Model')
    parser.add_argument('--data_dir', type=str, default='data/RAVDESS', help='Path to dataset')
    parser.add_argument('--feature_type', type=str, default='mfcc', choices=['mfcc', 'logmel'])
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--cache_dir', type=str, default='features')
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Feature config
    cfg = FeatureConfig()
    feature_dim = cfg.n_mfcc if args.feature_type == 'mfcc' else cfg.n_mels
    
    # Create datasets
    print("Loading training dataset...")
    train_dataset = SERDataset(
        args.data_dir, split='train', cfg=cfg,
        cache_dir=args.cache_dir, feature_type=args.feature_type
    )
    
    print("Loading validation dataset...")
    val_dataset = SERDataset(
        args.data_dir, split='val', cfg=cfg,
        scaler=train_dataset.scaler,
        label_encoder=train_dataset.le,
        cache_dir=args.cache_dir, feature_type=args.feature_type
    )
    
    # Save scaler and encoder
    save_scaler_and_encoder(train_dataset.scaler, train_dataset.le, args.save_dir)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"\nDataset Info:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Num classes: {len(train_dataset.le.classes_)}")
    print(f"  Classes: {train_dataset.le.classes_}")
    
    # Model
    num_classes = len(train_dataset.le.classes_)
    model = EmotionCNN(num_classes=num_classes, feature_dim=feature_dim).to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training loop
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        scheduler.step(val_loss)
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth')
        save_checkpoint(model, optimizer, epoch, val_loss, val_acc, checkpoint_path)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.save_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, best_path)
            print(f"✓ New best model! Val Acc: {val_acc:.2f}%")
    
    # Save training curves
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        os.path.join(args.results_dir, 'training_curves.png')
    )
    
    # Save final metrics
    metrics = {
        'best_val_accuracy': best_val_acc,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1]
    }
    save_metrics(metrics, os.path.join(args.results_dir, 'training_metrics.json'))
    
    print(f"\nTraining complete! Best Val Accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()
