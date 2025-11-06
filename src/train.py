from __future__ import annotations
import os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from .dataset import SERDataset
from .features import FeatureConfig
from .models import CNN1D_SER




def train(args):
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


cfg = FeatureConfig(sr=16000, n_mfcc=40, n_mels=64, duration=3.0)


# build train dataset to fit scaler and label encoder
train_ds = SERDataset(args.data_dir, split="train", cfg=cfg,
scaler=None, label_encoder=None,
feature_type=args.feature_type,
val_ratio=args.val_ratio, test_ratio=args.test_ratio)
scaler: StandardScaler = train_ds.scaler
le: LabelEncoder = train_ds.le


# Rebuild datasets with fitted transformers
val_ds = SERDataset(args.data_dir, split="val", cfg=cfg,
scaler=scaler, label_encoder=le,
feature_type=args.feature_type,
val_ratio=args.val_ratio, test_ratio=args.test_ratio)
test_ds = SERDataset(args.data_dir, split="test", cfg=cfg,
scaler=scaler, label_encoder=le,
feature_type=args.feature_type,
val_ratio=args.val_ratio, test_ratio=args.test_ratio)


train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, b
