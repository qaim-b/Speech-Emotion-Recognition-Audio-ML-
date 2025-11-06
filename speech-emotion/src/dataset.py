from __future__ import annotations
import os, glob, random
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset
import torch
from .features import FeatureConfig, extract_from_path, save_cache, load_cache

EMOTION_MAP = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

def parse_emotion_from_filename(fname: str):
    parts = fname.split('-')
    if len(parts) >= 3 and parts[2] in EMOTION_MAP:
        return EMOTION_MAP[parts[2]]
    return None

def list_audio_files(root: str):
    exts = ("*.wav", "*.WAV")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    return files

class SERDataset(Dataset):
    def __init__(self, data_dir: str, split: str, cfg: FeatureConfig,
                 scaler=None, label_encoder=None,
                 cache_dir="features", feature_type="mfcc",
                 val_ratio=0.1, test_ratio=0.1, seed=42):
        assert feature_type in {"mfcc", "logmel"}
        self.cfg = cfg
        self.feature_type = feature_type
        self.cache_dir = cache_dir
        self.split = split

        all_files = [f for f in list_audio_files(data_dir) if parse_emotion_from_filename(os.path.basename(f))]
        random.Random(seed).shuffle(all_files)
        n = len(all_files)
        n_test, n_val = int(n * test_ratio), int(n * val_ratio)
        test_files = all_files[:n_test]
        val_files = all_files[n_test:n_test+n_val]
        train_files = all_files[n_test+n_val:]
        self.files_by_split = {"train": train_files, "val": val_files, "test": test_files}
        self.files = self.files_by_split[split]

        labels = [parse_emotion_from_filename(os.path.basename(f)) for f in self.files]
        self.le = label_encoder or LabelEncoder().fit(labels)
        self.y = self.le.transform(labels)

        self.scaler = scaler
        if self.scaler is None and split == "train":
            X_collect = []
            print(f"Computing scaler for {split} split...")
            for fp in self.files:
                cache_path = self._cache_path(fp)
                if not os.path.exists(cache_path):
                    feats = extract_from_path(fp, self.cfg)
                    save_cache(cache_path, feats)
                feats = load_cache(cache_path)
                X_collect.append(feats[self.feature_type])
            X_stack = np.concatenate([x.reshape(-1, x.shape[-1]) for x in X_collect], axis=0)
            self.scaler = StandardScaler().fit(X_stack)
            print(f"Scaler computed on {X_stack.shape[0]} frames")

    def _cache_path(self, fp: str):
        rel = os.path.relpath(fp)
        base = os.path.splitext(rel)[0].replace(os.sep, "_")
        return os.path.join(self.cache_dir, base + ".npz")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        cache_path = self._cache_path(fp)
        
        if not os.path.exists(cache_path):
            feats = extract_from_path(fp, self.cfg)
            save_cache(cache_path, feats)
        
        feats = load_cache(cache_path)
        X = feats[self.feature_type]  # (time, features)
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # (1, time, features)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        
        return X, y
