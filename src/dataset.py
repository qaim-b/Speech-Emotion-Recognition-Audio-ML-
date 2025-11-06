from __future__ import annotations
train_files = all_files[n_test+n_val:]


self.files_by_split = {"train": train_files, "val": val_files, "test": test_files}
self.files = self.files_by_split[split]


# Labels
labels = [parse_emotion_from_filename(os.path.basename(f)) for f in self.files]
if label_encoder is None:
self.le = LabelEncoder()
self.le.fit(labels)
else:
self.le = label_encoder
self.y = self.le.transform(labels)


# Build/fit scaler on train only
self.scaler = scaler
if self.scaler is None and split == "train":
# Scan to compute scaler using selected feature type
X_collect = []
for fp in self.files:
cache_path = self._cache_path(fp)
if not os.path.exists(cache_path):
feats = extract_from_path(fp, self.cfg)
save_cache(cache_path, feats)
feats = load_cache(cache_path)
X_collect.append(feats[self.feature_type])
# stack time x feat, then standardize feature dim
X_stack = np.concatenate([x.reshape(-1, x.shape[-1]) for x in X_collect], axis=0)
self.scaler = StandardScaler().fit(X_stack)


def _cache_path(self, fp: str) -> str:
rel = os.path.relpath(fp)
base = os.path.splitext(rel)[0].replace(os.sep, "_")
return os.path.join(self.cache_dir, base + ".npz")


def __len__(self):
return len(self.files)


def __getitem__(self, idx: int):
fp = self.files[idx]
cache_path = self._cache_path(fp)
if not os.path.exists(cache_path):
feats = extract_from_path(fp, self.cfg)
save_cache(cache_path, feats)
feats = load_cache(cache_path)[self.feature_type] # (T, F)
# scale feature dims
T, F = feats.shape
feats_2d = feats.reshape(-1, F)
feats_scaled = self.scaler.transform(feats_2d).reshape(T, F) if self.scaler else feats
x = np.expand_dims(feats_scaled.T, axis=0).astype(np.float32) # (C=1, F, T)
y = self.y[idx]
return x, y
