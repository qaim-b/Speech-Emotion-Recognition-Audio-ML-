from __future__ import annotations
import librosa
from dataclasses import dataclass


@dataclass
class FeatureConfig:
sr: int = 16000
n_mfcc: int = 40
n_mels: int = 64
hop_length: int = 512
n_fft: int = 1024
duration: float = 3.0 # seconds; clips are trimmed/padded to this


def load_audio_mono(path: str, sr: int) -> np.ndarray:
y, _ = librosa.load(path, sr=sr, mono=True)
return y


def trim_or_pad(y: np.ndarray, sr: int, duration: float) -> np.ndarray:
target_len = int(sr * duration)
if len(y) > target_len:
y = y[:target_len]
elif len(y) < target_len:
y = np.pad(y, (0, target_len - len(y)))
return y


def extract_features(y: np.ndarray, cfg: FeatureConfig) -> dict[str, np.ndarray]:
# MFCC (T x n_mfcc) and log-mel (T x n_mels)
mfcc = librosa.feature.mfcc(y=y, sr=cfg.sr, n_mfcc=cfg.n_mfcc,
hop_length=cfg.hop_length, n_fft=cfg.n_fft)
mel = librosa.feature.melspectrogram(y=y, sr=cfg.sr, n_mels=cfg.n_mels,
hop_length=cfg.hop_length, n_fft=cfg.n_fft)
logmel = librosa.power_to_db(mel + 1e-10)
# time-major to channel-first later
return {"mfcc": mfcc.T.astype(np.float32), "logmel": logmel.T.astype(np.float32)}


def extract_from_path(path: str, cfg: FeatureConfig) -> dict[str, np.ndarray]:
y = load_audio_mono(path, sr=cfg.sr)
y = trim_or_pad(y, sr=cfg.sr, duration=cfg.duration)
return extract_features(y, cfg)


def save_cache(cache_path: str, feats: dict[str, np.ndarray]) -> None:
os.makedirs(os.path.dirname(cache_path), exist_ok=True)
np.savez_compressed(cache_path, **feats)


def load_cache(cache_path: str) -> dict[str, np.ndarray]:
data = np.load(cache_path)
return {k: data[k] for k in data.files}
