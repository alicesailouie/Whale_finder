"""Killer whale dataset (DCLDE 2027 DC challenge): annotation table, clip fetching, spectrograms.

The full audio is ~1.6 TB, so nothing is downloaded wholesale. WAV clips are read with
HTTP byte-range requests straight from the public bucket; FLAC files (small) are cached.
"""
from __future__ import annotations

import io
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import gcd

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly, stft

from . import gcs

KW_DIR = gcs.DATA_DIR / "kw"
FS = 32_000          # common sample rate (Hz)
CLIP_S = 3.0         # clip length (s)
N_MELS = 96
FMIN, FMAX = 200.0, 15_500.0
RESIDENT = {"SRKW", "NRKW", "SAR"}

_HEADERS: dict[str, "WavInfo"] = {}


# ----------------------------------------------------------------------------- annotations
def load_annotations() -> pd.DataFrame:
    """Master annotation table joined to bucket object names, with tidy labels."""
    a = pd.read_csv(KW_DIR / "Annotations.csv", low_memory=False)
    a["base"] = a.Soundfile.str.split("/").str[-1]
    idx = pd.read_csv(KW_DIR / "audio_index.csv")
    a = a.merge(idx[["name", "base"]], on="base", how="inner")
    a["is_kw"] = (a.ClassSpecies == "KW").astype(int)
    # Detections flagged uncertain are excluded from KW training and testing.
    a = a[~((a.is_kw == 1) & (a.KW_certain == 0))]
    a["mid_s"] = (a.FileBeginSec + a.FileEndSec) / 2
    return a.reset_index(drop=True)


def build_index(index_path=None) -> pd.DataFrame:
    """List every audio object in the killer-whale folder (cached to audio_index.csv)."""
    rows = [(n, s) for n, s in gcs.list_objects(gcs.ROOT + "dclde_2027_killer_whales/")
            if n.lower().endswith((".wav", ".flac"))]
    idx = pd.DataFrame(rows, columns=["name", "size"])
    idx["base"] = idx.name.str.split("/").str[-1]
    idx.to_csv(index_path or KW_DIR / "audio_index.csv", index=False)
    return idx


# ----------------------------------------------------------------------------- audio access
@dataclass
class WavInfo:
    fs: int
    channels: int
    bits: int
    fmt: int          # 1 = PCM int, 3 = IEEE float
    data_offset: int
    data_bytes: int


def wav_info(name: str) -> WavInfo:
    if name in _HEADERS:
        return _HEADERS[name]
    head = gcs.fetch_range(name, 0, 65536)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise ValueError(f"not a RIFF/WAVE file: {name}")
    pos, fmt = 12, None
    while pos + 8 <= len(head):
        cid, size = head[pos:pos + 4], struct.unpack("<I", head[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            tag, ch, fs = struct.unpack("<HHI", head[pos + 8:pos + 16])
            bits = struct.unpack("<H", head[pos + 22:pos + 24])[0]
            if tag == 0xFFFE:   # WAVE_FORMAT_EXTENSIBLE: real format is in the sub-GUID
                tag = struct.unpack("<H", head[pos + 32:pos + 34])[0]
            fmt = (tag, ch, fs, bits)
        elif cid == b"data":
            info = WavInfo(fs=fmt[2], channels=fmt[1], bits=fmt[3], fmt=fmt[0],
                           data_offset=pos + 8, data_bytes=size)
            _HEADERS[name] = info
            return info
        pos += 8 + size + (size & 1)
    raise ValueError(f"no data chunk in first 64 KiB: {name}")


def _decode(raw: bytes, info: WavInfo) -> np.ndarray:
    bps = info.bits // 8
    n = len(raw) // (bps * info.channels)
    raw = raw[: n * bps * info.channels]
    if info.fmt == 3:
        x = np.frombuffer(raw, dtype="<f4" if bps == 4 else "<f8").astype(np.float32)
    elif bps == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 2 ** 15
    elif bps == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = (np.where(v >= 2 ** 23, v - 2 ** 24, v)).astype(np.float32) / 2 ** 23
    elif bps == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2 ** 31
    else:
        raise ValueError(f"unsupported bit depth {info.bits}")
    return x.reshape(-1, info.channels)[:, 0]


def read_clip(name: str, start_s: float, dur_s: float = CLIP_S) -> np.ndarray:
    """Return a mono clip resampled to ``FS``; zero-padded if it runs past the file."""
    if name.lower().endswith(".wav"):
        info = wav_info(name)
        frame = info.channels * info.bits // 8
        a = max(0, int(start_s * info.fs))
        b = min(info.data_bytes // frame, int((start_s + dur_s) * info.fs))
        if b <= a:
            x, fs = np.zeros(1, np.float32), info.fs
        else:
            raw = gcs.fetch_range(name, info.data_offset + a * frame, info.data_offset + b * frame)
            x, fs = _decode(raw, info), info.fs
    else:
        path = gcs.fetch(name)
        with sf.SoundFile(path) as f:
            fs = f.samplerate
            f.seek(max(0, min(int(start_s * fs), f.frames)))
            x = f.read(int(dur_s * fs), dtype="float32", always_2d=True)[:, 0]
    if fs != FS:
        g = gcd(FS, fs)
        x = resample_poly(x, FS // g, fs // g).astype(np.float32)
    n = int(dur_s * FS)
    return np.pad(x[:n], (0, max(0, n - len(x))))


# ----------------------------------------------------------------------------- features
def _mel_fb(n_fft: int = 1024) -> np.ndarray:
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = mel2hz(np.linspace(hz2mel(FMIN), hz2mel(FMAX), N_MELS + 2))
    freqs = np.linspace(0, FS / 2, n_fft // 2 + 1)
    fb = np.zeros((N_MELS, len(freqs)), np.float32)
    for i in range(N_MELS):
        lo, c, hi = edges[i:i + 3]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return fb


_FB = _mel_fb()


def logmel(x: np.ndarray) -> np.ndarray:
    """Log-mel spectrogram with per-band median removed (cheap site/gain normalisation)."""
    _, _, Z = stft(x, fs=FS, nperseg=1024, noverlap=512, boundary=None, padded=False)
    S = np.log(_FB @ (np.abs(Z) ** 2) + 1e-10)
    S -= np.median(S, axis=1, keepdims=True)
    return S.astype(np.float16)


# ----------------------------------------------------------------------------- sampling
def sample_clips(a: pd.DataFrame, per_cell: int = 300, seed: int = 0) -> pd.DataFrame:
    """Balanced sample: up to ``per_cell`` clips per (provider, label) cell.

    Label is the ecotype for killer whale calls and the species class otherwise.
    """
    a = a.assign(label=np.where(a.is_kw == 1, a.Ecotype.fillna("KW_unk"), a.ClassSpecies))
    a = a[a.label != "KW_unk"]
    a = a.sample(frac=1.0, random_state=seed)
    return a.groupby(["Provider", "label"]).head(per_cell).reset_index(drop=True)


def extract(clips: pd.DataFrame, workers: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Fetch every clip and compute its spectrogram. Returns (X, ok_mask)."""
    def one(row):
        try:
            x = read_clip(row.name_, max(0.0, row.mid_s - CLIP_S / 2))
            return logmel(x)
        except Exception as e:  # noqa: BLE001 - a few files are expected to be unreadable
            return e

    rows = [r for r in clips.rename(columns={"name": "name_"}).itertuples()]
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(one, rows))
    ok = np.array([not isinstance(o, Exception) for o in out])
    shape = next(o.shape for o in out if not isinstance(o, Exception))
    X = np.stack([o if not isinstance(o, Exception) else np.zeros(shape, np.float16) for o in out])
    return X, ok
