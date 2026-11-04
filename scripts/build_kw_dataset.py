"""Build the killer-whale clip dataset: a balanced sample of annotated 3 s clips -> log-mel.

Usage:  python scripts/build_kw_dataset.py [--per-cell 300]
Needs:  data/kw/Annotations.csv, data/kw/audio_index.csv  (python scripts/download.py kw)
Writes: data/kw/features.npz, data/kw/clips.csv
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from dclde import kw_data as K

ap = argparse.ArgumentParser()
ap.add_argument("--per-cell", type=int, default=300)
ap.add_argument("--workers", type=int, default=16)
args = ap.parse_args()

a = K.load_annotations()
clips = K.sample_clips(a, per_cell=args.per_cell)
print(f"{len(clips)} clips sampled")
print(clips.groupby(["Provider", "label"]).size().unstack(fill_value=0))

t = time.time()
X, ok = K.extract(clips, workers=args.workers)
print(f"extracted in {time.time() - t:.0f}s; {(~ok).sum()} failed")
clips = clips[ok].reset_index(drop=True)
X = X[ok]
np.savez_compressed(K.KW_DIR / "features.npz", X=X)
clips[["name", "Provider", "Dataset", "label", "is_kw", "Ecotype", "ClassSpecies", "mid_s", "UTC"]] \
    .to_csv(K.KW_DIR / "clips.csv", index=False)
print("saved", X.shape)
