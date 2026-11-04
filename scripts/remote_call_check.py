"""Stream the raw DAS window around a localised call from the cloud and check the fit.

No DAS files are downloaded: only the HDF5 chunks covering each call are fetched over HTTPS.
For each of a few well-constrained L-2 calls, the script plots the SNR image on both cables
with the annotated picks and the arrival curve predicted by our L-2 position.

Usage:  python scripts/remote_call_check.py [--n 3]
Needs:  results/localization/L2_fixes.csv  (python scripts/run_localization.py)
Writes: results/localization/remote_call_check.png
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dclde import das_loc as L
from dclde import das_remote as R

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3)
args = ap.parse_args()

OUT = ROOT / "results" / "localization"
summary = pd.read_csv(OUT / "summary.csv", index_col=0)["value"]
c = float(summary["sound_speed_m_s"])
l2 = pd.read_csv(OUT / "L2_fixes.csv")
picks = L.load_picks()
cables = {n: L.load_cable(n) for n in ("North", "South")}

best = l2[l2.quality_ok].sort_values("n_picks", ascending=False).head(args.n)
fig, axs = plt.subplots(len(best), 2, figsize=(11, 3.2 * len(best)), squeeze=False)
for row, fix in zip(axs, best.itertuples()):
    for ax, key in zip(row, fix.call_key.split("+")):
        g = picks[picks.call_key == key]
        t = time.time()
        w = R.call_window(g, pad_m=3000, pad_s=4)
        print(f"{key}: fetched {w.shape} window in {time.time() - t:.1f}s")
        ax.pcolormesh(w.time, w.dist / 1e3, w, cmap="magma", vmin=0,
                      vmax=float(np.nanquantile(w, 0.995)), shading="auto")
        start = pd.to_datetime(g.segment_file.iloc[0], format="%Y-%m-%d_%H_%M_%S")
        d = np.linspace(float(w.dist.min()), float(w.dist.max()), 300)
        pred = R.predicted_arrivals(cables[g.cable.iloc[0]], fix._asdict(), d, start, c)
        ax.plot(pred, d / 1e3, "c-", lw=1, label="predicted from L-2 fix")
        ax.plot(g.time, g.dist / 1e3, "wo", ms=3, mfc="none", label="annotated picks")
        ax.set(title=f"{g.cable.iloc[0]} cable, {start:%Y-%m-%d %H:%M:%S} UTC",
               xlabel="s since file start", ylabel="km along cable")
axs[0, 0].legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "remote_call_check.png", dpi=130)
print(OUT / "remote_call_check.png")
