"""Plot one 70 s DAS SNR image with its annotated fin whale picks (README figure).

Usage:  python scripts/plot_das_waterfall.py
Downloads one ~850 MB NetCDF file from the bucket on first run.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

from dclde import das_loc as L
from dclde import gcs

NAME = "dclde/2027/dclde_2027_das-finwhale-localization/detections/batch_1/Denoised_SNR_North_2021-11-04_02-00-02.nc"
ds = xr.open_dataset(gcs.fetch(NAME), engine="h5netcdf")
img = ds.SNR_hf.isel(dist=slice(None, None, 10), time=slice(None, None, 2)).load()

picks = L.load_picks()
p = picks[(picks.segment_file == "2021-11-04_02_00_02") & (picks.cable == "North")]

fig, ax = plt.subplots(figsize=(10, 5))
ax.pcolormesh(img.time, img.dist / 1e3, img, cmap="magma", vmin=0, vmax=float(img.quantile(0.995)), shading="auto")
for k, g in p.groupby("call_key"):
    ax.plot(g.time, g.dist / 1e3, "o", ms=3, mfc="none", mec="cyan", mew=0.8)
ax.set(xlabel="time since 02:00:02 UTC (s)", ylabel="distance along North cable (km)",
       title="Fin whale calls on 54 km of fibre (denoised SNR, HF band) with annotated picks")
fig.tight_layout()
out = ROOT / "results" / "localization" / "waterfall.png"
fig.savefig(out, dpi=150)
print(out)
