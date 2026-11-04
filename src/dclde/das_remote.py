"""Read the DCLDE 2027 DAS data straight from the public NOAA bucket, without downloading files.

Each ~850 MB NetCDF file stores its two SNR images (``SNR_hf``, ``SNR_lf``; distance x time)
as gzip-compressed HDF5 chunks of 696 channels x 737 samples (~1.4 km x 3.7 s). Opening a
file over HTTPS reads only its metadata; slicing a window then fetches only the chunks that
overlap it.

Good for: windows around calls, a few km x a few seconds (seconds per window).
Slow for: strided or full-file scans from a home connection. Those touch nearly every
chunk, so run them on a machine in Google Cloud (bucket location: US multi-region) instead.

Example
-------
>>> from dclde import das_remote as R
>>> files = R.list_files()                                   # 240 files, no download
>>> w = R.window("North", "2021-11-04 02:00:02", dist_m=(45e3, 52e3), time_s=(3, 15))
>>> w.plot()                                                 # xarray DataArray
"""
from __future__ import annotations

import functools
import re

import numpy as np
import pandas as pd
import xarray as xr

from . import gcs

PREFIX = gcs.ROOT + "dclde_2027_das-finwhale-localization/detections/"
_STAMP = re.compile(r"Denoised_SNR_(North|South)_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})\.nc$")


@functools.lru_cache(maxsize=1)
def list_files() -> pd.DataFrame:
    """Every DAS NetCDF file in the bucket: cable, start time, size and object name."""
    rows = []
    for name, size in gcs.list_objects(PREFIX):
        m = _STAMP.search(name)
        if m:
            cable, day, hh, mm, ss = m.groups()
            rows.append(dict(cable=cable, start=pd.Timestamp(f"{day} {hh}:{mm}:{ss}"),
                             batch=name.split("/")[-2], size_mb=size / 1e6, name=name))
    return pd.DataFrame(rows).sort_values(["cable", "start"]).reset_index(drop=True)


def find_file(cable: str, start) -> str:
    """Object name for one cable and file start time (a string or Timestamp)."""
    f = list_files()
    hit = f[(f.cable == cable.capitalize()) & (f.start == pd.Timestamp(start))]
    if hit.empty:
        raise KeyError(f"no {cable} file starting at {start}")
    return hit.name.iloc[0]


@functools.lru_cache(maxsize=8)
def open_remote(name: str, block_size: int = 2 ** 21) -> xr.Dataset:
    """Lazily open one bucket object as an xarray Dataset (metadata only, ~5-10 s).

    Open handles are cached, so repeated windows from the same file skip the metadata read.
    Needs ``fsspec`` and ``aiohttp``.
    """
    import fsspec

    fh = fsspec.open(gcs.url(name), block_size=block_size, cache_type="blockcache").open()
    return xr.open_dataset(fh, engine="h5netcdf")


def window(cable: str, start, dist_m: tuple[float, float], time_s: tuple[float, float],
           band: str = "hf") -> xr.DataArray:
    """Load one distance x time window of the denoised SNR image (``band`` = 'hf' or 'lf').

    ``start`` is the file start time; ``time_s`` is in seconds since that start.
    """
    ds = open_remote(find_file(cable, start))
    return ds[f"SNR_{band}"].sel(dist=slice(*dist_m), time=slice(*time_s)).load()


def call_window(picks: pd.DataFrame, pad_m: float = 1000.0, pad_s: float = 2.0) -> xr.DataArray:
    """The window that covers one annotated call (rows of ``das_loc.load_picks()`` for one track)."""
    g = picks
    start = pd.to_datetime(g.segment_file.iloc[0], format="%Y-%m-%d_%H_%M_%S")
    band = "hf" if g.call_type.iloc[0] == "HF" else "lf"
    return window(g.cable.iloc[0], start,
                  (g.dist.min() - pad_m, g.dist.max() + pad_m),
                  (max(0.0, g.time.min() - pad_s), g.time.max() + pad_s), band=band)


def predicted_arrivals(cable, fix: dict, dist_m: np.ndarray, file_start, c: float) -> np.ndarray:
    """Arrival time (s since file start) along the cable for a fitted source ``fix``."""
    r = cable.position(dist_m)
    t0 = (pd.Timestamp("2021-11-01") + pd.to_timedelta(fix["t0"], unit="s") - pd.Timestamp(file_start)).total_seconds()
    return t0 + np.linalg.norm(r - np.array([fix["x"], fix["y"], fix["z"]]), axis=1) / c
