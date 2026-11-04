"""Fin whale localisation from DAS arrival-time picks (DCLDE 2027 challenges L-1 and L-2).

Model
-----
Each pick is an arrival time ``t_i`` of one call at a point ``r_i`` on the cable. For a
source at ``s`` emitting at ``t0`` and a constant effective sound speed ``c``:

    t_i = t0 + |s - r_i| / c

L-1 solves for (x, y, t0) from one cable with the whale depth fixed. The cable's
bends break the left/right mirror ambiguity a straight array would have.
L-2 solves for (x, y, z, t0) using picks from both cables.

Coordinates are the dataset's local ``x``/``y`` metres (from the geometry CSVs) and
``z`` = elevation in metres (negative below the sea surface), matching ``depth``
in the geometry files.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .gcs import DATA_DIR

DAS_DIR = DATA_DIR / "das"
C_DEFAULT = 1490.0        # m/s, effective speed of sound
Z_DEFAULT = -20.0         # m, assumed calling depth for the 2-D (L-1) fit
GRID = dict(x=(-20e3, 140e3), y=(-30e3, 70e3), step=1000.0)


# ----------------------------------------------------------------------------- data
@dataclass
class Cable:
    name: str
    chan_m: np.ndarray
    xyz: np.ndarray          # (n, 3)
    geo: pd.DataFrame = field(repr=False)

    def position(self, dist_m: np.ndarray) -> np.ndarray:
        """Interpolate cable (x, y, z) at distances along the fibre."""
        d = np.asarray(dist_m, float)
        return np.column_stack([np.interp(d, self.chan_m, self.xyz[:, k]) for k in range(3)])

    def lonlat(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map local x/y to lon/lat through the affine fit local→UTM (zone 10N)."""
        return xy_to_lonlat(x, y, self.geo)


def load_cable(name: str) -> Cable:
    g = pd.read_csv(DAS_DIR / "geometry" / f"{name.lower()}_DAS_multicoord.csv")
    g = g.sort_values("chan_m").drop_duplicates("chan_m")
    return Cable(name.capitalize(), g.chan_m.to_numpy(), g[["x", "y", "depth"]].to_numpy(), g)


def xy_to_lonlat(x, y, geo: pd.DataFrame):
    from pyproj import Transformer

    A = np.column_stack([geo.x, geo.y, np.ones(len(geo))])
    cx, *_ = np.linalg.lstsq(A, geo.utm_x, rcond=None)
    cy, *_ = np.linalg.lstsq(A, geo.utm_y, rcond=None)
    ux = cx[0] * np.asarray(x) + cx[1] * np.asarray(y) + cx[2]
    uy = cy[0] * np.asarray(x) + cy[1] * np.asarray(y) + cy[2]
    lon, lat = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True).transform(ux, uy)
    return lon, lat


def load_picks() -> pd.DataFrame:
    """All annotation picks, one row per (call, channel) with absolute time in seconds."""
    rows = []
    for f in sorted(glob.glob(str(DAS_DIR / "annotations" / "*" / "*.csv"))):
        d = pd.read_csv(f)
        if d.empty:
            continue
        d["batch"] = os.path.basename(os.path.dirname(f))
        d["segment_file"] = os.path.basename(f)[-23:-4]
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    start = pd.to_datetime(df.segment_file, format="%Y-%m-%d_%H_%M_%S")
    df["t_abs"] = (start - pd.Timestamp("2021-11-01")).dt.total_seconds() + df.time
    df["call_key"] = df.batch + "/" + df.segment_file + "/" + df.cable + "/" + df.call_id.astype(str)
    return split_tracks(df)


def split_tracks(df: pd.DataFrame, c: float = C_DEFAULT, slack_s: float = 0.5) -> pd.DataFrame:
    """Split annotation calls that actually contain several arrivals.

    Along the fibre, an arrival-time curve can never change faster than 1/c seconds
    per metre. Walking the picks in distance order, a pick joins the first track whose
    last pick it is physically consistent with; otherwise it starts a new track.
    Resulting keys get a ``#k`` suffix.
    """
    out = []
    for key, g in df.groupby("call_key", sort=False):
        g = g.sort_values("dist")
        tracks: list[list[int]] = []
        last: list[tuple[float, float]] = []
        for idx, d, t in zip(g.index, g.dist, g.t_abs):
            for k, (d0, t0) in enumerate(last):
                if abs(t - t0) <= abs(d - d0) / c + slack_s:
                    tracks[k].append(idx)
                    last[k] = (d, t)
                    break
            else:
                tracks.append([idx])
                last.append((d, t))
        for k, idxs in enumerate(tracks):
            out.append(df.loc[idxs].assign(call_key=f"{key}#{k}"))
    return pd.concat(out)


# ----------------------------------------------------------------------------- solver
@dataclass
class Fix:
    call_key: str
    cable: str
    call_type: str
    n_picks: int
    x: float
    y: float
    z: float
    t0: float
    rms_s: float
    sx: float = np.nan       # 1-sigma uncertainty, metres
    sy: float = np.nan
    sz: float = np.nan
    mirror_ratio: float = np.nan   # cost of best mirror-side solution / best cost (>>1 = unambiguous)
    x_alt: float = np.nan          # the competing (mirror-side) solution
    y_alt: float = np.nan


def _grid_init(r: np.ndarray, t: np.ndarray, z: float, c: float, n_best: int = 2):
    """Brute-force (x, y) grid search with t0 solved analytically; returns best local minima."""
    xs = np.arange(*GRID["x"], GRID["step"])
    ys = np.arange(*GRID["y"], GRID["step"])
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
    tt = np.linalg.norm(P[:, None, :] - r[None], axis=2) / c          # (ngrid, npicks)
    t0 = np.median(t[None] - tt, axis=1)
    cost = np.mean((t[None] - tt - t0[:, None]) ** 2, axis=1).reshape(X.shape)
    best = np.unravel_index(np.argmin(cost), cost.shape)
    out = [(xs[best[0]], ys[best[1]], cost[best])]
    # best point at least 5 km from the first one: candidate for the "mirror" solution
    far = np.hypot(X - xs[best[0]], Y - ys[best[1]]) > 5e3
    if n_best > 1 and far.any():
        i = np.argmin(np.where(far, cost, np.inf))
        i = np.unravel_index(i, cost.shape)
        out.append((xs[i[0]], ys[i[1]], cost[i]))
    return out


def _solve(r, t, p0, c, fit_z, z_fixed, zmin):
    def resid(p):
        s = np.array([p[0], p[1], p[2] if fit_z else z_fixed])
        return t - (p[-1] + np.linalg.norm(r - s, axis=1) / c)

    lo = [-np.inf, -np.inf] + ([zmin] if fit_z else []) + [-np.inf]
    hi = [np.inf, np.inf] + ([-1.0] if fit_z else []) + [np.inf]
    return least_squares(resid, p0, loss="soft_l1", f_scale=0.05, bounds=(lo, hi))


def _uncertainty(sol, n):
    J = sol.jac
    dof = max(n - J.shape[1], 1)
    s2 = np.sum(sol.fun ** 2) / dof
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        return np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        return np.full(J.shape[1], np.nan)


def locate(r: np.ndarray, t: np.ndarray, c: float = C_DEFAULT, fit_z: bool = False,
           z: float = Z_DEFAULT, zmin: float = -200.0):
    """Locate one call from cable positions ``r`` (n,3) and arrival times ``t`` (n,)."""
    cands = _grid_init(r, t, z, c)
    sols = []
    for x0, y0, _ in cands:
        t00 = np.median(t - np.linalg.norm(r - [x0, y0, z], axis=1) / c)
        p0 = [x0, y0] + ([z] if fit_z else []) + [t00]
        sols.append(_solve(r, t, p0, c, fit_z, z, zmin))
    costs = [np.mean(s.fun ** 2) for s in sols]
    k = int(np.argmin(costs))
    sol = sols[k]
    sig = _uncertainty(sol, len(t))
    p = sol.x
    zz = p[2] if fit_z else z
    alt = sols[1 - k].x if len(sols) > 1 else [np.nan, np.nan]
    mirror = costs[1 - k] / max(costs[k], 1e-12) if len(sols) > 1 else np.nan
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))
    return dict(x=p[0], y=p[1], z=zz, t0=p[-1], rms_s=rms, sx=sig[0], sy=sig[1],
                sz=sig[2] if fit_z else np.nan, mirror_ratio=mirror, x_alt=alt[0], y_alt=alt[1])


# ----------------------------------------------------------------------------- challenges
def run_L1(picks: pd.DataFrame, cables: dict[str, Cable], c: float = C_DEFAULT, min_picks: int = 5) -> pd.DataFrame:
    """L-1: 2-D (x, y) fix for every call, independently on each cable."""
    out = []
    for key, g in picks.groupby("call_key"):
        if len(g) < min_picks:
            continue
        cab = cables[g.cable.iloc[0]]
        r = cab.position(g.dist.to_numpy())
        fix = locate(r, g.t_abs.to_numpy(), c=c)
        out.append(Fix(key, cab.name, g.call_type.iloc[0], len(g), **fix))
    df = pd.DataFrame([f.__dict__ for f in out])
    return _add_lonlat(df, cables)


def associate(picks: pd.DataFrame, cables: dict[str, Cable], c: float = C_DEFAULT,
              window_s: float = 15.0, max_rms_s: float = 0.15, min_picks: int = 5) -> pd.DataFrame:
    """Pair North and South tracks that are the same call.

    Candidates: same call type, first arrivals within ``window_s``. A candidate pair is
    accepted when a joint two-cable fit explains all picks (rms < ``max_rms_s``), which
    a wrong pairing almost never does. Each track is used at most once, best fit first.
    """
    by = {k: g for k, g in picks.groupby("call_key") if len(g) >= min_picks}
    meta = pd.DataFrame([(k, g.cable.iloc[0], g.call_type.iloc[0], g.t_abs.min()) for k, g in by.items()],
                        columns=["key", "cable", "type", "t_first"])
    n, s = meta[meta.cable == "North"], meta[meta.cable == "South"]
    trials = []
    for a in n.itertuples():
        for b in s[(s.type == a.type) & ((s.t_first - a.t_first).abs() < window_s)].itertuples():
            gn, gs = by[a.key], by[b.key]
            r = np.vstack([cables["North"].position(gn.dist), cables["South"].position(gs.dist)])
            fix = locate(r, np.concatenate([gn.t_abs, gs.t_abs]), c=c)
            trials.append((a.key, b.key, fix["rms_s"]))
    trials = pd.DataFrame(trials, columns=["north_key", "south_key", "joint_rms_s"]).sort_values("joint_rms_s")
    used_n, used_s, keep = set(), set(), []
    for t in trials.itertuples():
        if t.joint_rms_s < max_rms_s and t.north_key not in used_n and t.south_key not in used_s:
            used_n.add(t.north_key)
            used_s.add(t.south_key)
            keep.append(t.Index)
    return trials.loc[keep].reset_index(drop=True)


def run_L2(picks: pd.DataFrame, cables: dict[str, Cable], pairs: pd.DataFrame,
           c: float = C_DEFAULT) -> pd.DataFrame:
    """L-2: 3-D (x, y, z) fix using both cables for every associated call pair."""
    by = dict(tuple(picks.groupby("call_key")))
    out = []
    for _, p in pairs.iterrows():
        gn, gs = by[p.north_key], by[p.south_key]
        r = np.vstack([cables["North"].position(gn.dist), cables["South"].position(gs.dist)])
        t = np.concatenate([gn.t_abs, gs.t_abs])
        fix = locate(r, t, c=c, fit_z=True)
        out.append(Fix(p.north_key + "+" + p.south_key, "Both", gn.call_type.iloc[0], len(t), **fix))
    df = pd.DataFrame([f.__dict__ for f in out])
    return _add_lonlat(df, cables)


def fit_sound_speed(picks, cables, keys, speeds=np.arange(1440, 1541, 10)) -> pd.Series:
    """Total residual over a set of well-constrained calls as a function of c."""
    by = dict(tuple(picks.groupby("call_key")))
    res = {}
    for c in speeds:
        tot = []
        for k in keys:
            g = by[k]
            r = cables[g.cable.iloc[0]].position(g.dist)
            tot.append(locate(r, g.t_abs.to_numpy(), c=c)["rms_s"])
        res[c] = float(np.median(tot))
    return pd.Series(res, name="rms_s")


def _add_lonlat(df, cables):
    if df.empty:
        return df
    lon, lat = cables["North"].lonlat(df.x.to_numpy(), df.y.to_numpy())
    df["lon"], df["lat"] = lon, lat
    t = pd.Timestamp("2021-11-01") + pd.to_timedelta(df.t0, unit="s")
    df.insert(1, "utc_emission", t.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z")
    return df
