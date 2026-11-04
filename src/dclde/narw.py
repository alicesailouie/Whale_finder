"""North Atlantic right whale abundance in Cape Cod Bay (DCLDE 2027 challenge DE-1).

Pipeline (following and extending Garcia, Tolkova et al. 2025, Endang Species Res 56:101):

1. Load the CNN upcall detections from the 5 MARUs; keep score >= 0.75.
2. Associate duplicate detections of the same call across recorders with a
   time-difference-of-arrival (TDoA) feasibility test on a grid over the bay.
3. Count distinct upcalls per day inside the median aerial-survey window.
4. Estimate abundance two ways:
   A. cue counting from literature call rates (no aerial data used), and
   B. calibration to aerial counts with quasi-Poisson regression, checked by
      leave-one-out cross-validation.
"""
from __future__ import annotations

import glob
from itertools import combinations, product

import numpy as np
import pandas as pd

from .gcs import DATA_DIR

NARW_DIR = DATA_DIR / "narw"

# Recorder positions (paper, Table 1): unit -> (lat, lon)
MARU = {1: (41.9337, -70.2286), 2: (41.9466, -70.4002), 3: (41.8413, -70.4693),
        4: (41.8397, -70.3148), 5: (41.8949, -70.3592)}
C = 1470.0              # m/s
TAU = 0.5               # s, admissible TDoA error
WINDOW = 15.3           # s, max TDoA in the array
THRESH = 0.75           # detector score threshold
PRECISION, RECALL, FP_PER_H = 0.857, 0.896, 4.8   # detector performance at THRESH (paper, Supplement)
FLIGHT_UTC = (14 + 12 / 60, 20 + 9 / 60)          # 09:12-15:09 EST, median flight interval
BAY = dict(lat=(41.74, 42.08), lon=(-70.62, -70.02), step=0.005)


# ----------------------------------------------------------------------------- detections
def load_detections(threshold: float = THRESH) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(NARW_DIR / "data" / "*.selections.txt"))):
        d = pd.read_csv(f, sep="\t", usecols=["Channel", "Begin File", "File Offset (s)", "Score"])
        rows.append(d[d.Score >= threshold])
    d = pd.concat(rows, ignore_index=True)
    stamp = d["Begin File"].str.extract(r"_(\d{8}_\d{6})Z")[0]
    d["t"] = pd.to_datetime(stamp, format="%Y%m%d_%H%M%S") + pd.to_timedelta(d["File Offset (s)"], unit="s")
    d = d.rename(columns={"Channel": "ch", "Score": "score"})[["t", "ch", "score"]]
    return d.sort_values("t").reset_index(drop=True)


# ----------------------------------------------------------------------------- association
def _toa_grid():
    lat = np.arange(*BAY["lat"], BAY["step"])
    lon = np.arange(*BAY["lon"], BAY["step"])
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    m_lat = 111_320.0
    m_lon = 111_320.0 * np.cos(np.radians(41.89))
    T = {}
    for k, (la, lo) in MARU.items():
        T[k] = np.hypot((LA - la) * m_lat, (LO - lo) * m_lon).ravel() / C
    return T


_T = None


def feasible(chans: tuple[int, ...], times: tuple[float, ...]) -> bool:
    """Can one source inside the bay explain these arrivals (and the silent units)?"""
    global _T
    if _T is None:
        _T = _toa_grid()
    ok = np.ones_like(_T[1], bool)
    ref_c, ref_t = chans[0], times[0]
    for c, t in zip(chans[1:], times[1:]):
        ok &= np.abs((_T[c] - _T[ref_c]) - (t - ref_t)) <= TAU
        if not ok.any():
            return False
    silent = [k for k in MARU if k not in chans]
    for k in chans:
        for l in silent:
            ok &= _T[k] < _T[l] + TAU   # a unit that heard it should not be much farther than one that didn't
    return bool(ok.any())


def associate(det: pd.DataFrame) -> pd.DataFrame:
    """Greedy TDoA association. Returns one row per distinct upcall.

    Detections are visited in time order. For the earliest unassigned detection, all
    unassigned detections on other units within ``WINDOW`` are candidates; the largest
    feasible set (one detection per unit, 5 down to 2) is accepted. A detection that
    cannot be matched is a call heard on one unit only.
    """
    t = (det.t - det.t.iloc[0]).dt.total_seconds().to_numpy()
    ch = det.ch.to_numpy()
    used = np.zeros(len(det), bool)
    calls_t, calls_n = [], []
    j_hi = 0
    for i in range(len(det)):
        if used[i]:
            continue
        while j_hi < len(det) and t[j_hi] <= t[i] + WINDOW:
            j_hi += 1
        cand = [j for j in range(i + 1, j_hi) if not used[j] and ch[j] != ch[i]]
        by_ch: dict[int, list[int]] = {}
        for j in cand:
            by_ch.setdefault(int(ch[j]), []).append(j)
        best = None
        for size in range(min(len(by_ch), 4), 0, -1):
            for chs in combinations(sorted(by_ch), size):
                for pick in product(*(by_ch[c][:3] for c in chs)):   # at most 3 options per unit
                    if feasible((int(ch[i]),) + chs, (t[i],) + tuple(t[p] for p in pick)):
                        best = pick
                        break
                if best:
                    break
            if best:
                break
        used[i] = True
        if best:
            used[list(best)] = True
        calls_t.append(det.t.iloc[i])
        calls_n.append(1 + (len(best) if best else 0))
    return pd.DataFrame({"t": calls_t, "n_units": calls_n})


# ----------------------------------------------------------------------------- counts
def daily_counts(calls: pd.DataFrame, det: pd.DataFrame) -> pd.DataFrame:
    def in_window(ts):
        h = ts.dt.hour + ts.dt.minute / 60 + ts.dt.second / 3600
        return (h >= FLIGHT_UTC[0]) & (h < FLIGHT_UTC[1])

    out = pd.DataFrame({
        "calls_all_day": calls.groupby(calls.t.dt.normalize()).size(),
        "calls_flight_window": calls[in_window(calls.t)].groupby(calls.t.dt.normalize()).size(),
        "detections_flight_window": det[in_window(det.t)].groupby(det.t.dt.normalize()).size(),
        "detections_unit5_flight_window": det[in_window(det.t) & (det.ch == 5)].groupby(det.t.dt.normalize()).size(),
    })
    days = pd.date_range("2019-02-17", "2019-06-12", freq="D")
    out = out.reindex(days).fillna(0).astype(int)
    out.index.name = "date"
    return out


def load_aerial(min_tracklines: int = 12) -> pd.DataFrame:
    a = pd.read_csv(NARW_DIR / "other" / "AerialSurvey.csv")
    a["date"] = pd.to_datetime(a.DATE, format="%d-%b-%y")
    a = a.rename(columns={"Total # of Individuals": "whales", "Num track lines surveyed": "tracklines"})
    a["weight"] = a.tracklines / 15
    a["used"] = a.tracklines >= min_tracklines
    return a[["date", "whales", "tracklines", "weight", "used"]]


# ----------------------------------------------------------------------------- method A
def cue_count(n_calls: np.ndarray, hours: float, call_rate: float = 6.0,
              precision: float = PRECISION, recall: float = RECALL) -> np.ndarray:
    """Abundance = true calls / (hours x per-whale call rate).

    Assumes every call made in the bay is within range of at least one unit (the
    array spans 22 km and calls are associated across 4-5 units, so detection
    range covers most of the 1560 km^2 bay), so P(detect) ~ detector recall.
    """
    return np.asarray(n_calls) * precision / recall / (hours * call_rate)


# ----------------------------------------------------------------------------- method B
def _design(x, form):
    x = np.asarray(x, float)
    f = {"linear": x, "sqrt": np.sqrt(x), "log": np.log(np.maximum(x, 1))}[form]
    return np.column_stack([np.ones_like(f), f])


def fit_qpois(x, y, w, form="log"):
    """Weighted Poisson GLM (log link) by IRLS; quasi-Poisson dispersion estimated afterwards."""
    X, y, w = _design(x, form), np.asarray(y, float), np.asarray(w, float)
    beta = np.array([np.log(max(y.mean(), 1)), 0.0])
    for _ in range(100):
        mu = np.exp(X @ beta)
        z = X @ beta + (y - mu) / mu
        W = w * mu
        new = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * z))
        if np.max(np.abs(new - beta)) < 1e-10:
            beta = new
            break
        beta = new
    mu = np.exp(X @ beta)
    phi = np.sum(w * (y - mu) ** 2 / mu) / (len(y) - 2)
    cov = phi * np.linalg.inv(X.T @ ((w * mu)[:, None] * X))
    return dict(beta=beta, phi=phi, cov=cov, form=form)


def predict(model, x):
    return np.exp(_design(x, model["form"]) @ model["beta"])


def prediction_interval(model, x, n_boot=4000, seed=0, level=0.95):
    """Parametric bootstrap: draw coefficients, then quasi-Poisson noise (negative binomial
    with variance phi*mu) around each predicted mean."""
    rng = np.random.default_rng(seed)
    B = rng.multivariate_normal(model["beta"], model["cov"], n_boot)
    mu = np.exp(_design(x, model["form"]) @ B.T)                 # (n, boot)
    phi = max(model["phi"], 1.0 + 1e-6)
    p = 1 / phi
    r = mu * p / (1 - p)
    sims = rng.negative_binomial(np.maximum(r, 1e-9), p)
    a = (1 - level) / 2
    return np.quantile(sims, a, axis=1), np.quantile(sims, 1 - a, axis=1)


def loo_rmse(x, y, w, form="log"):
    x, y, w = map(np.asarray, (x, y, w))
    err = []
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        mod = fit_qpois(x[m], y[m], w[m], form)
        err.append(predict(mod, x[i:i + 1])[0] - y[i])
    err = np.array(err)
    return float(np.sqrt(np.sum(w * err ** 2) / np.sum(w))), err


def weighted_rmse(pred, y, w):
    return float(np.sqrt(np.sum(w * (np.asarray(pred) - y) ** 2) / np.sum(w)))
