"""Synthetic check: plant a whale, simulate picks on the real cable geometry, recover it."""
import numpy as np
from dclde import das_loc as L


def _sim(cable, src, dists, c=L.C_DEFAULT, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    r = cable.position(dists)
    t = 100.0 + np.linalg.norm(r - src, axis=1) / c + rng.normal(0, noise, len(dists))
    return r, t


def test_L1_recovers_source_2d():
    cab = L.load_cable("South")
    src = np.array([60e3, 25e3, -20.0])
    r, t = _sim(cab, src, np.linspace(40e3, 80e3, 20))
    fix = L.locate(r, t)
    assert np.hypot(fix["x"] - src[0], fix["y"] - src[1]) < 500


def test_L2_recovers_source_3d():
    n, s = L.load_cable("North"), L.load_cable("South")
    src = np.array([50e3, 22e3, -40.0])
    rn, tn = _sim(n, src, np.linspace(30e3, 60e3, 15), seed=1)
    rs, ts = _sim(s, src, np.linspace(30e3, 70e3, 15), seed=2)
    fix = L.locate(np.vstack([rn, rs]), np.concatenate([tn, ts]), fit_z=True)
    assert np.hypot(fix["x"] - src[0], fix["y"] - src[1]) < 500
