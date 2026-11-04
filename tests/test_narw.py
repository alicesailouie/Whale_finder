"""The TDoA feasibility test accepts a real source in the bay and rejects impossible timings."""
import numpy as np
from dclde import narw as N


def _arrivals(lat, lon):
    m_lat, m_lon = 111_320.0, 111_320.0 * np.cos(np.radians(41.89))
    return {k: np.hypot((lat - a) * m_lat, (lon - b) * m_lon) / N.C for k, (a, b) in N.MARU.items()}


def test_real_source_is_feasible():
    t = _arrivals(41.90, -70.30)
    chans = tuple(sorted(t, key=t.get))
    assert N.feasible(chans, tuple(t[c] for c in chans))


def test_impossible_tdoa_rejected():
    # Units 1 and 3 are ~22 km apart (max TDoA ~15 s); 30 s apart is not one call.
    assert not N.feasible((1, 3), (0.0, 30.0))


def test_quasi_poisson_recovers_coefficients():
    rng = np.random.default_rng(0)
    x = rng.uniform(50, 2000, 400)
    y = rng.poisson(np.exp(-1.5 + 0.9 * np.log(x)))
    m = N.fit_qpois(x, y, np.ones_like(x), "log")
    assert abs(m["beta"][1] - 0.9) < 0.05
