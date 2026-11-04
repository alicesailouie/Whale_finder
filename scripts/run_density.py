"""DCLDE 2027 Density Estimation challenge (DE-1): right whale abundance in Cape Cod Bay.

Usage:  python scripts/run_density.py
Needs:  data/narw/  (python scripts/download.py narw)
Writes: results/density/*.csv and *.png
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dclde import narw as N

OUT = ROOT / "results" / "density"
OUT.mkdir(parents=True, exist_ok=True)

det = N.load_detections()
calls = N.associate(det)
calls.to_csv(OUT / "associated_calls.csv.gz", index=False)
daily = N.daily_counts(calls, det)
aer = N.load_aerial()
print(f"{len(det)} detections -> {len(calls)} distinct upcalls (paper: 485,880 -> 249,539)")
print("units per call:", calls.n_units.value_counts(normalize=True).sort_index().round(3).to_dict())

hours = N.FLIGHT_UTC[1] - N.FLIGHT_UTC[0]
cal = aer[aer.used].merge(daily, left_on="date", right_index=True)

# ---- Method A: cue counting with literature call rates (no aerial data used)
for r in (6.0, 8.0):
    daily[f"A_cuecount_r{r:.0f}"] = N.cue_count(daily.calls_flight_window, hours, call_rate=r)
    cal[f"A_r{r:.0f}"] = N.cue_count(cal.calls_flight_window, hours, call_rate=r)
# The call rate that would make method A unbiased on the aerial days (for interpretation only):
implied_rate = (cal.calls_flight_window * N.PRECISION / N.RECALL / hours / cal.whales)

# ---- Method B: quasi-Poisson calibration (paper's model family), every predictor x form
rows = []
for xcol in ("calls_flight_window", "detections_flight_window", "detections_unit5_flight_window"):
    for form in ("linear", "sqrt", "log"):
        m = N.fit_qpois(cal[xcol], cal.whales, cal.weight, form)
        rmse = N.weighted_rmse(N.predict(m, cal[xcol]), cal.whales, cal.weight)
        cv, _ = N.loo_rmse(cal[xcol], cal.whales, cal.weight, form)
        rows.append(dict(predictor=xcol, form=form, alpha=m["beta"][0], beta=m["beta"][1],
                         dispersion=m["phi"], RMSE=rmse, CV_RMSE=cv))
models = pd.DataFrame(rows).sort_values("CV_RMSE")
models.to_csv(OUT / "model_comparison.csv", index=False)
print(models.round(3).to_string(index=False))

best = models.iloc[0]
mod = N.fit_qpois(cal[best.predictor], cal.whales, cal.weight, best.form)
daily["B_estimate"] = N.predict(mod, daily[best.predictor])
lo, hi = N.prediction_interval(mod, daily[best.predictor].to_numpy())
daily["B_pi95_lo"], daily["B_pi95_hi"] = lo, hi
daily["extrapolated"] = daily[best.predictor] > cal[best.predictor].max()
daily = daily.join(aer.set_index("date")[["whales", "used"]].rename(columns={"whales": "aerial_count"}))
daily.to_csv(OUT / "daily_abundance.csv")

_, loo_err = N.loo_rmse(cal[best.predictor], cal.whales, cal.weight, best.form)
summary = pd.Series({
    "detections": len(det), "distinct_upcalls": len(calls),
    "aerial_days_used": len(cal),
    "A_r6_RMSE_vs_aerial": N.weighted_rmse(cal.A_r6, cal.whales, cal.weight),
    "A_r8_RMSE_vs_aerial": N.weighted_rmse(cal.A_r8, cal.whales, cal.weight),
    "A_r6_median_ratio_to_aerial": float(np.median(cal.A_r6 / cal.whales)),
    "implied_call_rate_median_per_whale_h": float(implied_rate.median()),
    "implied_call_rate_IQR": f"{implied_rate.quantile(.25):.1f}-{implied_rate.quantile(.75):.1f}",
    "B_best_model": f"{best.form}({best.predictor})",
    "B_RMSE": best.RMSE, "B_CV_RMSE": best.CV_RMSE,
    "B_LOO_median_abs_error_whales": float(np.median(np.abs(loo_err))),
    "B_peak_day": str(daily.B_estimate.idxmax().date()),
    "B_peak_estimate": float(daily.B_estimate.max()),
    "B_mean_Feb17_May15": float(daily.loc["2019-02-17":"2019-05-15", "B_estimate"].mean()),
})
summary.to_csv(OUT / "summary.csv", header=["value"])
print(summary.to_string())

# ---- figures
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.fill_between(daily.index, daily.B_pi95_lo, daily.B_pi95_hi, color="tab:blue", alpha=0.2, label="B: 95% prediction interval")
ax.plot(daily.index, daily.B_estimate, color="tab:blue", lw=1.5, label=f"B: calibrated ({best.form}, associated calls)")
ax.plot(daily.index, daily.A_cuecount_r6, color="tab:green", lw=1, ls="--", label="A: cue counting, 6 calls/whale/h")
u = daily[daily.used == True]  # noqa: E712
ax.scatter(u.index, u.aerial_count, color="tab:red", marker="*", s=90, zorder=5, label="aerial survey count")
ax.set(ylabel="right whales in Cape Cod Bay", title="North Atlantic right whale abundance, Cape Cod Bay, 2019",
       xlim=(pd.Timestamp("2019-02-15"), pd.Timestamp("2019-06-05")))
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "daily_abundance.png", dpi=150)

fig, ax = plt.subplots(figsize=(5, 4))
xs = np.linspace(max(cal[best.predictor].min(), 1), cal[best.predictor].max() * 1.3, 200)
ax.plot(xs, N.predict(mod, xs), "k-")
plo, phi_ = N.prediction_interval(mod, xs)
ax.fill_between(xs, plo, phi_, color="0.85")
ax.scatter(cal[best.predictor], cal.whales, s=40 * cal.weight, color="tab:red", zorder=3)
ax.set(xlabel="distinct upcalls, 09:12-15:09 EST", ylabel="whales seen from the air", title="Calibration (16 survey days)")
fig.tight_layout()
fig.savefig(OUT / "calibration.png", dpi=150)
