"""DCLDE 2027 Localization challenge: L-1 (2-D, one cable) and L-2 (3-D, two cables).

Usage:  python scripts/run_localization.py
Needs:  data/das/annotations and data/das/geometry  (python scripts/download.py das)
Writes: results/localization/*.csv and *.png
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

from dclde import das_loc as L

OUT = ROOT / "results" / "localization"
OUT.mkdir(parents=True, exist_ok=True)

picks = L.load_picks()
cables = {n: L.load_cable(n) for n in ("North", "South")}
print(f"{len(picks)} picks, {picks.call_key.nunique()} calls")

# 1. Calibrate the effective sound speed on the calls with the most picks.
counts = picks.groupby("call_key").size()
calib = counts[counts >= 20].sample(min(60, (counts >= 20).sum()), random_state=0).index
cs = L.fit_sound_speed(picks, cables, calib)
c = float(cs.idxmin())
print("sound-speed scan (rms s):\n", cs.round(4).to_string())
print(f"-> using c = {c:.0f} m/s")

# 2. L-1
l1 = L.run_L1(picks, cables, c=c)
# Quality flag: well-fitting, finite uncertainty under 5 km. End-on geometries leave range
# unconstrained and wander off to hundreds of km; those are kept but flagged.
l1["quality_ok"] = (l1.rms_s < 0.15) & (np.hypot(l1.sx, l1.sy) < 5e3)
l1.to_csv(OUT / "L1_fixes.csv", index=False)

# 3. Cross-cable association, then L-2
pairs = L.associate(picks, cables, c=c)
pairs.to_csv(OUT / "north_south_pairs.csv", index=False)
l2 = L.run_L2(picks, cables, pairs, c=c)
l2["quality_ok"] = (l2.rms_s < 0.15) & (np.hypot(l2.sx, l2.sy) < 5e3)
l2.to_csv(OUT / "L2_fixes.csv", index=False)

# 4. Score L-1 against L-2 (the best reference we have without released ground truth).
#    For each paired call, compare each cable's own 2-D fix (and its mirror alternative)
#    with the joint fix.
l2k = l2.assign(north_key=l2.call_key.str.split("+").str[0], south_key=l2.call_key.str.split("+").str[1])
rows = []
for side in ("north", "south"):
    m = l2k.merge(l1, left_on=f"{side}_key", right_on="call_key", suffixes=("_l2", ""))
    err = np.hypot(m.x - m.x_l2, m.y - m.y_l2)
    err_alt = np.hypot(m.x_alt - m.x_l2, m.y_alt - m.y_l2)
    rows.append(pd.DataFrame({"cable": side, "err_m": err, "alt_err_m": err_alt,
                              "right_side": err < err_alt, "mirror_ratio": m.mirror_ratio}))
score = pd.concat(rows, ignore_index=True)
score.to_csv(OUT / "L1_vs_L2.csv", index=False)
ok = score[score.right_side]

good = l1[l1.quality_ok]
summary = {
    "annotated_calls": int(picks.call_key.str.split("#").str[0].nunique()),
    "tracks_after_splitting": int(picks.call_key.nunique()),
    "L1_fixes": len(l1),
    "L1_quality_ok": int(l1.quality_ok.sum()),
    "L2_quality_ok": int(l2.quality_ok.sum()),
    "L1_median_rms_s": l1.rms_s.median(),
    "L1_median_1sigma_xy_m": np.nanmedian(np.hypot(l1.sx, l1.sy)),
    "L1_share_unambiguous (mirror_ratio>2)": (l1.mirror_ratio > 2).mean(),
    "L1_right_side_vs_L2": score.right_side.mean(),
    "L1_median_error_vs_L2_m (right side)": ok.err_m.median(),
    "north_south_pairs": len(pairs),
    "L2_fixes": len(l2),
    "L2_median_rms_s": l2.rms_s.median(),
    "L2_median_1sigma_xy_m": np.nanmedian(np.hypot(l2.sx, l2.sy)),
    "L2_median_depth_m": l2.z.median(),
    "L2_median_1sigma_z_m": np.nanmedian(l2.sz),
    "sound_speed_m_s": c,
}
m = score.assign(sep_m=score.err_m)
pd.Series(summary).to_csv(OUT / "summary.csv", header=["value"])
print(pd.Series(summary).to_string())

# 5. Map
fig, ax = plt.subplots(figsize=(10, 6))
for name, col in (("North", "tab:blue"), ("South", "tab:orange")):
    g = cables[name].geo
    ax.plot(g.x / 1e3, g.y / 1e3, color=col, lw=2, label=f"{name} cable")
ax.scatter(good.x / 1e3, good.y / 1e3, s=6, c="0.6", label="L-1 fixes (one cable)")
q2 = l2[l2.quality_ok]
sc = ax.scatter(q2.x / 1e3, q2.y / 1e3, s=14, c=(q2.t0 - q2.t0.min()) / 3600, cmap="viridis",
                label="L-2 fixes (both cables)")
fig.colorbar(sc, label="hours since first call")
ax.set(xlabel="x (km)", ylabel="y (km)", title="Fin whale call locations, OOI DAS, Nov 2021", aspect="equal",
       xlim=(-10, 130), ylim=(-25, 65))
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "map.png", dpi=150)

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.hist(ok.err_m / 1e3, bins=40, range=(0, 8))
ax.set(xlabel="L-1 (one cable) error vs L-2 fix (km)", ylabel="calls",
       title="L-1 accuracy, right-side fixes")
fig.tight_layout()
fig.savefig(OUT / "cable_agreement.png", dpi=150)
