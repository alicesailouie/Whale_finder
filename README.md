# DCLDE 2027 challenges: an unofficial, for-fun attempt

Code and results for the three [DCLDE 2027 challenges](https://www.dclde.org/challenges-and-data-sets),
built on the public datasets in NOAA's `noaa-passive-bioacoustic` bucket. This is not a submission: there
is no released test set yet, so every number below comes from held-out training data, which is described
case by case.

| Challenge | Task | Approach | Headline result |
|---|---|---|---|
| **L-1** | Fin whale 2-D localisation, one DAS cable | Least-squares time-of-arrival fit with a grid-search start | Median 355 m from the two-cable fix when the left/right side is right; side correct 65% of the time |
| **L-2** | Fin whale 3-D localisation, two DAS cables | Joint fit with cross-cable call association | 206 calls; median ±213 m (1σ) horizontally; depth not resolvable |
| **DC-1** | Detect killer whale calls | Small CNN on log-mel spectrograms, tested on unseen recorders | _see below_ |
| **DC-2** | Classify killer whale ecotype | Same CNN, 5 ecotypes | _see below_ |
| **DE-1** | Right whale abundance, Cape Cod Bay | TDoA call association + quasi-Poisson calibration to aerial counts | Leave-one-out error ±13 whales (RMSE); reproduces the published pipeline |

---

## Localization (L-1, L-2): fin whales on fibre-optic cables

![DAS waterfall](results/localization/waterfall.png)

Distributed acoustic sensing (DAS) turns a telecom fibre into tens of thousands of strain sensors.
Above, each "V" is one fin whale call sweeping along 54 km of the OOI North cable off Oregon. The tip
of the V is the point on the cable closest to the whale. The challenge provides the hand-picked arrival
times (cyan circles), so localisation never touches the 103 GB of raw strain data.

**Model.** For a whale at **s** calling at time *t₀*, a pick at cable point **rᵢ** arrives at
*tᵢ = t₀ + |s − rᵢ| / c*. Fixes come from robust least squares (soft-L1), started from a 1 km grid
search so the fit cannot get stuck in a local minimum.

**What I found in the data**
- Some annotation `call_id`s contain two different calls about 40 s apart. Picks are split into
  physically consistent tracks (an arrival curve cannot move along the fibre faster than 1/*c*).
- `call_id` is not shared between the North and South cables. Calls are paired across cables by
  testing whether one joint fit explains both sets of picks (RMS < 0.15 s), best fits first.
- The effective sound speed that minimises residuals is **1450 m/s**, below true water sound speed. That
  is expected, because straight-line rays ignore refraction and the seafloor path to the cable.
- **No ground-truth positions are in the bucket yet**, although the dataset description says they exist.
  L-1 is therefore scored against L-2, which is an independent, better-constrained estimate.

![map](results/localization/map.png)

| Metric | Value |
|---|---|
| Annotated calls → tracks after splitting | 794 → 801 |
| L-1 fixes / passing quality check | 769 / 538 |
| L-1 median residual | 71 ms |
| L-1 calls with an unambiguous side (mirror solution ≥ 2× worse) | 17% |
| L-1 side correct (vs L-2) | 65% |
| L-1 median error vs L-2, correct side | 355 m |
| L-2 cross-cable pairs | 206 |
| L-2 median residual | 91 ms |
| L-2 median 1σ horizontal | 213 m |
| L-2 depth | median −12 m, but 1σ ≈ 600 m, so not resolved |

On the map, the L-2 fixes form tight clusters (individual whales). One cluster south of the cables moves
steadily south-west over about 10 hours: a whale swimming while it calls.

**Limitations.** A single cable mostly cannot tell which side the whale is on, because the mirror image
fits almost as well. Depth is not recoverable with a constant-speed straight-ray model. A ray-traced or
modal propagation model would be the next step.

---

## Detection & Classification (DC-1, DC-2): killer whales in the Salish Sea

_Results pending; see `results/killer_whales/`._

---

## Density Estimation (DE-1): right whales in Cape Cod Bay

Five recorders (MARUs) logged 116 days in 2019. A CNN produced 495,038 upcall detections (score ≥ 0.75),
and on 16 days aircraft photographed every right whale in the bay.

1. **Association.** One call is often heard on several recorders. A detection set counts as one call if
   some source location inside the bay explains its time differences (τ = 0.5 s, *c* = 1470 m/s), and
   recorders that missed it are farther away. This yields **249,395 distinct upcalls**; the paper reports
   249,539.
2. **Counting.** Distinct calls per day in the median flight window (09:12–15:09 EST).
3. **Estimating abundance** in two ways:

| Method | Uses aerial data? | Error vs aerial counts |
|---|---|---|
| A. Cue counting, 6 upcalls per whale per hour (literature) | no | RMSE 41 whales; underestimates ~5× |
| B. Quasi-Poisson calibration, square-root model | yes | RMSE 11.4; **leave-one-out RMSE 13.2** |

![daily abundance](results/density/daily_abundance.png)

The comparison shows something interesting. To match the aerial counts, method A needs a population-level
daytime call rate of only **≈1.2 upcalls per whale per hour** (IQR 0.9–1.7), much less than the 6–8 per hour
often quoted. Right whales call much more at night, and many whales are silent at any given moment, so
borrowing a call rate is the weakest link in pure cue counting. Calibrating to the aerial counts avoids
the problem.

Our fitted coefficients match the paper's (for example, linear model α = 2.69, β = 0.003, against 2.69 and
0.00275 in the paper). The log and square-root models are effectively tied (CV RMSE 13.26 vs 13.25); the
paper chose the log model. The spikes on the plot, such as 27 April, are days with more calls than any
calibration day, so those estimates are extrapolations and are flagged in `daily_abundance.csv`.

---

## Running it

```bash
pip install -r requirements.txt
python scripts/download.py            # ~450 MB of annotations, geometry and detections
python scripts/run_localization.py    # L-1, L-2      (~1 min)
python scripts/run_density.py         # DE-1          (~1 min)
python scripts/build_kw_dataset.py    # streams ~9,600 killer whale clips (~15-30 min, ~10 GB cached)
python scripts/run_killer_whales.py   # DC-1, DC-2    (CPU, ~1 h)
python -m pytest
```

Data stays in `data/` (git-ignored). Raw DAS and the 1.6 TB of killer whale audio are never downloaded
in bulk: WAV clips are read with HTTP range requests.

```
src/dclde/
  gcs.py        public-bucket access (listing, download, byte ranges)
  das_loc.py    DAS geometry, pick loading, track splitting, L-1/L-2 solvers, association
  kw_data.py    killer whale annotations, clip streaming, log-mel features
  kw_model.py   CNN, augmentation, training
  narw.py       right whale detections, TDoA association, cue counting, quasi-Poisson
scripts/        one script per challenge + download + figures
results/        CSVs and figures produced by the scripts
tests/          synthetic checks of the solvers and models
```

## Data and credits

All data are from the DCLDE 2027 organisers via NOAA NCEI (CC-BY-4.0). Please cite the original datasets:

- DAS fin whales: DOI [10.25921/v2vh-8w16](https://doi.org/10.25921/v2vh-8w16); Goestchel et al. (JASA, accepted). Data from the OOI RCA DAS experiment.
- Killer whales: DOI [10.25921/15ey-mh50](https://doi.org/10.25921/15ey-mh50); [Palmer et al. 2025, *Sci. Data*](https://www.nature.com/articles/s41597-025-05281-5).
- Right whales: DOIs [10.25921/ab3a-c842](https://doi.org/10.25921/ab3a-c842), [10.25921/x68p-cj94](https://doi.org/10.25921/x68p-cj94); [Garcia, Tolkova et al. 2025, *Endang. Species Res.* 56:101](https://www.int-res.com/abstracts/esr/v56/esr01384). Aerial surveys by the Center for Coastal Studies (NOAA permit 19315-01).

Code: MIT licence (see `LICENSE`).
