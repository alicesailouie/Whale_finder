"""Download the small parts of the DCLDE 2027 data needed by this repo (~450 MB in total).

Usage:  python scripts/download.py [das] [kw] [narw]      (no argument = all three)

Raw DAS strain files (103 GB) and full killer-whale audio (1.6 TB) are NOT downloaded:
localisation only needs the arrival-time picks, and killer-whale clips are streamed with
byte-range requests by scripts/build_kw_dataset.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dclde import gcs

P = gcs.ROOT
SETS = {
    "das": [P + "dclde_2027_das-finwhale-localization/annotations/",
            P + "dclde_2027_das-finwhale-localization/geometry/"],
    "kw": [P + "dclde_2027_killer_whales/Annotations.csv",
           P + "dclde_2027_killer_whales/CC-BY-4.0.txt"],
    "narw": [P + "dclde2027_narw_detections_in_cape_cod_bay/",
             P + "dclde2027_acoustic_monitoring_in_cape_cod_bay/metadata/"],
}

for key in sys.argv[1:] or list(SETS):
    for prefix in SETS[key]:
        files = gcs.fetch_prefix(prefix)
        print(f"{key}: {len(files)} files from {prefix}")
    if key == "kw":
        from dclde import kw_data
        idx = kw_data.build_index()
        print(f"kw: indexed {len(idx)} audio files")
