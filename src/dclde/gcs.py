"""Minimal, dependency-free access to the public NOAA passive-bioacoustic bucket.

The DCLDE 2027 data live in ``gs://noaa-passive-bioacoustic/dclde/2027``.
The bucket is public, so plain HTTPS works and no Google credentials are needed.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterator

BUCKET = "noaa-passive-bioacoustic"
ROOT = "dclde/2027/"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
MEDIA = f"https://storage.googleapis.com/{BUCKET}/"

DATA_DIR = Path(os.environ.get("DCLDE_DATA", Path(__file__).resolve().parents[2] / "data"))


def list_objects(prefix: str) -> Iterator[tuple[str, int]]:
    """Yield ``(object_name, size_bytes)`` for every object under ``prefix``."""
    token = None
    while True:
        q = {"prefix": prefix, "fields": "items(name,size),nextPageToken", "maxResults": "1000"}
        if token:
            q["pageToken"] = token
        with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(q)}") as r:
            d = json.load(r)
        for it in d.get("items", []):
            yield it["name"], int(it["size"])
        token = d.get("nextPageToken")
        if not token:
            return


def url(name: str) -> str:
    return MEDIA + urllib.parse.quote(name)


# Short local folder names keep paths well under the Windows 260-character limit.
SHORT = {
    "dclde_2027_das-finwhale-localization/": "das/",
    "dclde_2027_killer_whales/": "kw/",
    "dclde2027_acoustic_monitoring_in_cape_cod_bay/": "narw_audio/",
    "dclde2027_narw_detections_in_cape_cod_bay/": "narw/",
}


def local_path(name: str) -> Path:
    rel = name[len(ROOT):] if name.startswith(ROOT) else name
    for long, short in SHORT.items():
        if rel.startswith(long):
            rel = short + rel[len(long):]
            break
    return DATA_DIR / rel


def fetch(name: str, overwrite: bool = False) -> Path:
    """Download one object into ``data/`` (mirroring the bucket layout) and return its path."""
    dst = local_path(name)
    if dst.exists() and not overwrite:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url(name)) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dst)
    return dst


def fetch_range(name: str, start: int, stop: int) -> bytes:
    """Read bytes ``[start, stop)`` of an object with an HTTP Range request."""
    req = urllib.request.Request(url(name), headers={"Range": f"bytes={start}-{stop - 1}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def fetch_prefix(prefix: str, suffixes: tuple[str, ...] = (), max_bytes: int | None = None) -> list[Path]:
    """Download every object under ``prefix`` (optionally filtered by suffix)."""
    out = []
    for name, size in list_objects(prefix):
        if suffixes and not name.lower().endswith(suffixes):
            continue
        if max_bytes is not None and size > max_bytes:
            continue
        out.append(fetch(name))
    return out
