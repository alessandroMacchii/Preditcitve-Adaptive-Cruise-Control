"""Builds outputs/elevation_cache.parquet in a RESUMABLE way.

Saves the cache after each batch: if the script dies (timeout, 429, network), just
re-launch it and it resumes from the missing points only. Once complete, Notebook 1
loads it and skips the API calls entirely.

Usage:  ./.venv/Scripts/python.exe build_elevation_cache.py
"""
import glob
import time
from pathlib import Path

import truststore
truststore.inject_into_ssl()  # SSL verification via the OS trust store
import requests
import numpy as np
import pandas as pd

ROUND_DECIMALS = 3            # ~111 m, consistent with SRTM (must match Notebook 1)
BATCH_SIZE = 100             # max allowed by Open-Meteo
SLEEP_BETWEEN = 1.5
RATE_LIMIT_WAIT = 60
URL = "https://api.open-meteo.com/v1/elevation"

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE = OUTPUT_DIR / "elevation_cache.parquet"


def unique_coords():
    files = sorted(glob.glob("content/**/*.parquet", recursive=True))
    df = pd.concat(
        (pd.read_parquet(f, columns=["Latitude_deg", "Longitude_deg"]) for f in files),
        ignore_index=True,
    )
    df["lat_round"] = df["Latitude_deg"].round(ROUND_DECIMALS)
    df["lon_round"] = df["Longitude_deg"].round(ROUND_DECIMALS)
    return df[["lat_round", "lon_round"]].drop_duplicates().reset_index(drop=True)


def fetch_one(params):
    """GET with patient waiting on the 429. Returns a list of elevations or raises."""
    last_err = None
    for attempt in range(12):
        try:
            r = requests.get(URL, params=params, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 0)) or RATE_LIMIT_WAIT
                print(f"    429: waiting {wait}s ({attempt+1}/12)...", flush=True)
                time.sleep(wait)
                last_err = "429"
                continue
            r.raise_for_status()
            return r.json()["elevation"]
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(2 * (2 ** min(attempt, 5)))
    raise RuntimeError(f"Batch failed: {last_err!r}")


def main():
    coords = unique_coords()
    print(f"Unique points (round {ROUND_DECIMALS}): {len(coords):,}", flush=True)

    # Resume: start from the existing cache, keep only the valid elevations
    if CACHE.exists():
        prev = pd.read_parquet(CACHE)
        prev = prev[prev["elevation_m"].notna()]
        done = coords.merge(prev, on=["lat_round", "lon_round"], how="left")
    else:
        done = coords.copy()
        done["elevation_m"] = np.nan

    todo = done[done["elevation_m"].isna()].reset_index(drop=True)
    print(f"Already in cache: {len(done)-len(todo):,} | To download: {len(todo):,}", flush=True)

    if len(todo) == 0:
        print("Cache already complete.", flush=True)
        done.to_parquet(CACHE, index=False)
        return

    n_batches = (len(todo) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi, start in enumerate(range(0, len(todo), BATCH_SIZE), 1):
        batch = todo.iloc[start:start + BATCH_SIZE]
        params = {
            "latitude": ",".join(map(str, batch["lat_round"].values)),
            "longitude": ",".join(map(str, batch["lon_round"].values)),
        }
        elevs = fetch_one(params)
        done.loc[batch.index, "elevation_m"] = elevs[:len(batch)]

        # incremental save -> resumable
        done.to_parquet(CACHE, index=False)
        if bi % 5 == 0 or bi == n_batches:
            valid = done["elevation_m"].notna().sum()
            print(f"  batch {bi}/{n_batches} | valid {valid:,}/{len(done):,}", flush=True)
        time.sleep(SLEEP_BETWEEN)

    valid = done["elevation_m"].notna().sum()
    print(f"DONE. Valid elevations: {valid:,}/{len(done):,} | "
          f"range {done['elevation_m'].min():.0f}-{done['elevation_m'].max():.0f} m", flush=True)


if __name__ == "__main__":
    main()
