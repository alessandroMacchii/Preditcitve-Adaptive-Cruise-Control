"""Costruisce outputs/elevation_cache.parquet in modo RESUMABLE.

Salva la cache dopo ogni batch: se lo script muore (timeout, 429, rete), basta
rilanciarlo e riprende dai soli punti mancanti. Una volta completa, il Notebook 1
la carica e salta del tutto le chiamate API.

Uso:  ./.venv/Scripts/python.exe build_elevation_cache.py
"""
import glob
import time
from pathlib import Path

import truststore
truststore.inject_into_ssl()  # verifica SSL via trust store del SO
import requests
import numpy as np
import pandas as pd

ROUND_DECIMALS = 3            # ~111 m, coerente con SRTM (deve combaciare col Notebook 1)
BATCH_SIZE = 100             # max consentito da Open-Meteo
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
    """GET con attesa paziente sul 429. Ritorna lista elevazioni o solleva."""
    last_err = None
    for attempt in range(12):
        try:
            r = requests.get(URL, params=params, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 0)) or RATE_LIMIT_WAIT
                print(f"    429: attendo {wait}s ({attempt+1}/12)...", flush=True)
                time.sleep(wait)
                last_err = "429"
                continue
            r.raise_for_status()
            return r.json()["elevation"]
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(2 * (2 ** min(attempt, 5)))
    raise RuntimeError(f"Batch fallito: {last_err!r}")


def main():
    coords = unique_coords()
    print(f"Punti unici (round {ROUND_DECIMALS}): {len(coords):,}", flush=True)

    # Riprendi: parti dalla cache esistente, tieni solo le elevazioni valide
    if CACHE.exists():
        prev = pd.read_parquet(CACHE)
        prev = prev[prev["elevation_m"].notna()]
        done = coords.merge(prev, on=["lat_round", "lon_round"], how="left")
    else:
        done = coords.copy()
        done["elevation_m"] = np.nan

    todo = done[done["elevation_m"].isna()].reset_index(drop=True)
    print(f"Gia' in cache: {len(done)-len(todo):,} | Da scaricare: {len(todo):,}", flush=True)

    if len(todo) == 0:
        print("Cache gia' completa.", flush=True)
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

        # salvataggio incrementale -> resumable
        done.to_parquet(CACHE, index=False)
        if bi % 5 == 0 or bi == n_batches:
            valid = done["elevation_m"].notna().sum()
            print(f"  batch {bi}/{n_batches} | validi {valid:,}/{len(done):,}", flush=True)
        time.sleep(SLEEP_BETWEEN)

    valid = done["elevation_m"].notna().sum()
    print(f"FATTO. Elevazioni valide: {valid:,}/{len(done):,} | "
          f"range {done['elevation_m'].min():.0f}-{done['elevation_m'].max():.0f} m", flush=True)


if __name__ == "__main__":
    main()
