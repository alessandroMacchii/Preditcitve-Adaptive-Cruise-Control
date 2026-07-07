# Data analysis — Vehicle Energy Dataset (enriched VED)

> Complete EDA of `outputs/ved_enriched.parquet` (NB1 output). All the numbers are **real**,
> computed on the whole dataset. It serves to understand what the data looks like and why certain
> modeling choices were made. For the exam picture see `PROJECT_REPORT.md`.

---

## 1. At a glance

| | |
|---|---|
| Rows | **17,922,869** |
| Columns | 21 |
| Memory in RAM | ~2.06 GB |
| Vehicles | **299** |
| Trips | **26,285** |
| Powertrain | ICE 58% · HEV 30% · PHEV 12% |
| Period | 2017-11-01 → 2018-11-10 (**374 days**) |
| Area | Ann Arbor (MI), ~12×10 km |
| Total fleet distance | ~**138,827 km** |
| Sample frequency | median **600 ms** (irregular) |

Origin: real OBD-II telemetry; elevation is **not** in the raw data → derived in NB1 (Open-Meteo /
SRTM), from which `elevation_m` and `slope`. Also added `accel_kmh_s`, `dist_m`, `dt_ms`.

## 2. Schema (21 columns)

- **Keys/time:** `VehId` (int), `Trip` (int), `Timestampms` (ms intra-trip), `Datetime`
  (timestamp, **constant per trip** = trip start).
- **Position/terrain:** `Latitude_deg`, `Longitude_deg`, `elevation_m`, `slope` (fraction).
- **Dynamics:** `Vehicle_Speed_km_per_h`, `accel_kmh_s`.
- **Engine:** `Engine_RPM_RPM`, `Absolute_Load_pct`, `MAF_g_per_sec`, `log_MAF`.
- **Diagnostics:** `Short_Term_Fuel_Trim_Bank_1_pct`, `Long_Term_Fuel_Trim_Bank_1_pct`.
- **Context:** `OAT_DegC` (outside temperature), `EngineType`, `Generalized_Weight`.
- **Per-row derived:** `dist_m`, `dt_ms` (distance and Δt from the previous point).

### 2.1 The 18 columns of the **raw** file (original VED)

Examples taken from a real row (`content/VED_171101_week.parquet`, vehicle moving at 40 km/h).

| Column | Type | Description | Example |
|---|---|---|---|
| `VehId` | int | Anonymous ID of the vehicle in the fleet (299 in total) | `8` |
| `Trip` | int | Trip ID for that vehicle (restarts from scratch for each `VehId`) | `706` |
| `Timestampms` | int | Milliseconds since the trip start (**intra-trip time**, restarts from 0 at each trip) | `0` |
| `Latitude_deg` | float | GPS latitude in decimal degrees (Ann Arbor area) | `42.2776` |
| `Longitude_deg` | float | GPS longitude in decimal degrees | `-83.6988` |
| `Vehicle_Speed_km_per_h` | float | Vehicle speed (km/h) | `40.0` |
| `MAF_g_per_sec` | float | **Mass Air Flow**: mass of air taken in by the engine (g/s) → **a fuel-consumption proxy** (valid only for combustion engines) | `22.13` |
| `Engine_RPM_RPM` | float | Engine speed (rpm) | `2285` |
| `Absolute_Load_pct` | float | **Absolute load** of the engine: % of the max air flow = "how hard it pushes" | `49.0` |
| `OAT_DegC` | float | **Outside Air Temperature**: outside air temperature (°C) | `6.25` |
| `Short_Term_Fuel_Trim_Bank_1_pct` | float | **Short-term** injection correction, cylinder bank 1 (%): the ECU adjusts the mixture instant by instant | `-3.91` |
| `Short_Term_Fuel_Trim_Bank_2_pct` | float | As above, **cylinder bank 2** (V engines) † | `-3.13` |
| `Long_Term_Fuel_Trim_Bank_1_pct` | float | **Long-term** correction (average learned over time), bank 1 (%) | `-3.13` |
| `Long_Term_Fuel_Trim_Bank_2_pct` | float | As above, bank 2 † | `-2.34` |
| `Datetime` | datetime | **Trip-start** timestamp (**constant** within the same trip) | `2017-11-01 14:04:46` |
| `EngineType` | str | Powertrain type: **ICE / HEV / PHEV** | `ICE` |
| `Generalized_Weight` | float | "Generalized" vehicle weight (kg, rounded for anonymity) | `2500` |
| `log_MAF` | float | Natural logarithm of `(MAF+1)` — a transformation (**already in the raw data**) that reduces the skew of the MAF distribution | `3.14` |

† They do **not** end up in the enriched file: the two **bank 2** trims are discarded when NB1 saves (only
bank 1 is kept). The 5 columns added in NB1 (`elevation_m`, `slope`, `accel_kmh_s`, `dist_m`,
`dt_ms`) take it from **18 raw → 21** in the enriched file (18 − 2 bank 2 + 5 derived).

### 2.2 The 5 **derived** columns added in NB1

Examples from a real enriched row (decelerating vehicle). All computed **point vs previous
point, within the same trip** (`groupby(['VehId','Trip']).shift(1)`).

| Column | Type | How it is computed | What it is for | Example |
|---|---|---|---|---|
| `elevation_m` | float | Elevation from the **3-dec dedup → Open-Meteo (SRTM)** then merge (not a per-row computation) | Base for `slope`, `dz_net`/`climb`/`descent` in NB2 | `253.0` |
| `dist_m` | float | Horizontal distance from the previous point (**Haversine**) | **Segmentation** ~250 m in NB2; denominator of `slope` | `46.19` |
| `dt_ms` | float | Δt from the previous point (`Timestampms − Timestampms_prev`) | `air_g = MAF·dt`; denominator of `accel` | `200.0` |
| `slope` | float | `dz / dist_m` (Δelevation/Δspace), guard `dist>1`, clip ±0.3 | **Terrain** feature (turned out weak) in NB2/NB3 | `-0.0433` |
| `accel_kmh_s` | float | `dv / (dt_ms/1000)` (Δspeed/Δtime), guard `dt>50ms`, clip −15..+10 | **Kinematics** (strong signal): `accel_abs_mean`, driving styles | `-10.0` |

> `dist_m`/`dt_ms` are **NaN on the first row of each trip** (26,285 rows, see §3); `slope`/`accel`
> there are set to 0. (`elevation_m` is not a "per-row" derivative but external data attached; I put it
> here because it is nonetheless added by NB1 and is not in the raw data.)

### 2.3 Features **built in the notebooks** (not in the parquet)

The columns above are per-*row* data. The models, however, work on **aggregated units** — a **road
segment** (NB2), a **spatial cell** (NB3 Part A), a **driver** (NB3 Part B) — and derive the
features by aggregating the enriched rows. Examples from real units computed on the data.

**NB2 — segment level (~250 m, ICE only).** Target + 20 *map-only* features. Example: a
stop-and-go segment (VehId 116, mean speed 13 km/h).

| Feature | From where | Meaning | Example |
|---|---|---|---|
| `maf_per_km` **(target)** | `Σ(MAF·dt) / km` | Air/km = **segment consumption** | `4652` |
| `seg_distance_m` | `Σ dist_m` | Segment length | `241.3` |
| `dz_net` | `elev_end − elev_start` | Net elevation change (terrain) | `2.0` |
| `climb_m` / `descent_m` | `Σ` climbs / descents | Cumulative climb/descent | `10.0` / `8.0` |
| `slope_mean` / `slope_max_abs` | mean / max\|·\| of `slope` | Mean slope / peak | `0.003` / `0.300` |
| `speed_mean/max/min/std` | stats of `Vehicle_Speed` | Speed profile | `13.3 / 26 / 0 / 8.5` |
| `accel_abs_mean` | mean \|`accel_kmh_s`\| | Accel/decel intensity | `2.41` |
| `stop_fraction` | fraction of rows `speed<2` | Stationary share (stop-and-go) — **top feature** | `0.196` |
| `entry_speed` | first `Vehicle_Speed` of the segment | Inherited kinetic energy | `20.0` |
| `next_dz_net` / `next_slope_mean` / `next_stop_fraction` | `shift(-1)` of the next segment | **Anticipation** (look-ahead, from the map) | `-28 / -0.006 / 0.333` |
| `Generalized_Weight` | `first` | Vehicle weight (context) | `2526` |
| `OAT_DegC` | mean | Outside temperature | `3.19` |
| `hour` / `month` | from `Datetime` | Hour/month (traffic/season context) | `13` / `11` |

**NB3 Part A — spatial-cell level (~11×8 m).** 7 *kinematics + geometry* features that **form** the
clusters, + `maf_mean` **descriptive only**. Example: a free-flowing cell (120 passages).

| Feature | From where | Role | Example |
|---|---|---|---|
| `speed_mean` / `speed_std` | mean/std `Vehicle_Speed` in the cell | clustering | `46.4` / `5.98` |
| `accel_mean` / `accel_std` | mean/std `accel_kmh_s` | clustering | `0.22` / `2.71` |
| `accel_abs_mean` | mean \|`accel`\| | clustering | `1.06` |
| `slope_mean` | mean `slope` | clustering (geometry) | `-0.030` |
| `stop_fraction` | fraction `speed<2` | clustering | `0.000` |
| `maf_mean` *(→ `maf_mean_descr`)* | mean `MAF` | **descriptive only** (does not cluster) | `9.16` |

**NB3 Part B — driver level (`VehId`).** 8 *kinematic* features (powertrain-agnostic). Example:
an HEV driver.

| Feature | From where | Meaning | Example |
|---|---|---|---|
| `speed_mean` / `speed_std` | mean/std `Vehicle_Speed` of the driver | Pace and variability | `38.5` / `21.0` |
| `speed_p85` | 85th percentile of speed | **Cruising** speed | `61.0` |
| `accel_abs_mean` / `accel_std` | mean\|·\| / std `accel` | Driving liveliness | `1.92` / `4.05` |
| `frac_hard_accel` / `frac_hard_decel` | fraction `accel>6` / `accel<−6` | **Hard accelerations/braking** (aggressiveness) | `0.063` / `0.057` |
| `stop_fraction` | fraction `speed<2` | Share of stationary time | `0.069` |

## 3. Missing values

Only two columns have NaN: **`dist_m` and `dt_ms`, 26,285 each** — exactly the number of trips.
They are the **first row of each trip** (there is no previous point from which to compute distance/Δt). Everything
else is complete. *(Practical implication: in the notebooks these NaNs must be zeroed before
`cumsum`/integrations — a trap already handled in NB2.)*

## 4. Powertrain (EngineType)

| Type | rows | % | vehicles | trips |
|---|---|---|---|---|
| ICE | 10,401,317 | 58.0% | **197** | 14,705 |
| HEV | 5,378,063 | 30.0% | **90** | 9,327 |
| PHEV | 2,143,489 | 12.0% | **12** | 2,253 |

⚠️ **Important caveat:** PHEVs are only **12 vehicles** (and 2,253 trips). Any "per PHEV"
conclusion is statistically fragile (few units) and should be taken as indicative. ICE and HEV are well
represented.

## 5. Temporal coverage

- **By month:** fairly uniform, with peaks in **November (2.19M) and December (1.96M)** → more winter
  data (useful for the temperature effect on consumption).
- **By hour** (of trip start): peaks at **noon (~1.5M) and 20–21 (~2M)**, minimum 5–8 in the
  morning → commuting pattern.
- **By day:** **Thursday** is the most represented; overall a reasonable distribution across
  the whole week.

## 6. Geography

`lat ∈ [42.220, 42.326]`, `lon ∈ [−83.804, −83.674]` → a box of ~**12×10 km** over Ann Arbor.
A compact urban/suburban area: no long highways, no mountains.

## 7. Statistics of the key variables

| Variable | median | mean | p95 | max | notes |
|---|---|---|---|---|---|
| Vehicle_Speed (km/h) | 44 | 41.4 | 83 | 173 | 5th pct = 0 (stops) |
| accel (km/h/s) | 0 | −0.1 | 7.5 | 10 | clip [−15, +10] |
| MAF (g/s) | 5.58 | 9.85 | 30.4 | 259 | target, skewed |
| Engine_RPM | 1255 | 1198 | 2373 | 6605 | |
| Absolute_Load (%) | 26.6 | 29.7 | 62.0 | 196 | clip at 200 in NB1 |
| OAT (°C) | 9 | 11.0 | 31 | 60 | range −40…60 (suspicious extremes) |
| slope (fraction) | 0 | 0 | 0 | 0.3 | **almost always 0** (§11) |
| elevation (m) | 268 | 268.8 | 298 | 324 | range 101 m |
| Generalized_Weight | 3500 | 3375 | 4500 | 6000 | |
| dist_m (per row) | 0 | 7.76 | 52.8 | 11,210 | 0 when stationary |
| Short_Term_Fuel_Trim (%) | 0 | 0.13 | — | 89.8 | diagnostics |
| Long_Term_Fuel_Trim (%) | 0.78 | 1.26 | — | 57.8 | diagnostics |

## 8. The target: MAF

- Strongly **right-skewed**: median 5.58 but mean 9.85, **skew 1.91**, tail up to 259 g/s.
- This is why **`log_MAF`** already exists in the raw data (range 0–5.56), useful if you want a more
  symmetric target.
- **9.5%** of the rows have MAF < 0.5 g/s (engine idling / electric traction in hybrids);
  MAF exactly 0 only 0.36%.

## 9. Irregular sampling

The Δt between samples is **not constant**: median **600 ms**, p95 1900 ms, p99 2400 ms, but **max
~7,444,700 ms (~2 hours!)**. **0.48%** of the Δt exceed 3 s (logging gaps).
**Implications:** acceleration must be computed on the real Δt (done in NB1); to integrate
consumption over the segment (NB2) the Δt must be **capped** (at 2 s) so as not to add "phantom air" on the gaps.

## 10. Trip and vehicle structure

- **Trips:** 26,285. Rows/trip median **536** (p05 133, p95 1716, max 10,813); distance/trip median
  **4.3 km** (p95 14 km); duration median **6.9 min** (p95 23 min). → short urban trips.
- **Vehicles:** 299. Rows/vehicle **very imbalanced**: median 38,673, **min 394, max 822,320**.
  A single vehicle alone has ~822k rows.

⚠️ The **per-vehicle imbalance** is why the tuning uses **GroupKFold per VehId**:
without it, a dominant vehicle would end up in both train and validation, inflating the performance.

## 11. The terrain — the main critical point

- **Elevation range of the whole city: only 101 m** (223 → 324). Gentle terrain.
- **92% of the rows have |slope| < 1%**; the percentiles from the 5th to the 95th are **exactly 0**.
- **Slope ↔ MAF correlation = 0.004** (null).

Double cause: (a) Ann Arbor is almost flat; (b) **quantization artifact** — in NB1 the
coordinates are deduped to 3 decimals (~111 m) for the elevation API, so all points inside
a cell receive the **same elevation** → `dz=0` → slope null inside the cell and "stepwise" only at
the borders (then clipped to ±0.3). **It is the reason for the project reframe** (terrain is not the
useful signal; the speed profile is).

## 12. Correlations with MAF (Pearson)

| feature | corr with MAF |
|---|---|
| Engine_RPM_RPM | **0.751** |
| Absolute_Load_pct | **0.515** |
| Vehicle_Speed_km_per_h | 0.326 |
| accel_kmh_s | 0.299 |
| Generalized_Weight | 0.143 |
| elevation_m | −0.046 |
| OAT_DegC | −0.015 |
| **slope** | **0.004** |

Crucial reading: **RPM (0.75) and Load (0.52) are quasi-deterministic with MAF** — it is the
quantitative proof of why they must be **excluded** from the map-only model (it would be like predicting MAF from
the formula that generates it). Speed and acceleration (0.33 / 0.30) are the "legitimate" signals; the
**slope is useless (0.004)**.

## 13. Motion and stops

**10.1%** of the rows are with a stationary vehicle (speed < 2 km/h): stops, traffic lights, traffic. Relevant
for NB2 (the per-segment `stop_fraction` turned out to be the dominant driver of consumption).

## 14. Comparison of the three powertrains

| | mean MAF | mean RPM | engine off while moving | MAF in decel | mean speed |
|---|---|---|---|---|---|
| **ICE** | 11.73 | 1398 | **0.0%** | 5.86 | 39.0 |
| **HEV** | 8.28 | 1079 | **21.0%** | 2.42 | 45.0 |
| **PHEV** | 4.65 | 526 | **24.4%** | 1.46 | 43.7 |

- HEV/PHEV run with the **engine off 21–24% of the time while moving** (electric traction) → MAF
  for them **does not measure the real consumption** (the electric energy, which VED does not record, is missing). It is the
  reason the consumption model (NB2) is **ICE-only**.
- The **MAF in deceleration** collapses for hybrids (5.86 → 2.42 → 1.46): the signature of **regenerative
  braking** (engine off on release). *(It was the material of the powertrain energy comparison
  in NB3, removed on 06/17: it remains a citable development.)*

## 15. Implications for modeling (summary)

1. **Map-only justified by the numbers:** RPM/Load correlate 0.75/0.52 with the target → excluded.
2. **Terrain = data limitation:** slope ~0 in 92% of cases, corr 0.004 → reframe on the speed
   profile; terrain documented as a limitation.
3. **GroupKFold per VehId:** per-vehicle data heavily imbalanced (max 822k rows).
4. **ICE-only for consumption:** MAF valid only for combustion engines; HEV/PHEV treated separately.
5. **Statistically fragile PHEV:** 12 vehicles → indicative conclusions.
6. **Capped Δt** in the integrations due to the sampling gaps (max ~2 h).
7. **Skewed target:** `log_MAF` available if symmetry is needed.
8. **First row of each trip:** `dist_m`/`dt_ms` NaN to be zeroed.
