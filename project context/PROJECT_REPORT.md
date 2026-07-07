# Project report — ML for energy and driving context (VED)

*(Application: driving assistant / adaptive cruise control. Framing revised 2026-06-14:
see §1.)*

> Single document for the **exam discussion**. It explains: what the project is, the dataset (with
> real explored numbers), the architecture, each notebook, the **critical points** and the **choices** —
> so it can be told to the professor. For the Q&A revision see `discussions.md`.
>
> All the numbers in this report come from a real exploration of `ved_enriched.parquet`
> (17.9M rows), not from estimates.

---

## 1. The idea in one sentence

**The project:** use **supervised + unsupervised** ML on real telemetry to give a
driving assistant / ACC two capabilities — **estimate consumption** (eco-driving) and **recognize the
road context** and the driving styles.

**Honesty up front (so it doesn't look "forced").** The starting idea was an ACC that exploits
**orography**: anticipate the climbs → modulate speed → consume less. Exploring the data,
this hypothesis was **disproved**: in Ann Arbor slope is almost absent and quantized (§3,
corr with MAF 0.004). The project was therefore **reframed** — the real predictive signal is
the anticipated **speed/traffic profile**, not the climbs — and terrain remains as a *documented
data limitation*. This path (hypothesis → disproved by the data → correction) **is part of the
method**, not a flaw: it is the difference between "choosing the data for the story" and "letting the
data tell the story".

The conceptual chain of a predictive ACC (who does what):

1. **Map (NB1)** → from elevation it derives the **slope** of the upcoming segment.
2. **Cost evaluator (NB2)** → given a driving scenario, it estimates **consumption**. It is a
   *meter*, not the driver.
3. **Context + driving-style recognizer (NB3)** → road type and driving profile.
4. **Planner** (out of scope, future) → tries several speed profiles, asks the evaluator for the
   cost, chooses the cheapest. *This* decides "how much to accelerate".

> Sentence to use at the exam: *"from the map I know slope and segment type (NB1+NB3); a
> model estimates the consumption of each driving strategy (NB2); a controller chooses the most
> efficient one."*

---

## 2. The dataset — Vehicle Energy Dataset (VED)

Real OBD-II telemetry from the University of Michigan, **Ann Arbor (MI)** area, collected in
54 weekly parquet files. No elevation in the raw data → we derive it (NB1).

### 2.1 Real numbers (after cleaning, from `ved_enriched.parquet`)

| Quantity | Value |
|---|---|
| Rows | **17,922,869** |
| Unique vehicles | **299** |
| Unique trips | **26,285** |
| EngineType | ICE 10.40M · HEV 5.38M · PHEV 2.14M |
| Time span | 2017-11-01 → 2018-11-10 (**374 days**) |
| Extent | lat 42.220–42.326 · lon −83.804…−83.674 (**~12×10 km**) |
| Total fleet distance | **~138,827 km** |
| Median trip | 536 rows, **~4.3 km** |

### 2.2 Key distributions

- **MAF** (`MAF_g_per_sec`, the consumption target): mean 9.85, **median 5.58**, max 259 g/s →
  very **right-skewed** (this is why `log_MAF` already exists in the raw data). The 99th pct is 44.9.
- **Speed**: mean 41.4 km/h, median 44; the 5th pct is 0 (many stationary-vehicle rows).
- **Acceleration**: mean ~0, std 3.9 km/h/s (clipped to [−15, +10]).
- **Weight** (`Generalized_Weight`): 2500–6000, median 3500.
- **Temperature** (`OAT_DegC`): median 9 °C, range −40…+60 (the extremes are suspicious: sensor
  outliers).

### 2.3 Irregular sampling — a data constraint

The `dt` between consecutive samples is not constant: **median 600 ms**, p99 2400 ms, but **max
~7,444,700 ms (~2 hours!)**. There are huge logging gaps. Methodological consequences:
- acceleration must be computed on the real `dt` (not a plain `diff()`) — done in NB1;
- to integrate consumption over the segment (NB2), the dt is **capped at 2 s**, otherwise the gaps
  would add "phantom air".

---

## 3. THE central critical point: slope is almost unusable on this data

It is the most important point to be able to defend, because it is the **conceptual heart** of the project
("the ACC exploits orography") and the data calls it into question.

### 3.1 The numbers

- Elevation: range **only 101 m** (223→324) across the whole city → **gentle terrain**.
- Slope: **92% of the rows have |slope| < 1%**; the percentiles from the 5th to the 95th are **exactly
  0.0000**.
- **Slope ↔ MAF correlation = 0.004** (practically null).

### 3.2 Why slope is almost always zero — two overlapping causes

1. **Ann Arbor is flat** (range 101 m): little real elevation change = little signal.
2. **Quantization artifact** (subtler, to explain well): to obtain elevation
   without making 18M API calls, in NB1 the coordinates are **deduped to 3 decimals
   (~111 m)**. All points inside the same 111 m cell receive **the same elevation** →
   `dz = 0` → `slope = 0`. Slope becomes **null almost everywhere and "stepwise"** only at
   cell boundaries (where it is also then clipped to ±0.3). This is why the percentiles are 0 and the
   99th pct jumps to 0.3 (the clip).

> **How to tell it to the professor (turning the limit into method):** *"The instantaneous slope is
> dominated by a resolution artifact: SRTM elevation (~30 m) and the 111 m dedup make it
> null inside the cell and spiky at the edge. It is not a code bug, it is the limit of the
> source. This is why in NB2 I switched from the instantaneous value to the **cumulative elevation
> change over the segment**, much more robust, and I documented the case as a data limitation — on hilly
> terrain slope would matter."*

This single critical point justifies half of the project's choices (map-only, switch to the
segment, counterfactual).

---

## 4. The 3 notebooks — what they do and the key choices


### NB1 — `data prep.ipynb` (preparation)
Consolidates the 54 parquet files, EDA, **outlier filter** (Load 0–200%, RPM 0–8000, Speed 0–200, MAF
0–300), motion filter, **elevation enrichment** via Open-Meteo (free, no key, SRTM)
with **3-decimal dedup** (~8.7k points, ~88 batches, resumable cache), **slope** via Haversine,
**acceleration** on real `dt`. → `ved_enriched.parquet`.
**Choices:** 3-dec dedup (not 4: unusable *and* finer than the source); local cache; slope/accel
clipping. **Known critical points:** the first row of each trip with slope/accel = 0; windows "in samples"
not in seconds (irregular sampling).

### NB2 — `consumption prediction.ipynb` (consumption / eco-driving, ICE only)
Predicts **`maf_per_km` on ~250 m segments** (efficiency, not trivial distance), with **map-only**
features + **anticipation** (`entry_speed` + look-ahead of the next segment). **XGBoost
default and tuned** (Optuna + GroupKFold per VehId). *(Notebook trimmed in two passes: kept only
the two XGBoosts — baseline/Ridge/Lasso/RF removed; on 06/18 also removed Pipeline/scaler, the
**SEG_LEN sensitivity (150/250/500)** and the eco/sport counterfactual — their results, verified
in the previous runs, remain cited below.)*
**Two key choices:**
1. **Segment, not instant.** Predicting the *instantaneous* MAF would be almost a physical tautology
   (terrain = noise, artificial counterfactual); the segment target `maf_per_km` solves it.
2. **ICE only.** MAF is a consumption proxy valid only for combustion engines; RPM/Load also excluded
   (map-only) and `EngineType` (now constant). *(Why not include hybrids "accounting for the
   time in electric mode"? Three technical reasons in §5.1.)*
**Real numbers (executed run):** R² ~**0.76**; **stop_fraction** (~0.25) and speed dominate; the
**terrain stays weak (~0.06)** at all segment lengths → confirms the data limitation.
Output: no file saved (saving section removed on 06/18; the results live in the notebook).

### NB3 — `clustering.ipynb` (road context + driving styles)
**Part A — road segments:** spatial cells (~11×8 m), ≥50 passages filter, **StandardScaler** (with
a demonstration of the failure without), **Elbow + Silhouette**, **K-Means++**, heatmap, **PCA**, interactive
Folium map. Clustering features = **kinematics + geometry only** (the consumption `maf_mean` is
only a *descriptive* column). Output: `cluster_profile.csv`, `cluster_map.html`.
**Part B — driving styles:** clustering of the **drivers** (`VehId`) on **kinematics** alone
(speed/accel/stops, no MAF → valid for all powertrains), with heatmap, naming and **PCA**.
*(Trimmed on 06/17: the **chi-square** test style×powertrain, the **style → consumption** ICE relation and the
**energy comparison** — engine-off-while-moving ICE 0% / HEV 21% / PHEV 24%, regenerative braking —
were **removed from the notebook** and remain **citable developments**. The "why hybrids are not in
the consumption model" is argued anyway in §5.1.)*

---

## 5. The big methodological decisions (and how to defend them)

| Decision | Why |
|---|---|
| **Map-only** (no RPM/Load) | those signals are not known in advance + quasi-circular with MAF. Lower R² but a model *usable* in an ACC. |
| **Temporal split per trip** | simulates deployment (train on the past, predict the future); no temporal leakage. |
| **GroupKFold per VehId** | the same vehicle not in train and validation together → CV not optimistic. |
| **Look-ahead = future road/speed, never future MAF** | it is the *point* of the project (map info), not leakage; the target never enters the features. |
| **Elevation dedup to 3 decimals** | consistent with SRTM ~30 m; 4 dec was unusable and finer than the source. |
| **Consumption per segment (NB2)** | the instantaneous target would be tautological; the segment uses the cumulative elevation change and makes the counterfactual meaningful. |
| **ICE-only consumption** | MAF is a valid proxy only for combustion engines; HEV/PHEV go electric (MAF=0, ~21–24% of the motion) and VED has no battery signals → mixing them would falsify the target. |
| **Powertrain treated separately (NB3)** | descriptive comparison ICE/HEV/PHEV + driving styles (kinematics, valid for all) → "the three vehicles are accounted for" without forcing MAF. |
| **Target `maf_per_km`** | measures efficiency, not trivial distance. |
| **NO naive cluster→consumption integration** | the cluster contains `maf_mean`/`rpm_mean` → it would be masked mean-target-encoding (leakage). Complementary pillars. |
| **Optuna instead of GridSearch** | Bayesian TPE: better solutions at equal budget (cost = trials × folds × fits). |
| **StandardScaler inside the Pipeline** | mandatory for K-Means/PCA (and for linear models); inside the Pipeline it avoids leakage (fit only on the train). |

### 5.1 Deep dive: why NOT to include hybrids in NB2 "accounting for the time in electric mode"

It is the most natural objection: *"HEV/PHEV go electric a certain % of the time (MAF=0); can't I
simply account for it and predict the same `maf_per_km` for them too?"*. The intuition is
correct as a **direction**, but on VED it does not hold, for three precise reasons. It is worth knowing how to
explain them because they show that the "ICE-only" choice is reasoned, not convenient.

**1. The *why* the engine is off is not observable.** For a hybrid `MAF=0` is not noise: it is the
ECU that decided to go electric, and that decision depends mainly on the
**battery state of charge (SoC)**, besides power demand, temperature and the manufacturer's
strategy. **VED records no battery signal** (SoC, current/voltage, electric-motor
power) — verified on the raw schema. So the exact same speed/acceleration
profile can have the engine on or off depending on a **latent variable we do not
observe**. Predicting the segment MAF for a hybrid means predicting a function of a hidden
factor → **high irreducible error**, and not through the model's fault.

**2. The "electric fraction" is not a feature known in advance.** If you compute it as the share of
rows with MAF=0 inside the segment, you are deriving it **from the target itself** → it is
**circular leakage** (I use MAF to build a feature with which I predict MAF). To use it
*honestly* it should be **predicted** first — which turns the problem into a **two-stage
(hurdle) model**: Stage 1 = P(engine on | map-only features); Stage 2 = E[MAF | engine on];
prediction = `P(on) × E[MAF | on]`. Conceptually correct, but Stage 1 is driven
precisely by the unobserved SoC (point 1) → a low and *structural* R² ceiling.

**3. The PHEV is the worst case.** A PHEV can travel the whole first part of the trip in
**pure electric** until the battery drains, then behave like a HEV. Without SoC **the two regimes
cannot be distinguished** from the driving features alone. And in VED the PHEVs are **12 vehicles out of 299**
→ not even enough data to learn the transition empirically.

**Practical consequence.** Simply adding `EngineType` as a feature and training on all
powertrains makes the model learn only an **"average discount"** per engine type, poorly calibrated at
the segment level, and **pollutes the eco-driving interpretation** (for ICE, MAF *is* consumption;
for hybrids, consumption splits between fuel and electric, which we do not measure). This is why the
consumption model stays **ICE-only**, and hybrids are treated **descriptively** in NB3 Part B
(engine-off fraction, regenerative braking, driving styles on kinematics — valid for all).
The *hurdle* model remains a good **future development** if a dataset with SoC is available (see §7).

---

## 6. Course topics covered

**Supervised:** Pipeline + ColumnTransformer, **XGBoost** (gradient boosting), GroupKFold, Optuna
(TPE), MAE/RMSE/MAPE/R², feature importance, data leakage & temporal split.
*(Note: NB2 was trimmed to XGBoost-only models → L1/L2 regularization and Random Forest
are no longer in the project. Reintegrable by keeping a Lasso if they need to be covered.)*
**Unsupervised:** K-Means/K-Means++, elbow, silhouette, motivated StandardScaler, PCA with loadings.
→ **Out of the project** remain **deep learning** (autoencoder/anomaly detection — the diagnostics
NB4 was removed because orthogonal to the eco-driving core), **CNNs** (no image
use case) and — after the NB3 pruning (17/06) — **t-SNE** and the **chi-square test**: topics
studied but no longer applied here.

---

## 7. Honest limitations and future developments

**Limitations (to declare before the professor asks):**
- **Flat terrain + quantization** → slope has a weak signal (a limit of the *data*, not of the
  method); NB2 confirms it (terrain weight ~0.06 at all segment lengths). On hilly
  terrain it would matter.
- **PHEV = only 12 vehicles** (out of 299): the conclusions on PHEVs are **indicative**.
- The cost models are **statistical, not physical simulators**: reliable only near the
  observed distribution.
- The **eco/sport counterfactual** (uniform ±10% scaling) is the weakest part: it gives a counterintuitive
  "eco +4%" because in the data low speed ↔ stop-and-go are intertwined. The strong proof is the
  **feature importance** (what drives consumption), not that demo.
- Data only from Ann Arbor → does not automatically generalize to other areas/vehicles.

**Future developments (consistent with the project):**
- **Counterfactual / mini-optimizer at equal time**: shows explicitly that the answer
  is not "drive slowly" (makes eco-driving non-trivial).
- **DP planner** "slow down before the climb": closes the ACC loop (NB2 already has the
  coupling features to support it).
- **Hurdle model for hybrids** (Stage 1: P(engine on); Stage 2: E[MAF | on]) on a
  dataset that includes the **battery SoC** — the only way to honestly extend consumption to
  HEV/PHEV (impossible today on VED, see §5.1).
- **SHAP** on the consumption model (how, not just how much, the features matter).
- **Eco-routing** visualization (links consumption + road context).
- *(Driver-level clustering — eco/aggressive profiles — has already been implemented in NB3,
  Part B.)*

---

## 8. Likely exam questions — ready answers

- *"Is your R² the right one?"* → The consumption model (segment, ICE) does **R² ~0.76**. With
  the engine (RPM/Load) it would rise, but it would cheat (quasi-circular with MAF); map-only is the honest
  and ACC-usable choice.
- *"Where is the value of slope, if it is irrelevant?"* → On *this* data it is weak: flat terrain
  + quantization at 111 m (slope null in 92% of the rows, corr 0.004); confirmed in NB2 (terrain
  weight ~0.06 at all scales). Documented as a **data limitation**.
- *"Aren't the look-ahead features leakage?"* → No: they are *future* slope/speed (info from
  map/planned profile), the **future MAF target never enters**.
- *"How did you handle the three powertrains? Isn't MAF falsified for hybrids?"* → Yes, and that is the point:
  HEV/PHEV go electric (MAF=0) 21–24% of the motion and VED has no battery → MAF is not
  consumption for them. So a **consumption model ICE-only**, and hybrids **described separately** (NB3:
  engine-off, regenerative braking, driving styles). PHEV = 12 vehicles → indicative.
- *"Aren't you just showing that economical driving is worth it?"* → No. The model is a **cost
  calculator**, not advice: without a time constraint "minimum consumption" = "stop". The real lever is
  **stop-and-go** (feature #1), not raw speed — slowing down may in fact increase consumption/km.
- *"Why didn't you merge cluster (NB3) and consumption (NB2)?"* → The cluster encodes `maf_mean`/`rpm_mean`
  → leakage (masked target encoding). Complementary pillars.

---

## 9. File map

| File | What |
|---|---|
| `data prep.ipynb` | NB1 — preparation + enrichment (input to all) |
| `consumption prediction.ipynb` | NB2 — consumption per segment, ICE only |
| `clustering.ipynb` | NB3 — road segments + driving styles |
| `outputs/ved_enriched.parquet` | consolidated dataset + elevation + slope (input to all) |
| `PROJECT_REPORT.md` | **this file** — overview for the exam |
| `VED_DATA_ANALYSIS.md` | complete EDA of the dataset (real numbers) |
| `discussions.md` | Q&A log for exam revision |
| `STATE.md` | live state of the pipeline + next steps |
