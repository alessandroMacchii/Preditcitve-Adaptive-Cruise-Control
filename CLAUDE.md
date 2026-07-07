# CLAUDE.md — Onboarding for new sessions

> File loaded automatically by Claude Code at the start of a session. Kept **deliberately
> concise**: it contains stable facts + pointers. The **current state** (always
> changing) lives in `STATE.md`: **read it first**, and update it when you do something.

## What the project is
Supervised + unsupervised ML on the **Vehicle Energy Dataset (VED)** to demonstrate two capabilities
useful to a **driving assistant / adaptive cruise control**: (1) estimate **consumption** as a
function of the *anticipated* speed profile (eco-driving), (2) **recognize the road
context** (segment types) and the **driving styles**. Alex's exam project.

> **Framing (revised 2026-06-14).** The initial idea was an ACC exploiting **orography**.
> Exploring the data, slope turned out to be **weak** (flat Ann Arbor + elevation
> quantization at ~111 m: slope null in 92% of the rows, corr with MAF 0.004). The project was
> therefore **reframed**: the real predictive signal is the anticipated **speed/traffic
> profile**, not the climbs; terrain remains documented as a *data limitation*. The
> alternative datasets evaluated (OBD-II without GPS, IMU) did not solve the problem → we stay on VED.
> History and numbers in `PROJECT_REPORT.md`.

## The 3 notebooks (renamed by Alex on 2026-06-18; translated to English on 2026-07-07)
1. `data prep.ipynb` *(formerly `01_data_prep_and_enrichment.ipynb`)* — loads the VED parquet files, EDA,
   enriches with elevation (Open-Meteo) and computes slope + acceleration.
   Output: `outputs/ved_enriched.parquet`.
2. `consumption prediction.ipynb` *(formerly `02_consumption_ecodriving.ipynb`)* —
   **consumption / eco-driving**: predicts `maf_per_km` on ~250 m segments, **ICE only** (MAF is a valid
   proxy only for combustion engines), 20 map-only features + anticipation. RPM/Load and EngineType
   excluded. *(Pruned on 06/18: removed Pipeline/ColumnTransformer/StandardScaler, SEG_LEN sensitivity,
   eco/sport counterfactual, diagnostics/saving — it no longer saves files in `outputs/`.)*
3. `clustering.ipynb` *(formerly `03_unsupervised_context_and_styles.ipynb`)* — **context + driving
   styles**: Part A clustering of road segments (K-Means/PCA/Folium map, kinematics+geometry
   only); Part B clustering of **drivers** by style (kinematics, powertrain-agnostic) + PCA.
   *(Pruned on 06/17: removed t-SNE, cluster↔EngineType sanity check, chi-square test style×powertrain,
   style→consumption, energy comparison — they remain as future developments.)*

> Execution state and what remains to commit: see `STATE.md` (Alex runs the notebooks).

## The decision that defines the project: the "MAP-ONLY" model
NB2 uses **only features known in advance to an ACC** (speed, acceleration, current/future
slope, future speed, context). The engine signals
`Engine_RPM_RPM`, `Absolute_Load_pct`, `rpm_roll10s_mean` are **deliberately excluded** because: (a) not known in advance
in the real use case, (b) quasi-circular with the target (MAF is ~ RPM × load). Including them
gives a higher R² but makes the model learn a shortcut and zeroes out the role of the road.

## How to run it
- Interpreter: `./.venv/Scripts/python.exe` (the project environment).
- The notebooks: run them in the `.venv` kernel. NB2 with Optuna is slow (~15–30 min).
- ⚠️ Possible **scikit-learn version mismatch** between the environment that saved the
  `.joblib` files and the `.venv`: re-run/save in the same environment (see STATE.md).

## Facts about the data (verified)
- `content/`: 54 parquet files, ~18.26M rows, schema verified. Area: **Ann Arbor (MI)**.
- `Datetime` is constant per trip; intra-trip time is `Timestampms`. `log_MAF` exists.
- **Gentle** terrain: slope has a limited signal (+ SRTM resolution ~30 m). It is a
  limitation of the *data*, not of the method — a good cue for "future developments".

## Known traps (do not reintroduce them)
- **Anticipation (look-ahead) NB2**: the `next_*` features are built with
  `groupby(['VehId','Trip']).shift(-1)` (next segment), never looking back. The
  future of the road/speed must enter, NEVER the future MAF (the target).
- **Elevation cache**: `build_elevation_cache.py` is resumable (saves after each batch).
  The NB1 elevation cell (§6) downloads only the missing points and raises if the cache stays incomplete.
  Open-Meteo is **free, no API key**; the only constraint is the 429 rate-limit (already handled).
- **NB2 counterfactual** *(removed on 06/18; applies if reintroduced)*: scale coherently all the
  `SPEED_COLS` (`speed_mean/max/min/std`, `entry_speed`, `accel_abs_mean`); the road/`next_*`
  features stay fixed.

## Context files (in `project context/`, read them as needed)
- `STATE.md` = live state of the pipeline + next steps (**read first**).
- `PROJECT_REPORT.md` = exam overview (dataset, critical points, choices) with real numbers.
- `VED_DATA_ANALYSIS.md` = complete EDA of the dataset (real numbers: schema, distributions, correlations, powertrain).
- `discussions.md` = Q&A log for exam revision (Alex's questions + concise answers).
- `ml_study_notes_1.md` = generic theoretical ML revision (PCA/t-SNE/UMAP, etc.), not project-specific.
- `README.md` (in root) = the MLOps exercise submission (follows the course template):
  problem, setup/execution, data, ML lifecycle, MLOps, risks, conventions and glossary.
  It replaced the old setup-only README.
- `DECISION_LOG.md` (in root) = decisions D1–D12 with alternatives and reasons (exercise submission).
- *(Removed on 07/07: the exercise folder with HANDOVER/ONBOARDING — unique contents
  recovered into the README — and the submission-templates folder with the instructor's templates, now absorbed.)*

## Conventions
- Documentation in **English** (the project was translated from Italian to English on 2026-07-07,
  by the user's request: no Italian must remain in code, notebooks, plots or docs).
- Before launching API calls or long/heavy (Optuna, big fits) or
  destructive actions, **ask Alex**.
