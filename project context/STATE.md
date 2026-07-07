# STATE.md — Current state of the project

> Live state of the pipeline: **read first**, update at every relevant change.
> **Last update:** 2026-07-07 (later in the day). **The whole project was translated from Italian to
> English** at the user's request: notebooks (markdown, code comments, print strings, plot labels,
> cluster/style names), all docs, `build_elevation_cache.py`, `requirements.txt`. The Italian-named
> files were **renamed** to English: the consumption notebook is now `consumption prediction.ipynb`;
> in `project context/` the docs are now `PROJECT_REPORT.md`, `VED_DATA_ANALYSIS.md`, `discussions.md`
> and `ml_study_notes_1.md`. The documentation convention in `CLAUDE.md` is now **English**.
>
> Earlier on 07/07: discovered the **`.venv` is broken** (see known issues); created
> `ClaudeExplained.md` (practical Claude Code permissions).
> On 02/07: aligned to the real on-disk state after Alex's work of **06/18**: the 3 notebooks were
> **renamed** (`01_...` → `data prep.ipynb`, `02_...` → `consumption prediction.ipynb`, `03_...` →
> `clustering.ipynb`) and **modified**. NB2 was **pruned**: removed
> Pipeline/ColumnTransformer/StandardScaler (XGBoost works directly on the columns),
> the SEG_LEN sensitivity cell, the eco/sport counterfactual and the diagnostics/saving section
> (→ **it no longer produces** `consumption_model.joblib` nor the csv). The NB3 heatmap fixes
> (z-score from non-rounded means + `maf_mean_descr`) are **applied and executed**.
> On 02/07 the entries of `discussions.md` that cited things removed from NB2 (#23, #42, #43) were corrected.
> **MLOps exercise (07/07 update):** the instructor requires only **README + decision log** →
> the submission is now in **root**: `README.md` (follows the instructor's template, it **replaced** the old
> setup README and absorbs its contents) and `DECISION_LOG.md` (D1–D12). The
> exercise folder (HANDOVER + ONBOARDING, unique contents recovered into the README: VED→parquet
> conversion gap, glossary, conventions/DoD, process/roles) and the submission-templates folder (the instructor's
> templates, now absorbed) were **deleted**.
>
> ⚠️ **UNCOMMITTED work:** the rename of the 3 notebooks (git sees them as deletes + untracked),
> the modifications to the notebooks themselves, the English translation of everything, `discussions.md`
> (entries #40–#43 + 02/07 corrections + translation), this STATE.md, CLAUDE.md and README.md (updated
> to the new names). Moreover `consumption prediction.ipynb` is **only partly executed**
> (5/23 cells with output): it must be **re-run in full** before committing.

## Current framing
**"ML for energy and driving context"** on VED, applied to a driving assistant / ACC.
Two pillars: **consumption/eco-driving** · **context + driving styles**. The terrain
(initial idea "orographic ACC") turned out to be a **weak signal** → reframed around the **speed
profile**; terrain = a data limitation. Full picture: `PROJECT_REPORT.md`; data:
`VED_DATA_ANALYSIS.md`; Q&A revision: `discussions.md`.

## Pipeline — the 3 notebooks (renamed on 06/18, rename NOT committed)
```
[RUN OK, TO COMMIT]         data prep.ipynb                (formerly 01_data_prep_and_enrichment)  13/13 cells executed → ved_enriched.parquet (~17.9M rows)
[PRUNED, TO RE-RUN]         consumption prediction.ipynb   (formerly 02_consumption_ecodriving)    only 5/23 cells executed; XGBoost only, 20 features
[RUN OK, TO COMMIT]         clustering.ipynb               (formerly 03_unsupervised_context_and_styles)  18/19 cells executed, heatmap fixes applied, K=4
```
**State (02/07):**
- **NB1 (`data prep.ipynb`)**: re-run on 06/18 with small modifications (removed the EDA distribution cells
  §4, refactored the elevation cell with `ROUND_DECIMALS=3`, new final sanity cell).
- **NB3 (`clustering.ipynb`)**: heatmap fixes **applied and executed**; K=4 (intersection/mixed-urban/
  free-flowing/highway); saves only `cluster_profile.csv` + `cluster_map.html` (in `outputs/`, 06/17).
- **NB2 (`consumption prediction.ipynb`)**: pruned on 06/18 (see note at the top) and **to be re-run in
  full** (Optuna ~15–30 min). It no longer saves a model/csv.
- **Alex runs the notebooks** in the `.venv` kernel.

`outputs/` — files expected after the runs:
```
ved_enriched.parquet           (input to all, from the NB1 run)   elevation_cache.parquet
cluster_profile.csv            cluster_map.html                   [NB3]
```
> **NB2 no longer saves anything** (diagnostics/saving section removed on 06/18): the old
> `consumption_model.joblib` / `consumption_results.csv` / `consumption_seglen_sensitivity.csv`, if
> they reappear on disk, are orphans. Other historical orphans to delete if present:
> `road_segment_clusters.parquet`, `cluster_map_static.png`, `anomaly_scores.parquet`,
> `telemetry_autoencoder.keras` (leftovers of the removed NB4).

## Environment
- Interpreter: `./.venv/Scripts/python.exe`.
- ⚠️ **`.venv` BROKEN (detected 07/07):** it points to a Python managed by `uv` that no longer exists
  (`No Python at ...uv\python\cpython-3.12.12...`). `uv run` also fails for the same reason.
  To recreate before re-running the notebooks: `python -m venv .venv` +
  `pip install -r requirements.txt` (or `uv python install` then recreate). The working system
  Python is the Microsoft Store 3.13.
- GPU (GTX 1660 Super): **not worth it** — small models/data, most is sklearn (CPU). Stays CPU.
- (`torch`/`keras` were installed only for NB4, now removed: no longer needed.)

## Decision history (condensed)
1. **Reframe (2026-06-14):** from "ACC exploiting orography" to "energy + context". Cause: slope
   null in 92% of the rows, corr with MAF 0.004 (flat Ann Arbor + elevation quantization 111 m).
2. **Dataset:** we stay on **VED**. Evaluated and discarded an OBD-II "allcars" (no GPS) and an IMU
   classification one (4 classes, ~1100 rows, duplicate files): none solved the terrain.
3. **ICE-only consumption:** MAF is a valid proxy only for combustion engines (HEV/PHEV go electric
   ~21–24% of the motion → MAF=0; VED has no battery signals, verified on the raw schema).
4. **3-notebook structure:** NB2 = consumption per segment (ICE-only); NB3 = road context +
   driving styles (Part B). *(The NB4 autoencoder/diagnostics was removed on 06/17, Alex's
   choice: orthogonal to the eco-driving core of the project. Consequence: deep learning/anomaly detection
   no longer covered by the project — they remain among the studied topics, not applied.)*
5. **Trimmed NB2:** kept **XGBoost only (default + tuned)**; removed baseline/Ridge/Lasso/RF
   (Alex's choice; XGBoost holds on the tabular→boosting *prior*). **Consequence:** L1/L2
   regularization and Random Forest are no longer in the project.
6. **Hybrids in NB2 (2026-06-15):** documented *why* consumption is not extended to HEV/PHEV
   "accounting for the time in electric mode" — SoC is not in VED → the electric fraction is an
   unobserved latent variable; using it from the target = leakage; a *hurdle* model with SoC is needed
   (future development). Written in `PROJECT_REPORT.md` §5.1 and in NB2 markdown §1. A stale
   line in the NB2 intro (still listing Ridge/Lasso/RF) was also corrected.

## What to verify after execution
- **NB2:** the feature importance — expected that `stop_fraction` and speed dominate, terrain ~0.06
  (data limitation). *(The SEG_LEN sensitivity table and the eco/sport counterfactual were
  removed on 06/18: the results of the previous runs remain citable, see `discussions.md` #23.)*
- **NB3 Part B:** style profile on kinematics + heatmap + PCA (K_STYLE=3). *(The chi-square test
  style×powertrain, style→ICE consumption and the energy comparison were removed on 06/17 — developments.)*
- Manual naming of the clusters (NB3) after seeing the heatmaps (`name_cluster` assigns from values, it is only
  a proposal to confirm). **Note:** in NB3 `K_FINAL=4` is fixed in the "Final choice" cell.

## Known issues still open (minor)
- **NB1:** the first row of each trip has `slope`/`accel`=0 (not NaN) due to an `np.where` on NaN →
  ~26k rows with fictitious 0 values remain. Low impact. (In NB2 the `dist_m`/`dt_ms` NaNs of those
  rows are zeroed anyway before segmentation.)
- ~~NB3 "50 m" cells~~ **resolved (17/06):** the markdown now says ~11×8 m (4 decimals). With cells this
  small the ≥50 passages filter keeps only heavily trafficked roads.
- **PHEV = only 12 vehicles** (out of 299): the conclusions on PHEVs are indicative.
- **NB3 pruned (17/06):** removed t-SNE, cluster↔EngineType, chi-square style×powertrain, style→consumption,
  energy comparison; imports consolidated in `[2]`; restored the PCA cell (it was orphaned after the
  deletions); markdown/intro/summaries realigned. **Two redundant summary cells remain**
  (`[38]` and `[39]`): Alex can merge them.
- **30 m elevation grid: explored and CANCELLED (17/06).** Idea: sample elevation at 30 m (native SRTM
  resolution) instead of the current 111 m. Everything restored to `ROUND_DECIMALS=3` (NB1 + `build_elevation_cache.py`).
  Reason: ~2h for the API rate-limit + marginal gain on a flat city. **Do not redo before the exam**;
  it remains a future development. Details and numbers in `discussions.md` #36.
- ~~NB3 heatmap fix~~ **resolved:** applied and executed in `clustering.ipynb` (z-score from the non-rounded
  means + `maf_mean_descr` row, see #39). Only remains to **commit**.
- **NB2 (`consumption prediction.ipynb`) to be re-run in full** (only 5/23 cells have output), then
  commit everything (notebook rename + context).

## Future developments (citable at the exam)
Counterfactual/optimizer **at equal time** (shows that "drive slowly" is not the answer) ·
DP planner "slow down before the climb" · SHAP on the consumption model · visual eco-routing ·
the removed NB3 blocks (chi-square style×powertrain, energy comparison, t-SNE).
