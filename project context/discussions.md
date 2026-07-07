# Discussions — Q&A log for exam revision

> Summary of Alex's questions and the key answers, in abbreviated form, to remember them when
> explaining the project. Updated at every new discussion. For the full detail:
> `PROJECT_REPORT.md`, `VED_DATA_ANALYSIS.md`.
>
> **Started:** 2026-06-17.

---

## 1. Notebook logic and how to explain it at the exam
**Q:** What is the notebooks' logic, why do they exist, how do I explain them to the professor?
**A:** Not disconnected files but the **modules of a driving assistant / predictive ACC**:
- **NB1** prep+enrichment (data, elevation, slope, accel) → `ved_enriched.parquet`.
- **NB2** cost evaluator: estimates the **consumption** (`maf_per_km`) of a driving strategy (supervised).
- **NB3** recognizer of **context + driving styles** (unsupervised).
- **Key sentence:** *"from the map I know slope and segment type (NB1+NB3); a model estimates the
  consumption of each strategy (NB2); a controller chooses the most efficient one."*
- **Honest twist:** the initial idea (orographic ACC) was **disproved by the data** (flat
  terrain) → reframed around the speed profile. *"Hypothesis → disproved → correction IS the method."*

## 2. Was the autoencoder (NB4) needed? → removed
**Q:** Is NB4 (autoencoder/diagnostics) not necessary for the future ACC, was it only for the exam?
**A:** Correct: NB4 is **orthogonal** to the eco-driving core (it is predictive maintenance, not on the
consumption↔speed loop) and the docs say it existed to cover the exam topic. **Removed on
06/17.** Honest consequence: the project **no longer covers deep learning/anomaly detection**.
- **At the exam:** *"the predictive pillars are consumption and context; diagnostics is a
  parallel module, I removed it because orthogonal to the focus."*

## 3. Common thread: "this is only the introduction"
**Q:** I want to close with *"this is the introduction, the adaptive CC will be built with more data"*.
**A:** Present the notebooks as **foundations + gap discovery**, not as a finished project:
1. I built and validated the **building blocks** (consumption, context) on real data;
2. I **mapped what is missing** for the complete system (real orography + battery SoC + the
   **planner**, i.e. the controller that chooses the speed).
- The closing sentence becomes a **roadmap**, not an alibi.

## 4. Can NB2 and NB3 be merged into one notebook?
**Q:** Now that NB4 is gone, do I merge NB2 and NB3?
**A:** Technically yes, but **better separate**:
- they map the two blocks of the course (**supervised** vs **unsupervised**);
- the physical separation **is** the proof of the anti-leakage point (see §5);
- independent execution/RAM (NB2 loads ICE+Optuna, NB3 aggregates 17.9M rows).
- Merging is worth it **only** if the submission requires a single file → two clean sections.

## 5. Why didn't I use the clusters (NB3) as features in the consumption (NB2)?
**Q:** Why don't I feed the cluster to the consumption model?
**A:** Because the cluster is **built from `maf_mean`/`rpm_mean`** → using it = injecting the mean of the
target → **leakage (masked target encoding)**. Inflated but circular R².
- **Subtlety:** the *wrong* justification is "an ACC does not know the road type" (false: it is known
  from the map). The *right* one is "the cluster is defined using the target".
- Moreover you would gain nothing: NB2 already has the raw kinematic features; of the cluster only the
  leaky part would remain.
- **Honest fix (future):** re-cluster **without** MAF/RPM/Load.

## 6. NB3: why 50×50 m cells?
**Q:** Why 50×50 m cells?
**A:** **They are not 50 m — it was a typo in the markdown, ✅ corrected on 06/17 (now says ~11×8 m).**
Rounding to 4 decimals gives cells of **~11 m (N-S) × ~8 m (E-W)**. (1° lat ≈ 111 km → 0.0001° ≈ 11 m;
in longitude × cos(42°) ≈ 8 m.)
- **Cell-size trade-off:** big → mixes different stretches; small → few measurements/cell.
- **Consequence (selection bias):** small cells + ≥50 passages filter → only the
  **trafficked roads** remain (281,494 → 77,325 cells). To be declared.

## 7. The ~11×8 cells: are they the road's length and width?
**Q:** Is the road 8 wide and 11 long?
**A:** No. 11 and 8 are the two sides of a **fixed map square** (vertical N-S / horizontal
E-W), different from each other because of the **convergence of the meridians** (cos of the latitude), **not**
because of the road. The road has its own direction and its own width (~7–10 m). The cell ≈ a short chunk of road.

## 8. Filter cell (2.1): show which cells are discarded + log scale
**Q:** The filter discards the sparse cells but doesn't show which → it looks like "few passages everywhere".
**A:** Two real flaws: (a) **linear** color scale crushed by the outliers (~4000); (b) the plot
shows only the survivors. **Fix applied to NB3:** two panels — left, discarded (gray) vs
kept (blue); right, density in **log scale**.

## 9. Cluster by road segment instead of by square? (map-matching)
**Q:** A square can contain two different roads — better to cluster by road segment?
**A:** You are right in principle: the correct unit is the **road segment** (map-matching on
**OpenStreetMap**: each GPS point → road arc; parallel roads = different arcs, intersections = nodes).
- But **high cost** (OSM dependency, snapping 17.9M points) and the **clusters would come out almost the same**
  (they are driven by the aggregated behaviors, not the exact geometry).
- **Recommendation:** keep it as **declared limitation #1 / future development** (it makes a good impression).
  A light possible mitigation: **heading-dispersion** feature to recognize the intersections.

## 10. How is slope computed?
**Q:** How is slope computed?
**A:** "Climb over distance": `slope = Δelevation / Δhorizontal distance`, between **two consecutive GPS
points of the same trip**. E.g. I climb 2 m over 40 m → 0.05 = 5%. It is **stretch-by-stretch**, not global.
**Code steps (NB1, "Slope computation" cell):**
1. **Sort** by `VehId, Trip, Timestampms` (consecutive = temporal order of the trip).
2. **Previous point** within the trip with `groupby(['VehId','Trip']).shift(1)` (lat/lon/elevation/time of the
   row before). The shift *within the trip* avoids linking end-of-trip↔start-of-next → the **first row
   of each trip has no "before"** → NaN (then dropped when saving).
3. **Numerator** `dz_m = elevation_m − elev_prev` (vertical elevation change).
4. **Denominator** `dist_m = haversine(lat_prev,lon_prev, lat,lon)` (horizontal distance on the Earth's
   sphere, **Haversine formula**, R=6,371 km).
5. **Division with two guards:**
   - `np.where(dist_m > 1.0, dz/dist, 0.0)` → if the car is almost stationary (Δdist < 1 m) it puts **0** instead of
     dividing by ~0 (no explosive values);
   - `.clip(-0.3, 0.3)` → caps at **±30%**; beyond that, in the city, it is an SRTM error not a real ramp.
- **Why 0 in ~92%:** it is the **numerator** → `dz_m = 0` every time the two consecutive points fall
  in the **same elevation block ~111×82 m** (same `lat_round`/`lon_round` → same `elevation_m`). It is not the
  denominator: it is that the **elevation does not change inside the block**. A correct method, weak only because of the
  quantized *elevation source*. See [[#11]] (elevation derived by us) and [[#33]] (why not more decimals).
- **Twin note:** **acceleration** is computed with the same logic but on time: `accel = Δspeed /
  Δt` (`dv/(dt_ms/1000)`, guard `dt > 50 ms`, clip −15..+10 km/h·s) to handle the **irregular sampling**
  (Δt varies ~100–1400 ms) — you cannot use a plain `diff()`.

## 11. Was elevation in the dataset or did we derive it?
**Q:** Was elevation in the raw data?
**A:** **We derived it in NB1.** The raw VED **has no elevation**; we have the GPS coordinates, we
round them to **3 decimals (~111 m)** → ~8,700 unique points → **Open-Meteo Elevation API** (SRTM,
free) → merge. This is why elevation is **constant inside each 111 m cell** → consecutive points
have the **same elevation** → `dz=0` → **slope=0** in 92% of the rows (not because it is flat there).

## 12. What if I made the cells smaller than 111 m?
**Q:** Does shrinking the cells recover slope?
**A:** **No** — the limit is not the dedup, it is the **source**. SRTM has a resolution of **~30 m** and a
vertical error of **several meters**:
- below 30 m **there is no new information** (you resample/interpolate the same pixel);
- at ~30 m you would have fewer zeros, but the new Δelevations would be **noise**, not real climbs (worse: it looks like
  signal);
- cost: 4 decimals → ~280k points (~32× API calls) + rate-limit;
- and Ann Arbor is **flat** (101 m total): there is no signal to recover.
- **Real solution:** high-resolution DEM (e.g. LiDAR USGS 3DEP 1 m) **or** a hilly city.

## 13. NB3 clustering: why MAF/RPM/Load together? What is Load? Why MAF as a feature?
**Q:** Why MAF as a clustering feature, why RPM too, and what is Load?
**A:** What they are (**bicycle** analogy): **RPM** = pedaling cadence (engine speed); **Load**
(`Absolute_Load_pct`) = how hard you push = **effort as % of max** (0% minimum, ~90% full throttle);
**MAF** = calories/s = **absolute consumption** (g air/s).
- **Not identical:** the same MAF from high RPM+low load *or* low RPM+high load → RPM/Load tell
  the *how*, MAF the *how much*.
- **Why MAF in Part A:** it gives the "energy signature" of the stretch (highway = high MAF, residential
  street = low). But it is **debatable**: MAF is a *consequence* of driving, not an independent property
  of the road; and `maf_mean` per cell is also contaminated by hybrids (MAF=0 in electric).
- **Redundant:** corr with MAF ≈ **0.75 (RPM)**, **0.52 (Load)** → keeping all three **overweights**
  the "engine effort" dimension (3 ~collinear features out of 10) and makes the cluster **leaky** (see §5).
- **Useful contrast:** **Part B** (styles) deliberately excludes MAF (powertrain-agnostic). A
  clean version of Part A would do the same: only **kinematics + geometry**.

## 14. Decision: engine signals removed from the segment clustering (NB3 Part A)
**Q:** At this point do we exclude MAF here too, like in the driving styles?
**A:** Yes, but to do it "like Part B" you must remove **all three** engine signals
(`maf_mean`/`rpm_mean`/`load_mean`), not just MAF: RPM and Load correlate with MAF (0.75/0.52), so
by themselves they would leave the same dimension and the **leakage**. **Applied to NB3:** the segment
clustering features are now = **kinematics + geometry** (7: `speed_*`, `accel_*`, `slope_mean`,
`stop_fraction`).
- **Consequences:** (1) the clusters change → `road_segment_clusters.parquet` to regenerate; (2) the scaling
  demo is now dominated by `speed_mean` (~90%) instead of `rpm_mean` (99.52%) — same lesson; (3)
  no more leakage. The NB3 markdown was also updated.

## 15. But isn't "engine effort" an extra help to separate the clusters?
**Q:** Even if leaky, doesn't knowing how much the engine consumes/strains help separate the stretches better?
**A:** No, and for reasons stronger than "leaky":
- **"Leaky" only counts downstream:** for the clustering itself (unsupervised) there is no leakage; it holds
  only if you reuse the clusters in NB2. By itself it is a weak argument — a fair observation.
- **It is not new info:** consumption is a **consequence** of speed/accel/slope (already present) →
  **collinear** → it does not open a new axis, it **overweights** the existing one (more redundant features ≠
  more separation).
- **The truly new part is the wrong confounder:** it depends on **who drives there** (fleet, hybrids at
  MAF=0), i.e. it separates by *who travels* the road, not by *type* of road → like coords/elevation,
  out.
- **Interpretive bonus:** by excluding it, "the clusters differ in consumption" becomes an *a posteriori*
  **discovery** instead of a **tautology**.
- **Depends on the goal:** for "road type/context" → out; for "energy-demand map" → in. Rule: put among
  the features what you want the clusters to **mean**.

## 16. Maf/rpm/load in the aggregation: are they still needed?
**Q:** In the aggregation (cell `[4]`) we still compute maf/rpm/load — are they needed?
**A:** For the **clustering** no (excluded). Today they are computed and saved in the parquet but **never used**.
- **`maf_mean` → keep it and *use it*:** add it to `cluster_profile` as a **descriptive** column
  (not for clustering) to **show a posteriori** that the clusters differ in consumption = the
  "discovery not tautology" of #15.
- **`rpm_mean`/`load_mean` → removable** (engine internals, useless even descriptively); or leave them
  (cheap, harmless).
- **Key distinction:** *clustering features* (kinematics+geometry, **create** the clusters) vs
  *descriptive columns* (maf_mean, **interpret** the clusters afterwards). At the exam: *"consumption does not form
  the clusters, it describes them."*
- **State:** ✅ **done** — `maf_mean_descr` added to `cluster_profile` (Part A) as a descriptive
  column, not for clustering. `rpm_mean`/`load_mean` remain computed in `agg` but unused
  (removable whenever wanted).

## 17. Does the ≥50 passages filter discard too many cells? Lower it?
**Q:** We discard ~72% of the cells (281k→77k). Are 50 passages not a lot? Is it worth lowering?
**A:** To frame well:
- **It is not a quantity problem:** 77k cells are already plenty for K-Means; the discarded 72% are
  **tiny and noisy** cells (little-used residential, cell edges) → garbage, not info.
- **50 = rows (GPS samples), not distinct vehicles** → 50 samples to estimate mean/std is a
  **reasonable minimum**, not high. Below ~20-30 the `std` become unstable.
- **Trade-off of lowering:** + coverage / less bias toward the arterials (a residential cluster
  could emerge); − noisier statistics → silhouette (already ~0.23) could worsen.
- **To be decided with the data, not by intuition:** sensitivity test `MIN_PASSAGES ∈ {20,50,100}` →
  n. cells + silhouette + n. clusters. If the structure is stable → 50 defensible; if at 20 a
  sensible cluster appears without a silhouette collapse → lower.
- **State:** ✅ **check done** (script on the real data, thresholds 10→150). **Outcome: 50 is kept.**
  Silhouette: lowering **worsens** it (~0.235 at 10-20), raising **saturates** (~0.260 from 75 up but with
  half the cells). At 50 = ~0.257 with the **maximum coverage** (~77k cells) → the knee of the curve.
  Lowering it degrades the separation; raising it is a negligible gain + less coverage. Justification
  added to the NB3 markdown (cell 2.1) and to the GUIDE.

## 19. Explanation: standardization and choice of K
**Q:** I didn't get the standardization and the choice of K.
**A (standardization):** K-Means uses **distance** → features with different scales get crushed.
Test in the notebook: without scaling `speed_mean` = **91% of the variance**, `slope_mean`/`stop_fraction`
= 0% (ignored). The **StandardScaler** brings each feature to **mean 0, std 1** (z-score =
`(x−mean)/std` = "how many std from the mean") → all weigh equally. Mandatory for distance-based methods
(K-Means/PCA/KNN); useless for trees (XGBoost). Analogy: grouping people by height/weight/salary
— without scaling, salary dominates (large numbers).
**A (choice of K):** K-Means does not choose K on its own. Two tools:
- **Elbow (inertia/WCSS):** cluster compactness; always decreases with K → look for the **elbow** (where
  it stops falling steeply), a visual reading.
- **Silhouette (−1..+1):** **separation**; +1 well inside the cluster, 0 on the border, <0 wrong cluster.
  Take the K with the maximum silhouette (here between 3 and 7 → **K=6**).
- **Honest caveat:** silhouette **low and flat** (~0.21–0.26 for every K) → clusters exist but **poorly
  separated** (the road types blur). So K=6 is an **interpretable compromise**, not a clear optimum.
  **Say it yourself at the exam**, confirming with the sensible clusters in the heatmap.

## 20. Naming the clusters from the values (instead of the random auto_name)
**Q:** Can you give the names yourself by looking at the heatmap, explaining based on which values?
**A:** Yes. The old `auto_name` used *relative* thresholds (top 30%) → **duplicates**. `name_cluster`
rewritten to assign **from the values** (robust to renumbering). 6 clusters (seed-42 run, 7 features):
- **Intersection/traffic light (stopped):** speed 3.6 · stop **0.77**
- **Urban stop-and-go:** speed 22 · stop **0.28**
- **Approaching intersections/decel.:** speed 37 · accel_mean **−1.26**
- **Restarts/acceleration:** speed 36 · accel_mean **+0.97**
- **Free-flowing/fluid arterial:** speed 57 · very regular (std 10)
- **Highway/fast flow:** speed **84**
- **Finding 1:** the clustering separated accel (+) vs decel (−) **at equal speed** (C3 vs C4).
- **Finding 2 (a posteriori discovery):** the descriptive consumption `maf_mean_descr` orders them
  sensibly (highway 17.6 > restarts 13.2 > free-flowing 10.0 > stop-and-go 7.9 > decel 7.5 >
  intersection 4.5) — *kinematic* clusters that differ in consumption = result, not tautology.
  (MAF here is instantaneous g/s = how hard the engine works, not consumption per km.)
- **Naming logic:** stop>0.50 → intersection; stop>0.18 → stop-and-go; speed≥75 → highway; speed≥50 →
  free-flowing; otherwise by the sign of accel_mean → restarts/decel.

## 21. HTML map: show/hide individual clusters
**Q:** Can I select to display only some colors (clusters) at a time on the map?
**A:** Yes. The map cell (NB3) was rewritten: each cluster is a separate **`FeatureGroup`** + a
**`LayerControl(collapsed=False)`** → panel in the top right with a **checkbox per cluster**
(multi-select, not radio). Color legend in the bottom left. To regenerate by re-running NB3.
- **Exam demo:** *"the map is interactive — I isolate only the intersections or only the fast arterials to
  see how they are distributed in the city."*

## 18. Is Load useful for the driver profiles (aggressiveness)?
**Q:** An aggressive driver presses the accelerator more → doesn't Load help profile the styles?
**A:** Right intuition (Load = pedal effort, more *direct* than the resulting acceleration), but **no**:
- **Load is NOT powertrain-agnostic** (same flaw as MAF): it is a PID of the **combustion engine** → for
  hybrids in electric the engine is off → **Load=0**. A PHEV would look "gentle" only because the engine is
  off → it would falsify the **style × powertrain** comparison (the purpose of Part B).
- **Aggressiveness already captured** fairly by `frac_hard_accel`/`frac_hard_decel`/`accel_*` (they come
  from speed → they hold for all powertrains).
- **Exam:** "Load = same problem as MAF (0 in electric); I give aggressiveness with
  `frac_hard_accel/decel`, powertrain-agnostic."

---

> **Round 2026-06-17 — questions on NB2 (consumption / eco-driving).**

## 22. Keras in NB2 instead of sklearn, like in NB3?
**Q:** Could Keras be used in NB2 like in the third notebook?
**A:** **Wrong premise: NB3 does not use Keras** — it is sklearn (K-Means/PCA). Keras/torch were
only in NB4 (removed). One *could* put an MLP, but it is the **wrong** choice: on **tabular
data** **gradient boosting beats neural nets** (an established prior); the dataset is not huge; I would lose
the **feature importance** (the heart of the "road yes, terrain no" argument); no GPU/fragile tuning.
- **Exam:** *"on tabular data boosting is state of the art; I reserve nets for raw signals."*

## 23. Why 250 m per segment?
**Q:** Why 250 m and not less/more?
**A:** **A trade-off**, verified empirically with a **sensitivity cell** (present in the previous versions
of NB2, removed in the 06/18 pruning). Short (50-100 m)
→ noisy per-segment statistics, `maf_per_km` unstable on small distances, useless `next_*` horizon;
and **below the SRTM resolution (~30 m) the elevation change is only noise** → a segment that is too short
does not add terrain information, it falls into the *instantaneous* MAF tautology. Long (500-1000 m)
→ **mix heterogeneous stretches** (arterial + traffic light in the same average), losing the resolution needed
to plan. **250 m** ≈ a homogeneous stretch ~10-25 s of driving = the physical planning scale of an ACC,
**above the SRTM noise** but still fine for anticipation. Justified by **data + domain**, not by intuition.
- **Empirical confirmation (sensitivity cell SEG_LEN ∈ {150, 250, 500}, pre-06/18 run):**
  R² stays **stable** and the **terrain weight stays weak (~0.06) at all lengths** → the choice
  of 250 m is not arbitrary and does not change the conclusions (the limit is the *data*, not the granularity).
  *Note: the cell was removed from the notebook on 06/18 (pruning); the result remains citable as a
  performed analysis.* [[#36]] [[#42]]

## 24. What does the section-5 cell do (Target and features, map-only)?
**Q:** What does "cell 5" do?
**A:** (cell `[10]`) It embodies the **MAP-ONLY** decision, three things: (1) target `maf_per_km`; (2) list of
**20 features** = geometry/terrain + kinematics + **anticipation** `next_*` + context (hour/month/
weight/OAT), **excluding** RPM/Load/instantaneous MAF; (3) `dropna` → `seg_model`. In short: **it fixes what
is predicted and with which inputs, all known in advance to an ACC.**

## 25. The car doesn't know the RPM in advance either → why was excluding them right?
**Q:** The real ACC doesn't know the RPM before the stretch, like us: where is the difference?
**A:** The distinction is **of which segment**. The model predicts the **upcoming segment** → the features
must be known **before driving it**: geometry (map) + speed profile that the controller
**chooses** to try. The RPM/Load/MAF **of the future segment** are the **result** of driving it (to
measure them I would already need to know its consumption = the target) **and** quasi-circular with MAF (≈RPM×load).
The "current RPM" of the current stretch exist but **do not describe** the hypothetical future stretch to evaluate.
- **Exam:** *"I exclude them not because they are secret, but because they are the result of driving the stretch I
  want to predict — using them is using the answer."*

## 26. Why don't I use the NB3 clusters as features in NB2? (updates #5)
**Q:** Isn't the cluster an extra feature for consumption?
**A:** Historically **leakage** (cluster built on `maf_mean`/`rpm_mean` = masked target encoding).
**Post-#14 update:** now NB3 clusters **without** maf/rpm/load → **no longer leaky**. But the
reason that **still holds**: **no new info** — the cluster is a **lossy compression**
of `speed_*`/`accel_*`/`stop_fraction` that NB2 **already has raw**, + it creates a **pipeline
dependency** NB3→NB2. *"The clusters **describe** the context, they do not **feed** the consumption model."*

## 27. What is the look-ahead? Is it the future speed?
**Q:** Is the look-ahead the speed I know will be in the next segment? then shouldn't it be excluded?
**A:** **No, it is not the realized speed.** They are 3 **road** features of the next segment
(`next_dz_net`, `next_slope_mean`, `next_stop_fraction`) via `shift(-1)` (cell `[8]`). **Legitimate**
because they derive from the **map/route** (slope and stops ahead = known from the navigator) → it is precisely
the **anticipation** to demonstrate. **Illegitimate** (never done): forwarding the **target or what determines it**
(`next_MAF`, future RPM, realized speed). Honest nuance: `next_stop_fraction` is semi-behavioral,
used as a map-knowable proxy for "stops ahead" → the most borderline look-ahead feature, to be declared.

## 28. Why split train/test in time and not shuffle the trips?
**Q:** Why divide temporally and not randomly by segment?
**A:** **Correlation leakage.** Segments of the **same trip** are strongly correlated (same
driver/vehicle/weather). Random split by segment → pieces of the same trip in train **and** test → the
model **memorizes** the trip → **inflated R²**. Splitting by **whole trips ordered in time** (cell `[12]`,
80% oldest → train) = simulates the **real deployment** (train on the past, predict new trips) +
protects from drift. Same logic in CV with **GroupKFold per VehId** (cell `[18]`).

## 29. XGBoost estimators/depth/lr random? Can they be improved?
**Q:** Are the XGBoost parameters chosen at random? can't they be improved?
**A:** **No, and they are already optimized.** The **default** (cell `[16]`: 500/6/0.1) are standard values =
**baseline**. The **final** model (sec. 9-10, cells `[18]`/`[20]`) uses **Optuna** (Bayesian TPE
search, 30 trials) on estimators/depth/lr **+ regularization**, evaluated in **GroupKFold per VehId**,
min MAE → `study.best_params`. So the tuning **is already there**, more serious than a manual grid. It can be
pushed further (more trials, wider ranges, early stopping) with diminishing returns.
- **Exam:** *"the defaults are the standard; the finals are chosen by Optuna with Bayesian optimization and per-vehicle
  CV — not random, optimized."*

---

> **Round 2026-06-17 (bis) — pruning of NB3 and doc alignment.**

## 30. Pruning of NB3: what was removed and what it implies
**Q:** I kept in NB3 only what I present; align markdown and context files. Put all the
imports in a single cell.
**A:** Done. **Removed** from Part A: t-SNE and the **cluster↔EngineType** sanity check; from Part B:
the **chi-square** test style×powertrain, the **style→consumption** (ICE) relation and the **energy comparison** between
engines. Consequences and interventions:
- **Bug introduced by the deletions → fixed:** the PCA scatter cell had been left **orphaned** (it used
  `X_pca`/`pca` never defined → `NameError`). I **restored the PCA computation** before the scatter
  (section 8) — PCA is a technique you want to show and the scatter was kept.
- **Imports consolidated** in cell `[2]` (removed the scattered re-imports of `LogNorm`, `StandardScaler`,
  `KMeans`, `silhouette_score`, `PCA`). `import folium` stays in the `try/except`: it is an **optional
  dependency**, moving it up would break the fallback to the static map.
- **Typo "50×50 m" → fixed** to `~11×8 m` (see #6).
- **"Three powertrains" narrative:** Part B now **profiles the styles** (kinematics, powertrain-agnostic)
  and shows the composition by `EngineType`, but **no longer** statistically demonstrates "the powertrains are
  driven differently" nor "why hybrids consume differently". That thesis remains argued in
  **NB2 §1** and in `PROJECT_REPORT §5.1`; the three removed blocks are now **citable
  developments**, not results.
- **At the exam:** if the professor asks about the powertrain comparison, say it is **documented conceptually**
  but no longer computed in the notebook (a scope choice), do not pass it off as done.
- **Loose ends (for Alex):** `K_FINAL` is **forced to 4** in `[14]`; **two redundant summary
  cells** remain at the bottom (mergeable); the "Real result" numbers of the GUIDE Part A are from
  an old run → to be reconfirmed by re-running.

## 31. Should the "motion filter" be removed for the clustering? Two datasets (supervised without stops / clustering with stops)?
**Q:** Since we do clustering, isn't it better to remove the motion filter? Or make a supervised dataset
without the stationary moments and one for the clustering with the stops too?
**A:** **A misunderstanding to clear up: the filter does NOT remove the stops.** It is in **NB1 §5**:
`df[(speed > 0) | (RPM > 100)]` → keeps the row if it moves **OR** the engine is on. So a car stopped
at the **traffic light with the engine on, stop-and-go, queues → KEPT**. It removes only `speed=0` **AND** `RPM≤100`
(**engine off and stationary**: parking, trip start/end) = **334,788 rows, ~1.8%**.
- **Removing it for the clustering → worsens it:** the useful stops (`stop_fraction` for intersections/traffic lights) are
  **already inside** (engine on). The removed rows are **parking/trip-edges** → reinserting them would create
  "always stationary" cells in the parking lots = risk of a **"stationary" garbage cluster** that is not a road type.
  So the filter **helps** the clustering.
- **Supervised "without the stops" → is the opposite of what is needed:** in NB2 the `stop_fraction` is the
  **most important feature** ("stop-and-go costs"). Removing it would destroy the signal. The degenerate
  segments (near-stop, distance ~0) are already discarded at the segment level (`seg_distance_m ≥
  0.5·SEG_LEN` + finite `maf_per_km`) — no need to do it per row.
- **Recommendation:** **a single dataset filtered as is.** It keeps all stops/idle/stop-and-go for
  both tasks and removes ~1.8% of dead rows. Two datasets = more complexity, zero gain, parking-noise
  in the clustering. The only sensible *opposite* refinement would be to better exclude the parking lots
  from the clustering, but the ≥50 passages filter + kinematic features already handle most of it.

## 32. Wouldn't binning at 2 decimals (more precise) have been better?
**Q:** At 2 decimals it would have been much more precise, no?
**A:** **Inverted: fewer decimals = LARGER cells = LESS precise.** The current 4 decimals are the *finest*
choice, not the coarsest. Scale (lat ~42.3°N, lon × cos≈0.74):
- **2 dec** (0.01°) → **~1.1 km × 0.82 km** = a **whole neighborhood**/cell (highway+residential+
  intersection mixed) → meaningless clustering.
- **3 dec** (0.001°) → ~111 m × 82 m (a block).
- **4 dec** (0.0001°) → **~11 m × 8 m** ← current (a chunk of road).
- **5 dec** → ~1.1 m, below the **GPS noise** (~5-10 m) → only noise.
- **Real limit = source, not decimals:** GPS ~5-10 m (going below ~11 m adds no info); for
  the **slope** the limit is SRTM ~30 m + elevation sampled at **3 dec (~111 m)** in NB1 → that is what
  weakens the slope, not the clustering bin.
- **The only defensible change = the opposite direction:** switching to **3 decimals (~111 m)** would give more
  samples/cell and less discard (today the ≥50 filter throws ~72% of the cells because at 11 m they are tiny),
  at the price of mixing more road in the same cell. A compromise, not a clear "better" → **4 decimals
  are fine**.
- ⚠️ **Beware (see #33):** there are **two different roundings** — the **clustering bin** (NB3)
  is at **4 decimals ~11 m**; the **elevation sampling** for the API (NB1) is at **3 decimals ~111 m**.

## 33. So wouldn't it have been better to do the elevation at 4 decimals (more precise)?
**Q:** You did the elevation at 3 decimals with the API; wasn't 4 (more precise) better?
**A:** Right, the elevation API is at **3 decimals** (`ROUND_DECIMALS=3`, ~111 m) — not to be confused with the **clustering
bin** which is already at **4 decimals** (~11 m). On the **elevation**, 4 decimals would **not** have helped,
and NB1 §6 explains it:
- **Limit = source, not decimals:** behind Open-Meteo there is **SRTM ~30 m**; points < 30 m fall into the
  **same pixel** → same elevation. At 4 dec (~11 m) you **resample the same pixel** → the new Δelevations are
  **interpolation noise**, not climbs (it would look like signal → worse).
- **Cost:** `round(4)` → ~282k points → ~2800 batches → **>45 min** (+ rate-limit); `round(3)` → ~8,700
  points → ~88 batches → **1-2 min**. Same useful info, 32× faster.
- **Flat Ann Arbor:** the whole city's elevation range **223–324 m ≈ 101 m** → no slope to recover.
- **Real solution (not more decimals):** high-resolution DEM (LiDAR USGS 3DEP 1 m) or a hilly city.
  Links to [[#11]] (elevation derived by us, dedup at 3 dec) and [[#12]] (shrinking the cells does not recover slope).

## 34. Is elevation in meters or feet? Sometimes it makes "absurd drops" — did we get the units wrong?
**Q:** Are we sure `elevation_m` is in meters and not feet? Sometimes there are absurd elevation drops.
**A:** **It is in meters, no unit error** (verified on the real data):
- Mean elevation **268.8 m**, range **223–324 m**. Real Ann Arbor ≈ **256 m a.s.l. (840 ft)** → **it matches**.
  In **feet** the mean (269 ft) would be **82 m** → impossible. Open-Meteo/SRTM always gives meters.
- The "**absurd drops**" **are not a bug**, they are **quantization**: the elevation is **in integer steps** (102
  distinct values = ~1 per meter). Between consecutive points **dz = 0 in 91.9%** (same ~111 m block); at
  the block boundary the whole difference jumps at once → `p99 |dz| = 30 m`, **max |dz| = 94 m** (almost
  the whole city elevation excursion of 101 m in one step → not a cliff, it is the SRTM step / sometimes a GPS jump).
- This is why `slope` is **clipped to ±0.3** → **2.5%** of the rows hit it (those are precisely those drops).
- **Moral:** they seem absurd *because* Ann Arbor is super-flat (101 m total) → each step is a huge
  slice of the relief. It is the **limit of the elevation source**, not a meters/feet error. Links [[#33]], [[#10]].

## 35. The THREE grids/units of the project + what "round to N decimals" means
**Q:** If the elevation blocks are 111×82, then saying the road segment is 11×8 is wrong? And why
can't I choose a 10×10 grid? What does rounding to the 3rd/4th decimal mean?
**A:** **Nothing is wrong: they are different grids.** Careful not to confuse THREE "pieces":
| Unit | How | Size | Where |
|---|---|---|---|
| **NB2 segment** | ~250 m *along* the trajectory (cumulative distance) | 250 m linear | consumption (NB2) |
| **Clustering cell** | `round(4)` on lat/lon | **~11×8 m** | road segments (NB3-A) |
| **Elevation block** | `round(3)` on lat/lon | **~111×82 m** | elevation (NB1) |
- **Rounding to N decimals = snapping the coordinate to the nearest grid with step 10⁻ᴺ degrees.**
  All points in the same square → same value → same cell. 1° lat ≈ 111,320 m, so each
  decimal **divides by 10**: 2 dec=1.1 km · 3 dec=111 m · 4 dec=11 m · 5 dec=1.1 m. **Only ×10 jumps**,
  nothing in between. Rectangular (11×8, not 11×11) because longitude is worth `cos(42.3°)≈0.74` of latitude.
- **One elevation block (111 m) contains ~100 clustering cells (11 m)** → ~100 nearby cells share the
  SAME elevation → another reason `slope_mean` is weak at the 11 m scale.
- **Can I choose 10×10 (or 30×30)?** Yes, but NOT with `.round()`: you need a **metric snap**
  `round(coord/step)*step` with `step = meters/111320` (and `/cos(lat)` for longitude). `.round()` is only
  the convenient shortcut chained to the ×10 steps. See [[#6]], [[#32]], [[#33]].

## 36. Attempt at a 30 m elevation grid — explored and CANCELLED (2026-06-17)
**Q:** Let's do the elevation at 30 m (= native SRTM resolution) so maybe slope improves in the other NBs too?
**A:** Idea **right in direction** (the current 111 m are coarser than SRTM → they throw away real detail), and
verified cost: ~**54,591 points** at 30 m. **Cancelled** before the presentation because:
- **Real time ~1.5–2 h:** the free Open-Meteo API limits ~500–600 coordinates/min → recurring 429
  (60s waits; more `(n/12)` attempts = the limit has not freed up yet; reaching 12 = hourly/daily
  cap → the script stops, but it is **resumable**, it saves every batch). Then NB1+NB2+NB3 would have to be re-run.
- **Marginal gain:** Ann Arbor is flat (101 m) + SRTM has a vertical error ~6–16 m → slope would
  remain secondary anyway, and at 30 m some new Δelevations would be **noise** (fake slopes).
- **Decision:** everything restored to 111 m (`ROUND_DECIMALS=3`) from the commits/backup; `ved_enriched.parquet`
  had not been touched. It remains an excellent **citable future development** ("align the sampling to the native
  30 m of SRTM, or use a LiDAR 1 m DEM"). Links [[#33]], [[#12]].

## 37. NB3 vocabulary: `agg`, mean/std/abs, `stop_fraction`
**Q:** What is `agg`? What do mean/abs/std and the stop fraction mean?
**A:**
- **`agg`** = the **cells DataFrame** (NB3-A): `df.groupby(['lat_bin','lon_bin']).agg(...)` transforms
  17.9M rows → ~281k cells (1 row = 1 cell ~11×8 m), with means/variances of the per-cell behavior. It is
  **the input of the clustering**. Chain: `agg` → ≥50 filter → ~77k → `agg_clean` (dropna) → K-Means. It also computes
  `maf/rpm/load_mean` but they **do not** enter the clusters (only `maf_mean_descr` descriptive afterwards).
- **mean** = sum/count (typical value). **std** = how spread the values are around the mean
  (low=regular, high=jumpy). **abs** = magnitude without sign (`|−3|=3`).
- **Why `accel_abs_mean` and not just the mean:** on a mixed stretch the mean acceleration ≈ 0 (the +
  and − cancel out) → hides jerky driving. Taking the absolute value first measures the
  **intensity** (accelerations+braking), which does not cancel out. `std` captures the same "nervousness" as dispersion.
- **`stop_fraction`** = `(speed < 2).mean()` = **fraction of practically stationary samples** (0–1). Trick:
  the mean of True/False = the proportion of Trues. `<2` (not `=0`) for sensor noise. Readings: 0=free-flowing,
  ~0.2=stop-and-go, ~0.5–0.8=intersection/traffic light. It is the **most important feature of NB2** and the one that recognizes
  the intersections in NB3.

## 38. Is MAF the throttle opening and Load how much you press the accelerator?
**Q:** So MAF = throttle-body opening and Load = how much you press the accelerator?
**A:** **No, two corrections** — they are both *engine-response measures*, not pedal/throttle positions:
- **MAF** = **mass of air that actually enters** the engine (g/s, from a sensor). It is not the valve: the throttle
  opening *influences* the MAF, but the MAF is the measured **result**. A consumption proxy (air ≈ gasoline, ~14.7:1).
- **Load** (`Absolute_Load_pct`) = **how hard the engine works in % of the max** (air charge per cycle
  normalized). It correlates with the pedal but is the **actual engine effort**, not the pedal position.
- **Demand vs response:** pedal (demand) → the throttle opens → MAF ↑ (air) → Load ↑ (effort) → more gasoline.
  Neither MAF nor Load is the pedal/throttle; and in VED there is **no** pedal or throttle position (only MAF/RPM/Load).
- Bicycle analogy: RPM=cadence, Load=how much you push (effort %), MAF=calories/s (consumption). Detail in [[#13]].

## 39. Cluster heatmap (NB3): `slope_mean` and the descriptive MAF are not visible
**Q:** In the heatmap of the 4 clusters `slope_mean` and the descriptive MAF column do not appear.
**A:** Two distinct causes:
- **`slope_mean` invisible = `.round(2)` bug.** `cluster_profile` does `.mean().round(2)`, but the slope
  is ~0.00x → rounded it becomes **0.00 for all 4 clusters** (verified: 0.0/-0.0/-0.0/0.0) →
  variance 0 between clusters → the z-score divides by std=0 → **NaN → empty row**. **Fix:** compute the
  heatmap z-score from the **NON-rounded** means (`agg_clean.groupby('cluster')[...].mean()`),
  not from `cluster_profile`. (The `cluster_profile` table can stay rounded, it is only for reading.)
- **`maf_mean_descr` absent = intended.** The heatmap plotted only `FEATURES_CLUSTER` (the 7 that *form*
  the clusters); MAF is descriptive. **Fix:** added as an **extra row separated by a line**
  (`ax.axhline(len(FEATURES_CLUSTER))`), labeled "not for clustering".
- **Honest reading:** after the fix `slope_mean` appears but is **almost flat** between the clusters (small z-scores)
  → confirms that the terrain does not separate the road types. `maf_mean_descr` instead is **very marked**
  (highway warm ~16.7, intersection cold ~6.3) → the kinematic clusters differ in consumption too =
  a result, not a tautology [[#15]] [[#16]]. **The heatmap cell must be re-run.**

## 40. Would a linear model have been better on this data?
**Q:** Since MAF is skewed and `log_MAF` exists, wouldn't a linear model have been better?
**A:** **No, it is the opposite.** Two planes not to confuse:
- **The skew concerns the target, not the model choice.** The log-transform (`log_MAF`) is
  if anything a **crutch for linear models** (they assume ~normal/homoscedastic residuals) — it serves *them*,
  not XGBoost. So that detail does not push toward the linear one. [[#38]]
- **The data is strongly non-linear + with interactions:** consumption vs speed is U/J-shaped (high in city
  stop-and-go, minimum at cruise, high again on the highway); `stop_fraction` (feature #1), accel and
  speed **interact**. A linear model captures only additive/linear effects → to get closer you would need
  polynomial features and interaction terms built **by hand**. XGBoost does it **automatically** (splits +
  trees). It is the project's *prior*: tabular with non-linearities → gradient boosting. R² ~0.76 map-only.
- **When the linear would have been preferable (honesty):** direct interpretability of the coefficients;
  little data (not the case, 17.9M rows); **extrapolation** out of range (trees do not extrapolate, the
  line does); covering **L1/L2 regularization** as a course topic. A baseline **Lasso** would be
  the only real reason to reintroduce it (STATE.md decision #5 had removed it) — **not because it predicts
  better**, but for interpretable comparison / to cover L1/L2.
- **Sentence for the professor:** *"MAF's skew does not justify a linear model — it is the linear model that would need
  the log-transform. The data has strong non-linearity and interactions, so the correct prior on
  tabular data is gradient boosting; a Lasso remains useful only as an interpretable baseline."*

## 41. What is linear regression and what would it serve instead of XGBoost?
**Q:** What is a model like linear regression and what would it serve me in place of XGBoost?
**A:** **It is the simplest model to predict a number:** it assumes the target as a **weighted sum** of the
features, `ŷ = w₀ + w₁x₁ + … + wₙxₙ` (e.g. `maf_per_km ≈ w₀ + w₁·speed_mean + w₂·stop_fraction + …`).
- **How it learns:** least squares (OLS) — it finds the weights `w` that minimize the sum of squared
  errors; geometrically the line/hyperplane closest to the points.
- **The value is in the coefficients `w`:** **sign** = direction (`w_stop_fraction>0` → more stops = more
  consumption), **magnitude** = sensitivity ("+0.1 of stop → +X of maf_per_km, all else equal").
- **Regularized variants (course topics):** **Ridge (L2)** shrinks all the coefficients
  (multicollinearity); **Lasso (L1)** can **zero them** → automatic feature selection.
- **What it would serve INSTEAD of XGBoost** (not to predict better, but for what boosting does worse):
  (1) **direct interpretability** of the coefficients (XGBoost requires feature importance/SHAP); (2)
  **an honest baseline** = the model to beat (if XGBoost 0.76 vs linear 0.60 → non-linearity is worth
  +0.16, demonstrated); (3) **extrapolation** out of range (trees give a flat value, the line continues
  the trend); (4) covering **L1/L2** in the syllabus (absent today, decision #5); (5) little data (not the case,
  17.9M rows).
- **The price of the linear model:** it captures only linear/additive relations → the U curve "consumption vs speed"
  it gets wrong without polynomial features (`speed²`) and interactions (`speed×stop_fraction`) built by hand;
  it must be scaled; it wants a more symmetric target (here `log_MAF` would come in); sensitive to outliers.
- **The right move in the project:** not "linear *instead of*" but a **Lasso as a baseline alongside** XGBoost
  → it covers the regularization, gives readable coefficients, quantifies how much boosting is worth. [[#40]]

## 42. What are the features we give XGBoost (NB2)?
**Q:** Explain the features inside XGBoost.
**A:** `FEATURES_NUM` = **20 numeric features**, all at the **~250 m segment** level, all **map-only**
(known in advance to an ACC). *No scaling: XGBoost is tree-based → invariant to monotone transformations
(the Pipeline with StandardScaler was in the previous versions, removed on 06/18 — see [[#43]]).*
Four groups:
- **Terrain/geometry** (total weight ~0.06 → data limitation): `seg_distance_m` (length),
  `dz_net` (net elevation change = elev_end−elev_start), `climb_m`/`descent_m` (cumulative meters climbed/descended),
  `slope_mean`, `slope_max_abs`. Choice: **cumulative elevation change over the segment**, more robust than the
  spiky/zeroed instantaneous slope. [[#36]]
- **Kinematics** (strong signal): `speed_mean`, `speed_max`, `speed_min`, `speed_std` (variability),
  `accel_abs_mean` (accel+braking intensity, the abs prevents +/− from cancelling), `stop_fraction`
  (fraction of near-stationary `speed<2` = stop-and-go), `entry_speed` (entry speed `first` = inherited
  kinetic energy = backward coupling). **Importance:** stop_fraction #1 (~0.13), speed_mean
  (~0.08), entry_speed (~0.07).
- **Anticipation/look-ahead** (the "ACC" heart): `next_dz_net`, `next_slope_mean`, `next_stop_fraction`,
  via `groupby(['VehId','Trip']).shift(-1)` = info of the **next segment**. The future of the
  road/speed enters, **NEVER the future MAF** (it would be target leakage). [[CLAUDE.md traps]]
- **Vehicle/environment context:** `Generalized_Weight` (vehicle weight class), `OAT_DegC`
  (outside temperature → cold/warm engine, climate), `hour` and `month` (traffic/season proxy).
  In the importance plot they end up in the "kinematics/context" group.
- **Deliberately excluded:** `Engine_RPM_RPM`, `Absolute_Load_pct`, `rpm_roll10s_mean` (not known in advance
  + quasi-circular with MAF ≈ RPM×load → a shortcut that zeroes out the role of the road); `EngineType`
  (ICE-only model → constant). **Target:** `maf_per_km` (efficiency, not distance). [[#40]]

## 43. What is the ColumnTransformer and why does it go with the StandardScaler?
**Q:** What was the ColumnTransformer and why does it go together with the StandardScaler?
**A:** They are **two distinct things**, often confused:
- **StandardScaler** = *transforms* the features to **mean 0, std 1** (same scale). Needed because `speed_mean`
  (tens) and `slope_mean` (~0.00x) have very different scales.
- **ColumnTransformer** = the *container* that says **to which columns** to apply which transformer (it does not
  scale anything by itself, it orchestrates). Classic case: numeric→StandardScaler, categorical→OneHotEncoder,
  recomposed into a single matrix.
- **In NB2 (versions up to 06/17):** `ColumnTransformer([('num', StandardScaler(), FEATURES_NUM)])`
  = a single group (the numeric ones), no categorical (`EngineType` excluded, ICE-only). **Why use it
  with a single transformer?** (1) a standard/scalable scaffold (tomorrow you add a categorical with one
  line); (2) it selects exactly the columns by name (you don't scale ID/target by mistake); (3) it fits
  into the **Pipeline** with the model. *On 06/18 the NB2 pruning removed Pipeline/ColumnTransformer/
  StandardScaler: XGBoost works directly on the columns (see Honest note below). The concept remains
  exam material.* [[#42]]
- **Why inside the Pipeline = anti-leakage:** the scaler is **fitted only on the train** (mean/std) and
  *applied* to test/folds with those parameters. Scaling the whole dataset by hand before the split → mean/std
  also see the test → **leakage** → false metrics. The Pipeline seals it, especially in CV (GroupKFold,
  scaler refitted at each fold). [[#22]]
- **Honest note:** XGBoost is tree-based → **invariant to monotone transformations**: the StandardScaler **does
  not change its predictions**. It was kept for good practice/reusability (a Lasso, or the NB3 K-Means/PCA
  *require* it) — and it is precisely for this that removing it from NB2 (06/18) is legitimate. In NB3 the scaling
  remains **mandatory** (explicit "without scaler" demonstration). [[#41]]
