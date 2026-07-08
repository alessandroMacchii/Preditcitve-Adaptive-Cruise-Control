# Decision Log

## D1 — Reframe: from the "orographic ACC" to "energy + context" `2026-06-14`

- **Context.** Starting idea: a predictive ACC that exploits **orography** ("slow down before
  the climb"). First notebooks built around slope.
- **Alternatives.** (a) Insist on orography; (b) change dataset (see D2); (c) reframe
  the project around the signal the data actually contains.
- **Decision.** (c): the predictive signal for consumption is the **anticipated speed/traffic
  profile**; terrain remains documented as a **data limitation**.
- **Why.** The EDA falsified the hypothesis: Ann Arbor is almost flat and the elevation (deduped to ~111 m)
  is quantized → **slope null in 92% of the rows**, slope–MAF correlation **0.004**. Insisting
  would have produced a model that "demonstrates" a signal that does not exist in the data. It is the
  *Test → Learn → Iterate* cycle of an MVP: the first cycle invalidated the hypothesis, the second
  restarted from the real signal.
- **Evidence.** `PROJECT_REPORT.md` (history and numbers), commit `12aa60e` (2026-06-15, the
  restructuring), `VED_DATA_ANALYSIS.md`.

## D2 — Staying on VED (dataset) `2026-06-14`

- **Context.** After D1 we asked whether another dataset had the orographic signal.
- **Alternatives.** (a) An OBD-II "allcars" dataset — discarded: **no GPS**, impossible to build
  the road context and anticipation; (b) an IMU classification dataset — discarded: 4 classes,
  ~1100 rows, duplicate files, unrelated to consumption; (c) stay on VED.
- **Decision.** (c) VED.
- **Why.** No alternative solved the terrain problem *and* all of them lost what
  makes the project possible (GPS + telemetry + heterogeneous fleet). Better a declared limitation
  than an unsuitable dataset.
- **Evidence.** `STATE.md` (decision history #2), `PROJECT_REPORT.md`.

## D3 — "MAP-ONLY" features for the consumption model `2026-06-15`

- **Context.** Which features to give the model that predicts `maf_per_km` for the upcoming segment.
- **Alternatives.** (a) All available features, including RPM and engine load → higher R²;
  (b) only features known **in advance** to an ACC (map, expected kinematics, look-ahead).
- **Decision.** (b): `Engine_RPM_RPM`, `Absolute_Load_pct`,
  `rpm_roll10s_mean` deliberately excluded (and `EngineType`, constant in the ICE-only model).
- **Why.** Two independent reasons: (1) in the real use case those signals **are not known in
  advance** for the future segment; (2) they are **quasi-circular** with the target (MAF ≈ RPM × load):
  including them makes the model learn a shortcut and zeroes out the role of the road — better R², worse
  model. It is the decision that defines the project.
- **Evidence.** `CLAUDE.md` (dedicated section), `discussions.md` #24–#25, the "Target and features"
  cell of NB2.

## D4 — Consumption modeled for ICE ONLY `2026-06-15`

- **Context.** The VED fleet has ICE, HEV and PHEV; the consumption target is based on MAF.
- **Alternatives.** (a) Model all powertrains; (b) all of them, "accounting for the time in
  electric mode"; (c) ICE only.
- **Decision.** (c) ICE only.
- **Why.** MAF is a consumption proxy valid **only for combustion engines**: hybrids travel in
  electric mode for ~21–24% of the motion with MAF=0 but real consumption ≠ 0. Option (b) was analyzed
  and discarded with a technical argument: VED **has no SoC** (battery state) → the electric
  fraction is an unobserved latent variable; estimating it from the target = **leakage**. The
  correct path (*hurdle* model with SoC) requires data that is not there → future development.
- **Evidence.** `PROJECT_REPORT.md` §5.1, `STATE.md` (history #3 and #6), NB2 markdown §1.

## D5 — Target per ~250 m segment (not instantaneous MAF) `2026-06-15`

- **Context.** At what granularity to predict consumption.
- **Alternatives.** (a) Instantaneous MAF per row; (b) short 50–100 m segments; (c) ~250 m;
  (d) 500–1000 m.
- **Decision.** (c) `maf_per_km` on ~250 m segments.
- **Why.** (a) is a physical quasi-tautology (predicting the present from the present, terrain is
  noise); (b) noisy statistics, unstable `maf_per_km`, and below the SRTM resolution (~30 m)
  the elevation change is only noise; (d) mixes heterogeneous stretches (arterial + traffic light in the same
  average) and loses the resolution useful for planning. 250 m ≈ a homogeneous stretch, ~10–25 s of driving =
  the planning scale of an ACC. **Empirical check**: sensitivity cell with
  SEG_LEN ∈ {150, 250, 500} → stable R² and terrain weight ~0.06 at all lengths (the choice
  does not condition the conclusions).
- **Evidence.** `discussions.md` #23; the sensitivity cell was later removed in the final notebook
  pruning, the results remain documented.

## D6 — Validation: temporal split per trip + GroupKFold per vehicle `2026-06-15`

- **Context.** How to honestly estimate the consumption model's performance.
- **Alternatives.** (a) Random split per row/segment; (b) temporal split per trip + GroupKFold
  per `VehId` in the tuning.
- **Decision.** (b).
- **Why.** A random split would put segments of the same trip (and vehicle) on both
  sides → the model "recognizes" the trip/vehicle instead of learning the road, inflated metrics.
  The temporal one simulates the real use (predicting the future); GroupKFold guarantees that no vehicle
  is in both train and validation. Same anti-leakage philosophy as the rule on the
  `next_*` features: never the future MAF among the inputs.
- **Evidence.** NB2 split/tuning cells, `discussions.md` #22, `CLAUDE.md` (known traps).

## D7 — Clustering: K=4 for the segments, features kinematics+geometry only `2026-06-17`

- **Context.** How many clusters, and on which features, for the road-segment typing.
- **Alternatives.** K from 2 to 10 (evaluated with Elbow + Silhouette); whether or not to include consumption
  (`maf_mean`) among the clustering features.
- **Decision.** **K=4** (intersection / mixed urban / free-flowing / highway); consumption **excluded**
  from the clustering features, kept only as a descriptive column; spatial cells ~11×8 m with a
  ≥50 passages filter; StandardScaler mandatory.
- **Why.** K=4 is the best elbow/silhouette compromise *and* gives interpretable clusters that
  trace the real road network on the map. Keeping consumption out of the features avoids
  circularity: this way "the kinematic clusters differ in consumption too" is a **result**
  (highway ~16.7 vs intersection ~6.3 mean MAF), not a tautology. The scaler is mandatory
  because the features have very different scales (explicit demonstration in the notebook).
- **Evidence.** `clustering.ipynb` (Elbow/Silhouette and "Final choice" cells), `discussions.md`
  #15–#16 and #39, `outputs/cluster_profile.csv` + `cluster_map.html`.

## D8 — 30 m elevation grid: explored and CANCELLED `2026-06-17`

- **Context.** The elevation is deduped to 3 decimals (~111 m); native SRTM goes down to ~30 m: was it
  worth redoing the cache at full resolution?
- **Alternatives.** (a) Resample everything to 30 m (~54,591 points to download); (b) stay at ~111 m.
- **Decision.** (b) — explored, quantified, **cancelled**; everything restored to
  `ROUND_DECIMALS=3`.
- **Why.** Cost ~2h of API calls (rate-limit) for an expected marginal gain on a flat
  city: the bottleneck is the orography of Ann Arbor, not the grid resolution.
  A cost/benefit decision ("estimating is complex, not doing it is diabolical"), not a technical one. It remains a
  sensible future development on hilly terrain.
- **Evidence.** `discussions.md` #36 (with the numbers), `STATE.md`.

---

## How to read this log

The decisions are not independent: D1 (reframe) generates D2 (dataset) and gives meaning to D3 (map-only) and
D5 (segment); D6 (validation) is the anti-leakage discipline that keeps the consumption model honest;
D8 (the cancelled 30 m grid) shows that even a **no** is a decision to track, with the numbers that justify it.

In course terms: this log is the defense against **technical debt** in its
original definition (Cunningham) — the misalignment between what is needed and what was built grows
when the choices are not explicit. Every entry here is a declared "loan", with its cost
written next to it.
