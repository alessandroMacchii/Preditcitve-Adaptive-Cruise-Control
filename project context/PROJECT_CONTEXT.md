# PROJECT_CONTEXT.md — Onboarding per sessioni Claude Code

> Leggere questo file all'inizio di ogni sessione. Lo stato corrente è in `STATE.md`.

## Utente

Alex, studente AI/ML del corso **UFS08-UFS09** (Machine Learning Supervised + Unsupervised) all'ITS Angelo Rizzoli. Progetto d'esame. Email: alemacchi18@gmail.com.

## Idea narrativa

I cruise control tradizionali mantengono una velocità fissa. Un cruise control **adattivo predittivo** — che conosce in anticipo l'orografia e la tipologia della strada (da mappe/V2X) — può modulare la velocità per ridurre il consumo. Il progetto dimostra questa idea su telemetria reale con due modelli ML complementari:

1. **Supervised**: predice il consumo aria istantaneo (`MAF_g_per_sec`, proxy del consumo carburante) dallo stato veicolo + feature *look-ahead* (pendenza/velocità futura). È il "simulatore di consumo" che un cruise control userebbe per confrontare strategie di velocità (analisi controfattuale eco/sport).
2. **Unsupervised**: clusterizza i tratti stradali di Ann Arbor in tipologie di guida (urbano, rettilineo veloce, salita...) tramite K-Means su celle spaziali aggregate. È il "riconoscitore di contesto" del cruise control.

## Dataset

**VED (Vehicle Energy Dataset)**, Università del Michigan, versione Refined da Kaggle.
- 54 file parquet settimanali in `content/` (schema identico, Phase 1 nov 2017–mag 2018 + Phase 2 mag–nov 2018, non distinte nel codice)
- **18.26M righe** di telemetria OBD-II, ~250 veicoli (ICE/HEV/PHEV), Ann Arbor MI (~12×10 km)
- Colonne chiave: `VehId`, `Trip`, `Timestampms` (ms relativi al trip), `Datetime` (**costante per trip** = inizio trip), `Latitude_deg`, `Longitude_deg`, `Vehicle_Speed_km_per_h`, `MAF_g_per_sec`, `log_MAF` (già presente nel raw), `Engine_RPM_RPM`, `Absolute_Load_pct`, `OAT_DegC`, fuel trim Bank 1/2, `EngineType`, `Generalized_Weight`, `__index_level_0__` (artefatto)
- Zero NaN dichiarati; sampling rate irregolare (100–1400 ms)
- **Niente altitudine** nel raw → enrichment via Open-Meteo Elevation API

## Struttura dei notebook

| Notebook | Cosa fa | Output |
|---|---|---|
| `01_data_prep_and_enrichment.ipynb` | Carica i 54 parquet, EDA, filtro outlier (Load>200%, RPM>8000, Speed>200, MAF>300), filtro movimento (Speed>0 ∨ RPM>100), elevation via Open-Meteo con dedup spaziale a **3 decimali** (~8.7k punti unici, ~88 batch), slope via Haversine, accelerazione con dt reale | `outputs/ved_enriched.parquet` |
| `02_supervised_maf_prediction.ipynb` | Sample stratificato 30% trip per EngineType. Feature: istantanee, rolling trailing (30/10 campioni), **look-ahead** (slope/speed prossimi 5/10/30 campioni, via `FixedForwardWindowIndexer`), temporali. Split temporale per trip 80/20. Pipeline+ColumnTransformer (StandardScaler + OHE drop='first'). 6 modelli: baseline mean, Ridge, Lasso, RF, XGBoost default, XGBoost+Optuna (50 trial TPE, GroupKFold per VehId). Metriche MAE/RMSE/MAPE/R². Feature importance. Controfattuale eco/sport | `xgb_maf_model.joblib`, `supervised_results.csv` |
| `03_unsupervised_road_clustering.ipynb` | Aggregazione in celle spaziali (round 4 decimali su lat/lon), filtro ≥50 passaggi, StandardScaler con dimostrazione del perché, Elbow+Silhouette per K, K-Means (k-means++, n_init=50), heatmap z-score + naming euristico, PCA con loadings, t-SNE bonus, mappa Folium | `road_segment_clusters.parquet`, `cluster_profile.csv`, `cluster_map.html/.png` |

File di supporto: `build_elevation_cache.py` — script standalone **resumabile** che costruisce `outputs/elevation_cache.parquet` (salva dopo ogni batch, gestisce 429 con Retry-After). Da usare PRIMA del Notebook 1; il notebook poi carica la cache e salta le API.

## Scelte tecniche chiave (e perché)

1. **Split train/test temporale per trip** (80% trip più vecchi → train): simula deployment reale, no leakage temporale
2. **GroupKFold per VehId** nel tuning: lo stesso veicolo non sta in train e validation insieme
3. **Feature look-ahead = pendenza/velocità FUTURE** (info da mappe), mai MAF futuro → non è leakage, è il punto del progetto
4. **Dedup spaziale a 3 decimali (~111 m)** prima di Open-Meteo: coerente con risoluzione SRTM ~30 m (il prompt originale diceva 4 decimali, poi rivisto: 4 decimali = 282k punti = inutilizzabile e più fine della sorgente)
5. **Cache locale elevation** (`elevation_cache.parquet`) per non rifare le chiamate
6. **Clustering su celle spaziali aggregate**, non su righe singole: l'unità è il tratto stradale
7. **Niente lat/lon tra le feature del clustering**: vogliamo tipologie, non vicinati
8. **StandardScaler obbligatorio** per Ridge/Lasso e K-Means/PCA (dimostrato esplicitamente nel NB3)
9. **OneHotEncoder(drop='first')** contro la dummy variable trap
10. **n_init=50** nel K-Means finale contro i minimi locali

## Argomenti del corso coperti

Pipeline + ColumnTransformer, Ridge/Lasso (L2/L1), Random Forest, XGBoost, GroupKFold CV, Optuna (tuning bayesiano TPE), MAE/RMSE/MAPE/R², feature importance, data leakage e split temporale, K-Means/K-Means++, elbow, silhouette, StandardScaler motivato, PCA con loadings, t-SNE.
**Trattati in aula ma NON presenti nel progetto:** autoencoder (Keras 3 + PyTorch backend: denso, convoluzionale, denoising, inpainting su MNIST) e CNN (Conv2D, MaxPooling, leaky_relu). Sono materia d'esame.

## Tono e stile richiesti da Alex

- Tecnico, conciso, in italiano; onesto e critico (non assecondare)
- Nei notebook: markdown narrativo prima delle celle di codice (mantenere lo stile)
- NON eseguire senza chiedere: Optuna 50 trial, fit su 18M righe, chiamate Open-Meteo
- OK eseguire celle veloci di esplorazione (head, describe, len)
- I PDF di teoria e i notebook del prof NON sono in questa cartella: chiedere ad Alex se servono

## Vincoli operativi

- `.claude/settings.json`: whitelist python/pip/jupyter/fs base; deny rm/rmdir/sudo/git push/git reset --hard. Non modificarlo senza chiedere.
- `.gitignore` esclude `content/`, `outputs/`, `.claude/`, `.venv/`
- Ambiente: Windows 10, venv in `.venv/` (usare `./.venv/Scripts/python.exe`)
- `truststore` necessario per le chiamate HTTPS (proxy/antivirus che intercetta SSL)
