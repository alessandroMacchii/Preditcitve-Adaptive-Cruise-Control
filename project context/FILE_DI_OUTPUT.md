# File di output (`outputs/`) — cosa sono e chi li produce

> Spiega ogni file nella cartella `outputs/`: cosa contiene, quale notebook lo genera, se è input o
> risultato. **Tutti i file attuali provengono dai 4 notebook correnti**: nessun residuo vecchio
> (i vecchi `xgb_maf_model.joblib`, `segment_fuel_model.joblib`, `supervised_results.csv`,
> `segment_*` sono stati cancellati il 2026-06-14).
>
> Nota: `outputs/` è in `.gitignore` (non versionata): tutto qui è **rigenerabile** eseguendo i
> notebook. Gli unici due file "preziosi" da non perdere a cuor leggero sono `ved_enriched.parquet`
> (rigenerarlo = rieseguire NB1, con chiamate API) e `elevation_cache.parquet` (la cache delle quote).

## Tabella riassuntiva

| File | Prodotto da | Tipo | Contenuto |
|---|---|---|---|
| `ved_enriched.parquet` | **NB1** | input di tutti | dataset consolidato + elevation + slope + accel (~17,9M righe, 361 MB) |
| `elevation_cache.parquet` | **NB1** / `build_elevation_cache.py` | cache | quote Open-Meteo per punto unico (evita di richiamare l'API) |
| `consumption_model.joblib` | **NB2** | modello | pipeline XGBoost tunato del consumo (`maf_per_km`, solo ICE) |
| `consumption_results.csv` | **NB2** | risultati | tabella metriche (MAE/RMSE/MAPE/R²) per modello |
| `consumption_seglen_sensitivity.csv` | **NB2** | risultati | R² e peso del terreno per SEG_LEN ∈ {150,250,500} |
| `road_segment_clusters.parquet` | **NB3** (Parte A) | risultati | celle stradali con feature aggregate + etichetta `cluster` |
| `cluster_profile.csv` | **NB3** (Parte A) | risultati | profilo medio dei cluster stradali |
| `cluster_map_static.png` | **NB3** (Parte A) | mappa | mappa di Ann Arbor colorata per cluster (statica) |
| `cluster_map.html` | **NB3** (Parte A) | mappa | mappa interattiva Folium (11 MB) |
| `telemetry_autoencoder.keras` | **NB4** | modello | autoencoder addestrato (Keras 3 / PyTorch) |
| `anomaly_scores.parquet` | **NB4** | risultati | campione 200k righe + `recon_err`, `anomaly_ae`, `anomaly_iso` |

## Dettaglio per notebook

### NB1 → input/cache
- **`ved_enriched.parquet`**: l'output fondamentale, lo leggono tutti gli altri notebook. Se lo
  cancelli devi rieseguire NB1 (lento, con chiamate Open-Meteo).
- **`elevation_cache.parquet`**: cache delle altitudini per coordinata arrotondata; serve solo a NB1
  per non riscaricare. Minuscola, tienila.

### NB2 (consumo) → `consumption_*`
- **`consumption_model.joblib`**: la pipeline finale (ColumnTransformer + XGBoost tunato) salvata con
  joblib; riutilizzabile per predire il consumo di nuovi segmenti.
- **`consumption_results.csv`**: confronto modelli.
- **`consumption_seglen_sensitivity.csv`**: sensibilità alla lunghezza del segmento.

### NB3 (contesto + stili) → `road_segment_clusters`, `cluster_*`
- **`road_segment_clusters.parquet`**: l'output della **Parte A** (tratti stradali), una riga per
  cella spaziale con statistiche aggregate + `cluster`.
- **`cluster_profile.csv`**, **`cluster_map_static.png`**, **`cluster_map.html`**: profili e mappe
  dei cluster stradali.
- ⚠️ La **Parte B** (stili di guida × powertrain) **non salva file**: i suoi risultati (tabella
  stile×EngineType, chi-quadro, confronto energetico) sono mostrati **dentro il notebook** (grafici
  e stampe). Se servono come file, vanno salvati esplicitamente.

### NB4 (diagnostica) → `telemetry_autoencoder`, `anomaly_scores`
- **`telemetry_autoencoder.keras`**: il modello autoencoder addestrato.
- **`anomaly_scores.parquet`**: il campione di 200k righe con l'errore di ricostruzione e i flag di
  anomalia (autoencoder e Isolation Forest); include `EngineType`.

## ⚠️ Disallineamenti da sapere (al 2026-06-14)
- I `consumption_*` provengono da un'esecuzione di NB2 **precedente alla semplificazione** (la
  `consumption_results.csv` elenca ancora **6 modelli**: baseline, Ridge, Lasso, RF, XGBoost
  default/tunato). Il NB2 attuale ne ha **solo 2** (XGBoost default + tunato). → **Rieseguendo NB2**
  il CSV si riallinea a 2 righe. Il `consumption_model.joblib` è comunque l'XGBoost tunato (valido).
- Se modifichi un notebook, ricordati che il suo output su disco resta quello dell'ultima esecuzione
  finché non lo rilanci.

## Pulizia
Per fare spazio si possono cancellare **tutti** i file generati (consumption_*, cluster_*,
road_segment_clusters, telemetry_autoencoder, anomaly_scores): si rigenerano rieseguendo i notebook.
**Non** cancellare `ved_enriched.parquet` né `elevation_cache.parquet`.
