# Cruise Control Adattivo Predittivo — Progetto VED

Progetto di Machine Learning supervised + unsupervised basato sul Vehicle Energy Dataset (VED) dell'Università del Michigan.

## Struttura

```
project_ved/
├── 01_data_prep_and_enrichment.ipynb   # Caricamento, EDA, enrichment elevation, slope
├── 02_supervised_maf_prediction.ipynb  # Modello supervised XGBoost per predire MAF
├── 03_unsupervised_road_clustering.ipynb # K-Means per clusterizzare tratti stradali
├── content/            # tutti i file parquet del VED (caricati ricorsivamente)
└── outputs/
    ├── ved_enriched.parquet         # Output del notebook 1, input dei notebook 2 e 3
    ├── elevation_cache.parquet      # Cache delle chiamate Open-Meteo (per non rifarle)
    ├── xgb_maf_model.joblib         # Modello supervised finale
    ├── road_segment_clusters.parquet # Segmenti con etichetta cluster
    ├── cluster_profile.csv          # Profilo numerico dei cluster
    ├── cluster_map_static.png       # Mappa statica
    ├── cluster_map.html             # Mappa interattiva (folium)
    └── supervised_results.csv       # Tabella comparativa modelli supervised
```

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna pyarrow requests folium joblib
```

## Esecuzione

I notebook vanno eseguiti in ordine:

1. **Notebook 1** — Una sola volta. Produce `ved_enriched.parquet`.
   - Tempo: ~5-10 min (incluse le chiamate API Open-Meteo)
   - Output: dataset arricchito di ~17M righe
2. **Notebook 2** — Modello supervised.
   - Tempo: ~30-60 min (Optuna tuning incluso)
3. **Notebook 3** — Clustering.
   - Tempo: ~5-10 min

## Note importanti

- Il **notebook 1** chiama Open-Meteo Elevation API (gratuita, no API key). Mette in cache i risultati in `elevation_cache.parquet`: la seconda volta che esegui non rifa le chiamate.
- Il **notebook 2** lavora su un sample del 30% dei trip (configurabile via `SAMPLE_FRAC`). Per il deliverable finale puoi alzare al 100% — ricorda che il tuning Optuna è la fase più lenta.
- Il **notebook 3** richiede `folium` solo per la mappa interattiva. Se non installato, la mappa statica `.png` viene comunque generata.

## Narrativa per la presentazione

"I cruise control attuali mantengono una velocità fissa. Un cruise control *predittivo* — che conosce in anticipo l'orografia della strada — può ridurre il consumo modulando dinamicamente la velocità. In questo progetto ho costruito due modelli ML che insieme dimostrano questa idea su 17M punti di telemetria reale.

Il **modello supervised** predice il consumo aria istantaneo (MAF) data lo stato veicolo + l'informazione look-ahead sulla strada. Il **modello unsupervised** classifica i tratti stradali in categorie di stile di guida ottimale (urbano, rettilineo veloce, salita). Insieme costituiscono la base di un cruise control adattivo intelligente."
