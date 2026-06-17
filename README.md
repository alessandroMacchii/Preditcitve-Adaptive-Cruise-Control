# ML per l'energia e il contesto di guida — Progetto VED

Progetto di Machine Learning **supervised + unsupervised** sul Vehicle Energy Dataset (VED)
dell'Università del Michigan (~17,9M righe di telemetria OBD-II reale, Ann Arbor MI).

**Cosa dimostra.** Tre capacità utili a un **assistente di guida / cruise control adattivo**:
1. **Consumo / eco-driving** — stimare il consumo per km in funzione del profilo di velocità
   *anticipato* (modello supervised, NB2).
2. **Contesto + stili di guida** — riconoscere tipologie di tratto e profili di guida, e confrontare
   i powertrain ICE/HEV/PHEV (clustering, NB3).

> **Nota di inquadramento.** L'idea di partenza era un ACC che sfrutta l'**orografia**. I dati hanno
> mostrato che su VED la pendenza è un segnale debole (Ann Arbor quasi piatta + altitudine
> quantizzata a ~111 m → slope nullo nel 92% delle righe, correlazione col consumo ~0). Il progetto
> è stato reinquadrato: il segnale reale è il **profilo di velocità/traffico** anticipato; il terreno
> resta documentato come *limite del dato*. Inoltre il MAF è un proxy di consumo valido **solo per
> gli ICE** (gli ibridi vanno spesso in elettrico → MAF=0): il modello di consumo è quindi solo-ICE,
> e i tre powertrain sono confrontati a parte. Dettagli in `project context/RELAZIONE_PROGETTO.md`.

## Struttura (3 notebook)

```
├── 01_data_prep_and_enrichment.ipynb        # Caricamento, EDA, enrichment elevation/slope/accel
├── 02_consumption_ecodriving.ipynb          # Consumo per km su segmenti ~250 m (solo ICE)
├── 03_unsupervised_context_and_styles.ipynb # A) tratti stradali  B) stili di guida × powertrain
├── content/                                 # parquet VED (caricati ricorsivamente)
└── outputs/
    ├── ved_enriched.parquet                 # Output NB1, input di tutti gli altri
    ├── elevation_cache.parquet              # Cache chiamate Open-Meteo
    ├── consumption_model.joblib             # Modello consumo a segmento (NB2)
    ├── road_segment_clusters.parquet        # Tratti con etichetta cluster (NB3)
    └── *.csv / *.png / *.html               # risultati e mappe
```

## Documentazione (in `project context/`)

- `RELAZIONE_PROGETTO.md` — panoramica per l'esame (dataset, criticità, scelte) con numeri reali.
- `ANALISI_DATI_VED.md` — analisi esplorativa completa del dataset (EDA con numeri reali).
- `discussioni.md` — registro Q&A per il ripasso d'esame.
- `STATE.md` — stato vivo della pipeline.

## Setup

```bash
pip install -r requirements.txt
```

## Esecuzione (in ordine)

1. **NB1** — una sola volta. Produce `ved_enriched.parquet` (~17,9M righe). ~5–10 min.
2. **NB2** — consumo a segmento (solo ICE). Optuna leggero → pochi minuti.
3. **NB3** — clustering tratti + stili di guida × powertrain. ~5–10 min.

## Note

- Il NB1 chiama Open-Meteo Elevation API (gratuita, no key) e mette in cache i risultati.
- Ambiente: `.venv` del progetto (`./.venv/Scripts/python.exe`), Windows.
