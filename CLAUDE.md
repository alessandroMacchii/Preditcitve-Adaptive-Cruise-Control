# CLAUDE.md — Onboarding per nuove sessioni

> File caricato automaticamente da Claude Code a inizio sessione. Tenuto **conciso di
> proposito**: contiene fatti stabili + puntatori. Lo **stato corrente** (sempre
> mutevole) sta in `STATE.md`: **leggilo per primo**, e aggiornalo quando fai qualcosa.

## Cos'è il progetto
ML supervised + unsupervised sul **Vehicle Energy Dataset (VED)** per dimostrare due capacità
utili a un **assistente di guida / cruise control adattivo**: (1) stimare il **consumo** in
funzione del profilo di velocità *anticipato* (eco-driving), (2) **riconoscere il contesto
stradale** (tipologie di tratto) e gli **stili di guida**. Progetto d'esame di Alex.

> **Inquadramento (rivisto 2026-06-14).** L'idea iniziale era un ACC che sfrutta l'**orografia**.
> Esplorando i dati la pendenza si è rivelata **debole** (Ann Arbor piatta + quantizzazione
> elevation a ~111 m: slope nullo nel 92% delle righe, corr con MAF 0,004). Il progetto è stato
> quindi **reinquadrato**: il segnale predittivo reale è il **profilo di velocità/traffico**
> anticipato, non le salite; il terreno resta documentato come *limite del dato*. I dataset
> alternativi valutati (OBD-II senza GPS, IMU) non risolvevano il problema → si resta su VED.
> Storia e numeri in `RELAZIONE_PROGETTO.md`.

## I 3 notebook (consolidati 2026-06-14)
1. `01_data_prep_and_enrichment.ipynb` — carica i parquet VED, EDA, arricchisce con altitudine
   (Open-Meteo) e calcola pendenza + accelerazione. Output: `outputs/ved_enriched.parquet`.
2. `02_consumption_ecodriving.ipynb` — **consumo / eco-driving**: predice `maf_per_km` su segmenti
   ~250 m, **solo ICE** (MAF è proxy valido solo per i termici), feature map-only + anticipazione.
   Esclusi RPM/Load e EngineType.
3. `03_unsupervised_context_and_styles.ipynb` — **contesto + stili di guida**: Parte A clustering
   dei tratti stradali (K-Means/PCA/mappa Folium, solo cinematica+geometria); Parte B clustering dei
   **guidatori** per stile (cinematica, powertrain-agnostica) + PCA. *(Sfoltito il 17/06: rimossi
   t-SNE, sanity check cluster↔EngineType, test chi-quadro stile×powertrain, stile→consumo, confronto
   energetico — restano come sviluppi.)*

> **I notebook nuovi non sono ancora stati eseguiti** (lo fa Alex).

## La decisione che definisce il progetto: modello "MAP-ONLY"
Il NB2 usa **solo feature note in anticipo a un ACC** (velocità, accelerazione, pendenza
attuale/futura, velocità futura, contesto). Sono **esclusi di proposito** i segnali-motore
`Engine_RPM_RPM`, `Absolute_Load_pct`, `rpm_roll10s_mean` perché: (a) non noti in anticipo
nel caso d'uso reale, (b) quasi-circolari col target (il MAF è ~ RPM × carico). Includerli
dà R² più alto ma fa imparare una scorciatoia e azzera il ruolo della strada.

## Come si esegue
- Interprete: `./.venv/Scripts/python.exe` (ambiente del progetto).
- I notebook: eseguirli nel kernel del `.venv`. NB2 con Optuna è lento (~15–30 min).
- ⚠️ Possibile **mismatch versione scikit-learn** tra l'ambiente che ha salvato i
  `.joblib` e il `.venv`: rieseguire/salvare nello stesso ambiente (vedi STATE.md).

## Fatti sui dati (verificati)
- `content/`: 54 parquet, ~18,26M righe, schema verificato. Area: **Ann Arbor (MI)**.
- `Datetime` è costante per trip; il tempo intra-trip è `Timestampms`. Esiste `log_MAF`.
- Terreno **dolce**: la pendenza ha segnale limitato (+ risoluzione SRTM ~30 m). È un
  limite del *dato*, non del metodo — buon spunto per "sviluppi futuri".

## Trappole note (non reintrodurle)
- **Anticipazione (look-ahead) NB2**: le feature `next_*` si costruiscono con
  `groupby(['VehId','Trip']).shift(-1)` (segmento successivo), mai guardando indietro. Deve
  entrare il futuro di strada/velocità, MAI il MAF futuro (target).
- **Cache elevation**: `build_elevation_cache.py` è resumabile (salva dopo ogni batch).
  La cella 18 del NB1 scarica solo i punti mancanti e solleva se la cache resta incompleta.
  Open-Meteo è **gratuita, no API key**; l'unico vincolo è il rate-limit 429 (già gestito).
- **Controfattuale NB2**: scalare in modo coerente tutte le `SPEED_COLS` (`speed_mean/max/min/std`,
  `entry_speed`, `accel_abs_mean`); le feature di strada/`next_*` restano fisse.

## File di contesto (in `project context/`, leggerli secondo necessità)
- `STATE.md` = stato vivo della pipeline + prossimi passi (**leggere per primo**).
- `RELAZIONE_PROGETTO.md` = panoramica per l'esame (dataset, criticità, scelte) con numeri reali.
- `ANALISI_DATI_VED.md` = EDA completa del dataset (numeri reali: schema, distribuzioni, correlazioni, powertrain).
- `discussioni.md` = registro Q&A per il ripasso d'esame (domande di Alex + risposte sintetiche).
- `notebook_studio_ML_1.md` = ripasso teorico generico di ML (PCA/t-SNE/UMAP, ecc.), non specifico del progetto.
- `README.md` (in root) = setup utente, aggiornato al reframe e ai 3 notebook.

> **Nota (2026-06-17):** sono stati rimossi da `project context/` i file `FAQ_DATI_E_MODELLO.md`,
> `FILE_DI_OUTPUT.md`, `GUIDA_CELLE_NB02_NB03.md`, `DISCUSSIONI_E_SVILUPPI.md` (scelta di Alex): se
> li vedi citati in vecchi documenti, sono puntatori morti.

## Convenzioni
- Documentazione di stato in italiano.
- Prima di lanciare chiamate API o azioni lunghe/pesanti (Optuna, fit grossi) o
  distruttive, **chiedere ad Alex**.
