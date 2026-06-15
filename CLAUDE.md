# CLAUDE.md — Onboarding per nuove sessioni

> File caricato automaticamente da Claude Code a inizio sessione. Tenuto **conciso di
> proposito**: contiene fatti stabili + puntatori. Lo **stato corrente** (sempre
> mutevole) sta in `STATE.md`: **leggilo per primo**, e aggiornalo quando fai qualcosa.

## Cos'è il progetto
ML supervised + unsupervised sul **Vehicle Energy Dataset (VED)** per dimostrare tre capacità
utili a un **assistente di guida / cruise control adattivo**: (1) stimare il **consumo** in
funzione del profilo di velocità *anticipato* (eco-driving), (2) **riconoscere il contesto
stradale** (tipologie di tratto), (3) **rilevare anomalie** di funzionamento (diagnostica).
Progetto d'esame di Alex.

> **Inquadramento (rivisto 2026-06-14).** L'idea iniziale era un ACC che sfrutta l'**orografia**.
> Esplorando i dati la pendenza si è rivelata **debole** (Ann Arbor piatta + quantizzazione
> elevation a ~111 m: slope nullo nel 92% delle righe, corr con MAF 0,004). Il progetto è stato
> quindi **reinquadrato**: il segnale predittivo reale è il **profilo di velocità/traffico**
> anticipato, non le salite; il terreno resta documentato come *limite del dato*. I dataset
> alternativi valutati (OBD-II senza GPS, IMU) non risolvevano il problema → si resta su VED.
> Storia e numeri in `RELAZIONE_PROGETTO.md`.

## I 4 notebook (consolidati 2026-06-14)
1. `01_data_prep_and_enrichment.ipynb` — carica i parquet VED, EDA, arricchisce con altitudine
   (Open-Meteo) e calcola pendenza + accelerazione. Output: `outputs/ved_enriched.parquet`.
2. `02_consumption_ecodriving.ipynb` — **consumo / eco-driving**: predice `maf_per_km` su segmenti
   ~250 m, **solo ICE** (MAF è proxy valido solo per i termici), feature map-only + anticipazione.
   Assorbe la lezione del vecchio target istantaneo (tautologico). Esclusi RPM/Load e EngineType.
3. `03_unsupervised_context_and_styles.ipynb` — **contesto + stili di guida**: Parte A clustering
   dei tratti stradali (K-Means/PCA/t-SNE/mappa); Parte B clustering dei **guidatori** (cinematica)
   con confronto **stile × powertrain** (chi-quadro) + confronto energetico ICE/HEV/PHEV.
4. `04_autoencoder_diagnostics.ipynb` — **diagnostica**: autoencoder (Keras/PyTorch) per anomaly
   detection sui fuel trim/sensori; include `EngineType` (il motore-spento-in-marcia degli ibridi
   è normale, non anomalia). Confronto con Isolation Forest.

> Storia: si è partiti da 5 notebook (NB2 MAF istantaneo + NB4 segmento + NB5 autoencoder); il NB2
> istantaneo è stato eliminato (ridondante), il segmento è diventato NB2 (solo-ICE), gli stili di
> guida sono confluiti in NB3. **I notebook nuovi non sono ancora stati eseguiti** (lo fa Alex).

## La decisione che definisce il progetto: modello "MAP-ONLY"
Il NB2 usa **solo feature note in anticipo a un ACC** (velocità, accelerazione, pendenza
attuale/futura, velocità futura, contesto). Sono **esclusi di proposito** i segnali-motore
`Engine_RPM_RPM`, `Absolute_Load_pct`, `rpm_roll10s_mean` perché: (a) non noti in anticipo
nel caso d'uso reale, (b) quasi-circolari col target (il MAF è ~ RPM × carico). Includerli
dà R² più alto ma fa imparare una scorciatoia e azzera il ruolo della strada.
→ **Storia completa con i numeri in `STORIA_PROGETTO.md`** (documento per il prof).

## Come si esegue
- Interprete: `./.venv/Scripts/python.exe` (ambiente del progetto).
- I notebook: eseguirli nel kernel del `.venv`. NB2 con Optuna è lento (~15–30 min).
- ⚠️ Possibile **mismatch versione scikit-learn** tra l'ambiente che ha salvato i
  `.joblib` e il `.venv`: rieseguire/salvare nello stesso ambiente (vedi STATE.md).

## Fatti sui dati (verificati — vedi memoria `ved-schema-facts`)
- `content/`: 54 parquet, ~18,26M righe, schema verificato. Area: **Ann Arbor (MI)**.
- `Datetime` è costante per trip; il tempo intra-trip è `Timestampms`. Esiste `log_MAF`.
- Terreno **dolce**: la pendenza ha segnale limitato (+ risoluzione SRTM ~30 m). È un
  limite del *dato*, non del metodo — buon spunto per "sviluppi futuri".

## Trappole note (non reintrodurle)
- **Feature look-ahead**: usare `FixedForwardWindowIndexer` + `shift(-1)`, NON il rolling
  trailing di default (bug storico, vedi memoria `ved-lookahead-bug`).
- **Cache elevation**: `build_elevation_cache.py` è resumabile (salva dopo ogni batch).
  La cella 18 del NB1 scarica solo i punti mancanti e solleva se la cache resta incompleta.
  Open-Meteo è **gratuita, no API key**; l'unico vincolo è il rate-limit 429 (già gestito).
- **Controfattuale NB2**: scalare *tutte* le feature di velocità in modo coerente e
  ricalcolare `speed_delta_future_5`; le feature di strada restano fisse.

## File di contesto (in `project context/`, leggerli secondo necessità)
- `STATE.md` = stato vivo della pipeline + prossimi passi (**leggere per primo**).
- `RELAZIONE_PROGETTO.md` = panoramica per l'esame (dataset, criticità, scelte) con numeri reali.
- `ANALISI_DATI_VED.md` = EDA completa del dataset (numeri reali: schema, distribuzioni, correlazioni, powertrain).
- `FAQ_DATI_E_MODELLO.md` = spiegazioni semplici (cos'è MAF/slope/fuel trim, map-only, ibridi, skew…) per lo studio.
- `FILE_DI_OUTPUT.md` = cosa contiene ogni file in `outputs/` e quale notebook lo produce.
- `GUIDA_CELLE_NB02_NB04.md` = spiegazione cella-per-cella di consumo (NB2) e autoencoder (NB4).
- `STORIA_PROGETTO.md` = come si è arrivati al modello map-only (numeri con/senza motore).
- `DISCUSSIONI_E_SVILUPPI.md` = architettura concettuale, perché NON integrare NB3→NB2 naïve
  (leakage), idee di sviluppo futuro.
- `README.md` (in root) = setup utente, aggiornato al reframe e ai 4 notebook.

## Convenzioni
- Documentazione di stato in italiano.
- Prima di lanciare chiamate API o azioni lunghe/pesanti (Optuna, fit grossi) o
  distruttive, **chiedere ad Alex**.
