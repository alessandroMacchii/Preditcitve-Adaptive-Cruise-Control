# STATE.md — Stato corrente del progetto

> Stato vivo della pipeline: **leggere per primo**, aggiornare a ogni modifica rilevante.
> **Ultimo aggiornamento:** 2026-06-17 (rimosso il NB4 autoencoder/diagnostica dal progetto su scelta di
> Alex: il progetto resta a **3 notebook**; cancellati notebook + output NB4 e ripuliti i riferimenti).

## Inquadramento attuale
**"ML per l'energia e il contesto di guida"** sul VED, applicato a un assistente di guida / ACC.
Due pilastri: **consumo/eco-driving** · **contesto + stili di guida**. Il terreno
(idea iniziale "ACC orografico") si è rivelato un **segnale debole** → reinquadrato sul **profilo di
velocità**; terreno = limite del dato. Quadro completo: `RELAZIONE_PROGETTO.md`; dati:
`ANALISI_DATI_VED.md`; concetti: `FAQ_DATI_E_MODELLO.md`.

## Pipeline — i 3 notebook
```
[OK]            01_data_prep_and_enrichment.ipynb       eseguito → ved_enriched.parquet (~17,9M righe)
[OUTPUT STALE]  02_consumption_ecodriving.ipynb         consumo maf_per_km a segmento, SOLO ICE, solo XGBoost
[OUTPUT STALE]  03_unsupervised_context_and_styles.ipynb  A) tratti stradali  B) stili×powertrain
```
⚠️ **Stato output disallineato.** I due notebook NB2/NB3 hanno output già presenti in `outputs/` (modello
consumo, cluster…), ma sono di una **run del 14/06**, mentre i notebook sono stati **rimodificati e
committati il 16/06** (commit `ce1ab41 "edits"`) *dopo* quella run. Quindi gli output esistono ma
**non riflettono il codice attuale** → vanno **rigenerati** rieseguendo i notebook nell'ordine
02 → 03 (01 è già fatto). I notebook sono validati strutturalmente (nbformat); **li esegue Alex**.

`outputs/` contiene attualmente (run 14/06):
```
ved_enriched.parquet           (input di tutti, dalla run NB1)   elevation_cache.parquet
consumption_model.joblib       consumption_results.csv          consumption_seglen_sensitivity.csv   [NB2]
road_segment_clusters.parquet  cluster_profile.csv  cluster_map.html  cluster_map_static.png          [NB3]
```

## Ambiente
- Interprete: `./.venv/Scripts/python.exe`.
- GPU (GTX 1660 Super): **non conviene** — modelli/dati piccoli, gran parte è sklearn (CPU). Resta CPU.
- (`torch`/`keras` erano installati solo per il NB4, ora rimosso: non più necessari.)

## Storico delle decisioni (condensato)
1. **Reframe (2026-06-14):** da "ACC che sfrutta l'orografia" a "energia + contesto". Causa: slope
   nullo nel 92% delle righe, corr con MAF 0,004 (Ann Arbor piatta + quantizzazione elevation 111 m).
2. **Dataset:** restiamo su **VED**. Valutati e scartati un OBD-II "allcars" (senza GPS) e un IMU di
   classificazione (4 classi, ~1100 righe, file duplicati): nessuno risolveva il terreno.
3. **Consumo solo-ICE:** il MAF è proxy valido solo per i termici (HEV/PHEV vanno in elettrico
   ~21–24% del moto → MAF=0; il VED non ha segnali batteria, verificato sullo schema grezzo).
4. **Struttura a 3 notebook:** NB2 = consumo a segmento (solo-ICE); NB3 = contesto stradale +
   stili di guida (Parte B). *(Il NB4 autoencoder/diagnostica è stato rimosso il 17/06, scelta di
   Alex: ortogonale al cuore eco-driving del progetto. Conseguenza: deep learning/anomaly detection
   non più coperti dal progetto — restano tra gli argomenti studiati, non applicati.)*
5. **NB2 snellito:** tenuti **solo XGBoost (default + tunato)**; rimossi baseline/Ridge/Lasso/RF
   (scelta di Alex; XGBoost regge sul *prior* tabellari→boosting). **Conseguenza:** la
   regolarizzazione L1/L2 e Random Forest non sono più nel progetto.
6. **Ibridi nel NB2 (2026-06-15):** documentato *perché* non si estende il consumo agli HEV/PHEV
   "tenendo conto del tempo in elettrico" — il SoC non è nel VED → la frazione-elettrica è una
   variabile latente non osservata; usarla dal target = leakage; serve un modello *hurdle* con SoC
   (sviluppo futuro). Scritto in `RELAZIONE_PROGETTO.md` §5.1 e nel markdown §1 del NB2. Corretta
   anche una riga stale nell'intro NB2 (elencava ancora Ridge/Lasso/RF).

## Cosa verificare dopo l'esecuzione
- **NB2:** la feature importance — atteso che dominino `stop_fraction` e velocità, terreno ~0,06
  (limite dato); la tabella di sensibilità a SEG_LEN; il controfattuale (può dare "eco +X%"
  controintuitivo: è l'entanglement velocità↔stop-and-go, non un bug → la prova forte è la feature
  importance).
- **NB3 Parte B:** stile × EngineType (chi-quadro: i powertrain sono guidati diversamente?);
  stile → consumo per gli ICE; confronto energetico (motore-spento, rigenerazione).
- Naming manuale dei cluster (NB3) dopo aver visto le heatmap (gli `auto_name` sono solo proposte).

## Problemi noti ancora aperti (minori)
- **NB1:** la prima riga di ogni trip ha `slope`/`accel`=0 (non NaN) per un `np.where` su NaN →
  restano ~26k righe con valori fittizi a 0. Impatto basso. (In NB2 i `dist_m`/`dt_ms` NaN di quelle
  righe sono comunque azzerati prima della segmentazione.)
- **NB3 Parte A:** le celle a 4 decimali sono ~11 m, non i "50 m" del markdown → correggere il testo
  o il binning. Con celle piccole il filtro ≥50 passaggi tiene solo le strade molto trafficate.
- **PHEV = solo 12 veicoli** (su 299): le conclusioni sui PHEV sono indicative.
- I notebook NB2/NB3 **da rieseguire** per allineare gli output al codice del 16/06 (gli output
  in `outputs/` sono di una run precedente — vedi nota "Stato output disallineato" sopra).

## Sviluppi futuri (citabili all'esame)
Controfattuale/ottimizzatore **a parità di tempo** (mostra che "vai piano" non è la risposta) ·
pianificatore DP "rallenta prima della salita" · SHAP sul modello di consumo · eco-routing visivo.
Dettagli in `DISCUSSIONI_E_SVILUPPI.md`.
