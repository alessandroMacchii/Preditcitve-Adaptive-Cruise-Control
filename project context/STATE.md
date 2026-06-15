# STATE.md — Stato corrente del progetto

> Stato vivo della pipeline: **leggere per primo**, aggiornare a ogni modifica rilevante.
> **Ultimo aggiornamento:** 2026-06-14 (consolidamento a 4 notebook + allineamento di tutti i doc).

## Inquadramento attuale
**"ML per l'energia e il contesto di guida"** sul VED, applicato a un assistente di guida / ACC.
Tre pilastri: **consumo/eco-driving** · **contesto + stili di guida** · **diagnostica**. Il terreno
(idea iniziale "ACC orografico") si è rivelato un **segnale debole** → reinquadrato sul **profilo di
velocità**; terreno = limite del dato. Quadro completo: `RELAZIONE_PROGETTO.md`; dati:
`ANALISI_DATI_VED.md`; concetti: `FAQ_DATI_E_MODELLO.md`.

## Pipeline — i 4 notebook
```
[OK]  01_data_prep_and_enrichment.ipynb        eseguito → ved_enriched.parquet (~17,9M righe)
[DA ESEGUIRE] 02_consumption_ecodriving.ipynb   consumo maf_per_km a segmento, SOLO ICE, solo XGBoost
[DA ESEGUIRE] 03_unsupervised_context_and_styles.ipynb  A) tratti stradali  B) stili×powertrain
[DA ESEGUIRE] 04_autoencoder_diagnostics.ipynb  autoencoder (Keras/PyTorch) + IsolationForest
```
I tre notebook nuovi/modificati sono **validati strutturalmente (nbformat) ma NON eseguiti**:
**li esegue Alex**. Ordine: 02 → 03 → 04 (01 è già fatto).

`outputs/` ora contiene solo `ved_enriched.parquet` + `elevation_cache.parquet` (il resto è stato
ripulito; verrà rigenerato eseguendo i notebook).

## Ambiente
- Interprete: `./.venv/Scripts/python.exe`.
- Installati `torch 2.12.0+cpu` e `keras 3.14.1` (backend torch verificato) per il NB4.
- GPU (GTX 1660 Super): **non conviene** — modelli/dati piccoli, gran parte è sklearn (CPU). Resta CPU.

## Storico delle decisioni (condensato)
1. **Reframe (2026-06-14):** da "ACC che sfrutta l'orografia" a "energia + contesto". Causa: slope
   nullo nel 92% delle righe, corr con MAF 0,004 (Ann Arbor piatta + quantizzazione elevation 111 m).
2. **Dataset:** restiamo su **VED**. Valutati e scartati un OBD-II "allcars" (senza GPS) e un IMU di
   classificazione (4 classi, ~1100 righe, file duplicati): nessuno risolveva il terreno.
3. **Consumo solo-ICE:** il MAF è proxy valido solo per i termici (HEV/PHEV vanno in elettrico
   ~21–24% del moto → MAF=0; il VED non ha segnali batteria, verificato sullo schema grezzo).
4. **Consolidamento a 4 notebook:** eliminato il vecchio NB2 "MAF istantaneo" (ridondante); il
   modello a segmento è diventato il NB2; gli stili di guida sono confluiti nel NB3 (Parte B);
   l'autoencoder è il NB4 (con `EngineType` one-hot).
5. **NB2 snellito:** tenuti **solo XGBoost (default + tunato)**; rimossi baseline/Ridge/Lasso/RF
   (scelta di Alex; XGBoost regge sul *prior* tabellari→boosting). **Conseguenza:** la
   regolarizzazione L1/L2 e Random Forest non sono più nel progetto.

## Cosa verificare dopo l'esecuzione
- **NB2:** la feature importance — atteso che dominino `stop_fraction` e velocità, terreno ~0,06
  (limite dato); la tabella di sensibilità a SEG_LEN; il controfattuale (può dare "eco +X%"
  controintuitivo: è l'entanglement velocità↔stop-and-go, non un bug → la prova forte è la feature
  importance).
- **NB3 Parte B:** stile × EngineType (chi-quadro: i powertrain sono guidati diversamente?);
  stile → consumo per gli ICE; confronto energetico (motore-spento, rigenerazione).
- **NB4:** curva di loss, distribuzione errore, accordo con Isolation Forest (Jaccard), casi estremi.
- Naming manuale dei cluster (NB3) dopo aver visto le heatmap (gli `auto_name` sono solo proposte).

## Problemi noti ancora aperti (minori)
- **NB1:** la prima riga di ogni trip ha `slope`/`accel`=0 (non NaN) per un `np.where` su NaN →
  restano ~26k righe con valori fittizi a 0. Impatto basso. (In NB2 i `dist_m`/`dt_ms` NaN di quelle
  righe sono comunque azzerati prima della segmentazione.)
- **NB3 Parte A:** le celle a 4 decimali sono ~11 m, non i "50 m" del markdown → correggere il testo
  o il binning. Con celle piccole il filtro ≥50 passaggi tiene solo le strade molto trafficate.
- **PHEV = solo 12 veicoli** (su 299): le conclusioni sui PHEV sono indicative.
- Tutti i notebook nuovi **da rieseguire** per avere output e numeri reali.

## Sviluppi futuri (citabili all'esame)
Controfattuale/ottimizzatore **a parità di tempo** (mostra che "vai piano" non è la risposta) ·
pianificatore DP "rallenta prima della salita" · SHAP sul modello di consumo · eco-routing visivo.
Dettagli in `DISCUSSIONI_E_SVILUPPI.md`.
