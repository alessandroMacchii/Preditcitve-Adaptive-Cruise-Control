# STATE.md — Stato corrente del progetto

> Stato vivo della pipeline: **leggere per primo**, aggiornare a ogni modifica rilevante.
> **Ultimo aggiornamento:** 2026-06-17 (sessione lunga). NB3 **sfoltito** (rimossi t-SNE,
> cluster↔EngineType, chi-quadro stile×powertrain, stile→consumo, confronto energetico; import consolidati,
> PCA ripristinata); **fix heatmap** (z-score dalle medie non arrotondate, `slope_mean` non spariva più +
> riga `maf_mean_descr`); **salvataggio NB3 snellito** (solo `cluster_profile.csv` + `cluster_map.html`);
> tabelle colonne in `ANALISI_DATI_VED.md` (§2.1/2.2/2.3); voci #30–#39 in `discussioni.md`. **Esplorata e
> ANNULLATA** la griglia quota a 30 m (vedi `discussioni.md` #36). In precedenza: NB4 rimosso → **3 notebook**.
>
> ⚠️ **Lavoro NON committato (per la prossima sessione):** `03_..ipynb` (fix heatmap + salvataggio) e i
> file di contesto `ANALISI_DATI_VED.md`, `STATE.md`, `discussioni.md`. NB1 è pulito (= HEAD). Vanno
> **rieseguiti i fix del NB3** e committato il tutto.

## Inquadramento attuale
**"ML per l'energia e il contesto di guida"** sul VED, applicato a un assistente di guida / ACC.
Due pilastri: **consumo/eco-driving** · **contesto + stili di guida**. Il terreno
(idea iniziale "ACC orografico") si è rivelato un **segnale debole** → reinquadrato sul **profilo di
velocità**; terreno = limite del dato. Quadro completo: `RELAZIONE_PROGETTO.md`; dati:
`ANALISI_DATI_VED.md`; ripasso Q&A: `discussioni.md`.

## Pipeline — i 3 notebook
```
[OK, committato]     01_data_prep_and_enrichment.ipynb        eseguito → ved_enriched.parquet (~17,9M righe)
[DA VERIFICARE]      02_consumption_ecodriving.ipynb          consumo maf_per_km a segmento, SOLO ICE, solo XGBoost
[RUN OK, FIX DA RIESEGUIRE] 03_unsupervised_context_and_styles.ipynb  A) tratti  B) stili (sfoltito 17/06)
```
**Stato (17/06, fine sessione):**
- **NB1** rieseguito e **committato** (ha gli output, incl. grafico slope). Pulito = HEAD.
- **NB3** rieseguito oggi a **K=4** (cluster: incrocio/urbano-misto/scorrevole/autostrada; output reali nel
  notebook committato). **MA** ci sono **fix non committati** (heatmap z-score + salvataggio snellito) →
  **rieseguire** quelle celle e **ricommittare**.
- **NB2:** verificare che sia stato rieseguito col codice attuale (solo-XGBoost). Se in dubbio, rieseguirlo.
- I notebook **li esegue Alex** nel kernel del `.venv`.

`outputs/` — file utili:
```
ved_enriched.parquet           (input di tutti, dalla run NB1)   elevation_cache.parquet
consumption_model.joblib       consumption_results.csv          consumption_seglen_sensitivity.csv   [NB2]
cluster_profile.csv            cluster_map.html                                                       [NB3]
```
> **NB3 salva solo l'essenziale (scelta di Alex):** `cluster_profile.csv` (profilo+nome dei cluster) e
> `cluster_map.html` (mappa interattiva). **Non** salva più `road_segment_clusters.parquet` né
> `cluster_map_static.png`. Orfani da cancellare se ancora su disco: quei due + `anomaly_scores.parquet`
> e `telemetry_autoencoder.keras` (residui del NB4 rimosso).

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
- **NB3 Parte B:** profilo stili sulla cinematica + heatmap + PCA (K_STYLE=3). *(Il test chi-quadro
  stile×powertrain, stile→consumo ICE e il confronto energetico sono stati rimossi il 17/06 — sviluppi.)*
- Naming manuale dei cluster (NB3) dopo aver visto le heatmap (`name_cluster` assegna dai valori, è solo
  una proposta da confermare). **Nota:** in NB3 `K_FINAL` è forzato a **4** (riga finale cella `[14]`).

## Problemi noti ancora aperti (minori)
- **NB1:** la prima riga di ogni trip ha `slope`/`accel`=0 (non NaN) per un `np.where` su NaN →
  restano ~26k righe con valori fittizi a 0. Impatto basso. (In NB2 i `dist_m`/`dt_ms` NaN di quelle
  righe sono comunque azzerati prima della segmentazione.)
- ~~NB3 celle "50 m"~~ **risolto (17/06):** il markdown ora dice ~11×8 m (4 decimali). Con celle così
  piccole il filtro ≥50 passaggi tiene solo le strade molto trafficate.
- **PHEV = solo 12 veicoli** (su 299): le conclusioni sui PHEV sono indicative.
- **NB3 sfoltito (17/06):** rimossi t-SNE, cluster↔EngineType, chi-quadro stile×powertrain, stile→consumo,
  confronto energetico; import consolidati in `[2]`; ripristinata la cella PCA (era orfana dopo le
  cancellazioni); markdown/intro/riepiloghi riallineati. **Restano due celle di riepilogo redondanti**
  (`[38]` e `[39]`): Alex può accorparle.
- **Griglia quota a 30 m: esplorata e ANNULLATA (17/06).** Idea: campionare la quota a 30 m (risoluzione
  nativa SRTM) invece dei 111 m attuali. Tutto ripristinato a `ROUND_DECIMALS=3` (NB1 + `build_elevation_cache.py`).
  Motivo: ~2h per il rate-limit API + guadagno marginale su città piatta. **Non rifarlo prima dell'esame**;
  resta sviluppo futuro. Dettagli e numeri in `discussioni.md` #36.
- **Fix heatmap NB3 (17/06, da committare):** lo z-score si calcola dalle medie **non arrotondate**
  (`.round(2)` azzerava `slope_mean` → riga NaN); aggiunta riga descrittiva `maf_mean_descr`. Vedi #39.
- **NB3 da rieseguire** per applicare i fix (heatmap + salvataggio) e poi committare. NB2 da verificare.

## Sviluppi futuri (citabili all'esame)
Controfattuale/ottimizzatore **a parità di tempo** (mostra che "vai piano" non è la risposta) ·
pianificatore DP "rallenta prima della salita" · SHAP sul modello di consumo · eco-routing visivo ·
i blocchi NB3 rimossi (chi-quadro stile×powertrain, confronto energetico, t-SNE).
