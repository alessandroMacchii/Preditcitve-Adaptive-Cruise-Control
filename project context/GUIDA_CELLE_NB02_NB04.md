# Guida cella-per-cella — NB02 (consumo) e NB04 (autoencoder)

> Documento di studio per la discussione d'esame. Per ogni **cella di codice** spiega *cosa fa* e
> *perché* (le scelte). I numeri `[n]` sono l'indice reale della cella nel notebook (markdown
> inclusi). Le celle markdown sono la narrativa già nel notebook e non vengono ripetute.
>
> Allineata ai notebook attuali (2026-06-14): NB2 = consumo a segmento **solo ICE**, **solo XGBoost**
> (default + tunato); NB4 = autoencoder con **EngineType**. Gli **stili di guida × powertrain** sono
> nel NB3 (Parte B), spiegati in `RELAZIONE_PROGETTO.md`. Contesto: `ANALISI_DATI_VED.md` (dati),
> `STATE.md` (stato).

---

# Parte 1 — `02_consumption_ecodriving.ipynb`

**In una frase:** predice il consumo per km (`maf_per_km`) di un tratto di strada da ~250 m, usando
solo feature note in anticipo, **sui soli veicoli ICE** (dove il MAF è un proxy di consumo valido).

## `[2]` Setup e caricamento + filtro ICE
**Cosa fa:** importa le librerie, fissa `RANDOM_STATE=42`, carica `ved_enriched.parquet` e **filtra
alle righe `EngineType=='ICE'`** (~10,4M).
**Perché:** il MAF (massa d'aria ≈ carburante) è un proxy di consumo valido **solo per i termici**;
HEV/PHEV vanno in elettrico (MAF=0 mentre consumano) e il VED non registra la batteria → includerli
falserebbe il target. Restano dati abbondanti (58% del dataset).

## `[4]` Parametri e helper di riga
**Cosa fa:** `SEG_LEN=250` m, filtri qualità (`MIN_SEG_FRAC`, `MIN_POINTS`), `DT_CLIP_S=2.0`. Ordina
per trip+tempo, **azzera i NaN** di `dist_m`/`dt_ms`, calcola `air_g = MAF·dt` (con Δt cappato),
salita/discesa cumulata, `accel_abs`, `is_stop`, `hour`, `month`.
**Perché — scelte chiave:** azzerare `dist_m`/`dt_ms` (NaN sulla **prima riga di ogni trip**)
altrimenti il `cumsum` si rompe; **cappare il Δt a 2 s** per non integrare "aria fantasma" sui buchi
di sampling; il **dislivello cumulato** è più robusto dello slope istantaneo (rumoroso a 30 m SRTM).

## `[6]` Costruzione dei segmenti (`build_segments`)
**Cosa fa:** dentro ogni trip accumula la distanza e taglia un segmento ogni `SEG_LEN` m; aggrega le
righe in: lunghezza/tempo/aria del segmento, quota inizio/fine, salita/discesa, statistiche di slope
e velocità, `entry_speed`, `accel_abs_mean`, `stop_fraction`, contesto. Deriva `dz_net` e il **target
`maf_per_km` = aria totale / km**. Filtra segmenti troppo corti/con pochi punti.
**Perché:** il target **per km** misura l'efficienza, non la distanza banale. È una funzione perché
serve anche al test di sensibilità (cella `[24]`).

## `[8]` Feature di accoppiamento (`add_coupling`)
**Cosa fa:** con `shift(-1)` dentro il trip aggiunge `next_dz_net`, `next_slope_mean`,
`next_stop_fraction` (cosa c'è nel **segmento successivo**); l'ultimo segmento → 0.
**Perché:** insieme a `entry_speed` permettono l'**anticipazione** (regolarsi ora per ciò che arriva).
È l'analogo, a livello di segmento, delle feature look-ahead.

## `[10]` Target e feature (map-only)
**Cosa fa:** `TARGET='maf_per_km'`, `FEATURES_NUM` (geometria + profilo velocità + accoppiamento +
contesto), `GROUP_COL='VehId'`.
**Perché:** **map-only** — solo ciò che un ACC conosce in anticipo. Esclusi `Engine_RPM_RPM` e
`Absolute_Load_pct` (output del motore, non noti a priori, quasi-circolari col MAF: correlano 0,75 e
0,52). `EngineType` non c'è (costante = ICE).

## `[12]` Split temporale per trip
**Cosa fa:** ordina i trip per data, 80% più vecchio in train, 20% più recente in test.
**Perché:** simula il deployment (passato→futuro), niente leakage temporale; split per trip interi.

## `[14]` Pipeline (StandardScaler) + funzione `report`
**Cosa fa:** `ColumnTransformer` con `StandardScaler` sulle numeriche; `report()` stampa
MAE/RMSE/MAPE/R² e accumula in `results`.
**Perché:** lo scaler è innocuo per gli alberi ma tenuto nella Pipeline (evita leakage: fit solo sul
train). Niente OneHotEncoder: non ci sono categoriche (solo-ICE).

## `[16]` XGBoost (default)
**Cosa fa:** addestra un XGBoost con iperparametri di default; ne misura le metriche.
**Perché:** è il candidato naturale per dati tabellari (lo sa l'esperienza) → lo si usa come
riferimento *prima* del tuning, per vedere quanto aggiunge il tuning.

## `[18]` Tuning con Optuna (TPE) + GroupKFold
**Cosa fa:** `N_TRIALS=30`, `N_SPLITS=4`; cerca 8 iperparametri minimizzando il MAE in
`GroupKFold(VehId)`.
**Perché:** Optuna/TPE (bayesiano) trova soluzioni migliori a parità di budget; **GroupKFold per
VehId** perché lo stesso veicolo non stia in train e validation (i dati per veicolo sono molto
sbilanciati: una sola auto ha ~822k righe). Costo ≈ trial × fold × fit.

## `[20]` Modello finale (XGBoost tunato) + confronto
**Cosa fa:** riaddestra XGBoost coi `best_params` su tutto il train, valuta sul test; tabella +
barplot dei due modelli.
**Perché:** dopo aver scelto gli iperparametri in CV, il modello definitivo usa tutti i dati di
train; la stima onesta è sul test mai visto. Il confronto default→tunato dice se il tuning è valso.

## `[22]` Feature importance (terreno / anticipazione / cinematica)
**Cosa fa:** importanze dell'XGBoost finale, colorate per famiglia, con somma per famiglia.
**Perché:** è il test che giustifica il progetto. Risultato atteso (dal run precedente): a dominare
**stop_fraction** e velocità; **terreno debole** (~0,06). Conferma il reframe (il valore è il profilo
di velocità, non l'orografia).

## `[24]` Sensibilità a `SEG_LEN` (`quick_score`)
**Cosa fa:** rifà segmentazione + XGBoost default per `SEG_LEN ∈ {150,250,500}`; confronta R² e peso
del terreno.
**Perché:** giustifica con i dati la scelta di 250 m (troppo corto → terreno torna rumore; troppo
lungo → poca granularità). Risultato: R² ~0,75 e terreno ~0,06 a tutte le scale → terreno
genuinamente assente, non artefatto di granularità.

## `[26]`–`[27]` Controfattuale eco/sport su una route
**Cosa fa:** sceglie un trip di test con dislivello; `scale_speed_seg()` scala in modo coerente le
feature di velocità (la geometria resta fissa); predice `maf_per_km` per reale/eco(−10%)/sport(+10%)
e integra in aria totale.
**Perché:** dimostra l'uso ACC (confrontare strategie sullo stesso percorso). **Limiti dichiarati:**
scaling *uniforme* (non ottimizzazione selettiva) su un **modello statistico**, non fisica. Può dare
un "eco +X%" controintuitivo perché nei dati bassa velocità ↔ stop-and-go sono intrecciati → la prova
forte resta la feature importance, non questa demo.

## `[29]` Diagnostica e salvataggio
**Cosa fa:** scatter predetto-vs-reale e residui; salva `consumption_model.joblib`,
`consumption_results.csv`, `consumption_seglen_sensitivity.csv`.
**Perché:** i residui rivelano bias/eteroschedasticità; salvare rende il modello riusabile.

> **Nota:** baseline, Ridge, Lasso e Random Forest sono stati **rimossi** (scelta di snellimento). La
> scelta di XGBoost poggia sul *prior* "tabellari → boosting", non su un confronto con i lineari.
> Conseguenza: la regolarizzazione L1/L2 e RF non sono più nel progetto.

---

# Parte 2 — `04_autoencoder_diagnostics.ipynb`

**In una frase:** un autoencoder impara a ricostruire la telemetria "normale"; ciò che ricostruisce
male (errore alto) è un candidato **anomalia** (diagnostica / manutenzione predittiva).

## `[2]` Setup
**Cosa fa:** imposta `os.environ["KERAS_BACKEND"]="torch"` **prima** di importare keras; importa
sklearn (StandardScaler, IsolationForest, PCA, split), fissa il seed.
**Perché:** Keras 3 è multi-backend; il backend va scelto prima dell'import (stack del corso:
Keras 3 + PyTorch).

## `[4]` Dati e feature
**Cosa fa:** definisce `AE_NUM` (9 segnali: velocità, accel, RPM, carico, MAF, OAT, slope, **2 fuel
trim**), `SAMPLE_N=200_000`; carica quelle colonne + `EngineType`, droppa NaN, campiona.
**Perché:** i **fuel trim** sono il cuore della diagnostica (valori grandi/persistenti = problemi).
Qui RPM/Load sono **leciti**: descriviamo lo stato a posteriori, non prediciamo in anticipo. Si
campiona perché i dati sono **ridondanti** (200k bastano a definire il "normale").

## `[6]` Standardizzazione + one-hot EngineType + split
**Cosa fa:** `StandardScaler` sulle numeriche, `EngineType` → one-hot, concatenati in `X`; split
train/val 80/20.
**Perché:** l'AE minimizza l'MSE: senza scaling le feature a range ampio (RPM, MAF) dominerebbero.
**EngineType incluso** così l'AE impara il "normale" *per powertrain* (il motore-spento-in-marcia
degli ibridi non viene scambiato per anomalia).

## `[8]` Architettura dell'autoencoder
**Cosa fa:** rete densa a clessidra (`… → 16 → 8 → 3 bottleneck → 8 → 16 → n`), ReLU, uscita lineare,
Adam, loss MSE; stampa `summary()`.
**Perché:** il **bottleneck a 3** forza la compressione (la rete deve tenere solo la struttura
essenziale; un collo largo copierebbe l'input). Uscita lineare = ricostruiamo valori reali; MSE =
naturale per dati continui ed è anche l'anomaly score.

## `[10]` Training
**Cosa fa:** `EarlyStopping(patience=5)`, fino a 50 epoche, batch 512; plotta la curva train/val.
**Perché:** l'early stopping evita overfitting e ripristina i pesi migliori; la curva è il controllo
visivo (train e val devono scendere insieme).

## `[12]` Errore di ricostruzione = anomaly score
**Cosa fa:** MSE per riga (`recon_err`), soglia al **99° percentile** (top 1% = anomalie),
distribuzione (log + zoom).
**Perché:** errore alto = la rete non sa rappresentare quel punto con la struttura del normale. La
soglia 1% è una **scelta operativa**, da dichiarare.

## `[14]` Contributo per-feature
**Cosa fa:** errore di ricostruzione per singola feature, media anomalie vs normali, rapporto.
**Perché:** rende l'anomaly score **interpretabile**: dice *quale* segnale (fuel trim? combinazione
RPM/carico?) rende anomalo il punto.

## `[16]` Confronto con Isolation Forest
**Cosa fa:** `IsolationForest(contamination=0.01)`, accordo con l'AE (intersezione, Jaccard).
**Perché:** metodo unsupervised **indipendente**; se concorda coi casi estremi, la fiducia aumenta —
è il modo onesto di "validare" senza etichette.

## `[18]` Ispezione dei casi più anomali
**Cosa fa:** i 10 punti con errore massimo, valori reali + EngineType, vs la mediana dei normali.
**Perché:** trasforma uno "score" in un'osservazione verificabile a occhio.

## `[20]` Visualizzazione PCA
**Cosa fa:** proietta in 2D (PCA) e colora normali vs anomalie.
**Perché:** verifica visiva che gli anomali stiano ai margini della nuvola.

## `[22]` Salvataggio
**Cosa fa:** salva `telemetry_autoencoder.keras` e `anomaly_scores.parquet`.
**Perché:** rende riusabile il rilevatore e documenta gli score.

### Domande d'esame su NB4
- *Perché un autoencoder e non una soglia?* → coglie anomalie **multivariate** (combinazioni), non
  solo valori singoli fuori scala.
- *A cosa serve il bottleneck?* → forza la compressione; senza, la rete copierebbe l'input.
- *Come validi senza etichette?* → distribuzione errore, ispezione casi, accordo con Isolation Forest.
- *Perché RPM/Load qui sì e nel consumo no?* → qui si descrive a posteriori, non si predice in
  anticipo: nessuna circolarità col caso d'uso ACC.

---

## Come eseguirli
Kernel: `.venv` del progetto. NB2: Optuna leggero → pochi minuti. NB4: training AE su CPU → pochi
minuti (Keras 3 + PyTorch già installati). Eseguire dopo che il NB1 ha prodotto `ved_enriched.parquet`.
