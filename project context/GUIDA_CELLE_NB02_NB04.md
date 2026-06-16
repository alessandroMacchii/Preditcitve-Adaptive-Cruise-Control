# Guida cella-per-cella — NB02 (consumo), NB03 (contesto/stili) e NB04 (autoencoder)

> Documento di studio per la discussione d'esame. Per ogni **cella di codice** spiega *cosa fa* e
> *perché* (le scelte). I numeri `[n]` sono l'indice reale della cella nel notebook (markdown
> inclusi). Le celle markdown sono la narrativa già nel notebook e non vengono ripetute.
>
> Allineata ai notebook eseguiti: NB2 = consumo a segmento **solo ICE**, **solo XGBoost** (default +
> tunato); NB3 = contesto stradale (Parte A) + stili di guida × powertrain (Parte B); NB4 =
> autoencoder con **EngineType**. Le righe **"Risultato reale:"** riportano i numeri usciti dai run,
> così leggendo col codice di fianco ritrovi a schermo gli stessi valori. Contesto:
> `ANALISI_DATI_VED.md` (dati), `STATE.md` (stato).

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
**Risultato reale:** default R²=0,753 (MAE 250) → tunato **R²=0,762** (MAE 246). Il tuning aggiunge
~0,01 di R²: piccolo ma reale.

## `[22]` Feature importance (terreno / anticipazione / cinematica)
**Cosa fa:** importanze dell'XGBoost finale, colorate per famiglia, con somma per famiglia.
**Perché:** è il test che giustifica il progetto. **Risultato reale:** dominano **cinematica/contesto**
(≈0,79: soprattutto `stop_fraction` e velocità); **anticipazione** (`entry_*`/`next_*`) 0,112;
**terreno** (slope/dz/climb/descent) 0,100 — il singolo `slope_mean` vale solo 0,025. Conferma il
reframe: il valore è il profilo di velocità/stop-and-go, non l'orografia. **Nota:** il terreno (~0,10)
è secondario ma **non** trascurabile come il vecchio "~0,06" — la storia "terreno debole vs cinematica"
regge, ma il numero corretto da citare è **~0,10**.

## `[24]` Sensibilità a `SEG_LEN` (`quick_score`)
**Cosa fa:** rifà segmentazione + XGBoost default per `SEG_LEN ∈ {150,250,500}`; confronta R² e peso
del terreno.
**Perché:** giustifica con i dati la scelta di 250 m (troppo corto → ci si riavvicina al MAF
istantaneo e il terreno collassa per la dedup elevation a 111 m; troppo lungo → poca granularità).
**Risultato reale:** per SEG_LEN 150/250/500 → R² 0,755/0,750/0,743 e peso terreno 0,088/0,082/0,100.
R² quasi piatto e terreno debole a **tutte** le scale → è genuinamente assente, non un artefatto di
granularità. (R² qui è col modello default, sotto allo 0,762 del tunato: normale.)

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
**Risultato reale:** la loss scende pulita da ~0,61 a ~0,28, con train e val **appaiati** → niente
overfitting.

## `[12]` Errore di ricostruzione = anomaly score
**Cosa fa:** MSE per riga (`recon_err`), soglia al **99° percentile** (top 1% = anomalie),
distribuzione (log + zoom).
**Perché:** errore alto = la rete non sa rappresentare quel punto con la struttura del normale. La
soglia 1% è una **scelta operativa**, da dichiarare.
**Risultato reale:** soglia (99° pct) = 2,411 → esattamente 2.000 anomalie (1,00%), per costruzione.

## `[14]` Contributo per-feature
**Cosa fa:** errore di ricostruzione per singola feature, media anomalie vs normali, rapporto.
**Perché:** rende l'anomaly score **interpretabile**: dice *quale* segnale (fuel trim? combinazione
RPM/carico?) rende anomalo il punto.
**Risultato reale:** in cima `Short_Term_Fuel_Trim_Bank_1` (errore medio ~8,4× più alto sui punti
anomali che sui normali) → coerente coi fuel trim estremi (fino a ~+90%).

## `[16]` Confronto con Isolation Forest
**Cosa fa:** `IsolationForest(contamination=0.01)` flagga il suo 1%; calcola intersezione e Jaccard
con le anomalie dell'AE.
**Perché:** metodo unsupervised **indipendente** → modo onesto di "validare" senza etichette.
**Risultato reale:** Jaccard **0,070** (262 punti in comune). Da leggere bene: in assoluto è **basso**
(i due metodi flaggano per lo più cose diverse), MA 262 concordi contro **~20 attesi a caso** (1%×1%×
200k) = **~13× sopra il caso**. Lettura corretta da portare all'esame: i due metodi sono
**complementari** (catturano anomalie di tipo diverso) e l'overlap non-casuale isola un **nucleo di
anomalie ad alta confidenza**. Da **non** dire: "l'accordo valida il modello" — Jaccard 0,07 non lo
sostiene.

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
- *Come validi senza etichette?* → distribuzione errore, ispezione casi, e overlap **non-casuale**
  (~13× il caso) con Isolation Forest. Attento: non è "accordo" pieno (Jaccard 0,07), ma un **nucleo
  comune robusto** + metodi complementari.
- *Perché RPM/Load qui sì e nel consumo no?* → qui si descrive a posteriori, non si predice in
  anticipo: nessuna circolarità col caso d'uso ACC.

---

# Parte 3 — `03_unsupervised_context_and_styles.ipynb`

**In una frase:** clustering **non supervisionato** in due parti — **A)** i *tratti stradali* di Ann
Arbor in tipologie (urbano / scorrevole / veloce), **B)** i *guidatori* in stili di guida, con
confronto tra i tre powertrain. Gli indici `[n]` sono la posizione della cella nel notebook.

> **Nota generale NB3:** i numeri dei cluster (0,1,2…) **non sono stabili** tra run di K-Means → fai
> sempre il naming guardando la heatmap, non l'ID. E le silhouette modeste (~0,23 tratti / ~0,29
> stili) vanno **dichiarate**, non nascoste: i cluster esistono ma sono poco separati.

## Parte A — contesto stradale (celle `[2]`–`[33]`)

## `[2]` Setup e caricamento
**Cosa fa:** import (StandardScaler, PCA, KMeans, silhouette_score), seed, carica
`ved_enriched.parquet` (17,9M righe).

## `[4]` Aggregazione spaziale → celle
**Cosa fa:** arrotonda lat/lon a 4 decimali (`lat_bin`/`lon_bin`) = griglia geografica; `groupby` per
cella e calcola le statistiche aggregate (speed mean/std, accel mean/std/abs, maf/rpm/load mean,
slope, elevation) + `stop_fraction`. **L'unità del clustering è la cella, non la riga.**
**Perché:** una "tipologia di tratto" è proprietà del luogo, non del singolo campione → si aggrega.
**Risultato reale:** **281.494 celle**, `n_passages` da 1 a 3.956.
⚠️ **Caveat:** il markdown dice "~50×50 m", ma 4 decimali di latitudine ≈ **11 m**, non 50 → le celle
sono più piccole di quanto scritto, e col filtro ≥50 passaggi restano solo le strade molto battute.
(Da correggere nel testo o nel binning.)

## `[6]` Filtro celle con pochi dati
**Cosa fa:** tiene solo le celle con `n_passages ≥ 50`; mappa scatter dei passaggi.
**Perché:** una media su 3 punti è rumore, non un comportamento di tratto.

## `[8]` Selezione feature
**Cosa fa:** 10 feature comportamentali (`speed_*`, `accel_*`, `maf/rpm/load_mean`, `slope_mean`,
`stop_fraction`); droppa i NaN (celle con 1 passaggio hanno `std` NaN).
**Perché:** escluse `n_passages` (meta-dato), `elevation` (quota **assoluta** → identificherebbe il
*luogo*, non il *tipo*), e le coordinate (stesso motivo: non vogliamo "vicinato A vs B").
**Risultato reale:** **77.325 celle**, 10 feature.

## `[10]` Standardizzazione (+ demo del fallimento)
**Cosa fa:** prima stampa la varianza relativa *senza* scaling, poi applica `StandardScaler`.
**Perché:** K-Means usa la distanza euclidea → senza scaling le feature a range ampio dominano.
**Risultato reale:** senza scaling **`rpm_mean` = 99,52%** di tutta la varianza → da solo deciderebbe i
cluster; dopo lo scaler tutte media 0 / std 1. È la prova didattica del perché serve lo scaler.

## `[12]` Scelta di K — elbow + silhouette
**Cosa fa:** K da 2 a 10; per ognuno inerzia (WCSS) e silhouette (su un sample di 10k per costo); due
grafici.
**Risultato reale:** silhouette **bassa e piatta** (k=2: 0,233; k=5: 0,232; tutte ~0,17–0,23) → i
cluster esistono ma sono **poco separati**.

## `[14]` K finale
**Cosa fa:** sceglie automaticamente il K col miglior silhouette tra 3 e 7.
**Risultato reale:** **K = 5**. ⚠️ La silhouette a 5 (0,232) ≈ quella a 2 (0,233): non è la silhouette
a "imporre" il 5, ma il compromesso gomito + interpretabilità. **All'esame dillo tu**, non far credere
che la silhouette giustifichi il 5.

## `[16]` K-Means finale
**Cosa fa:** `KMeans(K=5, n_init=50, init='k-means++')`; assegna `cluster` a ogni cella.
**Perché:** `n_init=50` riduce il rischio di minimo locale; k-means++ sceglie centroidi iniziali
distanti.
**Risultato reale:** cluster di dimensioni 29.880 / 19.872 / 13.913 / 7.848 / 5.812.

## `[18]`–`[19]` Profilo dei cluster + heatmap
**Cosa fa:** medie delle feature per cluster (`[18]`); heatmap z-score relativa (`[19]`).
**Risultato reale (leggibile a colpo d'occhio):** un cluster **veloce** (speed ~91 km/h, stop 0), uno
**urbano/incrocio** (speed ~11, stop 0,52), uno **scorrevole** (~54) e due "cittadini misti". È la
caratterizzazione che dà senso ai numeri — la sezione più importante della Parte A.

## `[21]` Naming euristico
**Cosa fa:** `auto_name()` assegna un'etichetta da stop_fraction/speed/slope e la mappa nei dati.
**Perché:** cluster numerati = inutili senza nome.
**Risultato reale:** propone C0 "Cittadino misto", C1 "Rettilineo veloce", C2 "Cittadino misto", C3
"Urbano/Incrocio", C4 "Rettilineo veloce". ⚠️ Nomi **duplicati** (due "misto", due "veloce")
→ l'euristica è grezza: **rinominali a mano** guardando la heatmap `[19]` (es. separa "scorrevole" da
"autostradale"). Il commento nel codice lo dice già.

## `[23]`–`[24]` PCA (loadings + scatter)
**Cosa fa:** PCA a 2 componenti; stampa varianza spiegata e loadings (`[23]`); scatter colorato per
cluster (`[24]`).
**Perché:** i **loadings** dicono *quali* feature pesano su PC1/PC2 (cosa separa i cluster); lo scatter
è il controllo visivo della separazione.

## `[26]`–`[27]` Mappa di Ann Arbor (statica + Folium)
**Cosa fa:** scatter geografico colorato per cluster (`[26]`, salva PNG); mappa interattiva Folium con
popup e legenda (`[27]`, salva HTML, subsample 10k punti).
**Perché:** è il **deliverable visivo** — la classificazione dei tratti diventa una mappa vera.

## `[29]` t-SNE (bonus)
**Cosa fa:** t-SNE su 5.000 punti, scatter per cluster.
**Perché:** visualizzazione **non lineare** (preserva la struttura locale). Limiti da ricordare:
stocastico, lento, **distanze tra cluster non interpretabili** quantitativamente.

## `[31]` Cluster ↔ EngineType (sanity check)
**Cosa fa:** per ogni cluster, composizione % di ICE/HEV/PHEV; barre impilate.
**Perché:** controllo — i tre powertrain dovrebbero percorrere le stesse strade (composizione
~uniforme); deviazioni marcate sarebbero interessanti.

## `[33]` Salvataggio Parte A
**Cosa fa:** salva `road_segment_clusters.parquet`, `cluster_profile.csv` e le mappe.

---

## Parte B — stili di guida × powertrain (celle `[36]`–`[48]`)

## `[36]` Profilo cinematico per guidatore
**Cosa fa:** cambia unità — `groupby('VehId')`; calcola **solo cinematica** (speed mean/std/p85, accel
abs/std, `frac_hard_accel`/`frac_hard_decel` con soglia |6|, stop_fraction); tiene i guidatori con
≥2.000 righe.
**Perché:** la cinematica è **powertrain-agnostica** (niente MAF/RPM) → confrontabile tra ICE/HEV/PHEV.
È il cardine che rende **onesto** il confronto tra i tre motori.
**Risultato reale:** **278 guidatori** (ICE 177, HEV 89, **PHEV 12**).

## `[38]` Scelta K stili
**Cosa fa:** StandardScaler + KMeans per K 2..8, elbow + silhouette; sceglie K tra 3 e 6.
**Risultato reale:** silhouette migliore a **k=3 (0,290)** → **K_STYLE = 3**. Separazione un filo
migliore della Parte A, ma sempre modesta.

## `[40]` K-Means stili + profilo + naming
**Cosa fa:** `KMeans(3)`, profilo medio per stile, heatmap z-score, naming euristico
(`Urbano stop-and-go` / `Aggressivo` / `Crociera/extraurbano` / `Moderato`).
**Risultato reale:** 3 stili con n = 91 / 131 / 56; si distinguono per velocità di crociera
(`speed_p85` ~61 / ~64 / ~85) e durezza di accel/decel.

## `[42]` PCA stili
**Cosa fa:** scatter dei 278 guidatori in 2D PCA, colorati per stile.
**Perché:** controllo visivo della separazione degli stili.

## `[44]` Stile × EngineType — chi-quadro (la domanda chiave)
**Cosa fa:** tabella di contingenza stile×EngineType (conteggi e % per colonna); `chi2_contingency`.
**Perché:** test formale — *i powertrain vengono guidati diversamente?*
**Risultato reale:** **chi² = 34,05, dof = 4, p < 0,001 → differenze significative.** ⚠️ **Caveat
forte:** **PHEV = 12 guidatori** divisi su 3 stili → le celle PHEV hanno frequenze attese **< 5**,
sotto cui l'assunzione del chi-quadro non tiene. Il p-value è affidabile per ICE vs HEV, **solo
indicativo** sui PHEV. Da dichiarare (o accorpa i PHEV / usa Fisher exact).

## `[46]` (Solo ICE) lo stile incide sul consumo?
**Cosa fa:** per i soli ICE calcola `maf_per_km` per guidatore e lo confronta tra stili (tabella +
boxplot).
**Perché:** è il **payoff eco-driving** — lo stile aggressivo/stop-and-go costa di più? (MAF valido
solo per gli ICE.)
**Risultato reale:** consumo medio per stile (ICE) nell'ordine atteso; leggi la tabella a schermo per
i g/km esatti per stile.

## `[48]` Confronto energetico tra powertrain
**Cosa fa:** (1) % di **motore spento in marcia** (`MAF<0,5` con speed>5) per powertrain; (2) **MAF
medio in decelerazione** (firma della frenata rigenerativa).
**Perché:** spiega **perché gli ibridi non entrano nel modello di consumo NB2** — il loro motore si
spegne, quindi il MAF non è il loro consumo.
**Risultato reale (da leggere a schermo):** ICE ~0% di motore spento e MAF>0 anche in decelerazione;
HEV/PHEV quota di motore spento marcata e MAF~0 in decel → trazione elettrica + rigenerazione.

### Domande d'esame su NB3
- *Perché la silhouette così bassa non è un problema?* → con dati di guida i tratti si sovrappongono;
  i cluster sono interpretabili (heatmap) anche se non nettamente separati. Si dichiara, non si nasconde.
- *Perché clusterizzi i guidatori sulla sola cinematica?* → è l'unica grandezza confrontabile tra i
  tre powertrain (il MAF è falsato per gli ibridi). Così il confronto stile×motore è onesto.
- *Il chi-quadro è valido coi PHEV?* → significativo nel complesso (p<0,001), ma con 12 PHEV le celle
  attese sono <5 → indicativo su quel powertrain.

---

## Come eseguirli
Kernel: `.venv` del progetto. NB2: Optuna leggero → pochi minuti. NB3: clustering su CPU → pochi
minuti (t-SNE è la parte più lenta). NB4: training AE su CPU → pochi minuti (Keras 3 + PyTorch già
installati). Eseguire dopo che il NB1 ha prodotto `ved_enriched.parquet`.
