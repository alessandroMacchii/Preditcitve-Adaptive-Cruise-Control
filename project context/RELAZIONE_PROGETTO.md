cc# Relazione di progetto — ML per l'energia e il contesto di guida (VED)

*(Applicazione: assistente di guida / cruise control adattivo. Inquadramento rivisto 2026-06-14:
vedi §1.)*

> Documento unico per la **discussione d'esame**. Spiega: cos'è il progetto, il dataset (con
> numeri reali esplorati), l'architettura, ogni notebook, le **criticità** e le **scelte** —
> in modo da poterlo raccontare al prof. Per il dettaglio cella-per-cella dei notebook nuovi
> vedi `GUIDA_CELLE_NB02_NB04.md`.
>
> Tutti i numeri di questa relazione vengono da un'esplorazione reale di `ved_enriched.parquet`
> (17,9M righe), non da stime.

---

## 1. L'idea in una frase

**Il progetto:** usare ML **supervised + unsupervised** su telemetria reale per dare a un
assistente di guida / ACC tre capacità — **stimare il consumo** (eco-driving), **riconoscere il
contesto stradale**, **rilevare anomalie** di funzionamento.

**Onestà subito (così non sembra "tirato").** L'idea di partenza era un ACC che sfrutta
l'**orografia**: anticipo le salite → modulo la velocità → consumo meno. Esplorando i dati,
questa ipotesi è stata **smentita**: su Ann Arbor la pendenza è quasi assente e quantizzata (§3,
corr con MAF 0,004). Il progetto è stato quindi **reinquadrato** — il segnale predittivo reale è
il **profilo di velocità/traffico** anticipato, non le salite — e il terreno resta come *limite
del dato documentato*. Questo percorso (ipotesi → smentita dai dati → correzione) **è parte del
metodo**, non un difetto: è la differenza tra "scegliere i dati per la storia" e "lasciare che i
dati raccontino la storia".

La catena concettuale di un ACC predittivo (chi fa cosa):

1. **Mappa (NB1)** → dall'altitudine ricava la **pendenza** del tratto in arrivo.
2. **Valutatore di costo (NB2)** → dato uno scenario di guida, stima il **consumo**. È un
   *misuratore*, non il guidatore.
3. **Riconoscitore di contesto + stili di guida (NB3)** → tipo di strada e profilo di guida.
4. **Diagnostica (NB4)** → rileva stati **anomali** del veicolo (manutenzione predittiva).
5. **Pianificatore** (fuori scope, futuro) → prova più profili di velocità, chiede il costo al
   valutatore, sceglie il più economico. *Questo* decide "quanto accelerare".

> Frase da usare all'esame: *"dalla mappa conosco pendenza e tipo di tratto (NB1+NB3); un
> modello stima il consumo di ogni strategia di guida (NB2); un controllore sceglie la più
> efficiente. In più, un autoencoder sorveglia lo stato di salute del veicolo (NB4)."*

---

## 2. Il dataset — Vehicle Energy Dataset (VED)

Telemetria OBD-II reale dell'Università del Michigan, area di **Ann Arbor (MI)**, raccolta in
54 file parquet settimanali. Nessuna altitudine nel grezzo → la deriviamo (NB1).

### 2.1 Numeri reali (dopo pulizia, da `ved_enriched.parquet`)

| Grandezza | Valore |
|---|---|
| Righe | **17.922.869** |
| Veicoli unici | **299** |
| Trip unici | **26.285** |
| EngineType | ICE 10,40M · HEV 5,38M · PHEV 2,14M |
| Arco temporale | 2017-11-01 → 2018-11-10 (**374 giorni**) |
| Estensione | lat 42,220–42,326 · lon −83,804…−83,674 (**~12×10 km**) |
| Distanza totale flotta | **~138.827 km** |
| Trip mediano | 536 righe, **~4,3 km** |

### 2.2 Distribuzioni chiave

- **MAF** (`MAF_g_per_sec`, il target del consumo): media 9,85, **mediana 5,58**, max 259 g/s →
  molto **asimmetrico a destra** (per questo nel grezzo esiste già `log_MAF`). Il 99° pct è 44,9.
- **Velocità**: media 41,4 km/h, mediana 44; il 5° pct è 0 (molte righe a veicolo fermo).
- **Accelerazione**: media ~0, std 3,9 km/h/s (clippata a [−15, +10]).
- **Peso** (`Generalized_Weight`): 2500–6000, mediana 3500.
- **Temperatura** (`OAT_DegC`): mediana 9 °C, range −40…+60 (gli estremi sono sospetti → candidati
  anomalia per il NB4).

### 2.3 Sampling irregolare — un vincolo del dato

Il `dt` tra campioni consecutivi non è costante: **mediana 600 ms**, p99 2400 ms, ma **max
~7.444.700 ms (~2 ore!)**. Ci sono buchi di logging enormi. Conseguenze metodologiche:
- l'accelerazione va calcolata su `dt` reale (non `diff()` semplice) — fatto nel NB1;
- per integrare il consumo sul segmento (NB4) si **cappa il dt a 2 s**, altrimenti i buchi
  sommerebbero "aria fantasma".

---

## 3. LA criticità centrale: la pendenza è quasi inutilizzabile su questo dato

È il punto più importante da saper difendere, perché è il **cuore concettuale** del progetto
("l'ACC sfrutta l'orografia") e i dati lo mettono in discussione.

### 3.1 I numeri

- Elevazione: range **solo 101 m** (223→324) su tutta la città → **terreno dolce**.
- Slope: il **92% delle righe ha |slope| < 1%**; i percentili dal 5° al 95° sono **esattamente
  0,0000**.
- **Correlazione slope ↔ MAF = 0,004** (praticamente nulla).

### 3.2 Perché lo slope è quasi sempre zero — due cause sovrapposte

1. **Ann Arbor è piatta** (range 101 m): poco dislivello reale = poco segnale.
2. **Artefatto di quantizzazione** (più sottile, da spiegare bene): per ottenere l'altitudine
   senza fare 18M chiamate API, nel NB1 le coordinate vengono **deduplicate a 3 decimali
   (~111 m)**. Tutti i punti dentro la stessa cella da 111 m ricevono **la stessa quota** →
   `dz = 0` → `slope = 0`. La pendenza diventa **nulla quasi ovunque e "a gradini"** solo ai
   confini di cella (dove poi viene anche clippata a ±0,3). Ecco perché i percentili sono 0 e il
   99° pct salta a 0,3 (il clip).

> **Come raccontarlo al prof (trasformare il limite in metodo):** *"Lo slope istantaneo è
> dominato da un artefatto di risoluzione: l'altitudine SRTM (~30 m) e la dedup a 111 m lo
> rendono nullo dentro cella e spiky al bordo. Non è un bug del codice, è il limite della
> sorgente. Per questo nel NB4 sono passato dal valore istantaneo al **dislivello cumulato sul
> segmento**, molto più robusto, e ho documentato il caso come limite del dato — su terreno
> collinare la pendenza conterebbe."*

Questa singola criticità giustifica metà delle scelte del progetto (map-only, passaggio al
segmento, controfattuale).

---

## 4. I 4 notebook — cosa fanno e le scelte chiave


### NB1 — `01_data_prep_and_enrichment.ipynb` (preparazione)
Consolida i 54 parquet, EDA, **filtro outlier** (Load 0–200%, RPM 0–8000, Speed 0–200, MAF
0–300), filtro movimento, **arricchimento altitudine** via Open-Meteo (gratuita, no key, SRTM)
con **dedup a 3 decimali** (~8,7k punti, ~88 batch, cache resumabile), **slope** via Haversine,
**accelerazione** su `dt` reale. → `ved_enriched.parquet`.
**Scelte:** dedup 3 dec (non 4: inutilizzabile *e* più fine della sorgente); cache locale; clip di
slope/accel. **Criticità note:** prima riga di ogni trip con slope/accel = 0; finestre "in campioni"
non in secondi (sampling irregolare).

### NB2 — `02_consumption_ecodriving.ipynb` (consumo / eco-driving, solo ICE)
Predice **`maf_per_km` su segmenti da ~250 m** (efficienza, non distanza banale), con feature
**map-only** + **anticipazione** (`entry_speed` + look-ahead del segmento successivo). **XGBoost
default e tunato** (Optuna + GroupKFold per VehId), **sensibilità a SEG_LEN (150/250/500)**,
controfattuale eco/sport. *(Notebook snellito: tenuti solo i due XGBoost; baseline/Ridge/Lasso/RF
rimossi.)*
**Due scelte chiave:**
1. **Segmento, non istante.** Predire il MAF *istantaneo* sarebbe quasi una tautologia fisica
   (terreno = rumore, controfattuale artificioso); il target a segmento `maf_per_km` lo risolve.
2. **Solo ICE.** Il MAF è proxy di consumo valido solo per i termici; esclusi anche RPM/Load
   (map-only) ed `EngineType` (ora costante). *(Perché non includere gli ibridi "tenendo conto del
   tempo in elettrico"? Tre motivi tecnici in §5.1.)*
**Numeri reali (run eseguito):** R² ~**0,76**; a dominare **stop_fraction** (~0,25) e velocità; il
**terreno resta debole (~0,06)** a tutte le lunghezze di segmento → conferma il limite del dato.
Output: `consumption_model.joblib`, `consumption_results.csv`.

### NB3 — `03_unsupervised_context_and_styles.ipynb` (contesto stradale + stili di guida)
**Parte A — tratti stradali:** celle spaziali, filtro ≥50 passaggi, **StandardScaler** (con
dimostrazione del fallimento senza), **Elbow + Silhouette**, **K-Means++**, heatmap, **PCA**,
**t-SNE**, mappa Folium. *(Criticità: celle a 4 dec ≈ 11 m, non 50 m.)*
**Parte B — stili di guida × powertrain (nuova):** clustering dei **guidatori** (`VehId`) sulla sola
**cinematica** (velocità/accel/soste, niente MAF → valida per tutti i powertrain). Confronto
**stile × EngineType** con **chi-quadro**; per i soli ICE, relazione **stile → consumo**; più il
**confronto energetico** (motore-spento-in-marcia: ICE 0%, HEV 21%, PHEV 24%; firma della **frenata
rigenerativa**). È qui che "si tiene conto dei tre veicoli".

### NB4 — `04_autoencoder_diagnostics.ipynb` (diagnostica)
**Autoencoder** (Keras 3 / backend PyTorch, bottleneck 3): errore di ricostruzione = **anomalia**.
Su 9 feature (velocità, accel, RPM, carico, MAF, OAT, slope, **2 fuel trim**) + **`EngineType`
one-hot** (così il motore-spento degli ibridi non è scambiato per anomalia), 200k righe. Soglia 99°
pct, contributo per-feature, confronto con **Isolation Forest**, ispezione casi, PCA.
**Perché esiste:** copre l'**autoencoder** (argomento d'esame) e dà un caso d'uso reale
(manutenzione predittiva). I **fuel trim** lo giustificano (range fino a +89,8%).

---

## 5. Le grandi decisioni metodologiche (e come difenderle)

| Decisione | Perché |
|---|---|
| **Map-only** (no RPM/Load) | quei segnali non sono noti in anticipo + quasi-circolari col MAF. R² più basso ma modello *usabile* in un ACC. |
| **Split temporale per trip** | simula il deployment (addestro sul passato, predico sul futuro); niente leakage temporale. |
| **GroupKFold per VehId** | lo stesso veicolo non in train e validation insieme → CV non ottimistica. |
| **Look-ahead = futuro di strada/velocità, mai MAF futuro** | è il *punto* del progetto (info da mappe), non leakage; il target non entra mai nelle feature. |
| **Dedup elevation a 3 decimali** | coerente con SRTM ~30 m; 4 dec era inutilizzabile e più fine della sorgente. |
| **Consumo a segmento (NB2)** | il target istantaneo sarebbe tautologico; il segmento usa il dislivello cumulato e rende sensato il controfattuale. |
| **Consumo solo-ICE** | il MAF è proxy valido solo per i termici; HEV/PHEV vanno in elettrico (MAF=0, ~21–24% del moto) e il VED non ha segnali batteria → mescolarli falserebbe il target. |
| **Powertrain a parte (NB3)** | confronto descrittivo ICE/HEV/PHEV + stili di guida (cinematica, valida per tutti) → "si tiene conto dei tre veicoli" senza forzare il MAF. |
| **Target `maf_per_km`** | misura l'efficienza, non la distanza banale. |
| **NO integrazione naïve cluster→consumo** | il cluster contiene `maf_mean`/`rpm_mean` → sarebbe mean-target-encoding mascherato (leakage). Pilastri complementari. |
| **Optuna invece di GridSearch** | TPE bayesiano: migliori soluzioni a parità di budget (costo = trial × fold × fit). |
| **StandardScaler dentro la Pipeline** | obbligatorio per Ridge/Lasso e K-Means/PCA/autoencoder; nella Pipeline evita leakage (fit solo sul train). |

### 5.1 Approfondimento: perché NON includere gli ibridi nel NB2 "tenendo conto del tempo in elettrico"

È l'obiezione più naturale: *"gli HEV/PHEV vanno in elettrico una certa % del tempo (MAF=0); non posso
semplicemente tenerne conto e predire lo stesso `maf_per_km` anche per loro?"*. L'intuizione è
corretta come **direzione**, ma sul VED non regge, per tre motivi precisi. Vale la pena saperli
spiegare perché dimostrano che la scelta "solo-ICE" è ragionata, non comoda.

**1. Il *perché* il motore è spento non è osservabile.** Per un ibrido `MAF=0` non è rumore: è la
centralina che ha deciso di andare in elettrico, e quella decisione dipende soprattutto dallo
**stato di carica della batteria (SoC)**, oltre che da richiesta di potenza, temperatura e strategia
del costruttore. **Il VED non registra alcun segnale di batteria** (SoC, corrente/tensione, potenza
del motore elettrico) — verificato sullo schema grezzo. Quindi lo stesso identico profilo di
velocità/accelerazione può avere motore acceso o spento a seconda di una **variabile latente che non
osserviamo**. Predire il MAF a segmento per un ibrido significa predire una funzione di un fattore
nascosto → **errore irriducibile alto**, e non per colpa del modello.

**2. La "frazione in elettrico" non è una feature nota in anticipo.** Se la si calcola come quota di
righe con MAF=0 dentro il segmento, la si sta ricavando **dal target stesso** → è
**leakage circolare** (uso il MAF per costruire una feature con cui predico il MAF). Per usarla
*onestamente* andrebbe **predetta** prima — il che trasforma il problema in un **modello a due stadi
(hurdle)**: Stadio 1 = P(motore acceso | feature map-only); Stadio 2 = E[MAF | motore acceso];
predizione = `P(acceso) × E[MAF | acceso]`. Concettualmente corretto, ma lo Stadio 1 è guidato
proprio dal SoC non osservato (punto 1) → soffitto di R² basso e *strutturale*.

**3. Il PHEV è il caso peggiore.** Un PHEV può percorrere tutta la prima parte del viaggio in
**elettrico puro** finché la batteria non si scarica, poi comportarsi come un HEV. Senza SoC **non si
distinguono i due regimi** dalle sole feature di guida. E nel VED i PHEV sono **12 veicoli su 299**
→ neanche abbastanza dati per imparare la transizione empiricamente.

**Conseguenza pratica.** Aggiungere semplicemente `EngineType` come feature e allenare su tutti i
powertrain fa imparare al modello solo uno **"sconto medio"** per tipo di motore, mal calibrato a
livello di segmento, e **inquina l'interpretazione eco-driving** (per gli ICE il MAF *è* il consumo;
per gli ibridi il consumo si divide tra carburante ed elettrico, che non misuriamo). Per questo il
modello di consumo resta **solo-ICE**, e gli ibridi sono trattati **descrittivamente** nel NB3 Parte B
(frazione motore-spento, frenata rigenerativa, stili di guida sulla cinematica — valida per tutti).
Il modello *hurdle* resta un buon **sviluppo futuro** se si dispone di un dataset con il SoC (vedi §7).

---

## 6. Argomenti del corso coperti

**Supervised:** Pipeline + ColumnTransformer, **XGBoost** (gradient boosting), GroupKFold, Optuna
(TPE), MAE/RMSE/MAPE/R², feature importance, data leakage & split temporale.
*(Nota: il NB2 è stato snellito a soli modelli XGBoost → la regolarizzazione L1/L2 e Random Forest
non sono più nel progetto. Reintegrabili tenendo un Lasso se serve coprirli.)*
**Unsupervised:** K-Means/K-Means++, elbow, silhouette, StandardScaler motivato, PCA con
loadings, t-SNE, **autoencoder** (NB4), anomaly detection (Isolation Forest).
→ Con l'autoencoder (NB4) l'unico grande argomento d'aula prima mancante è ora coperto. Resta
fuori solo la **CNN** (Conv2D/MaxPooling): non c'è un caso d'uso a immagini in questo progetto.

---

## 7. Limiti onesti e sviluppi futuri

**Limiti (da dichiarare prima che li chieda il prof):**
- **Terreno piatto + quantizzazione** → la pendenza ha segnale debole (limite del *dato*, non del
  metodo); il NB2 lo conferma (peso terreno ~0,06 a tutte le lunghezze di segmento). Su terreno
  collinare conterebbe.
- **PHEV = solo 12 veicoli** (su 299): le conclusioni sui PHEV sono **indicative**.
- I modelli di costo sono **statistici, non simulatori fisici**: affidabili solo vicino alla
  distribuzione osservata.
- Il **controfattuale eco/sport** (scaling uniforme ±10%) è la parte più debole: dà un "eco +4%"
  controintuitivo perché nei dati bassa velocità ↔ stop-and-go sono intrecciati. La prova forte è la
  **feature importance** (cosa guida il consumo), non quella demo.
- L'autoencoder non ha **etichette di guasto** → validazione qualitativa, niente precision/recall.
- Dati solo di Ann Arbor → non generalizzano automaticamente ad altre aree/veicoli.

**Sviluppi futuri (coerenti col progetto):**
- **Controfattuale / mini-ottimizzatore a parità di tempo**: mostra esplicitamente che la risposta
  non è "vai piano" (rende non banale l'eco-driving).
- **Pianificatore DP** "rallenta prima della salita": chiude il cerchio dell'ACC (il NB2 ha già le
  feature di accoppiamento per supportarlo).
- **Modello *hurdle* per gli ibridi** (Stadio 1: P(motore acceso); Stadio 2: E[MAF | acceso]) su un
  dataset che includa il **SoC della batteria** — l'unico modo per estendere onestamente il consumo a
  HEV/PHEV (oggi impossibile sul VED, vedi §5.1).
- **SHAP** sul modello di consumo (come, non solo quanto, contano le feature).
- **Eco-routing** visivo (collega consumo + contesto stradale).
- *(Il clustering a livello di guidatore — profili eco/aggressivo — è già stato realizzato nel NB3,
  Parte B.)*

---

## 8. Domande d'esame probabili — risposte pronte

- *"Il tuo R² è quello giusto?"* → Il modello di consumo (segmento, ICE) fa **R² ~0,76**. Col
  motore (RPM/Load) salirebbe, ma barerebbe (quasi-circolari col MAF); map-only è la scelta onesta
  e usabile in un ACC.
- *"Dov'è il valore della pendenza, se è irrilevante?"* → Su *questo* dato è debole: terreno piatto
  + quantizzazione a 111 m (slope nullo nel 92% delle righe, corr 0,004); confermato nel NB2 (peso
  terreno ~0,06 a tutte le scale). Documentato come **limite del dato**.
- *"Le feature look-ahead non sono leakage?"* → No: sono pendenza/velocità *future* (info da
  mappa/profilo pianificato), il **target MAF futuro non entra mai**.
- *"Come hai gestito i tre powertrain? Il MAF non è falsato per gli ibridi?"* → Sì, ed è il punto:
  HEV/PHEV vanno in elettrico (MAF=0) il 21–24% del moto e il VED non ha la batteria → il MAF non è
  consumo per loro. Quindi **modello di consumo solo-ICE**, e ibridi **descritti a parte** (NB3:
  motore-spento, frenata rigenerativa, stili di guida). PHEV = 12 veicoli → indicativi.
- *"Non dimostri solo che la guida economica conviene?"* → No. Il modello è un **calcolatore di
  costo**, non un consiglio: senza vincolo di tempo "minimo consumo" = "fermati". La leva vera è lo
  **stop-and-go** (feature #1), non la velocità grezza — rallentare può anzi aumentare il consumo/km.
- *"Perché non hai unito cluster (NB3) e consumo (NB2)?"* → Il cluster codifica `maf_mean`/`rpm_mean`
  → leakage (target encoding mascherato). Pilastri complementari.
- *"Perché RPM/Load nell'autoencoder (NB4) sì e nel consumo no?"* → L'AE *descrive* lo stato a
  posteriori (anomaly detection), non *predice* in anticipo: nessuna circolarità col caso d'uso ACC.
- *"L'autoencoder come lo validi senza etichette?"* → Qualitativamente: distribuzione errore,
  ispezione dei casi estremi, **accordo con Isolation Forest** (metodo indipendente).

---

## 9. Mappa dei file

| File | Cosa |
|---|---|
| `01_data_prep_and_enrichment.ipynb` | preparazione + enrichment (input di tutti) |
| `02_consumption_ecodriving.ipynb` | consumo a segmento, solo ICE |
| `03_unsupervised_context_and_styles.ipynb` | tratti stradali + stili di guida × powertrain |
| `04_autoencoder_diagnostics.ipynb` | anomaly detection (autoencoder) |
| `outputs/ved_enriched.parquet` | dataset consolidato + elevation + slope (input di tutti) |
| `RELAZIONE_PROGETTO.md` | **questo file** — panoramica per l'esame |
| `GUIDA_CELLE_NB02_NB04.md` | spiegazione cella-per-cella di consumo (NB2) e autoencoder (NB4) |
| `ANALISI_DATI_VED.md` | EDA completa del dataset (numeri reali) |
| `FAQ_DATI_E_MODELLO.md` | spiegazioni semplici dei concetti (MAF, slope, fuel trim, map-only…) |
| `DISCUSSIONI_E_SVILUPPI.md` | architettura ACC, perché no NB3→NB2 naïve, idee future |
| `STATE.md` | stato vivo della pipeline + prossimi passi |
