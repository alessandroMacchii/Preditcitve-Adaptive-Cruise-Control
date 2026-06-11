Sono Alex, studente di AI/ML del corso UFS08-UFS09 (Machine Learning Supervised + 
Unsupervised) all'ITS Angelo Rizzoli. Sto preparando un progetto d'esame e ti chiedo 
di farne un audit completo + creare file di contesto persistente per le sessioni 
future.

## CONTESTO DEL PROGETTO

**Idea narrativa:** Cruise control adattivo predittivo basato sul Vehicle Energy 
Dataset (VED) dell'Università del Michigan. Costruisco due modelli ML complementari 
che insieme dimostrano come un cruise control predittivo possa ottimizzare il 
consumo conoscendo in anticipo la strada (orografia, tipo di tratto).

**Dataset:** VED Refined, ~17M righe di telemetria OBD-II reale, 54 file parquet 
settimanali divisi in due fasi temporali (Phase 1: nov 2017-mag 2018, Phase 2: 
mag-nov 2018). Area geografica: Ann Arbor, Michigan (~12x10 km). ~250 veicoli unici 
(ICE/HEV/PHEV). 18 colonne tra cui VehId, Trip, Timestampms, Latitude_deg, 
Longitude_deg, Vehicle_Speed_km_per_h, MAF_g_per_sec, Engine_RPM_RPM, 
Absolute_Load_pct, OAT_DegC, fuel trim, EngineType, Generalized_Weight. Zero NaN. 
Sampling rate irregolare (100-1400ms).

**Struttura dei tre notebook:**

1. `01_data_prep_and_enrichment.ipynb` — Carica i 54 parquet, EDA, gestione outlier 
(Absolute_Load > 200%, RPM > 8000, Speed > 200, MAF > 300), arricchimento con 
elevation via Open-Meteo Elevation API (gratuita, no key) usando dedup spaziale a 
4 decimali (~150 chiamate batch invece di 17M), calcolo pendenza con formula 
Haversine, calcolo accelerazione gestendo sampling irregolare. Output: 
`outputs/ved_enriched.parquet`.

2. `02_supervised_maf_prediction.ipynb` — Target: MAF_g_per_sec (proxy del consumo). 
Sample stratificato 30% trip per EngineType. Feature engineering: istantanee, 
rolling passate (30s/10s), LOOK-AHEAD (slope/speed nei prossimi 5/10/30 punti — 
NON è data leakage perché un cruise control predittivo ha queste info da mappe). 
Split temporale per Trip. Pipeline + ColumnTransformer con StandardScaler + 
OneHotEncoder(drop='first'). Confronto 6 modelli: baseline mean, Ridge, Lasso, 
Random Forest, XGBoost default, XGBoost tuned con Optuna (50 trial bayesiani + 
GroupKFold). Metriche: MAE, RMSE, MAPE, R². Feature importance con colorazione 
look-ahead. Analisi controfattuale finale (eco vs sport vs reale).

3. `03_unsupervised_road_clustering.ipynb` — Aggregazione spaziale in celle ~50x50m. 
Filtro >=50 passaggi per cella. Standardizzazione con dimostrazione esplicita del 
fallimento senza scaling. Elbow + silhouette per K. K-Means con K-Means++ e 
n_init=50. Caratterizzazione cluster con heatmap z-score e naming auto-euristico. 
PCA con interpretazione loadings, t-SNE bonus. Mappa Folium interattiva di 
Ann Arbor colorata per cluster.

## ARGOMENTI DEL CORSO COPERTI

Pipeline + ColumnTransformer, regolarizzazione L1/L2 (Ridge, Lasso), Random Forest, 
XGBoost/gradient boosting, K-Fold CV (GroupKFold variante), tuning bayesiano con 
Optuna, metriche regressione (MAE/MAPE/R2/RMSE), feature importance, data leakage 
(con split temporale rigoroso), K-Means con K-Means++, elbow method, silhouette 
score, StandardScaler con motivazione, PCA con loadings, t-SNE per visualizzazione, 
riduzione dimensionalità.

Il prof ha anche introdotto in aula autoencoder (con Keras 3 + PyTorch backend) e 
reti convoluzionali (Conv2D, MaxPooling, leaky_relu) tramite un tutorial 
sull'MNIST: autoencoder denso, autoencoder convoluzionale, denoising autoencoder, 
inpainting autoencoder. Questo va oltre il programma originale del corso ML 
classico ma è materia d'esame.

## SCELTE TECNICHE CHIAVE

1. Split train/test temporale per Trip (no random)
2. GroupKFold raggruppando per VehId nel tuning
3. Feature look-ahead = pendenza/velocità FUTURE da mappe, NON consumo futuro 
   (no leakage)
4. Dedup spaziale a 4 decimali prima di chiamare Open-Meteo
5. Cache locale dell'elevation
6. Aggregazione spaziale per il clustering (celle 50x50m), NON clustering su righe
7. Niente lat/lon tra le feature del clustering (voglio tipologie, non vicinati)
8. StandardScaler obbligatorio sia per Ridge/Lasso che per K-Means/PCA
9. drop='first' nel OneHotEncoder per evitare dummy variable trap
10. n_init=50 in K-Means

## COSA TI CHIEDO DI FARE

### Fase 1 — Esplora e capisci

1. Lista tutti i file del progetto nella cartella corrente (notebook, README, 
   requirements, GUIDA_PROGETTO.md se esiste, dataset se presente).
2. Leggi i tre notebook (01, 02, 03) cella per cella. Capisci la logica del codice, 
   non solo gli output.
3. Leggi README.md, GUIDA_PROGETTO.md e requirements.txt se ci sono.
4. Controlla se esiste già una cartella `content/ved_phase_1/` e 
   `content/ved_phase_2/` con i parquet, oppure se ci sono solo gli zip.
5. Controlla se esiste già una cartella `outputs/` con file generati 
   (`ved_enriched.parquet`, ecc.) per capire a che stato di esecuzione sono.
6. Verifica che esista `.claude/settings.json` (l'ho creato io manualmente). 
   Controlla che contenga whitelist per python/pip/jupyter/filesystem base e 
   deny per rm/rmdir/sudo. Se manca qualche comando utile, segnalalo ma non 
   modificarlo senza chiedermi.

### Fase 2 — Crea file di contesto persistente

Crea nella root del progetto questi due file in modo che sessioni future di Claude 
Code abbiano subito tutto il contesto:

1. **`PROJECT_CONTEXT.md`** — Documento di onboarding per nuove sessioni. Deve 
   contenere:
   - Identità dell'utente (Alex, studente AI/ML, corso UFS08-UFS09)
   - Idea narrativa del progetto (in 2-3 paragrafi)
   - Descrizione del dataset
   - Struttura dei 3 notebook + cosa fa ciascuno
   - Scelte tecniche chiave e perché (le 10 elencate sopra)
   - Argomenti del corso coperti
   - Tono e stile preferiti dall'utente (vedi sotto)

2. **`STATE.md`** — Documento di stato corrente. Deve contenere:
   - A che punto è il progetto (notebook eseguiti, output generati, modello salvato)
   - Cosa è già stato fatto vs cosa manca
   - Eventuali bug, problemi o TODO trovati
   - Suggerimenti di miglioramento prioritari
   - Da aggiornare ogni volta che si fa qualcosa di rilevante nel progetto

### Fase 3 — Analisi critica

Dopo aver letto tutto, scrivimi un report in chat (non un file) con:

**Cose fatte bene:**
- Quali scelte sono solide e ben giustificate
- Quali parti del codice sono ben strutturate e leggibili

**Cose fatte male o migliorabili:**
- Bug, problemi logici, scelte discutibili
- Codice ridondante o inefficiente
- Documentazione/markdown mancante o poco chiara
- Argomenti del corso poco valorizzati

**Suggerimenti di miglioramento prioritizzati:**
- Miglioramenti che potrebbero essere fatti restando dentro al programma del corso
- Eventuali cose che si potrebbero aggiungere per coprire argomenti del corso 
  attualmente assenti
- Eventuali integrazioni tra i notebook (es. usare i cluster del 03 nel 02)

**Domande potenziali d'esame su cui potrei essere in difficoltà:**
- Punti del progetto su cui la motivazione è debole o assente
- Scelte che potrebbero essere attaccate dal prof
- Concetti che andrebbero spiegati meglio nei markdown

### Fase 4 — Proposta di prossimo passo

Alla fine del report, fammi una proposta concreta: cosa faresti TU se fossi nei 
miei panni e dovessi presentare questo progetto all'esame tra una settimana? 
Dammi un piano operativo, non solo un'analisi astratta.

## TONO E STILE PREFERITI

- Parlami tecnicamente, sono uno studente di ML universitario
- Risposte concise, non loquaci. Evita di ripetere tre volte la stessa cosa.
- In italiano
- Onesto: se una mia scelta nel codice è sbagliata o migliorabile, dimmelo 
  apertamente. Non assecondarmi.
- Quando modifichi file, mantieni lo stile esistente (markdown narrativo prima 
  delle celle di codice)
- Non eseguire blocchi di codice lunghi senza chiedere (es. Optuna 50 trial, 
  fit XGBoost su 17M righe)
- Ok eseguire celle veloci di esplorazione (head, ls, len, describe)

## LIMITI

- Non hai accesso ai PDF di teoria del corso e ai notebook didattici del prof 
  (ce li ho io localmente ma non in questa cartella). Se ti servono per 
  contesto specifico, chiedimi e li condivido.
- Non eseguire le chiamate Open-Meteo Elevation API senza autorizzazione 
  (sono ~150 chiamate, gratuite, ma preferisco controllare quando le fai)

Inizia con la Fase 1. Quando hai esplorato tutto, fammi un riassunto di cosa hai 
trovato e poi procedi con la Fase 2 (creazione file di contesto), Fase 3 
(analisi critica) e Fase 4 (proposta di prossimo passo). Buon lavoro.