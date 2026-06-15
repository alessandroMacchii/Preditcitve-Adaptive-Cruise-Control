# Discussioni, decisioni di design e sviluppi futuri

> **Nota (aggiornata 2026-06-14):** l'architettura concettuale qui descritta è ancora valida, ma il
> progetto è stato reinquadrato da "ACC orografico" a *"ML per l'energia e il contesto di guida"*
> (terreno = limite del dato) e **consolidato in 4 notebook**: 01 prep · 02 consumo a segmento
> (solo-ICE, solo XGBoost) · 03 contesto stradale + stili di guida × powertrain · 04 autoencoder.
> Dove sotto si legge "NB2 istantaneo / NB4 / NB5", vanno reinterpretati con questa struttura nuova.
> Quadro aggiornato in `RELAZIONE_PROGETTO.md` e `STATE.md`.

> Raccolta dei ragionamenti fatti con Claude (sessione 2026-06-13), così una nuova
> sessione non riparte da zero. Per lo **stato operativo** vedi `STATE.md`; per la
> **storia del modello map-only** vedi `STORIA_PROGETTO.md`.

---

## 1. Architettura concettuale (chi fa cosa)

Errore mentale comune già chiarito: **NB2 non "decide quanto accelerare"** e **la pendenza
non viene dal NB3**. La catena reale di un ACC predittivo:

1. **NB1 (mappa)** → dall'altitudine ricava la **pendenza** del tratto in arrivo. È già una
   feature di NB2 (`slope`, `slope_future_5/10/30`). La pendenza viene da qui, NON dal NB3.
2. **NB2 (valutatore di costo)** → dato uno scenario di guida (velocità + pendenza + look-ahead),
   stima il **MAF** = sforzo/consumo *in quell'istante*. È un **misuratore**, non il guidatore.
3. **Pianificatore/controllore (fuori scope, fatto a mano nella demo)** → propone più profili di
   velocità, chiede a NB2 il costo di ciascuno, sceglie il più economico. *Questo* determina
   "quanto accelerare". Nel progetto è l'esperimento controfattuale eco/sport.
4. **NB3 (contesto)** → classifica le strade per **tipologia** (urbano, rettilineo veloce,
   salita…). Fornisce contesto categorico, NON la pendenza.

Frase corretta da usare all'esame: *"dalla mappa conosco pendenza e tipo di tratto in arrivo
(NB1 + NB3); NB2 stima il consumo di ogni possibile strategia di guida; un controllore sceglie
la più efficiente."* I modelli sono i mattoni; chi decide è il controllore che li usa.

## 2. Cosa fa il Notebook 3 (riassunto)

Unsupervised, **non predice nulla**: raggruppa i tratti stradali per tipologia.
Griglia spaziale → statistiche aggregate per cella (velocità, accel, MAF, RPM, pendenza,
frazione soste) → filtro celle con <50 passaggi → StandardScaler → scelta K (elbow+silhouette)
→ K-Means → naming dei cluster → PCA/t-SNE → mappa di Ann Arbor (PNG + HTML folium).
È **veloce** (niente Optuna). Usare RPM/load qui è **legittimo**: NB3 *descrive* a posteriori,
non predice in anticipo (nessun conflitto col map-only di NB2).

## 3. Performance NB2 — perché era lentissimo e cosa è stato fatto

**Causa:** il tuning era stato alzato a `N_TRIALS=100`, `N_SPLITS=5` = **500 fit** di XGBoost,
con ricerca fino a 800 alberi / profondità 12, su ~1,3M righe per fit → ore.

**Fix applicati (2026-06-13):**
- `N_TRIALS` 100→25, `N_SPLITS` 5→3 (da 500 a 75 fit).
- Range ridotte: `n_estimators` 200–500, `max_depth` 4–8 (alberi più piccoli = fit rapidi).
- `SAMPLE_FRAC` 0.30→0.15.
- Risultato atteso: **~10–15 min** invece di ore. Qualità praticamente identica (XGBoost default
  map-only fa già R² ~0.62; il tunato guadagna pochi centesimi).
- Regola da ricordare: **costo tuning ≈ N_TRIALS × N_SPLITS × costo_di_un_fit**.

**GPU (opzionato, non adottato):** solo XGBoost ne beneficerebbe (Ridge/Lasso/RandomForest di
sklearn restano su CPU). Si attiva con `XGBRegressor(tree_method='hist', device='cuda')` se c'è
una GPU NVIDIA + CUDA. Con il tuning già snellito non è necessario; valutarlo solo se si torna a
budget di tuning alti. (Il controllo GPU non è stato eseguito su richiesta dell'utente.)

## 4. Integrazione NB2 ↔ NB3: NON farla nella versione naïve

Idea tentante: prendere il `cluster` del NB3 e darlo come feature a NB2. **Sconsigliato così com'è**,
per tre motivi:

1. **Data leakage / circolarità:** il cluster del NB3 è costruito da `maf_mean`, `rpm_mean`,
   `load_mean` aggregati per cella → l'etichetta *codifica già la media del target*. Usarla per
   predire il MAF = mean-target-encoding mascherata. R² gonfiato e ingannevole.
2. **Viola lo split temporale:** il clustering è fittato su tutti i dati (test incluso), quindi la
   label porterebbe informazione del periodo di test dentro il train.
3. **Distrugge il map-only:** reintroduce RPM/load (dentro il cluster) dopo averli tolti da NB2.

**Non serve forzarla per "coprire" l'unsupervised:** il NB3 da solo copre già il requisito.
Opzioni sensate:
- **(consigliata)** tenerli come due pilastri complementari; citare l'integrazione come *sviluppo futuro*;
- **(se si vuole davvero)** versione *pulita*: ricostruire i cluster usando **solo feature note a
  priori** (pendenza, geometria, soste — niente `maf_mean`/`rpm_mean`/`load_mean`) e fittando il
  clustering **solo sul train**. Difendibile, ma più lavoro.

## 5. Idee di sviluppo futuro (usando gli stessi argomenti d'esame)

Ordinate per rapporto valore/rischio.

### Più consigliate
- **Anomaly detection (unsupervised) + autoencoder.** Sfrutta i `Short/Long_Term_Fuel_Trim` e i
  valori-sensore impossibili. Isolation Forest / One-Class SVM + un piccolo **autoencoder**
  (errore di ricostruzione = anomalia). Vantaggi: copre l'**autoencoder** (argomento d'esame non
  ancora trattato), caso d'uso reale (diagnostica/manutenzione predittiva), secondo pilastro
  unsupervised accanto al clustering. Limite: niente etichette vere → valutazione qualitativa
  (distribuzione score, esempi), da dichiarare. NB: anomaly detection è unsupervised *qui* perché
  mancano le label; con label sarebbe classificazione supervised sbilanciata.
- **SHAP sul modello map-only.** Oltre `feature_importances_`: spiega *come* (segno/forma) MAF
  dipende da pendenza, velocità, delta futuro. Poco lavoro, rafforza la parte interpretativa.
- **Clustering a livello di guidatore.** Stesse tecniche del NB3 ma unità = `VehId`: profili di
  guida (aggressivo/eco/misto). Secondo deliverable unsupervised pulito.

### Altri angoli validi
- **Confronto fisico per EngineType:** HEV/PHEV hanno frenata rigenerativa → MAF in discesa/decel
  diverso dagli ICE. Spiega perché `EngineType` è feature utile.
- **Effetto temperatura/stagione:** `OAT_DegC` + 374 giorni → motore a freddo consuma di più.
- **Mappa del costo / eco-routing:** collega NB2+NB3 in una demo visiva (percorso a minor consumo).

### Da valutare con cautela
- **Classificazione supervised** (es. predire `EngineType` dal comportamento): aggiunge confusion
  matrix/ROC-AUC, ma rischia di essere o quasi impossibile (i tipi guidano simile) o "troppo facile"
  via RPM (leakage). Se fatta, definire bene cosa significhi il risultato.
