# Analisi dei dati — Vehicle Energy Dataset (VED arricchito)

> EDA completa di `outputs/ved_enriched.parquet` (output del NB1). Tutti i numeri sono **reali**,
> calcolati sull'intero dataset. Serve a capire com'è fatto il dato e perché sono state prese certe
> scelte di modeling. Per il quadro d'esame vedi `RELAZIONE_PROGETTO.md`.

---

## 1. Colpo d'occhio

| | |
|---|---|
| Righe | **17.922.869** |
| Colonne | 21 |
| Memoria in RAM | ~2,06 GB |
| Veicoli | **299** |
| Trip | **26.285** |
| Powertrain | ICE 58% · HEV 30% · PHEV 12% |
| Periodo | 2017-11-01 → 2018-11-10 (**374 giorni**) |
| Area | Ann Arbor (MI), ~12×10 km |
| Distanza totale flotta | ~**138.827 km** |
| Frequenza campioni | mediana **600 ms** (irregolare) |

Origine: telemetria OBD-II reale; l'altitudine **non** è nel grezzo → derivata nel NB1 (Open-Meteo /
SRTM), da cui `elevation_m` e `slope`. Aggiunti anche `accel_kmh_s`, `dist_m`, `dt_ms`.

## 2. Schema (21 colonne)

- **Chiavi/tempo:** `VehId` (int), `Trip` (int), `Timestampms` (ms intra-trip), `Datetime`
  (timestamp, **costante per trip** = inizio viaggio).
- **Posizione/terreno:** `Latitude_deg`, `Longitude_deg`, `elevation_m`, `slope` (frazione).
- **Dinamica:** `Vehicle_Speed_km_per_h`, `accel_kmh_s`.
- **Motore:** `Engine_RPM_RPM`, `Absolute_Load_pct`, `MAF_g_per_sec`, `log_MAF`.
- **Diagnostica:** `Short_Term_Fuel_Trim_Bank_1_pct`, `Long_Term_Fuel_Trim_Bank_1_pct`.
- **Contesto:** `OAT_DegC` (temperatura esterna), `EngineType`, `Generalized_Weight`.
- **Derivate per riga:** `dist_m`, `dt_ms` (distanza e Δt dal punto precedente).

## 3. Valori mancanti

Solo due colonne hanno NaN: **`dist_m` e `dt_ms`, 26.285 ciascuna** — esattamente il numero di trip.
Sono la **prima riga di ogni trip** (non c'è un punto precedente da cui calcolare distanza/Δt). Tutto
il resto è completo. *(Implicazione pratica: nei notebook questi NaN vanno azzerati prima di
`cumsum`/integrazioni — trappola già gestita nel NB2.)*

## 4. Powertrain (EngineType)

| Tipo | righe | % | veicoli | trip |
|---|---|---|---|---|
| ICE | 10.401.317 | 58,0% | **197** | 14.705 |
| HEV | 5.378.063 | 30,0% | **90** | 9.327 |
| PHEV | 2.143.489 | 12,0% | **12** | 2.253 |

⚠️ **Caveat importante:** i PHEV sono solo **12 veicoli** (e 2.253 trip). Qualsiasi conclusione
"per PHEV" è statisticamente fragile (poche unità) e va presa come indicativa. ICE e HEV sono ben
rappresentati.

## 5. Copertura temporale

- **Per mese:** abbastanza uniforme, con picchi a **novembre (2,19M) e dicembre (1,96M)** → più dati
  invernali (utile per l'effetto temperatura sul consumo).
- **Per ora** (di inizio trip): picchi a **mezzogiorno (~1,5M) e 20–21 (~2M)**, minimo 5–8 del
  mattino → pattern di pendolarismo.
- **Per giorno:** il **giovedì** è il più rappresentato; nel complesso distribuzione ragionevole su
  tutta la settimana.

## 6. Geografia

`lat ∈ [42,220, 42,326]`, `lon ∈ [−83,804, −83,674]` → un riquadro di ~**12×10 km** su Ann Arbor.
Area urbana/suburbana compatta: niente autostrade lunghe, niente montagne.

## 7. Statistiche delle variabili chiave

| Variabile | mediana | media | p95 | max | note |
|---|---|---|---|---|---|
| Vehicle_Speed (km/h) | 44 | 41,4 | 83 | 173 | 5° pct = 0 (soste) |
| accel (km/h/s) | 0 | −0,1 | 7,5 | 10 | clip [−15, +10] |
| MAF (g/s) | 5,58 | 9,85 | 30,4 | 259 | target, asimmetrico |
| Engine_RPM | 1255 | 1198 | 2373 | 6605 | |
| Absolute_Load (%) | 26,6 | 29,7 | 62,0 | 196 | clip a 200 nel NB1 |
| OAT (°C) | 9 | 11,0 | 31 | 60 | range −40…60 (estremi sospetti) |
| slope (frazione) | 0 | 0 | 0 | 0,3 | **quasi sempre 0** (§11) |
| elevation (m) | 268 | 268,8 | 298 | 324 | range 101 m |
| Generalized_Weight | 3500 | 3375 | 4500 | 6000 | |
| dist_m (per riga) | 0 | 7,76 | 52,8 | 11.210 | 0 quando fermo |
| Short_Term_Fuel_Trim (%) | 0 | 0,13 | — | 89,8 | diagnostica |
| Long_Term_Fuel_Trim (%) | 0,78 | 1,26 | — | 57,8 | diagnostica |

## 8. Il target: MAF

- Fortemente **asimmetrico a destra**: mediana 5,58 ma media 9,85, **skew 1,91**, coda fino a 259 g/s.
- Per questo nel grezzo esiste già **`log_MAF`** (range 0–5,56), utile se si vuole un target più
  simmetrico.
- **9,5%** delle righe ha MAF < 0,5 g/s (motore al minimo / trazione elettrica negli ibridi);
  MAF esattamente 0 solo lo 0,36%.

## 9. Sampling irregolare

Il Δt tra campioni **non è costante**: mediana **600 ms**, p95 1900 ms, p99 2400 ms, ma **max
~7.444.700 ms (~2 ore!)**. Lo **0,48%** dei Δt supera i 3 s (buchi di logging).
**Implicazioni:** l'accelerazione va calcolata sul Δt reale (fatto nel NB1); per integrare il
consumo sul segmento (NB2) il Δt va **cappato** (a 2 s) per non sommare "aria fantasma" sui buchi.

## 10. Struttura di trip e veicoli

- **Trip:** 26.285. Righe/trip mediana **536** (p05 133, p95 1716, max 10.813); distanza/trip mediana
  **4,3 km** (p95 14 km); durata mediana **6,9 min** (p95 23 min). → trip urbani brevi.
- **Veicoli:** 299. Righe/veicolo **molto sbilanciate**: mediana 38.673, **min 394, max 822.320**.
  Un veicolo da solo ha ~822k righe.

⚠️ Lo **sbilanciamento per veicolo** è il motivo per cui nel tuning si usa **GroupKFold per VehId**:
senza, un veicolo dominante finirebbe sia in train sia in validation, gonfiando la performance.

## 11. Il terreno — la criticità principale

- **Range di quota di tutta la città: solo 101 m** (223 → 324). Terreno dolce.
- **Il 92% delle righe ha |slope| < 1%**; i percentili dal 5° al 95° sono **esattamente 0**.
- **Correlazione slope ↔ MAF = 0,004** (nulla).

Doppia causa: (a) Ann Arbor è quasi piatta; (b) **artefatto di quantizzazione** — nel NB1 le
coordinate sono deduplicate a 3 decimali (~111 m) per l'API altitudine, quindi tutti i punti dentro
una cella ricevono la **stessa quota** → `dz=0` → slope nullo dentro cella e "a gradini" solo ai
bordi (poi clippato a ±0,3). **È il motivo del reframe del progetto** (il terreno non è il segnale
utile; lo è il profilo di velocità).

## 12. Correlazioni con il MAF (Pearson)

| feature | corr con MAF |
|---|---|
| Engine_RPM_RPM | **0,751** |
| Absolute_Load_pct | **0,515** |
| Vehicle_Speed_km_per_h | 0,326 |
| accel_kmh_s | 0,299 |
| Generalized_Weight | 0,143 |
| elevation_m | −0,046 |
| OAT_DegC | −0,015 |
| **slope** | **0,004** |

Lettura cruciale: **RPM (0,75) e Load (0,52) sono quasi-deterministici col MAF** — è la riprova
quantitativa del perché vanno **esclusi** dal modello map-only (sarebbe come predire il MAF dalla
formula che lo genera). La velocità e l'accelerazione (0,33 / 0,30) sono i segnali "leciti"; lo
**slope è inutile (0,004)**.

## 13. Movimento e soste

Il **10,1%** delle righe è a veicolo fermo (speed < 2 km/h): soste, semafori, traffico. Rilevante
per il NB2 (lo `stop_fraction` per segmento è risultato il driver dominante del consumo).

## 14. Confronto tra i tre powertrain

| | MAF medio | RPM medio | motore spento in marcia | MAF in decel | speed media |
|---|---|---|---|---|---|
| **ICE** | 11,73 | 1398 | **0,0%** | 5,86 | 39,0 |
| **HEV** | 8,28 | 1079 | **21,0%** | 2,42 | 45,0 |
| **PHEV** | 4,65 | 526 | **24,4%** | 1,46 | 43,7 |

- HEV/PHEV girano col **motore spento il 21–24% del tempo in marcia** (trazione elettrica) → il MAF
  per loro **non misura il consumo reale** (manca l'energia elettrica, che il VED non registra). È il
  motivo per cui il modello di consumo (NB2) è **solo-ICE**.
- Il **MAF in decelerazione** crolla per gli ibridi (5,86 → 2,42 → 1,46): firma della **frenata
  rigenerativa** (motore spento in rilascio). Materiale per il confronto powertrain nel NB3.

## 15. Implicazioni per il modeling (sintesi)

1. **Map-only giustificato dai numeri:** RPM/Load correlano 0,75/0,52 col target → esclusi.
2. **Terreno = limite del dato:** slope ~0 nel 92% dei casi, corr 0,004 → reframe sul profilo di
   velocità; terreno documentato come limite.
3. **GroupKFold per VehId:** dati per veicolo fortemente sbilanciati (max 822k righe).
4. **Solo-ICE per il consumo:** MAF valido solo per i termici; HEV/PHEV trattati a parte.
5. **PHEV fragili statisticamente:** 12 veicoli → conclusioni indicative.
6. **Δt cappato** nelle integrazioni per via dei buchi di sampling (max ~2 h).
7. **Target asimmetrico:** disponibile `log_MAF` se serve simmetria.
8. **Prima riga di ogni trip:** `dist_m`/`dt_ms` NaN da azzerare.
