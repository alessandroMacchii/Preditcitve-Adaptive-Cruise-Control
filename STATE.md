# STATE.md — Stato corrente del progetto

> Aggiornare questo file ogni volta che si fa qualcosa di rilevante.
> **Ultimo aggiornamento:** 2026-06-11 (fix check cache cella 18 NB1)

## Stato pipeline: ⛔ BLOCCATA alla cache elevation

```
[OK] Dati grezzi          54 parquet in content/, 18.258.138 righe, schema verificato
[~ ] elevation_cache      700 / 8.707 punti validi (8%) — INCOMPLETA
[KO] ved_enriched.parquet 0 RIGHE — generato con cache vuota, il dropna finale ha eliminato tutto
[KO] Notebook 2           mai eseguito con dati validi (nessun output: no joblib, no csv)
[KO] Notebook 3           mai eseguito con dati validi (nessun output: no parquet, no mappe)
```

### Causa del blocco
Il Notebook 1 è stato eseguito quando le elevation non erano disponibili → `elevation_m` quasi tutto NaN → `dropna(subset=['elevation_m'])` in cella 27 ha prodotto un parquet **vuoto**. Successivamente `build_elevation_cache.py` (resumabile) ha scaricato 700 punti prima di fermarsi (presumibilmente rate-limit 429 di Open-Meteo).

### Per sbloccare (in ordine)
1. Rilanciare `./.venv/Scripts/python.exe build_elevation_cache.py` finché la cache non è completa (8.007 punti mancanti ≈ 81 batch ≈ 3–5 min se Open-Meteo non rate-limita; lo script riprende da dove si è fermato). **Chiedere ad Alex prima di lanciare le chiamate API.** In alternativa la cella 18 del NB1 ora completa la cache da sola (vedi punto 2).
2. ~~Check cache debole in cella 18~~ **RISOLTO 2026-06-11**: la cella 18 ora fa merge con la cache, scarica SOLO i punti mancanti (resumabile come lo script), salva la cache aggiornata e solleva RuntimeError se dopo il fetch restano NaN. Non può più produrre un enriched vuoto in silenzio.
3. Rieseguire NB1 per intero (~5–10 min senza API).
4. Eseguire NB2 (lento: Optuna ~15–30 min) e NB3 (~5–10 min).

## Cosa è già fatto

- Struttura dei 3 notebook completa e coerente (codice + markdown narrativo)
- `build_elevation_cache.py` resumabile con gestione 429/Retry-After
- Bug look-ahead già corretto in passato: le feature "future" usavano rolling trailing; ora usano `FixedForwardWindowIndexer` con `shift(-1)` (vedi memoria `ved-lookahead-bug`)
- requirements.txt, README.md, .gitignore, .claude/settings.json presenti
- PROJECT_CONTEXT.md e STATE.md creati (questo audit)

## Cosa manca

- [ ] Completare elevation cache (bloccante, richiede autorizzazione di Alex)
- [ ] Eseguire NB1 → NB2 → NB3 end-to-end con dati reali
- [ ] Verificare i risultati reali (i notebook non hanno mai prodotto numeri: R², silhouette, K scelto, cluster naming sono tutti da vedere per la prima volta)
- [ ] Naming manuale dei cluster dopo aver visto la heatmap (l'euristica `auto_name` è solo una proposta)
- [ ] Aggiornare il riepilogo del NB2 ("5 modelli" ma ne elenca 6)

## Bug e problemi noti (da audit 2026-06-09)

1. **NB1 cella 22/25 — slope/accel della prima riga di ogni trip = 0.0, non NaN**: `np.where(NaN > soglia, …, 0.0)` restituisce 0.0 perché il confronto con NaN è False. Il `dropna(subset=['slope','accel_kmh_s'])` in cella 27 quindi NON rimuove le prime righe dei trip come il commento promette: restano con slope/accel fasulli a 0. Impatto basso (~1 riga per trip) ma incoerente col markdown.
2. ~~**NB1 cella 18 — check cache debole**~~: RISOLTO 2026-06-11, vedi punto 2 dello sblocco.
3. **NB3 cella 4 — la cella NON è ~50×50 m**: round a 4 decimali = 0.0001° ≈ 11 m in lat × ~8 m in lon (a 42°N). Il markdown dice 50×50. O si corregge il testo, o si usa un binning vero a 50 m (es. round a 3 decimali su griglia metrica o `floor(coord/0.0005)`). Con celle da ~11 m il filtro ≥50 passaggi tiene solo le strade molto trafficate.
4. **NB2 cella 6 — finestre in CAMPIONI, non secondi**: i nomi (`speed_roll30s_mean`, `slope_future_5`…) e il markdown parlano di secondi, ma rolling(30) = 30 campioni; col sampling irregolare (100–1400 ms) 30 campioni = 3–42 s. O si dichiara l'approssimazione nel markdown, o si usa rolling time-based.
5. **NB2 cella 38 — controfattuale incompleto**: scala speed, roll mean e speed_future ma NON ricalcola `speed_delta_future_5` (che andrebbe scalato anch'esso) e lascia RPM/Load invariati (quest'ultimo limite è dichiarato nel markdown, il delta no).
6. **NB2 — `Absolute_Load_pct` tra le feature è quasi-circolare col target**: Absolute Load è calcolato dalla centralina a partire dalla massa d'aria (≈ MAF normalizzato per RPM). Predire MAF da Load+RPM è quasi deterministico → R² gonfiato e feature look-ahead schiacciate nell'importance. Punto d'attacco forte all'esame. Vedi analisi critica.
7. **Cartella `-p` spuria** nella root (artefatto di un `mkdir -p` su PowerShell). Rimuovere a mano.
8. **README** elenca `project_ved/` come nome cartella e promette output non ancora esistenti; promette anche "~150 chiamate" ma i batch reali sono ~88 da 100 punti.

## Suggerimenti prioritari (sintesi — dettaglio nel report dell'audit)

1. Sbloccare la pipeline (cache → NB1 → NB2 → NB3) e vedere i numeri reali
2. Decidere cosa fare di `Absolute_Load_pct` (toglierla o aggiungere un modello "map-only" di confronto)
3. Correggere etichette 50m/secondi nei markdown (costo zero, evita domande imbarazzanti)
4. Integrazione NB3 → NB2: usare il cluster della cella come feature categorica nel supervised (chiude la narrativa)
5. Valutare se aggiungere un piccolo autoencoder (materia d'esame non coperta)
