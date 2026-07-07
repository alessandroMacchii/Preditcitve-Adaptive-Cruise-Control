# Discussioni — registro Q&A per il ripasso d'esame

> Sintesi delle domande di Alex e delle risposte chiave, in forma abbreviata, per ricordarle quando
> si spiega il progetto. Si aggiorna a ogni nuova discussione. Per il dettaglio completo:
> `RELAZIONE_PROGETTO.md`, `ANALISI_DATI_VED.md`.
>
> **Avviato:** 2026-06-17.

---

## 1. Logica dei notebook e come spiegarli all'esame
**D:** Qual è la logica dei notebook, perché esistono, come li spiego al prof?
**R:** Non file slegati ma i **moduli di un assistente di guida / ACC predittivo**:
- **NB1** prep+enrichment (dati, quota, slope, accel) → `ved_enriched.parquet`.
- **NB2** valutatore di costo: stima il **consumo** (`maf_per_km`) di una strategia di guida (supervised).
- **NB3** riconoscitore di **contesto + stili di guida** (unsupervised).
- **Frase chiave:** *"dalla mappa conosco pendenza e tipo di tratto (NB1+NB3); un modello stima il
  consumo di ogni strategia (NB2); un controllore sceglie la più efficiente."*
- **Colpo di scena onesto:** l'idea iniziale (ACC orografico) è stata **smentita dai dati** (terreno
  piatto) → reinquadrata sul profilo di velocità. *"Ipotesi → smentita → correzione È il metodo."*

## 2. L'autoencoder (NB4) serviva? → rimosso
**D:** Il NB4 (autoencoder/diagnostica) non è necessario per il futuro ACC, era solo per l'esame?
**R:** Corretto: NB4 è **ortogonale** al cuore eco-driving (è manutenzione predittiva, non sta sul
loop consumo↔velocità) e i doc dicono che esisteva per coprire l'argomento d'esame. **Rimosso il
17/06.** Conseguenza onesta: il progetto **non copre più deep learning/anomaly detection**.
- **All'esame:** *"i pilastri predittivi sono consumo e contesto; la diagnostica è un modulo
  parallelo, l'ho tolta perché ortogonale al focus."*

## 3. Filo conduttore: "questa è solo l'introduzione"
**D:** Voglio chiudere con *"questa è l'introduzione, il CC adattivo si costruirà con più dati"*.
**R:** Presentare i notebook come **fondazioni + scoperta del gap**, non come progetto finito:
1. ho costruito e validato i **mattoni** (consumo, contesto) su dati reali;
2. ho **mappato cosa manca** per il sistema completo (orografia vera + SoC batteria + il
   **pianificatore**, cioè il controllore che sceglie la velocità).
- La frase finale diventa una **roadmap**, non un alibi.

## 4. NB2 e NB3 si possono unire in un notebook?
**D:** Ora che NB4 non c'è, unisco NB2 e NB3?
**R:** Tecnicamente sì, ma **meglio separati**:
- mappano i due blocchi del corso (**supervised** vs **unsupervised**);
- la separazione fisica **è** la prova del punto anti-leakage (vedi §5);
- esecuzione/RAM indipendenti (NB2 carica ICE+Optuna, NB3 aggrega 17,9M righe).
- Unirli conviene **solo** se la consegna richiede un singolo file → due sezioni nette.

## 5. Perché non ho usato i cluster (NB3) come feature nel consumo (NB2)?
**D:** Perché non do il cluster in pasto al modello di consumo?
**R:** Perché il cluster è **costruito da `maf_mean`/`rpm_mean`** → usarlo = iniettare la media del
target → **leakage (target encoding mascherato)**. R² gonfiato ma circolare.
- **Sottigliezza:** la motivazione *sbagliata* è "un ACC non conosce il tipo di strada" (falso: si sa
  da mappa). Quella *giusta* è "il cluster è definito usando il target".
- In più non guadagneresti nulla: NB2 ha già le feature cinematiche grezze; del cluster resterebbe
  solo la parte leaky.
- **Fix onesto (futuro):** ri-clusterizzare **senza** MAF/RPM/Load.

## 6. NB3: perché celle 50×50 m?
**D:** Perché celle da 50×50 m?
**R:** **Non sono 50 m — era un refuso nel markdown, ✅ corretto il 17/06 (ora dice ~11×8 m).**
Arrotondare a 4 decimali dà celle da **~11 m (N-S) × ~8 m (E-O)**. (1° lat ≈ 111 km → 0,0001° ≈ 11 m;
in longitudine × cos(42°) ≈ 8 m.)
- **Trade-off dimensione cella:** grande → mescola tratti diversi; piccola → poche misure/cella.
- **Conseguenza (bias di selezione):** celle piccole + filtro ≥50 passaggi → restano solo le
  **strade trafficate** (281.494 → 77.325 celle). Da dichiarare.

## 7. Le celle ~11×8: sono lunghezza e larghezza della strada?
**D:** La strada è larga 8 e lunga 11?
**R:** No. 11 e 8 sono i due lati di un **quadretto di mappa fisso** (verticale N-S / orizzontale
E-O), diversi tra loro per la **convergenza dei meridiani** (cos della latitudine), **non** per la
strada. La strada ha sua direzione e sua larghezza (~7–10 m). La cella ≈ uno spezzone corto di strada.

## 8. Cella del filtro (2.1): mostrare quali celle si scartano + scala log
**D:** Il filtro scarta le celle scarse ma non mostra quali → sembra "pochi passaggi ovunque".
**R:** Due difetti veri: (a) scala colore **lineare** schiacciata dagli outlier (~4000); (b) il plot
mostra solo i sopravvissuti. **Fix applicato al NB3:** due pannelli — sinistra scartate (grigio) vs
tenute (blu); destra densità in **scala log**.

## 9. Clusterizzare per tratto di strada invece che per quadrato? (map-matching)
**D:** Un quadrato può contenere due strade diverse — meglio clusterizzare per segmento stradale?
**R:** Hai ragione sul principio: l'unità corretta è il **tratto di strada** (map-matching su
**OpenStreetMap**: ogni punto GPS → arco stradale; strade parallele = archi diversi, incroci = nodi).
- Ma **costo alto** (dipendenza OSM, snapping 17,9M punti) e i **cluster verrebbero quasi uguali**
  (li guidano i comportamenti aggregati, non la geometria esatta).
- **Raccomandazione:** tienilo come **limite dichiarato #1 / sviluppo futuro** (fa fare bella figura).
  Mitigazione leggera possibile: feature di **dispersione dell'heading** per riconoscere gli incroci.

## 10. Come si calcola lo slope?
**D:** Come è calcolata la pendenza?
**R:** "Salita fratto percorso": `slope = Δquota / Δdistanza orizzontale`, tra **due punti GPS
consecutivi dello stesso trip**. Es. salgo 2 m in 40 m → 0,05 = 5%. È **tratto-per-tratto**, non globale.
**Passi del codice (NB1, cella "Calcolo della pendenza"):**
1. **Ordina** per `VehId, Trip, Timestampms` (consecutivo = ordine temporale del viaggio).
2. **Punto precedente** dentro il trip con `groupby(['VehId','Trip']).shift(1)` (lat/lon/quota/tempo della
   riga prima). Lo shift *dentro il trip* evita di legare fine-viaggio↔inizio-successivo → la **prima riga
   di ogni trip non ha "prima"** → NaN (poi droppata nel salvataggio).
3. **Numeratore** `dz_m = elevation_m − elev_prev` (dislivello verticale).
4. **Denominatore** `dist_m = haversine(lat_prev,lon_prev, lat,lon)` (distanza orizzontale sulla sfera
   terrestre, **formula di Haversine**, R=6.371 km).
5. **Divisione con due guardie:**
   - `np.where(dist_m > 1.0, dz/dist, 0.0)` → se l'auto è quasi ferma (Δdist < 1 m) mette **0** invece di
     dividere per ~0 (niente valori esplosivi);
   - `.clip(-0.3, 0.3)` → tappa a **±30%**; oltre, in città, è errore SRTM non rampa vera.
- **Perché 0 nel ~92%:** è il **numeratore** → `dz_m = 0` ogni volta che i due punti consecutivi cadono
  nello **stesso blocco-quota ~111×82 m** (stessi `lat_round`/`lon_round` → stessa `elevation_m`). Non è il
  denominatore: è che la **quota non cambia dentro il blocco**. Metodo corretto, debole solo per la
  *sorgente quota* quantizzata. Vedi [[#11]] (quota ricavata da noi) e [[#33]] (perché non più decimali).
- **Nota gemella:** l'**accelerazione** è calcolata con la stessa logica ma sul tempo: `accel = Δvelocità /
  Δt` (`dv/(dt_ms/1000)`, guardia `dt > 50 ms`, clip −15..+10 km/h·s) per gestire il **sampling irregolare**
  (Δt varia ~100–1400 ms) — non si può usare un `diff()` semplice.

## 11. L'elevation era nel dataset o l'abbiamo ricavata?
**D:** La quota c'era nei dati grezzi?
**R:** **Ricavata noi nel NB1.** Il VED grezzo **non ha quota**; abbiamo le coordinate GPS, le
arrotondiamo a **3 decimali (~111 m)** → ~8.700 punti unici → **Open-Meteo Elevation API** (SRTM,
gratis) → merge. Per questo la quota è **costante dentro ogni cella da 111 m** → punti consecutivi
hanno la **stessa quota** → `dz=0` → **slope=0** nel 92% delle righe (non perché sia piatto lì).

## 12. E se facessi le celle più piccole di 111 m?
**D:** Rimpicciolendo le celle si recupera lo slope?
**R:** **No** — il limite non è la dedup, è la **sorgente**. SRTM ha risoluzione **~30 m** e errore
verticale di **diversi metri**:
- sotto i 30 m **non c'è nuova informazione** (resampli/interpoli lo stesso pixel);
- a ~30 m avresti meno zeri, ma i nuovi Δquota sarebbero **rumore**, non salite vere (peggio: sembra
  segnale);
- costo: 4 decimali → ~280k punti (~32× chiamate API) + rate-limit;
- e Ann Arbor è **piatta** (101 m totali): non c'è segnale da recuperare.
- **Vera soluzione:** DEM ad alta risoluzione (es. LiDAR USGS 3DEP 1 m) **oppure** una città collinare.

## 13. NB3 clustering: perché MAF/RPM/Load insieme? Cos'è Load? Perché il MAF come feature?
**D:** Perché il MAF come feature di clustering, perché anche gli RPM, e Load cos'è?
**R:** Cosa sono (analogia **bici**): **RPM** = cadenza di pedalata (regime); **Load**
(`Absolute_Load_pct`) = quanto spingi forte = **sforzo in % del max** (0% minimo, ~90% a tavoletta);
**MAF** = calorie/s = **consumo assoluto** (g aria/s).
- **Non identiche:** stesso MAF da RPM alti+carico basso *o* RPM bassi+carico alto → RPM/Load dicono
  il *come*, il MAF il *quanto*.
- **Perché il MAF nella Parte A:** dà la "firma energetica" del tratto (superstrada = MAF alto, via
  residenziale = basso). Ma è **discutibile**: il MAF è una *conseguenza* della guida, non proprietà
  indipendente della strada; e `maf_mean` per cella è pure sporcato dagli ibridi (MAF=0 in elettrico).
- **Ridondanti:** corr col MAF ≈ **0,75 (RPM)**, **0,52 (Load)** → tenerle tutte e tre **sovrappesa**
  la dimensione "sforzo motore" (3 feature ~collineari su 10) e rende il cluster **leaky** (vedi §5).
- **Contrasto utile:** la **Parte B** (stili) esclude apposta il MAF (powertrain-agnostica). Una
  versione pulita della Parte A farebbe lo stesso: solo **cinematica + geometria**.

## 14. Decisione: tolti i segnali-motore dal clustering dei tratti (NB3 Parte A)
**D:** A questo punto escludiamo il MAF anche qui, come negli stili di guida?
**R:** Sì, ma per farlo "come la Parte B" vanno tolti **tutti e tre** i segnali-motore
(`maf_mean`/`rpm_mean`/`load_mean`), non solo il MAF: RPM e Load correlano col MAF (0,75/0,52), quindi
da soli lascerebbero la stessa dimensione e il **leakage**. **Applicato al NB3:** feature del
clustering tratti ora = **cinematica + geometria** (7: `speed_*`, `accel_*`, `slope_mean`,
`stop_fraction`).
- **Conseguenze:** (1) i cluster cambiano → `road_segment_clusters.parquet` da rigenerare; (2) la demo
  dello scaling ora la domina `speed_mean` (~90%) invece di `rpm_mean` (99,52%) — stessa lezione; (3)
  niente più leakage. Aggiornati anche i markdown del NB3.

## 15. Ma lo "sforzo motore" non è un aiuto in più per separare i cluster?
**D:** Anche se leaky, sapere quanto consuma/sforza il motore non aiuta a separare meglio i tratti?
**R:** No, e per motivi più forti del "leaky":
- **"Leaky" conta solo a valle:** per il clustering in sé (non supervisionato) non c'è leakage; vale
  solo se riusi i cluster nel NB2. Da solo è un argomento debole — giusta osservazione.
- **Non è info nuova:** il consumo è una **conseguenza** di velocità/accel/pendenza (già presenti) →
  **collineare** → non apre un asse nuovo, **sovrappesa** quello esistente (più feature ridondanti ≠
  più separazione).
- **La parte davvero nuova è il confondente sbagliato:** dipende da **chi guida lì** (flotta, ibridi a
  MAF=0), cioè separa per *chi percorre* la strada, non per *tipo* di strada → come coords/elevation,
  fuori.
- **Bonus interpretativo:** escludendolo, "i cluster differiscono nel consumo" diventa una **scoperta**
  a posteriori invece di una **tautologia**.
- **Dipende dall'obiettivo:** per "tipo/contesto di strada" → fuori; per "mappa di domanda energetica"
  → dentro. Regola: metti tra le feature ciò che vuoi che i cluster **significhino**.

## 16. Maf/rpm/load nell'aggregazione: servono ancora?
**D:** Nell'aggregazione (cella `[4]`) calcoliamo ancora maf/rpm/load — servono?
**R:** Per il **clustering** no (esclusi). Oggi sono calcolati e salvati nel parquet ma **mai usati**.
- **`maf_mean` → tenerlo e *usarlo*:** aggiungerlo a `cluster_profile` come colonna **descrittiva**
  (non di clustering) per **mostrare a posteriori** che i cluster differiscono nel consumo = la
  "scoperta non tautologia" di #15.
- **`rpm_mean`/`load_mean` → eliminabili** (interni motore, inutili anche descrittivamente); o lasciarli
  (cheap, innocui).
- **Distinzione chiave:** *feature di clustering* (cinematica+geometria, **creano** i cluster) vs
  *colonne descrittive* (maf_mean, **interpretano** i cluster dopo). All'esame: *"il consumo non forma
  i cluster, li descrive."*
- **Stato:** ✅ **fatto** — `maf_mean_descr` aggiunta al `cluster_profile` (Parte A) come colonna
  descrittiva, non di clustering. `rpm_mean`/`load_mean` restano calcolati in `agg` ma inutilizzati
  (eliminabili quando si vuole).

## 17. Il filtro ≥50 passaggi scarta troppe celle? Abbassarlo?
**D:** Scartiamo ~72% delle celle (281k→77k). 50 passaggi non sono tanti? Conviene abbassare?
**R:** Da inquadrare bene:
- **Non è un problema di quantità:** 77k celle sono già abbondanti per K-Means; il 72% scartato sono
  celle **minuscole e rumorose** (residenziali poco battute, bordi-cella) → spazzatura, non info.
- **50 = righe (campioni GPS), non veicoli distinti** → 50 campioni per stimare media/std è un
  **minimo ragionevole**, non alto. Sotto ~20-30 le `std` diventano instabili.
- **Trade-off di abbassare:** + copertura / meno bias verso le arterie (potrebbe emergere un cluster
  residenziale); − statistiche più rumorose → silhouette (già ~0,23) potrebbe peggiorare.
- **Da decidere coi dati, non a intuito:** test di sensibilità `MIN_PASSAGES ∈ {20,50,100}` →
  n. celle + silhouette + nº cluster. Se la struttura è stabile → 50 difendibile; se a 20 spunta un
  cluster sensato senza crollo silhouette → abbassare.
- **Stato:** ✅ **check fatto** (script sui dati reali, soglie 10→150). **Esito: si tiene 50.**
  Silhouette: abbassando **peggiora** (~0,235 a 10-20), alzando **satura** (~0,260 da 75 in su ma con
  metà celle). A 50 = ~0,257 col **massimo di copertura** (~77k celle) → ginocchio della curva.
  Abbassarla degrada la separazione; alzarla è guadagno trascurabile + meno copertura. Giustificazione
  aggiunta al markdown del NB3 (cella 2.1) e alla GUIDA.

## 19. Spiegazione: standardizzazione e scelta di K
**D:** Non ho capito la standardizzazione e la scelta di K.
**R (standardizzazione):** K-Means usa la **distanza** → feature con scale diverse si schiacciano.
Prova nel notebook: senza scaling `speed_mean` = **91% della varianza**, `slope_mean`/`stop_fraction`
= 0% (ignorate). Lo **StandardScaler** porta ogni feature a **media 0, std 1** (z-score =
`(x−media)/std` = "quante std dalla media") → tutte pesano uguale. Obbligatoria per metodi a distanza
(K-Means/PCA/KNN); inutile per alberi (XGBoost). Analogia: raggruppare persone per altezza/peso/stipendio
— senza scalare domina lo stipendio (numeri grandi).
**R (scelta di K):** K-Means non sceglie K da solo. Due strumenti:
- **Elbow (inerzia/WCSS):** compattezza dei cluster; scende sempre con K → si cerca il **gomito** (dove
  smette di scendere ripido), lettura visiva.
- **Silhouette (−1..+1):** **separazione**; +1 ben dentro il cluster, 0 sul confine, <0 cluster sbagliato.
  Si prende il K con silhouette massima (qui tra 3 e 7 → **K=6**).
- **Caveat onesto:** silhouette **bassa e piatta** (~0,21–0,26 per ogni K) → cluster esistono ma **poco
  separati** (i tipi di strada sfumano). Quindi K=6 è un **compromesso interpretabile**, non un ottimo
  netto. **All'esame dirlo tu**, confermando coi cluster sensati nella heatmap.

## 20. Nominare i cluster dai valori (invece dell'auto_name a caso)
**D:** Puoi dare tu i nomi guardando la heatmap, spiegando in base a quali valori?
**R:** Sì. Il vecchio `auto_name` usava soglie *relative* (top 30%) → **doppioni**. Riscritto
`name_cluster` per assegnare **dai valori** (robusto al rinumeramento). 6 cluster (run seed 42, 7 feat):
- **Incrocio/semaforo (fermo):** speed 3,6 · stop **0,77**
- **Urbano stop-and-go:** speed 22 · stop **0,28**
- **Avvicinamento incroci/decel.:** speed 37 · accel_mean **−1,26**
- **Ripartenze/accelerazione:** speed 36 · accel_mean **+0,97**
- **Scorrevole/arteria fluida:** speed 57 · regolarissima (std 10)
- **Autostrada/scorrimento veloce:** speed **84**
- **Finding 1:** il clustering ha separato accel (+) vs decel (−) **a parità di velocità** (C3 vs C4).
- **Finding 2 (scoperta a posteriori):** il consumo descrittivo `maf_mean_descr` li ordina
  sensatamente (autostrada 17,6 > ripartenze 13,2 > scorrevole 10,0 > stop-and-go 7,9 > decel 7,5 >
  incrocio 4,5) — cluster *cinematici* che differiscono nel consumo = risultato, non tautologia.
  (MAF qui è istantaneo g/s = quanto lavora il motore, non consumo per km.)
- **Logica naming:** stop>0,50 → incrocio; stop>0,18 → stop-and-go; speed≥75 → autostrada; speed≥50 →
  scorrevole; altrimenti per segno di accel_mean → ripartenze/decel.

## 21. Mappa HTML: mostrare/nascondere singoli cluster
**D:** Posso selezionare di visualizzare solo alcuni colori (cluster) alla volta sulla mappa?
**R:** Sì. Riscritta la cella mappa (NB3): ogni cluster è un **`FeatureGroup`** separato + un
**`LayerControl(collapsed=False)`** → pannello in alto a destra con una **casella per cluster**
(selezione multipla, non radio). Legenda colori in basso a sinistra. Da rigenerare rieseguendo il NB3.
- **Demo esame:** *"la mappa è interattiva — isolo solo gli incroci o solo le arterie veloci per
  vedere come si distribuiscono in città."*

## 18. Il Load è utile per i profili guidatore (aggressività)?
**D:** Un guidatore aggressivo preme di più l'acceleratore → il Load non aiuta a profilare gli stili?
**R:** Intuizione giusta (Load = sforzo sul pedale, più *diretto* dell'accelerazione risultante), ma **no**:
- **Load NON è powertrain-agnostico** (stesso difetto del MAF): è un PID del **motore termico** → per gli
  ibridi in elettrico il motore è spento → **Load=0**. Un PHEV sembrerebbe "gentile" solo perché motore
  spento → falserebbe il confronto **stile × powertrain** (lo scopo della Parte B).
- **Aggressività già catturata** in modo equo da `frac_hard_accel`/`frac_hard_decel`/`accel_*` (vengono
  dalla velocità → valgono per tutti i powertrain).
- **Esame:** "Load = stesso problema del MAF (0 in elettrico); l'aggressività la do con
  `frac_hard_accel/decel`, powertrain-agnostiche."

---

> **Tornata 2026-06-17 — domande sul NB2 (consumo / eco-driving).**

## 22. Keras nel NB2 invece di sklearn, come nel NB3?
**D:** Si potrebbe usare Keras nel NB2 come nel terzo notebook?
**R:** **Premessa errata: il NB3 non usa Keras** — è sklearn (K-Means/PCA). Keras/torch erano
solo del NB4 (rimosso). Si *potrebbe* mettere una MLP, ma è la scelta **sbagliata**: su **dati
tabellari** il **gradient boosting batte le reti** (prior consolidato); dataset non enorme; perderei
la **feature importance** (il cuore dell'argomento "strada sì, terreno no"); niente GPU/tuning fragile.
- **Esame:** *"su tabellari il boosting è lo stato dell'arte; le reti le riservo a segnali grezzi."*

## 23. Perché 250 m per segmento?
**D:** Perché 250 m e non meno/più?
**R:** **Trade-off**, verificato empiricamente con una **cella di sensibilità** (presente nelle versioni
precedenti del NB2, rimossa nello sfoltimento del 18/06). Corti (50-100 m)
→ statistiche per-segmento rumorose, `maf_per_km` instabile su distanze piccole, orizzonte `next_*`
inutile; e **sotto la risoluzione SRTM (~30 m) il dislivello è solo rumore** → un segmento troppo corto
non aggiunge informazione di terreno, ricade nella tautologia del MAF *istantaneo*. Lunghi (500-1000 m)
→ **mescolano tratti eterogenei** (arteria + semaforo nella stessa media), persa la risoluzione che serve
a pianificare. **250 m** ≈ tratto omogeneo ~10-25 s di guida = scala fisica della pianificazione di un ACC,
**sopra il rumore SRTM** ma ancora fine per l'anticipazione. Giustificato **dati + dominio**, non a intuito.
- **Conferma empirica (cella sensibilità SEG_LEN ∈ {150, 250, 500}, run della versione pre-18/06):**
  l'R² resta **stabile** e il peso del **terreno resta debole (~0,06) a tutte le lunghezze** → la scelta
  di 250 m non è arbitraria e non cambia le conclusioni (il limite è il *dato*, non la granularità).
  *Nota: la cella è stata rimossa dal notebook il 18/06 (sfoltimento); il risultato resta citabile come
  analisi fatta.* [[#36]] [[#42]]

## 24. Cosa fa la cella della sez. 5 (Target e feature, map-only)?
**D:** Cosa fa la "cella 5"?
**R:** (cella `[10]`) Incarna la decisione **MAP-ONLY**, tre cose: (1) target `maf_per_km`; (2) lista
**20 feature** = geometria/terreno + cinematica + **anticipazione** `next_*` + contesto (hour/month/
peso/OAT), **escludendo** RPM/Load/MAF istantaneo; (3) `dropna` → `seg_model`. In breve: **fissa cosa
si predice e con quali ingressi, tutti noti in anticipo a un ACC.**

## 25. Anche la macchina non conosce gli RPM in anticipo → perché escluderli era giusto?
**D:** L'ACC reale non sa gli RPM prima del tratto, come noi: dov'è la differenza?
**R:** La distinzione è **di quale segmento**. Il modello predice il **segmento in arrivo** → le feature
devono essere note **prima di guidarlo**: geometria (mappa) + profilo di velocità che il controllore
**sceglie** di provare. RPM/Load/MAF **del segmento futuro** sono il **risultato** del guidarlo (per
misurarli dovrei già conoscerne il consumo = il target) **e** quasi-circolari col MAF (≈RPM×carico).
Gli "RPM attuali" del tratto di ora esistono ma **non descrivono** il tratto futuro ipotetico da valutare.
- **Esame:** *"li escludo non perché segreti, ma perché sono il risultato del guidare il tratto che
  voglio predire — usarli è usare la risposta."*

## 26. Perché non uso i cluster del NB3 come feature nel NB2? (aggiorna #5)
**D:** Il cluster non è una feature in più per il consumo?
**R:** Storicamente **leakage** (cluster costruito su `maf_mean`/`rpm_mean` = target encoding mascherato).
**Aggiornamento post-#14:** ora il NB3 clusterizza **senza** maf/rpm/load → **non più leaky**. Ma la
ragione che **regge comunque**: **nessuna info nuova** — il cluster è una **compressione con perdita**
di `speed_*`/`accel_*`/`stop_fraction` che il NB2 **ha già grezze**, + crea una **dipendenza di
pipeline** NB3→NB2. *"I cluster **descrivono** il contesto, non **alimentano** il modello di consumo."*

## 27. Cos'è il look-ahead? È la velocità futura?
**D:** Il look-ahead è la velocità che so esserci nel prossimo segmento? allora non andrebbe esclusa?
**R:** **No, non è la velocità realizzata.** Sono 3 feature di **strada** del segmento successivo
(`next_dz_net`, `next_slope_mean`, `next_stop_fraction`) via `shift(-1)` (cella `[8]`). **Lecite**
perché derivano dalla **mappa/route** (pendenza e stop davanti = noti dal navigatore) → è proprio
l'**anticipazione** da dimostrare. **Illecito** (mai fatto): forward del **target o di chi lo determina**
(`next_MAF`, RPM futuri, velocità realizzata). Sfumatura onesta: `next_stop_fraction` è semi-comportamentale,
usata come proxy map-knowable di "stop davanti" → la feature look-ahead più borderline, da dichiarare.

## 28. Perché split train/test nel tempo e non mischiare i trip?
**D:** Perché dividere temporalmente e non a caso i segmenti?
**R:** **Leakage da correlazione.** Segmenti dello **stesso trip** sono fortemente correlati (stesso
driver/veicolo/meteo). Split casuale per segmento → pezzi dello stesso trip in train **e** test → il
modello **memorizza** il trip → **R² gonfiato**. Split per **trip interi ordinati nel tempo** (cella `[12]`,
80% più vecchi → train) = simula il **deployment reale** (alleno sul passato, predico viaggi nuovi) +
protegge dal drift. Stessa logica in CV con **GroupKFold per VehId** (cella `[18]`).

## 29. estimators/depth/lr di XGBoost a caso? si migliorano?
**D:** I parametri XGBoost sono scelti a caso? non si possono migliorare?
**R:** **No, e sono già ottimizzati.** Il **default** (cella `[16]`: 500/6/0.1) sono valori standard =
**baseline**. Il modello **finale** (sez. 9-10, celle `[18]`/`[20]`) usa **Optuna** (ricerca bayesiana
TPE, 30 trial) su estimators/depth/lr **+ regolarizzazione**, valutata in **GroupKFold per VehId**,
min MAE → `study.best_params`. Quindi il tuning **c'è già**, più serio di una grid manuale. Si può
spingere oltre (più trial, range più ampi, early stopping) con rendimenti decrescenti.
- **Esame:** *"i default sono lo standard; i finali li sceglie Optuna con ottimizzazione bayesiana e CV
  per veicolo — non a caso, ottimizzati."*

---

> **Tornata 2026-06-17 (bis) — sfoltimento del NB3 e allineamento doc.**

## 30. Sfoltimento del NB3: cosa è stato tolto e cosa implica
**D:** Ho tenuto nel NB3 solo ciò che presento; allinea markdown e file di contesto. Metti tutti gli
import in un'unica cella.
**R:** Fatto. **Rimosse** dalla Parte A: t-SNE e il sanity check **cluster↔EngineType**; dalla Parte B:
test **chi-quadro** stile×powertrain, relazione **stile→consumo** (ICE) e **confronto energetico** tra
motori. Conseguenze e interventi:
- **Bug introdotto dalle cancellazioni → corretto:** la cella PCA scatter era rimasta **orfana** (usava
  `X_pca`/`pca` mai definiti → `NameError`). Ho **ripristinato il calcolo PCA** prima dello scatter
  (sezione 8) — la PCA è una tecnica che vuoi mostrare e lo scatter era tenuto.
- **Import consolidati** nella cella `[2]` (tolti i re-import sparsi di `LogNorm`, `StandardScaler`,
  `KMeans`, `silhouette_score`, `PCA`). `import folium` resta nel `try/except`: è una **dipendenza
  opzionale**, spostarlo su romperebbe il fallback alla mappa statica.
- **Refuso "50×50 m" → corretto** in `~11×8 m` (vedi #6).
- **Narrativa "tre powertrain":** la Parte B ora **profila gli stili** (cinematica, powertrain-agnostica)
  e mostra la composizione per `EngineType`, ma **non** dimostra più statisticamente "i powertrain sono
  guidati diversamente" né "perché gli ibridi consumano diverso". Quella tesi resta argomentata nel
  **NB2 §1** e in `RELAZIONE §5.1`; i tre blocchi tolti sono ora **sviluppi citabili**, non risultati.
- **All'esame:** se il prof chiede del confronto powertrain, dire che è **documentato concettualmente**
  ma non più calcolato nel notebook (scelta di scope), non spacciarlo per fatto.
- **Pendenze residue (per Alex):** `K_FINAL` è **forzato a 4** in `[14]`; restano **due celle di
  riepilogo** redondanti in fondo (accorpabili); i numeri "Risultato reale" della GUIDA Parte A sono di
  una run vecchia → da riconfermare rieseguendo.

## 31. Il "filtro sul movimento" va tolto per il clustering? Due dataset (supervised senza soste / clustering con soste)?
**D:** Visto che facciamo clustering, non conviene togliere il filtro sul movimento? O fare un dataset
supervised senza i momenti fermi e uno per il clustering con anche i fermi?
**R:** **Equivoco da chiarire: il filtro NON rimuove le soste.** È nel **NB1 §5**:
`df[(speed > 0) | (RPM > 100)]` → tiene la riga se si muove **O** il motore è acceso. Quindi auto ferma
al **semaforo col motore acceso, stop-and-go, code → TENUTI**. Rimuove solo `speed=0` **E** `RPM≤100`
(**motore spento e fermo**: parcheggi, avvii/fine trip) = **334.788 righe, ~1,8%**.
- **Toglierlo per il clustering → peggiora:** le soste utili (`stop_fraction` per incroci/semafori) sono
  **già dentro** (motore acceso). Le righe tolte sono **parcheggi/bordi-trip** → reinserirle creerebbe
  celle "sempre ferme" nei parcheggi = rischio **cluster-spazzatura "fermo"** che non è un tipo di strada.
  Il filtro quindi **aiuta** il clustering.
- **Supervised "senza i fermi" → è il contrario di ciò che serve:** nel NB2 lo `stop_fraction` è la
  **feature più importante** ("lo stop-and-go costa"). Toglierlo distruggerebbe il segnale. I segmenti
  degeneri (quasi-sosta, distanza ~0) sono già scartati a livello di segmento (`seg_distance_m ≥
  0,5·SEG_LEN` + `maf_per_km` finito) — non serve farlo per riga.
- **Raccomandazione:** **un solo dataset filtrato com'è.** Conserva tutte soste/idle/stop-and-go per
  entrambi i task e rimuove ~1,8% di righe morte. Due dataset = più complessità, zero guadagno, rumore-
  parcheggi nel clustering. L'unico affinamento *opposto* sensato sarebbe escludere meglio i parcheggi
  dal clustering, ma il filtro ≥50 passaggi + feature cinematiche lo gestisce già in gran parte.

## 32. Non conveniva fare il binning a 2 decimali (più preciso)?
**D:** A 2 decimali sarebbe stato molto più preciso, no?
**R:** **Invertito: meno decimali = celle più GRANDI = MENO preciso.** I 4 decimali attuali sono la
scelta *più* fine, non meno. Scala (lat ~42,3°N, lon × cos≈0,74):
- **2 dec** (0,01°) → **~1,1 km × 0,82 km** = un **intero quartiere**/cella (autostrada+residenziale+
  incrocio mescolati) → clustering senza senso.
- **3 dec** (0,001°) → ~111 m × 82 m (un isolato).
- **4 dec** (0,0001°) → **~11 m × 8 m** ← attuale (uno spezzone di strada).
- **5 dec** → ~1,1 m, sotto il **rumore GPS** (~5-10 m) → solo rumore.
- **Limite vero = sorgente, non i decimali:** GPS ~5-10 m (scendere sotto ~11 m non aggiunge info); per
  lo **slope** il limite è SRTM ~30 m + quota campionata a **3 dec (~111 m)** nel NB1 → è quello a
  indebolire la pendenza, non il bin del clustering.
- **Unico cambio difendibile = direzione opposta:** passare a **3 decimali (~111 m)** darebbe più
  campioni/cella e meno scarto (oggi il filtro ≥50 butta ~72% delle celle perché a 11 m sono minuscole),
  al prezzo di mescolare più strada nella stessa cella. Compromesso, non un "meglio" netto → **4 decimali
  vanno bene**.
- ⚠️ **Attenzione (vedi #33):** ci sono **due arrotondamenti diversi** — il **bin del clustering** (NB3)
  è a **4 decimali ~11 m**; il **campionamento della quota** per l'API (NB1) è a **3 decimali ~111 m**.

## 33. Allora la quota non conveniva farla a 4 decimali (più precisa)?
**D:** Hai fatto la quota a 3 decimali con l'API; non era meglio 4 (più preciso)?
**R:** Giusto, l'API quota è a **3 decimali** (`ROUND_DECIMALS=3`, ~111 m) — da non confondere col **bin
del clustering** che è già a **4 decimali** (~11 m). Sulla **quota**, 4 decimali **non** avrebbe aiutato,
e il NB1 §6 lo spiega:
- **Limite = sorgente, non decimali:** dietro Open-Meteo c'è **SRTM ~30 m**; punti < 30 m cadono nello
  **stesso pixel** → stessa quota. A 4 dec (~11 m) **ricampioni lo stesso pixel** → i nuovi Δquota sono
  **rumore di interpolazione**, non salite (sembrerebbe segnale → peggio).
- **Costo:** `round(4)` → ~282k punti → ~2800 batch → **>45 min** (+ rate-limit); `round(3)` → ~8.700
  punti → ~88 batch → **1-2 min**. Stessa info utile, 32× più veloce.
- **Ann Arbor piatta:** range quota di tutta la città **223–324 m ≈ 101 m** → niente slope da recuperare.
- **Vera soluzione (non più decimali):** DEM ad alta risoluzione (LiDAR USGS 3DEP 1 m) o città collinare.
  Collega a [[#11]] (quota ricavata noi, dedup a 3 dec) e [[#12]] (rimpicciolire le celle non recupera lo slope).

## 34. La quota è in metri o feet? A volte fa "drop assurdi" — abbiamo sbagliato le unità?
**D:** Siamo sicuri che `elevation_m` sia in metri e non feet? A volte ci sono cali di quota assurdi.
**R:** **È in metri, nessun errore di unità** (verificato sui dati reali):
- Quota media **268,8 m**, range **223–324 m**. Ann Arbor reale ≈ **256 m s.l.m. (840 ft)** → **combacia**.
  In **feet** la media (269 ft) sarebbe **82 m** → impossibile. Open-Meteo/SRTM dà sempre metri.
- I "**drop assurdi**" **non sono un bug**, sono **quantizzazione**: la quota è **a gradini interi** (102
  valori distinti = ~1 per metro). Tra punti consecutivi **dz = 0 nel 91,9%** (stesso blocco ~111 m); al
  confine di blocco salta tutta la differenza in un colpo → `p99 |dz| = 30 m`, **max |dz| = 94 m** (quasi
  tutta l'escursione cittadina di 101 m in un passo → non è un dirupo, è il gradino SRTM / a volte salto GPS).
- Per questo lo `slope` è **clippato a ±0,3** → il **2,5%** delle righe ci sbatte (sono proprio quei drop).
- **Morale:** sembrano assurdi *perché* Ann Arbor è piattissima (101 m totali) → ogni gradino è una fetta
  enorme del rilievo. È il **limite della sorgente quota**, non un errore metri/feet. Collega [[#33]], [[#10]].

## 35. Le TRE griglie/unità del progetto + cosa vuol dire "arrotondare a N decimali"
**D:** Se i blocchi quota sono 111×82, allora dire che il segmento stradale è 11×8 è sbagliato? E perché
non posso scegliere una griglia 10×10? Cosa significa arrotondare al 3°/4° decimale?
**R:** **Niente è sbagliato: sono griglie diverse.** Attenzione a non confondere TRE "pezzi":
| Unità | Come | Dimensione | Dove |
|---|---|---|---|
| **Segmento NB2** | ~250 m *lungo* la traiettoria (distanza cumulata) | 250 m lineari | consumo (NB2) |
| **Cella di clustering** | `round(4)` su lat/lon | **~11×8 m** | tratti stradali (NB3-A) |
| **Blocco quota** | `round(3)` su lat/lon | **~111×82 m** | elevation (NB1) |
- **Arrotondare a N decimali = agganciare la coordinata alla griglia più vicina con passo 10⁻ᴺ gradi.**
  Tutti i punti nello stesso quadretto → stesso valore → stessa cella. 1° lat ≈ 111.320 m, quindi ogni
  decimale **divide per 10**: 2 dec=1,1 km · 3 dec=111 m · 4 dec=11 m · 5 dec=1,1 m. **Solo scatti ×10**,
  niente in mezzo. Rettangolare (11×8, non 11×11) perché la longitudine vale `cos(42,3°)≈0,74` della latitudine.
- **Un blocco quota (111 m) contiene ~100 celle di clustering (11 m)** → ~100 celle vicine condividono la
  STESSA quota → altro motivo per cui `slope_mean` è debole a scala 11 m.
- **Posso scegliere 10×10 (o 30×30)?** Sì, ma NON con `.round()`: serve uno **snap metrico**
  `round(coord/step)*step` con `step = metri/111320` (e `/cos(lat)` per la longitudine). `.round()` è solo
  la scorciatoia comoda incatenata ai passi ×10. Vedi [[#6]], [[#32]], [[#33]].

## 36. Tentativo griglia quota a 30 m — esplorato e ANNULLATO (2026-06-17)
**D:** Facciamo la quota a 30 m (= risoluzione nativa SRTM) così magari migliora lo slope anche negli altri NB?
**R:** Idea **giusta in direzione** (i 111 m attuali sono più grossi di SRTM → buttano dettaglio reale), e
costo verificato: ~**54.591 punti** a 30 m. **Annullato** prima della presentazione perché:
- **Tempo reale ~1,5–2 h:** l'API gratuita Open-Meteo limita ~500–600 coordinate/min → 429 ricorrenti
  (attese di 60s; più tentativi `(n/12)` = il limite non si è ancora liberato; arrivare a 12 = tetto
  orario/giornaliero → lo script si ferma, ma è **resumable**, salva ogni batch). Poi servirebbe rieseguire NB1+NB2+NB3.
- **Guadagno marginale:** Ann Arbor è piatta (101 m) + SRTM ha errore verticale ~6–16 m → lo slope resterebbe
  comunque secondario, e a 30 m alcuni Δquota nuovi sarebbero **rumore** (slope finti).
- **Decisione:** ripristinato tutto a 111 m (`ROUND_DECIMALS=3`) dai commit/backup; `ved_enriched.parquet`
  non era stato toccato. Resta un ottimo **sviluppo futuro citabile** ("allineare il campionamento ai 30 m
  nativi SRTM, o usare un DEM LiDAR 1 m"). Collega [[#33]], [[#12]].

## 37. Vocabolario NB3: `agg`, media/std/abs, `stop_fraction`
**D:** Cos'è `agg`? Cosa vogliono dire media/abs/std e la frazione di sosta?
**R:**
- **`agg`** = il **DataFrame delle celle** (NB3-A): `df.groupby(['lat_bin','lon_bin']).agg(...)` trasforma
  17,9M righe → ~281k celle (1 riga = 1 cella ~11×8 m), con medie/varianze del comportamento per cella. È
  **l'input del clustering**. Catena: `agg` → filtro ≥50 → ~77k → `agg_clean` (dropna) → K-Means. Calcola
  anche `maf/rpm/load_mean` ma **non** entrano nei cluster (solo `maf_mean_descr` descrittiva dopo).
- **media** = somma/conteggio (valore tipico). **std** = quanto i valori sono sparsi attorno alla media
  (basso=regolare, alto=ballerino). **abs** = grandezza senza segno (`|−3|=3`).
- **Perché `accel_abs_mean` e non solo media:** su un tratto misto la media dell'accelerazione ≈ 0 (i +
  e − si annullano) → nasconde la guida a strappi. Prendendo prima il valore assoluto si misura
  l'**intensità** (accelerate+frenate), che non si annulla. `std` cattura la stessa "nervosità" come dispersione.
- **`stop_fraction`** = `(speed < 2).mean()` = **frazione di campioni praticamente fermi** (0–1). Trucco:
  media di Vero/Falso = proporzione di Veri. `<2` (non `=0`) per il rumore del sensore. Letture: 0=scorrevole,
  ~0,2=stop-and-go, ~0,5–0,8=incrocio/semaforo. È la feature **più importante del NB2** e quella che riconosce
  gli incroci nel NB3.

## 38. Il MAF è l'apertura della farfalla e il Load è quanto premi l'acceleratore?
**D:** Quindi MAF = apertura del corpo farfallato e Load = quanto premi l'acceleratore?
**R:** **No, due correzioni** — sono entrambi *misure di risposta del motore*, non posizioni di pedale/farfalla:
- **MAF** = **massa d'aria che entra davvero** nel motore (g/s, da sensore). Non è la valvola: l'apertura
  farfalla *influenza* il MAF, ma il MAF è il **risultato** misurato. Proxy del consumo (aria ≈ benzina, ~14,7:1).
- **Load** (`Absolute_Load_pct`) = **quanto lavora il motore in % del max** (carica d'aria per ciclo
  normalizzata). Correla col pedale ma è lo **sforzo effettivo del motore**, non la posizione del pedale.
- **Richiesta vs risposta:** pedale (richiesta) → farfalla si apre → MAF ↑ (aria) → Load ↑ (sforzo) → più benzina.
  Né MAF né Load sono il pedale/farfalla; e nel VED **non** c'è la posizione di pedale o farfalla (solo MAF/RPM/Load).
- Analogia bici: RPM=cadenza, Load=quanto spingi (sforzo %), MAF=calorie/s (consumo). Dettaglio in [[#13]].

## 39. Heatmap dei cluster (NB3): non si vede `slope_mean` né il MAF descrittivo
**D:** Nella heatmap dei 4 cluster non compaiono `slope_mean` e la colonna descrittiva del MAF.
**R:** Due cause distinte:
- **`slope_mean` invisibile = bug del `.round(2)`.** `cluster_profile` fa `.mean().round(2)`, ma lo slope
  vale ~0,00x → arrotondato diventa **0.00 per tutti e 4 i cluster** (verificato: 0.0/-0.0/-0.0/0.0) →
  varianza 0 tra cluster → lo z-score divide per std=0 → **NaN → riga vuota**. **Fix:** calcolare lo
  z-score della heatmap dalle medie **NON arrotondate** (`agg_clean.groupby('cluster')[...].mean()`),
  non da `cluster_profile`. (La tabella `cluster_profile` può restare arrotondata, serve solo a leggere.)
- **`maf_mean_descr` assente = voluto.** La heatmap plottava solo `FEATURES_CLUSTER` (le 7 che *formano*
  i cluster); il MAF è descrittivo. **Fix:** aggiunto come **riga extra separata da una linea**
  (`ax.axhline(len(FEATURES_CLUSTER))`), etichettata "non di clustering".
- **Lettura onesta:** dopo il fix `slope_mean` compare ma è **quasi piatto** tra i cluster (z-score piccoli)
  → conferma che il terreno non separa i tipi di strada. `maf_mean_descr` invece è **molto marcato**
  (autostrada caldo ~16,7, incrocio freddo ~6,3) → i cluster cinematici differiscono anche nel consumo =
  risultato, non tautologia [[#15]] [[#16]]. **Da rieseguire la cella heatmap.**

## 40. Sarebbe stato meglio un modello lineare su questi dati?
**D:** Visto che il MAF è asimmetrico ed esiste `log_MAF`, non sarebbe stato meglio usare un modello lineare?
**R:** **No, è il contrario.** Due piani da non confondere:
- **L'asimmetria riguarda il target, non la scelta del modello.** La log-trasformazione (`log_MAF`) è
  semmai una **stampella per i modelli lineari** (assumono residui ~normali/omoschedastici) — serve *loro*,
  non a XGBoost. Quindi quel dettaglio non spinge verso il lineare. [[#38]]
- **I dati sono fortemente non-lineari + con interazioni:** consumo vs velocità è a U/J (alto in città
  stop-and-go, minimo in crociera, di nuovo alto in autostrada); `stop_fraction` (feature #1), accel e
  velocità **interagiscono**. Un lineare cattura solo effetti additivi/lineari → per avvicinarsi servirebbero
  feature polinomiali e termini d'interazione costruiti **a mano**. XGBoost lo fa **automaticamente** (split +
  alberi). È il *prior* del progetto: tabellari con non-linearità → gradient boosting. R² ~0,76 map-only.
- **Quando il lineare sarebbe stato preferibile (onestà):** interpretabilità diretta dei coefficienti;
  pochi dati (non è il caso, 17,9M righe); **estrapolazione** fuori range (gli alberi non estrapolano, la
  retta sì); coprire la **regolarizzazione L1/L2** come argomento di corso. Un **Lasso** baseline sarebbe
  l'unico vero motivo per reintrodurlo (decisione #5 di STATE.md lo aveva rimosso) — **non perché predica
  meglio**, ma per confronto interpretabile / coprire L1/L2.
- **Frase per il prof:** *"L'asimmetria del MAF non giustifica un lineare — è il lineare che avrebbe bisogno
  della log-trasformazione. I dati hanno non-linearità e interazioni forti, quindi il prior corretto su
  tabellari è il gradient boosting; un Lasso resta utile solo come baseline interpretabile."*

## 41. Cos'è la regressione lineare e a cosa servirebbe invece di XGBoost?
**D:** Cos'è un modello tipo regressione lineare e a cosa mi servirebbe al posto di XGBoost?
**R:** **È il modello più semplice per predire un numero:** assume il target come **somma pesata** delle
feature, `ŷ = w₀ + w₁x₁ + … + wₙxₙ` (es. `maf_per_km ≈ w₀ + w₁·speed_mean + w₂·stop_fraction + …`).
- **Come impara:** minimi quadrati (OLS) — trova i pesi `w` che minimizzano la somma degli errori al
  quadrato; geometricamente la retta/iperpiano più vicino ai punti.
- **Il valore sta nei coefficienti `w`:** **segno** = direzione (`w_stop_fraction>0` → più soste = più
  consumo), **grandezza** = sensibilità ("+0,1 di stop → +X di maf_per_km, a parità di resto").
- **Varianti regolarizzate (argomenti di corso):** **Ridge (L2)** rimpicciolisce tutti i coefficienti
  (multicollinearità); **Lasso (L1)** può **azzerarli** → selezione automatica di feature.
- **A cosa servirebbe INVECE di XGBoost** (non per predire meglio, ma per ciò che il boosting fa peggio):
  (1) **interpretabilità diretta** dei coefficienti (XGBoost richiede feature importance/SHAP); (2)
  **baseline onesta** = il modello da battere (se XGBoost 0,76 vs lineare 0,60 → la non-linearità vale
  +0,16, dimostrato); (3) **estrapolazione** fuori range (gli alberi danno valore piatto, la retta continua
  il trend); (4) coprire **L1/L2** nel programma (oggi assenti, decisione #5); (5) pochi dati (non è il caso,
  17,9M righe).
- **Il prezzo del lineare:** cattura solo relazioni lineari/additive → la curva a U "consumo vs velocità"
  la sbaglia senza feature polinomiali (`speed²`) e interazioni (`speed×stop_fraction`) costruite a mano;
  va scalato; vuole un target più simmetrico (qui rientrerebbe `log_MAF`); sensibile agli outlier.
- **Mossa giusta nel progetto:** non "lineare *al posto di*" ma un **Lasso come baseline accanto** a XGBoost
  → copre la regolarizzazione, dà coefficienti leggibili, quantifica quanto vale il boosting. [[#40]]

## 42. Quali sono le feature che diamo a XGBoost (NB2)?
**D:** Spiegami le feature dentro XGBoost.
**R:** `FEATURES_NUM` = **20 feature numeriche**, tutte a livello di **segmento ~250 m**, tutte **map-only**
(note in anticipo a un ACC). *Niente scaling: XGBoost è ad albero → invariante a trasformazioni monotone
(la Pipeline con StandardScaler c'era nelle versioni precedenti, rimossa il 18/06 — vedi [[#43]]).*
Quattro gruppi:
- **Terreno/geometria** (peso totale ~0,06 → limite del dato): `seg_distance_m` (lunghezza),
  `dz_net` (dislivello netto = elev_end−elev_start), `climb_m`/`descent_m` (metri saliti/scesi cumulati),
  `slope_mean`, `slope_max_abs`. Scelta: **dislivello cumulato sul segmento**, più robusto dello slope
  istantaneo spiky/azzerato. [[#36]]
- **Cinematica** (segnale forte): `speed_mean`, `speed_max`, `speed_min`, `speed_std` (variabilità),
  `accel_abs_mean` (intensità accel+frenate, l'abs evita che +/− si annullino), `stop_fraction`
  (frazione quasi-fermi `speed<2` = stop-and-go), `entry_speed` (velocità d'ingresso `first` = energia
  cinetica ereditata = accoppiamento all'indietro). **Importance:** stop_fraction #1 (~0,13), speed_mean
  (~0,08), entry_speed (~0,07).
- **Anticipazione/look-ahead** (cuore "ACC"): `next_dz_net`, `next_slope_mean`, `next_stop_fraction`,
  via `groupby(['VehId','Trip']).shift(-1)` = info del **segmento successivo**. Entra il futuro di
  strada/velocità, **MAI il MAF futuro** (sarebbe leakage del target). [[CLAUDE.md trappole]]
- **Contesto veicolo/ambiente:** `Generalized_Weight` (classe di peso del veicolo), `OAT_DegC`
  (temperatura esterna → motore freddo/caldo, clima), `hour` e `month` (proxy di traffico/stagione).
  Nel grafico importance finiscono nel gruppo "cinematica/contesto".
- **Escluse di proposito:** `Engine_RPM_RPM`, `Absolute_Load_pct`, `rpm_roll10s_mean` (non noti in anticipo
  + quasi-circolari col MAF ≈ RPM×carico → scorciatoia che azzera il ruolo della strada); `EngineType`
  (modello solo-ICE → costante). **Target:** `maf_per_km` (efficienza, non distanza). [[#40]]

## 43. Cos'è il ColumnTransformer e perché va con lo StandardScaler?
**D:** Cos'era il ColumnTransformer e perché sta insieme allo StandardScaler?
**R:** Sono **due cose distinte**, spesso confuse:
- **StandardScaler** = *trasforma* le feature a **media 0, std 1** (stessa scala). Serve perché `speed_mean`
  (decine) e `slope_mean` (~0,00x) hanno scale diversissime.
- **ColumnTransformer** = il *contenitore* che dice **a quali colonne** applicare quale trasformatore (non
  scala nulla da solo, orchestra). Caso classico: numeriche→StandardScaler, categoriche→OneHotEncoder,
  ricomposte in un'unica matrice.
- **Nel NB2 (versioni fino al 17/06):** `ColumnTransformer([('num', StandardScaler(), FEATURES_NUM)])`
  = un solo gruppo (le numeriche), niente categoriche (`EngineType` escluso, solo-ICE). **Perché usarlo
  con un solo trasformatore?** (1) scaffold standard/scalabile (domani aggiungi una categorica con una
  riga); (2) seleziona esattamente le colonne per nome (non scali ID/target per sbaglio); (3) si incastra
  nella **Pipeline** col modello. *Il 18/06 lo sfoltimento del NB2 ha rimosso Pipeline/ColumnTransformer/
  StandardScaler: XGBoost lavora direttamente sulle colonne (vedi Nota onesta sotto). Il concetto resta
  materia d'esame.* [[#42]]
- **Perché dentro la Pipeline = anti-leakage:** lo scaler viene **fittato solo sul train** (media/std) e
  *applicato* a test/fold con quei parametri. Scalare a mano tutto il dataset prima dello split → media/std
  vedono anche il test → **leakage** → metriche false. La Pipeline lo blinda, soprattutto in CV (GroupKFold,
  scaler rifittato a ogni fold). [[#22]]
- **Nota onesta:** XGBoost è ad albero → **invariante a trasformazioni monotone**: lo StandardScaler **non
  cambia le sue predizioni**. Si teneva per buona pratica/riusabilità (un Lasso, o il K-Means/PCA del NB3
  lo *richiedono*) — ed è proprio per questo che toglierlo dal NB2 (18/06) è legittimo. Nel NB3 lo scaling
  resta **obbligatorio** (dimostrazione "senza scaler" esplicita). [[#41]]
