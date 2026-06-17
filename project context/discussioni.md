# Discussioni — registro Q&A per il ripasso d'esame

> Sintesi delle domande di Alex e delle risposte chiave, in forma abbreviata, per ricordarle quando
> si spiega il progetto. Si aggiorna a ogni nuova discussione. Per il dettaglio completo:
> `RELAZIONE_PROGETTO.md`, `FAQ_DATI_E_MODELLO.md`, `GUIDA_CELLE_NB02_NB03.md`.
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
**R:** `slope = Δquota / Δdistanza`, **punto contro il precedente** dentro lo stesso trip (`shift(1)`):
`dz = quota_ora − quota_prima`, `dist` = distanza orizzontale (Haversine). Es. salgo 2 m in 40 m →
slope 0,05 = 5%. Non è un valore "globale", è tratto-per-tratto.

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
  niente più leakage. Aggiornati anche i markdown del NB3 e la `GUIDA_CELLE_NB02_NB03.md`.

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
**R:** **Trade-off** + c'è già la **cella di sensibilità** (sez. 12) che lo verifica. Corti (50-100 m)
→ statistiche per-segmento rumorose, `maf_per_km` instabile su distanze piccole, orizzonte `next_*`
inutile. Lunghi (500-1000 m) → **mescolano tratti eterogenei** (arteria + semaforo nella stessa media),
persa la risoluzione che serve a pianificare. **250 m** ≈ tratto omogeneo ~10-25 s di guida = scala
fisica della pianificazione di un ACC. Giustificato **dati + dominio**, non a intuito.

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
