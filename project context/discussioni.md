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
- esecuzione/RAM indipendenti (NB2 carica ICE+Optuna, NB3 17,9M righe+t-SNE).
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
**R:** **Non sono 50 m — è un refuso nel markdown.** Arrotondare a 4 decimali dà celle da **~11 m
(N-S) × ~8 m (E-O)**. (1° lat ≈ 111 km → 0,0001° ≈ 11 m; in longitudine × cos(42°) ≈ 8 m.)
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
