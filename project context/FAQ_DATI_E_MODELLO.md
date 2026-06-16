# FAQ — capire i dati e il modello (spiegazioni semplici)

> Risposte alle domande di Alex sul dataset e sul progetto. Linguaggio volutamente semplice.
> Numeri da `ANALISI_DATI_VED.md`. Per il quadro d'esame vedi `RELAZIONE_PROGETTO.md`.

---

## A) Cosa sono i dati

### Che cos'è il dataset, in pratica?
È telemetria **OBD-II reale**: ~300 veicoli veri che girano per Ann Arbor (Michigan) per ~1 anno.
Ogni riga = una **fotografia di un veicolo in un istante** durante un viaggio (velocità, giri motore,
aria aspirata, ecc.). Quindi sì: è una raccolta continua dei veicoli mentre circolano.

### Ogni quanto vengono presi i dati? (~1 volta al secondo?)
Più o meno, ma **irregolare**: il tempo tra due campioni ha mediana **600 ms** (quindi spesso ~1,6
volte al secondo), ma varia da 100 ms a oltre 2 s, con qualche **buco lungo** (max ~2 ore = il logger
si è fermato). Per questo molti calcoli usano il Δt **reale** invece di assumere 1 secondo fisso.

### Cos'è `Timestampms`?
Sono i **millisecondi dall'inizio del trip** (tempo *dentro* il viaggio), non l'orario del giorno.
L'orario di calendario è `Datetime` (che è **costante per tutto il trip** = momento di partenza).
Quindi: `Datetime` ti dice *quando* è partito il viaggio; `Timestampms` ordina le righe *nel tempo*
dentro quel viaggio (0, 600, 1200 ms…).

### I dati di un trip sono correlati tra loro o sono punti sparsi?
Sono una **sequenza ordinata nel tempo** (sono correlati: la velocità a t dipende da t-1). E noi
**sfruttiamo** questa correlazione: accelerazione, pendenza, medie mobili, look-ahead e segmenti sono
sempre calcolati **dentro lo stesso trip, in ordine di tempo** (mai mescolando trip diversi). Anche
lo split train/test è **per trip interi**, così non si "bara" usando il futuro di un viaggio per
predirne il passato.

---

## B) Le singole variabili

### Cos'è il `MAF`? L'ho calcolato io?
**No, è un sensore reale** del motore (Mass Air Flow = massa d'aria aspirata, in **grammi al
secondo**), già presente nel datato grezzo. Noi NON lo calcoliamo: lo usiamo come **bersaglio** da
predire. (Quello che abbiamo calcolato noi è: pendenza, accelerazione, distanza, Δt.)

### Un MAF di 5 è tanto o poco?
È in **grammi d'aria al secondo**. Riferimenti tipici:
- minimo/idle: ~2–4 g/s
- guida tranquilla / crociera: ~5–15 g/s
- accelerazione forte / salita sotto carico: 30–50+ g/s

Quindi **5 g/s = guida leggera, basso consumo**. Nel nostro dataset: mediana 5,58, p95 30, max 259.
Poiché l'aria è ~proporzionale al carburante bruciato (rapporto aria/benzina ~14,7:1), il MAF è un
**proxy del consumo istantaneo**: più alto = più carburante.

### Cos'è il `Fuel Trim Bank`?
Il **fuel trim** è la **correzione che la centralina applica alla miscela aria/benzina**, in %.
- *Short-term* = correzione istantanea; *Long-term* = correzione media imparata nel tempo.
- *Bank 1 / Bank 2* = i due "banchi" di cilindri (i motori a V ne hanno due; quelli in linea uno solo
  → Bank 2 spesso vuoto/zero).

Valori grandi e persistenti (es. +20%) = la centralina sta compensando molto → **sintomo di un
problema** (presa d'aria, sensore che deriva…): un valore anomalo è un campanello d'allarme. Sono le
variabili tipiche di una **diagnostica / manutenzione predittiva** (caso d'uso non trattato in questo
progetto).

### Lo `slope` è una frazione? Uno slope di 1 = sale di 1 metro?
Lo slope è **pendenza = Δquota / Δdistanza** (metri/metri) → un **numero puro (frazione)**, non metri.
- `slope = 0,05` → 5% → sale **5 m ogni 100 m** percorsi.
- `slope = 1` → 100% → 45° → salirebbe **1 m ogni 1 m** orizzontale = un muro (irreale).

Nel nostro dato è limitato (clip) a ±0,3 (±30%). Quindi **no**: slope=1 non vuol dire "1 metro", vuol
dire una pendenza ripidissima. Per sapere "di quanto sale" guardi la **differenza di quota** (`dz`),
non lo slope.

### Lo slope è quasi sempre 0 → il terreno non è quasi mai in pendenza?
Due cose insieme:
1. **Ann Arbor è davvero quasi piatta** (dislivello totale 101 m su tutta la città).
2. **Artefatto:** per ottenere l'altitudine abbiamo raggruppato le coordinate a celle di ~111 m
   (per non fare 18M chiamate API). Tutti i punti dentro una cella hanno la **stessa quota** → Δquota
   = 0 → slope = 0. Quindi slope=0 spesso significa "stesso quadretto di mappa", non "perfettamente
   in piano".

Risultato: lo slope è quasi inutilizzabile (corr col MAF = 0,004) → è il motivo per cui il progetto
**non** poggia più sul terreno.

### `accel` con mediana 0 è un dato sbagliato?
**No, è corretto e atteso.** Per gran parte del tempo l'auto va a **velocità ~costante** (crociera o
ferma) → accelerazione ~0. Quindi è normale che *almeno metà* degli istanti abbia accel ≈ 0. L'azione
sta nelle **code** (p5 = −6,95, p95 = +7,5). La media è ~0 perché accelerazioni e frenate si bilanciano
nel tempo (non puoi accelerare all'infinito).

### `Generalized_Weight` è in libbre?
**Sì, quasi certamente in libbre (lb).** Valori 2500–6000, mediana 3500. In kg sarebbe assurdo (3500
kg per un'auto). In libbre: 3500 lb ≈ **1588 kg** = peso normalissimo di un'auto; 6000 lb ≈ un
SUV/pickup. È un peso "generalizzato" (a fasce), non al kg esatto.

### Quando la distanza è 0 (veicolo fermo) non è meglio scartarli?
Non in blocco, e per scelta:
- Il NB1 **ha già** tolto le righe davvero inerti (tiene solo `speed>0` **o** `rpm>100`, cioè motore
  acceso).
- Le righe "ferme ma motore acceso" (semafori, traffico) le **teniamo apposta**: al minimo si consuma
  comunque, e soprattutto lo **stop-and-go è il fattore numero 1 del consumo per km** (la feature
  `stop_fraction` è risultata la più importante nel modello!). Buttarle cancellerebbe il segnale più
  utile.

---

## C) Gli ibridi (il punto delicato)

### Come viene gestito il periodo in cui le ibride vanno in elettrico?
Quando una HEV/PHEV va in elettrico, **il motore termico è spento** → `MAF = 0` e RPM bassi/zero,
**mentre il veicolo si muove e consuma energia (dalla batteria)**. Il problema: **il VED non registra
la batteria** (nessun segnale elettrico). Quindi quel consumo è **invisibile**. Lo gestiamo così:
1. **Modello di consumo (NB2) → solo ICE**, dove MAF = carburante è onesto.
2. **NB3:** li descriviamo (quanto vanno in elettrico, frenata rigenerativa) invece di forzarli in un
   modello di consumo.

### I calcoli sul MAF medio non sono "sballati" dal fatto che si spengono?
**Esatto, e l'hai colto.** Il MAF medio degli ibridi è *trascinato in basso* dai momenti a motore
spento (MAF=0), che **non** significano "consuma poco", ma "sta andando in elettrico". Quindi
confrontare il MAF medio tra powertrain come fosse "consumo" è fuorviante. Proprio per questo: il
modello di consumo è **solo-ICE**, e il confronto powertrain è **descrittivo** (con il caveat del
motore spento), mai presentato come "l'HEV consuma meno benzina".

---

## D) Il modello

### Cosa fa il modello, cosa predice, con quali feature?
Il modello di consumo (**NB2**) predice il **`maf_per_km`** di un **tratto di strada da ~250 m**:
cioè *quanta aria (≈ carburante) serve per percorrere quel tratto, per chilometro* — una misura di
**efficienza**.

- **Input (feature):** solo cose note **in anticipo** →
  - profilo di velocità pianificato: velocità media/max/min/std, velocità d'ingresso, accelerazione,
    frazione di sosta;
  - geometria del tratto: lunghezza, dislivello, pendenza;
  - **look-ahead**: cosa c'è nel tratto successivo (dislivello/sosta in arrivo);
  - contesto: peso, temperatura, ora.
- **NON usa** RPM, carico (Load) né il MAF come input.
- **Output:** il consumo per km predetto per quel tratto.

**A cosa serve:** un cruise control può chiedere al modello *"quanto mi costa questo tratto se lo
guido così?"* e confrontare strategie di guida diverse per scegliere la più economica.

### Cosa vuol dire "map-only"?
Vuol dire: **il modello usa solo informazioni che un assistente di guida conoscerebbe in anticipo**
(dalla mappa + dal profilo di velocità che ha pianificato), e **non** i segnali che il motore produce
*mentre* guida (RPM, carico, MAF).

Perché? Un ACC predittivo deve valutare un viaggio **ipotetico futuro** ("e se rallentassi qui?"): in
quello scenario gli RPM futuri **non esistono ancora** (dipendono da come guiderai). Quindi un modello
che ha bisogno degli RPM funziona solo *a posteriori*, non come pianificatore. "Map-only" = uso solo
feature **prevedibili in anticipo**. (E in più, RPM/Load sono quasi una copia del MAF — corr 0,75/0,52
— quindi usarli sarebbe "barare".)

---

## E) Concetti statistici veloci

### Cos'è lo `skew` (asimmetria)?
Misura quanto una distribuzione è **storta**:
- skew = 0 → simmetrica;
- skew > 0 → **coda lunga a destra** (tanti valori piccoli, pochi molto grandi che tirano su la media).

Il MAF ha **skew 1,91** = molto storto a destra → la media (9,85) è più alta della mediana (5,58). Per
questo esiste `log_MAF` (il logaritmo "raddrizza" la distribuzione, utile per i modelli lineari).

---

## F) Implicazioni per il modeling (spiegate)

Ogni numero dell'EDA giustifica una scelta:

1. **RPM e Load correlano 0,75 e 0,52 col MAF** → sono quasi la formula del MAF → **esclusi**
   (modello map-only). Includerli darebbe un R² più alto ma "finto".
2. **Slope ~0 nel 92% delle righe, corr 0,004** → il terreno non è un segnale utile → progetto
   reinquadrato sul **profilo di velocità**; terreno = limite del dato.
3. **Dati per veicolo sbilanciatissimi** (un'auto ha 822k righe, un'altra 394) → nel tuning si usa
   **GroupKFold per VehId**, così lo stesso veicolo non sta in train e validation insieme (altrimenti
   il modello "lo riconosce" e i risultati sembrano migliori di quanto sono).
4. **MAF non valido per gli ibridi** (motore spento 21–24% in marcia) → modello di consumo **solo-ICE**;
   ibridi trattati a parte.
5. **PHEV = solo 12 veicoli** → le conclusioni sui PHEV sono **indicative**, da dichiarare.
6. **Buchi di campionamento** (Δt fino a ~2 h) → nell'integrazione del consumo il Δt va **cappato**
   (a 2 s) per non sommare "aria fantasma".
7. **MAF asimmetrico** (skew 1,91) → disponibile `log_MAF` se serve un target più simmetrico.
8. **Prima riga di ogni trip** ha `dist_m`/`dt_ms` mancanti → vanno **azzerati** prima di
   cumulare/integrare (altrimenti i conti si rompono).
9. **Lo stop-and-go è il driver principale** del consumo per km → le righe "ferme ma motore acceso"
   si **tengono** (non si scartano): sono informazione, non rumore.
