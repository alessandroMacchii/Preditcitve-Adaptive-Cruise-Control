# Storia del progetto — come si è arrivati al modello "map-only"

> **Nota (reframe 2026-06-14):** questo documento resta valido — la scelta map-only è ancora il
> cuore dei modelli di consumo. Ma l'inquadramento generale del progetto è cambiato da "ACC che
> sfrutta l'orografia" a *"ML per l'energia e il contesto di guida"* (il terreno si è rivelato un
> segnale debole). Quadro aggiornato in `RELAZIONE_PROGETTO.md`.

> Documento di accompagnamento alla discussione. Racconta **perché** il modello del
> Notebook 2 è fatto così, partendo dall'idea iniziale, passando per un problema
> scoperto guardando i dati, fino alla scelta finale. I numeri citati sono reali.

---

## 1. L'idea di partenza

Il progetto nasce da una domanda concreta: un **cruise control adattivo predittivo**
— a differenza di quello tradizionale a velocità fissa — conosce in anticipo la strada
che ha davanti (dalle mappe: salite, discese) e potrebbe *anticipare*, modulando la
velocità per consumare meno.

Per costruirne il "cervello" serve un modello che risponda a una domanda:

> dato lo stato del veicolo e ciò che la strada/guida faranno nei prossimi secondi,
> **quanto sforzo (consumo) costa?**

Come misura dello sforzo usiamo il **MAF** (`MAF_g_per_sec`, massa d'aria aspirata dal
motore), che è un proxy diretto del carburante bruciato. Il dataset è il **Vehicle
Energy Dataset (VED)**: ~18 milioni di righe di telemetria reale raccolta ad Ann Arbor
(Michigan), arricchita nel Notebook 1 con l'altitudine (Open-Meteo/SRTM) e quindi con
la **pendenza** della strada.

## 2. Il primo modello (e un ottimo R² sospetto)

Il primo approccio è stato naturale: dare al modello **tutte** le variabili disponibili,
compresi i giri motore (`Engine_RPM_RPM`) e il carico (`Absolute_Load_pct`), insieme a
velocità, accelerazione, pendenza e alle feature *look-ahead* (pendenza e velocità
nei prossimi 5/10/30 campioni).

Risultato, con XGBoost tunato (split temporale, test su trip futuri):

| Metrica | Valore |
|---|---|
| R² | **0.84** |
| MAE | 2.29 g/s |

Un R² di 0.84 sembra un successo. Ma guardando **da dove** veniva, è emerso un problema.

## 3. Il problema: il modello prendeva una scorciatoia

La **feature importance** del modello "con motore" racconta tutto:

| Feature | Importanza |
|---|---|
| `Engine_RPM_RPM` | **0.53** |
| `Generalized_Weight` | 0.12 |
| `speed_delta_future_5` | 0.08 |
| `rpm_roll10s_mean` | 0.05 |
| `Absolute_Load_pct` | 0.05 |
| ... | ... |
| **tutte le feature di pendenza sommate** | **0.007** |

Tre fatti, tutti problematici:

1. **Gli RPM da soli valgono metà del modello** (53%); i segnali-motore insieme
   (RPM + carico + media RPM) pesano **~64%**.
2. **La pendenza — il cuore concettuale del progetto — è irrilevante**: tutte le sue
   varianti sommate fanno lo 0,7%. Anche Lasso le azzerava.
3. Persino la velocità grezza finiva in fondo alla classifica (0,004).

### Perché succede

Non è un bug: è **fisica**. La massa d'aria aspirata è quasi proporzionale ai giri
motore (≈ cilindrata × RPM × efficienza volumetrica), e il MAF è essenzialmente
RPM × carico. Dare in input RPM e carico per predire il MAF è come chiedere di
"predire" il prezzo totale dando prezzo unitario e quantità: il modello **inverte una
formula**, non impara una relazione del mondo. E poiché gli RPM hanno già *incorporato*
la reazione del motore alla salita, la pendenza diventa ridondante e sparisce.

### Perché è un problema serio, non estetico

Oltre a gonfiare l'R², gli RPM e il carico hanno un difetto fatale per il caso d'uso:
**non sono noti in anticipo**. Un cruise control predittivo deve valutare un profilo di
velocità *ipotetico* ("e se rallentassi prima della salita?"); per quello scenario gli
RPM futuri non esistono — dipendono da marcia e acceleratore, cioè dalla guida che il
sistema sta ancora decidendo. Un modello che richiede gli RPM in input funziona solo
*a posteriori*: è un misuratore, non un pianificatore.

La prova del nove è l'**esperimento controfattuale**: simulando una guida "eco" (−10%
di velocità) sullo stesso tratto, il modello con motore prevedeva un risparmio di
appena **−1.2%**. Ridicolo — perché scalavamo la velocità ma lasciavamo *congelati*
RPM e carico, cioè il 64% del modello. Il controfattuale era di fatto rotto.

## 4. La decisione: modello "map-only"

La correzione è stata **togliere i segnali-motore** (`Engine_RPM_RPM`,
`Absolute_Load_pct`, `rpm_roll10s_mean`) e tenere solo ciò che un cruise control
conosce davvero in anticipo:

- velocità e accelerazione;
- **pendenza** attuale e futura (dalla mappa);
- **velocità futura** e delta di velocità previsto (il profilo di guida pianificato);
- contesto: peso, tipo motore, ora/mese, temperatura.

Da qui il nome: il modello usa solo informazione derivabile da **mappa** + **profilo
di velocità**.

## 5. Il risultato: un R² più basso ma onesto

Confronto diretto (XGBoost, stesso campione, stesse condizioni):

| Modello | R² | MAE | Feature dominante |
|---|---|---|---|
| Con motore (RPM + carico) | ~0.82 | 2.60 | `Engine_RPM_RPM` (0.57) |
| **Map-only** | **~0.54** | 4.90 | `speed_delta_future_5` (0.34) |

L'R² **cala da ~0.82 a ~0.54**. Non è un peggioramento: è la **misura onesta** di quanto
il modello precedente barava. Quel calo quantifica esattamente quanto valeva la
scorciatoia degli RPM.

E soprattutto, nel map-only **le feature giuste tornano protagoniste**. Importanza:

| Feature | Importanza |
|---|---|
| `speed_delta_future_5` (look-ahead) | 0.34 |
| `speed_future_10` (look-ahead) | 0.29 |
| `accel_kmh_s` | 0.10 |
| `Generalized_Weight` | 0.08 |
| ... | ... |

Le feature **look-ahead** sul profilo di velocità, da sole, valgono circa **i due terzi**
del modello. È esattamente la tesi del progetto: il valore sta nell'*anticipazione*, ed
è quella che il modello ora sfrutta. Il controfattuale eco/sport, con le feature scalate
in modo coerente, diventa finalmente sensato.

## 6. Un'onestà in più: e la pendenza?

C'è una sfumatura da non nascondere. Anche nel modello map-only la **pendenza resta
debole**: a dominare è il look-ahead sulla *velocità*, non sulla *pendenza*. Perché?

- **Ann Arbor è quasi pianeggiante**: le pendenze reali sono piccole, quindi c'è poco
  segnale da cui imparare.
- **La risoluzione dell'altitudine** (SRTM ~30 m, coordinate arrotondate a ~111 m) leviga
  le pendenze su tratti brevi.
- La risposta del veicolo alla salita si manifesta comunque attraverso
  accelerazione e variazione di velocità, che il modello *già* vede.

Non è una debolezza del metodo, ma un limite del **dato**: su un terreno collinare o
montano la pendenza avrebbe quasi certamente un ruolo maggiore. È un'ottima base per la
sezione "limiti e sviluppi futuri".

## 7. In sintesi (per la discussione)

- Partiti da un modello con R² 0.84 che sembrava ottimo.
- Scoperto, guardando la feature importance, che viveva al 64% sui segnali-motore e
  ignorava la strada.
- Capito che quei segnali sono (a) quasi-circolari col target e (b) non disponibili in
  anticipo → il modello non era utilizzabile in un cruise control predittivo.
- Tolti i segnali-motore → R² ~0.54, **onesto**, con le feature look-ahead finalmente
  centrali e un controfattuale che funziona.
- Documentato un limite reale del dataset (terreno piatto) come spunto futuro.

> Il risultato non è "l'R² più alto possibile", ma **un modello coerente con il problema
> reale** e la consapevolezza di cosa stava facendo quello precedente. È questa la parte
> di metodo del progetto.
