# Notebook di Studio — Machine Learning

> Guida ragionata a tutti gli argomenti del corso. L'obiettivo non è elencare definizioni, ma capire **perché** si fanno certe scelte e **quando** un approccio batte un altro. Ogni sezione chiude con un riquadro decisionale "Quando usarlo".

---

## Indice

1. [Cos'è il Machine Learning e i paradigmi di apprendimento](#1)
2. [Gli strumenti: NumPy e pandas (e perché si vettorizza)](#2)
3. [Il workflow di un progetto ML e i concetti trasversali](#3)
4. [Loss, funzione di costo e metriche](#4)
5. [Overfitting, underfitting e il trade-off bias–varianza](#5)
6. [Regressione lineare](#6)
7. [Codifica delle variabili categoriche](#7)
8. [Feature engineering: interazioni, feature polinomiali, collinearità](#8)
9. [Standardizzazione: quando serve davvero](#9)
10. [Regolarizzazione L1 e L2](#10)
11. [Data leakage e Pipeline](#11)
12. [Classificazione: la regressione logistica](#12)
13. [Metriche per la classificazione (il cuore del "quando")](#13)
14. [La soglia di decisione (threshold)](#14)
15. [Alberi decisionali e Random Forest](#15)
16. [Gradient Boosting e XGBoost](#16)
17. [Model selection: Grid, Random e Bayesian search](#17)
18. [Clustering: K-Means](#18)
19. [Riduzione della dimensionalità: PCA](#19)
20. [Visualizzazione: PCA vs t-SNE vs UMAP](#20)
21. [Reti neurali e autoencoder](#21)
22. [Anomaly detection con autoencoder](#22)
23. [Mappa decisionale finale: "quando uso cosa"](#23)

---

<a name="1"></a>
## 1. Cos'è il Machine Learning e i paradigmi di apprendimento

Un programma tradizionale è una regola scritta a mano: input → regole → output. Il machine learning ribalta la logica: dai al sistema **input e output** e lui ricava le regole da solo. Questo è utile quando le regole sono troppe, troppo sottili o cambiano nel tempo (riconoscere una cifra scritta a mano: nessuno saprebbe scrivere le centinaia di `if` necessari).

### I quattro paradigmi

| Paradigma | Cosa riceve | Cosa impara | Esempio |
|---|---|---|---|
| **Supervisionato** | input + etichetta corretta | a mappare input → output | prezzo di una casa, spam/non spam |
| **Non supervisionato** | solo input, niente etichette | struttura nascosta nei dati | clustering clienti, anomaly detection |
| **Self-supervised** | input che genera da solo la propria etichetta | come il supervisionato ma senza annotazione umana | next-token prediction degli LLM |
| **Rinforzo** | un ambiente e una ricompensa | una strategia (policy) che massimizza il reward nel tempo | un agente che gioca, un robot |

**Perché distinguerli conta.** La scelta del paradigma è la *prima* decisione e dipende interamente da quali dati hai. Se hai etichette → supervisionato. Se non le hai (o costano troppo) → non supervisionato. Il self-supervised è la grande intuizione dietro i modelli moderni: trasforma dati non etichettati in un problema supervisionato gratis (la "risposta" è una parte dell'input stesso, es. la parola successiva).

### Struttura dei dati

- **Strutturati**: tabelle (righe = esempi, colonne = feature). Qui dominano regressione lineare, alberi, boosting.
- **Non strutturati**: immagini, audio, testo. Qui dominano le reti neurali, che imparano da sole le feature rilevanti.

Nel supervisionato il dataset si scrive come matrice delle **feature** $X$ (le variabili in ingresso, "i segnali") e vettore dei **target** $y$ (ciò che vogliamo prevedere). Ogni riga di $X$ è un esempio, ogni colonna una feature.

### Tipi di feature

- **Numeriche**: quantità (altezza, prezzo).
- **Categoriche nominali**: gruppi senza ordine (città, colore).
- **Categoriche ordinali**: gruppi con ordine (titolo di studio, soddisfazione 1–5).

Questa distinzione non è pedante: determina **come codificare** la variabile (sezione 7) e quali assunzioni il modello finirà per fare.

> **Quando usarlo** — La domanda di partenza è sempre: *ho le risposte (etichette) o no?* Da lì discende tutto il resto.

---

<a name="2"></a>
## 2. Gli strumenti: NumPy e pandas (e perché si vettorizza)

### NumPy: l'array omogeneo

Un `ndarray` è una struttura N-dimensionale **omogenea** (tutti gli elementi dello stesso tipo) e a dimensione fissa. Questa rigidità è esattamente ciò che lo rende veloce: i dati stanno in un blocco contiguo di memoria e le operazioni vengono eseguite da codice C compilato.

Concetti chiave del tutorial:

- **Slicing N-dimensionale**: `a[riga, colonna]`, con `:` per "tutto".
- **Slice assignment**: `a[:, 0] = 5` sovrascrive in blocco.
- **Fancy indexing booleano**: una maschera `a[a > 0]` seleziona solo gli elementi che soddisfano la condizione. È il modo "vettorizzato" di fare un filtro.
- **Operazioni elementwise (ufunc)**: `a + b`, `np.exp(a)` agiscono su ogni elemento senza scrivere cicli.
- **Broadcasting**: NumPy "allinea" automaticamente array di forme diverse seguendo due regole — (1) le dimensioni mancanti vengono aggiunte a sinistra, (2) le dimensioni di taglia 1 vengono espanse. Così puoi sommare uno scalare a una matrice, o un vettore riga a ogni riga di una matrice, senza copiare nulla.
- **Riduzioni**: `sum`, `mean`, `any`, `all` con `axis` collassano l'array lungo un asse.
- **Reshape**: restituisce una *view* (stessa memoria, nuova forma) quando possibile — economico ma occhio agli effetti collaterali se modifichi.
- **View vs copia**: uno slice è una view (modificarlo modifica l'originale); `np.copy` crea dati indipendenti.

### Perché vettorizzare: il caso del Game of Life

Il file `life.py` mostra due implementazioni della stessa regola (Conway's Game of Life):

1. **Python puro**: due cicli `for` annidati su una griglia $N \times N$ → complessità $O(N^2)$ con overhead dell'interprete a ogni cella.
2. **NumPy + convoluzione 2D**: il conteggio dei vicini di *tutte* le celle in una sola operazione, con un kernel 3×3, e l'applicazione delle regole con un unico `np.where`.

La versione vettorizzata è molto più veloce per tre motivi: il loop gira in **C compilato** (non nell'interprete Python), i dati sono **contigui** in memoria (cache-friendly), e le operazioni sfruttano **istruzioni SIMD** della CPU. La lezione generale: *ogni volta che scrivi un ciclo `for` su un array, chiediti se esiste l'equivalente vettorizzato*. In ML i dataset hanno milioni di righe e la differenza tra le due versioni è tra "secondi" e "ore".

### pandas: l'analisi tabellare

Il `DataFrame` è la tabella centrale. Punti chiave del tutorial:

- **Accesso**: `df.colonna` o `df["colonna"]` (la seconda è obbligatoria se il nome ha spazi o conflitti).
- **`.loc` (per etichetta) vs `.iloc` (per posizione)**: distinzione cruciale per non sbagliare riga.
- **Maschere booleane** e `.query("price < 1e6")` per filtrare.
- **L'allineamento per indice**: ogni Series/DataFrame ha un Index; le operazioni si allineano *per etichetta*, non per posizione. È potente ma è una fonte classica di bug silenziosi.
- **GroupBy** (split → apply → combine): raggruppa, applica un'aggregazione, ricombina.
- **Dati mancanti** (`NaN`): si possono eliminare (`dropna`, ma perdi righe) o **imputare** (riempire con media/mediana/moda). L'imputazione è preferibile quando dropping ti farebbe perdere troppi dati.

> **Quando usarlo** — pandas per esplorare, pulire e fare feature engineering su dati tabellari; NumPy per il calcolo numerico puro e ogni volta che la performance conta. Regola d'oro: **niente cicli Python su grandi array, sempre operazioni vettorizzate**.

---

<a name="3"></a>
## 3. Il workflow di un progetto ML e i concetti trasversali

Il notebook sulla regressione immobiliare (King County) mostra il flusso end-to-end che si ripete in quasi ogni progetto:

1. **EDA (Exploratory Data Analysis)**: capire i dati con istogrammi, box plot, conteggi. Serve a scoprire anomalie e relazioni *prima* di modellare.
2. **Data cleaning**: gestire valori anomali (case a $0 o a $10M con superficie minuscola) e mancanti.
3. **Feature engineering**: costruire/selezionare le variabili giuste.
4. **Train/test split**: separare i dati per una valutazione onesta.
5. **Modello + predizioni**: addestrare e prevedere.
6. **Valutazione**: metriche + analisi dei residui.

### Perché si separano training e test

Il modello deve **generalizzare** a dati mai visti, non memorizzare quelli di addestramento. Il test set è la simulazione del "mondo reale": lo si tocca **solo alla fine**. Se lo usi per prendere decisioni, le tue stime di performance diventano ottimisticamente false.

Per le decisioni *durante* lo sviluppo (quale modello, quale iperparametro) serve un terzo blocco, il **validation set**:

- **Training set** → addestra i parametri.
- **Validation set** → confronta modelli e sceglie iperparametri.
- **Test set** → valutazione finale imparziale, una volta sola.

`random_state` (es. `=42`) fissa il seme casuale dello split: garantisce **riproducibilità** (stessi risultati a ogni esecuzione).

> **Quando usarlo** — Sempre. La regola "il test set non si tocca finché non hai finito" è metodologia non negoziabile: violarla è il modo più comune per illudersi che un modello sia buono.

---

<a name="4"></a>
## 4. Loss, funzione di costo e metriche

Tre concetti che si confondono facilmente ma servono a cose diverse.

- **Funzione di loss**: misura l'errore su **un singolo esempio**.
- **Funzione di costo**: la **media** delle loss su tutto il training set. È ciò che l'algoritmo **minimizza** durante l'addestramento.
- **Metrica**: un indice **per noi umani**, calcolato a valle, per giudicare quanto è buono il modello. Non è (necessariamente) ciò che il modello ottimizza.

La distinzione chiave: la **loss deve essere ottimizzabile** (derivabile, smussa), la **metrica deve essere interpretabile**. A volte coincidono, spesso no. Esempio: una rete di classificazione minimizza la cross-entropy (loss) ma noi la giudichiamo con l'accuracy o l'F1 (metriche), perché "percentuale di errori" non è derivabile e quindi non si può ottimizzare direttamente con la discesa del gradiente.

### Loss tipiche

- **MSE (Mean Squared Error)** — regressione:
  $$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
  Eleva al quadrato l'errore → penalizza molto gli errori grandi. È smussa e derivabile, ideale per l'ottimizzazione.

- **Cross-Entropy** — classificazione:
  $$\mathcal{L} = -\log(\hat{y}_{\text{classe corretta}})$$
  Misura quanto la distribuzione di probabilità predetta si discosta da quella vera. Cresce verso infinito quando il modello assegna probabilità bassa alla classe giusta → punisce duramente la sicurezza sbagliata.

### Metriche di regressione

- **MAE (Mean Absolute Error)**: errore medio assoluto, nella stessa unità del target ($ per le case). Facile da spiegare a un non addetto.
- **MAPE (Mean Absolute Percentage Error)**: errore in percentuale, utile per comunicare il margine relativo ("sbagliamo in media del 18%").
- **R² (coefficiente di determinazione)**: quota di varianza del target spiegata dal modello.
  $$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$$
  Va da $0$ a $1$ (può essere negativo se il modello è peggio della media). $R^2 = 1$ → perfetto; $R^2 = 0$ → non fa meglio del predire sempre la media.

**MAE vs MSE — quando usare quale.** MSE/RMSE se gli errori grandi sono particolarmente costosi (vuoi punirli di più) e per ottimizzare; MAE se vuoi una misura robusta agli outlier e facile da interpretare. MAPE quando l'errore relativo conta più di quello assoluto (ma attenzione: esplode se i valori veri sono vicini a zero).

> **Quando usarlo** — Scegli la **loss** in base al tipo di problema (MSE per regressione, cross-entropy per classificazione). Scegli la **metrica** in base a cosa devi comunicare e a quale errore ti fa più male.

---

<a name="5"></a>
## 5. Overfitting, underfitting e il trade-off bias–varianza

Questo è il concetto centrale di tutto il ML supervisionato.

- **Underfitting**: il modello è troppo semplice per catturare la struttura dei dati. Errore alto *sia* su training *sia* su test. Ha **alto bias** (errore sistematico: assume una forma sbagliata, es. una retta per dati curvi).
- **Overfitting**: il modello è troppo complesso e "impara a memoria" il training set, rumore compreso. Errore basso su training ma alto su test. Ha **alta varianza** (è ipersensibile ai dati specifici visti).

### Il trade-off

| | Bias | Varianza | Sintomo |
|---|---|---|---|
| Modello troppo semplice | alto | bassa | underfitting |
| Modello troppo complesso | basso | alta | overfitting |
| Modello giusto | equilibrio | equilibrio | buona generalizzazione |

L'esperimento delle feature polinomiali lo mostra benissimo: all'aumentare del grado del polinomio, l'errore di **training scende sempre**, ma quello di **test prima scende e poi risale**. Il punto di minimo del test error è il modello che generalizza meglio. Il "gap" tra test error e train error misura l'overfitting.

**Perché succede.** Un modello con tanti parametri ha abbastanza "libertà" per passare esattamente per ogni punto di training, inclusi quelli rumorosi. Ma il rumore è casuale: non si ripresenta nel test set, quindi quella precisione è inutile o dannosa fuori dal training.

**Rasoio di Occam**: a parità di risultati, il modello più semplice è preferibile. La regolarizzazione (sezione 10) è la formalizzazione di questo principio.

### Strategie contro l'overfitting

- Scegliere la complessità giusta (grado del polinomio, profondità dell'albero) via validation/cross-validation.
- Regolarizzazione (L1/L2).
- Più dati di training (più è difficile memorizzarli tutti).
- Monitorare *sempre* il test/validation error, non solo il training.

> **Quando preoccuparsi** — Se train error ≈ test error ma entrambi alti → underfitting, aumenta la complessità. Se train error basso ma test error alto → overfitting, semplifica o regolarizza. Un modello leggermente in underfitting è spesso più sicuro di uno in overfitting.

---

<a name="6"></a>
## 6. Regressione lineare

È il punto di partenza di tutto il ML supervisionato perché è semplice, interpretabile e veloce.

### L'idea

Ipotizza una relazione **lineare** tra feature e target. Nel caso univariato è una retta:
$$\hat{y} = w_1 x + b$$
dove $w_1$ è il coefficiente (pendenza, "quanto cambia $y$ per ogni unità di $x$") e $b$ è il bias/intercetta ("valore di $y$ quando $x=0$"). Nel caso multivariato:
$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

### Come "impara"

Minimizza la funzione di costo MSE: cerca i coefficienti che rendono minima la somma dei quadrati dei residui ($y_i - \hat{y}_i$). Geometricamente, trova la retta/iperpiano che passa "più in mezzo" possibile alla nuvola di punti.

### Interpretazione dei coefficienti

Ogni $w_i$ è l'**effetto marginale** della feature $x_i$ su $y$, *a parità di tutte le altre* (clausola *ceteris paribus*). Esempio del notebook: il coefficiente di `sqft_living` dice di quanti dollari aumenta il prezzo per ogni piede quadrato in più. Questa interpretabilità è il grande vantaggio della regressione lineare rispetto a modelli "black box".

### Analisi dei residui (la parte che spesso si salta)

I residui (reale − predetto) raccontano se le **assunzioni** del modello reggono. Se il modello è buono, i residui dovrebbero essere:

- centrati intorno a 0 (nessun bias sistematico),
- a varianza costante (omoscedastici — niente "imbuto"),
- senza pattern (se vedi una curva nei residui, la relazione vera non era lineare),
- distribuiti circa normalmente.

Quando i residui mostrano un imbuto o una curva, è un segnale che serve una trasformazione. Nel notebook immobiliare, applicare il **logaritmo al prezzo** (`np.log(price)`) rende i residui più normali e gestisce il fatto che gli errori crescono con il prezzo: un modello che sbaglia di $50k su una casa da $5M è ottimo, su una da $100k è pessimo. Modellare $\log(\text{prezzo})$ trasforma gli errori in *percentuali*, più sensate per i prezzi.

### Progressione del notebook (perché aggiungere feature aiuta)

Il notebook costruisce il modello a strati e ogni strato migliora R²:
1. solo `sqft_living` → modello povero;
2. + bedrooms, bathrooms, condition, waterfront, view → meglio;
3. + interazioni (sezione 8);
4. + `statezip` con one-hot encoding → la geografia spiega tantissimo del prezzo;
5. + log del target → residui più sani.

> **Quando usarla** — Come **baseline** sempre: è veloce, interpretabile, e ti dice subito se il problema è "facile". È la scelta giusta quando ti serve spiegare *perché* il modello prevede una certa cosa (es. contesti regolatori, business). Quando la relazione è chiaramente non lineare o ci sono molte interazioni complesse, passa a feature polinomiali o, meglio, a modelli ad albero/boosting.

---

<a name="7"></a>
## 7. Codifica delle variabili categoriche

I modelli matematici vogliono numeri. Le variabili qualitative (colore, città) vanno trasformate, e *come* le trasformi cambia le assunzioni che il modello farà.

### Ordinal Encoding

Assegna un intero a ogni categoria (Basso=1, Medio=2, Alto=3).

**Il problema**: introduce un'assunzione fortissima sulla *distanza*. Il modello tratta la differenza 1→2 come identica a 2→3, e assume un *ordine*. Per variabili davvero ordinali (titolo di studio) può andare; per variabili **nominali** ("Rosso"=1, "Verde"=2, "Blu"=3) è un disastro: stai dicendo al modello che Blu > Verde > Rosso e che Blu è "tre volte" Rosso, una relazione inesistente che lo porta a cercare pattern lineari fasulli.

### One-Hot Encoding

Crea una colonna binaria (0/1) per ogni categoria. "Genere = {M, F, Altro}" → tre colonne, ognuna 1 solo per la riga giusta. Niente ordine artificiale, niente distanze inventate. È lo standard per le variabili **nominali**.

**La regola del "drop one"** (variabili dummy): se una variabile ha $k$ categorie, ne includi solo $k-1$. Perché? Intuitivamente per **ridondanza**: se non sei M e non sei Altro, allora sei F per forza — la terza colonna non aggiunge informazione. Tecnicamente, includere tutte le colonne crea **multicollinearità perfetta** (la loro somma è sempre 1, identica all'intercetta): la matrice diventa non invertibile (singolare) e i coefficienti della regressione lineare non sono calcolabili. La categoria esclusa diventa il **livello di riferimento** (baseline) e ogni coefficiente dummy si interpreta come "differenza media di $y$ rispetto alla baseline".

> **Nota importante**: il drop-one serve per i modelli **lineari**. Per gli alberi/boosting non è necessario (non soffrono di multicollinearità allo stesso modo).

### Target Encoding

Sostituisce ogni categoria con la **media del target** per quella categoria.

- **Pro**: una sola colonna anche per variabili con centinaia di categorie (riduce la dimensionalità). Utile per feature ad alta cardinalità (es. CAP, codici prodotto) dove il one-hot esploderebbe in migliaia di colonne.
- **Contro**: rischio altissimo di **overfitting/data leakage**, perché stai infilando informazione sul target dentro la feature. Va fatto con cautela (calcolato solo sul training, con smoothing o cross-fitting).

### Tabella decisionale

| Tecnica | Quando | Rischio |
|---|---|---|
| **Ordinal** | variabile *davvero* ordinale (poche categorie con ordine reale) | impone distanze/ordini falsi su variabili nominali |
| **One-Hot** | variabile nominale a **bassa** cardinalità | esplosione di colonne se le categorie sono tante |
| **Target** | variabile nominale ad **alta** cardinalità | overfitting / leakage se fatto male |

> **Quando usarlo** — Default per le nominali: **One-Hot** (con drop-one se il modello è lineare). Ordinal *solo* se l'ordine è reale e significativo. Target encoding solo quando la cardinalità è troppo alta per il one-hot e sai gestire il leakage.

---

<a name="8"></a>
## 8. Feature engineering: interazioni, feature polinomiali, collinearità

"I dati sono argilla; il feature engineering è l'arte di modellarli." Spesso migliora le performance più del cambio di modello.

### Termini di interazione

Un modello **additivo** assume che l'effetto di una variabile sia *indipendente* dalle altre. Ma nella realtà spesso interagiscono. Esempio del PDF: il prezzo di una casa in funzione di superficie e stato ("da ristrutturare" vs "nuova"). Un modello additivo:
$$\text{Prezzo} = \beta_0 + \beta_1 \text{Superficie} + \beta_2 \text{StatoNuovo}$$
assume che passare a "Nuova" aggiunga un valore *fisso*, uguale per un monolocale e per una villa. Ma intuitivamente la ristrutturazione vale molto di più su 500 m² che su 30 m². Si aggiunge allora un **termine di interazione** (il prodotto):
$$\text{Prezzo} = \beta_0 + \beta_1 \text{Superficie} + \beta_2 \text{StatoNuovo} + \beta_3 (\text{Superficie} \times \text{StatoNuovo})$$
Ora la *pendenza* (l'importanza della superficie) cambia a seconda dello stato. Nel notebook immobiliare questa idea diventa `sqft_above * waterfront`, `sqft_above * view`, e l'interazione tra zona (`statezip`) e superficie — il prezzo al m² dipende dal quartiere.

### Feature polinomiali

Trasformano una feature in $[x, x^2, x^3, \dots]$. Questo permette a un modello **lineare** di adattarsi a curve non lineari: il modello resta lineare nei *coefficienti*, ma non più nelle feature. Esempio: per dati che seguono una parabola, $\hat{y} = w_0 + w_1 x + w_2 x^2$ si adatta perfettamente dove una retta falliva.

**Attenzione al grado**: è la leva di complessità che genera overfitting. Grado troppo basso → underfitting; troppo alto → il modello serpeggia tra i punti di training e crolla sul test. Il grado giusto si sceglie con la cross-validation.

**Perché standardizzare prima**: $x^{10}$ con $x=15$ è un numero gigantesco → problemi numerici. Standardizzare ($x \to$ media 0, std 1) tiene le potenze in un range gestibile.

### Collinearità e multicollinearità

Si verifica quando due o più feature sono fortemente correlate tra loro.

- **Problema**: il modello non riesce a isolare l'effetto individuale di ciascuna (se due variabili si muovono insieme, "chi delle due" causa l'effetto?).
- **Conseguenza**: i coefficienti della regressione lineare diventano **instabili**, con varianza elevata, e impossibili da interpretare con fiducia.
- Caso limite: la **dummy variable trap** (sezione 7), multicollinearità *perfetta*.

La regolarizzazione L2 (Ridge) è un buon rimedio alla collinearità, perché stabilizza i coefficienti.

> **Quando usarlo** — Aggiungi interazioni quando hai motivo di credere che l'effetto di una variabile dipenda da un'altra (quasi sempre nel mondo reale). Usa feature polinomiali quando la relazione è curva ma vuoi restare con un modello lineare/interpretabile. Tieni d'occhio la collinearità appena hai feature correlate o many-dummy.

---

<a name="9"></a>
## 9. Standardizzazione: quando serve davvero

Lo `StandardScaler` applica lo Z-score: $z = \frac{x - \mu}{\sigma}$, portando ogni feature a media 0 e deviazione standard 1.

**Perché farlo**: molti algoritmi ragionano in termini di *distanze* o di *scala dei coefficienti*. Se una feature va da 0 a 1.000.000 e un'altra da 0 a 5, la prima domina i calcoli solo perché ha numeri più grandi, non perché sia più importante.

### Dove è indispensabile

- **K-Means** e tutto ciò che usa distanze euclidee (senza scaling, la feature con range più ampio decide i cluster).
- **PCA** (cerca direzioni di massima varianza: senza scaling, la feature con varianza numerica più alta vince a prescindere).
- **Regressione regolarizzata (Ridge/Lasso)**: la penalità sui coefficienti è sensibile alla scala.
- **SVM, reti neurali**: gradienti più stabili, convergenza più rapida (es. dividere i pixel per 255 in MNIST).
- **Regressione logistica**: sensibile alla scala (nel notebook è dentro la pipeline).

### Dove NON serve (o quasi)

- **Alberi decisionali, Random Forest, XGBoost**: splittano su soglie di singole feature ("sqft > 1500?"), quindi sono **invarianti a trasformazioni monotone**. Standardizzare non cambia nulla. È uno dei motivi per cui i modelli ad albero sono comodi: meno preprocessing.

### La regola anti-leakage (fondamentale)

Si fa `fit` dello scaler **solo sul training set** (calcola $\mu$ e $\sigma$ lì), poi si applica `transform` a training *e* test. Mai `fit` sul test. Vedi sezione 11.

> **Quando usarlo** — Sempre per modelli basati su distanze, gradienti o penalità (K-Means, PCA, logistica, SVM, reti, Ridge/Lasso). Non necessario per i modelli ad albero. In dubbio: standardizza, non fa male (tranne perdere un po' di interpretabilità).

---

<a name="10"></a>
## 10. Regolarizzazione L1 e L2

L'idea: aggiungere alla funzione di costo una **penalità sulla grandezza dei coefficienti**, per scoraggiare modelli troppo complessi. È il rasoio di Occam reso matematico — il modello deve "pagare" per ogni coefficiente grande, quindi li tiene piccoli a meno che non servano davvero.

### L2 — Ridge Regression

$$\text{Costo} = \text{MSE} + \lambda \sum_{j} w_j^2$$

Penalizza il **quadrato** dei coefficienti. Effetto: li **rimpicciolisce verso zero ma non li annulla mai del tutto**. Distribuisce il "peso" tra feature correlate invece di sceglierne una sola.

### L1 — Lasso Regression

$$\text{Costo} = \text{MSE} + \lambda \sum_{j} |w_j|$$

Penalizza il **valore assoluto**. Effetto: può portare alcuni coefficienti **esattamente a zero**. Questo la rende una **feature selection automatica**: le variabili inutili vengono spente.

### Perché L1 azzera e L2 no (intuizione geometrica)

La penalità L1 ha una "regione di vincolo" a forma di rombo (con spigoli sugli assi); l'ottimo tende a cadere su uno spigolo, dove alcuni coefficienti sono esattamente 0. La penalità L2 ha una regione circolare (liscia, senza spigoli): l'ottimo cade in un punto generico, con coefficienti piccoli ma non nulli.

### Il ruolo di λ (lambda)

È l'iperparametro che controlla la **forza** della penalità:
- $\lambda = 0$ → nessuna regolarizzazione (regressione normale, rischio overfitting).
- $\lambda$ grande → penalità forte, coefficienti molto piccoli (rischio underfitting).

Si sceglie con validation/cross-validation. Spesso si campiona su scala **logaritmica** (sezione 17).

### Tabella decisionale

| | L2 (Ridge) | L1 (Lasso) |
|---|---|---|
| Effetto sui coefficienti | piccoli, mai zero | alcuni esattamente zero |
| Feature selection | no | sì (automatica) |
| Multicollinearità | gestita bene | sceglie una delle correlate |
| Quando | molte feature tutte un po' utili | sospetti che molte feature siano inutili |

C'è anche **Elastic Net**, che combina L1 e L2 per avere il meglio di entrambe.

> **Quando usarlo** — Ridge (L2) se vuoi stabilizzare un modello con molte feature correlate o gestire la collinearità, mantenendole tutte. Lasso (L1) se vuoi un modello **sparso e interpretabile** che selezioni da solo le variabili importanti. In generale, regolarizza ogni volta che il modello mostra overfitting e non puoi aggiungere dati.

---

<a name="11"></a>
## 11. Data leakage e Pipeline

Il **data leakage** è il bug metodologico più insidioso: informazione che non dovrebbe essere disponibile "trapela" nell'addestramento, gonfiando le performance in test ma facendo fallire il modello nel mondo reale. Sintomo classico: un'accuratezza "troppo bella per essere vera" (99,9% su dati rumorosi).

### Due tipi

1. **Target leakage**: una feature contiene informazione sul target che *non sarebbe disponibile* al momento della predizione reale. Esempio: usare "prescrizione di antibiotici" per predire "ha la polmonite". La prescrizione è una *conseguenza* della diagnosi, non una causa predittiva. In produzione, quando devi prevedere, quella feature non esiste ancora.

2. **Train-test contamination**: le statistiche del test set "inquinano" il preprocessing. Esempio classico: calcolare media e deviazione standard per standardizzare su *tutto* il dataset *prima* dello split. Così il training set "sa" già qualcosa della distribuzione del test.

### Come prevenirlo: l'ordine sacro

1. **Split** subito (train/test).
2. **Fit** dei trasformatori (scaler, imputer, encoder) **solo sul training**.
3. **Transform** applicato a entrambi con i parametri del training.

### Le Pipeline di scikit-learn

Una `Pipeline` concatena trasformatori e modello in un unico oggetto. Ogni step riceve l'output del precedente. Quando chiami `pipeline.fit(X_train)`, ogni trasformatore fa `fit` *solo* sul training; quando fai `predict(X_test)`, applica solo `transform`. **Blinda automaticamente contro il train-test contamination**, soprattutto dentro la cross-validation (dove lo scaler va rifittato a ogni fold — farlo a mano è un errore comune).

Il `ColumnTransformer` permette di applicare preprocessing diverso a colonne diverse: es. numeriche → `SimpleImputer` + `StandardScaler`, categoriche → `OneHotEncoder`, tutto in un colpo solo:

```
preprocessor = ColumnTransformer([
    ("num", Pipeline([SimpleImputer(mean), StandardScaler()]), numeriche),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categoriche),
])
model = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])
```

`handle_unknown="ignore"` gestisce categorie viste in test ma non in training (cruciale in produzione).

> **Quando usarlo** — **Sempre** che ci sia preprocessing. La Pipeline non è un vezzo: è lo standard professionale che rende il flusso corretto-by-design e riproducibile. Ogni volta che fai scaling/encoding/imputation a mano fuori da una pipeline, stai rischiando un leakage.

---

<a name="12"></a>
## 12. Classificazione: la regressione logistica

Nonostante il nome, è un **classificatore**. Risolve il problema: "una retta/iperpiano produce qualsiasi numero da $-\infty$ a $+\infty$, ma io voglio una probabilità tra 0 e 1".

### La funzione sigmoide

Prende l'output lineare $z = \mathbf{w}^T\mathbf{x} + b$ e lo schiaccia in $[0,1]$:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

La sigmoide ha la forma a S: valori molto negativi → vicino a 0, molto positivi → vicino a 1, e $z=0$ → 0,5. L'output si interpreta come **probabilità** di appartenere alla classe positiva. La classe predetta si ottiene con una soglia (default 0,5):
$$\hat{y} = \begin{cases} 1 & \text{se } \sigma(z) \geq 0.5 \\ 0 & \text{altrimenti} \end{cases}$$

### Come impara

Minimizza la **cross-entropy** (non l'MSE): vuole assegnare probabilità alta alla classe vera. La cross-entropy punisce severamente la "sicurezza sbagliata" (probabilità 0,99 sulla classe sbagliata → loss enorme).

### Nel notebook (Breast Cancer Wisconsin)

569 biopsie, 30 feature, target binario maligno/benigno. La logistica è dentro una **Pipeline** con `StandardScaler` (è sensibile alla scala) e `max_iter` aumentato per garantire la convergenza dell'ottimizzatore. Si confrontano accuracy e F1 su train e test per diagnosticare overfitting.

> **Quando usarla** — Classificazione binaria quando vuoi **probabilità interpretabili** e un modello lineare/spiegabile, come baseline. È veloce, robusta, e ti dà coefficienti leggibili. Quando i confini tra classi sono fortemente non lineari, passa ad alberi/boosting o reti.

---

<a name="13"></a>
## 13. Metriche per la classificazione (il cuore del "quando")

Questa è la sezione più importante per la domanda "quando si sceglie X invece di Y", perché qui la scelta della metrica dipende interamente da **quale errore ti fa più male**.

### La matrice di confusione

È la base di tutto. Per la classificazione binaria:

| | Predetto Positivo | Predetto Negativo |
|---|---|---|
| **Reale Positivo** | TP (vero positivo) | FN (falso negativo) |
| **Reale Negativo** | FP (falso positivo) | TN (vero negativo) |

La diagonale (TP, TN) sono le predizioni corrette; fuori diagonale (FP, FN) gli errori. Nel caso multiclasse (MNIST, 10×10) la cella $C_{ij}$ dice quante volte la classe vera $i$ è stata predetta come $j$ — utile per vedere *quali* classi si confondono (es. 4↔9, 3↔5).

### Le metriche

**Accuracy** — frazione di predizioni corrette:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
Intuitiva ma **ingannevole su classi sbilanciate**: se il 99% degli esempi è negativo, un modello che predice sempre "negativo" ha 99% di accuracy ed è inutile.

**Recall (sensibilità)** — dei positivi reali, quanti ne becco:
$$\text{Recall} = \frac{TP}{TP + FN}$$
Importante quando i **falsi negativi** sono costosi (mancare una diagnosi, una frode).

**Precision** — delle mie predizioni positive, quante sono giuste:
$$\text{Precision} = \frac{TP}{TP + FP}$$
Importante quando i **falsi positivi** sono costosi (un filtro antispam che cestina email vere).

**F1-score** — media armonica di precision e recall:
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
La media *armonica* (non aritmetica) penalizza lo squilibrio: se una delle due è bassa, F1 crolla. Ottima su dataset sbilanciati e quando vuoi un compromesso tra i due tipi di errore.

### Il trade-off precision/recall

Sono in tensione: per aumentare il recall abbassi la soglia (dici "positivo" più spesso → becchi più positivi reali ma anche più falsi allarmi → precision scende), e viceversa. Non puoi massimizzare entrambi; scegli in base al dominio.

### ROC e AUC

Finora abbiamo usato predizioni binarie con soglia 0,5. Ma il modello produce **probabilità continue**. La **curva ROC** mostra come variano TPR (= recall) e FPR (= falsi positivi / negativi reali) al variare della soglia, da 0 a 1.

$$\text{TPR} = \frac{TP}{TP+FN} \qquad \text{FPR} = \frac{FP}{FP+TN}$$

L'**AUC (Area Under the Curve)** sintetizza tutto in un numero:
- AUC = 1,0 → classificatore perfetto;
- AUC = 0,5 → come tirare una moneta.

Il pregio dell'AUC: è **indipendente dalla soglia** e dallo sbilanciamento, quindi è ottima per **confrontare modelli** in modo equo.

### Tabella decisionale (da tenere a mente)

| Metrica | Usala quando | Esempio |
|---|---|---|
| **Accuracy** | classi bilanciate, errori ugualmente costosi | riconoscimento cifre |
| **Recall** | i falsi negativi sono costosi | diagnosi medica, frodi |
| **Precision** | i falsi positivi sono costosi | filtro antispam |
| **F1** | classi sbilanciate, vuoi un compromesso | anomaly detection |
| **ROC/AUC** | confrontare modelli indipendentemente dalla soglia | scegliere tra due classificatori |

> **Quando usarla** — Parti *sempre* dalla matrice di confusione. Poi scegli la metrica chiedendoti: *un falso positivo o un falso negativo, quale mi costa di più?* Su dati sbilanciati non fidarti mai della sola accuracy.

---

<a name="14"></a>
## 14. La soglia di decisione (threshold)

Un punto sottovalutato ma potente. `predict()` applica una soglia di default **0,5** alle probabilità per decidere la classe. Ma `predict_proba()` restituisce le **probabilità grezze**, e tu puoi scegliere una soglia diversa.

- Soglia **bassa** (es. 0,01) → il modello dice "positivo" molto facilmente → **recall alto, precision bassa** (becchi quasi tutti i positivi, ma con molti falsi allarmi).
- Soglia **alta** (es. 0,99) → dice "positivo" solo se molto sicuro → **precision alta, recall basso**.

La soglia è quindi una **leva di business**, non un parametro del modello: la sposti a seconda di quale errore vuoi minimizzare. In oncologia abbassi la soglia (meglio un falso allarme che una diagnosi mancata); in un filtro antispram la alzi (meglio lasciar passare un po' di spam che cestinare email importanti).

Nel notebook di anomaly detection ECG, la soglia ottimale non è scelta a occhio: si fa una **ricerca** sulla griglia di soglie candidate e si tiene quella che **massimizza l'F1-score sul training set**, poi la si applica al test. Questo è il modo corretto: ottimizzi la soglia su dati di sviluppo, la validi su dati nuovi.

> **Quando regolarla** — Ogni volta che il costo dei due errori è asimmetrico, o le classi sono sbilanciate. Non lasciare 0,5 per default solo perché è il default: scegli la soglia in funzione della metrica che conta per te.

---

<a name="15"></a>
## 15. Alberi decisionali e Random Forest

### Albero decisionale

Pone una sequenza di domande sì/no sulle feature ("petalo > 2,5 cm?") e scende lungo i rami fino a una foglia che dà la predizione. È il modello **più interpretabile** in assoluto: puoi disegnarlo e leggere esattamente la logica.

**Pro**: nessuno scaling necessario (split su soglie → invariante alla scala), gestisce relazioni non lineari e interazioni in automatico, leggibile.

**Contro**: un singolo albero **overfitta facilmente**. Se lo lasci crescere senza vincoli, costruisce una foglia per quasi ogni esempio → memorizza il training. È anche **instabile**: piccole variazioni nei dati cambiano completamente la struttura (alta varianza).

**`max_depth`** è la leva principale di complessità: limitarla forza un albero più semplice, più leggibile e meno incline all'overfitting. Albero profondo = basso bias, alta varianza; albero corto = alto bias, bassa varianza (lo stesso trade-off di sempre).

### Random Forest

Combina molti alberi (`n_estimators`), ciascuno addestrato su:
- un **campione casuale** delle righe (bootstrap / *bagging*),
- un **sottoinsieme casuale** delle feature a ogni split (`max_features="sqrt"`).

La predizione finale è la **media** (regressione) o il **voto di maggioranza** (classificazione) degli alberi.

**Perché funziona — l'intuizione chiave**: un singolo albero ha alta varianza (sbaglia in modo idiosincratico). Mediando tanti alberi *de-correlati* (resi diversi dalla casualità), gli errori individuali si **cancellano a vicenda**, mentre il segnale comune si rafforza. È il principio della "saggezza della folla": molti pareri mediocri ma indipendenti battono un singolo esperto instabile. La randomizzazione delle feature è essenziale: senza, tutti gli alberi userebbero la feature più forte allo stesso modo e sarebbero troppo simili (correlati) perché la media aiuti.

**Trade-off**: la foresta perde l'interpretabilità del singolo albero (non puoi più disegnarla), ma guadagna molto in robustezza e accuratezza.

> **Quando usarli** — Albero singolo quando l'**interpretabilità** è prioritaria (devi spiegare le decisioni) e accetti un po' meno accuratezza. Random Forest come **default robusto** su dati tabellari: poco preprocessing, poco tuning, ottime performance, resistente all'overfitting. Quando vuoi spremere l'ultima goccia di performance, passa al boosting (sezione 16).

---

<a name="16"></a>
## 16. Gradient Boosting e XGBoost

### Bagging vs Boosting (la distinzione concettuale)

- **Bagging** (Random Forest): costruisce alberi **in parallelo e indipendenti**, poi media. Riduce la **varianza**.
- **Boosting** (XGBoost): costruisce alberi **in sequenza**, dove ogni nuovo albero corregge gli **errori residui** del precedente. Riduce il **bias** (e con regolarizzazione anche la varianza).

L'intuizione del boosting: invece di tanti modelli forti che votano, costruisci tanti modelli **deboli** (alberi poco profondi) ognuno dei quali si concentra su ciò che gli altri hanno sbagliato. Sommando questi piccoli "aggiustamenti" si ottiene un modello molto accurato.

### Parametri chiave di XGBoost

- **`learning_rate` (eta)**: quanto "pesa" ogni nuovo albero. Piccolo (es. 0,05) → ogni albero corregge poco → serve più alberi ma generalizza meglio. È il classico trade-off "passi piccoli e tanti" vs "passi grandi e pochi".
- **`n_estimators`**: numero massimo di alberi (boosting rounds).
- **`max_depth`**: profondità di ogni albero (di solito bassa, 3–8: alberi deboli).
- **`reg_lambda`**: regolarizzazione L2 sui pesi delle foglie.

### Early stopping

XGBoost monitora la metrica su un **validation set** e **interrompe** l'addestramento quando smette di migliorare (`early_stopping_rounds=10` = "fermati se non migliora per 10 round"). È un anti-overfitting elegante: invece di indovinare `n_estimators`, ne metti tanti e lasci che si fermi da solo al punto giusto. Per questo servono tre blocchi: train, validation (per l'early stopping), test (valutazione finale).

### Comodità pratiche

- Gestisce le **variabili categoriche nativamente** (`enable_categorical=True`, colonne come `category`), senza one-hot manuale.
- Due interfacce: nativa (`DMatrix`, più controllo) e compatibile sklearn (`fit/predict`, più comoda per integrarsi nelle pipeline).
- Non richiede scaling (è basato su alberi).

> **Quando usarlo** — È il **re dei dati tabellari**: quando vuoi la massima accuratezza su tabelle e accetti un po' più di tuning e meno interpretabilità. Random Forest se vuoi un risultato solido con zero sforzo; XGBoost se vuoi vincere. Per dati non strutturati (immagini, testo) servono invece le reti neurali (sezione 21).

---

<a name="17"></a>
## 17. Model selection: Grid, Random e Bayesian search

I **parametri** (es. i pesi) si imparano durante il training. Gli **iperparametri** (learning rate, profondità, λ) vanno fissati *prima* e si scelgono cercando quelli che danno la migliore performance in validazione. Tre strategie, costo crescente di intelligenza.

### Grid Search

Prova **tutte le combinazioni** di una griglia di valori predefinita (prodotto cartesiano), valutando ognuna in cross-validation.

- **Pro**: semplice, esaustiva, trova l'ottimo *dentro la griglia*.
- **Contro**: il costo **esplode** combinatorialmente (maledizione della dimensionalità). 5 parametri × 10 valori = $10^5$ modelli. Con `cv=5`, ×5. Nel notebook: griglia $4\times3\times3 = 36$ combinazioni × 5 fold = **180 modelli**.

### Random Search

Campiona casualmente `n_iter` combinazioni da **distribuzioni di probabilità**.

- **Perché spesso batte la grid a parità di tempo**: molti iperparametri contano poco. La grid spreca tempo a variare meticolosamente parametri irrilevanti; la random esplora più valori *diversi* di quelli che contano davvero. Con lo stesso budget, copre meglio lo spazio.
- **Contro**: i campioni sono **indipendenti** — non impara dai risultati precedenti.

**La distribuzione log-uniforme** (`loguniform`): per parametri che spaziano su più ordini di grandezza (learning rate da $10^{-5}$ a $10^2$), il campionamento *uniforme* sprecherebbe quasi tutti i campioni nella parte alta dell'intervallo (tra 50 e 100 ci sono "più numeri" che tra 0,00001 e 0,1). Il log-uniforme dà **uguale probabilità a ogni ordine di grandezza**, esplorando sensatamente tutta la scala. È il motivo per cui learning rate e λ si cercano sempre in scala logaritmica.

### Ottimizzazione Bayesiana (Optuna / TPE)

Non procede a caso né esaustivamente: costruisce un **modello surrogato** della funzione obiettivo (come si comporta la performance al variare degli iperparametri) e lo usa per **decidere dove guardare dopo**.

Ciclo: (1) il surrogato propone la combinazione più promettente, (2) la si valuta davvero, (3) il risultato aggiorna il surrogato. Bilancia **exploration** (provare zone sconosciute) ed **exploitation** (raffinare le zone buone). I trial **non sono indipendenti**: converge più in fretta verso le regioni promettenti.

Optuna offre visualizzazioni utili: optimization history (la curva deve scendere → sta convergendo), slice plot (impatto di ogni parametro), contour (interazioni tra coppie), parallel coordinates (quali "traiettorie" portano ai risultati migliori). Può salvare lo studio su database e riprenderlo.

### Cross-validation (il pilastro sotto tutto)

Quando i dati sono pochi, un singolo split train/validation è rischioso (dipende troppo da *quale* split è capitato). La **K-Fold CV** divide i dati in $K$ parti: allena $K$ volte, ogni volta una fold diversa fa da validation e le altre $K-1$ da training, poi **media** le performance. Dà una stima più stabile e affidabile, usando tutti i dati sia per allenare sia per validare.

### Tabella decisionale

| Metodo | Strategia | Quando |
|---|---|---|
| **Grid** | tutte le combinazioni | pochi iperparametri, vuoi garanzia sulla griglia |
| **Random** | campionamento casuale | spazio ampio, budget limitato, alcuni parametri irrilevanti |
| **Bayesian** | guidato dai risultati | tuning serio, valutazioni costose, vuoi efficienza massima |

> **Quando usarlo** — Grid solo per spazi piccoli (≤ 3 parametri, pochi valori). Random come default pratico per spazi medio-grandi. Bayesian quando ogni addestramento è costoso e vuoi spremere il massimo con pochi trial. **Sempre** dentro la cross-validation, e con scala logaritmica per i parametri che variano di ordini di grandezza.

---

<a name="18"></a>
## 18. Clustering: K-Means

Primo algoritmo **non supervisionato**: niente etichette, l'obiettivo è raggruppare i punti simili. K-Means partiziona i dati in $k$ cluster non sovrapposti.

### L'algoritmo (iterativo)

1. **Inizializzazione**: scegli $k$ centroidi iniziali (casuali tra i punti).
2. **Assegnazione**: ogni punto va al centroide più vicino (distanza euclidea).
3. **Aggiornamento**: ogni centroide si sposta nella media dei punti che gli sono stati assegnati.
4. **Ripeti** 2–3 finché i centroidi non si muovono più (convergenza).

### L'inerzia (cosa minimizza)

$$\text{Inerzia} = \sum_{i=1}^{k}\sum_{x \in C_i} \|x - \mu_i\|^2$$

È la somma delle distanze al quadrato di ogni punto dal proprio centroide (Within-Cluster Sum of Squares). Inerzia bassa = cluster compatti. K-Means cerca di minimizzarla.

### Scegliere k: il metodo del gomito (Elbow)

L'inerzia **diminuisce sempre** all'aumentare di $k$ (con $k$ = numero di punti, inerzia = 0). Quindi non puoi semplicemente minimizzarla. Si traccia inerzia vs $k$ e si cerca il "gomito": il punto dove l'inerzia smette di calare rapidamente. Quello è il miglior compromesso tra compattezza e numero di cluster.

**Ma attenzione alla soggettività**: non esiste un $k$ "oggettivamente giusto". I dati hanno spesso una gerarchia (Sport → Calcio → Serie A): il numero giusto di cluster dipende dal *livello di astrazione* che ti serve. Il clustering è tanto arte quanto scienza.

### Minimi locali e k-means++

K-Means è **greedy**: converge sempre, ma a un minimo che può essere **locale**, non globale — dipende dall'inizializzazione fortunata o sfortunata dei centroidi. Soluzioni:
- eseguirlo più volte (`n_init`) con inizializzazioni diverse e tenere il risultato a inerzia minima;
- **k-means++**: sceglie centroidi iniziali ben distanziati, accelerando la convergenza e riducendo il rischio di minimi locali pessimi. È il default di scikit-learn.

### Standardizzare prima

**Fondamentale**: K-Means usa distanze euclidee. Senza scaling, la feature con range più ampio domina e distorce i cluster. `StandardScaler` mette tutte le feature sullo stesso piano.

### Applicazione: quantizzazione dei colori

Bell'esempio del notebook: ogni pixel è un punto nello spazio RGB 3D. K-Means trova $k$ colori rappresentativi (i centroidi) e rimpiazza ogni pixel con il colore del centroide più vicino → immagine ridotta a $k$ colori. È compressione tramite clustering.

> **Quando usarlo** — Quando vuoi **segmentare** dati non etichettati in gruppi (clienti, documenti, colori) e hai un'idea ragionevole di quanti gruppi cercare. Ricorda: standardizza prima, usa più inizializzazioni, e tratta $k$ come una scelta guidata dal problema, non come una verità. K-Means assume cluster sferici e di dimensioni simili: se i tuoi cluster hanno forme strane, servono altri algoritmi (DBSCAN, gaussian mixtures).

---

<a name="19"></a>
## 19. Riduzione della dimensionalità: PCA

La **PCA** (Principal Component Analysis) è la tecnica principale di riduzione **lineare** della dimensionalità.

### Perché ridurre le dimensioni

- **Visualizzazione**: portare dati a 2–3 dimensioni per graficarli.
- **Compressione**: meno feature → training più veloce, meno memoria.
- **Rimozione del rumore**: scartando le componenti a varianza minima si elimina spesso il rumore di fondo.
- **Multicollinearità**: le nuove feature sono non correlate per costruzione.

### L'intuizione geometrica

La PCA è una **rotazione** del sistema di coordinate. Immagina la nuvola di punti:
1. Trova la direzione di **massima varianza** (dove i dati sono più "allungati") → questa è la **prima componente principale** (PC1).
2. Ruota gli assi così che PC1 punti in quella direzione.
3. La seconda componente è **ortogonale** alla prima e cattura la massima varianza residua. E così via.

Non stai buttando feature a caso: stai **guardando i dati dall'angolazione più informativa**. Poi tieni solo le prime componenti (quelle che spiegano più varianza) e scarti le ultime, perdendo pochissima informazione.

### Proprietà delle componenti

1. **Non correlate**: ortogonali tra loro → correlazione lineare zero. Risolve la multicollinearità.
2. **Ordinate per varianza decrescente**: PC1 spiega più di PC2, che spiega più di PC3... Questo permette di scegliere quante tenerne.

### Strumenti pratici (dai notebook)

- `explained_variance_ratio_`: quanta varianza spiega ogni componente.
- La **curva di varianza cumulata**: scegli il numero di componenti che cattura, es., il 90–95% della varianza.
- `inverse_transform`: **ricostruisce** i dati originali dalle componenti — con poche componenti la ricostruzione è approssimata (sul MNIST: 5 componenti → cifre sfocate, 250 → quasi perfette). Mostra visivamente il trade-off compressione/qualità.
- Le `components_` sulle immagini MNIST sono dei "volti fantasma" (eigen-digits): i pattern principali di variazione tra le cifre.

### Il trade-off su un caso reale

Nel notebook, una Random Forest sul MNIST: addestrata su tutte le 784 feature vs su poche componenti PCA. Con 80–250 componenti l'accuratezza resta alta ma il training è molto più rapido. È il punto di PCA: **velocità in cambio di una minima perdita di informazione**.

> **Quando usarla** — Quando hai **troppe feature** (alta dimensionalità) e vuoi velocizzare/comprimere, ridurre il rumore o gestire la multicollinearità, **mantenendo una trasformazione lineare e invertibile**. Standardizza sempre prima. Limite: cattura solo relazioni *lineari*; per strutture non lineari servono t-SNE/UMAP (sezione 20). E le componenti perdono il significato fisico delle feature originali (meno interpretabili).

---

<a name="20"></a>
## 20. Visualizzazione: PCA vs t-SNE vs UMAP

Tre tecniche per portare dati ad alta dimensione in 2D, ma con scopi diversi. Il notebook le confronta tutte sul MNIST.

### PCA

**Lineare, globale, deterministica.** Preserva le direzioni di massima varianza globale. Veloce e riproducibile. Limite: se la struttura dei dati è curva/non lineare (come le cifre MNIST nello spazio dei pixel), una proiezione lineare a 2D sovrappone le classi e si vede poco.

### t-SNE

**Non lineare, locale.** Cerca di preservare le **vicinanze locali**: punti vicini nello spazio originale restano vicini in 2D. Produce cluster ben separati e visivamente belli.

- **Parametro chiave: `perplexity`** — quanti "vicini" considerare (tipicamente 5–50). Cambia molto il risultato: troppo bassa → frammenta, troppo alta → fonde i cluster. Nel notebook si esplora una griglia perplexity × numero di componenti PCA.
- **Attenzioni**: lento sui dataset grandi (per questo spesso si fa **PCA prima** per ridurre a ~50 dimensioni, poi t-SNE), stocastico (run diversi → mappe diverse), e **le distanze tra cluster non sono interpretabili** (la dimensione e la distanza dei gruppi in t-SNE non hanno significato quantitativo).

### UMAP

**Non lineare, locale + un po' di globale.** Simile a t-SNE ma generalmente **più veloce** e tende a preservare meglio la **struttura globale** (le distanze relative tra cluster sono più sensate). Scala meglio su grandi dataset.

### Tabella decisionale

| | PCA | t-SNE | UMAP |
|---|---|---|---|
| Tipo | lineare | non lineare | non lineare |
| Preserva | varianza globale | vicinanze locali | locale + globale |
| Velocità | molto veloce | lenta | veloce |
| Deterministica | sì | no | no (ma più stabile) |
| Uso tipico | compressione, preprocessing | visualizzazione di cluster | visualizzazione, più scalabile |

> **Quando usarle** — **PCA** quando ti serve compressione/preprocessing reale (riduci feature da dare a un modello) o una proiezione lineare interpretabile. **t-SNE/UMAP** quando l'obiettivo è *solo visualizzare* la struttura a cluster di dati complessi — non per fare feature per un modello, e senza leggere troppo nelle distanze. UMAP è la scelta moderna preferita per velocità e struttura globale. Trucco pratico: PCA → 50 dim → poi t-SNE/UMAP.

---

<a name="21"></a>
## 21. Reti neurali e autoencoder

Quando i dati sono **non strutturati** (immagini, segnali) i modelli tabellari faticano: servono reti neurali, che imparano da sole le feature rilevanti.

### MLP (Multi-Layer Perceptron) — il caso MNIST

La rete del notebook confusion_matrix: due strati nascosti densi da 100 neuroni (ReLU) + output softmax a 10 classi. Concetti:

- **ReLU** ($f(x) = \max(0, x)$): introduce non-linearità (senza funzioni di attivazione, una pila di layer lineari resterebbe lineare). Evita il *vanishing gradient* delle vecchie sigmoidi negli strati profondi.
- **Leaky ReLU**: variante che lascia passare un piccolo gradiente anche per input negativi → evita i "neuroni morti" (neuroni bloccati a zero che non imparano più). Per questo gli autoencoder del corso la preferiscono.
- **Softmax** sull'output: trasforma i logit grezzi in una distribuzione di probabilità che somma a 1 (una probabilità per classe).
- **Normalizzazione dei pixel** [0,255] → [0,1]: senza, i gradienti diventano enormi e il training è instabile/lento.
- **Optimizer Adam**: discesa del gradiente con learning rate adattivo per ogni parametro. È il default robusto: converge in fretta senza tuning fine del learning rate.
- **Loss**: SparseCategoricalCrossentropy (etichette come interi). Cross-entropy perché è classificazione.
- **`validation_split`**: tiene da parte una fetta del training per monitorare l'overfitting epoca per epoca senza toccare il test.

### Autoencoder

Una rete **non supervisionata** che impara a comprimere e ricostruire l'input:
```
Input x → [Encoder] → z (bottleneck/spazio latente) → [Decoder] → x̂ ≈ x
```
Il **bottleneck** (strato stretto centrale) è il trucco: costringendo i dati a passare per pochi neuroni, la rete è *obbligata* a tenere solo l'informazione essenziale e scartare il ridondante. La loss (MSE) si calcola tra input e output — **niente etichette**, il target è l'input stesso.

Varianti dal notebook:

- **Denso**: solo layer `Dense`. Tratta ogni pixel come indipendente.
- **Convoluzionale**: usa `Conv2D` per sfruttare la **struttura spaziale** (pixel vicini sono correlati). Molto più efficace sulle immagini. Encoder = Conv + MaxPooling (downsampling), Decoder = UpSampling + Conv (ricostruzione).
- **Denoising**: input = immagine + rumore, target = immagine pulita. Impara a **rimuovere il rumore**. Forza la rete a capire la *struttura* dell'immagine, non i singoli pixel → bottleneck più robusto. Il rumore viene applicato *on-the-fly* a ogni epoca (data augmentation).
- **Inpainting**: input = immagine con una patch oscurata, target = immagine completa. Impara a **ricostruire regioni mancanti** capendo il contesto.

### Scelte progettuali ricorrenti

- **Adam** come optimizer (adattivo, default solido).
- **MSE** come loss (valori continui dei pixel; penalizza errori grandi).
- **MAE** come metrica (più interpretabile: stessa unità dei pixel).
- **EarlyStopping** (`patience=5`, `restore_best_weights=True`): ferma il training quando la val_loss smette di migliorare e ripristina i pesi migliori → anti-overfitting.
- **ModelCheckpoint** (`save_best_only=True`): salva solo i pesi migliori.

> **Quando usarli** — Le reti quando i dati sono non strutturati (immagini, audio, testo) e i pattern sono troppo complessi per i modelli tabellari. Gli **autoencoder** quando vuoi: comprimere senza etichette, rimuovere rumore, o — soprattutto — fare **anomaly detection** (sezione 22). Su dati tabellari semplici, invece, XGBoost di solito batte una rete con molto meno sforzo.

---

<a name="22"></a>
## 22. Anomaly detection con autoencoder

Bellissima applicazione che mette insieme metà del corso. Problema (notebook ECG5000): rilevare battiti cardiaci anomali. Le anomalie sono **rare e varie** — non puoi addestrarci sopra un classificatore supervisionato normale (poche etichette, classi sbilanciatissime, anomalie "nuove" mai viste).

### L'idea (geniale nella sua semplicità)

1. Addestra un autoencoder **solo sui dati normali** (battiti sani). Impara a ricostruirli benissimo.
2. Su un input anomalo, la rete — che ha visto solo normali — **fatica a ricostruirlo** → produce un **errore di ricostruzione alto**.
3. L'errore di ricostruzione diventa il **punteggio di anomalia**:
   $$e(x) = \sqrt{\sum_t (x_t - \hat{x}_t)^2} \quad (\text{distanza euclidea L2})$$
4. Superata una **soglia**, il campione è anomalo.

### Come si sceglie la soglia (collega alla sezione 14)

- Si calcolano gli errori di ricostruzione su normali e anomali e si guardano le due **distribuzioni** sovrapposte (istogramma). Più sono separate, più è facile dividere.
- Si cerca la soglia che **massimizza l'F1-score sul training set** (griglia di soglie candidate), poi la si applica al test.
- Si valuta su test con **recall, precision, F1** — non accuracy, perché le classi sono sbilanciate e l'F1 è la metrica giusta (vedi sezione 13).
- Le anomalie sono trattate come **classe positiva** (è ciò che vogliamo "beccare").

### Perché funziona meglio del supervisionato qui

Non devi conoscere *come* sono fatte le anomalie: ne basta una definizione implicita ("tutto ciò che non assomiglia al normale"). Così rilevi anche anomalie **mai viste prima**, cosa impossibile per un classificatore addestrato solo su tipi noti di anomalia.

> **Quando usarla** — Quando le anomalie sono **rare, eterogenee o sconosciute** e hai abbondanza di esempi "normali" (frodi, guasti industriali, battiti patologici, difetti su texture). Modella la normalità, misura quanto un nuovo dato se ne discosta, taglia con una soglia ottimizzata in F1. È il paradigma da preferire quando un approccio supervisionato classico non ha abbastanza esempi positivi per imparare.

---

<a name="23"></a>
## 23. Mappa decisionale finale: "quando uso cosa"

### Che tipo di problema è?

- **Ho etichette numeriche continue** → Regressione (lineare → poly/Ridge/Lasso → XGBoost).
- **Ho etichette categoriche** → Classificazione (logistica → alberi/RF → XGBoost → rete).
- **Non ho etichette, voglio gruppi** → Clustering (K-Means).
- **Non ho etichette, voglio meno dimensioni** → PCA (o t-SNE/UMAP per visualizzare).
- **Voglio trovare il raro/strano** → Anomaly detection (autoencoder + soglia).
- **Dati non strutturati (immagini/audio/testo)** → Reti neurali.

### Quale modello per dati tabellari (in ordine di "prova prima questo")

1. **Regressione lineare/logistica** — baseline, interpretabile, veloce.
2. **Random Forest** — robusto, poco tuning, gestisce non-linearità.
3. **XGBoost** — massima accuratezza, più tuning.

### Quale metrica

- Regressione: **MAE** (interpretabile) o **R²** (varianza spiegata); MSE/RMSE se punire errori grandi; MAPE per errore relativo.
- Classificazione bilanciata: **Accuracy**.
- Classificazione sbilanciata o errori asimmetrici: **Precision** (FP costosi), **Recall** (FN costosi), **F1** (compromesso).
- Confronto tra modelli: **ROC/AUC**.

### Quale ottimizzazione iperparametri

- Spazio piccolo → **Grid Search**.
- Spazio medio/grande, budget limitato → **Random Search** (con scale log).
- Valutazioni costose, tuning serio → **Bayesian/Optuna**.
- In tutti i casi → dentro la **cross-validation**.

### Quale regolarizzazione / anti-overfitting

- Modello lineare con feature correlate → **Ridge (L2)**.
- Vuoi feature selection automatica → **Lasso (L1)**.
- Alberi → limita **`max_depth`**; foreste/boosting → più alberi + early stopping.
- Reti → **EarlyStopping**, dropout, più dati.
- Sempre → monitora il gap train/test.

### Preprocessing: serve lo scaling?

- **Sì** per: K-Means, PCA, logistica, SVM, reti, Ridge/Lasso (tutto ciò che usa distanze, gradienti o penalità).
- **No** per: alberi, Random Forest, XGBoost (invarianti alla scala).
- **Sempre** dentro una **Pipeline**, con `fit` solo sul training (anti-leakage).

### I tre principi che attraversano tutto il corso

1. **Generalizzazione, non memorizzazione** — il test set esiste solo per misurare quanto bene il modello va su dati mai visti. Non toccarlo finché non hai finito.
2. **Il preprocessing si impara solo dal training** — split prima, fit dopo, transform per entrambi. Le Pipeline lo garantiscono.
3. **La scelta giusta dipende dal costo dell'errore** — metrica, soglia, modello: non c'è un "migliore" assoluto, c'è il migliore *per il tuo problema e per quale sbaglio ti fa più male*.
