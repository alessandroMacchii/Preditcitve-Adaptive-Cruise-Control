# Study Notebook — Machine Learning

> A reasoned guide to all the course topics. The goal is not to list definitions, but to understand **why** certain choices are made and **when** one approach beats another. Each section closes with a "When to use it" decision box.

---

## Table of contents

1. [What Machine Learning is and the learning paradigms](#1)
2. [The tools: NumPy and pandas (and why you vectorize)](#2)
3. [The workflow of an ML project and the cross-cutting concepts](#3)
4. [Loss, cost function and metrics](#4)
5. [Overfitting, underfitting and the bias–variance trade-off](#5)
6. [Linear regression](#6)
7. [Encoding of categorical variables](#7)
8. [Feature engineering: interactions, polynomial features, collinearity](#8)
9. [Standardization: when it really matters](#9)
10. [L1 and L2 regularization](#10)
11. [Data leakage and Pipeline](#11)
12. [Classification: logistic regression](#12)
13. [Metrics for classification (the heart of the "when")](#13)
14. [The decision threshold](#14)
15. [Decision trees and Random Forest](#15)
16. [Gradient Boosting and XGBoost](#16)
17. [Model selection: Grid, Random and Bayesian search](#17)
18. [Clustering: K-Means](#18)
19. [Dimensionality reduction: PCA](#19)
20. [Visualization: PCA vs t-SNE vs UMAP](#20)
21. [Neural networks and autoencoders](#21)
22. [Anomaly detection with autoencoders](#22)
23. [Final decision map: "when I use what"](#23)

---

<a name="1"></a>
## 1. What Machine Learning is and the learning paradigms

A traditional program is a hand-written rule: input → rules → output. Machine learning flips the logic: you give the system **inputs and outputs** and it derives the rules by itself. This is useful when the rules are too many, too subtle or change over time (recognizing a handwritten digit: nobody could write the hundreds of `if` statements needed).

### The four paradigms

| Paradigm | What it receives | What it learns | Example |
|---|---|---|---|
| **Supervised** | input + correct label | to map input → output | house price, spam/not spam |
| **Unsupervised** | only inputs, no labels | hidden structure in the data | customer clustering, anomaly detection |
| **Self-supervised** | input that generates its own label | like supervised but without human annotation | next-token prediction of LLMs |
| **Reinforcement** | an environment and a reward | a strategy (policy) that maximizes the reward over time | an agent that plays, a robot |

**Why distinguishing them matters.** The choice of paradigm is the *first* decision and depends entirely on which data you have. If you have labels → supervised. If you don't (or they cost too much) → unsupervised. Self-supervised is the great insight behind modern models: it turns unlabeled data into a supervised problem for free (the "answer" is a part of the input itself, e.g. the next word).

### Data structure

- **Structured**: tables (rows = examples, columns = features). Here linear regression, trees, boosting dominate.
- **Unstructured**: images, audio, text. Here neural networks dominate, learning the relevant features by themselves.

In supervised learning the dataset is written as the **feature** matrix $X$ (the input variables, "the signals") and the **target** vector $y$ (what we want to predict). Each row of $X$ is an example, each column a feature.

### Feature types

- **Numeric**: quantities (height, price).
- **Nominal categorical**: groups without order (city, color).
- **Ordinal categorical**: groups with order (education level, satisfaction 1–5).

This distinction is not pedantic: it determines **how to encode** the variable (section 7) and which assumptions the model will end up making.

> **When to use it** — The starting question is always: *do I have the answers (labels) or not?* Everything else follows from there.

---

<a name="2"></a>
## 2. The tools: NumPy and pandas (and why you vectorize)

### NumPy: the homogeneous array

An `ndarray` is an N-dimensional **homogeneous** structure (all elements of the same type) with fixed size. This rigidity is exactly what makes it fast: the data sits in a contiguous block of memory and the operations are executed by compiled C code.

Key concepts from the tutorial:

- **N-dimensional slicing**: `a[row, column]`, with `:` for "everything".
- **Slice assignment**: `a[:, 0] = 5` overwrites in bulk.
- **Boolean fancy indexing**: a mask `a[a > 0]` selects only the elements that satisfy the condition. It is the "vectorized" way to filter.
- **Elementwise operations (ufunc)**: `a + b`, `np.exp(a)` act on each element without writing loops.
- **Broadcasting**: NumPy automatically "aligns" arrays of different shapes following two rules — (1) missing dimensions are added on the left, (2) size-1 dimensions are expanded. So you can add a scalar to a matrix, or a row vector to each row of a matrix, without copying anything.
- **Reductions**: `sum`, `mean`, `any`, `all` with `axis` collapse the array along an axis.
- **Reshape**: returns a *view* (same memory, new shape) when possible — cheap but watch the side effects if you modify it.
- **View vs copy**: a slice is a view (modifying it modifies the original); `np.copy` creates independent data.

### Why vectorize: the Game of Life case

The file `life.py` shows two implementations of the same rule (Conway's Game of Life):

1. **Pure Python**: two nested `for` loops over an $N \times N$ grid → complexity $O(N^2)$ with interpreter overhead at every cell.
2. **NumPy + 2D convolution**: counting the neighbors of *all* the cells in a single operation, with a 3×3 kernel, and applying the rules with a single `np.where`.

The vectorized version is much faster for three reasons: the loop runs in **compiled C** (not in the Python interpreter), the data is **contiguous** in memory (cache-friendly), and the operations exploit **SIMD instructions** of the CPU. The general lesson: *every time you write a `for` loop over an array, ask yourself whether the vectorized equivalent exists*. In ML the datasets have millions of rows and the difference between the two versions is between "seconds" and "hours".

### pandas: tabular analysis

The `DataFrame` is the central table. Key points from the tutorial:

- **Access**: `df.column` or `df["column"]` (the second is mandatory if the name has spaces or conflicts).
- **`.loc` (by label) vs `.iloc` (by position)**: a crucial distinction to avoid picking the wrong row.
- **Boolean masks** and `.query("price < 1e6")` to filter.
- **Alignment by index**: every Series/DataFrame has an Index; operations align *by label*, not by position. It is powerful but a classic source of silent bugs.
- **GroupBy** (split → apply → combine): group, apply an aggregation, recombine.
- **Missing data** (`NaN`): you can drop them (`dropna`, but you lose rows) or **impute** them (fill with mean/median/mode). Imputation is preferable when dropping would make you lose too much data.

> **When to use it** — pandas to explore, clean and do feature engineering on tabular data; NumPy for pure numeric computation and every time performance matters. Golden rule: **no Python loops over large arrays, always vectorized operations**.

---

<a name="3"></a>
## 3. The workflow of an ML project and the cross-cutting concepts

The real-estate regression notebook (King County) shows the end-to-end flow that repeats in almost every project:

1. **EDA (Exploratory Data Analysis)**: understand the data with histograms, box plots, counts. It serves to discover anomalies and relations *before* modeling.
2. **Data cleaning**: handle anomalous values (houses at $0 or $10M with a tiny surface) and missing ones.
3. **Feature engineering**: build/select the right variables.
4. **Train/test split**: separate the data for an honest evaluation.
5. **Model + predictions**: train and predict.
6. **Evaluation**: metrics + residual analysis.

### Why training and test are separated

The model must **generalize** to unseen data, not memorize the training data. The test set is the simulation of the "real world": you touch it **only at the end**. If you use it to make decisions, your performance estimates become optimistically false.

For the decisions *during* development (which model, which hyperparameter) you need a third block, the **validation set**:

- **Training set** → trains the parameters.
- **Validation set** → compares models and chooses hyperparameters.
- **Test set** → final unbiased evaluation, once only.

`random_state` (e.g. `=42`) fixes the random seed of the split: it guarantees **reproducibility** (same results at every run).

> **When to use it** — Always. The rule "the test set is not touched until you are done" is non-negotiable methodology: violating it is the most common way to fool yourself that a model is good.

---

<a name="4"></a>
## 4. Loss, cost function and metrics

Three concepts that are easily confused but serve different things.

- **Loss function**: measures the error on **a single example**.
- **Cost function**: the **average** of the losses over the whole training set. It is what the algorithm **minimizes** during training.
- **Metric**: an index **for us humans**, computed downstream, to judge how good the model is. It is not (necessarily) what the model optimizes.

The key distinction: the **loss must be optimizable** (differentiable, smooth), the **metric must be interpretable**. Sometimes they coincide, often they don't. Example: a classification network minimizes cross-entropy (loss) but we judge it with accuracy or F1 (metrics), because "percentage of errors" is not differentiable and therefore cannot be optimized directly with gradient descent.

### Typical losses

- **MSE (Mean Squared Error)** — regression:
  $$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
  It squares the error → heavily penalizes large errors. It is smooth and differentiable, ideal for optimization.

- **Cross-Entropy** — classification:
  $$\mathcal{L} = -\log(\hat{y}_{\text{correct class}})$$
  Measures how much the predicted probability distribution diverges from the true one. It grows toward infinity when the model assigns a low probability to the right class → it harshly punishes wrong confidence.

### Regression metrics

- **MAE (Mean Absolute Error)**: mean absolute error, in the same unit as the target ($ for the houses). Easy to explain to a non-expert.
- **MAPE (Mean Absolute Percentage Error)**: error as a percentage, useful for communicating the relative margin ("we are wrong on average by 18%").
- **R² (coefficient of determination)**: the share of target variance explained by the model.
  $$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$$
  It goes from $0$ to $1$ (it can be negative if the model is worse than the mean). $R^2 = 1$ → perfect; $R^2 = 0$ → does no better than always predicting the mean.

**MAE vs MSE — when to use which.** MSE/RMSE if large errors are particularly costly (you want to punish them more) and for optimization; MAE if you want a measure robust to outliers and easy to interpret. MAPE when the relative error matters more than the absolute one (but beware: it explodes if the true values are close to zero).

> **When to use it** — Choose the **loss** based on the type of problem (MSE for regression, cross-entropy for classification). Choose the **metric** based on what you must communicate and which error hurts you the most.

---

<a name="5"></a>
## 5. Overfitting, underfitting and the bias–variance trade-off

This is the central concept of all supervised ML.

- **Underfitting**: the model is too simple to capture the structure of the data. High error *both* on training *and* on test. It has **high bias** (systematic error: it assumes the wrong form, e.g. a line for curved data).
- **Overfitting**: the model is too complex and "learns by heart" the training set, noise included. Low error on training but high on test. It has **high variance** (it is hypersensitive to the specific data seen).

### The trade-off

| | Bias | Variance | Symptom |
|---|---|---|---|
| Model too simple | high | low | underfitting |
| Model too complex | low | high | overfitting |
| Right model | balance | balance | good generalization |

The polynomial-features experiment shows it very well: as the polynomial degree increases, the **training error always drops**, but the **test error first drops and then rises again**. The minimum point of the test error is the model that generalizes best. The "gap" between test error and train error measures the overfitting.

**Why it happens.** A model with many parameters has enough "freedom" to pass exactly through every training point, including the noisy ones. But the noise is random: it does not reappear in the test set, so that precision is useless or harmful outside the training.

**Occam's razor**: with equal results, the simpler model is preferable. Regularization (section 10) is the formalization of this principle.

### Strategies against overfitting

- Choose the right complexity (polynomial degree, tree depth) via validation/cross-validation.
- Regularization (L1/L2).
- More training data (the harder it is to memorize it all).
- *Always* monitor the test/validation error, not just the training one.

> **When to worry** — If train error ≈ test error but both high → underfitting, increase the complexity. If train error low but test error high → overfitting, simplify or regularize. A model slightly underfitting is often safer than one overfitting.

---

<a name="6"></a>
## 6. Linear regression

It is the starting point of all supervised ML because it is simple, interpretable and fast.

### The idea

It assumes a **linear** relationship between features and target. In the univariate case it is a line:
$$\hat{y} = w_1 x + b$$
where $w_1$ is the coefficient (slope, "how much $y$ changes per unit of $x$") and $b$ is the bias/intercept ("value of $y$ when $x=0$"). In the multivariate case:
$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

### How it "learns"

It minimizes the MSE cost function: it looks for the coefficients that make the sum of the squared residuals ($y_i - \hat{y}_i$) minimal. Geometrically, it finds the line/hyperplane that passes as "in the middle" as possible of the cloud of points.

### Interpretation of the coefficients

Each $w_i$ is the **marginal effect** of the feature $x_i$ on $y$, *all else being equal* (the *ceteris paribus* clause). Example from the notebook: the coefficient of `sqft_living` says by how many dollars the price increases for each additional square foot. This interpretability is the great advantage of linear regression over "black box" models.

### Residual analysis (the part often skipped)

The residuals (actual − predicted) tell whether the model's **assumptions** hold. If the model is good, the residuals should be:

- centered around 0 (no systematic bias),
- with constant variance (homoscedastic — no "funnel"),
- without pattern (if you see a curve in the residuals, the true relation was not linear),
- distributed approximately normally.

When the residuals show a funnel or a curve, it is a signal that a transformation is needed. In the real-estate notebook, applying the **logarithm to the price** (`np.log(price)`) makes the residuals more normal and handles the fact that errors grow with the price: a model that is wrong by $50k on a $5M house is excellent, on a $100k one it is terrible. Modeling $\log(\text{price})$ turns the errors into *percentages*, more sensible for prices.

### Notebook progression (why adding features helps)

The notebook builds the model in layers and each layer improves R²:
1. only `sqft_living` → a poor model;
2. + bedrooms, bathrooms, condition, waterfront, view → better;
3. + interactions (section 8);
4. + `statezip` with one-hot encoding → geography explains a lot of the price;
5. + log of the target → healthier residuals.

> **When to use it** — As a **baseline** always: it is fast, interpretable, and immediately tells you whether the problem is "easy". It is the right choice when you need to explain *why* the model predicts a certain thing (e.g. regulatory, business contexts). When the relation is clearly non-linear or there are many complex interactions, move to polynomial features or, better, to tree/boosting models.

---

<a name="7"></a>
## 7. Encoding of categorical variables

Mathematical models want numbers. Qualitative variables (color, city) must be transformed, and *how* you transform them changes the assumptions the model will make.

### Ordinal Encoding

Assigns an integer to each category (Low=1, Medium=2, High=3).

**The problem**: it introduces a very strong assumption about *distance*. The model treats the difference 1→2 as identical to 2→3, and assumes an *order*. For truly ordinal variables (education level) it can be fine; for **nominal** variables ("Red"=1, "Green"=2, "Blue"=3) it is a disaster: you are telling the model that Blue > Green > Red and that Blue is "three times" Red, a nonexistent relation that leads it to look for fake linear patterns.

### One-Hot Encoding

Creates a binary column (0/1) for each category. "Gender = {M, F, Other}" → three columns, each 1 only for the right row. No artificial order, no invented distances. It is the standard for **nominal** variables.

**The "drop one" rule** (dummy variables): if a variable has $k$ categories, you include only $k-1$. Why? Intuitively for **redundancy**: if you are not M and not Other, then you are necessarily F — the third column adds no information. Technically, including all the columns creates **perfect multicollinearity** (their sum is always 1, identical to the intercept): the matrix becomes non-invertible (singular) and the linear-regression coefficients are not computable. The excluded category becomes the **reference level** (baseline) and each dummy coefficient is interpreted as "the average difference of $y$ relative to the baseline".

> **Important note**: the drop-one is for **linear** models. For trees/boosting it is not necessary (they don't suffer from multicollinearity in the same way).

### Target Encoding

Replaces each category with the **mean of the target** for that category.

- **Pro**: a single column even for variables with hundreds of categories (it reduces the dimensionality). Useful for high-cardinality features (e.g. ZIP codes, product codes) where one-hot would explode into thousands of columns.
- **Con**: a very high risk of **overfitting/data leakage**, because you are inserting information about the target into the feature. It must be done with caution (computed only on the training set, with smoothing or cross-fitting).

### Decision table

| Technique | When | Risk |
|---|---|---|
| **Ordinal** | *truly* ordinal variable (few categories with a real order) | imposes false distances/orders on nominal variables |
| **One-Hot** | nominal variable with **low** cardinality | explosion of columns if the categories are many |
| **Target** | nominal variable with **high** cardinality | overfitting / leakage if done badly |

> **When to use it** — Default for nominal ones: **One-Hot** (with drop-one if the model is linear). Ordinal *only* if the order is real and meaningful. Target encoding only when the cardinality is too high for one-hot and you know how to handle the leakage.

---

<a name="8"></a>
## 8. Feature engineering: interactions, polynomial features, collinearity

"Data is clay; feature engineering is the art of shaping it." It often improves performance more than changing the model.

### Interaction terms

An **additive** model assumes that the effect of a variable is *independent* of the others. But in reality they often interact. Example from the PDF: the price of a house as a function of surface and condition ("to renovate" vs "new"). An additive model:
$$\text{Price} = \beta_0 + \beta_1 \text{Surface} + \beta_2 \text{NewCondition}$$
assumes that switching to "New" adds a *fixed* value, equal for a studio and for a villa. But intuitively renovation is worth much more on 500 m² than on 30 m². So you add an **interaction term** (the product):
$$\text{Price} = \beta_0 + \beta_1 \text{Surface} + \beta_2 \text{NewCondition} + \beta_3 (\text{Surface} \times \text{NewCondition})$$
Now the *slope* (the importance of the surface) changes depending on the condition. In the real-estate notebook this idea becomes `sqft_above * waterfront`, `sqft_above * view`, and the interaction between zone (`statezip`) and surface — the price per m² depends on the neighborhood.

### Polynomial features

They transform a feature into $[x, x^2, x^3, \dots]$. This allows a **linear** model to fit non-linear curves: the model stays linear in the *coefficients*, but no longer in the features. Example: for data following a parabola, $\hat{y} = w_0 + w_1 x + w_2 x^2$ fits perfectly where a line failed.

**Beware of the degree**: it is the complexity lever that generates overfitting. Degree too low → underfitting; too high → the model wiggles between the training points and collapses on the test. The right degree is chosen with cross-validation.

**Why standardize first**: $x^{10}$ with $x=15$ is a gigantic number → numerical problems. Standardizing ($x \to$ mean 0, std 1) keeps the powers in a manageable range.

### Collinearity and multicollinearity

It occurs when two or more features are strongly correlated with each other.

- **Problem**: the model cannot isolate the individual effect of each (if two variables move together, "which of the two" causes the effect?).
- **Consequence**: the linear-regression coefficients become **unstable**, with high variance, and impossible to interpret with confidence.
- Limit case: the **dummy variable trap** (section 7), *perfect* multicollinearity.

L2 (Ridge) regularization is a good remedy for collinearity, because it stabilizes the coefficients.

> **When to use it** — Add interactions when you have reason to believe that the effect of a variable depends on another (almost always in the real world). Use polynomial features when the relation is curved but you want to stay with a linear/interpretable model. Keep an eye on collinearity as soon as you have correlated features or many dummies.

---

<a name="9"></a>
## 9. Standardization: when it really matters

The `StandardScaler` applies the Z-score: $z = \frac{x - \mu}{\sigma}$, bringing each feature to mean 0 and standard deviation 1.

**Why do it**: many algorithms reason in terms of *distances* or *coefficient scale*. If one feature goes from 0 to 1,000,000 and another from 0 to 5, the first dominates the computations only because it has larger numbers, not because it is more important.

### Where it is indispensable

- **K-Means** and everything that uses Euclidean distances (without scaling, the feature with the widest range decides the clusters).
- **PCA** (it looks for directions of maximum variance: without scaling, the feature with the highest numeric variance wins regardless).
- **Regularized regression (Ridge/Lasso)**: the penalty on the coefficients is scale-sensitive.
- **SVM, neural networks**: more stable gradients, faster convergence (e.g. dividing the pixels by 255 in MNIST).
- **Logistic regression**: scale-sensitive (in the notebook it is inside the pipeline).

### Where it is NOT needed (or almost)

- **Decision trees, Random Forest, XGBoost**: they split on thresholds of individual features ("sqft > 1500?"), so they are **invariant to monotone transformations**. Standardizing changes nothing. It is one of the reasons tree models are convenient: less preprocessing.

### The anti-leakage rule (fundamental)

You `fit` the scaler **only on the training set** (it computes $\mu$ and $\sigma$ there), then you apply `transform` to training *and* test. Never `fit` on the test. See section 11.

> **When to use it** — Always for models based on distances, gradients or penalties (K-Means, PCA, logistic, SVM, nets, Ridge/Lasso). Not necessary for tree models. In doubt: standardize, it does no harm (except losing a bit of interpretability).

---

<a name="10"></a>
## 10. L1 and L2 regularization

The idea: add to the cost function a **penalty on the magnitude of the coefficients**, to discourage models that are too complex. It is Occam's razor made mathematical — the model must "pay" for every large coefficient, so it keeps them small unless they are really needed.

### L2 — Ridge Regression

$$\text{Cost} = \text{MSE} + \lambda \sum_{j} w_j^2$$

It penalizes the **square** of the coefficients. Effect: it **shrinks them toward zero but never zeroes them completely**. It distributes the "weight" among correlated features instead of choosing only one.

### L1 — Lasso Regression

$$\text{Cost} = \text{MSE} + \lambda \sum_{j} |w_j|$$

It penalizes the **absolute value**. Effect: it can bring some coefficients **exactly to zero**. This makes it an **automatic feature selection**: the useless variables are turned off.

### Why L1 zeroes and L2 does not (geometric intuition)

The L1 penalty has a diamond-shaped "constraint region" (with corners on the axes); the optimum tends to fall on a corner, where some coefficients are exactly 0. The L2 penalty has a circular region (smooth, without corners): the optimum falls at a generic point, with small but non-zero coefficients.

### The role of λ (lambda)

It is the hyperparameter that controls the **strength** of the penalty:
- $\lambda = 0$ → no regularization (normal regression, overfitting risk).
- $\lambda$ large → strong penalty, very small coefficients (underfitting risk).

It is chosen with validation/cross-validation. It is often sampled on a **logarithmic** scale (section 17).

### Decision table

| | L2 (Ridge) | L1 (Lasso) |
|---|---|---|
| Effect on the coefficients | small, never zero | some exactly zero |
| Feature selection | no | yes (automatic) |
| Multicollinearity | handled well | picks one of the correlated ones |
| When | many features all a bit useful | you suspect many features are useless |

There is also **Elastic Net**, which combines L1 and L2 to get the best of both.

> **When to use it** — Ridge (L2) if you want to stabilize a model with many correlated features or handle collinearity, keeping them all. Lasso (L1) if you want a **sparse and interpretable** model that selects the important variables by itself. In general, regularize every time the model shows overfitting and you cannot add data.

---

<a name="11"></a>
## 11. Data leakage and Pipeline

**Data leakage** is the most insidious methodological bug: information that should not be available "leaks" into the training, inflating the test performance but making the model fail in the real world. Classic symptom: an accuracy "too good to be true" (99.9% on noisy data).

### Two types

1. **Target leakage**: a feature contains information about the target that *would not be available* at the moment of the real prediction. Example: using "antibiotic prescription" to predict "has pneumonia". The prescription is a *consequence* of the diagnosis, not a predictive cause. In production, when you must predict, that feature does not exist yet.

2. **Train-test contamination**: the test set's statistics "pollute" the preprocessing. Classic example: computing the mean and standard deviation to standardize over the *whole* dataset *before* the split. This way the training set already "knows" something about the test distribution.

### How to prevent it: the sacred order

1. **Split** right away (train/test).
2. **Fit** the transformers (scaler, imputer, encoder) **only on the training**.
3. **Transform** applied to both with the training parameters.

### scikit-learn Pipelines

A `Pipeline` chains transformers and model into a single object. Each step receives the output of the previous one. When you call `pipeline.fit(X_train)`, each transformer does `fit` *only* on the training; when you do `predict(X_test)`, it applies only `transform`. **It automatically seals against train-test contamination**, especially inside cross-validation (where the scaler must be refitted at each fold — doing it by hand is a common mistake).

The `ColumnTransformer` allows applying different preprocessing to different columns: e.g. numeric → `SimpleImputer` + `StandardScaler`, categorical → `OneHotEncoder`, all in one go:

```
preprocessor = ColumnTransformer([
    ("num", Pipeline([SimpleImputer(mean), StandardScaler()]), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
])
model = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])
```

`handle_unknown="ignore"` handles categories seen in test but not in training (crucial in production).

> **When to use it** — **Always** when there is preprocessing. The Pipeline is not a frill: it is the professional standard that makes the flow correct-by-design and reproducible. Every time you do scaling/encoding/imputation by hand outside a pipeline, you are risking a leakage.

---

<a name="12"></a>
## 12. Classification: logistic regression

Despite the name, it is a **classifier**. It solves the problem: "a line/hyperplane produces any number from $-\infty$ to $+\infty$, but I want a probability between 0 and 1".

### The sigmoid function

It takes the linear output $z = \mathbf{w}^T\mathbf{x} + b$ and squashes it into $[0,1]$:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

The sigmoid has an S shape: very negative values → close to 0, very positive → close to 1, and $z=0$ → 0.5. The output is interpreted as the **probability** of belonging to the positive class. The predicted class is obtained with a threshold (default 0.5):
$$\hat{y} = \begin{cases} 1 & \text{if } \sigma(z) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$

### How it learns

It minimizes the **cross-entropy** (not the MSE): it wants to assign a high probability to the true class. Cross-entropy severely punishes "wrong confidence" (probability 0.99 on the wrong class → huge loss).

### In the notebook (Breast Cancer Wisconsin)

569 biopsies, 30 features, binary malignant/benign target. The logistic is inside a **Pipeline** with `StandardScaler` (it is scale-sensitive) and `max_iter` increased to guarantee the optimizer's convergence. Accuracy and F1 on train and test are compared to diagnose overfitting.

> **When to use it** — Binary classification when you want **interpretable probabilities** and a linear/explainable model, as a baseline. It is fast, robust, and gives you readable coefficients. When the boundaries between classes are strongly non-linear, move to trees/boosting or nets.

---

<a name="13"></a>
## 13. Metrics for classification (the heart of the "when")

This is the most important section for the question "when do you choose X instead of Y", because here the choice of metric depends entirely on **which error hurts you the most**.

### The confusion matrix

It is the basis of everything. For binary classification:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP (true positive) | FN (false negative) |
| **Actual Negative** | FP (false positive) | TN (true negative) |

The diagonal (TP, TN) are the correct predictions; off-diagonal (FP, FN) the errors. In the multiclass case (MNIST, 10×10) the cell $C_{ij}$ tells how many times the true class $i$ was predicted as $j$ — useful to see *which* classes get confused (e.g. 4↔9, 3↔5).

### The metrics

**Accuracy** — fraction of correct predictions:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
Intuitive but **deceptive on imbalanced classes**: if 99% of the examples are negative, a model that always predicts "negative" has 99% accuracy and is useless.

**Recall (sensitivity)** — of the real positives, how many I catch:
$$\text{Recall} = \frac{TP}{TP + FN}$$
Important when the **false negatives** are costly (missing a diagnosis, a fraud).

**Precision** — of my positive predictions, how many are right:
$$\text{Precision} = \frac{TP}{TP + FP}$$
Important when the **false positives** are costly (a spam filter that trashes real emails).

**F1-score** — harmonic mean of precision and recall:
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
The *harmonic* mean (not arithmetic) penalizes imbalance: if one of the two is low, F1 collapses. Excellent on imbalanced datasets and when you want a compromise between the two types of error.

### The precision/recall trade-off

They are in tension: to increase the recall you lower the threshold (you say "positive" more often → you catch more real positives but also more false alarms → precision drops), and vice versa. You cannot maximize both; you choose based on the domain.

### ROC and AUC

So far we have used binary predictions with a 0.5 threshold. But the model produces **continuous probabilities**. The **ROC curve** shows how TPR (= recall) and FPR (= false positives / real negatives) vary as the threshold varies, from 0 to 1.

$$\text{TPR} = \frac{TP}{TP+FN} \qquad \text{FPR} = \frac{FP}{FP+TN}$$

The **AUC (Area Under the Curve)** summarizes everything in one number:
- AUC = 1.0 → perfect classifier;
- AUC = 0.5 → like flipping a coin.

The merit of the AUC: it is **threshold-independent** and imbalance-independent, so it is excellent for **comparing models** fairly.

### Decision table (to keep in mind)

| Metric | Use it when | Example |
|---|---|---|
| **Accuracy** | balanced classes, equally costly errors | digit recognition |
| **Recall** | false negatives are costly | medical diagnosis, fraud |
| **Precision** | false positives are costly | spam filter |
| **F1** | imbalanced classes, you want a compromise | anomaly detection |
| **ROC/AUC** | comparing models independently of the threshold | choosing between two classifiers |

> **When to use it** — *Always* start from the confusion matrix. Then choose the metric by asking yourself: *a false positive or a false negative, which costs me more?* On imbalanced data never trust accuracy alone.

---

<a name="14"></a>
## 14. The decision threshold

An underrated but powerful point. `predict()` applies a default threshold of **0.5** to the probabilities to decide the class. But `predict_proba()` returns the **raw probabilities**, and you can choose a different threshold.

- **Low** threshold (e.g. 0.01) → the model says "positive" very easily → **high recall, low precision** (you catch almost all the positives, but with many false alarms).
- **High** threshold (e.g. 0.99) → it says "positive" only if very confident → **high precision, low recall**.

The threshold is therefore a **business lever**, not a model parameter: you move it depending on which error you want to minimize. In oncology you lower the threshold (better a false alarm than a missed diagnosis); in a spam filter you raise it (better to let some spam through than to trash important emails).

In the ECG anomaly-detection notebook, the optimal threshold is not chosen by eye: a **search** is done over the grid of candidate thresholds and the one that **maximizes the F1-score on the training set** is kept, then applied to the test. This is the correct way: you optimize the threshold on development data, you validate it on new data.

> **When to adjust it** — Every time the cost of the two errors is asymmetric, or the classes are imbalanced. Don't leave 0.5 by default just because it is the default: choose the threshold according to the metric that matters to you.

---

<a name="15"></a>
## 15. Decision trees and Random Forest

### Decision tree

It poses a sequence of yes/no questions on the features ("petal > 2.5 cm?") and descends the branches until a leaf that gives the prediction. It is the **most interpretable** model of all: you can draw it and read exactly the logic.

**Pro**: no scaling necessary (splits on thresholds → scale-invariant), handles non-linear relations and interactions automatically, readable.

**Con**: a single tree **overfits easily**. If you let it grow without constraints, it builds a leaf for almost every example → it memorizes the training. It is also **unstable**: small variations in the data completely change the structure (high variance).

**`max_depth`** is the main complexity lever: limiting it forces a simpler tree, more readable and less prone to overfitting. Deep tree = low bias, high variance; short tree = high bias, low variance (the same trade-off as always).

### Random Forest

It combines many trees (`n_estimators`), each trained on:
- a **random sample** of the rows (bootstrap / *bagging*),
- a **random subset** of the features at each split (`max_features="sqrt"`).

The final prediction is the **mean** (regression) or the **majority vote** (classification) of the trees.

**Why it works — the key intuition**: a single tree has high variance (it errs in an idiosyncratic way). By averaging many *de-correlated* trees (made different by the randomness), the individual errors **cancel each other out**, while the common signal strengthens. It is the principle of the "wisdom of the crowd": many mediocre but independent opinions beat a single unstable expert. The feature randomization is essential: without it, all the trees would use the strongest feature the same way and would be too similar (correlated) for the average to help.

**Trade-off**: the forest loses the interpretability of the single tree (you can no longer draw it), but it gains a lot in robustness and accuracy.

> **When to use them** — A single tree when **interpretability** is a priority (you must explain the decisions) and you accept a bit less accuracy. Random Forest as a **robust default** on tabular data: little preprocessing, little tuning, great performance, resistant to overfitting. When you want to squeeze the last drop of performance, move to boosting (section 16).

---

<a name="16"></a>
## 16. Gradient Boosting and XGBoost

### Bagging vs Boosting (the conceptual distinction)

- **Bagging** (Random Forest): builds trees **in parallel and independent**, then averages. It reduces the **variance**.
- **Boosting** (XGBoost): builds trees **in sequence**, where each new tree corrects the **residual errors** of the previous one. It reduces the **bias** (and with regularization also the variance).

The intuition of boosting: instead of many strong models voting, you build many **weak** models (shallow trees) each of which focuses on what the others got wrong. Summing these small "adjustments" gives a very accurate model.

### Key XGBoost parameters

- **`learning_rate` (eta)**: how much each new tree "weighs". Small (e.g. 0.05) → each tree corrects little → you need more trees but it generalizes better. It is the classic "small and many steps" vs "big and few steps" trade-off.
- **`n_estimators`**: maximum number of trees (boosting rounds).
- **`max_depth`**: depth of each tree (usually low, 3–8: weak trees).
- **`reg_lambda`**: L2 regularization on the leaf weights.

### Early stopping

XGBoost monitors the metric on a **validation set** and **stops** training when it stops improving (`early_stopping_rounds=10` = "stop if it doesn't improve for 10 rounds"). It is an elegant anti-overfitting: instead of guessing `n_estimators`, you put many and let it stop by itself at the right point. This is why you need three blocks: train, validation (for the early stopping), test (final evaluation).

### Practical conveniences

- It handles **categorical variables natively** (`enable_categorical=True`, columns as `category`), without manual one-hot.
- Two interfaces: native (`DMatrix`, more control) and sklearn-compatible (`fit/predict`, more convenient to integrate into pipelines).
- It does not require scaling (it is tree-based).

> **When to use it** — It is the **king of tabular data**: when you want the maximum accuracy on tables and you accept a bit more tuning and less interpretability. Random Forest if you want a solid result with zero effort; XGBoost if you want to win. For unstructured data (images, text) you need neural networks instead (section 21).

---

<a name="17"></a>
## 17. Model selection: Grid, Random and Bayesian search

The **parameters** (e.g. the weights) are learned during training. The **hyperparameters** (learning rate, depth, λ) must be fixed *beforehand* and are chosen by looking for those that give the best validation performance. Three strategies, of increasing "intelligence" cost.

### Grid Search

It tries **all the combinations** of a predefined grid of values (Cartesian product), evaluating each in cross-validation.

- **Pro**: simple, exhaustive, finds the optimum *within the grid*.
- **Con**: the cost **explodes** combinatorially (curse of dimensionality). 5 parameters × 10 values = $10^5$ models. With `cv=5`, ×5. In the notebook: grid $4\times3\times3 = 36$ combinations × 5 folds = **180 models**.

### Random Search

It samples randomly `n_iter` combinations from **probability distributions**.

- **Why it often beats the grid at equal time**: many hyperparameters count little. The grid wastes time meticulously varying irrelevant parameters; the random one explores more *different* values of those that really count. With the same budget, it covers the space better.
- **Con**: the samples are **independent** — it does not learn from the previous results.

**The log-uniform distribution** (`loguniform`): for parameters that span several orders of magnitude (learning rate from $10^{-5}$ to $10^2$), *uniform* sampling would waste almost all the samples in the high part of the interval (between 50 and 100 there are "more numbers" than between 0.00001 and 0.1). The log-uniform gives **equal probability to each order of magnitude**, sensibly exploring the whole scale. It is the reason learning rate and λ are always searched on a logarithmic scale.

### Bayesian optimization (Optuna / TPE)

It proceeds neither randomly nor exhaustively: it builds a **surrogate model** of the objective function (how the performance behaves as the hyperparameters vary) and uses it to **decide where to look next**.

Cycle: (1) the surrogate proposes the most promising combination, (2) it is actually evaluated, (3) the result updates the surrogate. It balances **exploration** (trying unknown areas) and **exploitation** (refining the good areas). The trials **are not independent**: it converges faster toward the promising regions.

Optuna offers useful visualizations: optimization history (the curve must go down → it is converging), slice plot (impact of each parameter), contour (interactions between pairs), parallel coordinates (which "trajectories" lead to the best results). It can save the study to a database and resume it.

### Cross-validation (the pillar under everything)

When the data is scarce, a single train/validation split is risky (it depends too much on *which* split you got). **K-Fold CV** divides the data into $K$ parts: it trains $K$ times, each time a different fold acts as validation and the other $K-1$ as training, then **averages** the performance. It gives a more stable and reliable estimate, using all the data both to train and to validate.

### Decision table

| Method | Strategy | When |
|---|---|---|
| **Grid** | all the combinations | few hyperparameters, you want a guarantee over the grid |
| **Random** | random sampling | large space, limited budget, some irrelevant parameters |
| **Bayesian** | guided by the results | serious tuning, costly evaluations, you want maximum efficiency |

> **When to use it** — Grid only for small spaces (≤ 3 parameters, few values). Random as the practical default for medium-large spaces. Bayesian when each training is costly and you want to squeeze the most with few trials. **Always** inside cross-validation, and with a logarithmic scale for the parameters that vary by orders of magnitude.

---

<a name="18"></a>
## 18. Clustering: K-Means

The first **unsupervised** algorithm: no labels, the goal is to group similar points. K-Means partitions the data into $k$ non-overlapping clusters.

### The algorithm (iterative)

1. **Initialization**: choose $k$ initial centroids (random among the points).
2. **Assignment**: each point goes to the nearest centroid (Euclidean distance).
3. **Update**: each centroid moves to the mean of the points assigned to it.
4. **Repeat** 2–3 until the centroids no longer move (convergence).

### The inertia (what it minimizes)

$$\text{Inertia} = \sum_{i=1}^{k}\sum_{x \in C_i} \|x - \mu_i\|^2$$

It is the sum of the squared distances of each point from its own centroid (Within-Cluster Sum of Squares). Low inertia = compact clusters. K-Means tries to minimize it.

### Choosing k: the Elbow method

The inertia **always decreases** as $k$ increases (with $k$ = number of points, inertia = 0). So you cannot simply minimize it. You plot inertia vs $k$ and look for the "elbow": the point where the inertia stops falling rapidly. That is the best compromise between compactness and number of clusters.

**But beware of the subjectivity**: there is no "objectively right" $k$. The data often has a hierarchy (Sport → Football → Serie A): the right number of clusters depends on the *level of abstraction* you need. Clustering is as much art as science.

### Local minima and k-means++

K-Means is **greedy**: it always converges, but to a minimum that may be **local**, not global — it depends on the lucky or unlucky initialization of the centroids. Solutions:
- run it several times (`n_init`) with different initializations and keep the minimum-inertia result;
- **k-means++**: it chooses well-spaced initial centroids, accelerating convergence and reducing the risk of very bad local minima. It is scikit-learn's default.

### Standardize first

**Fundamental**: K-Means uses Euclidean distances. Without scaling, the feature with the widest range dominates and distorts the clusters. `StandardScaler` puts all the features on the same plane.

### Application: color quantization

A nice example from the notebook: each pixel is a point in the 3D RGB space. K-Means finds $k$ representative colors (the centroids) and replaces each pixel with the color of the nearest centroid → an image reduced to $k$ colors. It is compression via clustering.

> **When to use it** — When you want to **segment** unlabeled data into groups (customers, documents, colors) and you have a reasonable idea of how many groups to look for. Remember: standardize first, use multiple initializations, and treat $k$ as a problem-driven choice, not as a truth. K-Means assumes spherical clusters of similar size: if your clusters have strange shapes, you need other algorithms (DBSCAN, gaussian mixtures).

---

<a name="19"></a>
## 19. Dimensionality reduction: PCA

**PCA** (Principal Component Analysis) is the main technique of **linear** dimensionality reduction.

### Why reduce the dimensions

- **Visualization**: bring data to 2–3 dimensions to plot it.
- **Compression**: fewer features → faster training, less memory.
- **Noise removal**: by discarding the minimum-variance components you often eliminate the background noise.
- **Multicollinearity**: the new features are uncorrelated by construction.

### The geometric intuition

PCA is a **rotation** of the coordinate system. Imagine the cloud of points:
1. Find the direction of **maximum variance** (where the data is most "stretched") → this is the **first principal component** (PC1).
2. Rotate the axes so that PC1 points in that direction.
3. The second component is **orthogonal** to the first and captures the maximum residual variance. And so on.

You are not throwing away features at random: you are **looking at the data from the most informative angle**. Then you keep only the first components (those that explain the most variance) and discard the last, losing very little information.

### Properties of the components

1. **Uncorrelated**: orthogonal to each other → zero linear correlation. It solves the multicollinearity.
2. **Ordered by decreasing variance**: PC1 explains more than PC2, which explains more than PC3... This allows choosing how many to keep.

### Practical tools (from the notebooks)

- `explained_variance_ratio_`: how much variance each component explains.
- The **cumulative variance curve**: choose the number of components that captures, e.g., 90–95% of the variance.
- `inverse_transform`: **reconstructs** the original data from the components — with few components the reconstruction is approximate (on MNIST: 5 components → blurry digits, 250 → almost perfect). It visually shows the compression/quality trade-off.
- The `components_` on the MNIST images are "ghost faces" (eigen-digits): the main patterns of variation between the digits.

### The trade-off on a real case

In the notebook, a Random Forest on MNIST: trained on all the 784 features vs on a few PCA components. With 80–250 components the accuracy stays high but the training is much faster. That is the point of PCA: **speed in exchange for a minimal loss of information**.

> **When to use it** — When you have **too many features** (high dimensionality) and you want to speed up/compress, reduce the noise or handle the multicollinearity, **keeping a linear and invertible transformation**. Always standardize first. Limit: it captures only *linear* relations; for non-linear structures you need t-SNE/UMAP (section 20). And the components lose the physical meaning of the original features (less interpretable).

---

<a name="20"></a>
## 20. Visualization: PCA vs t-SNE vs UMAP

Three techniques to bring high-dimensional data into 2D, but with different purposes. The notebook compares them all on MNIST.

### PCA

**Linear, global, deterministic.** It preserves the directions of maximum global variance. Fast and reproducible. Limit: if the data structure is curved/non-linear (like the MNIST digits in pixel space), a linear 2D projection overlaps the classes and little is seen.

### t-SNE

**Non-linear, local.** It tries to preserve the **local neighborhoods**: points close in the original space stay close in 2D. It produces well-separated and visually pleasing clusters.

- **Key parameter: `perplexity`** — how many "neighbors" to consider (typically 5–50). It changes the result a lot: too low → it fragments, too high → it merges the clusters. In the notebook a grid perplexity × number of PCA components is explored.
- **Cautions**: slow on large datasets (this is why you often do **PCA first** to reduce to ~50 dimensions, then t-SNE), stochastic (different runs → different maps), and **the distances between clusters are not interpretable** (the size and distance of the groups in t-SNE have no quantitative meaning).

### UMAP

**Non-linear, local + a bit of global.** Similar to t-SNE but generally **faster** and it tends to preserve the **global structure** better (the relative distances between clusters are more sensible). It scales better on large datasets.

### Decision table

| | PCA | t-SNE | UMAP |
|---|---|---|---|
| Type | linear | non-linear | non-linear |
| Preserves | global variance | local neighborhoods | local + global |
| Speed | very fast | slow | fast |
| Deterministic | yes | no | no (but more stable) |
| Typical use | compression, preprocessing | cluster visualization | visualization, more scalable |

> **When to use them** — **PCA** when you need real compression/preprocessing (reduce features to feed a model) or an interpretable linear projection. **t-SNE/UMAP** when the goal is *only to visualize* the cluster structure of complex data — not to make features for a model, and without reading too much into the distances. UMAP is the preferred modern choice for speed and global structure. Practical trick: PCA → 50 dim → then t-SNE/UMAP.

---

<a name="21"></a>
## 21. Neural networks and autoencoders

When the data is **unstructured** (images, signals) the tabular models struggle: you need neural networks, which learn the relevant features by themselves.

### MLP (Multi-Layer Perceptron) — the MNIST case

The network of the confusion_matrix notebook: two dense hidden layers of 100 neurons (ReLU) + softmax output with 10 classes. Concepts:

- **ReLU** ($f(x) = \max(0, x)$): introduces non-linearity (without activation functions, a stack of linear layers would stay linear). It avoids the *vanishing gradient* of the old sigmoids in deep layers.
- **Leaky ReLU**: a variant that lets a small gradient through even for negative inputs → it avoids "dead neurons" (neurons stuck at zero that no longer learn). This is why the course's autoencoders prefer it.
- **Softmax** on the output: turns the raw logits into a probability distribution that sums to 1 (one probability per class).
- **Pixel normalization** [0,255] → [0,1]: without it, the gradients become enormous and training is unstable/slow.
- **Adam optimizer**: gradient descent with an adaptive learning rate per parameter. It is the robust default: it converges quickly without fine tuning of the learning rate.
- **Loss**: SparseCategoricalCrossentropy (labels as integers). Cross-entropy because it is classification.
- **`validation_split`**: it sets aside a slice of the training to monitor overfitting epoch by epoch without touching the test.

### Autoencoder

An **unsupervised** network that learns to compress and reconstruct the input:
```
Input x → [Encoder] → z (bottleneck/latent space) → [Decoder] → x̂ ≈ x
```
The **bottleneck** (narrow central layer) is the trick: by forcing the data to pass through few neurons, the network is *forced* to keep only the essential information and discard the redundant one. The loss (MSE) is computed between input and output — **no labels**, the target is the input itself.

Variants from the notebook:

- **Dense**: only `Dense` layers. It treats each pixel as independent.
- **Convolutional**: uses `Conv2D` to exploit the **spatial structure** (nearby pixels are correlated). Much more effective on images. Encoder = Conv + MaxPooling (downsampling), Decoder = UpSampling + Conv (reconstruction).
- **Denoising**: input = image + noise, target = clean image. It learns to **remove the noise**. It forces the network to understand the *structure* of the image, not the individual pixels → a more robust bottleneck. The noise is applied *on-the-fly* at each epoch (data augmentation).
- **Inpainting**: input = image with an obscured patch, target = complete image. It learns to **reconstruct missing regions** by understanding the context.

### Recurring design choices

- **Adam** as the optimizer (adaptive, solid default).
- **MSE** as the loss (continuous pixel values; penalizes large errors).
- **MAE** as the metric (more interpretable: same unit as the pixels).
- **EarlyStopping** (`patience=5`, `restore_best_weights=True`): stops training when the val_loss stops improving and restores the best weights → anti-overfitting.
- **ModelCheckpoint** (`save_best_only=True`): saves only the best weights.

> **When to use them** — Nets when the data is unstructured (images, audio, text) and the patterns are too complex for tabular models. **Autoencoders** when you want to: compress without labels, remove noise, or — above all — do **anomaly detection** (section 22). On simple tabular data, instead, XGBoost usually beats a net with much less effort.

---

<a name="22"></a>
## 22. Anomaly detection with autoencoders

A beautiful application that brings together half the course. Problem (ECG5000 notebook): detect anomalous heartbeats. The anomalies are **rare and varied** — you cannot train a normal supervised classifier on them (few labels, very imbalanced classes, "new" anomalies never seen).

### The idea (brilliant in its simplicity)

1. Train an autoencoder **only on the normal data** (healthy heartbeats). It learns to reconstruct them very well.
2. On an anomalous input, the network — which has only seen normal ones — **struggles to reconstruct it** → it produces a **high reconstruction error**.
3. The reconstruction error becomes the **anomaly score**:
   $$e(x) = \sqrt{\sum_t (x_t - \hat{x}_t)^2} \quad (\text{L2 Euclidean distance})$$
4. Beyond a **threshold**, the sample is anomalous.

### How the threshold is chosen (links to section 14)

- You compute the reconstruction errors on normal and anomalous ones and look at the two overlapping **distributions** (histogram). The more separated they are, the easier to divide.
- You look for the threshold that **maximizes the F1-score on the training set** (grid of candidate thresholds), then apply it to the test.
- You evaluate on test with **recall, precision, F1** — not accuracy, because the classes are imbalanced and F1 is the right metric (see section 13).
- The anomalies are treated as the **positive class** (it is what we want to "catch").

### Why it works better than supervised here

You don't need to know *how* the anomalies look: an implicit definition suffices ("everything that does not resemble the normal"). This way you detect even anomalies **never seen before**, something impossible for a classifier trained only on known types of anomaly.

> **When to use it** — When the anomalies are **rare, heterogeneous or unknown** and you have an abundance of "normal" examples (fraud, industrial faults, pathological heartbeats, texture defects). Model the normality, measure how much a new datum deviates from it, cut with an F1-optimized threshold. It is the paradigm to prefer when a classic supervised approach does not have enough positive examples to learn.

---

<a name="23"></a>
## 23. Final decision map: "when I use what"

### What type of problem is it?

- **I have continuous numeric labels** → Regression (linear → poly/Ridge/Lasso → XGBoost).
- **I have categorical labels** → Classification (logistic → trees/RF → XGBoost → net).
- **I have no labels, I want groups** → Clustering (K-Means).
- **I have no labels, I want fewer dimensions** → PCA (or t-SNE/UMAP to visualize).
- **I want to find the rare/strange** → Anomaly detection (autoencoder + threshold).
- **Unstructured data (images/audio/text)** → Neural networks.

### Which model for tabular data (in order of "try this first")

1. **Linear/logistic regression** — baseline, interpretable, fast.
2. **Random Forest** — robust, little tuning, handles non-linearity.
3. **XGBoost** — maximum accuracy, more tuning.

### Which metric

- Regression: **MAE** (interpretable) or **R²** (explained variance); MSE/RMSE if punishing large errors; MAPE for relative error.
- Balanced classification: **Accuracy**.
- Imbalanced classification or asymmetric errors: **Precision** (costly FP), **Recall** (costly FN), **F1** (compromise).
- Comparison between models: **ROC/AUC**.

### Which hyperparameter optimization

- Small space → **Grid Search**.
- Medium/large space, limited budget → **Random Search** (with log scales).
- Costly evaluations, serious tuning → **Bayesian/Optuna**.
- In all cases → inside **cross-validation**.

### Which regularization / anti-overfitting

- Linear model with correlated features → **Ridge (L2)**.
- You want automatic feature selection → **Lasso (L1)**.
- Trees → limit **`max_depth`**; forests/boosting → more trees + early stopping.
- Nets → **EarlyStopping**, dropout, more data.
- Always → monitor the train/test gap.

### Preprocessing: is scaling needed?

- **Yes** for: K-Means, PCA, logistic, SVM, nets, Ridge/Lasso (everything that uses distances, gradients or penalties).
- **No** for: trees, Random Forest, XGBoost (scale-invariant).
- **Always** inside a **Pipeline**, with `fit` only on the training (anti-leakage).

### The three principles that run through the whole course

1. **Generalization, not memorization** — the test set exists only to measure how well the model does on unseen data. Don't touch it until you are done.
2. **Preprocessing is learned only from the training** — split first, fit after, transform for both. The Pipelines guarantee it.
3. **The right choice depends on the cost of the error** — metric, threshold, model: there is no absolute "best", there is the best *for your problem and for which mistake hurts you the most*.
