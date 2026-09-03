# K Nearest Neighbours

A forensic classifier that identifies what a fragment of broken glass came from —
a house window, a car window, a bottle, a piece of tableware or a headlamp — using
nothing but its refractive index and eight oxide percentages.

| Notebook | Model | Dataset | Task |
|---|---|---|---|
| [`glass-origin-classification.ipynb`](glass-origin-classification.ipynb) | K nearest neighbours, k = 1, Manhattan distance | `glass.csv` | Assign a glass fragment to one of six origins |

## Problem

A window breaks during a burglary and a two millimetre fragment ends up on a
suspect's sleeve. The fragment carries no serial number and looks like every other
piece of broken glass, but the answer decides whether it places the suspect at the
scene or only proves they once dropped a jar.

What a laboratory can measure on a piece that small is its chemistry, and the
chemistry is not arbitrary: glass is manufactured to a recipe and the recipe
follows the use. A windscreen has to survive bending, so it is float glass with
magnesium; a headlamp has to stand heat, so it carries barium; tableware is made
without potassium. Fragments of the same origin should therefore sit close
together in the space of the nine measurements — which is exactly the assumption a
distance-based classifier is built on.

- **Response:** `Type`, the origin of the fragment, a code from 1 to 7 that names a
  category rather than a quantity.
- **Predictors:** nine laboratory measurements — the refractive index `RI` and the
  weight percentages of sodium, magnesium, aluminium, silicon, potassium, calcium,
  barium and iron.
- **Preparation:** no empty cells; one duplicated fragment removed so no row could
  land in the training and test sets at once and be its own nearest neighbour; all
  nine columns standardised, then a 149 / 64 stratified split.

## Dataset

`glass.csv` — the Glass Identification dataset that B. German collected at the
Home Office Forensic Science Service in the United Kingdom, distributed through
the UCI repository. **214 rows, 10 columns**: nine measurements plus the response.
Every column is numeric and no cell is empty. One row is a duplicate, leaving 213.

The classes are unbalanced and one is missing outright:

| Origin | Fragments | Share |
|---|---|---|
| 1 — Building windows, float | 69 | 32.4 % |
| 2 — Building windows, non float | 76 | 35.7 % |
| 3 — Vehicle windows, float | 17 | 8.0 % |
| 5 — Containers | 13 | 6.1 % |
| 6 — Tableware | 9 | 4.2 % |
| 7 — Headlamps | 29 | 13.6 % |

Code 4, non-float vehicle windows, has no fragments in the file, so this is a six
class problem and a model trained here cannot recognise that origin at all.
Answering "building window" to everything would already be right 35.7 % of the
time, which is the floor every accuracy below has to be read against.

## Results

Three settings — the number of neighbours, the distance and the vote — were chosen
inside the training set by five-fold cross validation repeated five times over a
grid of 100 configurations. The search was run twice under two criteria, and they
disagreed:

| Selection criterion | Configuration | Cross validated score |
|---|---|---|
| Accuracy | k = 6, Manhattan, distance-weighted vote | 0.7328 accuracy |
| Macro F1 | k = 1, Manhattan, uniform vote | 0.6912 macro F1 |

The model selected on macro F1 was kept. On the 64 test fragments:

| Metric | Value |
|---|---|
| Test accuracy | 0.750 |
| Test macro F1 | 0.732 |
| Majority class floor | 0.359 |
| Cross validated macro F1 on the training set | 0.691 |

| Origin | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Building windows, float | 0.77 | 0.81 | 0.79 | 21 |
| Building windows, non float | 0.65 | 0.74 | 0.69 | 23 |
| Vehicle windows, float | 0.50 | 0.40 | 0.44 | 5 |
| Containers | 1.00 | 0.75 | 0.86 | 4 |
| Tableware | 1.00 | 0.50 | 0.67 | 2 |
| Headlamps | 1.00 | 0.89 | 0.94 | 9 |

**The model solves part of the forensic question and not the rest.** It identifies
headlamps, containers and tableware at a precision of 1.00 on this test set, so
when it names one of those three it can be believed, and the chemistry behind that
is visible in the data — barium marks the headlamps, low magnesium with high
calcium marks the containers. It cannot tell a float vehicle window from a float
building window: 2 of 5 found, at a precision of 0.50. Those two are made by the
same process from nearly the same recipe, and the nine measurements do not carry
enough to separate them. No choice of k fixes that.

**The interesting result is the disagreement between the two criteria, not the
headline number.** Optimising accuracy returns k = 6 with a weighted vote, which
on the test set classifies all five vehicle windows as something else and finds
one of the two tableware fragments, for a macro F1 of 0.606. Consulting six
neighbours smooths the boundaries, and smoothing pushes every prediction towards
whatever is locally more frequent — which in a region shared by several origins is
the class with 53 training fragments rather than the one with 7. Counted per
fragment that trade is profitable; counted per class it is not. Since the purpose
is to name what a fragment *is*, not to be right about the common case, macro F1
was chosen as the criterion before the test set was opened.

**The elbow curve never bends.** Accuracy against k falls almost monotonically
from 0.695 at k = 1 to 0.582 at k = 20, with a standard deviation band about 0.06
wide. There is no interior maximum to find: with 7, 9 and 12 training fragments in
the three small classes, a vote that reaches ten or fifteen neighbours fills up
with building windows before it leaves the region. The dataset is too small and
its classes too close for averaging to help.

**Scaling and the distance metric both earned their place, and both were
measured.** On raw values magnesium and calcium alone carry 48 % of every distance
while the refractive index carries 0.1 %, purely because one is a percentage and
the other a ratio near 1. Standardising is worth about four points of macro F1 at
k = 1. Manhattan distance beats Euclidean at both kinds of vote — 0.733 against
0.695 weighted, 0.725 against 0.695 uniform — because squaring the differences
lets one large discrepancy dominate, while the Manhattan sum lets nine moderate
ones contribute in proportion.

**The kept model cannot express doubt.** With k = 1 the vote is always 1.00: the
single neighbour consulted agrees with itself. A fragment falling in the middle of
the window cloud, where the answer is close to arbitrary, arrives with the same
apparent certainty as a headlamp identified by its barium. If a confidence value
were needed in deployment, the k = 6 weighted model provides one — at the cost of
never returning the rare origins.

## Techniques used

Duplicate detection and removal with the reasoning specific to distance models,
class balance analysis on an unbalanced six-class target, per-feature distribution
plots, correlation heatmap, PCA projection for visualisation, measurement of how
much each raw column contributes to a Euclidean distance, standardisation inside a
pipeline so the scaler is refitted in every fold, stratified train/test split,
repeated stratified five-fold cross validation, the elbow method over k,
`GridSearchCV` across neighbours, Minkowski power and vote weighting, selection
under two competing criteria with `cross_val_predict` used to compare candidates
without spending the test set, confusion matrices, per-class precision/recall/F1
and macro F1, decision-region plots on the principal components, and model
persistence with `joblib`.

## Running it

From the repository root:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open the notebook from this folder and run the cells in order. `glass.csv`
lives next to the notebook and is read by relative path, so nothing has to be
configured. On Google Colab, upload the CSV to `sample_data/` and the notebook
finds it there instead.
