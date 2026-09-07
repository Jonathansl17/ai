# Support Vector Machines

A triage filter for cardiotocograph traces: given the 21 numbers a monitoring
program extracts from a recording of a fetal heart rate, decide whether the trace
is normal, suspect or pathological — and, more usefully, decide which traces an
obstetrician should read first.

| Notebook | Model | Dataset | Task |
|---|---|---|---|
| [`fetal-health-classification.ipynb`](fetal-health-classification.ipynb) | SVM with a radial basis kernel, C = 100, gamma = 0.1 | `fetal_health.csv` | Classify a cardiotocograph trace as normal, suspect or pathological |

## Problem

A cardiotocograph is the belt strapped around a pregnant woman's abdomen in the
last weeks of pregnancy and during labour. It records the heart rate of the fetus
and the contractions of the uterus at once, and an obstetrician reads the two
together to decide whether the fetus is coping. Reading a trace well takes years,
agreement between two experienced readers is far from perfect, and the traces
arrive at three in the morning in a ward where the person on duty is doing four
other things.

The consequences are not symmetric. A fetus in distress read as healthy can end in
permanent injury or death; a healthy fetus read as distressed ends in an extra
hour of monitoring or, at worst, a caesarean that was not needed. That asymmetry
is what shapes the evaluation section of the notebook.

- **Response:** `fetal_health`, the consensus of three expert obstetricians on a
  three point scale — normal, suspect, pathological.
- **Predictors:** 21 features the SisPorto software extracts from each trace —
  rates of accelerations, decelerations and contractions per second, measures of
  short and long term variability of the heart rate, and summaries of the
  histogram of the rate over the recording.
- **Preparation:** no empty cells; 13 duplicated recordings removed so no row
  could be split across the training and test sets and be classified from memory;
  all 21 columns standardised inside the pipeline, then a 1584 / 529 stratified
  split.

## Dataset

`fetal_health.csv` — the cardiotocography dataset of Ayres de Campos and
colleagues at the University of Porto, distributed through the UCI repository.
**2126 rows, 22 columns**: 21 predictors plus the response, all `float64`, with no
empty cell anywhere. Thirteen rows are duplicates, leaving 2113.

| Diagnosis | Recordings | Share |
|---|---|---|
| Normal | 1646 | 77.9 % |
| Suspect | 292 | 13.8 % |
| Pathological | 175 | 8.3 % |

Answering "normal" to every trace would be right 77.9 % of the time, so an
accuracy of 0.78 is the starting point and not a result. After the split, 44
pathological recordings are left to judge the model on, and one recording moves
that recall by more than two points.

## Results

Three kernels were each tuned on their own grid and compared on the same five
folds, judged on macro F1:

| Kernel | Best macro F1 | Configuration |
|---|---|---|
| Radial basis | 0.8749 | C = 100, gamma = 0.1 |
| Polynomial | 0.8640 | C = 10, degree 3, gamma = 0.1 |
| Linear | 0.8003 | C = 100 |

On the 529 test recordings, taking the highest scoring class:

| Metric | Value |
|---|---|
| Test accuracy | 0.911 |
| Test macro F1 | 0.834 |
| Answering "normal" to everything | 0.779 |
| Cross validated macro F1 on the training set | 0.875 |
| Macro average AUC | 0.974 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.958 | 0.949 | 0.954 | 412 |
| Suspect | 0.699 | 0.795 | 0.744 | 73 |
| Pathological | 0.868 | 0.750 | 0.805 | 44 |

**The number that matters is not in that table.** As a three way classifier the
model sends **6 of the 44 pathological recordings to Normal**, and in this
application those six are the whole cost. Five more are called suspect, which in
practice still raises an alarm, so 33 are correctly named, 5 are flagged and 6 go
through as healthy.

**Those six are a decision problem, not a modelling one.** The one-against-rest
AUC for the pathological class is **0.982** — the model is not failing to see
those recordings, it is seeing them and the default rule, which assigns each
record to its highest scoring class, placing them on the wrong side. Where the
ranking is that good and the decision that unbalanced, the fix is the rule.

**So the rule was rebuilt.** The three way answer is replaced by a two way
decision — send this trace to a specialist, or let it through — taken by comparing
the model's probability for normal against a threshold. The threshold was chosen
on the training folds alone, with each record scored by the folds that did not
contain it, as the smallest value that missed no pathological recording there:
0.80.

| | Highest scoring class wins | Normal score below 0.80 |
|---|---|---|
| Pathological caught, of 44 | 38 | **43** |
| Pathological let through | 6 | **1** |
| Suspect caught, of 73 | 62 | 69 |
| Normal sent for review, of 412 | 21 (5.1 %) | 36 (8.7 %) |

Nothing was retrained between those two columns. The price of finding five more
babies in distress is fifteen extra second looks per four hundred healthy
recordings. Eliminating the last missed case would mean raising the threshold to
0.90, which costs 68 false alarms instead of 36.

The threshold reached a perfect score on the training folds and fell one case
short on the test set, which is the expected behaviour of a value chosen to *just
barely* achieve something on one sample, and is stated rather than smoothed over.

**Scaling was the single most consequential decision, and it was measured rather
than assumed.** At the same C and gamma, on the same folds:

| Variant | Accuracy | Macro F1 | Recall Suspect | Recall Pathological |
|---|---|---|---|---|
| Scaled, balanced class weights | 0.9312 | 0.8757 | 0.767 | 0.840 |
| Scaled, no class weights | 0.9293 | 0.8726 | 0.767 | 0.832 |
| Not scaled, `gamma='scale'` | 0.8592 | 0.7965 | 0.840 | 0.916 |
| Not scaled, tuned gamma | 0.7923 | 0.3816 | 0.050 | 0.092 |

The reason is in the units: three histogram columns carry 64 % of every raw
distance while the six event-rate columns carry 0.00004 %, and one of those six,
`prolongued_decelerations`, is the feature most correlated with the diagnosis in
the whole file. A kernel computed on raw values therefore cannot see the strongest
predictor at all. The class weighting, by contrast, was worth 0.3 points of macro
F1 and is reported as such rather than credited with more.

**The curved boundary is worth seven and a half points over the flat one**, and
the exploration predicted it: `percentage_of_time_with_abnormal_long_term_variability`
is highest for the *suspect* class rather than the pathological one, so the three
classes do not lie along a line in that direction. A flat boundary cannot express
that; a radial one wraps around the normal region and leaves both abnormal classes
outside it in different directions. That the polynomial kernel lands within a
point of the radial one suggests the remaining limit is in the data.

**The C and gamma surface is flat near its top.** The grid runs from 0.686 in the
smooth corner (C = 0.1, gamma = 0.001) to 0.875 at the best cell, with neighbours
within one or two points of it, and falls back to 0.857 at C = 1000 where the
boundary has begun to wrap itself around individual training records.

### One caveat that belongs with the numbers

`SVC` produces probabilities by fitting a calibration on top of its decision
function, through an internal cross validation, so the class with the highest
probability is not always the class `predict` returns. On this test set the two
disagree on 10 of 529 records, under 2 %. The confusion matrix and the
classification report come from `predict`; the ROC curves and the threshold rule
are read off `predict_proba`.

The model is not a replacement for the obstetrician who reads the trace. It is a
filter deciding which traces get read first, and the number it should be trusted
with is the probability, not the label.

## Techniques used

Duplicate detection and removal, class balance analysis on a three-class
imbalanced target, per-class box plots and event-rate tables, correlation heatmap
with high-correlation pair extraction, PCA projection for visualisation,
measurement of how much each raw column contributes to a kernel distance,
standardisation inside a pipeline so the scaler is refitted in every fold,
stratified train/test split, stratified five-fold cross validation, linear,
polynomial and radial basis kernels each tuned on their own grid, `GridSearchCV`
over C, gamma and degree, a C/gamma heat map, `class_weight='balanced'` measured
against no weighting, ablation of the scaler, `cross_val_predict` for out-of-fold
comparisons and out-of-fold probabilities, confusion matrices in counts and row
shares, per-class precision/recall/F1 and macro F1, one-against-rest ROC curves
and AUC, threshold selection on the training folds for an asymmetric-cost decision
rule, and model persistence with `joblib`.

## Running it

From the repository root:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open the notebook from this folder and run the cells in order.
`fetal_health.csv` lives next to the notebook and is read by relative path, so
nothing has to be configured. On Google Colab, upload the CSV to `sample_data/`
and the notebook finds it there instead.
