# Support Vector Machines

Two projects built on the same algorithm and pointed at inputs that could hardly be
more different. One reads a table of 21 numbers a monitoring program already
extracted from a signal. The other reads photographs, where no numbers exist until
the notebook invents them, and most of the work is deciding what to measure about
an image so that a model which only understands rows of numbers can separate the
classes.

| Notebook | Model | Data | Task |
|---|---|---|---|
| [`fetal-health-classification.ipynb`](fetal-health-classification.ipynb) | SVM with a radial basis kernel, C = 100, gamma = 0.1 | `fetal_health.csv`, 2126 rows of 21 numeric features | Classify a cardiotocograph trace as normal, suspect or pathological |
| [`face-mask-image-classification.ipynb`](face-mask-image-classification.ipynb) | SVM with a radial basis kernel, C = 10, on 1821 features built from each image | 7553 photographs of faces | Decide whether the face in a photograph is wearing a mask |

---

## Cardiotocograph triage

A triage filter for cardiotocograph traces: given the 21 numbers a monitoring
program extracts from a recording of a fetal heart rate, decide whether the trace
is normal, suspect or pathological — and, more usefully, decide which traces an
obstetrician should read first.

### Problem

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

### Dataset

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

### Results

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

#### One caveat that belongs with the numbers

`SVC` produces probabilities by fitting a calibration on top of its decision
function, through an internal cross validation, so the class with the highest
probability is not always the class `predict` returns. On this test set the two
disagree on 10 of 529 records, under 2 %. The confusion matrix and the
classification report come from `predict`; the ROC curves and the threshold rule
are read off `predict_proba`.

The model is not a replacement for the obstetrician who reads the trace. It is a
filter deciding which traces get read first, and the number it should be trusted
with is the probability, not the label.

### Techniques used

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

---

## Face mask detection in photographs

Given a photograph of a face, decide whether the person is wearing a mask. It is
the check a camera at the entrance of a building had to make thousands of times a
day during the pandemic, and it is a decision people make in a glance and cannot
write down as a rule.

### Problem

The difficulty is not the classifier, it is that a support vector machine reads a
row of numbers and a photograph is not one. No measurement of an image says mask:
masks come in different colours, fabrics and shapes, faces appear at any angle and
under any light, and the photographs were taken by different cameras at different
distances. What the two classes really differ in is what occupies the lower half of
the face, a nose and a mouth against a smooth surface crossed by a border.

So the whole first half of the project is designing a description of an image that
keeps that difference and drops everything else. Every photograph is brought to 64
by 64 pixels and turned into 1821 numbers in two blocks:

- **Shape, 1764 numbers.** A histogram of oriented gradients: the image is turned
  to grey, cut into squares of 8 by 8 pixels, and the direction in which the
  brightness changes is measured at every pixel and collected into nine bins per
  square. Squares are grouped into blocks of 2 by 2 and each block is normalised on
  its own, which is what makes the description survive a change of lighting.
- **Colour, 57 numbers.** Three 16 bin histograms of hue, saturation and value,
  plus the mean colour of the upper half of the picture, of the lower half, and the
  difference between them, since a mask is a flat region of one colour placed
  exactly there.

- **Response:** whether the face is covered, with the covered class treated as
  positive.
- **Preparation:** every image resized to a common 64 by 64, so that files ranging
  from 26 by 37 to 5760 by 5412 pixels all produce rows of the same length. Nine
  files that are not plain RGB are converted. All 1821 columns are standardised
  inside the pipeline, so the scaler is refitted on every training fold, then a
  75 / 25 stratified split leaves 5664 images to learn from and 1889 to be judged
  on.

### Dataset

A collection of 7553 photographs of faces, split into two folders that carry the
label. It is not in this repository, since it is 163 MB, well past what belongs in
a git history. The notebook downloads and unpacks it by itself on the first run and
skips the download when the folder is already there, so nothing has to be set up by
hand.

| Class | Photographs | Share |
|---|---|---|
| Without a mask | 3828 | 50.7 % |
| With a mask | 3725 | 49.3 % |

The classes are almost the same size, so answering the larger one to every
photograph scores 0.5066 and the floor of the evaluation is essentially a coin
toss, which is the most demanding version of that baseline.

### Results

On the 1889 test photographs, none of which took part in any decision:

| Metric | Value |
|---|---|
| Test accuracy | 0.9052 |
| F1 of the mask class | 0.9030 |
| Area under the ROC curve | 0.9710 |
| Answering the larger class to everything | 0.5066 |
| Mistakes | 179 of 1889 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Without a mask | 0.8986 | 0.9164 | 0.9074 | 957 |
| With a mask | 0.9124 | 0.8938 | 0.9030 | 932 |

**The accuracy is not the number this problem is judged by.** The two mistakes cost
different things: a masked person stopped and asked again is an inconvenience, an
uncovered face admitted as covered is the failure the check exists to prevent. The
model admitted **80 uncovered faces of 957** and stopped **99 covered faces of
932**, so the expensive mistake is the less frequent one, about one uncovered face
in twelve, and the cheap one absorbs the difference. Accuracy hides that entirely
by adding both cells together.

**The area under the curve of 0.9710 sits well above the accuracy of 0.9052**, and
that gap is the useful part. It says the model ranks the photographs better than
its decisions suggest, so the ordering is good and the default rule, the sign of
the decision function, is not where this application should read it. Raising the
threshold would cut the 80 further at a price the curve shows to be low near that
point.

**Both halves of the description earn their place, and they were measured rather
than assumed.** The same model on the same folds, given different columns:

| Description | Columns | Accuracy | Spread over folds |
|---|---|---|---|
| Shape only | 1764 | 0.8500 | 0.0043 |
| Colour only | 57 | 0.8084 | 0.0122 |
| Shape and colour | 1821 | 0.8708 | 0.0070 |

Shape carries the problem, as intended. What is more interesting is that 57 colour
numbers reach 0.8084 on their own, far above the floor, and that adding them to the
1764 shape columns is still worth 2.1 points. Every gap there is several times the
fold to fold spread, so none of it is noise.

**The curved boundary is worth 5.4 points over the flat one**, 0.8632 against
0.8092 at the same penalty, so these classes do not sit on either side of a plane.
The penalty itself reaches a plateau rather than a peak, 0.8708 at C = 10 and the
same to four decimals at C = 100, so the smallest value on the plateau was kept and
the result does not depend on hitting one exact number.

**The model is held up by a lot of data.** There are 3266 support vectors, 57.7 %
of the training images, split almost evenly between the classes. That says the two
classes overlap heavily in this description rather than sitting apart with a few
hard cases at the frontier, and it is also why the saved model is 24.8 MB
compressed: every support vector is stored with all 1821 of its columns, and every
prediction measures the new photograph against all of them.

#### Where it fails, and why

Nine of the twelve mistakes inspected are covered faces called uncovered, and their
masks are almost never the blue or white surgical mask that dominates the
collection. A navy mask on a child, a black cloth mask worn with sunglasses, a
purple mask with a valve. That is the colour half of the description meeting the
limit the exploration predicted for it: it learned that masks are blue or white, so
another colour costs the model its second source of evidence.

The rest come from reading a whole frame instead of a face. A profile with a
watermark bar across the bottom, a crowd with several masked people at different
distances, a printed sheet of small instructional photographs. All of these are
limits of the description rather than of the classifier, and cropping to a detected
face before describing it is the single change most likely to help.

### Techniques used

Image loading and resizing to a common frame, histogram of oriented gradients with
block normalisation, HSV colour histograms and half-image colour means, feature
caching, PCA projection for visualisation, class balance analysis, measurement of
how much each raw column contributes to a kernel distance, standardisation inside a
pipeline so the scaler is refitted in every fold, stratified train/test split,
stratified three-fold cross validation on a stratified search sample, linear against
radial basis kernels, a validation curve over the penalty C, an ablation comparing
each half of the feature description on identical folds, support vector count as a
diagnostic of class overlap, confusion matrix read as two asymmetric costs,
per-class precision/recall/F1, ROC curve and AUC from the decision function,
inspection of the misclassified images, and model persistence with `joblib`.

---

## Running the notebooks

From the repository root:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open a notebook from this folder and run the cells in order.

`fetal_health.csv` lives next to its notebook and is read by relative path, so
nothing has to be configured. On Google Colab, upload the CSV to `sample_data/` and
the notebook finds it there instead.

The image collection is not in this repository because of its size. Its notebook
downloads and unpacks it on the first run and skips the download when the folder is
already present, so it works the same from a clone and on Colab without setting
anything up.
