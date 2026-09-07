# HOG and Colour Features with a Support Vector Machine

Given a photograph of a face, decide whether the person is wearing a mask. It is
the check a camera at the entrance of a building had to make thousands of times a
day during the pandemic, and it is a decision people make in a glance and cannot
write down as a rule.

| Notebook | Model | Data | Task |
|---|---|---|---|
| [`face-mask-image-classification.ipynb`](face-mask-image-classification.ipynb) | SVM with a radial basis kernel, C = 10, on 1821 features built from each image | 7553 photographs of faces | Decide whether the face in a photograph is wearing a mask |

## Problem

The difficulty is not the classifier, it is that a support vector machine reads a
row of numbers and a photograph is not one. No measurement of an image says mask:
masks come in different colours, fabrics and shapes, faces appear at any angle and
under any light, and the photographs were taken by different cameras at different
distances. What the two classes really differ in is what occupies the lower half of
the face, a nose and a mouth against a smooth surface crossed by a border.

So most of the work is designing a description of an image that keeps that
difference and drops everything else. Every photograph is brought to 64 by 64
pixels and turned into 1821 numbers in two blocks:

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

## Dataset

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

## Results

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

### Where it fails, and why

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

## Techniques used

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

## Running it

From the repository root:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open the notebook from this folder and run the cells in order. The image
collection is not in the repository because of its size, so the first data cell
downloads and unpacks it, and skips the download when the folder is already there.
It works the same from a clone and on Google Colab, with nothing to configure.
