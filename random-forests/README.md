# Random Forests

A forest of 300 trees that estimates whether a person earns more than 50 000
dollars a year from the description a census holds about them, and picks the cut
applied to its probability instead of accepting the default one.

## Problem

Income is useful to know and hard to ask for directly. A bank offering a premium
product, an office planning a subsidy and a survey correcting for non-response
all hold a description of the person — age, studies, occupation, marital
situation, hours worked — but not what they earn.

- **Response:** `income`, recoded to 1 for the people above 50 000 dollars a
  year, who are the minority class at 24.1 percent.
- **Predictors:** 14 attributes, reduced to 12. `education` is dropped because
  it is the same variable as `education-num` written as a name instead of an
  ordered code, and `fnlwgt` because it is a census sampling weight that
  describes the survey design rather than the person — its correlation with the
  response is −0.0095 and its AUC used alone as a score is 0.4927.
- **Model:** a bagged random forest, its number of trees read off the
  out-of-bag curve and its `min_samples_leaf` and `max_features` chosen by
  5-fold cross validation on AUC.
- **Preparation:** 24 duplicated rows removed so no record lands in both splits;
  the `?` that this file writes instead of leaving a cell empty is kept as an
  explicit level, since the absence itself carries signal (10.4 percent above
  50 000 for an unknown workclass against 21.9 percent in the private sector);
  `get_dummies` takes the 12 predictors to 91 columns; 26 029 / 6 508 stratified
  split.

## Dataset

`income_evaluation.csv` — the Adult dataset, extracted from the 1994 United
States census. 32 561 rows, 15 columns: 14 predictors plus the response. Six
numeric columns and nine text ones. No cell is empty, but 2 399 rows (7.4
percent) carry a `?` in `workclass`, `occupation` or `native-country`. After
removing the 24 duplicates, 32 537 records remain: 24 698 at or below 50 000 and
7 839 above (75.9 / 24.1).

## Results

| Metric | At the cut of 0.5 | At the cut of 0.36 |
|---|---|---|
| Accuracy | 0.860 | 0.848 |
| Precision on the class above 50K | 0.769 | 0.664 |
| Recall on the class above 50K | 0.596 | 0.747 |
| F1 on the class above 50K | 0.672 | 0.703 |
| People above 50K found, of 1 568 | 935 | 1 172 |
| People wrongly flagged | 281 | 593 |

| Metric | Value |
|---|---|
| Trees, chosen by the out-of-bag curve | 300 |
| `min_samples_leaf`, by 5-fold CV on AUC | 2 |
| `max_features`, by 5-fold CV on AUC | `sqrt` — 9 columns per node |
| Accuracy on training | 0.8974 |
| Out-of-bag accuracy | 0.8655 |
| Accuracy on test | 0.8596 |
| Accuracy of answering "≤50K" to everybody | 0.7591 |
| AUC on test | 0.9152 |

**The forest works.** Its AUC of 0.9152 means that, given one person from each
class, it ranks them correctly nine times out of ten, and its test accuracy is
10 points above the answer that requires no model. The out-of-bag estimate
announced that result before the test set was touched — 0.8655 against the
0.8596 measured afterwards.

**Accuracy is still the wrong number to judge it by.** At the default cut it
finds only 935 of the 1 568 high earners, a recall of 0.596, and lets 633
through. Missing four in ten costs almost nothing in accuracy, because that
class is a quarter of the data, and it costs everything to whoever wanted to find
them.

**The cut was therefore chosen, not accepted.** `cross_val_predict` scores every
training record with the folds that did not contain it, and the cut maximising
the F1 of the minority class on those scores is 0.36. On the test set that cut
scores an F1 of 0.703, against the 0.706 of the best cut available on the test
set itself — the procedure landed one hundredth away from the best it could have
done without ever looking at those records.

**Two parameters mattered, one did not.** Leaving `min_samples_leaf` at its
default of 1 costs 0.0245 of AUC, because with 91 indicator columns a leaf can
almost always isolate a single person and the trees end up voting on individuals
they memorised. Letting every node see every column costs 0.0142, since removing
the column randomness turns 300 trees into 300 copies of nearly the same tree.
The number of trees was flat from 200 onwards: 200 trees give 0.8501 of
out-of-bag accuracy and 500 give 0.8511, so the last 300 buy one thousandth.

**Two importance measures were used and they disagree usefully.** By impurity the
ranking is capital gain 0.150, education code 0.146, married with a civilian
spouse 0.121, age 0.107. By permutation on the test set it is capital gain
(0.0361 of AUC lost), education code (0.0235), married with a civilian spouse
(0.0223), age (0.0199). The exception is `relationship_Husband`, fifth by
impurity and seventh by permutation: shuffling one of two columns that state the
same fact costs little, because the other one remains.

## What is wrong with this model

The README would be dishonest without these. The data is the 1994 census and the
threshold is 50 000 dollars of that year, so the model answers a question about a
population that no longer exists. The 159 records whose capital gain is exactly
99 999 are the census top code rather than a real amount, and all of them are
above 50 000, so a label of the answer travels inside the strongest predictor.
And the gap in the data between men and women — 30.6 percent above 50 000 against
11.0 percent — is reproduced by the model, which will assign a lower probability
to a woman with otherwise identical attributes. The recommendation in the
notebook is to deploy the forest as a scoring tool and never as a verdict about a
person.

## Techniques used

Class balance and duplicate checks, detection of missing values encoded as a
sentinel string, per-class histograms and box plots, categorical rate tables,
correlation heatmap and pairplot, redundancy and leakage analysis of individual
columns, one-hot encoding, stratified train/test split, out-of-bag scoring,
5-fold cross validation for hyperparameter selection, bagged random forests,
impurity and permutation feature importances, confusion matrices,
precision/recall/F1, ROC curves and AUC, F1-based threshold selection with
`cross_val_predict`, and model persistence with `joblib`.

## Running it

From the repository root:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open `income-level-classification.ipynb` from this folder and run the cells
in order. The dataset lives next to the notebook and is read by relative path, so
nothing has to be configured. On Google Colab, upload `income_evaluation.csv` to
`sample_data/` and the notebook finds it there instead.
