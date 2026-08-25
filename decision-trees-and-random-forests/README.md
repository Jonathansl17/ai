# Decision Trees and Random Forests

Two tree-based classifiers, each on its own dataset, built to answer the same
question in two different ways: what does a tree model actually learn, and where
does the probability it produces stop being a decision.

| Notebook | Model | Dataset | Task |
|---|---|---|---|
| [`credit-default-classification.ipynb`](credit-default-classification.ipynb) | Single decision tree | `credit.csv` | Will a credit applicant repay or default |
| [`income-level-classification.ipynb`](income-level-classification.ipynb) | Random forest of 300 trees | `income_evaluation.csv` | Does a person earn more than 50 000 dollars a year |

Both notebooks share the same discipline: every parameter is chosen inside the
training set, the test set is touched once, and the cut applied to the predicted
probability is treated as a decision to be made rather than as a default to be
accepted.

---

## Decision tree — credit default

### Problem

A bank has to decide, before handing out the money, whether an applicant will
repay the loan or default on it. The two mistakes do not cost the same:
approving a defaulter loses the loan, while rejecting a client who would have
paid loses only the interest, so the notebook prices them five to one.

- **Response:** `default` — 700 clients repaid, 300 defaulted (70 / 30).
- **Predictors:** 20 attributes of the applicant, mixing categorical ones such
  as `checking_balance` or `credit_history` with numeric ones such as `amount`,
  `age` and `months_loan_duration`.
- **Preparation:** no missing values and no duplicated rows; categorical columns
  turned into indicator columns with `get_dummies`, which takes the 20
  predictors to 61 columns, then an 800 / 200 stratified split.

### Dataset

`credit.csv` — the German credit dataset. 1 000 rows, 21 columns: 20 predictors
plus the response. Thirteen text columns and seven numeric ones. Nothing is
empty and no row repeats.

### Results

| Metric | Value |
|---|---|
| Depth chosen by 5-fold cross validation on AUC | 3 (mean AUC 0.6883) |
| Accuracy on training | 0.708 |
| Accuracy on test | 0.695 |
| Accuracy of answering "repaid" to everybody | 0.700 |
| AUC on test | 0.693 |
| Defaulters caught, of 60 | 0 |
| Precision / recall on defaulters | 0.000 / 0.000 |
| Cost at the cut of 0.5, five to one | 301 |
| Cost at the cheapest cut, 0.07 | 106 |

**The tree, as the standard recipe produces it, is not usable.** It catches none
of the 60 defaulters in the test set, and its accuracy of 0.695 sits below the
0.700 that answering "repaid" to every applicant would give, so it is one client
worse than having no model at all.

**It did not fail to learn — it failed to decide.** Its AUC of 0.693 says it
ranks a defaulter above a paying client about seven times out of ten, clearly
better than a coin flip. The break is in the step that turns that ranking into
an answer: a tree of depth 3 has eight leaves, produces seven distinct
probabilities on the test set, and six of them fall below 0.5, so the default cut
flags almost nobody. With 30 percent of defaulters in the data, a leaf voting by
majority rarely votes "default".

**Moving the cut is what makes it do something.** At 0.10 the recall goes from
0.000 to 0.933 and the cost from 301 to 106 units, paid for by rejecting 86 of
the 140 clients who would have repaid. Under the five to one rule that trade is
worth making, but it is a different way of running a credit desk and the bank
has to accept it explicitly.

**The tree is also extremely narrow.** The indicator of an unknown checking
balance alone takes 0.522 of the total importance, the loan duration 0.166, and
only 7 of the 61 columns are used at all — everything the dataset knows about
savings, employment, property or purpose is discarded by three levels of
questions.

The cross validation curve explains why no better tree was available: every
depth from 2 to 8 scores between 0.678 and 0.688, a range of one hundredth of
AUC, so the procedure was choosing between sizes that are all equally weak.

---

## Random forest — income level

### Problem

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
- **Preparation:** 24 duplicated rows removed so no record lands in both splits;
  the `?` that this file writes instead of leaving a cell empty is kept as an
  explicit level, since the absence itself carries signal (10.4 percent above
  50 000 for an unknown workclass against 21.9 percent in the private sector);
  `get_dummies` takes the 12 predictors to 91 columns; 26 029 / 6 508 stratified
  split.

### Dataset

`income_evaluation.csv` — the Adult dataset, extracted from the 1994 United
States census. 32 561 rows, 15 columns: 14 predictors plus the response. Six
numeric columns and nine text ones. No cell is empty, but 2 399 rows (7.4
percent) carry a `?` in `workclass`, `occupation` or `native-country`. After
removing the 24 duplicates, 32 537 records remain: 24 698 at or below 50 000 and
7 839 above (75.9 / 24.1).

### Results

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

### What is wrong with this model

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

---

## Techniques used

Class balance and duplicate checks, detection of missing values encoded as a
sentinel string, per-class histograms and box plots, categorical rate tables,
correlation heatmap and pairplot, redundancy and leakage analysis of individual
columns, one-hot encoding, stratified train/test split, 5-fold cross validation
for hyperparameter selection, out-of-bag scoring, Gini impurity decision trees,
bagged random forests, impurity and permutation feature importances, confusion
matrices, precision/recall/F1, ROC curves and AUC, cost-based and F1-based
threshold selection with `cross_val_predict`, and model persistence with
`joblib`.

## Running it

From the repository root:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open either notebook from this folder and run the cells in order. The
datasets live next to the notebooks and are read by relative path, so nothing has
to be configured. On Google Colab, upload the `.csv` files of this folder to
`sample_data/` and the notebooks find them there instead.
