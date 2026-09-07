# Decision Trees

A single decision tree trained to answer, before the money leaves the bank,
whether a credit applicant will repay or default — and to show where the
probability a tree produces stops being a decision.

## Problem

A bank has to decide, before handing out the money, whether an applicant will
repay the loan or default on it. The two mistakes do not cost the same:
approving a defaulter loses the loan, while rejecting a client who would have
paid loses only the interest, so the notebook prices them five to one.

- **Response:** `default` — 700 clients repaid, 300 defaulted (70 / 30).
- **Predictors:** 20 attributes of the applicant, mixing categorical ones such
  as `checking_balance` or `credit_history` with numeric ones such as `amount`,
  `age` and `months_loan_duration`.
- **Model:** a Gini impurity decision tree, its depth chosen by 5-fold cross
  validation on AUC inside the training set.
- **Preparation:** no missing values and no duplicated rows; categorical columns
  turned into indicator columns with `get_dummies`, which takes the 20
  predictors to 61 columns, then an 800 / 200 stratified split.

## Dataset

`credit.csv` — the German credit dataset. 1 000 rows, 21 columns: 20 predictors
plus the response. Thirteen text columns and seven numeric ones. Nothing is
empty and no row repeats. Class balance: 700 repaid against 300 defaulted.

## Results

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

## Techniques used

Class balance and duplicate checks, per-class histograms and box plots,
categorical rate tables, correlation heatmap, one-hot encoding, stratified
train/test split, 5-fold cross validation for depth selection, Gini impurity
decision trees, tree plotting and rule reading, impurity feature importances,
confusion matrices, precision/recall/F1, ROC curves and AUC, cost-based
threshold selection under an asymmetric five to one loss, and model persistence
with `joblib`.

## Running it

From the repository root:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open `credit-default-classification.ipynb` from this folder and run the
cells in order. The dataset lives next to the notebook and is read by relative
path, so nothing has to be configured. On Google Colab, upload `credit.csv` to
`sample_data/` and the notebook finds it there instead.
