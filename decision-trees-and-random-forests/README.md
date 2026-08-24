# Decision Trees and Random Forests

Credit risk classification on the German credit dataset, comparing a single
decision tree against a random forest, and turning the resulting probabilities
into a decision using the cost the bank actually pays rather than the default
0.5 cut.

Notebook: [`credit-default-classification.ipynb`](credit-default-classification.ipynb)
Dataset: `credit.csv` — 1,000 applicants of a German bank, 20 predictors plus
the response. No missing values, no duplicated rows.

## Problem

A bank has to decide whether an applicant will repay a loan or default on it.
The two mistakes are not symmetric: approving a defaulter costs far more than
turning away a client who would have repaid.

- **Response:** `default` — 700 clients repaid, 300 defaulted (70/30).
- **Predictors:** 20 attributes of the applicant, mixing categorical ones such
  as `checking_balance` or `credit_history` with numeric ones such as `amount`,
  `age` and `months_loan_duration`.
- **Preparation:** categorical columns turned into indicator columns with
  `get_dummies`, then an 800/200 stratified train/test split.

## Models

| Model | Accuracy | AUC | Defaulters caught (of 60) |
|---|---|---|---|
| Decision tree — depth 3, chosen by 5-fold cross validation | 0.695 | 0.693 | 0 |
| Random forest — 500 trees | 0.775 | 0.818 | 27 |

**The tree is not usable.** It catches none of the 60 defaulters in the test
set, its precision and recall on that class are both 0.000, and its accuracy of
0.695 is below the 0.70 that answering "repaid" to everybody would give. The
cross validation curve explains why: every depth from 2 to 8 scores between
0.678 and 0.688, a range of one hundredth of AUC, so the procedure was choosing
between sizes that are all equally weak rather than finding a good one.

**The forest works and is still not enough alone.** The gap of 0.125 in AUC is
the number that matters, since it says the forest orders clients by risk more
reliably wherever the decision cut is later placed. At the 0.5 cut it still lets
33 of 60 defaulters through, because the variables of this dataset overlap
heavily between the two classes and no single one separates a defaulter from a
paying client.

## Choosing the decision threshold

Both models return a probability of default, not a yes or no, and the accuracy
above comes from cutting that probability at 0.5. The notebook recomputes the
errors across every cut from 0.05 to 0.95 under a cost rule of five units per
defaulter approved against one unit per good client turned away. The resulting
cost curve is U shaped, and its minimum sits well below 0.5 — the forest is
unchanged, only the point at which its probabilities become a decision is set
from the cost of the mistake instead of from an arbitrary default.

## Techniques used

Class balance and duplicate checks, per-class histograms and box plots,
categorical crosstabs of repayment rate, correlation heatmap and pairplot,
one-hot encoding, stratified train/test split, 5-fold cross validation for depth
selection, Gini impurity trees, bagged forests, feature importances, confusion
matrices, precision/recall/F1, ROC curves and AUC, and cost-based threshold
selection.

## Running it

From the repository root:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open the notebook from this folder and run the cells in order. The datasets
live next to the notebooks and are read by relative path, so nothing has to be
configured. On Google Colab, upload the `.csv` files of this folder to
`sample_data/` and the notebooks find them there instead.
