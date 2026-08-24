# Linear Regression

Two ordinary least squares models built with `statsmodels` and validated against
the assumptions the method rests on, so every reported p-value and confidence
interval is one that is actually allowed to be read.

## Simple linear regression — insurance premium from BMI

Notebook: [`simple-linear-regression.ipynb`](simple-linear-regression.ipynb)
Dataset: `insurance.csv` — 1,338 clients of a health insurer, 7 columns.

An insurer charges every client a different annual premium. The question is
whether the body mass index alone explains the amount billed.

- **Response:** `charges`, the amount billed in dollars.
- **Predictor:** `bmi`, the body mass index.
- **Model:** `log(charges) ~ bmi`.

The regression on the raw `charges` fails its diagnostics: the residuals pile up
on one side and their spread grows with the fitted value, which invalidates the
inference. Taking the logarithm of the response fixes both problems, and only
then is the significance test trustworthy.

**Result.** The relationship is real and positive — each extra BMI point is
associated with roughly 1.9 % more on the premium, with a p-value far below
0.05 — but the R² stays under 2 %. BMI genuinely moves the premium and explains
almost none of it, so the model is honest evidence of an effect and useless as a
price predictor on its own.

## Multiple linear regression — minimum temperature from weather variables

Notebook: [`multiple-linear-regression.ipynb`](multiple-linear-regression.ipynb)
Dataset: `weatherAUS.csv` — 145,460 daily records from Australian weather
stations, 23 columns.

- **Response:** `MinTemp`, the minimum temperature of the day in °C.
- **Predictors:** the numeric variables of the daily record — temperatures,
  humidity, pressure, wind speed and rainfall.
- **Models compared:** Model A uses every numeric variable; Model B uses six
  variables chosen to remove the redundant ones.

| Model | Test R² | Mean absolute error | Worst VIF |
|---|---|---|---|
| A — all numeric variables | 0.908 | 1.52 °C | 55.7 |
| B — six selected variables | 0.883 | 1.73 °C | below 10 |

**Result.** Model A is the more accurate of the two, by 2.6 points of R² and
0.22 °C. Model B is the one recommended anyway: using every numeric variable
produces severe multicollinearity — `Temp3pm` and `MaxTemp` reach VIF values of
55.7 and 46.7 against a threshold of 10 — which makes the individual
coefficients unstable and unreadable even while the predictions stay good. Model
B trades 0.22 °C of error for coefficients that mean something.

The fitted model is stripped of its 46,588 training rows with `remove_data()`
before being saved, since what gets deployed is the equation and not the data.

## Techniques used

Pearson and Spearman correlation, Shapiro normality test, residual and
homoscedasticity diagnostics, logarithmic transformation of the response,
variance inflation factor for multicollinearity, train/test split, R² / MAE /
RMSE evaluation, and model persistence with `joblib`.

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
