# Machine Learning Notebooks

Applied machine learning projects written in Python, each one following the
CRISP-DM cycle end to end: business understanding, data understanding, data
preparation, modeling, evaluation and deployment. Every notebook explains what
each plot shows, why each decision was taken and what the resulting numbers mean,
so the reasoning is readable and not only the code.

## Projects

| Project | Algorithms | Dataset | Task |
|---|---|---|---|
| [Linear Regression](linear-regression) | Simple and multiple ordinary least squares | `insurance.csv`, `weatherAUS.csv` | Predict an insurance premium from BMI, and the minimum temperature of a day from weather variables |
| [Decision Trees and Random Forests](decision-trees-and-random-forests) | Decision tree, random forest | `credit.csv`, `income_evaluation.csv` | Classify whether a credit applicant will repay or default, and whether a person earns more than 50 000 dollars a year |

## Running the notebooks

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Each notebook reads its dataset from its own folder, and falls back to the Colab
`sample_data` path when it is not there, so it runs both locally and in Google
Colab without editing anything.

## Author

Jonathan Sancho Loaiza — Computer Engineering, Instituto Tecnológico de Costa Rica.
