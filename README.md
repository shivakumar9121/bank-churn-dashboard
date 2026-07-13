# Bank Customer Churn - Predictive Modeling & Risk Scoring

This is my project for the "Predictive Modeling and Risk Scoring for Bank
Customer Churn" assignment (Unified Mentor - European Central Bank brief).

Basic idea: instead of just looking back at who churned and why, build a
model that scores every customer with a churn probability *before* they
leave, so retention teams can actually do something about it.

Dataset is `European_Bank.csv` - 10,000 customers from a European bank
(France/Spain/Germany), ~20% churn rate.

## What's in here

- `data/` - the raw csv
- `notebooks/EDA.ipynb` - EDA, already run so you can see the plots without
  re-executing anything
- `src/data_prep.py` - cleaning + feature engineering, used by both the
  training script and the app
- `src/train_model.py` - trains 5 models and picks the best one by ROC-AUC
- `src/explain.py` - SHAP explainability, saves plots to `outputs/`
- `app/app.py` - the Streamlit app (risk calculator, prob distribution,
  feature importance, what-if simulator)
- `models/` - saved model + scaler + metrics (already generated, but you can
  regenerate by re-running the scripts below)
- `reports/` - research paper + executive summary
- `outputs/` - all the chart images used in the report

## How to actually run this

Install everything first:
```
pip install -r requirements.txt
```

Then, from the project root:

```
python src/train_model.py      # trains all 5 models, saves the best one to models/
python src/explain.py          # runs SHAP, saves plots to outputs/
streamlit run app/app.py       # launches the dashboard
```

That's it. `models/` already has everything saved from my last run so you
technically don't have to retrain before opening the app - but if you're
submitting this as your own project I'd recommend running it yourself at
least once so the numbers are freshly generated on your machine.

If you want to redo the EDA notebook:
```
jupyter nbconvert --to notebook --execute --inplace notebooks/EDA.ipynb
```

One thing to be careful about - all these scripts assume you're running
them from the project root folder (the one this README is in), because the
paths to `data/` and `models/` are relative. If you cd into `src/` first and
run `python train_model.py` from there it'll break.

## Modeling notes

Dropped `CustomerId`, `Surname` and `Year` since they don't carry any
predictive signal (Year is literally constant across the whole dataset).
One-hot encoded Geography and Gender, scaled the numeric columns.

Added a few engineered features since the assignment specifically asked
for them:
- Balance / Salary ratio
- Products / (Tenure + 1) - a rough "product density"
- IsActiveMember x NumOfProducts - this one turned out to matter a lot
- Age x Tenure

Trained Logistic Regression as a baseline, then Decision Tree, Random
Forest, Gradient Boosting and XGBoost. Used class weighting on all of them
since churn is imbalanced (~80/20). Picked the winner by ROC-AUC on a
stratified 80/20 test split - Gradient Boosting won in my run (~0.87
ROC-AUC), but don't be surprised if it comes out slightly different for you
since there's some randomness in training even with a fixed seed depending
on package versions.

For explainability, used SHAP's TreeExplainer on the winning model. Age and
NumOfProducts consistently come out as the two biggest drivers, which
matches what shows up in the EDA too.

## A few things I noticed while doing the EDA (worth mentioning if you
write up your own report)

- Germany churns a lot more than France or Spain despite having a similar
  number of customers - not something I expected going in.
- Customers with 3-4 products actually churn *more* than customers with 1-2,
  which is backwards from what you'd normally assume (more products = more
  sticky customer). This is one of the stronger signals in the whole
  dataset.
- Inactive members churn at roughly double the rate of active ones. Out of
  everything in the data, this is probably the most useful one for a bank
  to act on since it's something they can actually influence.

## Known limitations / stuff I'd improve with more time

- No hyperparameter tuning beyond some manual guesses - a proper grid
  search or Optuna run would probably squeeze out a bit more performance.
- The "what-if simulator" in the app assumes you can independently change
  one feature at a time, which isn't totally realistic (e.g. changing
  NumOfProducts probably correlates with other things in reality).
- Only one train/test split was used for the final numbers - k-fold CV
  would give a more reliable estimate, it's mentioned as optional in the
  brief so I skipped it to save time.
