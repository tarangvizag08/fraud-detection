# Fraud Detection — Cost-Aware Threshold Selection

## What this does

Trains a fraud classifier on the ULB credit card fraud dataset (the standard Kaggle one — 283k European card transactions, 473 of them fraud) and picks its decision threshold by actual business cost instead of the usual F1 score. A missed fraud and a false alarm don't cost a bank the same amount, so the metric shouldn't treat them the same either.

## Why "train a model, report accuracy" isn't the real task here

With 473 frauds out of 283k transactions, a model that predicts "never fraud" is already 99.8% accurate and completely useless. The actual question is where to draw the line between "flag this for review" and "let it through," and what getting that line wrong costs in real money — not how high one number can go.

## What I found

- LogReg and a regularized HistGradientBoosting model land within noise of each other on cross-validated PR-AUC (~0.756 both) — more model complexity didn't buy anything on this data.
- Pricing every missed fraud at the *mean* fraud amount pushes the threshold toward catching tiny card-testing transactions worth a few rupees each, at the cost of a lot of extra false alarms. Pricing each miss at its *actual* amount gives a very different, more defensible answer.
- Even at the best threshold, around a quarter of fraud value goes uncaught — and most of that isn't sitting just below the cutoff. It's transactions the model scores near zero: a real blind spot in the model, not a tuning problem a threshold can fix.

## Files

- `fraud_model_v4.py` — trains LogReg and HistGB, compares them on cross-validated PR-AUC, selects a threshold by actual (amount-weighted) cost rather than F1, and reports fraud *value* caught alongside fraud *case count* caught.
- `fraud_rule_search.py` — searches a simple amount-plus-score business rule on top of the model, evaluated out-of-fold so the result isn't just overfit to one test split.

## Running it

Needs `creditcard.csv` (ULB/Kaggle credit card fraud dataset) in the same folder — not included here, it's ~144MB and not mine to redistribute.

```bash
pip install scikit-learn pandas numpy matplotlib joblib
python fraud_model_v4.py
python fraud_rule_search.py
```

## Stack

Python, scikit-learn, pandas, matplotlib.
