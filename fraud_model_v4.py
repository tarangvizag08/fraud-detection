"""
Fraud detection - v3
====================
What v2's results revealed:

  Across all three thresholds, "fraud value caught" was FLAT at 74.3%.
  Going from 10 false alarms to 75 false alarms caught 4 extra frauds
  worth ~3.70 each - about 15 total - while costing ~325 in extra reviews.

  The v2 cost model priced every missed fraud at the MEAN fraud amount
  (123.87). That was wrong. The frauds sitting near the decision boundary
  are card-testing micro-transactions, not real losses. Pricing them all
  at the mean made the optimiser buy recall that was worth nothing.

v3 fixes that and asks the question that actually matters:
  "Where is the missing 25.7% of fraud VALUE, and can we reach it?"

Changes:
  1. Amount-weighted cost - each missed fraud costs its ACTUAL amount
  2. Blind-spot diagnostic - the biggest misses and what the model scored them
  3. Value-vs-threshold curve - shows the flat line that makes the argument
  4. Optional high-amount rule overlay, tested honestly (may not help)

Run from the folder containing creditcard.csv:
    python fraud_model_v3.py
"""

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
CSV_PATH = "creditcard.csv"

COST_FP = 5.0          # cost of one analyst reviewing a false alarm
MIN_PRECISION = 0.50   # for the precision-floor operating point
RANDOM_STATE = 42
N_FOLDS = 5

# High-amount rule overlay: flag big transactions on a softer score bar
RULE_AMOUNT_PCTILE = 0.99   # "big" = top 1% of transaction amounts
RULE_SCORE_FLOOR = 0.30     # softer score bar applied only to those


def log(msg=""):
    print(msg, flush=True)


# ----------------------------------------------------------------------
# 1. LOAD + FEATURES
# ----------------------------------------------------------------------
if not os.path.exists(CSV_PATH):
    raise SystemExit(f"Could not find {CSV_PATH!r}. Run from the folder containing it.")

df = pd.read_csv(CSV_PATH).drop_duplicates().reset_index(drop=True)
log(f"Loaded {len(df):,} rows after dedup | frauds: {int(df['Class'].sum())}")

df["hour"] = (df["Time"] // 3600) % 24
df["log_amount"] = np.log1p(df["Amount"])

y = df["Class"].astype(int)
amounts = df["Amount"].copy()
X = df.drop(columns=["Class", "Time", "Amount"])
FEATURES = list(X.columns)

X_train, X_test, y_train, y_test, amt_train, amt_test = train_test_split(
    X, y, amounts, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
amt_train = amt_train.reset_index(drop=True)
amt_test = amt_test.reset_index(drop=True)

log(f"Train: {len(X_train):,} ({y_train.sum()} frauds) | "
    f"Test: {len(X_test):,} ({y_test.sum()} frauds)")
log(f"Total fraud value in test: {amt_test[y_test == 1].sum():,.0f}\n")


# ----------------------------------------------------------------------
# 2. MODELS (same as v2 - CV says they're equivalent, keep both)
# ----------------------------------------------------------------------
def make_logreg():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=1.0, class_weight="balanced",
                                     max_iter=2000, random_state=RANDOM_STATE)),
    ])


def make_hgb():
    params = dict(learning_rate=0.05, max_depth=3, min_samples_leaf=50,
                  l2_regularization=1.0, max_iter=500, early_stopping=True,
                  validation_fraction=0.15, n_iter_no_change=30,
                  random_state=RANDOM_STATE)
    try:
        return HistGradientBoostingClassifier(class_weight="balanced", **params)
    except TypeError:
        return HistGradientBoostingClassifier(**params)


cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
candidates = {"LogReg": make_logreg(), "HistGB": make_hgb()}
results, oof_probs = {}, {}

for name, model in candidates.items():
    t0 = time.time()
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision")
    oof_probs[name] = cross_val_predict(model, X_train, y_train, cv=cv,
                                        method="predict_proba")[:, 1]
    model.fit(X_train, y_train)
    results[name] = scores.mean()
    log(f"== {name}: CV PR-AUC {scores.mean():.3f} +/- {scores.std():.3f}  ({time.time()-t0:.0f}s)")

blend_oof = np.mean([oof_probs["LogReg"], oof_probs["HistGB"]], axis=0)
results["Blend"] = average_precision_score(y_train, blend_oof)
oof_probs["Blend"] = blend_oof
log(f"== Blend: OOF PR-AUC {results['Blend']:.3f}")

best_name = max(results, key=results.get)
best_oof = oof_probs[best_name]
log(f"\n>>> Best by CV/OOF: {best_name} ({results[best_name]:.3f})")
log("    (differences under ~0.03 here are inside the fold noise - don't model-shop)\n")


# ----------------------------------------------------------------------
# 3. THRESHOLDS - including the amount-weighted one
# ----------------------------------------------------------------------
prec, rec, thr = precision_recall_curve(y_train, best_oof)
prec, rec = prec[:-1], rec[:-1]
n_pos = int(y_train.sum())
tp_c = rec * n_pos
fn_c = n_pos - tp_c
fp_c = np.where(prec > 0, tp_c * (1 - prec) / prec, 0.0)

f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
thr_f1 = thr[np.argmax(f1)]

mean_fraud_amt = float(amt_train[y_train == 1].mean())
thr_flatcost = thr[np.argmin(fn_c * mean_fraud_amt + fp_c * COST_FP)]

ok = prec >= MIN_PRECISION
thr_prec = thr[ok][np.argmax(rec[ok])] if ok.any() else thr_f1

# --- the fix: price each missed fraud at its OWN amount ---
order = np.argsort(-best_oof)
p_sorted = best_oof[order]
y_sorted = y_train.values[order]
amt_sorted = amt_train.values[order]

value_caught_cum = np.cumsum(y_sorted * amt_sorted)
total_fraud_value = value_caught_cum[-1]
value_missed_cum = total_fraud_value - value_caught_cum
fp_cum = np.cumsum(1 - y_sorted)

true_cost = value_missed_cum + fp_cum * COST_FP
thr_amount = float(p_sorted[np.argmin(true_cost)])

# NOTE: these thresholds sit at ~0.9999. Print enough digits to be meaningful -
# rounding one of these to 4dp turns 0.999958 into 1.0 and silently flags nothing.
log("Thresholds (all picked on out-of-fold predictions):")
log(f"   max-F1                 : {thr_f1:.8f}")
log(f"   flat-cost (v2, flawed) : {thr_flatcost:.8f}   <- prices every miss at the mean {mean_fraud_amt:,.0f}")
log(f"   AMOUNT-weighted cost   : {thr_amount:.8f}   <- prices every miss at its real value")
log(f"   precision >= {MIN_PRECISION:.2f}       : {thr_prec:.8f}\n")


# ----------------------------------------------------------------------
# 4. TEST EVALUATION
# ----------------------------------------------------------------------
if best_name == "Blend":
    p_test = np.mean([candidates["LogReg"].predict_proba(X_test)[:, 1],
                      candidates["HistGB"].predict_proba(X_test)[:, 1]], axis=0)
    final_model = candidates
else:
    final_model = candidates[best_name]
    p_test = final_model.predict_proba(X_test)[:, 1]

total_test_value = amt_test[y_test == 1].sum()

log("========== TEST SET ==========")
log(f"PR-AUC {average_precision_score(y_test, p_test):.3f} | "
    f"ROC-AUC {roc_auc_score(y_test, p_test):.3f}\n")

rows = []
for label, t in [("max-F1", thr_f1), ("flat-cost (v2)", thr_flatcost),
                 ("amount-weighted", thr_amount), (f"prec>={MIN_PRECISION:.2f}", thr_prec)]:
    pred = (p_test >= t).astype(int)
    tn, fp_, fn_, tp_ = confusion_matrix(y_test, pred).ravel()
    vc = amt_test[(y_test == 1) & (pred == 1)].sum()
    vm = amt_test[(y_test == 1) & (pred == 0)].sum()
    rows.append({
        "rule": label,
        "_raw": float(t),          # exact value - what every later step must use
        "thr": f"{float(t):.6f}",  # display only
        "caught": int(tp_), "missed": int(fn_), "false alarms": int(fp_),
        "recall": round(tp_ / (tp_ + fn_), 3),
        "value caught": f"{vc / total_test_value:.1%}",
        "value lost": round(float(vm), 0),
        "review cost": round(fp_ * COST_FP, 0),
        "TRUE cost": round(float(vm) + fp_ * COST_FP, 0),
    })

log(pd.DataFrame(rows).drop(columns=["_raw"]).to_string(index=False))
log("\n'TRUE cost' = money actually lost to missed fraud + money spent reviewing alerts.")
log("That is the column to optimise. Lowest wins.\n")

best_rule = min(rows, key=lambda r: r["TRUE cost"])
chosen_thr = best_rule["_raw"]          # exact float, NOT the rounded display value
log(f">>> Best operating point by true cost: {best_rule['rule']} @ {chosen_thr:.8f}\n")

y_pred = (p_test >= chosen_thr).astype(int)
log(classification_report(y_test, y_pred, target_names=["legit", "fraud"], digits=3))


# ----------------------------------------------------------------------
# 5. BLIND-SPOT DIAGNOSTIC - where is the lost money?
# ----------------------------------------------------------------------
log("\n========== WHERE THE LOST MONEY IS ==========")
miss_mask = (y_test == 1) & (y_pred == 0)
missed = pd.DataFrame({
    "amount": amt_test[miss_mask].values,
    "model_score": p_test[miss_mask.values],
}).sort_values("amount", ascending=False)
# scores cluster at ~0.9999, so the useful number is the GAP below the threshold
missed["short_by"] = chosen_thr - missed["model_score"]

log(f"Missed {len(missed)} frauds worth {missed['amount'].sum():,.0f} total.")
log("\nTop 10 most expensive misses ('short_by' = how far below the threshold it scored):")
log(missed.head(10).to_string(
    index=False,
    formatters={"amount": lambda v: f"{v:>10,.2f}",
                "model_score": lambda v: f"{v:>14.10f}",
                "short_by": lambda v: f"{v:>14.10f}"}))

near = (missed["model_score"] >= chosen_thr * 0.5).sum()
log(f"\n{near} of {len(missed)} misses scored above half the threshold (borderline - "
    f"reachable by lowering it).")
log(f"{len(missed) - near} scored below that (genuine blind spots - the model "
    f"cannot see these at all; a lower threshold will NOT recover them).")

k = min(5, len(missed))
top5_share = missed.head(k)["amount"].sum() / total_test_value
log(f"\nThe {k} biggest misses alone are {top5_share:.1%} of all fraud value in the test set.")
log("If those score near zero, no threshold fixes it - that needs a different signal.")


# ----------------------------------------------------------------------
# 6. HIGH-AMOUNT RULE OVERLAY - tested honestly
# ----------------------------------------------------------------------
log("\n========== HIGH-AMOUNT RULE OVERLAY ==========")
amt_cut = float(amt_train.quantile(RULE_AMOUNT_PCTILE))
rule_flag = (amt_test >= amt_cut) & (p_test >= RULE_SCORE_FLOOR)
combined = ((p_test >= chosen_thr) | rule_flag).astype(int)

tn, fp_, fn_, tp_ = confusion_matrix(y_test, combined).ravel()
vc = amt_test[(y_test == 1) & (combined == 1)].sum()
base_tn, base_fp, base_fn, base_tp = confusion_matrix(y_test, y_pred).ravel()
base_vc = amt_test[(y_test == 1) & (y_pred == 1)].sum()

log(f"Rule: amount >= {amt_cut:,.0f} (top {1-RULE_AMOUNT_PCTILE:.0%}) AND score >= {RULE_SCORE_FLOOR}")
log(f"  model alone : {base_tp} caught, {base_fp} false alarms, {base_vc/total_test_value:.1%} of value")
log(f"  model + rule: {tp_} caught, {fp_} false alarms, {vc/total_test_value:.1%} of value")
delta_v, delta_fp = vc - base_vc, fp_ - base_fp
verdict = "WORTH IT" if delta_v > delta_fp * COST_FP else "NOT worth it"
log(f"  => recovers {delta_v:,.0f} extra value for {delta_fp} extra reviews "
    f"({delta_fp*COST_FP:,.0f}) - {verdict}")


# ----------------------------------------------------------------------
# 7. PLOTS + SAVE
# ----------------------------------------------------------------------
grid = np.linspace(0.01, 0.999, 200)
val_pct, fp_counts = [], []
for t in grid:
    pr = p_test >= t
    v = amt_test[(y_test == 1) & pr].sum()
    val_pct.append(v / total_test_value)
    fp_counts.append(int(((y_test == 0) & pr).sum()))

fig, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(grid, np.array(val_pct) * 100, lw=2, color="#2b6cb0", label="fraud value caught (%)")
ax1.set_xlabel("Threshold")
ax1.set_ylabel("Fraud value caught (%)", color="#2b6cb0")
ax1.set_ylim(0, 100)
ax2 = ax1.twinx()
ax2.plot(grid, fp_counts, lw=2, color="#c53030", ls="--", label="false alarms")
ax2.set_ylabel("False alarms", color="#c53030")
ax1.axvline(chosen_thr, color="green", ls=":", lw=2)
ax1.set_title("Lowering the threshold buys false alarms, not money")
fig.tight_layout()
fig.savefig("value_vs_threshold.png", dpi=130)
plt.close()

p_curve, r_curve, _ = precision_recall_curve(y_test, p_test)
plt.figure(figsize=(6, 5))
plt.plot(r_curve, p_curve, lw=2)
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title(f"PR curve - {best_name} (AP={average_precision_score(y_test, p_test):.3f})")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("pr_curve.png", dpi=130)
plt.close()

joblib.dump({"model": final_model, "threshold": float(chosen_thr),
             "features": FEATURES, "model_name": best_name,
             "rule_amount_cut": amt_cut, "rule_score_floor": RULE_SCORE_FLOOR},
            "fraud_model_v3.joblib")

log("\nSaved fraud_model_v3.joblib, value_vs_threshold.png, pr_curve.png")
