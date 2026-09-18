"""
Fraud detection - business rule layer search
============================================
Your v3 run showed two things worth chasing:

  1. 68% of lost value sits in "blind spots" - frauds the model scores
     below 0.30. No threshold reaches those. Example: a 1,096.99 fraud
     scored 0.1362.

  2. The hand-picked rule (amount>=1,016 AND score>=0.30) missed that
     exact fraud by 0.164 on the score floor. The idea wasn't wrong,
     the two numbers were.

So instead of guessing the rule parameters, search them - and search them
on OUT-OF-FOLD training predictions, never on test. The test set is touched
exactly once, at the end, to report the honest result.

Run from the folder containing creditcard.csv:
    python fraud_rule_search.py
"""

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

CSV_PATH = "creditcard.csv"
COST_FP = 5.0
RANDOM_STATE = 42


def log(m=""):
    print(m, flush=True)


def true_cost(y, pred, amt, cost_fp=COST_FP):
    """Money actually lost to missed fraud + money spent reviewing alerts."""
    y = np.asarray(y); pred = np.asarray(pred); amt = np.asarray(amt)
    value_lost = amt[(y == 1) & (pred == 0)].sum()
    n_fp = int(((y == 0) & (pred == 1)).sum())
    return float(value_lost) + n_fp * cost_fp, float(value_lost), n_fp


# ---------------------------------------------------------------- data
df = pd.read_csv(CSV_PATH).drop_duplicates().reset_index(drop=True)
df["hour"] = (df["Time"] // 3600) % 24
df["log_amount"] = np.log1p(df["Amount"])

y = df["Class"].astype(int)
amounts = df["Amount"].copy()
X = df.drop(columns=["Class", "Time", "Amount"])

X_tr, X_te, y_tr, y_te, amt_tr, amt_te = train_test_split(
    X, y, amounts, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
for s in (y_tr, y_te, amt_tr, amt_te):
    s.reset_index(drop=True, inplace=True)

log(f"Train {len(X_tr):,} ({y_tr.sum()} frauds) | Test {len(X_te):,} ({y_te.sum()} frauds)")

# ---------------------------------------------------------------- model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=2000, random_state=RANDOM_STATE)),
])
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
model.fit(X_tr, y_tr)
p_te = model.predict_proba(X_te)[:, 1]
log("Model fitted, out-of-fold predictions ready.\n")

# ------------------------------------------- step 1: threshold, full sweep
# v3 only compared 4 candidate thresholds. Sweep the whole range instead -
# the borderline frauds needing thr < 0.95 were never tested.
order = np.argsort(-oof)
p_sorted, y_sorted, a_sorted = oof[order], y_tr.values[order], amt_tr.values[order]
value_caught = np.cumsum(y_sorted * a_sorted)
value_missed = value_caught[-1] - value_caught
fp_cum = np.cumsum(1 - y_sorted)
sweep_cost = value_missed + fp_cum * COST_FP
thr_best = float(p_sorted[np.argmin(sweep_cost)])

log(f"Best threshold from FULL sweep (on OOF): {thr_best:.8f}")
base_pred_oof = (oof >= thr_best).astype(int)
c, v, f = true_cost(y_tr, base_pred_oof, amt_tr)
log(f"   OOF: cost {c:,.0f} (value lost {v:,.0f}, {f} false alarms)\n")

# ------------------------------------------- step 2: search the rule layer
log("Searching rule layer: flag if (amount >= A) AND (score >= S) ...")
amount_cuts = np.percentile(amt_tr[amt_tr > 0], [80, 85, 90, 93, 95, 97, 98, 99, 99.5])
score_floors = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]

base_cost_oof = c
best = {"cost": base_cost_oof, "A": None, "S": None}
grid = []

for A in amount_cuts:
    for S in score_floors:
        rule = (amt_tr.values >= A) & (oof >= S)
        combined = (base_pred_oof.astype(bool) | rule).astype(int)
        cost_, v_, f_ = true_cost(y_tr, combined, amt_tr)
        grid.append({"A": A, "S": S, "cost": cost_, "value_lost": v_, "fp": f_})
        if cost_ < best["cost"]:
            best = {"cost": cost_, "A": float(A), "S": float(S)}

grid_df = pd.DataFrame(grid).sort_values("cost")
log("\nTop 8 rule settings by OOF true cost:")
log(grid_df.head(8).to_string(
    index=False,
    formatters={"A": lambda v: f"{v:>9,.0f}", "S": lambda v: f"{v:>5.2f}",
                "cost": lambda v: f"{v:>10,.0f}", "value_lost": lambda v: f"{v:>11,.0f}"}))
log(f"\nModel alone (no rule), OOF cost: {base_cost_oof:,.0f}")

if best["A"] is None:
    log("\n>>> No rule setting beat the model alone on OOF data.")
    log("    That is a real result: the blind spots are not recoverable by an")
    log("    amount-based rule. Present it as a tested negative, not a gap.")
    use_rule = False
else:
    improvement = base_cost_oof - best["cost"]
    log(f"\n>>> Best rule: amount >= {best['A']:,.0f} AND score >= {best['S']:.2f}")
    log(f"    OOF cost {best['cost']:,.0f} vs {base_cost_oof:,.0f} alone "
        f"(saves {improvement:,.0f})")
    use_rule = True

# ------------------------------------------- step 3: ONE honest test run
log("\n" + "=" * 60)
log("TEST SET - touched once, with settings chosen above")
log("=" * 60)

total_value = amt_te[y_te == 1].sum()
base_pred_te = (p_te >= thr_best).astype(int)
cost_b, v_b, f_b = true_cost(y_te, base_pred_te, amt_te)
tn, fp_, fn_, tp_ = confusion_matrix(y_te, base_pred_te).ravel()
log(f"\nModel alone @ {thr_best:.8f}")
log(f"   caught {tp_}/{tp_+fn_} | false alarms {fp_} | "
    f"value caught {1 - v_b/total_value:.1%} | TRUE cost {cost_b:,.0f}")

if use_rule:
    rule_te = (amt_te.values >= best["A"]) & (p_te >= best["S"])
    comb_te = (base_pred_te.astype(bool) | rule_te).astype(int)
    cost_r, v_r, f_r = true_cost(y_te, comb_te, amt_te)
    tn, fp_r, fn_r, tp_r = confusion_matrix(y_te, comb_te).ravel()
    log(f"\nModel + rule (amount >= {best['A']:,.0f} AND score >= {best['S']:.2f})")
    log(f"   caught {tp_r}/{tp_r+fn_r} | false alarms {fp_r} | "
        f"value caught {1 - v_r/total_value:.1%} | TRUE cost {cost_r:,.0f}")
    delta = cost_b - cost_r
    log(f"\n   => rule {'HELPS' if delta > 0 else 'DOES NOT help'} on test: "
        f"{'saves' if delta > 0 else 'costs'} {abs(delta):,.0f}")
    log(f"      recovered {v_b - v_r:,.0f} of fraud value for {fp_r - fp_} extra reviews")
    if delta <= 0:
        log("\n   NOTE: it helped on OOF but not on test. With 95 test frauds that")
        log("   is exactly the overfitting you'd expect. Report the OOF result AND")
        log("   this - the honesty is worth more than the number.")

# ------------------------------------------- variance warning
log("\n" + "=" * 60)
missed_te = amt_te[(y_te == 1) & (base_pred_te == 0)].sort_values(ascending=False)
top3 = missed_te.head(3).sum()
log(f"SANITY: your 3 biggest misses are {top3:,.0f} = {top3/total_value:.0%} of all")
log(f"test fraud value. With only {int(y_te.sum())} test frauds, every value-based")
log("number here swings hugely on a handful of transactions. Quote these with")
log("error bars, or quote the OOF numbers instead - they average over 5 folds.")

plt.figure(figsize=(7, 5))
piv = grid_df.pivot_table(index="S", columns="A", values="cost")
plt.imshow(piv.values, aspect="auto", origin="lower", cmap="viridis_r")
plt.colorbar(label="OOF true cost (lower = better)")
plt.xticks(range(len(piv.columns)), [f"{c:,.0f}" for c in piv.columns], rotation=45, ha="right")
plt.yticks(range(len(piv.index)), [f"{i:.2f}" for i in piv.index])
plt.xlabel("Amount cut"); plt.ylabel("Score floor")
plt.title("Rule layer search (out-of-fold)")
plt.tight_layout()
plt.savefig("rule_search.png", dpi=130)
plt.close()
log("\nSaved rule_search.png")
