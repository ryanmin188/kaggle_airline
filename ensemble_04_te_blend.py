import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = r"D:\kaggle\airline\data\train.csv"

TE_OOF_PATH = (
    r"D:\kaggle\airline\conditional_te_experiment"
    r"\conditional_te_lgbm_oof.csv"
)

TE_TEST_PATH = (
    r"D:\kaggle\airline\conditional_te_experiment"
    r"\conditional_te_lgbm_test.csv"
)

OLD_OOF_PATH = (
    r"D:\kaggle\airline\ensemble_02_best_oof_predictions.csv"
)

OLD_TEST_PATH = (
    r"D:\kaggle\airline\ensemble_02_best_test_predictions.csv"
)

SAMPLE_PATH = (
    r"D:\kaggle\airline\data\sample_submission.csv"
)

OUTPUT_DIR = (
    r"D:\kaggle\airline\te_ensemble"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("TE + ORIGINAL 3-MODEL ENSEMBLE")
print("=" * 80)

train = pd.read_csv(
    TRAIN_PATH
)

test = pd.read_csv(
    r"D:\kaggle\airline\data\test.csv"
)

te_oof = pd.read_csv(
    TE_OOF_PATH
)

te_test = pd.read_csv(
    TE_TEST_PATH
)

old_oof = pd.read_csv(
    OLD_OOF_PATH
)

old_test = pd.read_csv(
    OLD_TEST_PATH
)

y = (
    train["satisfaction"]
    .astype(int)
    .values
)

test_ids = test["id"]


# ============================================================
# SHOW FILE STRUCTURE
# ============================================================

print("\nTE OOF columns:")
print(
    te_oof.columns.tolist()
)

print("\nTE TEST columns:")
print(
    te_test.columns.tolist()
)

print("\nOld OOF columns:")
print(
    old_oof.columns.tolist()
)

print("\nOld TEST columns:")
print(
    old_test.columns.tolist()
)


# ============================================================
# ALIGN TE OOF
# ============================================================

te_oof = (
    te_oof
    .set_index("id")
    .loc[train["id"]]
)

te_oof_pred = (
    te_oof["prediction"]
    .astype(float)
    .values
)


# ============================================================
# ALIGN TE TEST
# ============================================================

te_test = (
    te_test
    .set_index("id")
    .loc[test_ids]
)

te_test_pred = (
    te_test["prediction"]
    .astype(float)
    .values
)


# ============================================================
# ALIGN OLD OOF
# ============================================================

old_oof = (
    old_oof
    .set_index("id")
    .loc[train["id"]]
)

# Your actual column:
# best_ensemble_oof

if "best_ensemble_oof" not in old_oof.columns:
    raise ValueError(
        "best_ensemble_oof not found in old OOF file.\n"
        f"Columns: {old_oof.columns.tolist()}"
    )

old_oof_pred = (
    old_oof["best_ensemble_oof"]
    .astype(float)
    .values
)


# ============================================================
# ALIGN OLD TEST
# ============================================================

old_test = (
    old_test
    .set_index("id")
    .loc[test_ids]
)


# ------------------------------------------------------------
# YOUR ACTUAL OLD TEST COLUMN IS:
#
# ensemble_prediction
# ------------------------------------------------------------

if "ensemble_prediction" in old_test.columns:

    old_test_pred = (
        old_test["ensemble_prediction"]
        .astype(float)
        .values
    )

else:

    # fallback
    candidates = [
        "best_ensemble_test",
        "best_ensemble_pred",
        "prediction",
    ]

    old_test_col = None

    for c in candidates:

        if c in old_test.columns:

            old_test_col = c
            break

    if old_test_col is None:

        raise ValueError(
            "Could not identify old ensemble "
            "test prediction column.\n"
            f"Columns: "
            f"{old_test.columns.tolist()}"
        )

    old_test_pred = (
        old_test[old_test_col]
        .astype(float)
        .values
    )


# ============================================================
# BASIC CHECKS
# ============================================================

assert len(te_oof_pred) == len(y)
assert len(old_oof_pred) == len(y)

assert len(te_test_pred) == len(test)
assert len(old_test_pred) == len(test)

assert np.isfinite(
    te_oof_pred
).all()

assert np.isfinite(
    old_oof_pred
).all()

assert np.isfinite(
    te_test_pred
).all()

assert np.isfinite(
    old_test_pred
).all()


# ============================================================
# INDIVIDUAL PERFORMANCE
# ============================================================

te_auc = roc_auc_score(
    y,
    te_oof_pred,
)

old_auc = roc_auc_score(
    y,
    old_oof_pred,
)

corr = np.corrcoef(
    te_oof_pred,
    old_oof_pred,
)[0, 1]


print("\n" + "=" * 80)
print("INDIVIDUAL PERFORMANCE")
print("=" * 80)

print(
    f"Conditional-TE LGBM OOF : "
    f"{te_auc:.6f}"
)

print(
    f"Original 3-model OOF     : "
    f"{old_auc:.6f}"
)

print(
    f"Prediction correlation   : "
    f"{corr:.6f}"
)


# ============================================================
# WEIGHT SEARCH
#
# w = weight on Conditional-TE LGBM
# 1-w = weight on Original 3-model Ensemble
#
# 0.005 step
# ============================================================

print("\n" + "=" * 80)
print("BLEND SEARCH")
print("=" * 80)

results = []

best_auc = -np.inf
best_w = None
best_oof = None


for w in np.arange(
    0.0,
    1.0001,
    0.005,
):

    blend_oof = (
        w * te_oof_pred
        + (1.0 - w)
        * old_oof_pred
    )

    auc = roc_auc_score(
        y,
        blend_oof,
    )

    results.append(
        {
            "te_weight": w,
            "old_ensemble_weight": 1.0 - w,
            "auc": auc,
            "delta_vs_te": auc - te_auc,
            "delta_vs_old": auc - old_auc,
        }
    )

    if auc > best_auc:

        best_auc = auc
        best_w = w
        best_oof = blend_oof.copy()


results_df = (
    pd.DataFrame(results)
    .sort_values(
        "auc",
        ascending=False,
    )
    .reset_index(drop=True)
)


# ============================================================
# TOP RESULTS
# ============================================================

print("\nTop 15 blend weights:")

print(
    results_df.head(15).to_string(
        index=False
    )
)


# ============================================================
# BEST BLEND
# ============================================================

old_weight = 1.0 - best_w


print("\n" + "=" * 80)
print("BEST BLEND")
print("=" * 80)

print(
    f"Conditional-TE weight : "
    f"{best_w:.3f}"
)

print(
    f"Old ensemble weight   : "
    f"{old_weight:.3f}"
)

print(
    f"Conditional-TE OOF    : "
    f"{te_auc:.6f}"
)

print(
    f"Old ensemble OOF      : "
    f"{old_auc:.6f}"
)

print(
    f"Best blend OOF        : "
    f"{best_auc:.6f}"
)

print(
    f"Improvement vs TE     : "
    f"{best_auc - te_auc:+.6f}"
)

print(
    f"Improvement vs old    : "
    f"{best_auc - old_auc:+.6f}"
)


# ============================================================
# SAVE SEARCH RESULTS
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "te_old_ensemble_blend_search.csv",
)

results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# BEST TEST BLEND
# ============================================================

best_test_pred = (
    best_w * te_test_pred
    + old_weight * old_test_pred
)


# ============================================================
# SAVE OOF
# ============================================================

best_oof_path = os.path.join(
    OUTPUT_DIR,
    "te_old_ensemble_best_oof.csv",
)

pd.DataFrame({
    "id": train["id"],
    "satisfaction": y,
    "prediction": best_oof,
}).to_csv(
    best_oof_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

best_test_path = os.path.join(
    OUTPUT_DIR,
    "te_old_ensemble_best_test.csv",
)

pd.DataFrame({
    "id": test_ids,
    "prediction": best_test_pred,
}).to_csv(
    best_test_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SUBMISSION
#
# IMPORTANT:
# Continuous probability only.
# NO threshold.
# ============================================================

sample = pd.read_csv(
    SAMPLE_PATH
)

target_col = [
    c
    for c in sample.columns
    if c != "id"
][0]

submission = pd.DataFrame({
    "id": test_ids,
    target_col: best_test_pred,
})

submission_path = os.path.join(
    OUTPUT_DIR,
    "submission_12_te_old_ensemble.csv",
)

submission.to_csv(
    submission_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# TEST PREDICTION STATS
# ============================================================

print("\n" + "=" * 80)
print("TEST PREDICTION")
print("=" * 80)

print(
    f"min  = "
    f"{best_test_pred.min():.6f}"
)

print(
    f"max  = "
    f"{best_test_pred.max():.6f}"
)

print(
    f"mean = "
    f"{best_test_pred.mean():.6f}"
)

print(
    f"std  = "
    f"{best_test_pred.std():.6f}"
)


# ============================================================
# FILES
# ============================================================

print("\n" + "=" * 80)
print("SAVED FILES")
print("=" * 80)

print(
    results_path
)

print(
    best_oof_path
)

print(
    best_test_path
)

print(
    submission_path
)

print("\nDone.")