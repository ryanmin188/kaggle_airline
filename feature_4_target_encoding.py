import os
import warnings

import numpy as np
import pandas as pd
import sklearn
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# ============================================================
# PATH
# ============================================================

TRAIN_PATH = r"D:\kaggle\airline\data\train.csv"
TEST_PATH = r"D:\kaggle\airline\data\test.csv"
SAMPLE_PATH = r"D:\kaggle\airline\data\sample_submission.csv"

OUTPUT_DIR = r"D:\kaggle\airline\proper_te_experiment"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
OUTER_FOLDS = 5
INNER_TE_FOLDS = 5

CURRENT_BEST_OOF = 0.959225


# ============================================================
# LIGHTGBM PARAMETERS
#
# Based on the stronger Proper-TE route rather than our old
# original LightGBM parameters.
# ============================================================

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "min_child_samples": 50,
    "colsample_bytree": 0.70,
    "subsample": 0.80,
    "subsample_freq": 1,
    "reg_alpha": 0.30,
    "reg_lambda": 1.00,
    "n_estimators": 8000,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}


# ============================================================
# LOAD
# ============================================================

print("=" * 90)
print("PROPER TARGET ENCODING + LIGHTGBM")
print("=" * 90)

print(f"scikit-learn : {sklearn.__version__}")
print(f"LightGBM     : {lgb.__version__}")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

TARGET = "satisfaction"
ID_COL = "id"

y = train[TARGET].astype(np.int8).to_numpy()

print(f"\nTrain shape: {train.shape}")
print(f"Test shape : {test.shape}")
print(f"Target mean: {y.mean():.6f}")


# ============================================================
# ORIGINAL 21 PREDICTORS
#
# 23 columns in train:
# id + 21 predictors + satisfaction
#
# We deliberately use ALL 21 original predictors for TE.
# ============================================================

ORIGINAL_FEATURES = [
    c for c in train.columns
    if c not in [ID_COL, TARGET]
]

print("\nOriginal predictors:")
for i, c in enumerate(ORIGINAL_FEATURES, 1):
    print(f"{i:02d}. {c}")

print(
    f"\nNumber of original predictors = "
    f"{len(ORIGINAL_FEATURES)}"
)


# ============================================================
# BUILD MODEL FEATURES
#
# Keep original information.
# Add:
#   traveler_segment
#   arrival_missing
#
# These are NOT part of the basic TE matrix below.
# ============================================================

CATEGORICAL_COLS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

NUMERIC_COLS = [
    c for c in ORIGINAL_FEATURES
    if c not in CATEGORICAL_COLS
]


def build_model_frame(df):
    X = df.drop(
        columns=[ID_COL, TARGET],
        errors="ignore",
    ).copy()

    # --------------------------------------------------------
    # Useful categorical interaction
    # --------------------------------------------------------

    X["traveler_segment"] = (
        df["Type of Travel"].astype(str)
        + "|"
        + df["Customer Type"].astype(str)
        + "|"
        + df["Class"].astype(str)
    )

    # --------------------------------------------------------
    # Missingness indicator
    # --------------------------------------------------------

    X["arrival_missing"] = (
        df["Arrival Delay in Minutes"]
        .isna()
        .astype("int8")
    )

    return X


X_full = build_model_frame(train)
X_test_full = build_model_frame(test)


# ============================================================
# PREPARE ORIGINAL MODEL FEATURES
#
# Numerical columns remain numerical.
# Categorical columns are pandas categorical.
# ============================================================

for c in NUMERIC_COLS:
    median_value = train[c].median()

    X_full[c] = (
        X_full[c]
        .fillna(median_value)
        .astype("float32")
    )

    X_test_full[c] = (
        X_test_full[c]
        .fillna(median_value)
        .astype("float32")
    )


MODEL_CAT_COLS = CATEGORICAL_COLS + [
    "traveler_segment"
]

for c in MODEL_CAT_COLS:

    combined = pd.concat(
        [
            X_full[c],
            X_test_full[c],
        ],
        axis=0,
    ).astype(str)

    categories = pd.Index(
        combined.unique()
    )

    dtype = pd.api.types.CategoricalDtype(
        categories=categories
    )

    X_full[c] = X_full[c].astype(dtype)
    X_test_full[c] = X_test_full[c].astype(dtype)


print(
    f"\nBase model feature count = "
    f"{X_full.shape[1]}"
)


# ============================================================
# TARGET ENCODER INPUT
#
# IMPORTANT:
# ALL 21 ORIGINAL VARIABLES are treated as categories.
#
# Numeric values such as:
#   Age = 25
#   Flight Distance = 850
#   Departure Delay = 17
#
# become exact-value categories for TargetEncoder.
#
# This is intentional.
# ============================================================

TE_TRAIN = train[
    ORIGINAL_FEATURES
].copy()

TE_TEST = test[
    ORIGINAL_FEATURES
].copy()


for c in ORIGINAL_FEATURES:
    TE_TRAIN[c] = (
        TE_TRAIN[c]
        .astype(str)
    )

    TE_TEST[c] = (
        TE_TEST[c]
        .astype(str)
    )


print(
    "\nTE matrix:"
    f" {TE_TRAIN.shape[1]} variables"
)


# ============================================================
# OUTER CV
# ============================================================

skf = StratifiedKFold(
    n_splits=OUTER_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# OOF / TEST ARRAYS
# ============================================================

oof = np.zeros(
    len(train),
    dtype=np.float32,
)

test_pred = np.zeros(
    len(test),
    dtype=np.float64,
)

fold_scores = []
best_iterations = []


# ============================================================
# OUTER LOOP
# ============================================================

for fold, (tr_idx, va_idx) in enumerate(
    skf.split(
        np.zeros(len(y)),
        y,
    ),
    1,
):

    print("\n" + "=" * 90)
    print(
        f"OUTER FOLD {fold}/{OUTER_FOLDS}"
    )
    print("=" * 90)

    # --------------------------------------------------------
    # Base features
    # --------------------------------------------------------

    X_tr = X_full.iloc[
        tr_idx
    ].copy()

    X_va = X_full.iloc[
        va_idx
    ].copy()

    X_te = X_test_full.copy()

    # --------------------------------------------------------
    # Target Encoder
    #
    # THIS is the key part.
    #
    # smooth='auto'
    # cv=5
    # shuffle=True
    #
    # fit_transform on training fold performs internal
    # cross-fitting, preventing each training observation
    # from simply seeing its own target.
    # --------------------------------------------------------

    encoder = TargetEncoder(
        target_type="binary",
        smooth="auto",
        cv=INNER_TE_FOLDS,
        shuffle=True,
        random_state=20261002,
    )

    print(
        "Fitting TargetEncoder "
        "(inner CV=5, smooth='auto')..."
    )

    te_tr = encoder.fit_transform(
        TE_TRAIN.iloc[tr_idx],
        y[tr_idx],
    ).astype(np.float32)

    te_va = encoder.transform(
        TE_TRAIN.iloc[va_idx]
    ).astype(np.float32)

    te_te = encoder.transform(
        TE_TEST
    ).astype(np.float32)

    # --------------------------------------------------------
    # Add 21 TE columns
    # --------------------------------------------------------

    for j, col in enumerate(
        ORIGINAL_FEATURES
    ):

        te_name = (
            "TE_"
            + col
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
    )

        X_tr[te_name] = te_tr[:, j]
        X_va[te_name] = te_va[:, j]
        X_te[te_name] = te_te[:, j]

    print(
        f"Added {te_tr.shape[1]} TE features."
    )

    print(
        f"Final feature count = "
        f"{X_tr.shape[1]}"
    )

    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    model = lgb.LGBMClassifier(
        **LGB_PARAMS
    )

    print(
        "Training LightGBM..."
    )

    model.fit(
        X_tr,
        y[tr_idx],
        eval_set=[
            (
                X_va,
                y[va_idx],
            )
        ],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(
                150,
                verbose=False,
            )
        ],
    )

    best_iter = (
        model.best_iteration_
        if model.best_iteration_
        else LGB_PARAMS["n_estimators"]
    )

    best_iterations.append(
        best_iter
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_pred = model.predict_proba(
        X_va
    )[:, 1]

    fold_auc = roc_auc_score(
        y[va_idx],
        val_pred,
    )

    oof[va_idx] = val_pred

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    fold_test_pred = model.predict_proba(
        X_te
    )[:, 1]

    test_pred += (
        fold_test_pred
        / OUTER_FOLDS
    )

    fold_scores.append(
        fold_auc
    )

    print(
        f"Best iteration = {best_iter}"
    )

    print(
        f"Fold {fold} AUC = "
        f"{fold_auc:.6f}"
    )


# ============================================================
# FINAL OOF
# ============================================================

oof_auc = roc_auc_score(
    y,
    oof,
)

fold_mean = np.mean(
    fold_scores
)

fold_std = np.std(
    fold_scores
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("FINAL PROPER-TE RESULT")
print("=" * 90)

print(
    "Fold AUCs:"
)

for i, score in enumerate(
    fold_scores,
    1,
):
    print(
        f"Fold {i}: "
        f"{score:.6f}"
    )

print(
    f"\nFold mean AUC = "
    f"{fold_mean:.6f}"
)

print(
    f"Fold std      = "
    f"{fold_std:.6f}"
)

print(
    f"OOF AUC       = "
    f"{oof_auc:.6f}"
)

print(
    f"Current 3-model benchmark = "
    f"{CURRENT_BEST_OOF:.6f}"
)

print(
    f"Delta vs current benchmark = "
    f"{oof_auc - CURRENT_BEST_OOF:+.6f}"
)

print(
    "\nBest iterations:"
)

print(
    best_iterations
)


# ============================================================
# SAVE OOF
# ============================================================

oof_df = pd.DataFrame({
    "id": train[ID_COL],
    "satisfaction": y,
    "prediction": oof,
})

oof_path = os.path.join(
    OUTPUT_DIR,
    "proper_te_lgbm_oof.csv",
)

oof_df.to_csv(
    oof_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_pred_df = pd.DataFrame({
    "id": test[ID_COL],
    "prediction": test_pred,
})

test_pred_path = os.path.join(
    OUTPUT_DIR,
    "proper_te_lgbm_test.csv",
)

test_pred_df.to_csv(
    test_pred_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SUBMISSION
#
# IMPORTANT:
# ROC AUC uses continuous probability.
# DO NOT threshold at 0.5.
# ============================================================

sample = pd.read_csv(
    SAMPLE_PATH
)

submission_target_col = [
    c
    for c in sample.columns
    if c != ID_COL
][0]

submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    submission_target_col: test_pred,
})

submission_path = os.path.join(
    OUTPUT_DIR,
    "submission_proper_te_lgbm.csv",
)

submission.to_csv(
    submission_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# PREDICTION SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("OUTPUT")
print("=" * 90)

print(
    f"OOF prediction file : "
    f"{oof_path}"
)

print(
    f"Test prediction file: "
    f"{test_pred_path}"
)

print(
    f"Submission file     : "
    f"{submission_path}"
)

print(
    "\nTest prediction statistics:"
)

print(
    f"min  = {test_pred.min():.6f}"
)

print(
    f"max  = {test_pred.max():.6f}"
)

print(
    f"mean = {test_pred.mean():.6f}"
)

print(
    f"std  = {test_pred.std():.6f}"
)

print("\nDone.")