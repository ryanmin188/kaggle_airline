import os
import time

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# ============================================================
# 0. CONFIGURATION
# ============================================================

TRAIN_PATH = r"D:\kaggle\airline\data\train.csv"
TEST_PATH = r"D:\kaggle\airline\data\test.csv"
OUTPUT_DIR = r"D:\kaggle\airline"

TARGET = "satisfaction"
ID_COL = "id"

RANDOM_STATE = 42
N_FOLDS = 5

# ------------------------------------------------------------
# Current best LightGBM benchmark
# ------------------------------------------------------------

LGBM_BENCHMARK = 0.959015


# ============================================================
# 1. CATBOOST PARAMETERS
#
# This is the first CatBoost benchmark.
# We are NOT doing extensive hyperparameter tuning here.
# The purpose is model diversification.
# ============================================================

CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",

    "iterations": 1200,
    "learning_rate": 0.03,

    "depth": 8,

    "l2_leaf_reg": 5,

    "random_strength": 1.0,

    "random_seed": RANDOM_STATE,

    "thread_count": -1,

    "verbose": False,

    "allow_writing_files": False,
}


# ============================================================
# 2. BEST CATEGORICAL INTERACTIONS
#
# Confirmed by previous feature engineering experiments.
# ============================================================

BEST_CATEGORICAL_INTERACTIONS = {
    "Customer_Type__Type_of_Travel__Class": [
        "Customer Type",
        "Type of Travel",
        "Class",
    ],

    "Gender__Type_of_Travel__Class": [
        "Gender",
        "Type of Travel",
        "Class",
    ],
}


# ============================================================
# 3. LOAD TRAIN
# ============================================================

print("=" * 80)
print("LOADING TRAIN DATA")
print("=" * 80)

start_time = time.time()

train = pd.read_csv(TRAIN_PATH)

print(f"Train shape: {train.shape}")


# ============================================================
# 4. BUILD FINAL FEATURE SET
#
# Exactly:
#
# Original features
# +
# Customer Type × Type of Travel × Class
# +
# Gender × Type of Travel × Class
#
# No Delay features.
# No Service aggregation features.
# ============================================================

def build_final_features(df):

    X = df.copy()

    # --------------------------------------------------------
    # Remove target and ID when present
    # --------------------------------------------------------

    columns_to_drop = []

    if TARGET in X.columns:
        columns_to_drop.append(TARGET)

    if ID_COL in X.columns:
        columns_to_drop.append(ID_COL)

    X = X.drop(
        columns=columns_to_drop
    ).copy()

    # --------------------------------------------------------
    # Create confirmed interactions
    # --------------------------------------------------------

    for (
        interaction_name,
        source_columns
    ) in BEST_CATEGORICAL_INTERACTIONS.items():

        X[interaction_name] = (
            X[source_columns[0]]
            .astype("string")
        )

        for col in source_columns[1:]:

            X[interaction_name] = (
                X[interaction_name]
                + "_"
                + X[col].astype("string")
            )

    # --------------------------------------------------------
    # CatBoost categorical variables
    #
    # Convert them to strings.
    # This is important because CatBoost expects categorical
    # features to be represented as strings / categorical data.
    # --------------------------------------------------------

    categorical_features = [
        "Gender",
        "Customer Type",
        "Type of Travel",
        "Class",
        "Customer_Type__Type_of_Travel__Class",
        "Gender__Type_of_Travel__Class",
    ]

    for col in categorical_features:

        if col not in X.columns:
            raise ValueError(
                f"Missing categorical feature: {col}"
            )

        X[col] = (
            X[col]
            .astype("string")
            .fillna("Missing")
            .astype(str)
        )

    return X, categorical_features


# ============================================================
# 5. PREPARE TRAIN FEATURES
# ============================================================

X, categorical_features = build_final_features(
    train
)

y = train[TARGET].astype(int)

print("\nFinal feature matrix:")
print(f"Shape: {X.shape}")

print("\nCategorical features:")

for col in categorical_features:
    print(f" - {col}")


# ============================================================
# 6. VALIDATE FEATURE COUNT
# ============================================================

print("\n" + "=" * 80)
print("FEATURE SET")
print("=" * 80)

for i, col in enumerate(
    X.columns,
    start=1
):
    print(
        f"{i:2d}. {col}"
    )


# ============================================================
# 7. SAME 5-FOLD CV AS LIGHTGBM
# ============================================================

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# 8. STORAGE
# ============================================================

oof_predictions = np.zeros(
    len(train),
    dtype=float
)

test_predictions = None

fold_scores = []

feature_importance_list = []


# ============================================================
# 9. CHECK TEST DATA
# ============================================================

test_exists = os.path.exists(
    TEST_PATH
)

if test_exists:

    test = pd.read_csv(
        TEST_PATH
    )

    print("\n" + "=" * 80)
    print("TEST DATA FOUND")
    print("=" * 80)

    print(
        f"Test shape: {test.shape}"
    )

    X_test, test_cat_features = (
        build_final_features(
            test
        )
    )

    # Make absolutely sure feature order matches
    if list(X_test.columns) != list(X.columns):

        raise ValueError(
            "Train and test feature columns do not match "
            "after feature engineering."
        )

    test_predictions = np.zeros(
        len(test),
        dtype=float
    )

else:

    test = None

    X_test = None

    print("\nTest file not found:")
    print(TEST_PATH)

    print(
        "OOF evaluation will still run."
    )


# ============================================================
# 10. CATBOOST 5-FOLD TRAINING
# ============================================================

print("\n" + "=" * 80)
print("CATBOOST 5-FOLD CROSS-VALIDATION")
print("=" * 80)

print("\nParameters:")

for key, value in CATBOOST_PARAMS.items():
    print(
        f"{key}: {value}"
    )


for fold, (
    train_idx,
    valid_idx
) in enumerate(
    skf.split(X, y),
    start=1
):

    print("\n" + "-" * 80)

    print(
        f"FOLD {fold}/{N_FOLDS}"
    )

    print("-" * 80)

    X_train = X.iloc[
        train_idx
    ]

    X_valid = X.iloc[
        valid_idx
    ]

    y_train = y.iloc[
        train_idx
    ]

    y_valid = y.iloc[
        valid_idx
    ]

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = CatBoostClassifier(
        **CATBOOST_PARAMS
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "Training CatBoost..."
    )

    model.fit(
        X_train,
        y_train,

        cat_features=categorical_features,

        eval_set=(
            X_valid,
            y_valid
        ),

        verbose=False
    )

    # --------------------------------------------------------
    # Validation probability
    # --------------------------------------------------------

    valid_pred = (
        model.predict_proba(
            X_valid
        )[:, 1]
    )

    # --------------------------------------------------------
    # Fold AUC
    # --------------------------------------------------------

    fold_auc = roc_auc_score(
        y_valid,
        valid_pred
    )

    fold_scores.append(
        fold_auc
    )

    # --------------------------------------------------------
    # Store OOF
    # --------------------------------------------------------

    oof_predictions[
        valid_idx
    ] = valid_pred

    # --------------------------------------------------------
    # Test prediction
    # --------------------------------------------------------

    if test_exists:

        fold_test_pred = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        test_predictions += (
            fold_test_pred
            / N_FOLDS
        )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = (
        model.get_feature_importance()
    )

    fold_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": importance,
        "fold": fold,
    })

    feature_importance_list.append(
        fold_importance
    )

    print(
        f"Fold {fold} AUC: "
        f"{fold_auc:.6f}"
    )


# ============================================================
# 11. CV SUMMARY
# ============================================================

mean_auc = np.mean(
    fold_scores
)

std_auc = np.std(
    fold_scores,
    ddof=0
)

improvement_vs_lgbm = (
    mean_auc
    - LGBM_BENCHMARK
)


print("\n" + "=" * 80)
print("CATBOOST FINAL CV RESULT")
print("=" * 80)

for i, score in enumerate(
    fold_scores,
    start=1
):

    print(
        f"Fold {i}: "
        f"{score:.6f}"
    )

print("-" * 80)

print(
    f"Mean ROC-AUC: "
    f"{mean_auc:.6f}"
)

print(
    f"Std ROC-AUC:  "
    f"{std_auc:.6f}"
)

print(
    f"LightGBM benchmark: "
    f"{LGBM_BENCHMARK:.6f}"
)

print(
    f"CatBoost - LightGBM: "
    f"{improvement_vs_lgbm:+.6f}"
)


# ============================================================
# 12. OOF AUC
#
# This is the most important number for future ensemble work.
# ============================================================

oof_auc = roc_auc_score(
    y,
    oof_predictions
)

print("\n" + "=" * 80)
print("OOF PREDICTION CHECK")
print("=" * 80)

print(
    f"OOF ROC-AUC: "
    f"{oof_auc:.6f}"
)

print(
    f"OOF - LightGBM benchmark: "
    f"{oof_auc - LGBM_BENCHMARK:+.6f}"
)


# ============================================================
# 13. SAVE OOF PREDICTIONS
#
# This file will be used later for:
# - prediction correlation
# - blending
# - ensemble optimization
# ============================================================

oof_output = pd.DataFrame({
    "id": train[ID_COL].values,
    "satisfaction": y.values,
    "catboost_oof": oof_predictions,
})

oof_path = (
    OUTPUT_DIR
    + r"\catboost_oof_predictions.csv"
)

oof_output.to_csv(
    oof_path,
    index=False
)

print(
    f"\nSaved OOF predictions:"
    f"\n{oof_path}"
)


# ============================================================
# 14. SAVE TEST PREDICTIONS
# ============================================================

if test_exists:

    test_output = pd.DataFrame({
        "id": test[ID_COL].values,
        "catboost_prediction": test_predictions,
    })

    test_path = (
        OUTPUT_DIR
        + r"\catboost_test_predictions.csv"
    )

    test_output.to_csv(
        test_path,
        index=False
    )

    print(
        f"\nSaved averaged test predictions:"
        f"\n{test_path}"
    )


# ============================================================
# 15. FEATURE IMPORTANCE
# ============================================================

importance_all = pd.concat(
    feature_importance_list,
    ignore_index=True
)

importance_summary = (
    importance_all
    .groupby("feature")["importance"]
    .agg(
        mean_importance="mean",
        std_importance="std"
    )
    .sort_values(
        "mean_importance",
        ascending=False
    )
    .reset_index()
)

importance_path = (
    OUTPUT_DIR
    + r"\catboost_feature_importance.csv"
)

importance_summary.to_csv(
    importance_path,
    index=False
)

print(
    f"\nSaved feature importance:"
    f"\n{importance_path}"
)


# ============================================================
# 16. SAVE MODEL SUMMARY
# ============================================================

summary = pd.DataFrame([{

    "model":
        "CatBoost",

    "iterations":
        CATBOOST_PARAMS["iterations"],

    "learning_rate":
        CATBOOST_PARAMS["learning_rate"],

    "depth":
        CATBOOST_PARAMS["depth"],

    "l2_leaf_reg":
        CATBOOST_PARAMS["l2_leaf_reg"],

    "5fold_mean_auc":
        mean_auc,

    "5fold_std":
        std_auc,

    "oof_auc":
        oof_auc,

    "lightgbm_benchmark":
        LGBM_BENCHMARK,

    "difference_vs_lightgbm":
        oof_auc - LGBM_BENCHMARK,

    "n_features":
        X.shape[1],

    "n_categorical_features":
        len(categorical_features),

}])


summary_path = (
    OUTPUT_DIR
    + r"\catboost_model_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print(
    f"\nSaved model summary:"
    f"\n{summary_path}"
)


# ============================================================
# 17. FINAL REPORT
# ============================================================

elapsed = (
    time.time()
    - start_time
)

print("\n" + "=" * 80)
print("MODEL 02 — CATBOOST FINISHED")
print("=" * 80)

print(
    f"5-Fold Mean AUC: "
    f"{mean_auc:.6f}"
)

print(
    f"5-Fold Std: "
    f"{std_auc:.6f}"
)

print(
    f"OOF AUC: "
    f"{oof_auc:.6f}"
)

print(
    f"Difference vs LightGBM: "
    f"{oof_auc - LGBM_BENCHMARK:+.6f}"
)

print(
    f"Elapsed time: "
    f"{elapsed / 60:.2f} minutes"
)

print("\nFold scores:")

for i, score in enumerate(
    fold_scores,
    start=1
):

    print(
        f"Fold {i}: "
        f"{score:.6f}"
    )

print("\nOutput files:")

print(
    "1. catboost_oof_predictions.csv"
)

if test_exists:
    print(
        "2. catboost_test_predictions.csv"
    )

print(
    "3. catboost_feature_importance.csv"
)

print(
    "4. catboost_model_summary.csv"
)