import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

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

# Existing benchmarks
LGBM_BENCHMARK = 0.959015
LGBM_CAT_BLEND_BENCHMARK = 0.959083


# ============================================================
# 1. XGBOOST PARAMETERS
#
# First benchmark only.
# We are NOT doing a full hyperparameter search.
# ============================================================

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",

    "n_estimators": 900,
    "learning_rate": 0.03,

    "max_depth": 8,

    "min_child_weight": 5,

    "subsample": 0.8,
    "colsample_bytree": 0.8,

    "gamma": 0,

    "reg_alpha": 0,
    "reg_lambda": 1,

    "tree_method": "hist",

    "random_state": RANDOM_STATE,

    "n_jobs": -1,

    "verbosity": 0,

    # Native pandas categorical support
    "enable_categorical": True,
}


# ============================================================
# 2. FINAL CATEGORICAL INTERACTIONS
#
# These were confirmed in the previous feature-engineering
# experiments and are now LOCKED.
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

train = pd.read_csv(
    TRAIN_PATH
)

print(
    f"Train shape: {train.shape}"
)


# ============================================================
# 4. BUILD FINAL XGBOOST FEATURES
#
# Exactly the same logical feature set as final LightGBM
# and CatBoost.
# ============================================================

def build_final_features(df):

    X = df.copy()

    columns_to_drop = []

    if TARGET in X.columns:
        columns_to_drop.append(
            TARGET
        )

    if ID_COL in X.columns:
        columns_to_drop.append(
            ID_COL
        )

    X = X.drop(
        columns=columns_to_drop
    ).copy()

    categorical_features = [
        "Gender",
        "Customer Type",
        "Type of Travel",
        "Class",
    ]

    # --------------------------------------------------------
    # Create the two confirmed interactions
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

        categorical_features.append(
            interaction_name
        )

    # --------------------------------------------------------
    # XGBoost native categorical handling
    #
    # Must use pandas "category" dtype.
    # --------------------------------------------------------

    for col in categorical_features:

        if col not in X.columns:

            raise ValueError(
                f"Missing categorical feature: "
                f"{col}"
            )

        X[col] = (
            X[col]
            .astype("string")
            .fillna("Missing")
            .astype("category")
        )

    return (
        X,
        categorical_features
    )


# ============================================================
# 5. PREPARE TRAIN
# ============================================================

X_train, categorical_features = (
    build_final_features(
        train
    )
)

y = train[
    TARGET
].astype(int)


print("\nFinal feature matrix:")
print(
    f"Shape: {X_train.shape}"
)

print("\nCategorical features:")

for col in categorical_features:
    print(
        f" - {col}"
    )


# ============================================================
# 6. FEATURE LIST
# ============================================================

print("\n" + "=" * 80)
print("FINAL FEATURE SET")
print("=" * 80)

for i, col in enumerate(
    X_train.columns,
    start=1
):

    print(
        f"{i:2d}. {col}"
    )


# ============================================================
# 7. LOAD TEST
# ============================================================

if not os.path.exists(
    TEST_PATH
):

    raise FileNotFoundError(
        f"\nTest file not found:\n"
        f"{TEST_PATH}"
    )

test = pd.read_csv(
    TEST_PATH
)

print("\n" + "=" * 80)
print("TEST DATA")
print("=" * 80)

print(
    f"Test shape: {test.shape}"
)

X_test, test_categorical_features = (
    build_final_features(
        test
    )
)


# ============================================================
# 8. ENSURE TRAIN / TEST FEATURE MATCH
# ============================================================

if list(X_train.columns) != list(
    X_test.columns
):

    raise ValueError(
        "\nTrain and test feature columns "
        "do not match."
    )


# Make category ordering identical.
#
# This avoids possible mismatches in native categorical
# handling between train and test.

for col in categorical_features:

    combined_categories = pd.Index(
        X_train[col].cat.categories
    ).union(
        X_test[col].cat.categories
    )

    X_train[col] = (
        X_train[col]
        .cat.set_categories(
            combined_categories
        )
    )

    X_test[col] = (
        X_test[col]
        .cat.set_categories(
            combined_categories
        )
    )


# ============================================================
# 9. SAME 5-FOLD CV
# ============================================================

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# 10. STORAGE
# ============================================================

xgb_oof = np.zeros(
    len(train),
    dtype=float
)

xgb_test = np.zeros(
    len(test),
    dtype=float
)

fold_scores = []

feature_importance_list = []


# ============================================================
# 11. XGBOOST 5-FOLD
# ============================================================

print("\n" + "=" * 80)
print("XGBOOST 5-FOLD CROSS-VALIDATION")
print("=" * 80)

print("\nParameters:")

for key, value in XGB_PARAMS.items():

    print(
        f"{key}: {value}"
    )


for fold, (
    train_idx,
    valid_idx
) in enumerate(
    skf.split(X_train, y),
    start=1
):

    print("\n" + "-" * 80)

    print(
        f"FOLD {fold}/{N_FOLDS}"
    )

    print("-" * 80)

    X_tr = X_train.iloc[
        train_idx
    ]

    X_val = X_train.iloc[
        valid_idx
    ]

    y_tr = y.iloc[
        train_idx
    ]

    y_val = y.iloc[
        valid_idx
    ]

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = xgb.XGBClassifier(
        **XGB_PARAMS
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "Training XGBoost..."
    )

    model.fit(
        X_tr,
        y_tr,

        eval_set=[
            (
                X_val,
                y_val
            )
        ],

        verbose=False
    )

    # --------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------

    val_pred = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    # --------------------------------------------------------
    # Fold AUC
    # --------------------------------------------------------

    fold_auc = roc_auc_score(
        y_val,
        val_pred
    )

    fold_scores.append(
        fold_auc
    )

    # --------------------------------------------------------
    # OOF
    # --------------------------------------------------------

    xgb_oof[
        valid_idx
    ] = val_pred

    # --------------------------------------------------------
    # Test prediction
    # --------------------------------------------------------

    test_pred = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    xgb_test += (
        test_pred
        / N_FOLDS
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = (
        model.feature_importances_
    )

    importance_df = pd.DataFrame({

        "feature":
            X_train.columns,

        "importance":
            importance,

        "fold":
            fold,
    })

    feature_importance_list.append(
        importance_df
    )

    print(
        f"Fold {fold} AUC: "
        f"{fold_auc:.6f}"
    )


# ============================================================
# 12. CV SUMMARY
# ============================================================

mean_auc = np.mean(
    fold_scores
)

std_auc = np.std(
    fold_scores,
    ddof=0
)

oof_auc = roc_auc_score(
    y,
    xgb_oof
)


print("\n" + "=" * 80)
print("XGBOOST FINAL CV RESULT")
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
    f"5-Fold Mean ROC-AUC: "
    f"{mean_auc:.6f}"
)

print(
    f"5-Fold Std ROC-AUC:  "
    f"{std_auc:.6f}"
)

print(
    f"OOF ROC-AUC:         "
    f"{oof_auc:.6f}"
)

print(
    f"Difference vs "
    f"LightGBM benchmark: "
    f"{oof_auc - LGBM_BENCHMARK:+.6f}"
)

print(
    f"Difference vs current "
    f"LGBM+CatBoost blend: "
    f"{oof_auc - LGBM_CAT_BLEND_BENCHMARK:+.6f}"
)


# ============================================================
# 13. SAVE OOF PREDICTIONS
# ============================================================

oof_output = pd.DataFrame({

    "id":
        train[ID_COL].values,

    "satisfaction":
        y.values,

    "xgboost_oof":
        xgb_oof,
})


oof_path = (
    OUTPUT_DIR
    + r"\xgboost_oof_predictions.csv"
)

oof_output.to_csv(
    oof_path,
    index=False
)

print(
    f"\nSaved XGBoost OOF:"
    f"\n{oof_path}"
)


# ============================================================
# 14. SAVE TEST PREDICTIONS
# ============================================================

test_output = pd.DataFrame({

    "id":
        test[ID_COL].values,

    "xgboost_prediction":
        xgb_test,
})


test_output_path = (
    OUTPUT_DIR
    + r"\xgboost_test_predictions.csv"
)

test_output.to_csv(
    test_output_path,
    index=False
)

print(
    f"\nSaved XGBoost test predictions:"
    f"\n{test_output_path}"
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
    + r"\xgboost_feature_importance.csv"
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
        "XGBoost",

    "n_estimators":
        XGB_PARAMS["n_estimators"],

    "learning_rate":
        XGB_PARAMS["learning_rate"],

    "max_depth":
        XGB_PARAMS["max_depth"],

    "min_child_weight":
        XGB_PARAMS["min_child_weight"],

    "subsample":
        XGB_PARAMS["subsample"],

    "colsample_bytree":
        XGB_PARAMS["colsample_bytree"],

    "5fold_mean_auc":
        mean_auc,

    "5fold_std":
        std_auc,

    "oof_auc":
        oof_auc,

    "lgbm_benchmark":
        LGBM_BENCHMARK,

    "lgbm_catboost_blend_benchmark":
        LGBM_CAT_BLEND_BENCHMARK,

    "difference_vs_lgbm":
        oof_auc
        - LGBM_BENCHMARK,

    "difference_vs_lgbm_catboost":
        oof_auc
        - LGBM_CAT_BLEND_BENCHMARK,

    "n_features":
        X_train.shape[1],

    "n_categorical_features":
        len(categorical_features),

}])


summary_path = (
    OUTPUT_DIR
    + r"\xgboost_model_summary.csv"
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
print("MODEL 03 — XGBOOST FINISHED")
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
    f"vs LightGBM: "
    f"{oof_auc - LGBM_BENCHMARK:+.6f}"
)

print(
    f"vs LGBM + CatBoost: "
    f"{oof_auc - LGBM_CAT_BLEND_BENCHMARK:+.6f}"
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
    "1. xgboost_oof_predictions.csv"
)

print(
    "2. xgboost_test_predictions.csv"
)

print(
    "3. xgboost_feature_importance.csv"
)

print(
    "4. xgboost_model_summary.csv"
)