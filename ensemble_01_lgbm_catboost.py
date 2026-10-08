import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

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

# Existing CatBoost outputs
CATBOOST_OOF_PATH = (
    r"D:\kaggle\airline\catboost_oof_predictions.csv"
)

CATBOOST_TEST_PATH = (
    r"D:\kaggle\airline\catboost_test_predictions.csv"
)

# Known LightGBM benchmark
LGBM_BENCHMARK = 0.959015


# ============================================================
# 1. FINAL LIGHTGBM PARAMETERS — LOCKED
# ============================================================

LGB_PARAMS = {
    "objective": "binary",
    "n_estimators": 960,
    "learning_rate": 0.01,
    "num_leaves": 127,
    "max_depth": -1,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}


# ============================================================
# 2. FINAL CATEGORICAL INTERACTIONS — LOCKED
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
# 3. LOAD DATA
# ============================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

start_time = time.time()

train = pd.read_csv(TRAIN_PATH)

print(
    f"Train shape: {train.shape}"
)

if not os.path.exists(
    CATBOOST_OOF_PATH
):
    raise FileNotFoundError(
        "\nCatBoost OOF file not found:\n"
        + CATBOOST_OOF_PATH
    )

if not os.path.exists(
    CATBOOST_TEST_PATH
):
    raise FileNotFoundError(
        "\nCatBoost test prediction file not found:\n"
        + CATBOOST_TEST_PATH
    )


# ============================================================
# 4. BUILD FINAL LIGHTGBM FEATURES
# ============================================================

def build_final_lgbm_features(df):

    X = df.copy()

    columns_to_drop = []

    if TARGET in X.columns:
        columns_to_drop.append(TARGET)

    if ID_COL in X.columns:
        columns_to_drop.append(ID_COL)

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
    # Add confirmed interactions
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
    # Convert categorical variables
    # --------------------------------------------------------

    for col in categorical_features:

        X[col] = (
            X[col]
            .astype("category")
        )

    return X, categorical_features


# ============================================================
# 5. PREPARE TRAIN
# ============================================================

X_train, categorical_features = (
    build_final_lgbm_features(
        train
    )
)

y = train[
    TARGET
].astype(int)

print("\nFinal LightGBM feature shape:")
print(
    X_train.shape
)

print("\nCategorical features:")
for col in categorical_features:
    print(
        f" - {col}"
    )


# ============================================================
# 6. LOAD TEST
# ============================================================

test = pd.read_csv(
    TEST_PATH
)

X_test, test_categorical_features = (
    build_final_lgbm_features(
        test
    )
)

# ------------------------------------------------------------
# Ensure exact feature order
# ------------------------------------------------------------

if list(X_train.columns) != list(
    X_test.columns
):

    raise ValueError(
        "\nTrain and test feature columns "
        "do not match."
    )


# ============================================================
# 7. SAME 5-FOLD SPLITS
# ============================================================

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# 8. STORAGE
# ============================================================

lgbm_oof = np.zeros(
    len(train),
    dtype=float
)

lgbm_test = np.zeros(
    len(test),
    dtype=float
)

lgbm_fold_scores = []


# ============================================================
# 9. LIGHTGBM 5-FOLD OOF
# ============================================================

print("\n" + "=" * 80)
print("LIGHTGBM 5-FOLD OOF")
print("=" * 80)

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
    # Train
    # --------------------------------------------------------

    model = lgb.LGBMClassifier(
        **LGB_PARAMS
    )

    model.fit(
        X_tr,
        y_tr,
        categorical_feature=categorical_features
    )

    # --------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------

    val_pred = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    fold_auc = roc_auc_score(
        y_val,
        val_pred
    )

    lgbm_fold_scores.append(
        fold_auc
    )

    lgbm_oof[
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

    lgbm_test += (
        test_pred
        / N_FOLDS
    )

    print(
        f"Fold {fold} AUC: "
        f"{fold_auc:.6f}"
    )


# ============================================================
# 10. LIGHTGBM OOF SUMMARY
# ============================================================

lgbm_mean_auc = np.mean(
    lgbm_fold_scores
)

lgbm_std_auc = np.std(
    lgbm_fold_scores,
    ddof=0
)

lgbm_oof_auc = roc_auc_score(
    y,
    lgbm_oof
)

print("\n" + "=" * 80)
print("LIGHTGBM OOF SUMMARY")
print("=" * 80)

for i, score in enumerate(
    lgbm_fold_scores,
    start=1
):

    print(
        f"Fold {i}: "
        f"{score:.6f}"
    )

print("-" * 80)

print(
    f"5-Fold Mean AUC: "
    f"{lgbm_mean_auc:.6f}"
)

print(
    f"5-Fold Std:      "
    f"{lgbm_std_auc:.6f}"
)

print(
    f"OOF AUC:         "
    f"{lgbm_oof_auc:.6f}"
)

print(
    f"Difference vs known 0.959015: "
    f"{lgbm_mean_auc - LGBM_BENCHMARK:+.6f}"
)


# ============================================================
# 11. SAVE LIGHTGBM OOF
# ============================================================

lgbm_oof_output = pd.DataFrame({

    "id":
        train[ID_COL].values,

    "satisfaction":
        y.values,

    "lgbm_oof":
        lgbm_oof,
})

lgbm_oof_path = (
    OUTPUT_DIR
    + r"\lgbm_oof_predictions.csv"
)

lgbm_oof_output.to_csv(
    lgbm_oof_path,
    index=False
)

print(
    f"\nSaved LightGBM OOF:"
    f"\n{lgbm_oof_path}"
)


# ============================================================
# 12. SAVE LIGHTGBM TEST PREDICTIONS
# ============================================================

lgbm_test_output = pd.DataFrame({

    "id":
        test[ID_COL].values,

    "lgbm_prediction":
        lgbm_test,
})

lgbm_test_path = (
    OUTPUT_DIR
    + r"\lgbm_test_predictions.csv"
)

lgbm_test_output.to_csv(
    lgbm_test_path,
    index=False
)

print(
    f"\nSaved LightGBM test predictions:"
    f"\n{lgbm_test_path}"
)


# ============================================================
# 13. LOAD CATBOOST OOF
# ============================================================

print("\n" + "=" * 80)
print("LOADING CATBOOST OOF")
print("=" * 80)

cat_oof_df = pd.read_csv(
    CATBOOST_OOF_PATH
)

cat_test_df = pd.read_csv(
    CATBOOST_TEST_PATH
)

print(
    f"CatBoost OOF shape: "
    f"{cat_oof_df.shape}"
)

print(
    f"CatBoost test shape: "
    f"{cat_test_df.shape}"
)


# ============================================================
# 14. VALIDATE CATBOOST OOF
# ============================================================

required_cat_oof_columns = [
    "id",
    "catboost_oof",
]

for col in required_cat_oof_columns:

    if col not in cat_oof_df.columns:

        raise ValueError(
            f"Missing column in CatBoost OOF: "
            f"{col}"
        )

required_cat_test_columns = [
    "id",
    "catboost_prediction",
]

for col in required_cat_test_columns:

    if col not in cat_test_df.columns:

        raise ValueError(
            f"Missing column in CatBoost test: "
            f"{col}"
        )


# ============================================================
# 15. MERGE OOF BY ID
#
# Never assume row order.
# Explicitly align by ID.
# ============================================================

oof_df = pd.DataFrame({

    "id":
        train[ID_COL].values,

    "satisfaction":
        y.values,

    "lgbm_oof":
        lgbm_oof,
})

oof_df = oof_df.merge(
    cat_oof_df[
        [
            "id",
            "catboost_oof",
        ]
    ],
    on="id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 16. CHECK MISSING CATBOOST OOF
# ============================================================

missing_cat_oof = (
    oof_df["catboost_oof"]
    .isna()
    .sum()
)

if missing_cat_oof > 0:

    raise ValueError(
        f"Missing CatBoost OOF "
        f"predictions for "
        f"{missing_cat_oof} rows."
    )


# ============================================================
# 17. PREDICTION CORRELATION
# ============================================================

lgbm_cat_corr = (
    oof_df[
        [
            "lgbm_oof",
            "catboost_oof",
        ]
    ]
    .corr()
    .loc[
        "lgbm_oof",
        "catboost_oof"
    ]
)

print("\n" + "=" * 80)
print("OOF PREDICTION CORRELATION")
print("=" * 80)

print(
    f"Pearson correlation: "
    f"{lgbm_cat_corr:.6f}"
)


# ============================================================
# 18. INDIVIDUAL OOF AUC CHECK
# ============================================================

cat_oof_auc = roc_auc_score(
    oof_df["satisfaction"],
    oof_df["catboost_oof"]
)

lgbm_oof_auc_check = roc_auc_score(
    oof_df["satisfaction"],
    oof_df["lgbm_oof"]
)

print("\n" + "=" * 80)
print("INDIVIDUAL OOF AUC")
print("=" * 80)

print(
    f"LightGBM OOF: "
    f"{lgbm_oof_auc_check:.6f}"
)

print(
    f"CatBoost OOF: "
    f"{cat_oof_auc:.6f}"
)

print(
    f"Difference: "
    f"{cat_oof_auc - lgbm_oof_auc_check:+.6f}"
)


# ============================================================
# 19. WEIGHTED BLENDING
#
# weight = share assigned to LightGBM
#
# Final prediction:
#
# blend =
# weight * LGBM
# +
# (1-weight) * CatBoost
#
# Scan every 0.5 percentage point.
# ============================================================

print("\n" + "=" * 80)
print("WEIGHTED BLENDING SEARCH")
print("=" * 80)

blend_results = []

best_auc = -np.inf
best_weight = None
best_prediction = None


for weight in np.arange(
    0.0,
    1.0001,
    0.005
):

    blend_pred = (
        weight
        * oof_df["lgbm_oof"].values
        +
        (1.0 - weight)
        * oof_df["catboost_oof"].values
    )

    auc = roc_auc_score(
        oof_df["satisfaction"],
        blend_pred
    )

    blend_results.append({

        "lgbm_weight":
            weight,

        "catboost_weight":
            1.0 - weight,

        "oof_auc":
            auc,

        "improvement_vs_lgbm":
            auc
            - lgbm_oof_auc_check,
    })

    if auc > best_auc:

        best_auc = auc

        best_weight = weight

        best_prediction = (
            blend_pred.copy()
        )


blend_df = pd.DataFrame(
    blend_results
).sort_values(
    "oof_auc",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# 20. BEST BLEND RESULT
# ============================================================

best_cat_weight = (
    1.0 - best_weight
)

best_improvement = (
    best_auc
    - lgbm_oof_auc_check
)

print("\n" + "=" * 80)
print("BEST BLEND RESULT")
print("=" * 80)

print(
    f"LightGBM weight: "
    f"{best_weight:.3f}"
)

print(
    f"CatBoost weight: "
    f"{best_cat_weight:.3f}"
)

print(
    f"Blend OOF AUC: "
    f"{best_auc:.6f}"
)

print(
    f"LightGBM OOF AUC: "
    f"{lgbm_oof_auc_check:.6f}"
)

print(
    f"Improvement: "
    f"{best_improvement:+.6f}"
)


# ============================================================
# 21. TOP 10 BLENDING RESULTS
# ============================================================

print("\n" + "=" * 80)
print("TOP 10 BLENDING WEIGHTS")
print("=" * 80)

print(
    blend_df.head(10).to_string(
        index=False
    )
)


# ============================================================
# 22. SAVE ALL BLENDING RESULTS
# ============================================================

blend_results_path = (
    OUTPUT_DIR
    + r"\ensemble_01_blending_results.csv"
)

blend_df.to_csv(
    blend_results_path,
    index=False
)

print(
    f"\nSaved blending results:"
    f"\n{blend_results_path}"
)


# ============================================================
# 23. SAVE BEST BLENDED OOF
# ============================================================

oof_df["best_blend_oof"] = (
    best_prediction
)

best_oof_path = (
    OUTPUT_DIR
    + r"\ensemble_01_best_oof_predictions.csv"
)

oof_df.to_csv(
    best_oof_path,
    index=False
)

print(
    f"\nSaved best blended OOF:"
    f"\n{best_oof_path}"
)


# ============================================================
# 24. ALIGN TEST PREDICTIONS
# ============================================================

test_pred_df = pd.DataFrame({

    "id":
        test[ID_COL].values,

    "lgbm_prediction":
        lgbm_test,
})

test_pred_df = test_pred_df.merge(
    cat_test_df[
        [
            "id",
            "catboost_prediction",
        ]
    ],
    on="id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 25. CHECK TEST ALIGNMENT
# ============================================================

missing_cat_test = (
    test_pred_df[
        "catboost_prediction"
    ]
    .isna()
    .sum()
)

if missing_cat_test > 0:

    raise ValueError(
        f"Missing CatBoost test "
        f"predictions for "
        f"{missing_cat_test} rows."
    )


# ============================================================
# 26. GENERATE BEST BLENDED TEST PREDICTION
# ============================================================

test_pred_df[
    "best_blend_prediction"
] = (

    best_weight
    * test_pred_df[
        "lgbm_prediction"
    ]

    +

    best_cat_weight
    * test_pred_df[
        "catboost_prediction"
    ]
)


# ============================================================
# 27. SAVE TEST BLEND
# ============================================================

test_blend_path = (
    OUTPUT_DIR
    + r"\ensemble_01_best_test_predictions.csv"
)

test_pred_df.to_csv(
    test_blend_path,
    index=False
)

print(
    f"\nSaved best blended test prediction:"
    f"\n{test_blend_path}"
)


# ============================================================
# 28. SAVE ENSEMBLE SUMMARY
# ============================================================

summary = pd.DataFrame([{

    "model_1":
        "LightGBM",

    "model_1_oof_auc":
        lgbm_oof_auc_check,

    "model_2":
        "CatBoost",

    "model_2_oof_auc":
        cat_oof_auc,

    "prediction_correlation":
        lgbm_cat_corr,

    "best_lgbm_weight":
        best_weight,

    "best_catboost_weight":
        best_cat_weight,

    "best_blend_oof_auc":
        best_auc,

    "blend_improvement_vs_lgbm":
        best_improvement,

    "known_lgbm_benchmark":
        LGBM_BENCHMARK,

}])


summary_path = (
    OUTPUT_DIR
    + r"\ensemble_01_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 29. FINAL REPORT
# ============================================================

elapsed = (
    time.time()
    - start_time
)

print("\n" + "=" * 80)
print("ENSEMBLE 01 — LGBM + CATBOOST FINISHED")
print("=" * 80)

print(
    f"LightGBM OOF AUC: "
    f"{lgbm_oof_auc_check:.6f}"
)

print(
    f"CatBoost OOF AUC: "
    f"{cat_oof_auc:.6f}"
)

print(
    f"Prediction correlation: "
    f"{lgbm_cat_corr:.6f}"
)

print()

print(
    f"Best LightGBM weight: "
    f"{best_weight:.3f}"
)

print(
    f"Best CatBoost weight: "
    f"{best_cat_weight:.3f}"
)

print(
    f"Best Blend OOF AUC: "
    f"{best_auc:.6f}"
)

print(
    f"Improvement vs LightGBM: "
    f"{best_improvement:+.6f}"
)

print()

print(
    f"Elapsed time: "
    f"{elapsed / 60:.2f} minutes"
)

print("\nOutput files:")

print(
    "1. lgbm_oof_predictions.csv"
)

print(
    "2. lgbm_test_predictions.csv"
)

print(
    "3. ensemble_01_blending_results.csv"
)

print(
    "4. ensemble_01_best_oof_predictions.csv"
)

print(
    "5. ensemble_01_best_test_predictions.csv"
)

print(
    "6. ensemble_01_summary.csv"
)