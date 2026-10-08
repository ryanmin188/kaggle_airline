import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================
# 0. CONFIGURATION
# ============================================================

TRAIN_PATH = r"D:\kaggle\airline\data\train.csv"
TEST_PATH = r"D:\kaggle\airline\data\test.csv"
OUTPUT_DIR = r"D:\kaggle\airline"

LGBM_OOF_PATH = (
    r"D:\kaggle\airline\lgbm_oof_predictions.csv"
)

CATBOOST_OOF_PATH = (
    r"D:\kaggle\airline\catboost_oof_predictions.csv"
)

XGBOOST_OOF_PATH = (
    r"D:\kaggle\airline\xgboost_oof_predictions.csv"
)

LGBM_TEST_PATH = (
    r"D:\kaggle\airline\lgbm_test_predictions.csv"
)

CATBOOST_TEST_PATH = (
    r"D:\kaggle\airline\catboost_test_predictions.csv"
)

XGBOOST_TEST_PATH = (
    r"D:\kaggle\airline\xgboost_test_predictions.csv"
)

# Existing benchmark
CURRENT_BEST_BLEND = 0.959083


# ============================================================
# 1. LOAD OOF PREDICTIONS
# ============================================================

print("=" * 80)
print("LOADING OOF PREDICTIONS")
print("=" * 80)

start_time = time.time()

required_files = [
    LGBM_OOF_PATH,
    CATBOOST_OOF_PATH,
    XGBOOST_OOF_PATH,
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nFile not found:\n{path}"
        )


lgbm = pd.read_csv(
    LGBM_OOF_PATH
)

catboost = pd.read_csv(
    CATBOOST_OOF_PATH
)

xgboost = pd.read_csv(
    XGBOOST_OOF_PATH
)


print(
    f"LightGBM OOF shape: "
    f"{lgbm.shape}"
)

print(
    f"CatBoost OOF shape: "
    f"{catboost.shape}"
)

print(
    f"XGBoost OOF shape: "
    f"{xgboost.shape}"
)


# ============================================================
# 2. VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = {

    "LightGBM": [
        "id",
        "satisfaction",
        "lgbm_oof",
    ],

    "CatBoost": [
        "id",
        "catboost_oof",
    ],

    "XGBoost": [
        "id",
        "xgboost_oof",
    ],
}


for model_name, columns in (
    required_columns.items()
):

    df = {
        "LightGBM": lgbm,
        "CatBoost": catboost,
        "XGBoost": xgboost,
    }[model_name]

    for col in columns:

        if col not in df.columns:

            raise ValueError(
                f"{model_name} OOF is missing "
                f"column: {col}"
            )


# ============================================================
# 3. MERGE BY ID
#
# NEVER rely on row order.
# ============================================================

oof = lgbm[
    [
        "id",
        "satisfaction",
        "lgbm_oof",
    ]
].copy()


oof = oof.merge(
    catboost[
        [
            "id",
            "catboost_oof",
        ]
    ],
    on="id",
    how="left",
    validate="one_to_one",
)


oof = oof.merge(
    xgboost[
        [
            "id",
            "xgboost_oof",
        ]
    ],
    on="id",
    how="left",
    validate="one_to_one",
)


# ============================================================
# 4. CHECK ALIGNMENT
# ============================================================

prediction_columns = [
    "lgbm_oof",
    "catboost_oof",
    "xgboost_oof",
]

for col in prediction_columns:

    missing = (
        oof[col]
        .isna()
        .sum()
    )

    if missing > 0:

        raise ValueError(
            f"{col} contains "
            f"{missing} missing predictions."
        )


if len(oof) != len(lgbm):

    raise ValueError(
        "Merged OOF row count does not "
        "match LightGBM OOF."
    )


print("\n" + "=" * 80)
print("OOF ALIGNMENT CHECK")
print("=" * 80)

print(
    f"Final OOF rows: {len(oof)}"
)


# ============================================================
# 5. INDIVIDUAL MODEL OOF AUC
# ============================================================

y = oof[
    "satisfaction"
].astype(int)


lgbm_auc = roc_auc_score(
    y,
    oof["lgbm_oof"]
)

catboost_auc = roc_auc_score(
    y,
    oof["catboost_oof"]
)

xgboost_auc = roc_auc_score(
    y,
    oof["xgboost_oof"]
)


print("\n" + "=" * 80)
print("INDIVIDUAL OOF AUC")
print("=" * 80)

print(
    f"LightGBM : "
    f"{lgbm_auc:.6f}"
)

print(
    f"CatBoost : "
    f"{catboost_auc:.6f}"
)

print(
    f"XGBoost  : "
    f"{xgboost_auc:.6f}"
)


# ============================================================
# 6. PREDICTION CORRELATION
# ============================================================

corr_matrix = (
    oof[
        prediction_columns
    ]
    .corr()
)


print("\n" + "=" * 80)
print("OOF PREDICTION CORRELATION")
print("=" * 80)

print(
    corr_matrix.to_string()
)


# ============================================================
# 7. WEIGHTED THREE-MODEL BLENDING
#
# w_lgbm + w_catboost + w_xgboost = 1
#
# First:
# coarse search step = 0.01
#
# Then:
# local refinement around the best point
# with step = 0.001
# ============================================================

print("\n" + "=" * 80)
print("THREE-MODEL BLENDING SEARCH")
print("=" * 80)


lgb_pred = (
    oof["lgbm_oof"].values
)

cat_pred = (
    oof["catboost_oof"].values
)

xgb_pred = (
    oof["xgboost_oof"].values
)


# ============================================================
# 8. COARSE SEARCH
# ============================================================

coarse_results = []

best_auc = -np.inf

best_lgbm_weight = None
best_catboost_weight = None
best_xgboost_weight = None


step = 0.01


for lgbm_weight in np.arange(
    0.0,
    1.0001,
    step
):

    for catboost_weight in np.arange(
        0.0,
        1.0001 - lgbm_weight,
        step
    ):

        xgboost_weight = (
            1.0
            - lgbm_weight
            - catboost_weight
        )

        if xgboost_weight < -1e-9:
            continue

        xgboost_weight = round(
            xgboost_weight,
            10
        )

        blend_pred = (
            lgbm_weight * lgb_pred
            +
            catboost_weight * cat_pred
            +
            xgboost_weight * xgb_pred
        )

        auc = roc_auc_score(
            y,
            blend_pred
        )

        coarse_results.append({

            "lgbm_weight":
                lgbm_weight,

            "catboost_weight":
                catboost_weight,

            "xgboost_weight":
                xgboost_weight,

            "oof_auc":
                auc,

            "improvement_vs_lgbm":
                auc - lgbm_auc,

            "improvement_vs_current_blend":
                auc - CURRENT_BEST_BLEND,
        })

        if auc > best_auc:

            best_auc = auc

            best_lgbm_weight = (
                lgbm_weight
            )

            best_catboost_weight = (
                catboost_weight
            )

            best_xgboost_weight = (
                xgboost_weight
            )


coarse_df = (
    pd.DataFrame(
        coarse_results
    )
    .sort_values(
        "oof_auc",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\nCoarse best:")
print(
    f"LightGBM: "
    f"{best_lgbm_weight:.3f}"
)

print(
    f"CatBoost: "
    f"{best_catboost_weight:.3f}"
)

print(
    f"XGBoost: "
    f"{best_xgboost_weight:.3f}"
)

print(
    f"OOF AUC: "
    f"{best_auc:.6f}"
)


# ============================================================
# 9. LOCAL REFINEMENT
#
# Search ±0.03 around coarse optimum.
# Step = 0.001
# ============================================================

refined_results = []

coarse_lgbm = (
    best_lgbm_weight
)

coarse_catboost = (
    best_catboost_weight
)

coarse_xgboost = (
    best_xgboost_weight
)


for lgbm_weight in np.arange(
    max(0.0, coarse_lgbm - 0.03),
    min(1.0, coarse_lgbm + 0.03)
    + 0.0001,
    0.001
):

    for catboost_weight in np.arange(
        max(0.0, coarse_catboost - 0.03),
        min(1.0, coarse_catboost + 0.03)
        + 0.0001,
        0.001
    ):

        xgboost_weight = (
            1.0
            - lgbm_weight
            - catboost_weight
        )

        if xgboost_weight < 0:
            continue

        if xgboost_weight > 1:
            continue

        blend_pred = (
            lgbm_weight * lgb_pred
            +
            catboost_weight * cat_pred
            +
            xgboost_weight * xgb_pred
        )

        auc = roc_auc_score(
            y,
            blend_pred
        )

        refined_results.append({

            "lgbm_weight":
                lgbm_weight,

            "catboost_weight":
                catboost_weight,

            "xgboost_weight":
                xgboost_weight,

            "oof_auc":
                auc,

            "improvement_vs_lgbm":
                auc - lgbm_auc,

            "improvement_vs_current_blend":
                auc - CURRENT_BEST_BLEND,
        })


refined_df = (
    pd.DataFrame(
        refined_results
    )
    .sort_values(
        "oof_auc",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# 10. BEST THREE-MODEL BLEND
# ============================================================

best = (
    refined_df.iloc[0]
)

best_lgbm_weight = (
    best["lgbm_weight"]
)

best_catboost_weight = (
    best["catboost_weight"]
)

best_xgboost_weight = (
    best["xgboost_weight"]
)

best_blend_auc = (
    best["oof_auc"]
)


best_blend_improvement_lgbm = (
    best_blend_auc
    - lgbm_auc
)

best_blend_improvement_current = (
    best_blend_auc
    - CURRENT_BEST_BLEND
)


# ============================================================
# 11. BEST THREE-MODEL OOF PREDICTION
# ============================================================

best_blend_oof = (
    best_lgbm_weight * lgb_pred
    +
    best_catboost_weight * cat_pred
    +
    best_xgboost_weight * xgb_pred
)


# ============================================================
# 12. PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("BEST THREE-MODEL BLEND")
print("=" * 80)

print(
    f"LightGBM weight: "
    f"{best_lgbm_weight:.3f}"
)

print(
    f"CatBoost weight: "
    f"{best_catboost_weight:.3f}"
)

print(
    f"XGBoost weight: "
    f"{best_xgboost_weight:.3f}"
)

print("-" * 80)

print(
    f"LightGBM OOF AUC: "
    f"{lgbm_auc:.6f}"
)

print(
    f"Current LGBM+CatBoost: "
    f"{CURRENT_BEST_BLEND:.6f}"
)

print(
    f"Best 3-model Blend: "
    f"{best_blend_auc:.6f}"
)

print(
    f"Improvement vs LightGBM: "
    f"{best_blend_improvement_lgbm:+.6f}"
)

print(
    f"Improvement vs LGBM+CatBoost: "
    f"{best_blend_improvement_current:+.6f}"
)


# ============================================================
# 13. TOP 20 WEIGHTS
# ============================================================

print("\n" + "=" * 80)
print("TOP 20 THREE-MODEL WEIGHTS")
print("=" * 80)

print(
    refined_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# 14. SAVE BLENDING SEARCH RESULTS
# ============================================================

coarse_path = (
    OUTPUT_DIR
    + r"\ensemble_02_coarse_results.csv"
)

refined_path = (
    OUTPUT_DIR
    + r"\ensemble_02_refined_results.csv"
)

coarse_df.to_csv(
    coarse_path,
    index=False
)

refined_df.to_csv(
    refined_path,
    index=False
)


# ============================================================
# 15. SAVE BEST OOF PREDICTION
# ============================================================

oof[
    "best_ensemble_oof"
] = best_blend_oof


best_oof_output = oof[
    [
        "id",
        "satisfaction",
        "lgbm_oof",
        "catboost_oof",
        "xgboost_oof",
        "best_ensemble_oof",
    ]
].copy()


best_oof_path = (
    OUTPUT_DIR
    + r"\ensemble_02_best_oof_predictions.csv"
)

best_oof_output.to_csv(
    best_oof_path,
    index=False
)


# ============================================================
# 16. LOAD TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("LOADING TEST PREDICTIONS")
print("=" * 80)

test_required_files = [
    LGBM_TEST_PATH,
    CATBOOST_TEST_PATH,
    XGBOOST_TEST_PATH,
]

for path in test_required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nTest prediction file not found:\n"
            f"{path}"
        )


lgbm_test = pd.read_csv(
    LGBM_TEST_PATH
)

catboost_test = pd.read_csv(
    CATBOOST_TEST_PATH
)

xgboost_test = pd.read_csv(
    XGBOOST_TEST_PATH
)


# ============================================================
# 17. VALIDATE TEST COLUMNS
# ============================================================

if not {
    "id",
    "lgbm_prediction",
}.issubset(lgbm_test.columns):

    raise ValueError(
        "Invalid LightGBM test prediction file."
    )


if not {
    "id",
    "catboost_prediction",
}.issubset(catboost_test.columns):

    raise ValueError(
        "Invalid CatBoost test prediction file."
    )


if not {
    "id",
    "xgboost_prediction",
}.issubset(xgboost_test.columns):

    raise ValueError(
        "Invalid XGBoost test prediction file."
    )


# ============================================================
# 18. MERGE TEST PREDICTIONS BY ID
# ============================================================

test_pred = lgbm_test[
    [
        "id",
        "lgbm_prediction",
    ]
].copy()


test_pred = test_pred.merge(
    catboost_test[
        [
            "id",
            "catboost_prediction",
        ]
    ],
    on="id",
    how="left",
    validate="one_to_one"
)


test_pred = test_pred.merge(
    xgboost_test[
        [
            "id",
            "xgboost_prediction",
        ]
    ],
    on="id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# 19. TEST ALIGNMENT CHECK
# ============================================================

for col in [
    "lgbm_prediction",
    "catboost_prediction",
    "xgboost_prediction",
]:

    missing = (
        test_pred[col]
        .isna()
        .sum()
    )

    if missing > 0:

        raise ValueError(
            f"{col} has "
            f"{missing} missing predictions."
        )


print(
    f"Test rows: "
    f"{len(test_pred)}"
)


# ============================================================
# 20. GENERATE FINAL TEST ENSEMBLE
# ============================================================

test_pred[
    "ensemble_prediction"
] = (

    best_lgbm_weight
    * test_pred[
        "lgbm_prediction"
    ]

    +

    best_catboost_weight
    * test_pred[
        "catboost_prediction"
    ]

    +

    best_xgboost_weight
    * test_pred[
        "xgboost_prediction"
    ]
)


# ============================================================
# 21. SAVE TEST ENSEMBLE
# ============================================================

test_output_path = (
    OUTPUT_DIR
    + r"\ensemble_02_best_test_predictions.csv"
)

test_pred.to_csv(
    test_output_path,
    index=False
)


# ============================================================
# 22. SAVE SUMMARY
# ============================================================

summary = pd.DataFrame([{

    "lgbm_oof_auc":
        lgbm_auc,

    "catboost_oof_auc":
        catboost_auc,

    "xgboost_oof_auc":
        xgboost_auc,

    "lgbm_catboost_previous_blend_auc":
        CURRENT_BEST_BLEND,

    "best_lgbm_weight":
        best_lgbm_weight,

    "best_catboost_weight":
        best_catboost_weight,

    "best_xgboost_weight":
        best_xgboost_weight,

    "best_three_model_oof_auc":
        best_blend_auc,

    "improvement_vs_lgbm":
        best_blend_improvement_lgbm,

    "improvement_vs_previous_blend":
        best_blend_improvement_current,

    "lgbm_catboost_correlation":
        corr_matrix.loc[
            "lgbm_oof",
            "catboost_oof"
        ],

    "lgbm_xgboost_correlation":
        corr_matrix.loc[
            "lgbm_oof",
            "xgboost_oof"
        ],

    "catboost_xgboost_correlation":
        corr_matrix.loc[
            "catboost_oof",
            "xgboost_oof"
        ],

    "n_train":
        len(oof),

    "n_test":
        len(test_pred),
}])


summary_path = (
    OUTPUT_DIR
    + r"\ensemble_02_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 23. FINISH
# ============================================================

elapsed = (
    time.time()
    - start_time
)

print("\n" + "=" * 80)
print("ENSEMBLE 02 — THREE MODELS FINISHED")
print("=" * 80)

print(
    f"LightGBM OOF: "
    f"{lgbm_auc:.6f}"
)

print(
    f"CatBoost OOF: "
    f"{catboost_auc:.6f}"
)

print(
    f"XGBoost OOF: "
    f"{xgboost_auc:.6f}"
)

print()

print(
    f"Best weights:"
)

print(
    f"  LightGBM : "
    f"{best_lgbm_weight:.3f}"
)

print(
    f"  CatBoost : "
    f"{best_catboost_weight:.3f}"
)

print(
    f"  XGBoost  : "
    f"{best_xgboost_weight:.3f}"
)

print()

print(
    f"Best 3-model OOF AUC: "
    f"{best_blend_auc:.6f}"
)

print(
    f"Improvement vs LightGBM: "
    f"{best_blend_improvement_lgbm:+.6f}"
)

print(
    f"Improvement vs previous "
    f"LGBM+CatBoost: "
    f"{best_blend_improvement_current:+.6f}"
)

print()

print(
    f"Elapsed time: "
    f"{elapsed / 60:.2f} minutes"
)

print("\nOutput files:")

print(
    "1. ensemble_02_coarse_results.csv"
)

print(
    "2. ensemble_02_refined_results.csv"
)

print(
    "3. ensemble_02_best_oof_predictions.csv"
)

print(
    "4. ensemble_02_best_test_predictions.csv"
)

print(
    "5. ensemble_02_summary.csv"
)