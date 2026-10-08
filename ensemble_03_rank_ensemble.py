import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================
# 0. CONFIGURATION
# ============================================================

OUTPUT_DIR = r"D:\kaggle\airline"

# ------------------------------------------------------------
# OOF prediction files
# ------------------------------------------------------------

LGBM_OOF_PATH = (
    r"D:\kaggle\airline\lgbm_oof_predictions.csv"
)

CATBOOST_OOF_PATH = (
    r"D:\kaggle\airline\catboost_oof_predictions.csv"
)

XGBOOST_OOF_PATH = (
    r"D:\kaggle\airline\xgboost_oof_predictions.csv"
)

# ------------------------------------------------------------
# Test prediction files
# ------------------------------------------------------------

LGBM_TEST_PATH = (
    r"D:\kaggle\airline\lgbm_test_predictions.csv"
)

CATBOOST_TEST_PATH = (
    r"D:\kaggle\airline\catboost_test_predictions.csv"
)

XGBOOST_TEST_PATH = (
    r"D:\kaggle\airline\xgboost_test_predictions.csv"
)

# ------------------------------------------------------------
# Sample submission
# ------------------------------------------------------------

SAMPLE_PATH = (
    r"D:\kaggle\airline\data\sample_submission.csv"
)


# ============================================================
# 1. EXISTING BENCHMARKS
# ============================================================

LIGHTGBM_OOF_AUC = 0.959003

LGBM_CATBOOST_OOF_AUC = 0.959083

THREE_MODEL_PROB_OOF_AUC = 0.959225

THREE_MODEL_PROB_WEIGHTS = {
    "lgbm": 0.536,
    "catboost": 0.110,
    "xgboost": 0.354,
}


# ============================================================
# 2. START
# ============================================================

start_time = time.time()

print("=" * 80)
print("ENSEMBLE 03 — FAST RANK ENSEMBLE")
print("=" * 80)

print(
    "No model training will be performed."
)

print(
    "Only existing OOF/Test predictions will be used."
)


# ============================================================
# 3. CHECK FILES
# ============================================================

required_files = [
    LGBM_OOF_PATH,
    CATBOOST_OOF_PATH,
    XGBOOST_OOF_PATH,
    LGBM_TEST_PATH,
    CATBOOST_TEST_PATH,
    XGBOOST_TEST_PATH,
    SAMPLE_PATH,
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


# ============================================================
# 4. LOAD OOF PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("LOADING OOF PREDICTIONS")
print("=" * 80)

lgbm_oof_df = pd.read_csv(
    LGBM_OOF_PATH
)

catboost_oof_df = pd.read_csv(
    CATBOOST_OOF_PATH
)

xgboost_oof_df = pd.read_csv(
    XGBOOST_OOF_PATH
)

print(
    f"LightGBM OOF: {lgbm_oof_df.shape}"
)

print(
    f"CatBoost OOF: {catboost_oof_df.shape}"
)

print(
    f"XGBoost OOF:  {xgboost_oof_df.shape}"
)


# ============================================================
# 5. VALIDATE OOF COLUMNS
# ============================================================

if not {
    "id",
    "satisfaction",
    "lgbm_oof",
}.issubset(
    lgbm_oof_df.columns
):

    raise ValueError(
        "Invalid LightGBM OOF file."
    )


if not {
    "id",
    "catboost_oof",
}.issubset(
    catboost_oof_df.columns
):

    raise ValueError(
        "Invalid CatBoost OOF file."
    )


if not {
    "id",
    "xgboost_oof",
}.issubset(
    xgboost_oof_df.columns
):

    raise ValueError(
        "Invalid XGBoost OOF file."
    )


# ============================================================
# 6. MERGE OOF BY ID
# ============================================================

oof = lgbm_oof_df[
    [
        "id",
        "satisfaction",
        "lgbm_oof",
    ]
].copy()


oof = oof.merge(
    catboost_oof_df[
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
    xgboost_oof_df[
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
# 7. OOF ALIGNMENT CHECK
# ============================================================

prediction_columns = [
    "lgbm_oof",
    "catboost_oof",
    "xgboost_oof",
]

if len(oof) != len(
    lgbm_oof_df
):

    raise ValueError(
        "OOF row count changed after merging."
    )


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


y = (
    oof["satisfaction"]
    .astype(int)
    .to_numpy()
)


print("\n" + "=" * 80)
print("OOF ALIGNMENT PASSED")
print("=" * 80)

print(
    f"OOF rows: {len(oof)}"
)


# ============================================================
# 8. INDIVIDUAL OOF AUC
# ============================================================

lgbm_pred = (
    oof["lgbm_oof"]
    .to_numpy(dtype=np.float64)
)

cat_pred = (
    oof["catboost_oof"]
    .to_numpy(dtype=np.float64)
)

xgb_pred = (
    oof["xgboost_oof"]
    .to_numpy(dtype=np.float64)
)


lgbm_auc = roc_auc_score(
    y,
    lgbm_pred
)

catboost_auc = roc_auc_score(
    y,
    cat_pred
)

xgboost_auc = roc_auc_score(
    y,
    xgb_pred
)


print("\n" + "=" * 80)
print("INDIVIDUAL OOF AUC")
print("=" * 80)

print(
    f"LightGBM : {lgbm_auc:.6f}"
)

print(
    f"CatBoost : {catboost_auc:.6f}"
)

print(
    f"XGBoost  : {xgboost_auc:.6f}"
)


# ============================================================
# 9. FAST RANK FUNCTION
# ============================================================

def normalized_rank(values):
    """
    Convert predictions into percentile-like ranks.

    The ordering of predictions is preserved.
    Ties receive the same average rank.
    """

    series = pd.Series(values)

    ranks = (
        series.rank(
            method="average",
            pct=True
        )
        .to_numpy(
            dtype=np.float64
        )
    )

    return ranks


# ============================================================
# 10. CREATE OOF RANKS
# ============================================================

print("\n" + "=" * 80)
print("CREATING OOF RANKS")
print("=" * 80)

lgb_rank = normalized_rank(
    lgbm_pred
)

cat_rank = normalized_rank(
    cat_pred
)

xgb_rank = normalized_rank(
    xgb_pred
)


# ============================================================
# 11. RANK CORRELATION
# ============================================================

rank_corr = pd.DataFrame({

    "lgbm_rank":
        lgb_rank,

    "catboost_rank":
        cat_rank,

    "xgboost_rank":
        xgb_rank,

}).corr()


print("\n" + "=" * 80)
print("RANK CORRELATION")
print("=" * 80)

print(
    rank_corr.to_string()
)


# ============================================================
# 12. EQUAL-WEIGHT RANK ENSEMBLE
# ============================================================

equal_rank_prediction = (
    lgb_rank
    + cat_rank
    + xgb_rank
) / 3.0


equal_rank_auc = roc_auc_score(
    y,
    equal_rank_prediction
)


print("\n" + "=" * 80)
print("EQUAL-WEIGHT RANK ENSEMBLE")
print("=" * 80)

print(
    f"Equal Rank AUC: "
    f"{equal_rank_auc:.6f}"
)

print(
    f"Improvement vs LightGBM: "
    f"{equal_rank_auc - lgbm_auc:+.6f}"
)

print(
    f"Difference vs previous "
    f"probability ensemble: "
    f"{equal_rank_auc - THREE_MODEL_PROB_OOF_AUC:+.6f}"
)


# ============================================================
# 13. FAST WEIGHTED-RANK CANDIDATES
#
# Instead of thousands of combinations, test a compact set
# around the currently successful probability ensemble.
# ============================================================

candidate_weights = [

    # --------------------------------------------------------
    # Current best probability ensemble
    # --------------------------------------------------------

    (0.536, 0.110, 0.354),

    # --------------------------------------------------------
    # Equal / balanced
    # --------------------------------------------------------

    (1 / 3, 1 / 3, 1 / 3),

    (0.40, 0.30, 0.30),

    (0.45, 0.25, 0.30),

    (0.45, 0.20, 0.35),

    (0.50, 0.25, 0.25),

    (0.50, 0.20, 0.30),

    (0.50, 0.15, 0.35),

    (0.50, 0.10, 0.40),

    # --------------------------------------------------------
    # LGBM dominant
    # --------------------------------------------------------

    (0.55, 0.15, 0.30),

    (0.55, 0.10, 0.35),

    (0.60, 0.20, 0.20),

    (0.60, 0.15, 0.25),

    (0.60, 0.10, 0.30),

    (0.65, 0.15, 0.20),

    (0.65, 0.10, 0.25),

    (0.70, 0.15, 0.15),

    (0.70, 0.10, 0.20),

    (0.70, 0.05, 0.25),

    (0.75, 0.15, 0.10),

    (0.75, 0.10, 0.15),

    # --------------------------------------------------------
    # XGB-heavy
    # --------------------------------------------------------

    (0.40, 0.20, 0.40),

    (0.45, 0.20, 0.35),

    (0.45, 0.15, 0.40),

    # --------------------------------------------------------
    # Around current probability optimum
    # --------------------------------------------------------

    (0.51, 0.11, 0.38),

    (0.536, 0.15, 0.314),

    (0.536, 0.07, 0.394),

    (0.55, 0.11, 0.34),

    (0.56, 0.11, 0.33),

]


# Remove duplicates while preserving order.
candidate_weights = list(
    dict.fromkeys(
        candidate_weights
    )
)


# ============================================================
# 14. FAST WEIGHTED-RANK SEARCH
# ============================================================

print("\n" + "=" * 80)
print("FAST WEIGHTED RANK SEARCH")
print("=" * 80)

print(
    f"Testing "
    f"{len(candidate_weights)} "
    f"candidate combinations."
)


rank_results = []

best_rank_auc = -np.inf

best_rank_weights = None

best_rank_prediction = None


for index, (
    w_lgb,
    w_cat,
    w_xgb
) in enumerate(
    candidate_weights,
    start=1
):

    # Safety check
    weight_sum = (
        w_lgb
        + w_cat
        + w_xgb
    )

    if not np.isclose(
        weight_sum,
        1.0
    ):

        raise ValueError(
            f"Invalid weights: "
            f"{w_lgb}, "
            f"{w_cat}, "
            f"{w_xgb}"
        )

    prediction = (
        w_lgb * lgb_rank
        +
        w_cat * cat_rank
        +
        w_xgb * xgb_rank
    )

    auc = roc_auc_score(
        y,
        prediction
    )

    rank_results.append({

        "lgbm_weight":
            w_lgb,

        "catboost_weight":
            w_cat,

        "xgboost_weight":
            w_xgb,

        "oof_auc":
            auc,

        "improvement_vs_lgbm":
            auc - lgbm_auc,

        "improvement_vs_probability":
            auc
            - THREE_MODEL_PROB_OOF_AUC,

    })

    if auc > best_rank_auc:

        best_rank_auc = auc

        best_rank_weights = (
            w_lgb,
            w_cat,
            w_xgb
        )

        best_rank_prediction = (
            prediction.copy()
        )

    print(
        f"[{index:02d}/"
        f"{len(candidate_weights):02d}] "
        f"{w_lgb:.3f}/"
        f"{w_cat:.3f}/"
        f"{w_xgb:.3f} "
        f"→ "
        f"{auc:.6f}"
    )


rank_df = (
    pd.DataFrame(
        rank_results
    )
    .sort_values(
        "oof_auc",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# 15. BEST WEIGHTED RANK
# ============================================================

best_rank_lgb = (
    best_rank_weights[0]
)

best_rank_cat = (
    best_rank_weights[1]
)

best_rank_xgb = (
    best_rank_weights[2]
)


print("\n" + "=" * 80)
print("BEST WEIGHTED RANK ENSEMBLE")
print("=" * 80)

print(
    f"LightGBM weight: "
    f"{best_rank_lgb:.3f}"
)

print(
    f"CatBoost weight: "
    f"{best_rank_cat:.3f}"
)

print(
    f"XGBoost weight: "
    f"{best_rank_xgb:.3f}"
)

print(
    f"Rank OOF AUC: "
    f"{best_rank_auc:.6f}"
)

print(
    f"Improvement vs LightGBM: "
    f"{best_rank_auc - lgbm_auc:+.6f}"
)

print(
    f"Difference vs previous "
    f"probability ensemble: "
    f"{best_rank_auc - THREE_MODEL_PROB_OOF_AUC:+.6f}"
)


print("\nTop 10 rank candidates:")

print(
    rank_df.head(10).to_string(
        index=False
    )
)


# ============================================================
# 16. CURRENT BEST PROBABILITY ENSEMBLE
# ============================================================

probability_ensemble = (

    THREE_MODEL_PROB_WEIGHTS["lgbm"]
    * lgbm_pred

    +

    THREE_MODEL_PROB_WEIGHTS["catboost"]
    * cat_pred

    +

    THREE_MODEL_PROB_WEIGHTS["xgboost"]
    * xgb_pred
)


probability_ensemble_auc = (
    roc_auc_score(
        y,
        probability_ensemble
    )
)


# ============================================================
# 17. RANK OF PROBABILITY ENSEMBLE
# ============================================================

probability_ensemble_rank = (
    normalized_rank(
        probability_ensemble
    )
)

weighted_rank_rank = (
    normalized_rank(
        best_rank_prediction
    )
)


# ============================================================
# 18. FAST HYBRID SEARCH
#
# Only a small number of alpha values.
#
# alpha = probability share
# 1-alpha = rank share
# ============================================================

print("\n" + "=" * 80)
print("FAST RANK + PROBABILITY HYBRID")
print("=" * 80)


hybrid_alphas = [
    0.00,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    1.00,
]


hybrid_results = []

best_hybrid_auc = -np.inf

best_alpha = None

best_hybrid_prediction = None


for alpha in hybrid_alphas:

    hybrid_prediction = (
        alpha
        * probability_ensemble_rank

        +

        (1.0 - alpha)
        * weighted_rank_rank
    )

    auc = roc_auc_score(
        y,
        hybrid_prediction
    )

    hybrid_results.append({

        "probability_weight":
            alpha,

        "rank_weight":
            1.0 - alpha,

        "oof_auc":
            auc,

        "improvement_vs_lgbm":
            auc - lgbm_auc,

        "improvement_vs_probability":
            auc
            - THREE_MODEL_PROB_OOF_AUC,

    })

    print(
        f"alpha={alpha:.2f} "
        f"→ "
        f"AUC={auc:.6f}"
    )

    if auc > best_hybrid_auc:

        best_hybrid_auc = auc

        best_alpha = alpha

        best_hybrid_prediction = (
            hybrid_prediction.copy()
        )


hybrid_df = (
    pd.DataFrame(
        hybrid_results
    )
    .sort_values(
        "oof_auc",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\nBest hybrid:")

print(
    f"Probability weight: "
    f"{best_alpha:.2f}"
)

print(
    f"Rank weight: "
    f"{1.0 - best_alpha:.2f}"
)

print(
    f"Hybrid OOF AUC: "
    f"{best_hybrid_auc:.6f}"
)

print(
    f"Improvement vs LightGBM: "
    f"{best_hybrid_auc - lgbm_auc:+.6f}"
)

print(
    f"Difference vs previous "
    f"probability ensemble: "
    f"{best_hybrid_auc - THREE_MODEL_PROB_OOF_AUC:+.6f}"
)


# ============================================================
# 19. OVERALL OOF COMPARISON
# ============================================================

method_results = [

    {
        "method":
            "LightGBM",

        "oof_auc":
            lgbm_auc,
    },

    {
        "method":
            "Previous 3-model Probability",

        "oof_auc":
            probability_ensemble_auc,
    },

    {
        "method":
            "Equal Rank",

        "oof_auc":
            equal_rank_auc,
    },

    {
        "method":
            "Best Weighted Rank",

        "oof_auc":
            best_rank_auc,
    },

    {
        "method":
            "Best Hybrid",

        "oof_auc":
            best_hybrid_auc,
    },
]


method_df = (
    pd.DataFrame(
        method_results
    )
    .sort_values(
        "oof_auc",
        ascending=False
    )
    .reset_index(drop=True)
)


method_df[
    "improvement_vs_lgbm"
] = (
    method_df["oof_auc"]
    - lgbm_auc
)

method_df[
    "improvement_vs_previous_probability"
] = (
    method_df["oof_auc"]
    - THREE_MODEL_PROB_OOF_AUC
)


print("\n" + "=" * 80)
print("OVERALL OOF COMPARISON")
print("=" * 80)

print(
    method_df.to_string(
        index=False
    )
)


overall_best_method = (
    method_df.iloc[0]["method"]
)

overall_best_auc = (
    method_df.iloc[0]["oof_auc"]
)


# ============================================================
# 20. LOAD TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("LOADING TEST PREDICTIONS")
print("=" * 80)

lgbm_test_df = pd.read_csv(
    LGBM_TEST_PATH
)

catboost_test_df = pd.read_csv(
    CATBOOST_TEST_PATH
)

xgboost_test_df = pd.read_csv(
    XGBOOST_TEST_PATH
)

sample = pd.read_csv(
    SAMPLE_PATH
)


# ============================================================
# 21. VALIDATE TEST COLUMNS
# ============================================================

if not {
    "id",
    "lgbm_prediction",
}.issubset(
    lgbm_test_df.columns
):

    raise ValueError(
        "Invalid LightGBM test prediction file."
    )


if not {
    "id",
    "catboost_prediction",
}.issubset(
    catboost_test_df.columns
):

    raise ValueError(
        "Invalid CatBoost test prediction file."
    )


if not {
    "id",
    "xgboost_prediction",
}.issubset(
    xgboost_test_df.columns
):

    raise ValueError(
        "Invalid XGBoost test prediction file."
    )


# ============================================================
# 22. MERGE TEST BY ID
# ============================================================

test_pred = lgbm_test_df[
    [
        "id",
        "lgbm_prediction",
    ]
].copy()


test_pred = test_pred.merge(
    catboost_test_df[
        [
            "id",
            "catboost_prediction",
        ]
    ],
    on="id",
    how="left",
    validate="one_to_one",
)


test_pred = test_pred.merge(
    xgboost_test_df[
        [
            "id",
            "xgboost_prediction",
        ]
    ],
    on="id",
    how="left",
    validate="one_to_one",
)


# ============================================================
# 23. TEST ALIGNMENT CHECK
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
# 24. CREATE TEST RANKS
# ============================================================

test_lgb_rank = normalized_rank(
    test_pred[
        "lgbm_prediction"
    ].to_numpy(
        dtype=np.float64
    )
)

test_cat_rank = normalized_rank(
    test_pred[
        "catboost_prediction"
    ].to_numpy(
        dtype=np.float64
    )
)

test_xgb_rank = normalized_rank(
    test_pred[
        "xgboost_prediction"
    ].to_numpy(
        dtype=np.float64
    )
)


# ============================================================
# 25. TEST PROBABILITY ENSEMBLE
# ============================================================

test_probability_prediction = (

    THREE_MODEL_PROB_WEIGHTS["lgbm"]
    * test_pred[
        "lgbm_prediction"
    ].to_numpy(dtype=np.float64)

    +

    THREE_MODEL_PROB_WEIGHTS["catboost"]
    * test_pred[
        "catboost_prediction"
    ].to_numpy(dtype=np.float64)

    +

    THREE_MODEL_PROB_WEIGHTS["xgboost"]
    * test_pred[
        "xgboost_prediction"
    ].to_numpy(dtype=np.float64)
)


# ============================================================
# 26. TEST EQUAL RANK
# ============================================================

test_equal_rank_prediction = (

    test_lgb_rank
    +
    test_cat_rank
    +
    test_xgb_rank

) / 3.0


# ============================================================
# 27. TEST WEIGHTED RANK
# ============================================================

test_weighted_rank_prediction = (

    best_rank_lgb
    * test_lgb_rank

    +

    best_rank_cat
    * test_cat_rank

    +

    best_rank_xgb
    * test_xgb_rank
)


# ============================================================
# 28. TEST HYBRID
# ============================================================

test_probability_rank = (
    normalized_rank(
        test_probability_prediction
    )
)

test_weighted_rank_rank = (
    normalized_rank(
        test_weighted_rank_prediction
    )
)


test_hybrid_prediction = (

    best_alpha
    * test_probability_rank

    +

    (1.0 - best_alpha)
    * test_weighted_rank_rank
)


# ============================================================
# 29. SAVE TEST PREDICTIONS
# ============================================================

test_pred[
    "probability_ensemble"
] = (
    test_probability_prediction
)

test_pred[
    "equal_rank_ensemble"
] = (
    test_equal_rank_prediction
)

test_pred[
    "weighted_rank_ensemble"
] = (
    test_weighted_rank_prediction
)

test_pred[
    "hybrid_ensemble"
] = (
    test_hybrid_prediction
)


test_output_path = (
    OUTPUT_DIR
    + r"\ensemble_03_rank_test_predictions.csv"
)

test_pred.to_csv(
    test_output_path,
    index=False
)


# ============================================================
# 30. CREATE SUBMISSIONS
# ============================================================

target_columns = [
    col
    for col in sample.columns
    if col != "id"
]


if len(target_columns) != 1:

    raise ValueError(
        "sample_submission.csv should contain "
        "exactly one target column."
    )


target_col = target_columns[0]


# ------------------------------------------------------------
# 30A. Equal Rank
# ------------------------------------------------------------

submission_05 = sample.copy()

submission_05[target_col] = (
    test_equal_rank_prediction
)

submission_05_path = (
    OUTPUT_DIR
    + r"\submission_05_equal_rank.csv"
)

submission_05.to_csv(
    submission_05_path,
    index=False
)


# ------------------------------------------------------------
# 30B. Weighted Rank
# ------------------------------------------------------------

submission_06 = sample.copy()

submission_06[target_col] = (
    test_weighted_rank_prediction
)

submission_06_path = (
    OUTPUT_DIR
    + r"\submission_06_weighted_rank.csv"
)

submission_06.to_csv(
    submission_06_path,
    index=False
)


# ------------------------------------------------------------
# 30C. Hybrid
# ------------------------------------------------------------

submission_07 = sample.copy()

submission_07[target_col] = (
    test_hybrid_prediction
)

submission_07_path = (
    OUTPUT_DIR
    + r"\submission_07_rank_probability_hybrid.csv"
)

submission_07.to_csv(
    submission_07_path,
    index=False
)


# ============================================================
# 31. SAVE SEARCH RESULTS
# ============================================================

rank_results_path = (
    OUTPUT_DIR
    + r"\ensemble_03_rank_results.csv"
)

hybrid_results_path = (
    OUTPUT_DIR
    + r"\ensemble_03_hybrid_results.csv"
)

overall_summary_path = (
    OUTPUT_DIR
    + r"\ensemble_03_summary.csv"
)


rank_df.to_csv(
    rank_results_path,
    index=False
)

hybrid_df.to_csv(
    hybrid_results_path,
    index=False
)

method_df.to_csv(
    overall_summary_path,
    index=False
)


# ============================================================
# 32. SAVE OOF PREDICTIONS
# ============================================================

oof[
    "probability_ensemble"
] = probability_ensemble

oof[
    "equal_rank_ensemble"
] = equal_rank_prediction

oof[
    "weighted_rank_ensemble"
] = best_rank_prediction

oof[
    "hybrid_ensemble"
] = best_hybrid_prediction


oof_output_path = (
    OUTPUT_DIR
    + r"\ensemble_03_oof_predictions.csv"
)

oof.to_csv(
    oof_output_path,
    index=False
)


# ============================================================
# 33. SAVE COMPREHENSIVE SUMMARY
# ============================================================

summary = pd.DataFrame([{

    "lightgbm_oof_auc":
        lgbm_auc,

    "catboost_oof_auc":
        catboost_auc,

    "xgboost_oof_auc":
        xgboost_auc,

    "previous_probability_ensemble_auc":
        probability_ensemble_auc,

    "equal_rank_auc":
        equal_rank_auc,

    "best_rank_auc":
        best_rank_auc,

    "best_rank_lgbm_weight":
        best_rank_lgb,

    "best_rank_catboost_weight":
        best_rank_cat,

    "best_rank_xgboost_weight":
        best_rank_xgb,

    "best_hybrid_auc":
        best_hybrid_auc,

    "best_hybrid_probability_weight":
        best_alpha,

    "best_hybrid_rank_weight":
        1.0 - best_alpha,

    "overall_best_method":
        overall_best_method,

    "overall_best_oof_auc":
        overall_best_auc,

    "improvement_vs_lgbm":
        overall_best_auc
        - lgbm_auc,

    "improvement_vs_previous_probability":
        overall_best_auc
        - THREE_MODEL_PROB_OOF_AUC,

}])


summary_path = (
    OUTPUT_DIR
    + r"\ensemble_03_summary_full.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 34. FINAL REPORT
# ============================================================

elapsed = (
    time.time()
    - start_time
)


print("\n" + "=" * 80)
print("ENSEMBLE 03 — FINAL RESULTS")
print("=" * 80)

print(
    f"LightGBM OOF: "
    f"{lgbm_auc:.6f}"
)

print(
    f"Previous 3-model Probability: "
    f"{probability_ensemble_auc:.6f}"
)

print(
    f"Equal Rank: "
    f"{equal_rank_auc:.6f}"
)

print(
    f"Best Weighted Rank: "
    f"{best_rank_auc:.6f}"
)

print(
    f"Best Hybrid: "
    f"{best_hybrid_auc:.6f}"
)


print("\n" + "-" * 80)

print(
    f"Overall OOF winner: "
    f"{overall_best_method}"
)

print(
    f"Overall best OOF AUC: "
    f"{overall_best_auc:.6f}"
)

print(
    f"Improvement vs LightGBM: "
    f"{overall_best_auc - lgbm_auc:+.6f}"
)

print(
    f"Difference vs previous "
    f"0.959225 ensemble: "
    f"{overall_best_auc - THREE_MODEL_PROB_OOF_AUC:+.6f}"
)


print("\n" + "=" * 80)
print("BEST WEIGHTED RANK WEIGHTS")
print("=" * 80)

print(
    f"LightGBM: "
    f"{best_rank_lgb:.3f}"
)

print(
    f"CatBoost: "
    f"{best_rank_cat:.3f}"
)

print(
    f"XGBoost: "
    f"{best_rank_xgb:.3f}"
)


print("\n" + "=" * 80)
print("BEST HYBRID WEIGHTS")
print("=" * 80)

print(
    f"Probability: "
    f"{best_alpha:.2f}"
)

print(
    f"Rank: "
    f"{1.0 - best_alpha:.2f}"
)


print("\n" + "=" * 80)
print("SUBMISSIONS CREATED")
print("=" * 80)

print(
    f"1. {submission_05_path}"
)

print(
    f"2. {submission_06_path}"
)

print(
    f"3. {submission_07_path}"
)


print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(
    "1. ensemble_03_rank_results.csv"
)

print(
    "2. ensemble_03_hybrid_results.csv"
)

print(
    "3. ensemble_03_summary.csv"
)

print(
    "4. ensemble_03_summary_full.csv"
)

print(
    "5. ensemble_03_oof_predictions.csv"
)

print(
    "6. ensemble_03_rank_test_predictions.csv"
)

print(
    "7. submission_05_equal_rank.csv"
)

print(
    "8. submission_06_weighted_rank.csv"
)

print(
    "9. submission_07_rank_probability_hybrid.csv"
)


print("\n" + "=" * 80)
print("ENSEMBLE 03 FINISHED")
print("=" * 80)

print(
    f"Elapsed time: "
    f"{elapsed / 60:.2f} minutes"
)