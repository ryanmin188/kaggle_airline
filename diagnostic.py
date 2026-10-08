import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SAMPLE_PATH = (
    r"D:\kaggle\airline\data\sample_submission.csv"
)

TEST_PATH = (
    r"D:\kaggle\airline\data\test.csv"
)

OUTPUT_DIR = (
    r"D:\kaggle\airline\diagnostic_submissions"
)


LGBM_PATH = (
    r"D:\kaggle\airline\lgbm_test_predictions.csv"
)

CATBOOST_PATH = (
    r"D:\kaggle\airline\catboost_test_predictions.csv"
)

XGBOOST_PATH = (
    r"D:\kaggle\airline\xgboost_test_predictions.csv"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("LOADING TEST PREDICTIONS")
print("=" * 80)

sample = pd.read_csv(
    SAMPLE_PATH
)

test = pd.read_csv(
    TEST_PATH
)

lgbm = pd.read_csv(
    LGBM_PATH
)

catboost = pd.read_csv(
    CATBOOST_PATH
)

xgboost = pd.read_csv(
    XGBOOST_PATH
)


print(
    f"Sample:   {sample.shape}"
)

print(
    f"Test:     {test.shape}"
)

print(
    f"LightGBM: {lgbm.shape}"
)

print(
    f"CatBoost: {catboost.shape}"
)

print(
    f"XGBoost:  {xgboost.shape}"
)


# ============================================================
# VALIDATION
# ============================================================

required = {
    "LightGBM": (
        lgbm,
        "lgbm_prediction"
    ),

    "CatBoost": (
        catboost,
        "catboost_prediction"
    ),

    "XGBoost": (
        xgboost,
        "xgboost_prediction"
    ),
}


for model_name, (
    df,
    prediction_col
) in required.items():

    if "id" not in df.columns:
        raise ValueError(
            f"{model_name}: missing id column"
        )

    if prediction_col not in df.columns:
        raise ValueError(
            f"{model_name}: missing "
            f"{prediction_col}"
        )

    if len(df) != len(test):
        raise ValueError(
            f"{model_name}: row count does not "
            f"match test.csv"
        )


# ============================================================
# ALIGN ALL PREDICTIONS BY ID
# ============================================================

pred = sample[["id"]].copy()


pred = pred.merge(
    lgbm[
        [
            "id",
            "lgbm_prediction"
        ]
    ],
    on="id",
    how="left",
    validate="one_to_one"
)


pred = pred.merge(
    catboost[
        [
            "id",
            "catboost_prediction"
        ]
    ],
    on="id",
    how="left",
    validate="one_to_one"
)


pred = pred.merge(
    xgboost[
        [
            "id",
            "xgboost_prediction"
        ]
    ],
    on="id",
    how="left",
    validate="one_to_one"
)


# ============================================================
# CHECK
# ============================================================

for col in [
    "lgbm_prediction",
    "catboost_prediction",
    "xgboost_prediction"
]:

    if pred[col].isna().any():

        raise ValueError(
            f"Missing predictions in {col}"
        )


print("\n" + "=" * 80)
print("PREDICTION ALIGNMENT PASSED")
print("=" * 80)


# ============================================================
# BUILD CANDIDATE PREDICTIONS
# ============================================================

# ------------------------------------------------------------
# A. LightGBM only
# ------------------------------------------------------------

pred["lightgbm_only"] = (
    pred["lgbm_prediction"]
)


# ------------------------------------------------------------
# B. LightGBM + CatBoost
#
# Previous best OOF blend:
# 75% LightGBM
# 25% CatBoost
# ------------------------------------------------------------

pred["lgbm_catboost"] = (

    0.75
    * pred["lgbm_prediction"]

    +

    0.25
    * pred["catboost_prediction"]
)


# ------------------------------------------------------------
# C. XGBoost only
# ------------------------------------------------------------

pred["xgboost_only"] = (
    pred["xgboost_prediction"]
)


# ------------------------------------------------------------
# D. Current best three-model ensemble
#
# OOF optimized:
# 53.6% LightGBM
# 11.0% CatBoost
# 35.4% XGBoost
# ------------------------------------------------------------

pred["three_model"] = (

    0.536
    * pred["lgbm_prediction"]

    +

    0.110
    * pred["catboost_prediction"]

    +

    0.354
    * pred["xgboost_prediction"]
)


# ============================================================
# SUBMISSION CREATION FUNCTION
# ============================================================

def create_submission(
    prediction_series,
    filename,
):

    submission = sample.copy()

    target_columns = [
        col
        for col in sample.columns
        if col != "id"
    ]

    if len(target_columns) != 1:

        raise ValueError(
            "sample_submission.csv must contain "
            "exactly one target column."
        )

    target_col = (
        target_columns[0]
    )

    submission[target_col] = (
        prediction_series.values
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if len(submission) != len(test):

        raise ValueError(
            "Submission row count mismatch."
        )

    if submission["id"].duplicated().any():

        raise ValueError(
            "Duplicate IDs in submission."
        )

    if submission[target_col].isna().any():

        raise ValueError(
            "NaN predictions in submission."
        )

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    submission.to_csv(
        path,
        index=False
    )

    return path


# ============================================================
# GENERATE FOUR SUBMISSIONS
# ============================================================

print("\n" + "=" * 80)
print("GENERATING DIAGNOSTIC SUBMISSIONS")
print("=" * 80)


files = {}


files["LightGBM"] = create_submission(
    pred["lightgbm_only"],
    "submission_01_lightgbm.csv"
)


files["LGBM + CatBoost"] = create_submission(
    pred["lgbm_catboost"],
    "submission_02_lgbm_catboost.csv"
)


files["XGBoost"] = create_submission(
    pred["xgboost_only"],
    "submission_03_xgboost.csv"
)


files["Three-model Ensemble"] = create_submission(
    pred["three_model"],
    "submission_04_three_model.csv"
)


# ============================================================
# PRINT OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("DIAGNOSTIC SUBMISSIONS CREATED")
print("=" * 80)

for name, path in files.items():

    print(
        f"{name}:"
        f"\n  {path}"
    )


# ============================================================
# PREDICTION STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("PREDICTION STATISTICS")
print("=" * 80)

for name, column in [

    (
        "LightGBM",
        "lightgbm_only"
    ),

    (
        "LGBM + CatBoost",
        "lgbm_catboost"
    ),

    (
        "XGBoost",
        "xgboost_only"
    ),

    (
        "Three-model",
        "three_model"
    ),
]:

    print(
        f"\n{name}"
    )

    print(
        f"  Mean: "
        f"{pred[column].mean():.6f}"
    )

    print(
        f"  Min:  "
        f"{pred[column].min():.6f}"
    )

    print(
        f"  Max:  "
        f"{pred[column].max():.6f}"
    )


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(
    f"All files saved in:\n"
    f"{OUTPUT_DIR}"
)