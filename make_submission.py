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

PREDICTION_PATH = (
    r"D:\kaggle\airline\ensemble_02_best_test_predictions.csv"
)

OUTPUT_PATH = (
    r"D:\kaggle\airline\submission.csv"
)


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 80)
print("LOADING FILES")
print("=" * 80)

sample = pd.read_csv(
    SAMPLE_PATH
)

test = pd.read_csv(
    TEST_PATH
)

prediction = pd.read_csv(
    PREDICTION_PATH
)

print(
    f"Sample submission shape: {sample.shape}"
)

print(
    f"Test shape:              {test.shape}"
)

print(
    f"Prediction shape:        {prediction.shape}"
)


# ============================================================
# CHECK SAMPLE STRUCTURE
# ============================================================

print("\n" + "=" * 80)
print("SAMPLE SUBMISSION STRUCTURE")
print("=" * 80)

print(
    sample.head()
)

print(
    "\nColumns:"
)

for col in sample.columns:
    print(
        f" - {col}"
    )


# ============================================================
# CHECK PREDICTION STRUCTURE
# ============================================================

print("\n" + "=" * 80)
print("ENSEMBLE PREDICTION STRUCTURE")
print("=" * 80)

print(
    prediction.head()
)

print(
    "\nColumns:"
)

for col in prediction.columns:
    print(
        f" - {col}"
    )


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

if "id" not in sample.columns:
    raise ValueError(
        "sample_submission.csv does not contain 'id'."
    )

if "id" not in test.columns:
    raise ValueError(
        "test.csv does not contain 'id'."
    )

if "id" not in prediction.columns:
    raise ValueError(
        "ensemble prediction file does not contain 'id'."
    )

if "ensemble_prediction" not in prediction.columns:
    raise ValueError(
        "ensemble prediction file does not contain "
        "'ensemble_prediction'."
    )


# ============================================================
# CHECK ROW COUNTS
# ============================================================

if len(sample) != len(test):
    raise ValueError(
        "Sample submission and test row counts differ."
    )

if len(prediction) != len(test):
    raise ValueError(
        "Prediction and test row counts differ."
    )


# ============================================================
# CHECK ID ALIGNMENT
#
# We explicitly align by ID rather than assuming row order.
# ============================================================

test_ids = set(
    test["id"]
)

prediction_ids = set(
    prediction["id"]
)

sample_ids = set(
    sample["id"]
)


if test_ids != prediction_ids:

    missing = test_ids - prediction_ids

    extra = prediction_ids - test_ids

    raise ValueError(
        "\nPrediction IDs do not match test IDs."
        f"\nMissing IDs: {len(missing)}"
        f"\nExtra IDs:   {len(extra)}"
    )


if test_ids != sample_ids:

    missing = test_ids - sample_ids

    extra = sample_ids - test_ids

    raise ValueError(
        "\nSample submission IDs do not match test IDs."
        f"\nMissing IDs: {len(missing)}"
        f"\nExtra IDs:   {len(extra)}"
    )


# ============================================================
# ALIGN PREDICTIONS TO SAMPLE ORDER
# ============================================================

prediction_aligned = (
    sample[["id"]]
    .merge(
        prediction[
            [
                "id",
                "ensemble_prediction",
            ]
        ],
        on="id",
        how="left",
        validate="one_to_one",
    )
)


# ============================================================
# CHECK FOR MISSING PREDICTIONS
# ============================================================

missing_predictions = (
    prediction_aligned[
        "ensemble_prediction"
    ]
    .isna()
    .sum()
)

if missing_predictions > 0:

    raise ValueError(
        f"{missing_predictions} test IDs "
        "have no ensemble prediction."
    )


# ============================================================
# CREATE SUBMISSION
# ============================================================

submission = sample.copy()

target_columns = [
    col
    for col in sample.columns
    if col != "id"
]


# ------------------------------------------------------------
# We expect one target column.
# ------------------------------------------------------------

if len(target_columns) != 1:

    raise ValueError(
        "Expected sample_submission.csv to have "
        "exactly one target column besides 'id'.\n"
        f"Found: {target_columns}"
    )

target_col = target_columns[0]


print("\n" + "=" * 80)
print("CREATING SUBMISSION")
print("=" * 80)

print(
    f"Submission target column: "
    f"{target_col}"
)


# ============================================================
# USE PROBABILITY PREDICTIONS
#
# Since ROC-AUC is the evaluation metric, probabilities
# preserve ranking information and are preferable to
# hard 0/1 predictions.
# ============================================================

submission[target_col] = (
    prediction_aligned[
        "ensemble_prediction"
    ].values
)


# ============================================================
# FINAL VALIDATION
# ============================================================

if len(submission) != len(test):
    raise ValueError(
        "Final submission row count is incorrect."
    )

if submission["id"].duplicated().any():
    raise ValueError(
        "Duplicate IDs found in submission."
    )

if submission[target_col].isna().any():
    raise ValueError(
        "NaN values found in submission target."
    )


# ============================================================
# PREDICTION RANGE
# ============================================================

print(
    f"\nPrediction min: "
    f"{submission[target_col].min():.6f}"
)

print(
    f"Prediction max: "
    f"{submission[target_col].max():.6f}"
)

print(
    f"Prediction mean: "
    f"{submission[target_col].mean():.6f}"
)


# ============================================================
# SAVE
# ============================================================

submission.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL CHECK
# ============================================================

saved_submission = pd.read_csv(
    OUTPUT_PATH
)

print("\n" + "=" * 80)
print("SUBMISSION CREATED")
print("=" * 80)

print(
    f"Path:\n{OUTPUT_PATH}"
)

print(
    f"\nShape: "
    f"{saved_submission.shape}"
)

print(
    "\nFirst 10 rows:"
)

print(
    saved_submission.head(10).to_string(
        index=False
    )
)

print("\nColumns:")

for col in saved_submission.columns:
    print(
        f" - {col}"
    )

print("\nFile exists:")

print(
    os.path.exists(
        OUTPUT_PATH
    )
)

print(
    "\nREADY FOR KAGGLE SUBMISSION."
)