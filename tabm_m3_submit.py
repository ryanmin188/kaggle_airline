from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


# ============================================================
# 1. Paths
# ============================================================

ROOT = Path(r"D:\kaggle\airline")
DATA_DIR = ROOT / "data"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"

# Conditional-TE LGBM predictions
LGBM_TEST_PATH = (
    ROOT
    / "conditional_te_experiment"
    / "conditional_te_lgbm_test.csv"
)

# Lean RealMLP V1 predictions
REALMLP_TEST_PATH = (
    ROOT
    / "lean_realmlp_blend"
    / "lean_realmlp_test.csv"
)

# Multi-seed TabM predictions
TABM_TEST_PATH = (
    ROOT
    / "tabm_m3_multiseed"
    / "tabm_m3_multiseed_test_predictions.csv"
)

# Output
OUTPUT_DIR = ROOT / "tabm_m3_multiseed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBMISSION_PATH = (
    OUTPUT_DIR / "tabm_m3_multiseed_submission.csv"
)

REPORT_PATH = (
    OUTPUT_DIR / "tabm_m3_submit_report.txt"
)


# ============================================================
# 2. Configuration
# ============================================================

ID_COL = "id"
TARGET = "satisfaction"

# Original champion weights
LGBM_WEIGHT = 0.40
REALMLP_WEIGHT = 0.60

# Final blend weights
CHAMPION_WEIGHT = 0.80
TABM_WEIGHT = 0.20

# Do not upload automatically
AUTO_SUBMIT = False


# ============================================================
# 3. Utilities
# ============================================================

def check_file(path: Path | str) -> Path:
    """Check whether a required file exists."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"\nRequired file does not exist:\n{path}"
        )

    return path


def normalized_rank(values: np.ndarray) -> np.ndarray:
    """
    Convert predictions to normalized ranks.

    This matches the rank normalization used in the
    multi-seed experiment:
        rankdata(values, method='average') / len(values)
    """
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError(
            f"Expected a 1D prediction array, got shape {values.shape}"
        )

    if len(values) == 0:
        raise ValueError("Prediction array is empty.")

    if not np.isfinite(values).all():
        raise ValueError(
            "Predictions contain NaN or infinite values."
        )

    return (
        rankdata(values, method="average")
        / len(values)
    )


def load_test_predictions(
    path: Path | str,
    expected_ids: np.ndarray,
    model_name: str,
) -> tuple[np.ndarray, str]:
    """
    Load test predictions and align them to test.csv IDs.

    Requirements:
        - Must contain an ID column.
        - Must contain a numeric prediction column.
        - IDs must match test.csv exactly.
        - Duplicate IDs are not allowed.
    """
    path = check_file(path)
    df = pd.read_csv(path)

    print(f"\nLoading {model_name}")
    print(f"File: {path}")

    if ID_COL not in df.columns:
        raise ValueError(
            f"{model_name}: missing ID column '{ID_COL}'.\n"
            f"Columns found: {df.columns.tolist()}"
        )

    df[ID_COL] = pd.to_numeric(
        df[ID_COL],
        errors="raise",
    )

    if df[ID_COL].duplicated().any():
        raise ValueError(
            f"{model_name}: duplicate IDs found in {path}"
        )

    if len(df) != len(expected_ids):
        raise ValueError(
            f"{model_name}: row count mismatch.\n"
            f"Prediction file: {len(df)}\n"
            f"Expected: {len(expected_ids)}"
        )

    if set(df[ID_COL]) != set(expected_ids):
        raise ValueError(
            f"{model_name}: IDs do not match test.csv."
        )

    # Identify the prediction column.
    preferred_columns = [
        "prediction",
        "lgbm_prediction",
        "realmlp_prediction",
        "probability",
        "prob",
        "pred",
    ]

    prediction_col = None

    for col in preferred_columns:
        if col in df.columns and col != ID_COL:
            prediction_col = col
            break

    if prediction_col is None:
        numeric_candidates = [
            col
            for col in df.columns
            if col != ID_COL
            and pd.api.types.is_numeric_dtype(df[col])
        ]

        if len(numeric_candidates) == 1:
            prediction_col = numeric_candidates[0]
        else:
            raise ValueError(
                f"{model_name}: cannot uniquely identify "
                "the prediction column.\n"
                f"Columns: {df.columns.tolist()}\n"
                f"Numeric candidates: {numeric_candidates}"
            )

    # Align predictions to test.csv ID order.
    df = df.set_index(ID_COL).loc[expected_ids]

    predictions = pd.to_numeric(
        df[prediction_col],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if not np.isfinite(predictions).all():
        raise ValueError(
            f"{model_name}: predictions contain NaN or inf."
        )

    print(f"Prediction column: {prediction_col}")
    print(f"Rows: {len(predictions):,}")
    print(
        "Prediction range: "
        f"{predictions.min():.8f} ~ "
        f"{predictions.max():.8f}"
    )

    return predictions, prediction_col


# ============================================================
# 4. Main
# ============================================================

def main():

    print("=" * 80)
    print("TABM M3 MULTI-SEED FINAL SUBMISSION")
    print("=" * 80)

    # --------------------------------------------------------
    # Step 1. Check required files
    # --------------------------------------------------------

    for path in [
        TRAIN_PATH,
        TEST_PATH,
        SAMPLE_PATH,
        LGBM_TEST_PATH,
        REALMLP_TEST_PATH,
        TABM_TEST_PATH,
    ]:
        check_file(path)

    print("\nFixed prediction paths:")
    print("Conditional-TE LGBM:", LGBM_TEST_PATH)
    print("Lean RealMLP V1    :", REALMLP_TEST_PATH)
    print("Multi-seed TabM M3 :", TABM_TEST_PATH)

    # --------------------------------------------------------
    # Step 2. Load train/test/sample files
    # --------------------------------------------------------

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)

    if ID_COL not in train.columns:
        raise ValueError(
            f"train.csv is missing '{ID_COL}'."
        )

    if TARGET not in train.columns:
        raise ValueError(
            f"train.csv is missing '{TARGET}'."
        )

    if ID_COL not in test.columns:
        raise ValueError(
            f"test.csv is missing '{ID_COL}'."
        )

    if ID_COL not in sample.columns:
        raise ValueError(
            f"sample_submission.csv is missing '{ID_COL}'."
        )

    if TARGET not in sample.columns:
        raise ValueError(
            f"sample_submission.csv is missing '{TARGET}'."
        )

    train_ids = pd.to_numeric(
        train[ID_COL],
        errors="raise",
    ).to_numpy()

    test_ids = pd.to_numeric(
        test[ID_COL],
        errors="raise",
    ).to_numpy()

    sample_ids = pd.to_numeric(
        sample[ID_COL],
        errors="raise",
    ).to_numpy()

    y = train[TARGET].to_numpy()

    print("\nDataset information:")
    print("Train shape:", train.shape)
    print("Test shape :", test.shape)
    print("Sample shape:", sample.shape)

    if len(np.unique(test_ids)) != len(test_ids):
        raise ValueError(
            "Duplicate IDs found in test.csv."
        )

    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError(
            "Duplicate IDs found in sample_submission.csv."
        )

    if set(sample_ids) != set(test_ids):
        raise ValueError(
            "IDs in sample_submission.csv do not match test.csv."
        )

    # --------------------------------------------------------
    # Step 3. Load predictions from the three fixed models
    # --------------------------------------------------------

    lgb_pred, lgb_col = load_test_predictions(
        LGBM_TEST_PATH,
        test_ids,
        "Conditional-TE LGBM",
    )

    mlp_pred, mlp_col = load_test_predictions(
        REALMLP_TEST_PATH,
        test_ids,
        "Lean RealMLP V1",
    )

    tabm_pred, tabm_col = load_test_predictions(
        TABM_TEST_PATH,
        test_ids,
        "Multi-seed TabM M3",
    )

    # --------------------------------------------------------
    # Step 4. Rank normalization
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("RANK NORMALIZATION")
    print("=" * 80)

    lgb_rank = normalized_rank(lgb_pred)
    mlp_rank = normalized_rank(mlp_pred)
    tabm_rank = normalized_rank(tabm_pred)

    # --------------------------------------------------------
    # Step 5. Construct the original champion
    # --------------------------------------------------------

    champion_rank = (
        LGBM_WEIGHT * lgb_rank
        + REALMLP_WEIGHT * mlp_rank
    )

    # --------------------------------------------------------
    # Step 6. Blend champion with multi-seed TabM
    # --------------------------------------------------------

    final_predictions = (
        CHAMPION_WEIGHT * champion_rank
        + TABM_WEIGHT * tabm_rank
    )

    if not np.isfinite(final_predictions).all():
        raise ValueError(
            "Final predictions contain NaN or inf."
        )

    # --------------------------------------------------------
    # Step 7. Build submission
    # --------------------------------------------------------

    submission = pd.DataFrame({
        ID_COL: test_ids,
        TARGET: final_predictions,
    })

    # Match the exact ID order in sample_submission.csv.
    submission = (
        submission.set_index(ID_COL)
        .loc[sample_ids]
        .reset_index()
    )

    if submission.isna().any().any():
        raise ValueError(
            "Submission contains missing values."
        )

    if len(submission) != len(sample):
        raise ValueError(
            "Submission row count differs from sample submission."
        )

    if not np.array_equal(
        submission[ID_COL].to_numpy(),
        sample_ids,
    ):
        raise ValueError(
            "Submission IDs are not aligned with sample submission."
        )

    submission.to_csv(
        SUBMISSION_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Step 8. Save report
    # --------------------------------------------------------

    effective_lgb_weight = (
        CHAMPION_WEIGHT * LGBM_WEIGHT
    )

    effective_mlp_weight = (
        CHAMPION_WEIGHT * REALMLP_WEIGHT
    )

    report = [
        "TABM M3 MULTI-SEED SUBMISSION REPORT",
        "=" * 70,
        "",
        f"Train path: {TRAIN_PATH}",
        f"Test path: {TEST_PATH}",
        f"Sample path: {SAMPLE_PATH}",
        "",
        "Prediction files:",
        f"Conditional-TE LGBM: {LGBM_TEST_PATH}",
        f"Lean RealMLP V1: {REALMLP_TEST_PATH}",
        f"Multi-seed TabM M3: {TABM_TEST_PATH}",
        "",
        "Prediction columns:",
        f"Conditional-TE LGBM: {lgb_col}",
        f"Lean RealMLP V1: {mlp_col}",
        f"Multi-seed TabM M3: {tabm_col}",
        "",
        "Effective final weights:",
        f"Conditional-TE LGBM: {effective_lgb_weight:.4f}",
        f"Lean RealMLP V1: {effective_mlp_weight:.4f}",
        f"Multi-seed TabM M3: {TABM_WEIGHT:.4f}",
        "",
        f"Submission path: {SUBMISSION_PATH}",
        f"Rows: {len(submission)}",
        f"Prediction minimum: {submission[TARGET].min():.10f}",
        f"Prediction maximum: {submission[TARGET].max():.10f}",
        f"Prediction mean: {submission[TARGET].mean():.10f}",
        "",
        f"AUTO_SUBMIT: {AUTO_SUBMIT}",
    ]

    REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Step 9. Final output
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SUBMISSION CREATED SUCCESSFULLY")
    print("=" * 80)

    print("Submission:", SUBMISSION_PATH)
    print("Report    :", REPORT_PATH)

    print("\nEffective final weights:")
    print(f"Conditional-TE LGBM : {effective_lgb_weight:.0%}")
    print(f"Lean RealMLP V1     : {effective_mlp_weight:.0%}")
    print(f"Multi-seed TabM M3  : {TABM_WEIGHT:.0%}")

    print("\nPrediction range:")
    print(
        f"{submission[TARGET].min():.6f} ~ "
        f"{submission[TARGET].max():.6f}"
    )

    print("\nSubmission preview:")
    print(submission.head())

    print("\nValidation:")
    print(
        "Correct row count:",
        len(submission) == len(sample),
    )
    print(
        "Correct ID order:",
        np.array_equal(
            submission[ID_COL].to_numpy(),
            sample_ids,
        ),
    )
    print(
        "No missing values:",
        not submission.isna().any().any(),
    )
    print(
        "All predictions finite:",
        np.isfinite(
            submission[TARGET].to_numpy()
        ).all(),
    )

    print("\nAUTO_SUBMIT=False: no Kaggle upload was performed.")
    print("Review the three prediction paths before submitting.")


if __name__ == "__main__":
    main()
