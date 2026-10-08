import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================
# PATHS
# ============================================================
BASE_DIR = r"D:\PycharmProjects\KAGGLE_airline"
DATA_ROOT = r"D:\kaggle\airline"

DATA_DIR = os.path.join(DATA_ROOT, "data")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

# IMPORTANT:
# These experiment folders are under D:\kaggle\airline,
# NOT under D:\kaggle\airline\data
LGBM_EXP_DIR = os.path.join(
    DATA_ROOT,
    "conditional_te_experiment"
)

REALMLP_EXP_DIR = os.path.join(
    DATA_ROOT,
    "realmlp_experiment"
)

LGBM_OOF_PATH = os.path.join(
    LGBM_EXP_DIR,
    "conditional_te_lgbm_oof.csv"
)

LGBM_TEST_PATH = os.path.join(
    LGBM_EXP_DIR,
    "conditional_te_lgbm_test.csv"
)

REALMLP_OOF_PATH = os.path.join(
    REALMLP_EXP_DIR,
    "realmlp_enriched_oof.csv"
)

REALMLP_TEST_PATH = os.path.join(
    REALMLP_EXP_DIR,
    "realmlp_enriched_test.csv"
)

OUT_DIR = os.path.join(
    DATA_ROOT,
    "lgbm_realmlp_blend"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================
def find_prediction_column(df, name="prediction"):
    """
    Automatically find the prediction/probability column.
    """
    candidates = [
        "prediction",
        "pred",
        "probability",
        "prob",
        "satisfaction",
        "target"
    ]

    # First try exact names
    for c in candidates:
        if c in df.columns:
            return c

    # Otherwise find numeric columns excluding obvious ID/fold/target columns
    exclude = {
        "id", "fold", "target", "y", "satisfaction"
    }

    numeric_cols = [
        c
        for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(
        f"Cannot uniquely identify prediction column in {name}.\n"
        f"Columns: {list(df.columns)}"
    )


def load_prediction_file(path, name):
    """
    Load a prediction CSV and return id + prediction.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    df = pd.read_csv(path)

    if "id" not in df.columns:
        raise ValueError(
            f"{name} does not contain 'id'. "
            f"Columns = {list(df.columns)}"
        )

    pred_col = find_prediction_column(df, name)

    out = df[["id", pred_col]].copy()
    out.columns = ["id", "prediction"]

    out["id"] = pd.to_numeric(
        out["id"],
        errors="raise"
    )

    out["prediction"] = pd.to_numeric(
        out["prediction"],
        errors="raise"
    )

    if out["prediction"].isna().any():
        raise ValueError(
            f"{name}: prediction contains NaN."
        )

    return out


# ============================================================
# LOAD DATA
# ============================================================
print("=" * 90)
print("LGBM + REALMLP OOF BLEND SEARCH")
print("=" * 90)

train = pd.read_csv(TRAIN_PATH)

if "id" not in train.columns:
    raise ValueError(
        "train.csv must contain 'id'."
    )

if "satisfaction" not in train.columns:
    raise ValueError(
        "train.csv must contain 'satisfaction'."
    )

train["id"] = pd.to_numeric(
    train["id"],
    errors="raise"
)

y_df = train[
    ["id", "satisfaction"]
].copy()

y_df["satisfaction"] = (
    y_df["satisfaction"].astype(int)
)

lgbm_oof = load_prediction_file(
    LGBM_OOF_PATH,
    "LGBM OOF"
)

realmlp_oof = load_prediction_file(
    REALMLP_OOF_PATH,
    "RealMLP OOF"
)

print("\nLoaded:")
print("LGBM OOF   :", lgbm_oof.shape)
print("RealMLP OOF:", realmlp_oof.shape)
print("Train      :", train.shape)


# ============================================================
# ALIGN OOF BY ID
# ============================================================
oof = (
    y_df
    .merge(
        lgbm_oof,
        on="id",
        how="inner"
    )
    .rename(
        columns={"prediction": "lgbm"}
    )
    .merge(
        realmlp_oof,
        on="id",
        how="inner"
    )
    .rename(
        columns={"prediction": "realmlp"}
    )
)

if len(oof) != len(train):
    raise ValueError(
        f"OOF alignment failed: "
        f"train={len(train)}, merged={len(oof)}"
    )

oof = (
    oof
    .sort_values("id")
    .reset_index(drop=True)
)

y = oof["satisfaction"].to_numpy()

p_lgbm = oof["lgbm"].to_numpy()

p_realmlp = oof["realmlp"].to_numpy()

print(
    "\nOOF aligned rows:",
    len(oof)
)


# ============================================================
# BASELINE AUC
# ============================================================
auc_lgbm = roc_auc_score(
    y,
    p_lgbm
)

auc_realmlp = roc_auc_score(
    y,
    p_realmlp
)

corr = np.corrcoef(
    p_lgbm,
    p_realmlp
)[0, 1]

print("\n" + "=" * 90)
print("BASELINE")
print("=" * 90)

print(
    f"LGBM OOF AUC    = {auc_lgbm:.9f}"
)

print(
    f"RealMLP OOF AUC = {auc_realmlp:.9f}"
)

print(
    f"Prediction corr = {corr:.9f}"
)


# ============================================================
# WEIGHT SEARCH
#
# blend = w * LGBM + (1-w) * RealMLP
# ============================================================
print("\n" + "=" * 90)
print("SEARCHING BLEND WEIGHTS")
print("=" * 90)

results = []

# ------------------------------------------------------------
# Coarse search: every 1%
# ------------------------------------------------------------
coarse_weights = np.arange(
    0.00,
    1.0001,
    0.01
)

for w in coarse_weights:

    p = (
        w * p_lgbm
        + (1.0 - w) * p_realmlp
    )

    auc = roc_auc_score(
        y,
        p
    )

    results.append({
        "weight_lgbm": w,
        "weight_realmlp": 1.0 - w,
        "auc": auc
    })

coarse_df = pd.DataFrame(
    results
)

best_coarse = coarse_df.loc[
    coarse_df["auc"].idxmax()
]

best_w = float(
    best_coarse["weight_lgbm"]
)

# ------------------------------------------------------------
# Fine search: every 0.1% around coarse optimum
# ------------------------------------------------------------
fine_low = max(
    0.0,
    best_w - 0.02
)

fine_high = min(
    1.0,
    best_w + 0.02
)

fine_weights = np.arange(
    fine_low,
    fine_high + 0.00001,
    0.001
)

fine_results = []

for w in fine_weights:

    p = (
        w * p_lgbm
        + (1.0 - w) * p_realmlp
    )

    auc = roc_auc_score(
        y,
        p
    )

    fine_results.append({
        "weight_lgbm": w,
        "weight_realmlp": 1.0 - w,
        "auc": auc
    })

fine_df = pd.DataFrame(
    fine_results
)

best = fine_df.loc[
    fine_df["auc"].idxmax()
]

best_w = float(
    best["weight_lgbm"]
)

best_wr = float(
    best["weight_realmlp"]
)

best_auc = float(
    best["auc"]
)

print(
    f"\nBest LGBM weight    = {best_w:.3f}"
)

print(
    f"Best RealMLP weight = {best_wr:.3f}"
)

print(
    f"Best blend OOF AUC  = {best_auc:.9f}"
)

print(
    f"Gain vs LGBM        = "
    f"{best_auc - auc_lgbm:+.9f}"
)

print(
    f"Gain vs RealMLP     = "
    f"{best_auc - auc_realmlp:+.9f}"
)


# ============================================================
# TOP 10 WEIGHTS
# ============================================================
print("\n" + "=" * 90)
print("TOP 10 WEIGHTS")
print("=" * 90)

top10 = (
    fine_df
    .sort_values(
        "auc",
        ascending=False
    )
    .head(10)
    .reset_index(drop=True)
)

print(
    top10.to_string(
        index=False,
        formatters={
            "weight_lgbm": "{:.3f}".format,
            "weight_realmlp": "{:.3f}".format,
            "auc": "{:.9f}".format,
        }
    )
)


# ============================================================
# SAVE WEIGHT SEARCH
# ============================================================
search_path = os.path.join(
    OUT_DIR,
    "blend_weight_search.csv"
)

fine_df.to_csv(
    search_path,
    index=False
)


# ============================================================
# SAVE OOF BLENDED PREDICTIONS
# ============================================================
oof["blend_prediction"] = (
    best_w * oof["lgbm"]
    + best_wr * oof["realmlp"]
)

oof_path = os.path.join(
    OUT_DIR,
    "lgbm_realmlp_blend_oof.csv"
)

oof.to_csv(
    oof_path,
    index=False
)


# ============================================================
# LOAD TEST PREDICTIONS
# ============================================================
print("\n" + "=" * 90)
print("BUILDING TEST PREDICTION")
print("=" * 90)

lgbm_test = load_prediction_file(
    LGBM_TEST_PATH,
    "LGBM TEST"
)

realmlp_test = load_prediction_file(
    REALMLP_TEST_PATH,
    "RealMLP TEST"
)

test = pd.read_csv(
    TEST_PATH
)

test["id"] = pd.to_numeric(
    test["id"],
    errors="raise"
)

test_pred = (
    test[["id"]]
    .merge(
        lgbm_test,
        on="id",
        how="left"
    )
    .rename(
        columns={"prediction": "lgbm"}
    )
    .merge(
        realmlp_test,
        on="id",
        how="left"
    )
    .rename(
        columns={"prediction": "realmlp"}
    )
)

if test_pred["lgbm"].isna().any():
    raise ValueError(
        "Some test IDs are missing "
        "LGBM predictions."
    )

if test_pred["realmlp"].isna().any():
    raise ValueError(
        "Some test IDs are missing "
        "RealMLP predictions."
    )

test_pred["blend_prediction"] = (
    best_w * test_pred["lgbm"]
    + best_wr * test_pred["realmlp"]
)

print(
    "Test rows:",
    len(test_pred)
)

print(
    f"Blend min : "
    f"{test_pred['blend_prediction'].min():.6f}"
)

print(
    f"Blend max : "
    f"{test_pred['blend_prediction'].max():.6f}"
)

print(
    f"Blend mean: "
    f"{test_pred['blend_prediction'].mean():.6f}"
)

print(
    f"Blend std : "
    f"{test_pred['blend_prediction'].std():.6f}"
)


# ============================================================
# CREATE KAGGLE SUBMISSION
# ============================================================
sample = pd.read_csv(
    SAMPLE_PATH
)

if "id" not in sample.columns:
    raise ValueError(
        "sample_submission.csv must contain id."
    )

sample["id"] = pd.to_numeric(
    sample["id"],
    errors="raise"
)

submission = (
    sample[["id"]]
    .merge(
        test_pred[
            ["id", "blend_prediction"]
        ],
        on="id",
        how="left"
    )
)

if submission["blend_prediction"].isna().any():
    raise ValueError(
        "Some submission IDs have no prediction."
    )

# Detect target column in sample_submission
target_candidates = [
    c
    for c in sample.columns
    if c != "id"
]

if len(target_candidates) != 1:
    raise ValueError(
        "Could not uniquely identify "
        "submission target column. "
        f"Found: {target_candidates}"
    )

target_col = target_candidates[0]

submission[target_col] = (
    submission["blend_prediction"]
)

submission = submission[
    ["id", target_col]
]


# ============================================================
# SAVE SUBMISSION
# ============================================================
submission_path = os.path.join(
    OUT_DIR,
    "submission_lgbm_realmlp_blend.csv"
)

submission.to_csv(
    submission_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 90)
print("FINAL RESULT")
print("=" * 90)

print(
    f"LGBM OOF AUC       : "
    f"{auc_lgbm:.9f}"
)

print(
    f"RealMLP OOF AUC    : "
    f"{auc_realmlp:.9f}"
)

print(
    f"Prediction corr    : "
    f"{corr:.9f}"
)

print(
    f"Best LGBM weight   : "
    f"{best_w:.3f}"
)

print(
    f"Best RealMLP weight: "
    f"{best_wr:.3f}"
)

print(
    f"Blend OOF AUC      : "
    f"{best_auc:.9f}"
)

print(
    f"Blend improvement  : "
    f"{best_auc - auc_lgbm:+.9f}"
)

print("\nSaved:")
print(search_path)
print(oof_path)
print(submission_path)

print("\nDone.")