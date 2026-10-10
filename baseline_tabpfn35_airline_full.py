# -*- coding: utf-8 -*-
"""
Kaggle Playground S6E10 - Airline Satisfaction
TabPFN-3.5 + leakage-safe feature engineering + OOF/test predictions

HOW TO USE
1) Set MODE = "smoke" first to verify environment, model download, and feature pipeline.
2) After smoke run succeeds, set MODE = "full" and run again for 5-fold OOF + test predictions.
3) Never submit the smoke output. Only the full run creates a submission CSV.

Recommended environment: Python 3.10-3.12, recent PyTorch with CUDA, latest tabpfn.
"""

from __future__ import annotations

import gc
import time
import warnings
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tabpfn import TabPFNClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*categorical.*")


# ============================================================================
# 0. USER CONFIGURATION
# ============================================================================

DATA_DIR = Path(r"D:\kaggle\airline\data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

OUTPUT_DIR = Path(r"D:\kaggle\airline\tabpfn35_experiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# First run "smoke" to check the whole pipeline; then change to "full".
MODE = "smoke"  # "smoke" or "full"

# Full mode: 5-fold OOF; smoke mode overrides these to smaller settings.
FULL_N_SPLITS = 5
FULL_INNER_TE_SPLITS = 5
FULL_N_ESTIMATORS = 4
FULL_SUBSAMPLE_ROWS = 50_000
FULL_PREDICT_BATCH_SIZE = 2_048

# Small diagnostic run; it does NOT create a submission file.
SMOKE_TRAIN_ROWS = 30_000
SMOKE_TEST_ROWS = 2_000
SMOKE_N_SPLITS = 2
SMOKE_INNER_TE_SPLITS = 3
SMOKE_N_ESTIMATORS = 2
SMOKE_SUBSAMPLE_ROWS = 10_000
SMOKE_PREDICT_BATCH_SIZE = 1_024

RANDOM_STATE = 42
TE_SMOOTHING = 50.0
MISSING_TOKEN = "__MISSING__"

TARGET_COL = "satisfaction"
ID_COL = "id"

# Original airline-service ratings.
RATING_COLS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]

# These categorical combinations get leakage-safe target encodings.
TE_COLS = [
    "traveler_segment",
    "gender_travel_class",
    "class_travel_agegroup",
    "customer_travel",
    "customer_class",
]


# ============================================================================
# 1. ENVIRONMENT + DATA CHECKS
# ============================================================================

def print_environment() -> None:
    print("=" * 100)
    print("TABPFN-3.5 AIRLINE SATISFACTION")
    print("=" * 100)
    print(f"Python/PyTorch version : {torch.__version__}")
    print(f"CUDA available         : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU                    : {props.name}")
        print(f"GPU memory             : {props.total_memory / (1024 ** 3):.2f} GB")
        print(f"CUDA runtime (PyTorch)  : {torch.version.cuda}")
    else:
        raise RuntimeError(
            "PyTorch 当前没有识别到 CUDA GPU。此脚本按大样本 GPU 运行设计，"
            "不建议在 CPU 上直接跑完整五折。\n"
            "请先检查当前 PyCharm .venv 中的 torch.cuda.is_available()，"
            "必要时按 PyTorch 官网安装带 CUDA 的 PyTorch。"
        )

    print(f"Run mode               : {MODE}")
    print(f"Train path             : {TRAIN_PATH}")
    print(f"Test path              : {TEST_PATH}")
    print(f"Output directory       : {OUTPUT_DIR}")


def encode_target(y_raw: pd.Series) -> np.ndarray:
    """Map satisfaction labels to binary 0/1."""
    if pd.api.types.is_numeric_dtype(y_raw):
        values = pd.to_numeric(y_raw, errors="raise").to_numpy()
        unique = set(np.unique(values).tolist())

        if unique.issubset({0, 1}):
            return values.astype(np.int8)

        raise ValueError(
            f"Target is numeric but not binary 0/1: {sorted(unique)[:10]}"
        )

    normalized = y_raw.astype("string").str.strip().str.lower()

    mapping = {
        "satisfied": 1,
        "1": 1,
        "true": 1,
        "yes": 1,
        "neutral or dissatisfied": 0,
        "neutral": 0,
        "dissatisfied": 0,
        "0": 0,
        "false": 0,
        "no": 0,
    }

    y = normalized.map(mapping)

    if y.isna().any():
        unknown = (
            normalized[y.isna()]
            .value_counts(dropna=False)
            .head(10)
            .to_dict()
        )
        raise ValueError(
            f"Cannot map target labels to 0/1. Unrecognized labels: {unknown}"
        )

    return y.to_numpy(dtype=np.int8)


def load_data() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Cannot find train data: {TRAIN_PATH}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Cannot find test data: {TEST_PATH}")

    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    test = pd.read_csv(TEST_PATH, low_memory=False)

    if TARGET_COL not in train.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found in train.csv")

    if ID_COL not in train.columns or ID_COL not in test.columns:
        raise KeyError(f"Expected '{ID_COL}' column in both train.csv and test.csv")

    y = pd.Series(encode_target(train[TARGET_COL]), name=TARGET_COL)
    train_ids = train[ID_COL].copy()
    test_ids = test[ID_COL].copy()

    # Exclude ID and target. Never use ID as a predictor.
    feature_cols = [
        c for c in train.columns
        if c not in {ID_COL, TARGET_COL}
    ]

    missing_in_test = [c for c in feature_cols if c not in test.columns]

    if missing_in_test:
        raise ValueError(f"Features in train but not test: {missing_in_test}")

    X_train_raw = train[feature_cols].copy()
    X_test_raw = test[feature_cols].copy()

    print(f"Loaded train            : {train.shape}")
    print(f"Loaded test             : {test.shape}")
    print(f"Target positive rate    : {y.mean():.6f}")
    print(f"Original predictor count: {len(feature_cols)}")

    # Smoke mode uses stratified subsets only to validate the pipeline.
    if MODE.lower() == "smoke":
        n = min(SMOKE_TRAIN_ROWS, len(X_train_raw))
        idx_all = np.arange(len(X_train_raw))

        idx_keep, _ = train_test_split(
            idx_all,
            train_size=n,
            random_state=RANDOM_STATE,
            stratify=y.to_numpy(),
        )

        idx_keep = np.sort(idx_keep)

        X_train_raw = (
            X_train_raw.iloc[idx_keep]
            .reset_index(drop=True)
        )

        train_ids = (
            train_ids.iloc[idx_keep]
            .reset_index(drop=True)
        )

        y = pd.Series(
            y.iloc[idx_keep].to_numpy(),
            name=TARGET_COL,
        )

        n_test = min(SMOKE_TEST_ROWS, len(X_test_raw))
        rng = np.random.default_rng(RANDOM_STATE)

        test_keep = np.sort(
            rng.choice(len(X_test_raw), size=n_test, replace=False)
        )

        X_test_raw = (
            X_test_raw.iloc[test_keep]
            .reset_index(drop=True)
        )

        test_ids = (
            test_ids.iloc[test_keep]
            .reset_index(drop=True)
        )

        print(f"SMOKE subset train      : {len(X_train_raw):,}")
        print(f"SMOKE subset test       : {len(X_test_raw):,}")

    return (
        X_train_raw,
        X_test_raw,
        y,
        train_ids.reset_index(drop=True),
        test_ids.reset_index(drop=True),
    )


# ============================================================================
# 2. UNSUPERVISED FEATURE ENGINEERING
# ============================================================================

def combine_categories(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> pd.Series:
    cols = [c for c in columns if c in frame.columns]

    if not cols:
        return pd.Series(
            MISSING_TOKEN,
            index=frame.index,
            dtype="object",
        )

    parts = frame[cols].copy()

    for c in cols:
        parts[c] = (
            parts[c]
            .astype("string")
            .fillna(MISSING_TOKEN)
        )

    return parts.agg("|".join, axis=1).astype("object")


def safe_numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            np.nan,
            index=frame.index,
            dtype="float32",
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    ).astype("float32")


def engineer_features(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build features on train+test without using y.
    Returns aligned feature rows.
    """
    print("\n[Feature Engineering] Building raw + engineered features...")

    combined = pd.concat(
        [train_raw, test_raw],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    X = combined.copy()
    del combined
    gc.collect()

    # Ensure core numeric fields are numeric.
    numeric_candidates = [
        "Age",
        "Flight Distance",
        "Departure Delay in Minutes",
        "Arrival Delay in Minutes",
        *RATING_COLS,
    ]

    for col in numeric_candidates:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # ------------------------------------------------------------------------
    # 2.1 Passenger context and categorical interactions
    # ------------------------------------------------------------------------
    age = safe_numeric(X, "Age")
    distance = safe_numeric(X, "Flight Distance")
    dep = safe_numeric(X, "Departure Delay in Minutes")
    arr = safe_numeric(X, "Arrival Delay in Minutes")

    age_group = pd.cut(
        age,
        bins=[0, 12, 18, 25, 35, 45, 55, 65, 120],
        labels=[
            "0_12",
            "13_18",
            "19_25",
            "26_35",
            "36_45",
            "46_55",
            "56_65",
            "66_plus",
        ],
        include_lowest=True,
    )

    distance_group = pd.cut(
        distance,
        bins=[-1, 100, 250, 500, 1000, 1500, 2500, 4000, 10000],
        labels=[
            "0_100",
            "101_250",
            "251_500",
            "501_1000",
            "1001_1500",
            "1501_2500",
            "2501_4000",
            "4001_plus",
        ],
        include_lowest=True,
    )

    X["age_group"] = (
        age_group.astype("string")
        .fillna(MISSING_TOKEN)
    )

    X["distance_group"] = (
        distance_group.astype("string")
        .fillna(MISSING_TOKEN)
    )

    # Delay severity bands.
    delay_bins = [-0.1, 0, 5, 15, 30, 60, 120, np.inf]
    delay_labels = [
        "zero",
        "1_5",
        "6_15",
        "16_30",
        "31_60",
        "61_120",
        "over_120",
    ]

    dep_band = pd.cut(
        dep,
        bins=delay_bins,
        labels=delay_labels,
        include_lowest=True,
    )

    arr_band = pd.cut(
        arr,
        bins=delay_bins,
        labels=delay_labels,
        include_lowest=True,
    )

    X["departure_delay_band"] = (
        dep_band.astype("string")
        .fillna(MISSING_TOKEN)
    )

    X["arrival_delay_band"] = (
        arr_band.astype("string")
        .fillna(MISSING_TOKEN)
    )

    # Passenger segment interactions.
    X["traveler_segment"] = combine_categories(
        X,
        ["Customer Type", "Type of Travel", "Class"],
    )

    X["gender_travel_class"] = combine_categories(
        X,
        ["Gender", "Type of Travel", "Class"],
    )

    X["customer_travel"] = combine_categories(
        X,
        ["Customer Type", "Type of Travel"],
    )

    X["customer_class"] = combine_categories(
        X,
        ["Customer Type", "Class"],
    )

    X["class_travel_agegroup"] = combine_categories(
        X,
        ["Class", "Type of Travel", "age_group"],
    )

    X["travel_class_booking"] = combine_categories(
        X,
        ["Type of Travel", "Class", "Ease of Online booking"],
    )

    X["travel_class_boarding"] = combine_categories(
        X,
        ["Type of Travel", "Class", "Online boarding"],
    )

    X["agegroup_customer_travel"] = combine_categories(
        X,
        ["age_group", "Customer Type", "Type of Travel"],
    )

    # ------------------------------------------------------------------------
    # 2.2 Delay structure and missingness
    # ------------------------------------------------------------------------
    if "Departure Delay in Minutes" in X.columns:
        X["departure_delay_missing"] = dep.isna().astype("int8")

        X["departure_delay_log1p"] = (
            np.log1p(dep.clip(lower=0))
            .astype("float32")
        )

        X["departure_delay_positive"] = (
            dep.fillna(0) > 0
        ).astype("int8")

        X["departure_delay_over_15"] = (
            dep.fillna(0) > 15
        ).astype("int8")

    if "Arrival Delay in Minutes" in X.columns:
        X["arrival_delay_missing"] = arr.isna().astype("int8")

        X["arrival_delay_log1p"] = (
            np.log1p(arr.clip(lower=0))
            .astype("float32")
        )

        X["arrival_delay_positive"] = (
            arr.fillna(0) > 0
        ).astype("int8")

        X["arrival_delay_over_15"] = (
            arr.fillna(0) > 15
        ).astype("int8")

    X["delay_sum"] = (
        dep.fillna(0) + arr.fillna(0)
    ).astype("float32")

    X["delay_max"] = (
        pd.concat([dep, arr], axis=1)
        .max(axis=1)
        .astype("float32")
    )

    X["delay_difference"] = (
        dep - arr
    ).astype("float32")

    X["any_delay_over_15"] = (
        (dep.fillna(0) > 15)
        | (arr.fillna(0) > 15)
    ).astype("int8")

    X["both_delays_zero"] = (
        (dep.fillna(0) == 0)
        & (arr.fillna(0) == 0)
    ).astype("int8")

    X["flight_distance_log1p"] = (
        np.log1p(distance.clip(lower=0))
        .astype("float32")
    )

    X["age_x_log_distance"] = (
        age * np.log1p(distance.clip(lower=0))
    ).astype("float32")

    # ------------------------------------------------------------------------
    # 2.3 Aggregate service-rating profile
    # ------------------------------------------------------------------------
    rating_cols = [
        c for c in RATING_COLS
        if c in X.columns
    ]

    if rating_cols:
        rating_matrix = X[rating_cols].apply(
            pd.to_numeric,
            errors="coerce",
        )

        X["rating_mean"] = (
            rating_matrix.mean(axis=1)
            .astype("float32")
        )

        X["rating_std"] = (
            rating_matrix.std(axis=1)
            .astype("float32")
        )

        X["rating_min"] = (
            rating_matrix.min(axis=1)
            .astype("float32")
        )

        X["rating_max"] = (
            rating_matrix.max(axis=1)
            .astype("float32")
        )

        X["rating_sum"] = (
            rating_matrix.sum(axis=1, min_count=1)
            .astype("float32")
        )

        X["rating_nonmissing_count"] = (
            rating_matrix.notna()
            .sum(axis=1)
            .astype("int8")
        )

        X["rating_zero_count"] = (
            (rating_matrix == 0)
            .sum(axis=1)
            .astype("int8")
        )

        X["rating_low_count_le_2"] = (
            (rating_matrix <= 2)
            .sum(axis=1)
            .astype("int8")
        )

        X["rating_high_count_ge_4"] = (
            (rating_matrix >= 4)
            .sum(axis=1)
            .astype("int8")
        )

        X["rating_five_count"] = (
            (rating_matrix == 5)
            .sum(axis=1)
            .astype("int8")
        )

    # Selected rating differences.
    rating_pairs = [
        (
            "Inflight wifi service",
            "Online boarding",
            "wifi_boarding_gap",
        ),
        (
            "Ease of Online booking",
            "Online boarding",
            "booking_boarding_gap",
        ),
        (
            "Seat comfort",
            "Cleanliness",
            "seat_cleanliness_gap",
        ),
        (
            "Inflight entertainment",
            "Seat comfort",
            "entertainment_seat_gap",
        ),
        (
            "Inflight service",
            "On-board service",
            "inflight_onboard_gap",
        ),
        (
            "Food and drink",
            "Cleanliness",
            "food_cleanliness_gap",
        ),
        (
            "Departure/Arrival time convenient",
            "Ease of Online booking",
            "convenience_booking_gap",
        ),
    ]

    for left, right, new_col in rating_pairs:
        if left in X.columns and right in X.columns:
            X[new_col] = (
                safe_numeric(X, left)
                - safe_numeric(X, right)
            ).astype("float32")

    # ------------------------------------------------------------------------
    # 2.4 Unsupervised frequency features
    # ------------------------------------------------------------------------
    # Using train+test counts is label-free.
    frequency_cols = [
        "traveler_segment",
        "gender_travel_class",
        "class_travel_agegroup",
        "customer_travel",
        "customer_class",
        "Class",
        "Type of Travel",
        "Customer Type",
    ]

    for col in frequency_cols:
        if col not in X.columns:
            continue

        keys = (
            X[col]
            .astype("string")
            .fillna(MISSING_TOKEN)
        )

        counts = keys.value_counts(dropna=False)

        freq = (
            keys.map(counts)
            .fillna(0)
            .astype("float32")
        )

        safe_name = (
            col.replace(" ", "_")
            .replace("/", "_")
        )

        X[f"freq__{safe_name}"] = freq

        X[f"logfreq__{safe_name}"] = (
            np.log1p(freq)
            .astype("float32")
        )

    # ------------------------------------------------------------------------
    # 2.5 Convert categorical and numeric columns into model-ready dtypes
    # ------------------------------------------------------------------------
    categorical_cols = list(
        X.select_dtypes(
            include=["object", "string", "category"]
        ).columns
    )

    for col in categorical_cols:
        X[col] = (
            X[col]
            .astype("string")
            .fillna(MISSING_TOKEN)
            .astype("category")
        )

    for col in X.columns:
        if col not in categorical_cols:
            X[col] = pd.to_numeric(
                X[col],
                errors="coerce",
            ).astype("float32")

    print(f"Engineered feature count: {X.shape[1]}")
    print(f"Categorical columns     : {len(categorical_cols)}")
    print(f"Numeric columns         : {X.shape[1] - len(categorical_cols)}")

    return X


# ============================================================================
# 3. LEAKAGE-SAFE TARGET ENCODING
# ============================================================================

def normalize_keys(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .fillna(MISSING_TOKEN)
        .astype("object")
    )


def fit_smoothed_target_map(
    keys: pd.Series,
    y: np.ndarray,
    smoothing: float = TE_SMOOTHING,
) -> Tuple[pd.Series, float]:
    key_values = normalize_keys(keys).to_numpy()
    y_values = np.asarray(y, dtype=np.float64)

    global_mean = float(y_values.mean())

    temp = pd.DataFrame({
        "_key": key_values,
        "_y": y_values,
    })

    agg = (
        temp.groupby(
            "_key",
            sort=False,
            observed=True,
        )["_y"]
        .agg(["sum", "count"])
    )

    mapping = (
        agg["sum"] + smoothing * global_mean
    ) / (
        agg["count"] + smoothing
    )

    return mapping, global_mean


def apply_target_map(
    keys: pd.Series,
    mapping: pd.Series,
    global_mean: float,
) -> np.ndarray:
    key_values = normalize_keys(keys)

    encoded = (
        key_values.map(mapping)
        .fillna(global_mean)
    )

    return encoded.to_numpy(dtype=np.float32)


def add_leakage_safe_target_encodings(
    X_outer_train: pd.DataFrame,
    y_outer_train: np.ndarray,
    X_outer_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    inner_splits: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Outer-training rows receive inner-OOF target encodings.

    Outer validation/test encodings use only labels from outer training.
    A row's own label is not used to encode that same training row.
    """
    Xtr = X_outer_train.copy()
    Xva = X_outer_valid.copy()
    Xte = X_test.copy()

    y = np.asarray(y_outer_train, dtype=np.int8)

    inner_cv = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=seed,
    )

    inner_split_list = list(
        inner_cv.split(np.zeros(len(y)), y)
    )

    for col in TE_COLS:
        if col not in Xtr.columns:
            continue

        print(f"    [TE] {col}")

        train_encoded = np.zeros(
            len(Xtr),
            dtype=np.float32,
        )

        for inner_train_idx, inner_valid_idx in inner_split_list:
            mapping, global_mean = fit_smoothed_target_map(
                Xtr[col].iloc[inner_train_idx],
                y[inner_train_idx],
                TE_SMOOTHING,
            )

            train_encoded[inner_valid_idx] = apply_target_map(
                Xtr[col].iloc[inner_valid_idx],
                mapping,
                global_mean,
            )

        # Validation/test use the entire outer-training fold's labels.
        full_mapping, full_mean = fit_smoothed_target_map(
            Xtr[col],
            y,
            TE_SMOOTHING,
        )

        Xtr[f"TE__{col}"] = train_encoded

        Xva[f"TE__{col}"] = apply_target_map(
            Xva[col],
            full_mapping,
            full_mean,
        )

        Xte[f"TE__{col}"] = apply_target_map(
            Xte[col],
            full_mapping,
            full_mean,
        )

    # Align validation and test columns exactly with the training columns.
    Xva = Xva.reindex(columns=Xtr.columns)
    Xte = Xte.reindex(columns=Xtr.columns)

    return Xtr, Xva, Xte


# ============================================================================
# 4. TABPFN FIT + BATCHED PREDICTION
# ============================================================================

def predict_positive_probability(
    model: TabPFNClassifier,
    X: pd.DataFrame,
    batch_size: int,
) -> np.ndarray:
    chunks = []

    n = len(X)
    positive_class_index = list(model.classes_).index(1)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = X.iloc[start:end]

        p = model.predict_proba(batch)[:, positive_class_index]

        chunks.append(
            np.asarray(p, dtype=np.float32)
        )

        if end == n or (start // batch_size + 1) % 25 == 0:
            print(
                f"        predicted {end:,}/{n:,} rows",
                flush=True,
            )

    if not chunks:
        return np.empty(0, dtype=np.float32)

    return np.concatenate(chunks)


def build_model(
    n_estimators: int,
    subsample_rows: int,
    seed: int,
) -> TabPFNClassifier:
    """
    Build TabPFN-3.5 classifier.

    The number of rows used by each estimator is restricted to help control
    memory consumption. Actual GPU requirements still depend on the model,
    preprocessing, data shape and installed TabPFN version.
    """
    return TabPFNClassifier(
        n_estimators=n_estimators,
        device="cuda",
        random_state=seed,
        fit_mode="fit_preprocessors",
        memory_saving_mode="auto",
        keep_cache_on_device=False,
        kv_cache_precision="int8",
        inference_config={
            "SUBSAMPLE_SAMPLES": int(subsample_rows),
            "SAMPLE_SUBSAMPLING_METHOD": "stratified",
        },
        show_progress_bar=True,
    )


# ============================================================================
# 5. CROSS-VALIDATION, OOF, TEST PREDICTIONS AND SUBMISSION
# ============================================================================

def main() -> None:
    print_environment()

    if MODE.lower() not in {"smoke", "full"}:
        raise ValueError("MODE must be either 'smoke' or 'full'.")

    (
        X_train_raw,
        X_test_raw,
        y_series,
        train_ids,
        test_ids,
    ) = load_data()

    y = y_series.to_numpy(dtype=np.int8)

    train_ids = train_ids.reset_index(drop=True)
    test_ids = test_ids.reset_index(drop=True)

    # Engineer features without using target labels.
    X_all = engineer_features(
        X_train_raw,
        X_test_raw,
    )

    n_train = len(X_train_raw)

    X_base_train = (
        X_all.iloc[:n_train]
        .reset_index(drop=True)
    )

    X_base_test = (
        X_all.iloc[n_train:]
        .reset_index(drop=True)
    )

    del X_all, X_train_raw, X_test_raw
    gc.collect()

    # Mode-specific settings.
    if MODE.lower() == "full":
        n_splits = FULL_N_SPLITS
        inner_te_splits = FULL_INNER_TE_SPLITS
        n_estimators = FULL_N_ESTIMATORS
        subsample_rows = FULL_SUBSAMPLE_ROWS
        batch_size = FULL_PREDICT_BATCH_SIZE
        prefix = "tabpfn35"
    else:
        n_splits = SMOKE_N_SPLITS
        inner_te_splits = SMOKE_INNER_TE_SPLITS
        n_estimators = SMOKE_N_ESTIMATORS
        subsample_rows = SMOKE_SUBSAMPLE_ROWS
        batch_size = SMOKE_PREDICT_BATCH_SIZE
        prefix = "tabpfn35_smoke"

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof = np.full(
        len(y),
        np.nan,
        dtype=np.float32,
    )

    fold_assignment = np.full(
        len(y),
        -1,
        dtype=np.int16,
    )

    fold_scores = []
    test_fold_predictions = []

    print("\n" + "=" * 100)
    print("START CROSS-VALIDATION")
    print(f"Rows train/test         : {len(y):,} / {len(X_base_test):,}")
    print(f"Outer folds             : {n_splits}")
    print(f"Inner TE folds          : {inner_te_splits}")
    print(f"TabPFN n_estimators     : {n_estimators}")
    print(f"Rows sampled/estimator  : {subsample_rows:,}")
    print(f"Prediction batch size   : {batch_size:,}")
    print("=" * 100)

    for fold, (tr_idx, va_idx) in enumerate(
        cv.split(np.zeros(len(y)), y),
        start=1,
    ):
        fold_start = time.time()

        print("\n" + "#" * 100)
        print(
            f"FOLD {fold}/{n_splits} | "
            f"train={len(tr_idx):,} valid={len(va_idx):,}"
        )
        print("#" * 100)

        X_tr_base = (
            X_base_train.iloc[tr_idx]
            .reset_index(drop=True)
        )

        X_va_base = (
            X_base_train.iloc[va_idx]
            .reset_index(drop=True)
        )

        X_te_base = X_base_test.reset_index(drop=True)

        y_tr = y[tr_idx]
        y_va = y[va_idx]

        # Leakage-safe target encoding within this outer fold.
        X_tr, X_va, X_te = add_leakage_safe_target_encodings(
            X_outer_train=X_tr_base,
            y_outer_train=y_tr,
            X_outer_valid=X_va_base,
            X_test=X_te_base,
            inner_splits=inner_te_splits,
            seed=RANDOM_STATE + fold,
        )

        del X_tr_base, X_va_base, X_te_base
        gc.collect()

        current_subsample = min(
            subsample_rows,
            len(X_tr),
        )

        model = build_model(
            n_estimators=n_estimators,
            subsample_rows=current_subsample,
            seed=RANDOM_STATE + 100 * fold,
        )

        print("    Fitting TabPFN...")
        model.fit(X_tr, y_tr)

        print("    Predicting validation fold...")

        p_valid = predict_positive_probability(
            model,
            X_va,
            batch_size,
        )

        fold_auc = roc_auc_score(
            y_va,
            p_valid,
        )

        oof[va_idx] = p_valid
        fold_assignment[va_idx] = fold

        fold_scores.append({
            "fold": fold,
            "auc": float(fold_auc),
            "n_valid": len(va_idx),
        })

        print(f"    Fold {fold} ROC-AUC: {fold_auc:.8f}")

        print("    Predicting test set for this fold...")

        p_test = predict_positive_probability(
            model,
            X_te,
            batch_size,
        )

        test_fold_predictions.append(p_test)

        fold_elapsed = time.time() - fold_start

        print(
            f"    Fold elapsed: {fold_elapsed / 60:.1f} minutes"
        )

        del model, X_tr, X_va, X_te, p_valid, p_test
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # 5.1 Validate OOF completeness and average test predictions
    # ------------------------------------------------------------------------
    if np.isnan(oof).any():
        missing = int(np.isnan(oof).sum())
        raise RuntimeError(
            f"OOF predictions incomplete: {missing:,} missing rows"
        )

    if len(test_fold_predictions) != n_splits:
        raise RuntimeError(
            "Not all folds produced test predictions."
        )

    oof_auc = roc_auc_score(y, oof)

    test_pred = np.mean(
        np.vstack(test_fold_predictions),
        axis=0,
    ).astype(np.float32)

    # ------------------------------------------------------------------------
    # 5.2 Save files
    # ------------------------------------------------------------------------
    oof_df = pd.DataFrame({
        ID_COL: train_ids.to_numpy(),
        TARGET_COL: y,
        "pred_tabpfn35": oof,
        "fold": fold_assignment,
    })

    test_pred_df = pd.DataFrame({
        ID_COL: test_ids.to_numpy(),
        "pred_tabpfn35": test_pred,
    })

    fold_df = pd.DataFrame(fold_scores)

    oof_path = OUTPUT_DIR / f"{prefix}_oof.csv"
    test_pred_path = OUTPUT_DIR / f"{prefix}_test_predictions.csv"
    fold_path = OUTPUT_DIR / f"{prefix}_fold_metrics.csv"

    oof_df.to_csv(oof_path, index=False)
    test_pred_df.to_csv(test_pred_path, index=False)
    fold_df.to_csv(fold_path, index=False)

    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)

    print(f"OOF ROC-AUC             : {oof_auc:.8f}")
    print(
        "Fold AUCs               : "
        f"{[round(x['auc'], 8) for x in fold_scores]}"
    )

    print(f"OOF saved               : {oof_path}")
    print(f"Test predictions saved  : {test_pred_path}")
    print(f"Fold metrics saved      : {fold_path}")

    # ------------------------------------------------------------------------
    # 5.3 Generate a submission only in full mode
    # ------------------------------------------------------------------------
    if MODE.lower() == "full":
        submission = pd.DataFrame({
            ID_COL: test_ids.to_numpy(),
            TARGET_COL: test_pred,
        })

        if SAMPLE_SUB_PATH.exists():
            sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

            if (
                ID_COL in sample_sub.columns
                and TARGET_COL in sample_sub.columns
            ):
                # Preserve sample-submission ID order.
                submission = sample_sub[[ID_COL]].merge(
                    submission,
                    on=ID_COL,
                    how="left",
                    validate="one_to_one",
                )

                submission = submission[
                    [
                        c
                        for c in sample_sub.columns
                        if c in submission.columns
                    ]
                ]

        if submission[TARGET_COL].isna().any():
            raise RuntimeError(
                "Submission contains missing predictions after ID alignment."
            )

        if len(submission) != len(test_ids):
            raise RuntimeError(
                "Submission row count does not match test set."
            )

        sub_path = OUTPUT_DIR / "tabpfn35_submission.csv"

        submission.to_csv(
            sub_path,
            index=False,
        )

        print(f"Submission saved        : {sub_path}")
        print(
            "NOTE: This is a TabPFN-only submission file, "
            "not the blended champion."
        )

    else:
        print(
            "SMOKE MODE: no submission file created. "
            "Change MODE='full' for complete OOF/test predictions."
        )

    # ------------------------------------------------------------------------
    # 5.4 Basic sanity checks
    # ------------------------------------------------------------------------
    print("\nPrediction sanity checks:")

    print(
        f"  OOF min/max            : "
        f"{float(oof.min()):.6f} / {float(oof.max()):.6f}"
    )

    print(
        f"  Test min/max           : "
        f"{float(test_pred.min()):.6f} / {float(test_pred.max()):.6f}"
    )

    print(f"  Test prediction rows   : {len(test_pred):,}")
    print("Done.")


if __name__ == "__main__":
    main()