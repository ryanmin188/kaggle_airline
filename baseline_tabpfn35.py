# -*- coding: utf-8 -*-
"""
Kaggle Playground S6E10 - Airline Satisfaction
TabPFN-3.5 | Full 5-fold OOF | Feature Engineering | Safe Target Encoding

主要修正：
1. 显式分层抽样：每折真正只让 10,000 行进入 model.fit()
2. 默认预测批次 512，OOM 时自动减半重试
3. 使用省显存模式
4. 每完成一折就保存验证集和测试集预测
5. 支持断点续跑，并验证 checkpoint 的实验配置
6. 新的输出目录及文件名，不覆盖旧实验结果

数据目录：
D:\\kaggle\\airline\\data

输出目录：
D:\\kaggle\\airline\\tabpfn35_experiment
"""

from __future__ import annotations

import gc
import hashlib
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
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*categorical.*",
)


# ============================================================================
# 0. CONFIGURATION
# ============================================================================

DATA_DIR = Path(r"D:\kaggle\airline\data")

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

OUTPUT_DIR = Path(r"D:\kaggle\airline\tabpfn35_experiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 使用新的目录，绝不复用之前 50k / 4-estimator 的 checkpoint。
CHECKPOINT_DIR = (
    OUTPUT_DIR
    / "checkpoints_full_fit10k_est2_batch512_v2"
)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MODE = "full"  # "full" 或 "smoke"

RESUME_COMPLETED_FOLDS = True

# 修改变量工程时，请更新 FEATURE_VERSION。
FEATURE_VERSION = "baseline80_te5_fit10k_v2"

# -------------------- Full run --------------------

FULL_N_SPLITS = 5
FULL_INNER_TE_SPLITS = 5

# RTX 4060 Laptop 8GB：优先跑通，不先追求最大上下文。
FULL_N_ESTIMATORS = 2
FULL_FIT_ROWS = 10_000
FULL_PREDICT_BATCH_SIZE = 512

# OOM 自动回退到的最小预测批次。
MIN_PREDICT_BATCH_SIZE = 32

# -------------------- Smoke run --------------------

SMOKE_TRAIN_ROWS = 30_000
SMOKE_TEST_ROWS = 2_000
SMOKE_N_SPLITS = 2
SMOKE_INNER_TE_SPLITS = 3
SMOKE_N_ESTIMATORS = 2
SMOKE_FIT_ROWS = 10_000
SMOKE_PREDICT_BATCH_SIZE = 512

RANDOM_STATE = 42
TE_SMOOTHING = 50.0
MISSING_TOKEN = "__MISSING__"

TARGET_COL = "satisfaction"
ID_COL = "id"

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

# 使用内层交叉拟合目标编码的类别变量。
TE_COLS = [
    "traveler_segment",
    "gender_travel_class",
    "class_travel_agegroup",
    "customer_travel",
    "customer_class",
]


# ============================================================================
# 1. ENVIRONMENT
# ============================================================================

def print_environment() -> None:
    print("=" * 100)
    print("TABPFN-3.5 AIRLINE SATISFACTION")
    print("=" * 100)

    print(f"PyTorch version          : {torch.__version__}")
    print(f"CUDA available           : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch 没有识别到 CUDA GPU。"
            "请检查当前 PyCharm 虚拟环境中的 CUDA/PyTorch 配置。"
        )

    props = torch.cuda.get_device_properties(0)

    print(f"GPU                      : {props.name}")
    print(
        f"GPU memory               : "
        f"{props.total_memory / (1024 ** 3):.2f} GB"
    )
    print(f"CUDA runtime             : {torch.version.cuda}")
    print(f"Run mode                 : {MODE}")
    print(f"Feature version           : {FEATURE_VERSION}")
    print(f"Train path                : {TRAIN_PATH}")
    print(f"Test path                 : {TEST_PATH}")
    print(f"Output directory          : {OUTPUT_DIR}")
    print(f"Checkpoint directory      : {CHECKPOINT_DIR}")


# ============================================================================
# 2. DATA LOADING
# ============================================================================

def encode_target(y_raw: pd.Series) -> np.ndarray:
    """将满意度标签统一映射到 0/1。"""

    if pd.api.types.is_numeric_dtype(y_raw):
        values = pd.to_numeric(
            y_raw,
            errors="raise",
        ).to_numpy()

        unique = set(np.unique(values).tolist())

        if unique.issubset({0, 1}):
            return values.astype(np.int8)

        raise ValueError(
            f"Target is numeric but not binary 0/1: {sorted(unique)}"
        )

    normalized = (
        y_raw.astype("string")
        .str.strip()
        .str.lower()
    )

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
            f"Unrecognized target labels: {unknown}"
        )

    return y.to_numpy(dtype=np.int8)


def load_data():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find train data: {TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find test data: {TEST_PATH}"
        )

    train = pd.read_csv(
        TRAIN_PATH,
        low_memory=False,
    )

    test = pd.read_csv(
        TEST_PATH,
        low_memory=False,
    )

    if TARGET_COL not in train.columns:
        raise KeyError(
            f"Target column '{TARGET_COL}' not found in train.csv"
        )

    if ID_COL not in train.columns or ID_COL not in test.columns:
        raise KeyError(
            f"Expected '{ID_COL}' in both train.csv and test.csv"
        )

    y = pd.Series(
        encode_target(train[TARGET_COL]),
        name=TARGET_COL,
    )

    train_ids = train[ID_COL].copy()
    test_ids = test[ID_COL].copy()

    feature_cols = [
        c for c in train.columns
        if c not in {ID_COL, TARGET_COL}
    ]

    missing_in_test = [
        c for c in feature_cols
        if c not in test.columns
    ]

    if missing_in_test:
        raise ValueError(
            f"Features missing from test.csv: {missing_in_test}"
        )

    X_train_raw = train[feature_cols].copy()
    X_test_raw = test[feature_cols].copy()

    print(f"Loaded train              : {train.shape}")
    print(f"Loaded test               : {test.shape}")
    print(f"Target positive rate      : {y.mean():.6f}")
    print(f"Original predictor count  : {len(feature_cols)}")

    if MODE.lower() == "smoke":
        n_train = min(
            SMOKE_TRAIN_ROWS,
            len(X_train_raw),
        )

        indices = np.arange(len(X_train_raw))

        keep_idx, _ = train_test_split(
            indices,
            train_size=n_train,
            random_state=RANDOM_STATE,
            stratify=y.to_numpy(),
        )

        keep_idx = np.sort(keep_idx)

        X_train_raw = (
            X_train_raw.iloc[keep_idx]
            .reset_index(drop=True)
        )

        train_ids = (
            train_ids.iloc[keep_idx]
            .reset_index(drop=True)
        )

        y = pd.Series(
            y.iloc[keep_idx].to_numpy(),
            name=TARGET_COL,
        )

        n_test = min(
            SMOKE_TEST_ROWS,
            len(X_test_raw),
        )

        rng = np.random.default_rng(RANDOM_STATE)

        test_keep_idx = np.sort(
            rng.choice(
                len(X_test_raw),
                size=n_test,
                replace=False,
            )
        )

        X_test_raw = (
            X_test_raw.iloc[test_keep_idx]
            .reset_index(drop=True)
        )

        test_ids = (
            test_ids.iloc[test_keep_idx]
            .reset_index(drop=True)
        )

        print(f"SMOKE train rows          : {len(X_train_raw):,}")
        print(f"SMOKE test rows           : {len(X_test_raw):,}")

    return (
        X_train_raw,
        X_test_raw,
        y,
        train_ids.reset_index(drop=True),
        test_ids.reset_index(drop=True),
    )


# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

def combine_categories(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> pd.Series:

    cols = [
        c for c in columns
        if c in frame.columns
    ]

    if not cols:
        return pd.Series(
            MISSING_TOKEN,
            index=frame.index,
            dtype="object",
        )

    parts = frame[cols].copy()

    for col in cols:
        parts[col] = (
            parts[col]
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

    print(
        "\n[Feature Engineering] Building raw + engineered features..."
    )

    # 不依赖标签的特征可以在 train + test 上一起构造。
    combined = pd.concat(
        [train_raw, test_raw],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    X = combined.copy()

    del combined
    gc.collect()

    numeric_candidates = [
        "Age",
        "Flight Distance",
        "Departure Delay in Minutes",
        "Arrival Delay in Minutes",
        *RATING_COLS,
    ]

    for col in numeric_candidates:
        if col in X.columns:
            X[col] = pd.to_numeric(
                X[col],
                errors="coerce",
            )

    age = safe_numeric(X, "Age")
    distance = safe_numeric(X, "Flight Distance")
    dep = safe_numeric(X, "Departure Delay in Minutes")
    arr = safe_numeric(X, "Arrival Delay in Minutes")

    # ------------------------------------------------------------------------
    # 3.1 年龄、距离和延误分组
    # ------------------------------------------------------------------------

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
        bins=[
            -1, 100, 250, 500, 1000,
            1500, 2500, 4000, 10000,
        ],
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

    delay_bins = [
        -0.1, 0, 5, 15, 30, 60, 120, np.inf
    ]

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

    # ------------------------------------------------------------------------
    # 3.2 旅客类别交互
    # ------------------------------------------------------------------------

    X["traveler_segment"] = combine_categories(
        X,
        [
            "Customer Type",
            "Type of Travel",
            "Class",
        ],
    )

    X["gender_travel_class"] = combine_categories(
        X,
        [
            "Gender",
            "Type of Travel",
            "Class",
        ],
    )

    X["customer_travel"] = combine_categories(
        X,
        [
            "Customer Type",
            "Type of Travel",
        ],
    )

    X["customer_class"] = combine_categories(
        X,
        [
            "Customer Type",
            "Class",
        ],
    )

    X["class_travel_agegroup"] = combine_categories(
        X,
        [
            "Class",
            "Type of Travel",
            "age_group",
        ],
    )

    X["travel_class_booking"] = combine_categories(
        X,
        [
            "Type of Travel",
            "Class",
            "Ease of Online booking",
        ],
    )

    X["travel_class_boarding"] = combine_categories(
        X,
        [
            "Type of Travel",
            "Class",
            "Online boarding",
        ],
    )

    X["agegroup_customer_travel"] = combine_categories(
        X,
        [
            "age_group",
            "Customer Type",
            "Type of Travel",
        ],
    )

    # ------------------------------------------------------------------------
    # 3.3 延误与缺失特征
    # ------------------------------------------------------------------------

    if "Departure Delay in Minutes" in X.columns:
        X["departure_delay_missing"] = (
            dep.isna().astype("int8")
        )

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
        X["arrival_delay_missing"] = (
            arr.isna().astype("int8")
        )

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
    # 3.4 服务评分画像
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
            rating_matrix.sum(
                axis=1,
                min_count=1,
            ).astype("float32")
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
    # 3.5 无标签类别频数
    # ------------------------------------------------------------------------

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
    # 3.6 类别和数值的 dtype 统一
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

    print(f"Engineered feature count : {X.shape[1]}")
    print(f"Categorical columns      : {len(categorical_cols)}")
    print(
        f"Numeric columns          : "
        f"{X.shape[1] - len(categorical_cols)}"
    )

    return X


# ============================================================================
# 4. LEAKAGE-SAFE TARGET ENCODING
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
    外层训练折：
        内层 OOF 目标编码，防止每行使用自身标签构造其编码。

    外层验证集和测试集：
        只使用当前外层训练折的标签建立目标编码映射。
    """

    Xtr = X_outer_train.copy()
    Xva = X_outer_valid.copy()
    Xte = X_test.copy()

    y = np.asarray(
        y_outer_train,
        dtype=np.int8,
    )

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

    # 保证各数据集列顺序完全一致。
    Xva = Xva.reindex(columns=Xtr.columns)
    Xte = Xte.reindex(columns=Xtr.columns)

    return Xtr, Xva, Xte


# ============================================================================
# 5. EXPLICIT STRATIFIED SUBSAMPLING FOR TABPFN
# ============================================================================

def get_fit_indices(
    y: np.ndarray,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    """
    返回真正送入 TabPFN.fit() 的行索引。

    如果外层训练折大于 max_rows，则进行分层抽样。
    这一步明确发生在 model.fit() 之前，不依赖 TabPFN 内部隐式抽样。
    """

    n = len(y)

    if n <= max_rows:
        return np.arange(n, dtype=np.int64)

    indices = np.arange(n)

    fit_idx, _ = train_test_split(
        indices,
        train_size=max_rows,
        random_state=seed,
        stratify=y,
    )

    return np.sort(
        fit_idx.astype(np.int64)
    )


# ============================================================================
# 6. MODEL
# ============================================================================

def build_model(
    n_estimators: int,
    seed: int,
) -> TabPFNClassifier:

    return TabPFNClassifier(
        n_estimators=n_estimators,
        device="cuda",
        random_state=seed,

        # 适合多次预测且显存有限的情况。
        fit_mode="fit_preprocessors",

        # 强制开启省显存模式，而不是让模型自行判断。
        memory_saving_mode=True,

        # 不再依赖 TabPFN 内部抽样控制训练行数；
        # 实际训练样本由 get_fit_indices() 显式选定。
        inference_config={
            "SUBSAMPLE_SAMPLES": None,
        },

        show_progress_bar=True,
    )


def is_cuda_oom(exc: Exception) -> bool:
    message = (
        f"{type(exc).__name__}: {exc}"
    ).lower()

    return (
        "out of memory" in message
        or "cuda oom" in message
        or "cudamemory" in message
    )


def predict_positive_probability(
    model: TabPFNClassifier,
    X: pd.DataFrame,
    batch_size: int,
) -> np.ndarray:
    """
    分批预测。
    遇到 CUDA OOM 时自动把批次减半，并从当前批次重试。
    """

    n = len(X)

    if n == 0:
        return np.empty(
            0,
            dtype=np.float32,
        )

    classes = list(model.classes_)

    if 1 not in classes:
        raise RuntimeError(
            f"Positive class 1 not found in model.classes_: {classes}"
        )

    positive_class_index = classes.index(1)

    predictions = np.empty(
        n,
        dtype=np.float32,
    )

    current_batch = batch_size
    start = 0

    while start < n:
        end = min(
            start + current_batch,
            n,
        )

        batch = X.iloc[start:end]

        try:
            p = model.predict_proba(batch)

            predictions[start:end] = np.asarray(
                p[:, positive_class_index],
                dtype=np.float32,
            )

            start = end

            if (
                start == n
                or start % 10_000 < current_batch
            ):
                print(
                    f"        predicted {start:,}/{n:,} rows "
                    f"(batch={current_batch})",
                    flush=True,
                )

        except Exception as exc:
            if not is_cuda_oom(exc):
                raise

            del batch
            gc.collect()
            torch.cuda.empty_cache()

            if current_batch <= MIN_PREDICT_BATCH_SIZE:
                raise RuntimeError(
                    "即使将预测批次降低到 "
                    f"{MIN_PREDICT_BATCH_SIZE}，仍然发生 CUDA OOM。\n"
                    "请将 FULL_FIT_ROWS 降低到 5,000，"
                    "并使用一个新的 checkpoint 目录。"
                ) from exc

            new_batch = max(
                MIN_PREDICT_BATCH_SIZE,
                current_batch // 2,
            )

            print(
                f"\n    CUDA OOM detected. "
                f"Reducing prediction batch "
                f"{current_batch} -> {new_batch}, retrying...",
                flush=True,
            )

            current_batch = new_batch
            gc.collect()
            torch.cuda.empty_cache()

            # start 不改变，从当前批次重新预测。

    return predictions


# ============================================================================
# 7. CHECKPOINT SIGNATURE
# ============================================================================

def make_data_signature(
    train_ids: pd.Series,
    test_ids: pd.Series,
    y: np.ndarray,
) -> str:
    """
    使用行 ID 和 target 的哈希，避免数据发生变化时误用旧 checkpoint。
    """

    hasher = hashlib.sha256()

    hasher.update(
        pd.util.hash_pandas_object(
            train_ids,
            index=False,
        ).to_numpy().tobytes()
    )

    hasher.update(
        pd.util.hash_pandas_object(
            test_ids,
            index=False,
        ).to_numpy().tobytes()
    )

    hasher.update(
        np.asarray(
            y,
            dtype=np.int8,
        ).tobytes()
    )

    return hasher.hexdigest()[:20]


def make_config_tag(
    data_signature: str,
    n_splits: int,
    inner_te_splits: int,
    n_estimators: int,
    fit_rows_limit: int,
    batch_size: int,
) -> str:

    return (
        f"{FEATURE_VERSION}"
        f"|data={data_signature}"
        f"|outer={n_splits}"
        f"|inner={inner_te_splits}"
        f"|estimators={n_estimators}"
        f"|fit_rows={fit_rows_limit}"
        f"|batch={batch_size}"
        f"|seed={RANDOM_STATE}"
        f"|smooth={TE_SMOOTHING}"
    )


def save_checkpoint(
    checkpoint_path: Path,
    config_tag: str,
    valid_idx: np.ndarray,
    valid_pred: np.ndarray,
    test_pred: np.ndarray,
    fold_auc: float,
) -> None:

    # 先写临时文件，再替换正式文件，减少中途终止造成的损坏风险。
    temp_path = checkpoint_path.with_name(
        checkpoint_path.stem + ".tmp.npz"
    )

    np.savez_compressed(
        temp_path,
        config_tag=np.asarray(config_tag),
        valid_idx=valid_idx.astype(np.int32),
        valid_pred=valid_pred.astype(np.float32),
        test_pred=test_pred.astype(np.float32),
        fold_auc=np.asarray(
            fold_auc,
            dtype=np.float64,
        ),
    )

    temp_path.replace(checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    expected_config_tag: str,
    expected_valid_idx: np.ndarray,
    expected_test_rows: int,
):
    if not checkpoint_path.exists():
        return None

    try:
        with np.load(
            checkpoint_path,
            allow_pickle=False,
        ) as saved:

            saved_tag = str(
                saved["config_tag"].item()
            )

            saved_valid_idx = saved["valid_idx"].copy()
            saved_valid_pred = saved["valid_pred"].copy()
            saved_test_pred = saved["test_pred"].copy()
            saved_auc = float(saved["fold_auc"])

        valid = (
            saved_tag == expected_config_tag
            and np.array_equal(
                saved_valid_idx,
                expected_valid_idx,
            )
            and len(saved_valid_pred) == len(expected_valid_idx)
            and len(saved_test_pred) == expected_test_rows
            and np.isfinite(saved_valid_pred).all()
            and np.isfinite(saved_test_pred).all()
            and np.isfinite(saved_auc)
        )

        if not valid:
            print(
                f"    Checkpoint configuration mismatch: "
                f"{checkpoint_path.name}"
            )
            return None

        return (
            saved_valid_pred.astype(np.float32),
            saved_test_pred.astype(np.float32),
            saved_auc,
        )

    except Exception as exc:
        print(
            f"    Failed to load checkpoint "
            f"{checkpoint_path.name}: {exc}"
        )
        return None


# ============================================================================
# 8. MAIN
# ============================================================================

def main() -> None:
    print_environment()

    if MODE.lower() not in {"full", "smoke"}:
        raise ValueError(
            "MODE must be 'full' or 'smoke'."
        )

    (
        X_train_raw,
        X_test_raw,
        y_series,
        train_ids,
        test_ids,
    ) = load_data()

    y = y_series.to_numpy(
        dtype=np.int8
    )

    train_ids = train_ids.reset_index(drop=True)
    test_ids = test_ids.reset_index(drop=True)

    data_signature = make_data_signature(
        train_ids,
        test_ids,
        y,
    )

    print(f"Data signature            : {data_signature}")

    # 所有不使用 y 的工程特征在此构建。
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

    del X_all
    del X_train_raw
    del X_test_raw
    gc.collect()

    # ------------------------------------------------------------------------
    # 模式配置
    # ------------------------------------------------------------------------

    if MODE.lower() == "full":
        n_splits = FULL_N_SPLITS
        inner_te_splits = FULL_INNER_TE_SPLITS
        n_estimators = FULL_N_ESTIMATORS
        fit_rows_limit = FULL_FIT_ROWS
        batch_size = FULL_PREDICT_BATCH_SIZE
        prefix = "tabpfn35_fit10k"

    else:
        n_splits = SMOKE_N_SPLITS
        inner_te_splits = SMOKE_INNER_TE_SPLITS
        n_estimators = SMOKE_N_ESTIMATORS
        fit_rows_limit = SMOKE_FIT_ROWS
        batch_size = SMOKE_PREDICT_BATCH_SIZE
        prefix = "tabpfn35_smoke_fit10k"

    config_tag = make_config_tag(
        data_signature=data_signature,
        n_splits=n_splits,
        inner_te_splits=inner_te_splits,
        n_estimators=n_estimators,
        fit_rows_limit=fit_rows_limit,
        batch_size=batch_size,
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    split_list = list(
        cv.split(np.zeros(len(y)), y)
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

    test_fold_predictions = [
        None
    ] * n_splits

    print("\n" + "=" * 100)
    print("START CROSS-VALIDATION")
    print(f"Train/test rows           : {len(y):,} / {len(X_base_test):,}")
    print(f"Outer folds               : {n_splits}")
    print(f"Inner target-encoding folds: {inner_te_splits}")
    print(f"TabPFN estimators          : {n_estimators}")
    print(f"Actual fit-row limit       : {fit_rows_limit:,}")
    print(f"Initial prediction batch   : {batch_size:,}")
    print(f"Memory saving mode         : True")
    print(f"Resume completed folds     : {RESUME_COMPLETED_FOLDS}")
    print("=" * 100)

    # ------------------------------------------------------------------------
    # 外层 K 折
    # ------------------------------------------------------------------------

    for fold, (tr_idx, va_idx) in enumerate(
        split_list,
        start=1,
    ):

        fold_start = time.time()

        checkpoint_path = (
            CHECKPOINT_DIR
            / f"fold_{fold}.npz"
        )

        # --------------------------------------------------------------
        # 先加载已完成折。
        # --------------------------------------------------------------

        if (
            MODE.lower() == "full"
            and RESUME_COMPLETED_FOLDS
        ):
            checkpoint = load_checkpoint(
                checkpoint_path=checkpoint_path,
                expected_config_tag=config_tag,
                expected_valid_idx=va_idx,
                expected_test_rows=len(X_base_test),
            )

            if checkpoint is not None:
                saved_valid_pred, saved_test_pred, saved_auc = checkpoint

                oof[va_idx] = saved_valid_pred
                fold_assignment[va_idx] = fold

                test_fold_predictions[fold - 1] = saved_test_pred

                fold_scores.append({
                    "fold": fold,
                    "auc": saved_auc,
                    "n_valid": len(va_idx),
                })

                print(
                    f"\nFOLD {fold}/{n_splits}: "
                    f"checkpoint loaded, skipping training."
                )

                print(
                    f"    Saved fold AUC: {saved_auc:.8f}"
                )

                continue

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

        # 不必先复制全体测试数据；目标编码函数内部会复制。
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        # --------------------------------------------------------------
        # 在当前外层折中，先构建无泄漏目标编码。
        # --------------------------------------------------------------

        X_tr, X_va, X_te = add_leakage_safe_target_encodings(
            X_outer_train=X_tr_base,
            y_outer_train=y_tr,
            X_outer_valid=X_va_base,
            X_test=X_base_test,
            inner_splits=inner_te_splits,
            seed=RANDOM_STATE + fold,
        )

        del X_tr_base
        del X_va_base
        gc.collect()

        # --------------------------------------------------------------
        # 关键修复：显式选 10,000 行，再调用 model.fit。
        # --------------------------------------------------------------

        fit_idx = get_fit_indices(
            y=y_tr,
            max_rows=fit_rows_limit,
            seed=RANDOM_STATE + 10_000 + fold,
        )

        X_fit = (
            X_tr.iloc[fit_idx]
            .reset_index(drop=True)
        )

        y_fit = y_tr[fit_idx]

        print(
            f"    Outer training rows     : {len(X_tr):,}"
        )

        print(
            f"    Actual model.fit rows   : {len(X_fit):,}"
        )

        print(
            f"    Validation rows         : {len(X_va):,}"
        )

        print(
            f"    Test prediction rows    : {len(X_te):,}"
        )

        if len(X_fit) > fit_rows_limit:
            raise RuntimeError(
                "Internal error: fit rows exceed configured limit."
            )

        model = build_model(
            n_estimators=n_estimators,
            seed=RANDOM_STATE + 100 * fold,
        )

        print("    Fitting TabPFN...")

        try:
            model.fit(
                X_fit,
                y_fit,
            )

        except Exception as exc:
            if is_cuda_oom(exc):
                raise RuntimeError(
                    "TabPFN 在 model.fit() 阶段发生 CUDA OOM。\n"
                    "建议将 FULL_FIT_ROWS 从 10,000 降至 5,000，"
                    "更换 checkpoint 目录后重跑。"
                ) from exc

            raise

        # 训练集副本不再需要。
        del X_fit
        del y_fit
        del fit_idx
        del X_tr
        del y_tr

        gc.collect()
        torch.cuda.empty_cache()

        # --------------------------------------------------------------
        # 当前外层验证折预测
        # --------------------------------------------------------------

        print("    Predicting validation fold...")

        p_valid = predict_positive_probability(
            model=model,
            X=X_va,
            batch_size=batch_size,
        )

        fold_auc = roc_auc_score(
            y_va,
            p_valid,
        )

        oof[va_idx] = p_valid
        fold_assignment[va_idx] = fold

        print(
            f"    Fold {fold} ROC-AUC: {fold_auc:.8f}"
        )

        fold_scores.append({
            "fold": fold,
            "auc": float(fold_auc),
            "n_valid": len(va_idx),
        })

        # --------------------------------------------------------------
        # 当前折测试集预测
        # --------------------------------------------------------------

        print("    Predicting test set...")

        p_test = predict_positive_probability(
            model=model,
            X=X_te,
            batch_size=batch_size,
        )

        test_fold_predictions[fold - 1] = p_test

        # --------------------------------------------------------------
        # 保存当前折 checkpoint
        # --------------------------------------------------------------

        if MODE.lower() == "full":
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                config_tag=config_tag,
                valid_idx=va_idx,
                valid_pred=p_valid,
                test_pred=p_test,
                fold_auc=fold_auc,
            )

            print(
                f"    Checkpoint saved: {checkpoint_path}"
            )

        fold_elapsed = time.time() - fold_start

        print(
            f"    Fold elapsed: {fold_elapsed / 60:.1f} minutes"
        )

        # 释放当前折的模型及大型 DataFrame。
        del model
        del X_va
        del X_te
        del p_valid
        del p_test
        del y_va

        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # 9. 汇总 OOF 和测试集预测
    # =========================================================================

    if np.isnan(oof).any():
        missing = int(np.isnan(oof).sum())

        raise RuntimeError(
            f"OOF predictions incomplete: {missing:,} missing rows."
        )

    if any(
        p is None
        for p in test_fold_predictions
    ):
        raise RuntimeError(
            "Some folds did not produce test predictions."
        )

    oof_auc = roc_auc_score(
        y,
        oof,
    )

    test_pred = np.mean(
        np.vstack(test_fold_predictions),
        axis=0,
    ).astype(np.float32)

    # =========================================================================
    # 10. 输出结果
    # =========================================================================

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

    fold_df = (
        pd.DataFrame(fold_scores)
        .sort_values("fold")
        .reset_index(drop=True)
    )

    oof_path = (
        OUTPUT_DIR
        / f"{prefix}_oof.csv"
    )

    test_pred_path = (
        OUTPUT_DIR
        / f"{prefix}_test_predictions.csv"
    )

    fold_path = (
        OUTPUT_DIR
        / f"{prefix}_fold_metrics.csv"
    )

    oof_df.to_csv(
        oof_path,
        index=False,
    )

    test_pred_df.to_csv(
        test_pred_path,
        index=False,
    )

    fold_df.to_csv(
        fold_path,
        index=False,
    )

    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)

    print(
        f"OOF ROC-AUC               : {oof_auc:.8f}"
    )

    print(
        f"Fold AUCs                 : "
        f"{[
            round(float(x), 8)
            for x in fold_df['auc'].to_numpy()
        ]}"
    )

    print(f"OOF saved                 : {oof_path}")
    print(f"Test predictions saved    : {test_pred_path}")
    print(f"Fold metrics saved        : {fold_path}")

    # =========================================================================
    # 11. 生成单模型提交文件
    # =========================================================================

    if MODE.lower() == "full":
        submission = pd.DataFrame({
            ID_COL: test_ids.to_numpy(),
            TARGET_COL: test_pred,
        })

        if SAMPLE_SUB_PATH.exists():
            sample_sub = pd.read_csv(
                SAMPLE_SUB_PATH
            )

            if ID_COL in sample_sub.columns:
                submission = sample_sub[[ID_COL]].merge(
                    submission,
                    on=ID_COL,
                    how="left",
                    validate="one_to_one",
                )

        if submission[TARGET_COL].isna().any():
            raise RuntimeError(
                "Submission contains missing predictions."
            )

        if len(submission) != len(test_ids):
            raise RuntimeError(
                "Submission row count does not match test set."
            )

        sub_path = (
            OUTPUT_DIR
            / f"{prefix}_submission.csv"
        )

        submission.to_csv(
            sub_path,
            index=False,
        )

        print(
            f"Submission saved          : {sub_path}"
        )

        print(
            "NOTE: This is a TabPFN-only submission, "
            "not your champion blend."
        )

    else:
        print(
            "SMOKE mode: no submission was created."
        )

    # =========================================================================
    # 12. Sanity checks
    # =========================================================================

    print("\nPrediction sanity checks:")

    print(
        f"  OOF min/max              : "
        f"{float(oof.min()):.6f} / {float(oof.max()):.6f}"
    )

    print(
        f"  Test min/max             : "
        f"{float(test_pred.min()):.6f} / {float(test_pred.max()):.6f}"
    )

    print(
        f"  Test prediction rows     : {len(test_pred):,}"
    )

    print("Done.")


if __name__ == "__main__":
    main()