# -*- coding: utf-8 -*-
"""
TabM V2: Feature Representation + Distance Profile Screening
============================================================

实验目的：
1. 不再把所有连续变量都转成类别变量。
2. 只对 13 个离散服务评分增加类别表示。
3. 增加服务评分汇总及延误结构特征。
4. 增加精确 Flight Distance 画像特征。
5. 对原始变量和条件组合进行交叉拟合目标编码。
6. 使用一次分层 80/20 验证进行快速筛选。

本脚本不执行完整 5-fold CV，也不生成最终提交文件。

依赖：
    numpy
    pandas
    scikit-learn
    scipy
    torch
    tabm
    rtdl_num_embeddings

默认数据路径：
    D:\\kaggle\\airline\\data\\train.csv
    D:\\kaggle\\airline\\data\\test.csv
"""

import os
import gc
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from torch import nn
from torch.optim import AdamW

from scipy.stats import rankdata

from sklearn.model_selection import (
    StratifiedShuffleSplit,
    StratifiedKFold,
)
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score

from tabm import TabM
from rtdl_num_embeddings import (
    PiecewiseLinearEmbeddings,
    compute_bins,
)

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PATHS / SETTINGS
# =============================================================================

DATA_DIR = r"D:\kaggle\airline\data"

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

LGBM_OOF_PATH = (
    r"D:\kaggle\airline\conditional_te_experiment"
    r"\conditional_te_lgbm_oof.csv"
)

REALMLP_OOF_PATH = (
    r"D:\kaggle\airline\lean_realmlp_blend"
    r"\lean_realmlp_oof.csv"
)

OUTPUT_DIR = (
    r"D:\kaggle\airline\tabm_v2_feature_screen"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 2. GLOBAL CONFIGURATION
# =============================================================================

ID_COL = "id"
TARGET = "satisfaction"

SEED = 42

# 快速筛选，不是最终交叉验证
HOLDOUT_SIZE = 0.20

# 距离画像训练集交叉拟合折数
PROFILE_FOLDS = 4

# TabM training
MAX_EPOCHS = 20
MIN_EPOCHS = 5
PATIENCE = 4

LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0003
BATCH_SIZE = 2048

# TabM architecture
TABM_K = 16
TABM_BLOCKS = 2
TABM_WIDTH = 256
TABM_DROPOUT = 0.10

# Piecewise Linear Embeddings
PLE_N_BINS = 32
PLE_D_EMBEDDING = 8
MAX_BIN_SAMPLE = 100_000

# 筛选用的融合权重
# 这些权重只用于诊断，不代表最终应该选择其中的最高值。
BLEND_WEIGHTS = [
    0.00,
    0.025,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.30,
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
# 3. COLUMN DEFINITIONS
# =============================================================================

NUMERIC_COLS = [
    "Age",
    "Flight Distance",
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
    "Cleanliness",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

CAT_COLS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

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
    "Cleanliness",
]

# 航程画像中类别变量的固定取值集合
PROFILE_VOCAB = {
    "Gender": [
        "Female",
        "Male",
    ],
    "Customer Type": [
        "Loyal Customer",
        "disloyal Customer",
    ],
    "Type of Travel": [
        "Business travel",
        "Personal Travel",
    ],
    "Class": [
        "Business",
        "Eco",
        "Eco Plus",
    ],
}

MISSING = "__MISSING__"


# =============================================================================
# 4. GENERAL HELPERS
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(message: str) -> None:
    print(message, flush=True)


def require_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series
        .astype("string")
        .fillna(MISSING)
        .astype(str)
        .str.strip()
    )


def numeric_key(series: pd.Series) -> pd.Series:
    """
    将数值转换成稳定的字符串键。

    例如：
        5
        5.0

    均映射到同一个键，避免字符串表示不一致。
    """
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    result = []

    for value in values:
        if not np.isfinite(value):
            result.append(MISSING)
        else:
            result.append(
                format(float(value), ".12g")
            )

    return pd.Series(
        result,
        index=series.index,
        dtype=object,
    )


def normalized_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Cannot rank non-finite predictions."
        )

    return (
        rankdata(
            values,
            method="average",
        ) / len(values)
    )


def read_aligned_oof(
    path: str,
    train_ids: np.ndarray,
    y: np.ndarray,
    name: str,
) -> np.ndarray:
    """
    读取并按 train.csv 的 id 对齐已有 OOF 预测。
    """
    require_file(path)

    df = pd.read_csv(path)

    required = {
        ID_COL,
        TARGET,
        "prediction",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"{name} needs columns {sorted(required)}; "
            f"got {df.columns.tolist()}"
        )

    df[ID_COL] = pd.to_numeric(
        df[ID_COL],
        errors="raise",
    )

    if df[ID_COL].duplicated().any():
        raise ValueError(
            f"{name} has duplicate IDs."
        )

    reference_ids = set(
        train_ids.tolist()
    )

    candidate_ids = set(
        df[ID_COL].tolist()
    )

    if candidate_ids != reference_ids:
        raise ValueError(
            f"{name} IDs do not match train.csv."
        )

    aligned = (
        df.set_index(ID_COL)
        .loc[train_ids]
        .reset_index()
    )

    aligned_y = aligned[TARGET].to_numpy(
        dtype=np.int8
    )

    if not np.array_equal(
        aligned_y,
        y.astype(np.int8),
    ):
        raise ValueError(
            f"{name} target does not align with train.csv."
        )

    pred = aligned["prediction"].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(pred).all():
        raise ValueError(
            f"{name} contains non-finite predictions."
        )

    log(
        f"{name}: full OOF AUC="
        f"{roc_auc_score(y, pred):.9f}"
    )

    return pred


# =============================================================================
# 5. ROW-LEVEL FEATURE ENGINEERING
# =============================================================================

def build_row_features(df: pd.DataFrame):
    """
    构造与标签无关的逐行特征。

    返回：
        raw: 原始数据
        numeric: 数值输入
        categorical: 类别输入
        te: 目标编码的类别键
    """
    raw = df.copy().reset_index(drop=True)

    num = pd.DataFrame(
        index=raw.index
    )

    cat = pd.DataFrame(
        index=raw.index
    )

    te = pd.DataFrame(
        index=raw.index
    )

    # -------------------------------------------------------------------------
    # A. 保留原始数值变量
    # -------------------------------------------------------------------------

    for col in NUMERIC_COLS:
        num[col] = (
            pd.to_numeric(
                raw[col],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

    # -------------------------------------------------------------------------
    # B. 服务评分汇总特征
    # -------------------------------------------------------------------------

    ratings = raw[RATING_COLS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    num["rating_mean"] = ratings.mean(axis=1)

    num["rating_min"] = ratings.min(axis=1)

    num["rating_max"] = ratings.max(axis=1)

    num["rating_std"] = ratings.std(axis=1)

    num["rating_range"] = (
        num["rating_max"]
        - num["rating_min"]
    )

    num["rating_zero_count"] = (
        ratings == 0
    ).sum(axis=1)

    num["rating_low_count_le2"] = (
        ratings <= 2
    ).sum(axis=1)

    num["rating_high_count_ge4"] = (
        ratings >= 4
    ).sum(axis=1)

    # 数字服务体验
    digital_cols = [
        "Inflight wifi service",
        "Ease of Online booking",
        "Online boarding",
    ]

    num["digital_service_mean"] = (
        ratings[digital_cols]
        .mean(axis=1)
    )

    # 机舱和服务体验
    cabin_cols = [
        "Food and drink",
        "Seat comfort",
        "Inflight entertainment",
        "On-board service",
        "Leg room service",
        "Baggage handling",
        "Checkin service",
        "Cleanliness",
    ]

    num["cabin_experience_mean"] = (
        ratings[cabin_cols]
        .mean(axis=1)
    )

    # -------------------------------------------------------------------------
    # C. 延误结构特征
    # -------------------------------------------------------------------------

    dep_raw = pd.to_numeric(
        raw["Departure Delay in Minutes"],
        errors="coerce",
    )

    arr_raw = pd.to_numeric(
        raw["Arrival Delay in Minutes"],
        errors="coerce",
    )

    dep = (
        dep_raw
        .fillna(0)
        .clip(lower=0)
    )

    arr = (
        arr_raw
        .fillna(0)
        .clip(lower=0)
    )

    # 保留到达延误原始缺失信息
    num["arrival_delay_missing"] = (
        arr_raw.isna()
        .astype(np.float32)
    )

    num["total_delay"] = dep + arr

    num["delay_difference_arr_minus_dep"] = (
        arr - dep
    )

    num["delay_abs_difference"] = (
        arr - dep
    ).abs()

    num["log1p_departure_delay"] = np.log1p(dep)

    num["log1p_arrival_delay"] = np.log1p(arr)

    num["departure_delayed"] = (
        dep > 0
    ).astype(np.float32)

    num["arrival_delayed"] = (
        arr > 0
    ).astype(np.float32)

    num["both_delayed"] = (
        (dep > 0) & (arr > 0)
    ).astype(np.float32)

    # -------------------------------------------------------------------------
    # D. 原始类别特征
    # -------------------------------------------------------------------------

    for col in CAT_COLS:
        cat[col] = clean_text(raw[col])

    # -------------------------------------------------------------------------
    # E. 仅对低基数评分变量创建类别副本
    #
    # 不把 Age、Flight Distance、精确延误分钟数全部转成类别。
    # -------------------------------------------------------------------------

    for col in RATING_COLS:
        cat[f"cat_rating__{col}"] = clean_text(
            raw[col]
        )

    # -------------------------------------------------------------------------
    # F. 合理分箱：避免连续变量产生过多的类别
    # -------------------------------------------------------------------------

    age = pd.to_numeric(
        raw["Age"],
        errors="coerce",
    )

    distance = pd.to_numeric(
        raw["Flight Distance"],
        errors="coerce",
    )

    cat["age_group"] = (
        pd.cut(
            age,
            bins=[
                -np.inf,
                18,
                25,
                35,
                45,
                55,
                65,
                np.inf,
            ],
            labels=False,
        )
        .astype("string")
        .fillna(MISSING)
        .astype(str)
    )

    cat["distance_group"] = (
        pd.cut(
            distance,
            bins=[
                -np.inf,
                250,
                500,
                750,
                1000,
                1500,
                2000,
                3000,
                5000,
                np.inf,
            ],
            labels=False,
        )
        .astype("string")
        .fillna(MISSING)
        .astype(str)
    )

    cat["departure_delay_group"] = (
        pd.cut(
            dep_raw,
            bins=[
                -np.inf,
                0,
                5,
                15,
                30,
                60,
                120,
                180,
                np.inf,
            ],
            labels=False,
        )
        .astype("string")
        .fillna(MISSING)
        .astype(str)
    )

    cat["arrival_delay_group"] = (
        pd.cut(
            arr_raw,
            bins=[
                -np.inf,
                0,
                5,
                15,
                30,
                60,
                120,
                180,
                np.inf,
            ],
            labels=False,
        )
        .astype("string")
        .fillna(MISSING)
        .astype(str)
    )

    # -------------------------------------------------------------------------
    # G. 乘客群体交互
    # -------------------------------------------------------------------------

    cat["traveler_segment"] = (
        clean_text(raw["Type of Travel"])
        + "|"
        + clean_text(raw["Customer Type"])
        + "|"
        + clean_text(raw["Class"])
    )

    # -------------------------------------------------------------------------
    # H. 原始变量的目标编码键
    #
    # 一共 21 个原始预测变量。
    # 数值变量使用稳定的字符串键。
    # -------------------------------------------------------------------------

    for col in CAT_COLS:
        te[f"raw__{col}"] = clean_text(
            raw[col]
        )

    for col in NUMERIC_COLS:
        te[f"raw__{col}"] = numeric_key(
            raw[col]
        )

    # -------------------------------------------------------------------------
    # I. 服务评分 × 乘客群体条件编码
    # -------------------------------------------------------------------------

    segment = cat["traveler_segment"]

    for col in RATING_COLS:
        te[
            f"conditional__{col}__segment"
        ] = (
            numeric_key(raw[col])
            + "|"
            + segment
        )

    # -------------------------------------------------------------------------
    # J. 精确航程 × 乘客类型
    # -------------------------------------------------------------------------

    distance_key = numeric_key(
        raw["Flight Distance"]
    )

    te["distance__class"] = (
        distance_key
        + "|"
        + clean_text(raw["Class"])
    )

    te["distance__travel_type"] = (
        distance_key
        + "|"
        + clean_text(raw["Type of Travel"])
    )

    te["distance__customer_type"] = (
        distance_key
        + "|"
        + clean_text(raw["Customer Type"])
    )

    # -------------------------------------------------------------------------
    # K. 数值分箱 × 乘客群体变量
    # -------------------------------------------------------------------------

    distance_num = pd.to_numeric(
        raw["Flight Distance"],
        errors="coerce",
    )

    age_num = pd.to_numeric(
        raw["Age"],
        errors="coerce",
    )

    te["distance_250_bin"] = (
        np.floor(distance_num / 250)
        .astype("Int64")
        .astype("string")
        .fillna(MISSING)
        .astype(str)
    )

    te["age_5_bin"] = (
        np.floor(age_num / 5)
        .astype("Int64")
        .astype("string")
        .fillna(MISSING)
        .astype(str)
    )

    te["traveler_segment"] = segment

    # 使用 category 类型减少重复字符串的存储开销
    for col in te.columns:
        te[col] = te[col].astype("category")

    return {
        "raw": raw,
        "numeric": num,
        "categorical": cat,
        "te": te,
    }


# =============================================================================
# 6. EXACT-DISTANCE PROFILE FEATURES
# =============================================================================

def _distance_key(df: pd.DataFrame) -> pd.Series:
    key = pd.to_numeric(
        df["Flight Distance"],
        errors="coerce",
    )

    return key.fillna(-999999.0)


def make_route_profile(
    reference_raw: pd.DataFrame,
    query_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    使用 reference_raw 中的行构造画像，再映射到 query_raw。

    不使用 satisfaction 标签。

    这里的 route 是基于精确 Flight Distance 的代理变量，
    并非真实的出发机场—到达机场航线。
    """
    ref = (
        reference_raw
        .reset_index(drop=True)
        .copy()
    )

    qry = (
        query_raw
        .reset_index(drop=True)
        .copy()
    )

    ref_key = _distance_key(ref)
    qry_key = _distance_key(qry)

    out = pd.DataFrame(
        index=qry.index
    )

    # -------------------------------------------------------------------------
    # A. 精确距离样本量
    # -------------------------------------------------------------------------

    counts = ref_key.value_counts(
        dropna=False
    )

    out["route_log_count"] = np.log1p(
        qry_key
        .map(counts)
        .fillna(0)
        .to_numpy(dtype=np.float64)
    )

    # -------------------------------------------------------------------------
    # B. 按精确距离计算评分、年龄及延误的均值和标准差
    # -------------------------------------------------------------------------

    measures = (
        RATING_COLS
        + [
            "Age",
            "Departure Delay in Minutes",
            "Arrival Delay in Minutes",
        ]
    )

    ref_measures = pd.DataFrame(
        index=ref.index
    )

    for col in measures:
        ref_measures[col] = (
            pd.to_numeric(
                ref[col],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

    ref_measures["__distance_key"] = ref_key

    grouped = (
        ref_measures
        .groupby(
            "__distance_key",
            dropna=False,
        )[measures]
        .agg(["mean", "std"])
    )

    for col in measures:
        global_mean = ref_measures[col].mean()
        global_std = ref_measures[col].std()

        if not np.isfinite(global_mean):
            global_mean = 0.0

        if not np.isfinite(global_std):
            global_std = 0.0

        mean_map = grouped[(col, "mean")]
        std_map = grouped[(col, "std")]

        safe_name = (
            col.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
        )

        out[f"route_{safe_name}_mean"] = (
            qry_key
            .map(mean_map)
            .fillna(global_mean)
            .to_numpy(dtype=np.float64)
        )

        out[f"route_{safe_name}_std"] = (
            qry_key
            .map(std_map)
            .fillna(global_std)
            .to_numpy(dtype=np.float64)
        )

    # -------------------------------------------------------------------------
    # C. 同一距离上的乘客构成
    # -------------------------------------------------------------------------

    for col, vocab in PROFILE_VOCAB.items():
        ref_values = clean_text(ref[col])

        for value in vocab:
            indicator = (
                ref_values == value
            ).astype(np.float32)

            tmp = pd.DataFrame({
                "key": ref_key,
                "indicator": indicator,
            })

            share_map = (
                tmp
                .groupby(
                    "key",
                    dropna=False,
                )["indicator"]
                .mean()
            )

            global_share = (
                float(indicator.mean())
                if len(indicator)
                else 0.0
            )

            safe_value = (
                value.lower()
                .replace(" ", "_")
            )

            out[
                f"route_share_"
                f"{col.lower().replace(' ', '_')}_"
                f"{safe_value}"
            ] = (
                _distance_key(qry)
                .map(share_map)
                .fillna(global_share)
                .to_numpy(dtype=np.float64)
            )

    out = out.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return out


def crossfit_route_profiles(
    raw_fit: pd.DataFrame,
    y_fit: np.ndarray,
    n_splits: int = PROFILE_FOLDS,
) -> pd.DataFrame:
    """
    训练集画像交叉拟合。

    对每个子折：
        reference = 其他子折
        query = 当前子折

    这样每一行的画像都不使用该行自身。
    """
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED + 700,
    )

    result = None

    for fold, (idx_ref, idx_query) in enumerate(
        splitter.split(
            np.zeros(len(y_fit)),
            y_fit,
        ),
        start=1,
    ):
        fold_profile = make_route_profile(
            raw_fit.iloc[idx_ref],
            raw_fit.iloc[idx_query],
        )

        if result is None:
            result = pd.DataFrame(
                index=np.arange(len(raw_fit)),
                columns=fold_profile.columns,
                dtype=np.float32,
            )

        result.iloc[idx_query, :] = (
            fold_profile.to_numpy(
                dtype=np.float32
            )
        )

        log(
            f"  distance-profile cross-fit "
            f"{fold}/{n_splits} complete"
        )

    if result is None or result.isna().all(axis=None):
        raise RuntimeError(
            "Route-profile cross-fitting failed."
        )

    return result.reset_index(drop=True)


# =============================================================================
# 7. CATEGORICAL MATRIX PREPROCESSING
# =============================================================================

def encode_categories(
    cat_fit: pd.DataFrame,
    cat_query: pd.DataFrame,
):
    """
    仅使用 fit 分区建立类别编码。

    0 = 未见类别
    1+ = fit 分区中出现的已知类别
    """
    maps = {}

    fit_cols = []
    query_cols = []

    for col in cat_fit.columns:
        fit_values = clean_text(
            cat_fit[col]
        )

        query_values = clean_text(
            cat_query[col]
        )

        vocab = sorted(
            fit_values.unique().tolist()
        )

        mapping = {
            value: idx + 1
            for idx, value in enumerate(vocab)
        }

        maps[col] = mapping

        fit_cols.append(
            fit_values
            .map(mapping)
            .fillna(0)
            .to_numpy(dtype=np.int64)
        )

        query_cols.append(
            query_values
            .map(mapping)
            .fillna(0)
            .to_numpy(dtype=np.int64)
        )

    X_fit_cat = np.column_stack(
        fit_cols
    ).astype(
        np.int64,
        copy=True,
    )

    X_query_cat = np.column_stack(
        query_cols
    ).astype(
        np.int64,
        copy=True,
    )

    cardinalities = [
        len(maps[col]) + 1
        for col in cat_fit.columns
    ]

    return (
        X_fit_cat,
        X_query_cat,
        cardinalities,
    )


# =============================================================================
# 8. NUMERIC MATRIX PREPROCESSING
# =============================================================================

def create_numeric_matrices(
    num_fit: pd.DataFrame,
    num_query: pd.DataFrame,
    te_fit: pd.DataFrame,
    te_query: pd.DataFrame,
    profile_fit: pd.DataFrame,
    profile_query: pd.DataFrame,
):
    """
    数值输入由三部分组成：
        1. 原始数值及人工汇总特征
        2. 精确距离画像特征
        3. 目标编码特征
    """
    all_fit = pd.concat(
        [
            num_fit.reset_index(drop=True),
            profile_fit.reset_index(drop=True),
            te_fit.reset_index(drop=True),
        ],
        axis=1,
    )

    all_query = pd.concat(
        [
            num_query.reset_index(drop=True),
            profile_query.reset_index(drop=True),
            te_query.reset_index(drop=True),
        ],
        axis=1,
    )

    all_query = all_query.reindex(
        columns=all_fit.columns
    )

    all_fit = all_fit.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    all_query = all_query.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # 缺失值填补参数只从 fit 分区估计
    medians = (
        all_fit
        .median(axis=0, numeric_only=True)
        .fillna(0.0)
    )

    all_fit = (
        all_fit
        .fillna(medians)
        .fillna(0.0)
    )

    all_query = (
        all_query
        .fillna(medians)
        .fillna(0.0)
    )

    X_fit = all_fit.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    X_query = all_query.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if (
        not np.isfinite(X_fit).all()
        or not np.isfinite(X_query).all()
    ):
        raise ValueError(
            "Non-finite value remains in numeric model inputs."
        )

    return (
        X_fit,
        X_query,
        list(all_fit.columns),
    )


# =============================================================================
# 9. TARGET ENCODING
# =============================================================================

def fit_target_encoder(
    te_fit_raw: pd.DataFrame,
    te_query_raw: pd.DataFrame,
    y_fit: np.ndarray,
):
    """
    sklearn TargetEncoder 内部对训练部分执行交叉拟合。

    训练集编码：
        fit_transform

    验证集编码：
        transform

    验证集标签不会参与目标编码。
    """
    encoder = TargetEncoder(
        target_type="binary",
        smooth="auto",
        cv=5,
        shuffle=True,
        random_state=SEED + 900,
    )

    fit_values = encoder.fit_transform(
        te_fit_raw,
        y_fit,
    ).astype(np.float32)

    query_values = encoder.transform(
        te_query_raw
    ).astype(np.float32)

    te_fit = pd.DataFrame(
        fit_values,
        columns=[
            f"target_enc__{col}"
            for col in te_fit_raw.columns
        ],
    )

    te_query = pd.DataFrame(
        query_values,
        columns=[
            f"target_enc__{col}"
            for col in te_query_raw.columns
        ],
    )

    return te_fit, te_query


# =============================================================================
# 10. TABM MODEL
# =============================================================================

def make_model(
    X_num: np.ndarray,
    cat_cardinalities: list,
):
    """
    数值分支继续使用 PLE 数值嵌入。
    类别分支使用 TabM 的类别嵌入。
    """
    sample_size = min(
        MAX_BIN_SAMPLE,
        len(X_num),
    )

    rng = np.random.default_rng(SEED)

    sample_idx = rng.choice(
        len(X_num),
        size=sample_size,
        replace=False,
    )

    bin_source = torch.from_numpy(
        X_num[sample_idx].copy()
    ).float()

    bins = compute_bins(
        bin_source,
        n_bins=PLE_N_BINS,
    )

    embeddings = PiecewiseLinearEmbeddings(
        bins,
        d_embedding=PLE_D_EMBEDDING,
        activation=False,
        version="B",
    )

    model = TabM.make(
        n_num_features=X_num.shape[1],
        cat_cardinalities=cat_cardinalities,
        num_embeddings=embeddings,
        d_out=1,
        k=TABM_K,
        n_blocks=TABM_BLOCKS,
        d_block=TABM_WIDTH,
        dropout=TABM_DROPOUT,
    )

    return model.to(DEVICE)


@torch.no_grad()
def predict(
    model,
    X_num: np.ndarray,
    X_cat: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    model.eval()

    predictions = []

    for start in range(
        0,
        len(X_num),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(X_num),
        )

        xb_num = torch.from_numpy(
            X_num[start:end].copy()
        ).to(DEVICE)

        xb_cat = torch.from_numpy(
            X_cat[start:end].copy()
        ).to(DEVICE)

        logits = model(
            xb_num,
            xb_cat,
        ).squeeze(-1)

        probabilities = (
            torch.sigmoid(logits)
            .mean(dim=1)
            .cpu()
            .numpy()
        )

        predictions.append(probabilities)

    pred = np.concatenate(
        predictions
    )

    if not np.isfinite(pred).all():
        raise ValueError(
            "TabM produced non-finite predictions."
        )

    return pred


def train_one_epoch(
    model,
    optimizer,
    X_num,
    X_cat,
    y_array,
):
    model.train()

    permutation = np.random.permutation(
        len(y_array)
    )

    total_loss = 0.0
    total_count = 0

    for start in range(
        0,
        len(permutation),
        BATCH_SIZE,
    ):
        idx = permutation[
            start:start + BATCH_SIZE
        ]

        xb_num = torch.from_numpy(
            X_num[idx].copy()
        ).to(DEVICE)

        xb_cat = torch.from_numpy(
            X_cat[idx].copy()
        ).to(DEVICE)

        yb = torch.from_numpy(
            y_array[idx].astype(
                np.float32,
                copy=True,
            )
        ).to(DEVICE)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            xb_num,
            xb_cat,
        ).squeeze(-1)

        # TabM 输出形状：(batch_size, K)
        targets = yb.unsqueeze(1).expand_as(
            logits
        )

        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
        )

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        batch_size = len(idx)

        total_loss += (
            float(loss.item()) * batch_size
        )

        total_count += batch_size

    return total_loss / max(
        total_count,
        1,
    )


def train_with_early_stopping(
    X_fit_num,
    X_fit_cat,
    y_fit,
    X_val_num,
    X_val_cat,
    y_val,
    cat_cardinalities,
):
    model = make_model(
        X_fit_num,
        cat_cardinalities,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_auc = -np.inf
    best_epoch = 1
    best_state = None
    stale = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        loss = train_one_epoch(
            model,
            optimizer,
            X_fit_num,
            X_fit_cat,
            y_fit,
        )

        val_pred = predict(
            model,
            X_val_num,
            X_val_cat,
        )

        val_auc = roc_auc_score(
            y_val,
            val_pred,
        )

        if val_auc > best_auc + 1e-5:
            best_auc = val_auc
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            stale = 0
        else:
            stale += 1

        log(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"loss={loss:.6f} | "
            f"holdout AUC={val_auc:.9f} | "
            f"best={best_auc:.9f} @ {best_epoch}"
        )

        if (
            epoch >= MIN_EPOCHS
            and stale >= PATIENCE
        ):
            log(
                f"Early stopping at epoch {epoch}."
            )
            break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    return (
        model,
        best_epoch,
        best_auc,
    )


# =============================================================================
# 11. MAIN EXPERIMENT
# =============================================================================

def main():
    set_seed(SEED)

    log("=" * 96)
    log("TABM V2 FEATURE SCREEN — no full-CV / no submission")
    log(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        log(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    require_file(TRAIN_PATH)
    require_file(TEST_PATH)

    train = pd.read_csv(
        TRAIN_PATH
    )

    # 本次仅检查测试集存在，不训练最终提交模型
    _test = pd.read_csv(
        TEST_PATH,
        nrows=5,
    )

    required_cols = (
        [ID_COL, TARGET]
        + NUMERIC_COLS
        + CAT_COLS
    )

    missing = (
        set(required_cols)
        - set(train.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required train columns: {sorted(missing)}"
        )

    ids = pd.to_numeric(
        train[ID_COL],
        errors="raise",
    ).to_numpy()

    y = train[TARGET].astype(
        np.int8
    ).to_numpy()

    if len(np.unique(ids)) != len(ids):
        raise ValueError(
            "Duplicate IDs found."
        )

    if len(np.unique(y)) != 2:
        raise ValueError(
            "Non-binary target found."
        )

    log(
        f"Train shape: {train.shape}; "
        f"positive rate={y.mean():.6f}"
    )

    # -------------------------------------------------------------------------
    # A. 读取当前 Champion 的 OOF
    # -------------------------------------------------------------------------

    lgb_oof = read_aligned_oof(
        LGBM_OOF_PATH,
        ids,
        y,
        "Conditional-TE LGBM",
    )

    mlp_oof = read_aligned_oof(
        REALMLP_OOF_PATH,
        ids,
        y,
        "Lean RealMLP V1",
    )

    champion_oof = (
        0.40 * normalized_rank(lgb_oof)
        + 0.60 * normalized_rank(mlp_oof)
    )

    log(
        "Rebuilt Champion full OOF AUC="
        f"{roc_auc_score(y, champion_oof):.9f}"
    )

    # -------------------------------------------------------------------------
    # B. 创建一次分层 80/20 划分
    # -------------------------------------------------------------------------

    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=HOLDOUT_SIZE,
        random_state=SEED,
    )

    fit_idx, val_idx = next(
        split.split(
            np.zeros(len(y)),
            y,
        )
    )

    y_fit = y[fit_idx]
    y_val = y[val_idx]

    log(
        f"Screen fit rows={len(fit_idx):,}; "
        f"validation rows={len(val_idx):,}"
    )

    # -------------------------------------------------------------------------
    # C. 构造行级特征
    # -------------------------------------------------------------------------

    log(
        "Building row-level features for fit and holdout..."
    )

    row_fit = build_row_features(
        train.iloc[fit_idx].reset_index(drop=True)
    )

    row_val = build_row_features(
        train.iloc[val_idx].reset_index(drop=True)
    )

    raw_fit = row_fit["raw"]
    raw_val = row_val["raw"]

    num_fit = row_fit["numeric"]
    num_val = row_val["numeric"]

    cat_fit_raw = row_fit["categorical"]
    cat_val_raw = row_val["categorical"]

    te_fit_raw = row_fit["te"]
    te_val_raw = row_val["te"]

    # -------------------------------------------------------------------------
    # D. 距离画像
    # -------------------------------------------------------------------------

    log(
        "Creating cross-fitted exact-distance profiles for fit rows..."
    )

    profile_fit = crossfit_route_profiles(
        raw_fit,
        y_fit,
        PROFILE_FOLDS,
    )

    log(
        "Creating validation distance profiles from fit partition..."
    )

    profile_val = make_route_profile(
        raw_fit,
        raw_val,
    )

    # -------------------------------------------------------------------------
    # E. 目标编码
    # -------------------------------------------------------------------------

    log(
        "Fitting cross-fitted TargetEncoder..."
    )

    te_fit, te_val = fit_target_encoder(
        te_fit_raw,
        te_val_raw,
        y_fit,
    )

    del te_fit_raw
    del te_val_raw

    del row_fit["te"]
    del row_val["te"]

    gc.collect()

    # -------------------------------------------------------------------------
    # F. 生成数值矩阵和类别矩阵
    # -------------------------------------------------------------------------

    (
        X_fit_num,
        X_val_num,
        numeric_feature_names,
    ) = create_numeric_matrices(
        num_fit,
        num_val,
        te_fit,
        te_val,
        profile_fit,
        profile_val,
    )

    (
        X_fit_cat,
        X_val_cat,
        cat_cardinalities,
    ) = encode_categories(
        cat_fit_raw,
        cat_val_raw,
    )

    log(
        f"Numeric feature count: {X_fit_num.shape[1]}"
    )

    log(
        f"Categorical feature count: {X_fit_cat.shape[1]}"
    )

    log(
        "Categorical cardinalities "
        f"(max={max(cat_cardinalities)}): "
        f"{cat_cardinalities}"
    )

    # -------------------------------------------------------------------------
    # G. 训练 TabM
    # -------------------------------------------------------------------------

    log(
        "Training TabM V2..."
    )

    model, best_epoch, best_val_auc = train_with_early_stopping(
        X_fit_num,
        X_fit_cat,
        y_fit,
        X_val_num,
        X_val_cat,
        y_val,
        cat_cardinalities,
    )

    val_pred = predict(
        model,
        X_val_num,
        X_val_cat,
    )

    tabm_auc = roc_auc_score(
        y_val,
        val_pred,
    )

    # -------------------------------------------------------------------------
    # H. 在相同验证行上比较 Champion
    # -------------------------------------------------------------------------

    champion_val_rank = normalized_rank(
        champion_oof[val_idx]
    )

    tabm_val_rank = normalized_rank(
        val_pred
    )

    champion_val_auc = roc_auc_score(
        y_val,
        champion_oof[val_idx],
    )

    rho = pd.Series(
        champion_val_rank
    ).corr(
        pd.Series(tabm_val_rank),
        method="spearman",
    )

    # -------------------------------------------------------------------------
    # I. 融合诊断
    # -------------------------------------------------------------------------

    blend_rows = []

    for w in BLEND_WEIGHTS:
        blend = (
            (1.0 - w) * champion_val_rank
            + w * tabm_val_rank
        )

        auc = roc_auc_score(
            y_val,
            blend,
        )

        blend_rows.append({
            "tabm_weight": w,
            "champion_weight": 1.0 - w,
            "holdout_auc": auc,
            "gain_vs_champion": (
                auc - champion_val_auc
            ),
        })

    blend_df = (
        pd.DataFrame(blend_rows)
        .sort_values("tabm_weight")
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # J. 输出结果
    # -------------------------------------------------------------------------

    log("\n" + "=" * 96)
    log(
        "SCREENING RESULT "
        "(same 20% holdout for TabM and Champion)"
    )
    log("=" * 96)

    log(
        f"TabM V2 holdout AUC      = {tabm_auc:.9f}"
    )

    log(
        f"Champion holdout AUC     = {champion_val_auc:.9f}"
    )

    log(
        f"Best selected epoch      = {best_epoch}"
    )

    log(
        f"Spearman rank correlation= {rho:.6f}"
    )

    log("\nBlend screen:")

    log(
        blend_df.to_string(
            index=False,
            formatters={
                "holdout_auc": "{:.9f}".format,
                "gain_vs_champion": "{:+.9f}".format,
            },
        )
    )

    # -------------------------------------------------------------------------
    # K. 保存筛选结果
    # -------------------------------------------------------------------------

    oof_out = pd.DataFrame({
        ID_COL: ids[val_idx],
        TARGET: y_val,
        "tabm_v2_prediction": val_pred,
        "champion_oof_prediction": champion_oof[val_idx],
    })

    oof_path = os.path.join(
        OUTPUT_DIR,
        "tabm_v2_holdout_predictions.csv",
    )

    metrics_path = os.path.join(
        OUTPUT_DIR,
        "tabm_v2_blend_screen.csv",
    )

    oof_out.to_csv(
        oof_path,
        index=False,
        encoding="utf-8-sig",
    )

    blend_df.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    log(
        f"\nSaved predictions: {oof_path}"
    )

    log(
        f"Saved blend screen: {metrics_path}"
    )

    log(
        "Decision rule: only proceed to full CV "
        "if TabM adds repeatable holdout blend gain; "
        "do not submit this screen as a final model."
    )

    # -------------------------------------------------------------------------
    # L. 清理内存
    # -------------------------------------------------------------------------

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()