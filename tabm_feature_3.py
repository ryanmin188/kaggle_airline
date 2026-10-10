# -*- coding: utf-8 -*-
"""
TabM feature engineering ablation for Kaggle S6E10 Airline Satisfaction.

M0: raw numeric + raw categorical features
M1: M0 + rating summaries, zero/low-rating flags, delay features,
    rating-category views and coarse numeric bins
M2: M1 + cross-fitted exact-flight-distance profile features
M3: M2 + cross-fitted target encoding and frequency features

Validation:
    First fold of StratifiedKFold(5, shuffle=True, random_state=42).
    Assumes existing LGBM and RealMLP OOF files use this exact split.

This is a feature-screening experiment, not a final submission pipeline.
"""

import os
import gc
import random
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score

from tabm import TabM
from rtdl_num_embeddings import (
    PiecewiseLinearEmbeddings,
    compute_bins,
)

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PATHS AND CONFIGURATION
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
    r"D:\kaggle\airline\tabm_feature_ablation_v3"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

ID_COL = "id"
TARGET = "satisfaction"

SEED = 42
N_FOLDS = 5
PROFILE_FOLDS = 4

# Training
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

# PLE numerical embeddings
PLE_N_BINS = 32
PLE_D_EMBEDDING = 8
MAX_BIN_SAMPLE = 100_000

# Screening weights, not final weights
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
# 2. COLUMN DEFINITIONS
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

PROFILE_MEASURES = RATING_COLS + [
    "Age",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

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
# 3. GENERAL HELPERS
# =============================================================================

def log(message: str) -> None:
    print(message, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    Convert numeric values to stable string keys.
    For example, 5 and 5.0 receive the same key.
    """
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(dtype=np.float64)

    output = [
        MISSING
        if not np.isfinite(value)
        else format(float(value), ".12g")
        for value in values
    ]

    return pd.Series(
        output,
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
            "Non-finite values cannot be ranked."
        )

    return (
        rankdata(values, method="average")
        / len(values)
    )


def read_oof(
    path: str,
    train_ids: np.ndarray,
    y: np.ndarray,
    label: str,
) -> np.ndarray:
    """
    Read OOF predictions and align them to train.csv by id.
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
            f"{label} needs columns {sorted(required)}; "
            f"found {df.columns.tolist()}"
        )

    df[ID_COL] = pd.to_numeric(
        df[ID_COL],
        errors="raise",
    )

    if df[ID_COL].duplicated().any():
        raise ValueError(
            f"{label}: duplicate IDs."
        )

    if set(df[ID_COL]) != set(train_ids):
        raise ValueError(
            f"{label}: IDs do not match train.csv."
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
            f"{label}: target alignment mismatch."
        )

    pred = aligned["prediction"].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(pred).all():
        raise ValueError(
            f"{label}: non-finite predictions."
        )

    log(
        f"{label} full OOF AUC="
        f"{roc_auc_score(y, pred):.9f}"
    )

    return pred


# =============================================================================
# 4. ROW-LEVEL FEATURE ENGINEERING
# =============================================================================

def build_row_features(
    df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:

    raw = df.reset_index(drop=True).copy()

    num = pd.DataFrame(index=raw.index)
    cat_base = pd.DataFrame(index=raw.index)
    cat_engineered = pd.DataFrame(index=raw.index)
    te = pd.DataFrame(index=raw.index)

    # -------------------------------------------------------------------------
    # A. Original numeric features
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
    # B. Original categorical features
    # -------------------------------------------------------------------------

    for col in CAT_COLS:
        cat_base[col] = clean_text(raw[col])

    # -------------------------------------------------------------------------
    # C. Rating engineering
    #
    # Preserve raw values in the original numeric columns.
    # The aggregate features separately consider nonzero ratings.
    # -------------------------------------------------------------------------

    ratings = raw[RATING_COLS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    nonzero_ratings = ratings.mask(
        ratings == 0
    )

    num_eng = pd.DataFrame(
        index=raw.index
    )

    num_eng["rating_mean_all"] = (
        ratings.mean(axis=1)
    )

    num_eng["rating_mean_nonzero"] = (
        nonzero_ratings.mean(axis=1)
    )

    num_eng["rating_min_nonzero"] = (
        nonzero_ratings.min(axis=1)
    )

    num_eng["rating_max"] = ratings.max(
        axis=1
    )

    num_eng["rating_std_all"] = (
        ratings.std(axis=1)
    )

    num_eng["rating_std_nonzero"] = (
        nonzero_ratings.std(axis=1)
    )

    num_eng["rating_range"] = (
        num_eng["rating_max"]
        - num_eng["rating_min_nonzero"]
    )

    num_eng["rating_zero_count"] = (
        (ratings == 0)
        .sum(axis=1)
        .astype(float)
    )

    num_eng["rating_nonzero_count"] = (
        (ratings > 0)
        .sum(axis=1)
        .astype(float)
    )

    num_eng["rating_low_count_1_2"] = (
        ratings.isin([1, 2])
        .sum(axis=1)
        .astype(float)
    )

    num_eng["rating_high_count_4_5"] = (
        ratings.isin([4, 5])
        .sum(axis=1)
        .astype(float)
    )

    digital_cols = [
        "Inflight wifi service",
        "Ease of Online booking",
        "Online boarding",
    ]

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

    num_eng["digital_service_mean"] = (
        ratings[digital_cols]
        .mask(ratings[digital_cols] == 0)
        .mean(axis=1)
    )

    num_eng["cabin_service_mean"] = (
        ratings[cabin_cols]
        .mask(ratings[cabin_cols] == 0)
        .mean(axis=1)
    )

    # Discrete scores get a parallel categorical view in M1+.
    # Exact flight distance and exact delay values are NOT categorized directly.
    for col in RATING_COLS:
        cat_engineered[
            f"rating_cat__{col}"
        ] = clean_text(raw[col])

    # -------------------------------------------------------------------------
    # D. Coarse numeric bins
    # -------------------------------------------------------------------------

    age = pd.to_numeric(
        raw["Age"],
        errors="coerce",
    )

    distance = pd.to_numeric(
        raw["Flight Distance"],
        errors="coerce",
    )

    dep_raw = pd.to_numeric(
        raw["Departure Delay in Minutes"],
        errors="coerce",
    )

    arr_raw = pd.to_numeric(
        raw["Arrival Delay in Minutes"],
        errors="coerce",
    )

    dep = dep_raw.fillna(0).clip(lower=0)
    arr = arr_raw.fillna(0).clip(lower=0)

    def cut_string(
        values: pd.Series,
        bins: list,
    ) -> pd.Series:
        return (
            pd.cut(
                values,
                bins=bins,
                labels=False,
            )
            .astype("string")
            .fillna(MISSING)
            .astype(str)
        )

    cat_engineered["age_group"] = cut_string(
        age,
        [
            -np.inf, 18, 25, 35, 45, 55, 65,
            np.inf,
        ],
    )

    cat_engineered["distance_group"] = cut_string(
        distance,
        [
            -np.inf, 250, 500, 750, 1000,
            1500, 2000, 3000, 5000, np.inf,
        ],
    )

    cat_engineered["departure_delay_group"] = cut_string(
        dep_raw,
        [
            -np.inf, 0, 5, 15, 30, 60, 120,
            180, np.inf,
        ],
    )

    cat_engineered["arrival_delay_group"] = cut_string(
        arr_raw,
        [
            -np.inf, 0, 5, 15, 30, 60, 120,
            180, np.inf,
        ],
    )

    # -------------------------------------------------------------------------
    # E. Delay structure
    # -------------------------------------------------------------------------

    num_eng["arrival_delay_missing"] = (
        arr_raw.isna().astype(float)
    )

    num_eng["departure_delay_missing"] = (
        dep_raw.isna().astype(float)
    )

    num_eng["total_delay"] = dep + arr

    num_eng["arrival_minus_departure_delay"] = (
        arr - dep
    )

    num_eng["absolute_delay_difference"] = (
        (arr - dep).abs()
    )

    num_eng["log1p_departure_delay"] = (
        np.log1p(dep)
    )

    num_eng["log1p_arrival_delay"] = (
        np.log1p(arr)
    )

    num_eng["departure_delayed"] = (
        dep > 0
    ).astype(float)

    num_eng["arrival_delayed"] = (
        arr > 0
    ).astype(float)

    num_eng["both_delayed"] = (
        (dep > 0) & (arr > 0)
    ).astype(float)

    # -------------------------------------------------------------------------
    # F. Passenger segment
    # -------------------------------------------------------------------------

    segment = (
        clean_text(raw["Type of Travel"])
        + "|"
        + clean_text(raw["Customer Type"])
        + "|"
        + clean_text(raw["Class"])
    )

    cat_engineered["traveler_segment"] = segment

    # -------------------------------------------------------------------------
    # G. Target-encoding keys
    #
    # TargetEncoder performs inner cross-fitting on fitting rows.
    # All columns here are categorical keys represented as strings.
    # -------------------------------------------------------------------------

    for col in CAT_COLS:
        te[f"raw__{col}"] = clean_text(
            raw[col]
        ).astype(object)

    for col in NUMERIC_COLS:
        te[f"raw__{col}"] = numeric_key(
            raw[col]
        ).astype(object)

    for col in RATING_COLS:
        te[f"cond__{col}__segment"] = (
            numeric_key(raw[col])
            + "|"
            + segment
        ).astype(object)

    distance_key = numeric_key(
        raw["Flight Distance"]
    )

    te["distance__class"] = (
        distance_key
        + "|"
        + clean_text(raw["Class"])
    ).astype(object)

    te["distance__travel"] = (
        distance_key
        + "|"
        + clean_text(raw["Type of Travel"])
    ).astype(object)

    te["distance__customer"] = (
        distance_key
        + "|"
        + clean_text(raw["Customer Type"])
    ).astype(object)

    te["distance_250_bin"] = (
        np.floor(distance / 250)
        .astype("Int64")
        .astype("string")
        .fillna(MISSING)
        .astype(object)
    )

    te["age_5_bin"] = (
        np.floor(age / 5)
        .astype("Int64")
        .astype("string")
        .fillna(MISSING)
        .astype(object)
    )

    te["traveler_segment"] = segment.astype(object)

    return {
        "raw": raw,
        "numeric_base": num,
        "numeric_engineered": num_eng,
        "categorical_base": cat_base,
        "categorical_engineered": cat_engineered,
        "te_keys": te,
    }


# =============================================================================
# 5. EXACT-DISTANCE PROFILE FEATURES
# =============================================================================

def distance_key(df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(
            df["Flight Distance"],
            errors="coerce",
        )
        .fillna(-999999.0)
    )


def make_distance_profile(
    reference: pd.DataFrame,
    query: pd.DataFrame,
) -> pd.DataFrame:
    """
    Exact Flight Distance acts only as a route proxy.
    The dataset does not contain real origin/destination airport IDs.

    The profile uses reference-row covariates only; no satisfaction labels
    enter this calculation. Group means are shrunk toward global means.
    """
    ref = reference.reset_index(drop=True).copy()
    qry = query.reset_index(drop=True).copy()

    ref_key = distance_key(ref)
    query_key = distance_key(qry)

    out = pd.DataFrame(index=qry.index)

    # Relative frequency instead of raw count reduces scale differences
    # between profile reference sets of different sizes.
    counts = ref_key.value_counts(dropna=False)
    count_share = counts / max(len(ref_key), 1)

    fallback_frequency = 1.0 / max(len(ref_key), 1)

    out["profile_distance_log_frequency"] = np.log1p(
        1_000_000.0
        * query_key.map(count_share)
        .fillna(fallback_frequency)
        .to_numpy(dtype=np.float64)
    )

    # Numeric profiles
    ref_measures = pd.DataFrame(index=ref.index)
    query_measures = pd.DataFrame(index=qry.index)

    for col in PROFILE_MEASURES:
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

        query_measures[col] = (
            pd.to_numeric(
                qry[col],
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
        )[PROFILE_MEASURES]
        .agg(["mean", "std", "count"])
    )

    prior_strength = 20.0

    for col in PROFILE_MEASURES:
        values = ref_measures[col]

        global_mean = (
            float(values.mean())
            if values.notna().any()
            else 0.0
        )

        global_std = (
            float(values.std())
            if values.notna().sum() > 1
            else 0.0
        )

        group_mean = grouped[(col, "mean")]
        group_std = grouped[(col, "std")]
        group_n = grouped[(col, "count")]

        n = (
            query_key
            .map(group_n)
            .fillna(0)
            .to_numpy(dtype=np.float64)
        )

        mapped_mean = (
            query_key
            .map(group_mean)
            .to_numpy(dtype=np.float64)
        )

        mapped_mean = np.nan_to_num(
            mapped_mean,
            nan=global_mean,
        )

        # Shrink sparse groups toward the global mean.
        smoothed_mean = (
            mapped_mean * n
            + global_mean * prior_strength
        ) / (n + prior_strength)

        smoothed_std = (
            query_key
            .map(group_std)
            .fillna(global_std)
            .to_numpy(dtype=np.float64)
        )

        safe_name = (
            col.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
        )

        out[f"profile_{safe_name}_mean"] = smoothed_mean
        out[f"profile_{safe_name}_std"] = smoothed_std

        out[f"profile_{safe_name}_residual"] = (
            query_measures[col].to_numpy(dtype=np.float64)
            - smoothed_mean
        )

    # Passenger composition at each exact distance.
    for col, vocabulary in PROFILE_VOCAB.items():
        ref_values = clean_text(ref[col])

        for category in vocabulary:
            indicator = (
                ref_values == category
            ).astype(float)

            temp = pd.DataFrame({
                "key": ref_key,
                "value": indicator,
            })

            stats = (
                temp
                .groupby("key", dropna=False)["value"]
                .agg(["sum", "count"])
            )

            global_share = (
                float(indicator.mean())
                if len(indicator)
                else 0.0
            )

            sums = (
                query_key
                .map(stats["sum"])
                .fillna(0)
                .to_numpy(dtype=np.float64)
            )

            ns = (
                query_key
                .map(stats["count"])
                .fillna(0)
                .to_numpy(dtype=np.float64)
            )

            share = (
                sums + prior_strength * global_share
            ) / (
                ns + prior_strength
            )

            safe_col = col.lower().replace(" ", "_")
            safe_category = category.lower().replace(" ", "_")

            out[
                f"profile_share_{safe_col}_{safe_category}"
            ] = share

    out = out.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return out.reset_index(drop=True)


def crossfit_distance_profiles(
    raw_fit: pd.DataFrame,
    y_fit: np.ndarray,
) -> pd.DataFrame:
    """
    Build training-row profiles from other fitting rows only.
    """
    skf = StratifiedKFold(
        n_splits=PROFILE_FOLDS,
        shuffle=True,
        random_state=SEED + 700,
    )

    result = None

    for fold, (ref_idx, query_idx) in enumerate(
        skf.split(
            np.zeros(len(y_fit)),
            y_fit,
        ),
        start=1,
    ):
        part = make_distance_profile(
            raw_fit.iloc[ref_idx],
            raw_fit.iloc[query_idx],
        )

        if result is None:
            result = pd.DataFrame(
                index=np.arange(len(raw_fit)),
                columns=part.columns,
                dtype=np.float32,
            )

        result.iloc[query_idx, :] = (
            part.to_numpy(dtype=np.float32)
        )

        log(
            f"  distance profile cross-fit "
            f"{fold}/{PROFILE_FOLDS} complete"
        )

    if result is None or result.isna().all(axis=None):
        raise RuntimeError(
            "Distance-profile cross-fitting failed."
        )

    return result.reset_index(drop=True)


# =============================================================================
# 6. TARGET ENCODING AND FREQUENCY FEATURES
# =============================================================================

def fit_target_encoding(
    te_fit_keys: pd.DataFrame,
    te_val_keys: pd.DataFrame,
    y_fit: np.ndarray,
):
    """
    fit_transform cross-fits encodings for fitting rows.
    Validation rows use a mapping estimated from fitting rows only.
    """
    encoder = TargetEncoder(
        target_type="binary",
        smooth="auto",
        cv=5,
        shuffle=True,
        random_state=SEED + 900,
    )

    fit_values = encoder.fit_transform(
        te_fit_keys,
        y_fit,
    ).astype(np.float32)

    val_values = encoder.transform(
        te_val_keys
    ).astype(np.float32)

    columns = [
        f"te__{col}"
        for col in te_fit_keys.columns
    ]

    return (
        pd.DataFrame(
            fit_values,
            columns=columns,
        ),
        pd.DataFrame(
            val_values,
            columns=columns,
        ),
    )


def fit_frequency_features(
    te_fit_keys: pd.DataFrame,
    te_val_keys: pd.DataFrame,
):
    """
    Label-free frequencies estimated from fitting rows only.
    """
    selected = [
        "raw__Gender",
        "raw__Customer Type",
        "raw__Type of Travel",
        "raw__Class",
        "raw__Age",
        "raw__Flight Distance",
        "raw__Departure Delay in Minutes",
        "raw__Arrival Delay in Minutes",
        "distance__class",
        "distance__travel",
        "distance__customer",
        "cond__Inflight wifi service__segment",
        "cond__Online boarding__segment",
        "cond__Seat comfort__segment",
        "distance_250_bin",
        "age_5_bin",
        "traveler_segment",
    ]

    selected = [
        col for col in selected
        if col in te_fit_keys.columns
    ]

    out_fit = pd.DataFrame()
    out_val = pd.DataFrame()

    for col in selected:
        fit_values = te_fit_keys[col].astype(str)
        val_values = te_val_keys[col].astype(str)

        freq_map = fit_values.value_counts(
            normalize=True,
            dropna=False,
        )

        fallback = 1.0 / max(len(fit_values), 1)

        fit_freq = (
            fit_values
            .map(freq_map)
            .fillna(fallback)
            .to_numpy(dtype=np.float64)
        )

        val_freq = (
            val_values
            .map(freq_map)
            .fillna(fallback)
            .to_numpy(dtype=np.float64)
        )

        name = f"freq__{col}"

        out_fit[name] = np.log1p(
            1_000_000.0 * fit_freq
        ).astype(np.float32)

        out_val[name] = np.log1p(
            1_000_000.0 * val_freq
        ).astype(np.float32)

    return out_fit, out_val


# =============================================================================
# 7. MATRIX PREPARATION
# =============================================================================

def encode_categorical(
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
):
    fit_columns = []
    val_columns = []
    cardinalities = []

    for col in fit_df.columns:
        fit_values = clean_text(fit_df[col])
        val_values = clean_text(val_df[col])

        vocabulary = sorted(
            fit_values.unique().tolist()
        )

        mapping = {
            value: idx + 1
            for idx, value in enumerate(vocabulary)
        }

        fit_columns.append(
            fit_values
            .map(mapping)
            .fillna(0)
            .to_numpy(dtype=np.int64)
        )

        val_columns.append(
            val_values
            .map(mapping)
            .fillna(0)
            .to_numpy(dtype=np.int64)
        )

        cardinalities.append(
            len(vocabulary) + 1
        )

    X_fit_cat = np.column_stack(
        fit_columns
    ).astype(np.int64, copy=True)

    X_val_cat = np.column_stack(
        val_columns
    ).astype(np.int64, copy=True)

    return (
        X_fit_cat,
        X_val_cat,
        cardinalities,
    )


def encode_numeric(
    frames_fit: List[pd.DataFrame],
    frames_val: List[pd.DataFrame],
):
    fit = pd.concat(
        [
            frame.reset_index(drop=True)
            for frame in frames_fit
        ],
        axis=1,
    )

    val = pd.concat(
        [
            frame.reset_index(drop=True)
            for frame in frames_val
        ],
        axis=1,
    )

    if fit.columns.duplicated().any():
        raise ValueError(
            "Duplicated numeric feature names detected."
        )

    if val.columns.duplicated().any():
        raise ValueError(
            "Duplicated validation feature names detected."
        )

    val = val.reindex(
        columns=fit.columns
    )

    fit = (
        fit
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    val = (
        val
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    # Fit-fold medians only.
    medians = (
        fit
        .median(axis=0, numeric_only=True)
        .fillna(0.0)
    )

    fit = fit.fillna(medians).fillna(0.0)
    val = val.fillna(medians).fillna(0.0)

    X_fit = fit.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    X_val = val.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    if (
        not np.isfinite(X_fit).all()
        or not np.isfinite(X_val).all()
    ):
        raise ValueError(
            "Non-finite numeric values remain."
        )

    # Drop constant columns based on fitting rows only.
    keep = np.ptp(
        X_fit,
        axis=0,
    ) > 1e-12

    X_fit = X_fit[:, keep]
    X_val = X_val[:, keep]

    names = (
        np.asarray(fit.columns, dtype=object)[keep]
        .tolist()
    )

    if X_fit.shape[1] == 0:
        raise ValueError(
            "No numeric features remain."
        )

    return X_fit, X_val, names


# =============================================================================
# 8. TABM TRAINING
# =============================================================================

def create_model(
    X_num: np.ndarray,
    cat_cardinalities: List[int],
):
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

    numeric_embeddings = PiecewiseLinearEmbeddings(
        bins,
        d_embedding=PLE_D_EMBEDDING,
        activation=False,
        version="B",
    )

    model = TabM.make(
        n_num_features=X_num.shape[1],
        cat_cardinalities=cat_cardinalities,
        num_embeddings=numeric_embeddings,
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
) -> np.ndarray:
    model.eval()

    parts = []

    for start in range(
        0,
        len(X_num),
        BATCH_SIZE,
    ):
        end = min(
            start + BATCH_SIZE,
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

        parts.append(probabilities)

    pred = np.concatenate(parts)

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
) -> float:
    model.train()

    order = np.random.permutation(
        len(y_array)
    )

    total_loss = 0.0
    total_count = 0

    for start in range(
        0,
        len(order),
        BATCH_SIZE,
    ):
        idx = order[
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

        # TabM output shape: batch_size x K
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

        n = len(idx)

        total_loss += (
            float(loss.item()) * n
        )

        total_count += n

    return total_loss / max(
        total_count,
        1,
    )


def train_variant(
    name,
    X_fit_num,
    X_fit_cat,
    y_fit,
    X_val_num,
    X_val_cat,
    y_val,
    cat_cardinalities,
):
    log("\n" + "=" * 88)
    log(f"TRAINING {name}")
    log(
        f"Numeric features={X_fit_num.shape[1]}; "
        f"categorical features={X_fit_cat.shape[1]}"
    )

    # Same initial seed for controlled comparisons.
    set_seed(SEED)

    model = create_model(
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
    stale = 0
    best_state = None

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

        auc = roc_auc_score(
            y_val,
            val_pred,
        )

        if auc > best_auc + 1e-5:
            best_auc = auc
            best_epoch = epoch
            stale = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1

        log(
            f"{name} | Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"loss={loss:.6f} | "
            f"val AUC={auc:.9f} | "
            f"best={best_auc:.9f}@{best_epoch}"
        )

        if (
            epoch >= MIN_EPOCHS
            and stale >= PATIENCE
        ):
            log(
                f"{name}: early stopping at epoch {epoch}."
            )
            break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    pred = predict(
        model,
        X_val_num,
        X_val_cat,
    )

    final_auc = roc_auc_score(
        y_val,
        pred,
    )

    del model, optimizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pred, best_epoch, final_auc


# =============================================================================
# 9. MAIN EXPERIMENT
# =============================================================================

def main():
    set_seed(SEED)

    log("=" * 100)
    log("TABM FEATURE ABLATION V3: M0 / M1 / M2 / M3")
    log(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        log(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    require_file(TRAIN_PATH)
    require_file(TEST_PATH)

    train = pd.read_csv(TRAIN_PATH)

    # This is a feature-screening script, not a submission script.
    # Verify that test.csv has the expected columns.
    test_head = pd.read_csv(
        TEST_PATH,
        nrows=5,
    )

    required_train = (
        [ID_COL, TARGET]
        + NUMERIC_COLS
        + CAT_COLS
    )

    missing_train = (
        set(required_train)
        - set(train.columns)
    )

    if missing_train:
        raise ValueError(
            f"Missing train columns: {sorted(missing_train)}"
        )

    required_test = (
        [ID_COL]
        + NUMERIC_COLS
        + CAT_COLS
    )

    missing_test = (
        set(required_test)
        - set(test_head.columns)
    )

    if missing_test:
        raise ValueError(
            f"Missing test columns: {sorted(missing_test)}"
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
            "Duplicate training IDs detected."
        )

    if len(np.unique(y)) != 2:
        raise ValueError(
            "Expected binary satisfaction target."
        )

    log(
        f"Train shape={train.shape}; "
        f"positive rate={y.mean():.6f}"
    )

    # -------------------------------------------------------------------------
    # A. Load Champion OOF predictions
    #
    # This comparison is properly fold-aligned only if the source OOF files
    # used the same StratifiedKFold(5, shuffle=True, random_state=42) split.
    # Both current scripts are expected to use that split.
    # -------------------------------------------------------------------------

    lgb_oof = read_oof(
        LGBM_OOF_PATH,
        ids,
        y,
        "Conditional-TE LGBM",
    )

    mlp_oof = read_oof(
        REALMLP_OOF_PATH,
        ids,
        y,
        "Lean RealMLP V1",
    )

    champion_full_oof = (
        0.40 * normalized_rank(lgb_oof)
        + 0.60 * normalized_rank(mlp_oof)
    )

    log(
        "Rebuilt Champion full OOF AUC="
        f"{roc_auc_score(y, champion_full_oof):.9f}"
    )

    # -------------------------------------------------------------------------
    # B. Use the first fold from the same 5-fold split
    # -------------------------------------------------------------------------

    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    folds = list(
        splitter.split(
            np.zeros(len(y)),
            y,
        )
    )

    fit_idx, val_idx = folds[0]

    y_fit = y[fit_idx]
    y_val = y[val_idx]

    log(
        f"Using fold 1/{N_FOLDS}: "
        f"fit={len(fit_idx):,}; "
        f"validation={len(val_idx):,}"
    )

    log(
        "The Champion comparison assumes its OOF files "
        "used this exact fold split."
    )

    # -------------------------------------------------------------------------
    # C. Build row-level views once
    # -------------------------------------------------------------------------

    log("Building row-level features...")

    row_fit = build_row_features(
        train.iloc[fit_idx]
    )

    row_val = build_row_features(
        train.iloc[val_idx]
    )

    raw_fit = row_fit["raw"]
    raw_val = row_val["raw"]

    base_num_fit = row_fit["numeric_base"]
    base_num_val = row_val["numeric_base"]

    eng_num_fit = row_fit["numeric_engineered"]
    eng_num_val = row_val["numeric_engineered"]

    cat_base_fit = row_fit["categorical_base"]
    cat_base_val = row_val["categorical_base"]

    cat_eng_fit = row_fit["categorical_engineered"]
    cat_eng_val = row_val["categorical_engineered"]

    # -------------------------------------------------------------------------
    # D. M2/M3: cross-fitted training profiles
    # -------------------------------------------------------------------------

    log(
        "Building exact-distance profile features for M2/M3..."
    )

    profile_fit = crossfit_distance_profiles(
        raw_fit,
        y_fit,
    )

    profile_val = make_distance_profile(
        raw_fit,
        raw_val,
    )

    # -------------------------------------------------------------------------
    # E. M3: target encoding and frequency features
    # -------------------------------------------------------------------------

    log(
        "Building M3 target encodings and frequency features..."
    )

    te_fit, te_val = fit_target_encoding(
        row_fit["te_keys"],
        row_val["te_keys"],
        y_fit,
    )

    freq_fit, freq_val = fit_frequency_features(
        row_fit["te_keys"],
        row_val["te_keys"],
    )

    # -------------------------------------------------------------------------
    # F. Define feature ablations
    # -------------------------------------------------------------------------

    variants = {
        "M0_raw": {
            "num_fit": [
                base_num_fit,
            ],
            "num_val": [
                base_num_val,
            ],
            "cat_fit": cat_base_fit,
            "cat_val": cat_base_val,
        },

        "M1_rating_delay": {
            "num_fit": [
                base_num_fit,
                eng_num_fit,
            ],
            "num_val": [
                base_num_val,
                eng_num_val,
            ],
            "cat_fit": pd.concat(
                [
                    cat_base_fit,
                    cat_eng_fit,
                ],
                axis=1,
            ),
            "cat_val": pd.concat(
                [
                    cat_base_val,
                    cat_eng_val,
                ],
                axis=1,
            ),
        },

        "M2_plus_distance_profile": {
            "num_fit": [
                base_num_fit,
                eng_num_fit,
                profile_fit,
            ],
            "num_val": [
                base_num_val,
                eng_num_val,
                profile_val,
            ],
            "cat_fit": pd.concat(
                [
                    cat_base_fit,
                    cat_eng_fit,
                ],
                axis=1,
            ),
            "cat_val": pd.concat(
                [
                    cat_base_val,
                    cat_eng_val,
                ],
                axis=1,
            ),
        },

        "M3_plus_TE_frequency": {
            "num_fit": [
                base_num_fit,
                eng_num_fit,
                profile_fit,
                te_fit,
                freq_fit,
            ],
            "num_val": [
                base_num_val,
                eng_num_val,
                profile_val,
                te_val,
                freq_val,
            ],
            "cat_fit": pd.concat(
                [
                    cat_base_fit,
                    cat_eng_fit,
                ],
                axis=1,
            ),
            "cat_val": pd.concat(
                [
                    cat_base_val,
                    cat_eng_val,
                ],
                axis=1,
            ),
        },
    }

    # -------------------------------------------------------------------------
    # G. Champion validation baseline
    #
    # Rank both component predictions INSIDE the validation fold so their rank
    # scales match the rank of the TabM predictions being blended.
    # -------------------------------------------------------------------------

    champion_val_rank = (
        0.40 * normalized_rank(lgb_oof[val_idx])
        + 0.60 * normalized_rank(mlp_oof[val_idx])
    )

    champion_val_auc = roc_auc_score(
        y_val,
        champion_val_rank,
    )

    log(
        f"Champion fold-1 OOF rank-blend AUC="
        f"{champion_val_auc:.9f}"
    )

    # -------------------------------------------------------------------------
    # H. Run ablations and screen rank blends
    # -------------------------------------------------------------------------

    all_preds = pd.DataFrame({
        ID_COL: ids[val_idx],
        TARGET: y_val,
    })

    result_rows = []

    for name, config in variants.items():
        X_fit_num, X_val_num, numeric_names = encode_numeric(
            config["num_fit"],
            config["num_val"],
        )

        X_fit_cat, X_val_cat, cat_cards = encode_categorical(
            config["cat_fit"],
            config["cat_val"],
        )

        log(
            f"\n{name}: "
            f"numeric={X_fit_num.shape[1]}, "
            f"categorical={X_fit_cat.shape[1]}"
        )

        pred, best_epoch, auc = train_variant(
            name,
            X_fit_num,
            X_fit_cat,
            y_fit,
            X_val_num,
            X_val_cat,
            y_val,
            cat_cards,
        )

        all_preds[f"pred__{name}"] = pred

        pred_rank = normalized_rank(pred)

        spearman_rho = pd.Series(
            champion_val_rank
        ).corr(
            pd.Series(pred_rank),
            method="spearman",
        )

        result_rows.append({
            "variant": name,
            "n_numeric": X_fit_num.shape[1],
            "n_categorical": X_fit_cat.shape[1],
            "best_epoch": best_epoch,
            "holdout_auc": auc,
            "champion_fold_auc": champion_val_auc,
            "spearman_with_champion": spearman_rho,
        })

        # Rank-blend diagnostics on the same validation fold.
        for tabm_weight in BLEND_WEIGHTS:
            champion_weight = 1.0 - tabm_weight

            blend = (
                champion_weight * champion_val_rank
                + tabm_weight * pred_rank
            )

            blend_auc = roc_auc_score(
                y_val,
                blend,
            )

            result_rows.append({
                "variant": f"{name}__blend",
                "tabm_weight": tabm_weight,
                "champion_weight": champion_weight,
                "n_numeric": X_fit_num.shape[1],
                "n_categorical": X_fit_cat.shape[1],
                "best_epoch": best_epoch,
                "holdout_auc": blend_auc,
                "champion_fold_auc": champion_val_auc,
                "gain_vs_champion": (
                    blend_auc - champion_val_auc
                ),
                "spearman_with_champion": spearman_rho,
            })

        # Save incrementally in case a later run fails.
        all_preds.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "tabm_v3_fold1_predictions.csv",
            ),
            index=False,
            encoding="utf-8-sig",
        )

        pd.DataFrame(result_rows).to_csv(
            os.path.join(
                OUTPUT_DIR,
                "tabm_v3_results_partial.csv",
            ),
            index=False,
            encoding="utf-8-sig",
        )

        del X_fit_num, X_val_num
        del X_fit_cat, X_val_cat
        del pred

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # I. Save complete results
    # -------------------------------------------------------------------------

    results = pd.DataFrame(result_rows)

    results_path = os.path.join(
        OUTPUT_DIR,
        "tabm_v3_ablation_results.csv",
    )

    predictions_path = os.path.join(
        OUTPUT_DIR,
        "tabm_v3_fold1_predictions.csv",
    )

    results.to_csv(
        results_path,
        index=False,
        encoding="utf-8-sig",
    )

    all_preds.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # J. Summary
    # -------------------------------------------------------------------------

    log("\n" + "=" * 100)
    log("FINAL SCREENING SUMMARY")
    log("=" * 100)

    model_rows = results[
        ~results["variant"].str.endswith("__blend")
    ].copy()

    log(
        model_rows[
            [
                "variant",
                "n_numeric",
                "n_categorical",
                "best_epoch",
                "holdout_auc",
                "spearman_with_champion",
            ]
        ].to_string(index=False)
    )

    blend_rows = results[
        results["variant"].str.endswith("__blend")
    ].copy()

    best_blends = (
        blend_rows
        .sort_values(
            "holdout_auc",
            ascending=False,
        )
        .groupby("variant", as_index=False)
        .head(1)
    )

    log("\nBest blend per feature variant (screening only):")

    log(
        best_blends[
            [
                "variant",
                "tabm_weight",
                "holdout_auc",
                "gain_vs_champion",
            ]
        ].to_string(index=False)
    )

    log(f"\nSaved results: {results_path}")
    log(f"Saved holdout predictions: {predictions_path}")

    log(
        "Do not use the best weight directly as a final submission choice. "
        "Confirm promising variants on another fold or full OOF."
    )


if __name__ == "__main__":
    main()