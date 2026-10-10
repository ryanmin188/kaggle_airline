# -*- coding: utf-8 -*-
"""
TabM M3 independent fold confirmation
Kaggle Playground Series S6E10 - Airline Passenger Satisfaction

Features:
    Original numeric + categorical features
    Rating and delay features
    Cross-fitted exact-distance profile
    Cross-fitted target encodings
    Frequency features

Validation:
    Fold 2 from StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )

Training:
    Fixed 20 epochs, no early stopping / no epoch selection.
    Fixed blend: 80% Champion + 20% TabM M3.

IMPORTANT:
    Existing LGBM and RealMLP OOF must use the same
    five-fold split for the OOF blend comparison to be valid.

This script does not make a test-set submission.
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
# 1. PATHS
# =============================================================================

DATA_DIR = r"D:\kaggle\airline\data"

TRAIN_PATH = os.path.join(
    DATA_DIR,
    "train.csv",
)

LGBM_OOF_PATH = (
    r"D:\kaggle\airline\conditional_te_experiment"
    r"\conditional_te_lgbm_oof.csv"
)

REALMLP_OOF_PATH = (
    r"D:\kaggle\airline\lean_realmlp_blend"
    r"\lean_realmlp_oof.csv"
)

OUTPUT_DIR = (
    r"D:\kaggle\airline\tabm_m3_fold2_confirm"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# =============================================================================
# 2. CONFIGURATION
# =============================================================================

ID_COL = "id"
TARGET = "satisfaction"

SEED = 42
N_FOLDS = 5
TARGET_PROFILE_FOLDS = 4

# Fixed confirmation protocol
CONFIRM_FOLD = 2
FIXED_EPOCHS = 20
TABM_BLEND_WEIGHT = 0.20
CHAMPION_BLEND_WEIGHT = 0.80

# Training
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0003
BATCH_SIZE = 2048

# TabM architecture
TABM_K = 16
TABM_BLOCKS = 2
TABM_WIDTH = 256
TABM_DROPOUT = 0.10

# Numerical embeddings
PLE_N_BINS = 32
PLE_D_EMBEDDING = 8
MAX_BIN_SAMPLE = 100_000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
# 3. FEATURES
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

PROFILE_MEASURES = (
    RATING_COLS
    + [
        "Age",
        "Departure Delay in Minutes",
        "Arrival Delay in Minutes",
    ]
)

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

def log(message):
    print(message, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_file(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )


def clean_text(series):
    return (
        series
        .astype("string")
        .fillna(MISSING)
        .astype(str)
        .str.strip()
    )


def numeric_key(series):
    """
    Stable string key:
        numeric 5 and 5.0 become the same category.
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


def normalized_rank(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Predictions contain NaN or infinity."
        )

    return (
        rankdata(
            values,
            method="average",
        )
        / len(values)
    )


def load_oof(
    path,
    train_ids,
    y,
    label,
):
    """
    Align OOF data to train.csv using id.
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
            f"{label} must contain {sorted(required)}. "
            f"Found: {df.columns.tolist()}"
        )

    df[ID_COL] = pd.to_numeric(
        df[ID_COL],
        errors="raise",
    )

    if df[ID_COL].duplicated().any():
        raise ValueError(
            f"{label} contains duplicate IDs."
        )

    if set(df[ID_COL]) != set(train_ids):
        raise ValueError(
            f"{label} IDs do not match train.csv."
        )

    aligned = (
        df.set_index(ID_COL)
        .loc[train_ids]
        .reset_index()
    )

    aligned_y = aligned[TARGET].to_numpy(
        dtype=np.int8,
    )

    if not np.array_equal(
        aligned_y,
        y.astype(np.int8),
    ):
        raise ValueError(
            f"{label}: target alignment mismatch."
        )

    pred = aligned["prediction"].to_numpy(
        dtype=np.float64,
    )

    if not np.isfinite(pred).all():
        raise ValueError(
            f"{label} predictions are not finite."
        )

    log(
        f"{label} full OOF AUC = "
        f"{roc_auc_score(y, pred):.9f}"
    )

    return pred


# =============================================================================
# 5. ROW-LEVEL FEATURES
# =============================================================================

def build_row_features(df):
    """
    All features in this function are target-independent.

    Returns:
        raw
        numeric_base
        numeric_engineered
        categorical_base
        categorical_engineered
        te_keys
    """
    raw = df.reset_index(
        drop=True
    ).copy()

    num = pd.DataFrame(
        index=raw.index
    )

    cat_base = pd.DataFrame(
        index=raw.index
    )

    cat_eng = pd.DataFrame(
        index=raw.index
    )

    te = pd.DataFrame(
        index=raw.index
    )

    # -------------------------------------------------------------------------
    # A. Original numeric columns
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
    # B. Original categorical columns
    # -------------------------------------------------------------------------

    for col in CAT_COLS:
        cat_base[col] = clean_text(
            raw[col]
        )

    # -------------------------------------------------------------------------
    # C. Rating summaries
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

    num_eng["rating_max"] = (
        ratings.max(axis=1)
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
        (ratings == 0).sum(axis=1).astype(float)
    )

    num_eng["rating_nonzero_count"] = (
        (ratings > 0).sum(axis=1).astype(float)
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

    # -------------------------------------------------------------------------
    # D. Categorical views for discrete ratings
    # -------------------------------------------------------------------------

    for col in RATING_COLS:
        cat_eng[
            f"rating_cat__{col}"
        ] = clean_text(
            raw[col]
        )

    # -------------------------------------------------------------------------
    # E. Coarse bins instead of categories for all exact numeric values
    # -------------------------------------------------------------------------

    def cut_string(values, bins):
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

    cat_eng["age_group"] = cut_string(
        age,
        [
            -np.inf, 18, 25, 35, 45, 55, 65,
            np.inf,
        ],
    )

    cat_eng["distance_group"] = cut_string(
        distance,
        [
            -np.inf, 250, 500, 750, 1000,
            1500, 2000, 3000, 5000, np.inf,
        ],
    )

    cat_eng["departure_delay_group"] = cut_string(
        dep_raw,
        [
            -np.inf, 0, 5, 15, 30, 60, 120,
            180, np.inf,
        ],
    )

    cat_eng["arrival_delay_group"] = cut_string(
        arr_raw,
        [
            -np.inf, 0, 5, 15, 30, 60, 120,
            180, np.inf,
        ],
    )

    # -------------------------------------------------------------------------
    # F. Delay structures
    # -------------------------------------------------------------------------

    dep = dep_raw.fillna(0).clip(lower=0)
    arr = arr_raw.fillna(0).clip(lower=0)

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
        ((dep > 0) & (arr > 0)).astype(float)
    )

    # -------------------------------------------------------------------------
    # G. Passenger segment
    # -------------------------------------------------------------------------

    segment = (
        clean_text(raw["Type of Travel"])
        + "|"
        + clean_text(raw["Customer Type"])
        + "|"
        + clean_text(raw["Class"])
    )

    cat_eng["traveler_segment"] = segment

    # -------------------------------------------------------------------------
    # H. Target-encoding keys for 21 original predictors
    # -------------------------------------------------------------------------

    for col in CAT_COLS:
        te[f"raw__{col}"] = clean_text(
            raw[col]
        ).astype(object)

    for col in NUMERIC_COLS:
        te[f"raw__{col}"] = numeric_key(
            raw[col]
        ).astype(object)

    # -------------------------------------------------------------------------
    # I. Service score conditional TE
    # -------------------------------------------------------------------------

    for col in RATING_COLS:
        te[
            f"cond__{col}__segment"
        ] = (
            numeric_key(raw[col])
            + "|"
            + segment
        ).astype(object)

    # -------------------------------------------------------------------------
    # J. Exact distance × passenger type
    # -------------------------------------------------------------------------

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

    te["traveler_segment"] = (
        segment.astype(object)
    )

    return {
        "raw": raw,
        "numeric_base": num,
        "numeric_engineered": num_eng,
        "categorical_base": cat_base,
        "categorical_engineered": cat_eng,
        "te_keys": te,
    }


# =============================================================================
# 6. DISTANCE PROFILE FEATURES
# =============================================================================

def get_distance_key(df):
    return (
        pd.to_numeric(
            df["Flight Distance"],
            errors="coerce",
        )
        .fillna(-999999.0)
    )


def make_distance_profile(
    reference,
    query,
):
    """
    Build an unsupervised covariate profile.

    Flight Distance is only a proxy because the dataset does not provide
    actual origin/destination airport IDs.
    """
    ref = reference.reset_index(
        drop=True
    ).copy()

    qry = query.reset_index(
        drop=True
    ).copy()

    ref_key = get_distance_key(ref)
    query_key = get_distance_key(qry)

    out = pd.DataFrame(
        index=qry.index
    )

    # Relative frequency, not raw count
    counts = ref_key.value_counts(
        dropna=False
    )

    count_share = counts / max(
        len(ref_key),
        1,
    )

    out["profile_distance_log_frequency"] = (
        np.log1p(
            1_000_000.0
            * query_key
            .map(count_share)
            .fillna(1.0 / max(len(ref_key), 1))
            .to_numpy(dtype=np.float64)
        )
    )

    ref_measures = pd.DataFrame(
        index=ref.index
    )

    query_measures = pd.DataFrame(
        index=qry.index
    )

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
        vals = ref_measures[col]

        global_mean = (
            float(vals.mean())
            if vals.notna().any()
            else 0.0
        )

        global_std = (
            float(vals.std())
            if vals.notna().sum() > 1
            else 0.0
        )

        mean_map = grouped[(col, "mean")]
        std_map = grouped[(col, "std")]
        count_map = grouped[(col, "count")]

        n = (
            query_key
            .map(count_map)
            .fillna(0)
            .to_numpy(dtype=np.float64)
        )

        mapped_mean = (
            query_key
            .map(mean_map)
            .to_numpy(dtype=np.float64)
        )

        mapped_mean = np.nan_to_num(
            mapped_mean,
            nan=global_mean,
        )

        smoothed_mean = (
            mapped_mean * n
            + global_mean * prior_strength
        ) / (
            n + prior_strength
        )

        smoothed_std = (
            query_key
            .map(std_map)
            .fillna(global_std)
            .to_numpy(dtype=np.float64)
        )

        safe_name = (
            col.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
        )

        out[f"profile_{safe_name}_mean"] = (
            smoothed_mean
        )

        out[f"profile_{safe_name}_std"] = (
            smoothed_std
        )

        out[f"profile_{safe_name}_residual"] = (
            query_measures[col].to_numpy(dtype=np.float64)
            - smoothed_mean
        )

    # Passenger composition by exact distance
    for col, vocabulary in PROFILE_VOCAB.items():
        ref_values = clean_text(
            ref[col]
        )

        temp = pd.DataFrame({
            "key": ref_key,
            "value": ref_values,
        })

        for category in vocabulary:
            temp_indicator = pd.DataFrame({
                "key": ref_key,
                "indicator": (
                    ref_values == category
                ).astype(float),
            })

            stats = (
                temp_indicator
                .groupby(
                    "key",
                    dropna=False,
                )["indicator"]
                .agg(["sum", "count"])
            )

            global_share = (
                float(
                    (ref_values == category).mean()
                )
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

            safe_col = col.lower().replace(
                " ",
                "_",
            )

            safe_category = category.lower().replace(
                " ",
                "_",
            )

            out[
                f"profile_share_{safe_col}_{safe_category}"
            ] = share

    return (
        out
        .replace([np.inf, -np.inf], np.nan)
        .reset_index(drop=True)
    )


def crossfit_distance_profiles(
    raw_fit,
    y_fit,
):
    """
    Generate fit-row profiles without each row contributing to its own profile.
    Labels are used only for stratified splitting, never for profile values.
    """
    skf = StratifiedKFold(
        n_splits=TARGET_PROFILE_FOLDS,
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
            f"Distance profile cross-fit "
            f"{fold}/{TARGET_PROFILE_FOLDS} complete"
        )

    if result is None or result.isna().to_numpy().all():
        raise RuntimeError(
            "Distance profile generation failed."
        )

    return result.reset_index(drop=True)


# =============================================================================
# 7. TARGET ENCODING AND FREQUENCY
# =============================================================================

def fit_target_encoding(
    fit_keys,
    val_keys,
    y_fit,
):
    encoder = TargetEncoder(
        target_type="binary",
        smooth="auto",
        cv=5,
        shuffle=True,
        random_state=SEED + 900,
    )

    # Cross-fitted encodings for fit rows
    fit_values = encoder.fit_transform(
        fit_keys,
        y_fit,
    ).astype(np.float32)

    # Validation rows transform with fit-fold labels only
    val_values = encoder.transform(
        val_keys
    ).astype(np.float32)

    columns = [
        f"te__{col}"
        for col in fit_keys.columns
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
    fit_keys,
    val_keys,
):
    """
    Frequency statistics are fitted on fit rows only.
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
        col
        for col in selected
        if col in fit_keys.columns
    ]

    fit_out = pd.DataFrame()
    val_out = pd.DataFrame()

    for col in selected:
        fit_values = fit_keys[col].astype(str)
        val_values = val_keys[col].astype(str)

        freq_map = fit_values.value_counts(
            normalize=True,
            dropna=False,
        )

        fallback = 1.0 / max(
            len(fit_values),
            1,
        )

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

        fit_out[name] = np.log1p(
            1_000_000.0 * fit_freq
        ).astype(np.float32)

        val_out[name] = np.log1p(
            1_000_000.0 * val_freq
        ).astype(np.float32)

    return fit_out, val_out


# =============================================================================
# 8. INPUT MATRIX PREPARATION
# =============================================================================

def encode_categorical(
    fit_df,
    val_df,
):
    fit_columns = []
    val_columns = []
    cardinalities = []

    for col in fit_df.columns:
        fit_values = clean_text(
            fit_df[col]
        )

        val_values = clean_text(
            val_df[col]
        )

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
    ).astype(
        np.int64,
        copy=True,
    )

    X_val_cat = np.column_stack(
        val_columns
    ).astype(
        np.int64,
        copy=True,
    )

    return (
        X_fit_cat,
        X_val_cat,
        cardinalities,
    )


def encode_numeric(
    fit_frames,
    val_frames,
):
    fit = pd.concat(
        [
            frame.reset_index(drop=True)
            for frame in fit_frames
        ],
        axis=1,
    )

    val = pd.concat(
        [
            frame.reset_index(drop=True)
            for frame in val_frames
        ],
        axis=1,
    )

    if fit.columns.duplicated().any():
        raise ValueError(
            "Duplicate numeric feature names."
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

    # Fit-fold medians only
    medians = (
        fit
        .median(axis=0, numeric_only=True)
        .fillna(0.0)
    )

    fit = fit.fillna(
        medians
    ).fillna(0.0)

    val = val.fillna(
        medians
    ).fillna(0.0)

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
            "Non-finite numeric model input."
        )

    # Drop constant columns based on fitting rows only
    keep = (
        np.ptp(X_fit, axis=0) > 1e-12
    )

    X_fit = X_fit[:, keep]
    X_val = X_val[:, keep]

    names = (
        np.asarray(
            fit.columns,
            dtype=object,
        )[keep]
        .tolist()
    )

    return X_fit, X_val, names


# =============================================================================
# 9. TABM MODEL
# =============================================================================

def create_model(
    X_num,
    cat_cardinalities,
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
    X_num,
    X_cat,
):
    model.eval()

    outputs = []

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

        outputs.append(
            probabilities
        )

    pred = np.concatenate(outputs)

    if not np.isfinite(pred).all():
        raise ValueError(
            "Non-finite TabM predictions."
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

        # TabM output: batch_size x K.
        # Compute loss for every ensemble member.
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

    return (
        total_loss
        / max(total_count, 1)
    )


# =============================================================================
# 10. MAIN: FOLD 2 ONLY, FIXED 20 EPOCHS
# =============================================================================

def main():
    set_seed(SEED)

    log("=" * 100)
    log("TABM M3 FOLD-2 CONFIRMATION")
    log(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        log(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    require_file(TRAIN_PATH)
    require_file(LGBM_OOF_PATH)
    require_file(REALMLP_OOF_PATH)

    train = pd.read_csv(
        TRAIN_PATH
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
            "Duplicate train IDs."
        )

    # -------------------------------------------------------------------------
    # A. Load current Champion OOF predictions
    # -------------------------------------------------------------------------

    lgb_oof = load_oof(
        LGBM_OOF_PATH,
        ids,
        y,
        "Conditional-TE LGBM",
    )

    mlp_oof = load_oof(
        REALMLP_OOF_PATH,
        ids,
        y,
        "Lean RealMLP V1",
    )

    # Verify the full-OOF champion benchmark
    champion_full_oof = (
        0.40 * normalized_rank(lgb_oof)
        + 0.60 * normalized_rank(mlp_oof)
    )

    log(
        "Rebuilt Champion full OOF AUC = "
        f"{roc_auc_score(y, champion_full_oof):.9f}"
    )

    # -------------------------------------------------------------------------
    # B. Reconstruct the same five-fold split
    #
    # Confirm that the LGBM and RealMLP OOF files were created with this split.
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

    fit_idx, val_idx = folds[
        CONFIRM_FOLD - 1
    ]

    y_fit = y[fit_idx]
    y_val = y[val_idx]

    log(
        f"Fold {CONFIRM_FOLD}/{N_FOLDS}: "
        f"fit={len(fit_idx):,}; "
        f"validation={len(val_idx):,}"
    )

    # -------------------------------------------------------------------------
    # C. Build target-independent row features
    # -------------------------------------------------------------------------

    log("Building M3 row-level features...")

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

    cat_fit = pd.concat(
        [
            row_fit["categorical_base"],
            row_fit["categorical_engineered"],
        ],
        axis=1,
    )

    cat_val = pd.concat(
        [
            row_val["categorical_base"],
            row_val["categorical_engineered"],
        ],
        axis=1,
    )

    # -------------------------------------------------------------------------
    # D. Cross-fitted distance profiles
    # -------------------------------------------------------------------------

    log(
        "Building cross-fitted exact-distance profiles..."
    )

    profile_fit = crossfit_distance_profiles(
        raw_fit,
        y_fit,
    )

    # The validation profile uses fit-fold covariates only.
    profile_val = make_distance_profile(
        raw_fit,
        raw_val,
    )

    # -------------------------------------------------------------------------
    # E. Target encoding and frequency
    # -------------------------------------------------------------------------

    log("Fitting M3 target encoding...")

    te_fit, te_val = fit_target_encoding(
        row_fit["te_keys"],
        row_val["te_keys"],
        y_fit,
    )

    log("Fitting M3 frequency features...")

    freq_fit, freq_val = fit_frequency_features(
        row_fit["te_keys"],
        row_val["te_keys"],
    )

    # -------------------------------------------------------------------------
    # F. Numeric and categorical matrices
    # -------------------------------------------------------------------------

    X_fit_num, X_val_num, num_names = encode_numeric(
        [
            base_num_fit,
            eng_num_fit,
            profile_fit,
            te_fit,
            freq_fit,
        ],
        [
            base_num_val,
            eng_num_val,
            profile_val,
            te_val,
            freq_val,
        ],
    )

    X_fit_cat, X_val_cat, cat_cardinalities = (
        encode_categorical(
            cat_fit,
            cat_val,
        )
    )

    log(
        f"M3 numeric features: {X_fit_num.shape[1]}"
    )

    log(
        f"M3 categorical features: {X_fit_cat.shape[1]}"
    )

    log(
        f"Max categorical cardinality: "
        f"{max(cat_cardinalities)}"
    )

    # -------------------------------------------------------------------------
    # G. Train exactly 20 epochs
    #
    # Do not inspect validation AUC during training.
    # No early stopping and no model selection on this confirmation fold.
    # -------------------------------------------------------------------------

    log(
        f"Training M3 for exactly {FIXED_EPOCHS} epochs..."
    )

    set_seed(SEED + 2000 + CONFIRM_FOLD)

    model = create_model(
        X_fit_num,
        cat_cardinalities,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(
        1,
        FIXED_EPOCHS + 1,
    ):
        loss = train_one_epoch(
            model,
            optimizer,
            X_fit_num,
            X_fit_cat,
            y_fit,
        )

        log(
            f"Epoch {epoch:02d}/{FIXED_EPOCHS} "
            f"| train loss={loss:.6f}"
        )

    # -------------------------------------------------------------------------
    # H. Evaluate on Fold 2 once, after training is finished
    # -------------------------------------------------------------------------

    log("Predicting fold-2 validation rows...")

    tabm_pred = predict(
        model,
        X_val_num,
        X_val_cat,
    )

    del model, optimizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # I. Fixed 80/20 rank blend against OOF champion predictions
    #
    # IMPORTANT:
    # Component predictions are ranked inside the same validation fold,
    # not ranked using the full dataset.
    # -------------------------------------------------------------------------

    lgb_val = lgb_oof[val_idx]
    mlp_val = mlp_oof[val_idx]

    lgb_rank = normalized_rank(lgb_val)
    mlp_rank = normalized_rank(mlp_val)
    tabm_rank = normalized_rank(tabm_pred)

    champion_val_rank = (
        0.40 * lgb_rank
        + 0.60 * mlp_rank
    )

    fixed_blend = (
        CHAMPION_BLEND_WEIGHT * champion_val_rank
        + TABM_BLEND_WEIGHT * tabm_rank
    )

    lgb_auc = roc_auc_score(
        y_val,
        lgb_val,
    )

    mlp_auc = roc_auc_score(
        y_val,
        mlp_val,
    )

    tabm_auc = roc_auc_score(
        y_val,
        tabm_pred,
    )

    champion_auc = roc_auc_score(
        y_val,
        champion_val_rank,
    )

    blend_auc = roc_auc_score(
        y_val,
        fixed_blend,
    )

    delta = blend_auc - champion_auc

    rho = pd.Series(
        tabm_rank
    ).corr(
        pd.Series(champion_val_rank),
        method="spearman",
    )

    log("\n" + "=" * 100)
    log("FOLD-2 CONFIRMATION RESULTS")
    log("=" * 100)

    log(f"LGBM fold-2 OOF AUC       = {lgb_auc:.9f}")
    log(f"RealMLP fold-2 OOF AUC    = {mlp_auc:.9f}")
    log(f"TabM M3 fold-2 AUC        = {tabm_auc:.9f}")
    log(f"Champion fold-2 AUC       = {champion_auc:.9f}")
    log(f"Fixed 80/20 blend AUC     = {blend_auc:.9f}")
    log(f"Gain vs Champion          = {delta:+.9f}")
    log(f"Spearman rank correlation= {rho:.9f}")

    if delta > 0:
        log(
            "\nFold-2 result: fixed 20% TabM blend "
            "is positive on this fold."
        )
    elif delta < 0:
        log(
            "\nFold-2 result: fixed 20% TabM blend "
            "is negative on this fold."
        )
    else:
        log(
            "\nFold-2 result: no AUC change."
        )

    # -------------------------------------------------------------------------
    # J. Save predictions and summary
    # -------------------------------------------------------------------------

    pred_path = os.path.join(
        OUTPUT_DIR,
        "tabm_m3_fold2_predictions.csv",
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "tabm_m3_fold2_summary.csv",
    )

    pred_df = pd.DataFrame({
        ID_COL: ids[val_idx],
        TARGET: y_val,
        "lgbm_prediction": lgb_val,
        "realmlp_prediction": mlp_val,
        "tabm_m3_prediction": tabm_pred,
        "champion_rank": champion_val_rank,
        "tabm_m3_rank": tabm_rank,
        "fixed_80_20_blend_rank": fixed_blend,
    })

    pred_df.to_csv(
        pred_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame([{
        "confirm_fold": CONFIRM_FOLD,
        "fixed_epochs": FIXED_EPOCHS,
        "tabm_weight": TABM_BLEND_WEIGHT,
        "champion_weight": CHAMPION_BLEND_WEIGHT,
        "lgbm_fold_auc": lgb_auc,
        "realmlp_fold_auc": mlp_auc,
        "tabm_m3_fold_auc": tabm_auc,
        "champion_fold_auc": champion_auc,
        "fixed_blend_fold_auc": blend_auc,
        "gain_vs_champion": delta,
        "spearman_tabm_champion": rho,
        "n_numeric": X_fit_num.shape[1],
        "n_categorical": X_fit_cat.shape[1],
    }])

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    log(f"\nSaved predictions: {pred_path}")
    log(f"Saved summary: {summary_path}")

    log(
        "\nDo not select or tune the weight based on this fold. "
        "Use the fixed-weight result to decide whether M3 "
        "deserves a full-OOF confirmation."
    )


if __name__ == "__main__":
    main()