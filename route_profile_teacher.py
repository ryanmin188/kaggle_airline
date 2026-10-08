import os
import warnings
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# ============================================================
# ROUTE / DISTANCE PROFILE ABLATION
#
# Competition-only.
#
# IMPORTANT:
# This competition dataset does NOT contain actual
# Departure Airport / Arrival Airport / Route ID.
#
# Therefore "route" is represented by a DISTANCE-BASED
# ROUTE PROXY:
#
#   exact Flight Distance
#   Flight Distance × Class
#   Flight Distance × Type of Travel
#   Flight Distance × Customer Type
#   Flight Distance × Class × Type of Travel
#
# No external 129k data.
# No XGBoost teacher.
# No RealMLP.
# ============================================================


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = r"D:\kaggle\airline\data\train.csv"
TEST_PATH = r"D:\kaggle\airline\data\test.csv"
SAMPLE_PATH = r"D:\kaggle\airline\data\sample_submission.csv"

OUTPUT_DIR = r"D:\kaggle\airline\route_profile_ablation"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

SCREEN_FOLDS = 3
FINAL_FOLDS = 5

INNER_TE_FOLDS = 5

# Current real-model benchmarks
CURRENT_LGBM_OOF = 0.960664306
CURRENT_BEST_BLEND_OOF = 0.960878324


# ------------------------------------------------------------
# Screening model
# ------------------------------------------------------------

SCREEN_LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",

    "learning_rate": 0.035,
    "num_leaves": 63,

    "min_child_samples": 50,

    "colsample_bytree": 0.75,

    "subsample": 0.80,
    "subsample_freq": 1,

    "reg_alpha": 0.30,
    "reg_lambda": 1.00,

    "n_estimators": 2500,

    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}


# ------------------------------------------------------------
# Final model
# ------------------------------------------------------------

FINAL_LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",

    "learning_rate": 0.03,
    "num_leaves": 63,

    "min_child_samples": 50,

    "colsample_bytree": 0.70,

    "subsample": 0.80,
    "subsample_freq": 1,

    "reg_alpha": 0.30,
    "reg_lambda": 1.00,

    "n_estimators": 8000,

    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}


# ============================================================
# LOAD
# ============================================================

print("=" * 100)
print("COMPETITION-ONLY DISTANCE ROUTE PROFILE ABLATION")
print("=" * 100)

print("LightGBM:", lgb.__version__)

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

TARGET = "satisfaction"
ID_COL = "id"


# ============================================================
# TARGET
# ============================================================

y = np.asarray(
    train[TARGET].to_numpy(dtype=np.int8)
).reshape(-1)

assert y.ndim == 1, (
    f"Target must be 1-dimensional, got shape={y.shape}"
)

assert len(y) == len(train), (
    f"Target length mismatch: "
    f"len(y)={len(y)}, train={len(train)}"
)

print(f"\nTrain shape : {train.shape}")
print(f"Test shape  : {test.shape}")
print(f"Target mean : {y.mean():.6f}")
print(f"Target shape: {y.shape}")


# ============================================================
# COLUMN GROUPS
# ============================================================

CATS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

SERVICES = [
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

NUMERIC = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

ORIGINAL_FEATURES = [
    c
    for c in train.columns
    if c not in [
        ID_COL,
        TARGET,
    ]
]

print(
    f"\nOriginal predictors = "
    f"{len(ORIGINAL_FEATURES)}"
)


# ============================================================
# COLUMN VALIDATION
# ============================================================

required_columns = (
    CATS
    + SERVICES
    + NUMERIC
    + [ID_COL, TARGET]
)

missing_train = [
    c
    for c in required_columns
    if c not in train.columns
]

missing_test = [
    c
    for c in required_columns
    if c != TARGET and c not in test.columns
]

assert not missing_train, (
    "Missing required train columns: "
    + str(missing_train)
)

assert not missing_test, (
    "Missing required test columns: "
    + str(missing_test)
)


# ============================================================
# BASE MODEL FRAME
#
# 21 original
# + traveler_segment
# + arrival_missing
# = 23 columns
# ============================================================

def build_base(df):

    X = df.drop(
        columns=[
            ID_COL,
            TARGET,
        ],
        errors="ignore",
    ).copy()

    X["traveler_segment"] = (
        df["Type of Travel"].astype(str)
        + "|"
        + df["Customer Type"].astype(str)
        + "|"
        + df["Class"].astype(str)
    )

    X["arrival_missing"] = (
        df["Arrival Delay in Minutes"]
        .isna()
        .astype("int8")
    )

    return X


X_BASE = build_base(train)
X_TEST_BASE = build_base(test)


# ============================================================
# BASE PREPROCESS
# ============================================================

BASE_NUMERIC = [
    c
    for c in ORIGINAL_FEATURES
    if c not in CATS
]

BASE_CATEGORICAL = CATS + [
    "traveler_segment"
]


for c in BASE_NUMERIC:

    median_value = train[c].median()

    X_BASE[c] = (
        X_BASE[c]
        .fillna(median_value)
        .astype("float32")
    )

    X_TEST_BASE[c] = (
        X_TEST_BASE[c]
        .fillna(median_value)
        .astype("float32")
    )


for c in BASE_CATEGORICAL:

    combined = pd.concat(
        [
            X_BASE[c].astype(str),
            X_TEST_BASE[c].astype(str),
        ],
        ignore_index=True,
    )

    categories = pd.Index(
        combined.unique()
    )

    dtype = pd.api.types.CategoricalDtype(
        categories=categories
    )

    X_BASE[c] = X_BASE[c].astype(dtype)
    X_TEST_BASE[c] = X_TEST_BASE[c].astype(dtype)


print(
    f"Base model columns = "
    f"{X_BASE.shape[1]}"
)

assert X_BASE.shape[1] == 23, (
    f"Expected 23 base columns, got {X_BASE.shape[1]}"
)


# ============================================================
# BASE TE INPUT
#
# Same as existing 78-feature model.
# ============================================================

Z_BASE = train[
    ORIGINAL_FEATURES
].copy()

U_BASE = test[
    ORIGINAL_FEATURES
].copy()


for c in ORIGINAL_FEATURES:

    Z_BASE[c] = (
        Z_BASE[c]
        .astype(str)
    )

    U_BASE[c] = (
        U_BASE[c]
        .astype(str)
    )


# ============================================================
# CONDITIONAL TE
# ============================================================

def build_conditional(df):

    traveler_segment = (
        df["Type of Travel"].astype(str)
        + "|"
        + df["Customer Type"].astype(str)
        + "|"
        + df["Class"].astype(str)
    )

    type_customer_group = (
        df["Type of Travel"].astype(str)
        + "|"
        + df["Customer Type"].astype(str)
    )

    Z = pd.DataFrame(
        index=df.index
    )

    # --------------------------------------------------------
    # 13 service × traveler segment
    # --------------------------------------------------------

    for c in SERVICES:

        Z[c + "__segment"] = (
            df[c].astype(str)
            + "|"
            + traveler_segment
        )

    # --------------------------------------------------------
    # 5 selected service interactions
    # --------------------------------------------------------

    PAIRS = [
        (
            "Inflight wifi service",
            "Online boarding",
        ),
        (
            "Inflight wifi service",
            "Ease of Online booking",
        ),
        (
            "Online boarding",
            "Seat comfort",
        ),
        (
            "Seat comfort",
            "Inflight entertainment",
        ),
        (
            "Inflight entertainment",
            "Cleanliness",
        ),
    ]

    for c1, c2 in PAIRS:

        name = (
            c1
            + "__"
            + c2
            + "__type_customer"
        )

        Z[name] = (
            df[c1].astype(str)
            + "|"
            + df[c2].astype(str)
            + "|"
            + type_customer_group
        )

    return Z


Z_COND = build_conditional(train)
U_COND = build_conditional(test)


# ============================================================
# NUMERIC GROUP TE
# ============================================================

def build_numeric_groups(df):

    Z = pd.DataFrame(
        index=df.index
    )

    GROUPS = [
        "Type of Travel",
        "Customer Type",
        "Class",
    ]

    for numeric_col in [
        "Age",
        "Flight Distance",
    ]:

        for group_col in GROUPS:

            name = (
                numeric_col
                + "__"
                + group_col
            )

            Z[name] = (
                df[numeric_col].astype(str)
                + "|"
                + df[group_col].astype(str)
            )

    return Z


Z_NUM_GROUP = build_numeric_groups(train)
U_NUM_GROUP = build_numeric_groups(test)


# ============================================================
# FREQUENCY ENCODING
# ============================================================

def build_frequency_features(
    train_df,
    val_df,
    test_df,
):

    train_out = pd.DataFrame(
        index=train_df.index
    )

    val_out = pd.DataFrame(
        index=val_df.index
    )

    test_out = pd.DataFrame(
        index=test_df.index
    )

    for c in train_df.columns:

        freq_map = (
            train_df[c]
            .astype(str)
            .value_counts(
                normalize=True,
                dropna=False,
            )
        )

        feature_name = (
            "freq_"
            + c
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
            .replace("|", "_")
        )

        tr_freq = (
            train_df[c]
            .astype(str)
            .map(freq_map)
            .fillna(0)
            .to_numpy()
            .reshape(-1)
        )

        va_freq = (
            val_df[c]
            .astype(str)
            .map(freq_map)
            .fillna(0)
            .to_numpy()
            .reshape(-1)
        )

        te_freq = (
            test_df[c]
            .astype(str)
            .map(freq_map)
            .fillna(0)
            .to_numpy()
            .reshape(-1)
        )

        train_out[
            feature_name
        ] = np.log1p(
            tr_freq * 1_000_000
        ).astype("float32")

        val_out[
            feature_name
        ] = np.log1p(
            va_freq * 1_000_000
        ).astype("float32")

        test_out[
            feature_name
        ] = np.log1p(
            te_freq * 1_000_000
        ).astype("float32")

    return (
        train_out,
        val_out,
        test_out,
    )


# ============================================================
# DISTANCE / ROUTE PROXY
# ============================================================

def build_distance_keys(df):

    Z = pd.DataFrame(
        index=df.index
    )

    distance = (
        df["Flight Distance"]
        .fillna(-999)
        .astype(str)
    )

    cls = df["Class"].astype(str)

    travel = (
        df["Type of Travel"]
        .astype(str)
    )

    customer = (
        df["Customer Type"]
        .astype(str)
    )

    # --------------------------------------------------------
    # Exact distance
    # --------------------------------------------------------

    Z["distance_exact"] = distance

    # --------------------------------------------------------
    # Distance × Class
    # --------------------------------------------------------

    Z["distance_class"] = (
        distance
        + "|"
        + cls
    )

    # --------------------------------------------------------
    # Distance × Travel
    # --------------------------------------------------------

    Z["distance_travel"] = (
        distance
        + "|"
        + travel
    )

    # --------------------------------------------------------
    # Distance × Customer
    # --------------------------------------------------------

    Z["distance_customer"] = (
        distance
        + "|"
        + customer
    )

    # --------------------------------------------------------
    # Distance × Class × Travel
    # --------------------------------------------------------

    Z["distance_class_travel"] = (
        distance
        + "|"
        + cls
        + "|"
        + travel
    )

    return Z


Z_DISTANCE_KEYS = build_distance_keys(train)
U_DISTANCE_KEYS = build_distance_keys(test)


# ============================================================
# DISTANCE BINS
#
# Fold-specific:
# edges are calculated from outer-training Flight Distance
# only.
# ============================================================

def build_distance_bin_edges(
    distance_series,
    n_bins=20,
):

    distance_clean = (
        distance_series
        .dropna()
    )

    if len(distance_clean) == 0:

        return np.array(
            [
                -np.inf,
                np.inf,
            ],
            dtype=float,
        )

    quantile_grid = np.linspace(
        0,
        1,
        n_bins + 1,
    )

    edges = (
        distance_clean
        .quantile(
            quantile_grid
        )
        .drop_duplicates()
        .to_numpy()
        .copy()
    )

    # --------------------------------------------------------
    # Make absolutely sure the array is writable and 1-D.
    # --------------------------------------------------------

    edges = np.asarray(
        edges,
        dtype=float,
    ).reshape(-1)

    if len(edges) >= 3:

        edges[0] = -np.inf
        edges[-1] = np.inf

    else:

        edges = np.array(
            [
                -np.inf,
                np.inf,
            ],
            dtype=float,
        )

    return edges


def build_distance_bins(
    df,
    distance_bin_edges,
):

    Z = pd.DataFrame(
        index=df.index
    )

    bins = pd.cut(
        df["Flight Distance"],
        bins=distance_bin_edges,
        include_lowest=True,
        duplicates="drop",
    )

    bins = (
        bins
        .astype(str)
    )

    cls = (
        df["Class"]
        .astype(str)
    )

    travel = (
        df["Type of Travel"]
        .astype(str)
    )

    Z["distance_bin"] = bins

    Z["distance_bin_class_travel"] = (
        bins
        + "|"
        + cls
        + "|"
        + travel
    )

    return Z


# ============================================================
# GENERIC TARGET PROFILE
# ============================================================

def fit_profile_map(
    key_series,
    target,
):
    """
    Estimate profile statistics from one single key Series.

    Returns:
        count
        mean
        std
        positive_share

    IMPORTANT:
    key_series must be 1-dimensional.
    """

    # --------------------------------------------------------
    # Force keys to a 1-D NumPy array.
    # --------------------------------------------------------

    keys = np.asarray(
        key_series.astype(str).to_numpy()
    ).reshape(-1)

    # --------------------------------------------------------
    # Force target to a 1-D NumPy array.
    # --------------------------------------------------------

    target_array = np.asarray(
        target
    ).reshape(-1)

    assert keys.ndim == 1, (
        f"Profile keys must be 1-D, "
        f"got shape={keys.shape}"
    )

    assert target_array.ndim == 1, (
        f"Profile target must be 1-D, "
        f"got shape={target_array.shape}"
    )

    assert len(keys) == len(target_array), (
        "Profile key/target length mismatch: "
        f"{len(keys)} vs {len(target_array)}"
    )

    temp = pd.DataFrame({
        "key": keys,
        "y": target_array,
    })

    grouped = (
        temp
        .groupby(
            "key",
            observed=True,
        )["y"]
        .agg(
            [
                "count",
                "mean",
                "std",
            ]
        )
    )

    # For binary target:
    # positive_share == mean.
    grouped["positive_share"] = (
        grouped["mean"]
    )

    grouped["std"] = (
        grouped["std"]
        .fillna(0.0)
    )

    return grouped


# ============================================================
# MAP PROFILE TO DATA
# ============================================================

def apply_profile_map(
    key_series,
    profile_map,
    prefix,
    global_mean,
):

    keys = (
        key_series
        .astype(str)
    )

    count = (
        keys
        .map(profile_map["count"])
        .fillna(0.0)
        .to_numpy()
        .reshape(-1)
    )

    mean = (
        keys
        .map(profile_map["mean"])
        .fillna(global_mean)
        .to_numpy()
        .reshape(-1)
    )

    std = (
        keys
        .map(profile_map["std"])
        .fillna(0.0)
        .to_numpy()
        .reshape(-1)
    )

    positive = (
        keys
        .map(profile_map["positive_share"])
        .fillna(global_mean)
        .to_numpy()
        .reshape(-1)
    )

    result = pd.DataFrame(
        index=key_series.index
    )

    result[
        prefix + "_count"
    ] = np.log1p(
        count
    ).astype("float32")

    result[
        prefix + "_mean"
    ] = mean.astype("float32")

    result[
        prefix + "_std"
    ] = std.astype("float32")

    result[
        prefix + "_positive"
    ] = positive.astype("float32")

    return result


# ============================================================
# INNER-OOF PROFILE
#
# This is the critical anti-leakage function.
#
# Every outer-training observation gets a profile generated
# without using its own target.
# ============================================================

def build_crossfit_profiles(
    key_series,
    target,
    inner_folds=5,
    random_state=42,
):

    # --------------------------------------------------------
    # Force target to 1-D.
    # --------------------------------------------------------

    target_array = np.asarray(
        target
    ).reshape(-1)

    assert target_array.ndim == 1, (
        f"Target must be 1-D, "
        f"got shape={target_array.shape}"
    )

    assert len(target_array) == len(key_series), (
        "Target/key row mismatch: "
        f"target={len(target_array)}, "
        f"keys={len(key_series)}"
    )

    # --------------------------------------------------------
    # Empty result with one set of four features per key.
    # --------------------------------------------------------

    result = pd.DataFrame(
        index=key_series.index
    )

    for c in key_series.columns:

        result[
            c + "_count"
        ] = np.nan

        result[
            c + "_mean"
        ] = np.nan

        result[
            c + "_std"
        ] = np.nan

        result[
            c + "_positive"
        ] = np.nan


    # --------------------------------------------------------
    # Inner CV
    # --------------------------------------------------------

    inner_skf = StratifiedKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=random_state,
    )

    for inner_tr_idx, inner_va_idx in inner_skf.split(
        np.zeros(len(target_array)),
        target_array,
    ):

        inner_y = target_array[
            inner_tr_idx
        ]

        # IMPORTANT:
        # Fallback mean is estimated only from inner train.
        inner_global_mean = float(
            inner_y.mean()
        )

        # ----------------------------------------------------
        # Process ONE key column at a time.
        #
        # THIS IS THE FIX FOR:
        #
        # Profile key/target length mismatch:
        # 1865690 vs 373138
        #
        # 1865690 = 373138 × 5.
        # ----------------------------------------------------

        for c in key_series.columns:

            key_train = (
                key_series[c]
                .iloc[inner_tr_idx]
            )

            key_valid = (
                key_series[c]
                .iloc[inner_va_idx]
            )

            # ------------------------------------------------
            # Shape checks
            # ------------------------------------------------

            assert isinstance(
                key_train,
                pd.Series,
            ), (
                f"{c}: key_train should be Series, "
                f"got {type(key_train)}"
            )

            assert isinstance(
                key_valid,
                pd.Series,
            ), (
                f"{c}: key_valid should be Series, "
                f"got {type(key_valid)}"
            )

            assert len(key_train) == len(inner_y), (
                f"{c}: inner train length mismatch: "
                f"keys={len(key_train)}, "
                f"target={len(inner_y)}"
            )

            # ------------------------------------------------
            # Fit on inner-train only
            # ------------------------------------------------

            profile_map = fit_profile_map(
                key_train,
                inner_y,
            )

            # ------------------------------------------------
            # Map to inner-validation only
            # ------------------------------------------------

            mapped = apply_profile_map(
                key_valid,
                profile_map,
                c,
                inner_global_mean,
            )

            # ------------------------------------------------
            # Put values back into corresponding outer-train
            # observations.
            # ------------------------------------------------

            result.loc[
                key_series.index[
                    inner_va_idx
                ],
                mapped.columns,
            ] = mapped.to_numpy()


    # --------------------------------------------------------
    # Final missing-value cleanup
    # --------------------------------------------------------

    outer_global_mean = float(
        target_array.mean()
    )

    for c in result.columns:

        if c.endswith("_mean"):

            result[c] = (
                result[c]
                .fillna(
                    outer_global_mean
                )
            )

        elif c.endswith("_positive"):

            result[c] = (
                result[c]
                .fillna(
                    outer_global_mean
                )
            )

        else:

            result[c] = (
                result[c]
                .fillna(0.0)
            )

    return result.astype("float32")


# ============================================================
# FULL OUTER-TRAIN PROFILE
#
# Validation/test profiles use ONLY outer-training y.
# ============================================================

def build_full_profiles(
    train_keys,
    train_y,
    val_keys,
    test_keys,
):

    train_y_array = np.asarray(
        train_y
    ).reshape(-1)

    assert train_y_array.ndim == 1

    assert len(train_y_array) == len(train_keys), (
        "Outer-train profile target/key mismatch: "
        f"target={len(train_y_array)}, "
        f"keys={len(train_keys)}"
    )

    train_global_mean = float(
        train_y_array.mean()
    )

    val_out = pd.DataFrame(
        index=val_keys.index
    )

    test_out = pd.DataFrame(
        index=test_keys.index
    )

    for c in train_keys.columns:

        profile_map = fit_profile_map(
            train_keys[c],
            train_y_array,
        )

        val_mapped = apply_profile_map(
            val_keys[c],
            profile_map,
            c,
            train_global_mean,
        )

        test_mapped = apply_profile_map(
            test_keys[c],
            profile_map,
            c,
            train_global_mean,
        )

        val_out = pd.concat(
            [
                val_out,
                val_mapped,
            ],
            axis=1,
        )

        test_out = pd.concat(
            [
                test_out,
                test_mapped,
            ],
            axis=1,
        )

    return (
        val_out.astype("float32"),
        test_out.astype("float32"),
    )


# ============================================================
# DISTANCE PROFILE GROUPS
# ============================================================

PROFILE_GROUPS = {

    # --------------------------------------------------------
    # Exact distance
    # --------------------------------------------------------

    "route_exact": [
        "distance_exact_count",
        "distance_exact_mean",
        "distance_exact_std",
        "distance_exact_positive",
    ],

    # --------------------------------------------------------
    # Distance × Class
    # --------------------------------------------------------

    "route_class": [
        "distance_class_count",
        "distance_class_mean",
        "distance_class_std",
        "distance_class_positive",
    ],

    # --------------------------------------------------------
    # Distance × Travel
    # --------------------------------------------------------

    "route_travel": [
        "distance_travel_count",
        "distance_travel_mean",
        "distance_travel_std",
        "distance_travel_positive",
    ],

    # --------------------------------------------------------
    # Distance × Customer
    # --------------------------------------------------------

    "route_customer": [
        "distance_customer_count",
        "distance_customer_mean",
        "distance_customer_std",
        "distance_customer_positive",
    ],

    # --------------------------------------------------------
    # Distance × Class × Travel
    # --------------------------------------------------------

    "route_class_travel": [
        "distance_class_travel_count",
        "distance_class_travel_mean",
        "distance_class_travel_std",
        "distance_class_travel_positive",
    ],

    # --------------------------------------------------------
    # Distance bins
    # --------------------------------------------------------

    "distance_bin": [
        "distance_bin_count",
        "distance_bin_mean",
        "distance_bin_std",
        "distance_bin_positive",
    ],

    # --------------------------------------------------------
    # Distance bin × Class × Travel
    # --------------------------------------------------------

    "distance_bin_class_travel": [
        "distance_bin_class_travel_count",
        "distance_bin_class_travel_mean",
        "distance_bin_class_travel_std",
        "distance_bin_class_travel_positive",
    ],
}


FULL_PROFILE_GROUPS = [
    "route_exact",
    "route_class",
    "route_travel",
    "route_customer",
    "route_class_travel",
    "distance_bin",
    "distance_bin_class_travel",
]


# ============================================================
# BASE 78 FEATURES
# ============================================================

def build_base_78_fold(
    tr_idx,
    va_idx,
):

    X_tr = X_BASE.iloc[
        tr_idx
    ].copy()

    X_va = X_BASE.iloc[
        va_idx
    ].copy()

    X_te = X_TEST_BASE.copy()


    # ========================================================
    # 1. BASE TE
    # ========================================================

    enc_base = TargetEncoder(
        target_type="binary",
        smooth="auto",
        cv=INNER_TE_FOLDS,
        shuffle=True,
        random_state=20261002,
    )

    te_tr = enc_base.fit_transform(
        Z_BASE.iloc[tr_idx],
        y[tr_idx],
    ).astype("float32")

    te_va = enc_base.transform(
        Z_BASE.iloc[va_idx]
    ).astype("float32")

    te_te = enc_base.transform(
        U_BASE
    ).astype("float32")


    for j, c in enumerate(
        ORIGINAL_FEATURES
    ):

        name = (
            "te_"
            + c
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
        )

        X_tr[name] = te_tr[:, j]
        X_va[name] = te_va[:, j]
        X_te[name] = te_te[:, j]


    # ========================================================
    # 2. CONDITIONAL TE
    # ========================================================

    enc_cond = TargetEncoder(
        target_type="binary",
        smooth="auto",
        cv=INNER_TE_FOLDS,
        shuffle=True,
        random_state=20261003,
    )

    cond_tr = enc_cond.fit_transform(
        Z_COND.iloc[tr_idx],
        y[tr_idx],
    ).astype("float32")

    cond_va = enc_cond.transform(
        Z_COND.iloc[va_idx]
    ).astype("float32")

    cond_te = enc_cond.transform(
        U_COND
    ).astype("float32")


    for j in range(
        Z_COND.shape[1]
    ):

        name = (
            "cond_te_"
            + str(j)
        )

        X_tr[name] = cond_tr[:, j]
        X_va[name] = cond_va[:, j]
        X_te[name] = cond_te[:, j]


    # ========================================================
    # 3. NUMERIC GROUP TE
    # ========================================================

    enc_num = TargetEncoder(
        target_type="binary",
        smooth=20.0,
        cv=INNER_TE_FOLDS,
        shuffle=True,
        random_state=20261004,
    )

    num_tr = enc_num.fit_transform(
        Z_NUM_GROUP.iloc[tr_idx],
        y[tr_idx],
    ).astype("float32")

    num_va = enc_num.transform(
        Z_NUM_GROUP.iloc[va_idx]
    ).astype("float32")

    num_te = enc_num.transform(
        U_NUM_GROUP
    ).astype("float32")


    for j in range(
        Z_NUM_GROUP.shape[1]
    ):

        name = (
            "ng_te_"
            + str(j)
        )

        X_tr[name] = num_tr[:, j]
        X_va[name] = num_va[:, j]
        X_te[name] = num_te[:, j]


    # ========================================================
    # 4. FREQUENCY
    # ========================================================

    freq_tr_input = pd.concat(
        [
            Z_BASE.iloc[
                tr_idx
            ][NUMERIC].reset_index(
                drop=True
            ),

            Z_NUM_GROUP.iloc[
                tr_idx
            ].reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    freq_va_input = pd.concat(
        [
            Z_BASE.iloc[
                va_idx
            ][NUMERIC].reset_index(
                drop=True
            ),

            Z_NUM_GROUP.iloc[
                va_idx
            ].reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    freq_te_input = pd.concat(
        [
            U_BASE[
                NUMERIC
            ].reset_index(
                drop=True
            ),

            U_NUM_GROUP.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    freq_tr, freq_va, freq_te = (
        build_frequency_features(
            freq_tr_input,
            freq_va_input,
            freq_te_input,
        )
    )

    freq_tr.index = X_tr.index
    freq_va.index = X_va.index
    freq_te.index = X_te.index

    X_tr = pd.concat(
        [
            X_tr,
            freq_tr,
        ],
        axis=1,
    )

    X_va = pd.concat(
        [
            X_va,
            freq_va,
        ],
        axis=1,
    )

    X_te = pd.concat(
        [
            X_te,
            freq_te,
        ],
        axis=1,
    )


    # ========================================================
    # CLEAN
    # ========================================================

    for df in [
        X_tr,
        X_va,
        X_te,
    ]:

        df.replace(
            [np.inf, -np.inf],
            np.nan,
            inplace=True,
        )

    numeric_cols = X_tr.select_dtypes(
        include=[
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
        ]
    ).columns

    for c in numeric_cols:

        median_value = (
            X_tr[c]
            .median()
        )

        if pd.isna(median_value):
            median_value = 0.0

        X_tr[c] = (
            X_tr[c]
            .fillna(median_value)
        )

        X_va[c] = (
            X_va[c]
            .fillna(median_value)
        )

        X_te[c] = (
            X_te[c]
            .fillna(median_value)
        )


    # --------------------------------------------------------
    # Remove accidental duplicate names.
    # --------------------------------------------------------

    X_tr = X_tr.loc[
        :,
        ~X_tr.columns.duplicated()
    ]

    X_va = X_va[
        X_tr.columns
    ]

    X_te = X_te[
        X_tr.columns
    ]


    # --------------------------------------------------------
    # Critical check
    # --------------------------------------------------------

    assert X_tr.shape[1] == 78, (
        "Base representation changed: "
        f"expected 78, got {X_tr.shape[1]}"
    )

    return (
        X_tr,
        X_va,
        X_te,
    )


# ============================================================
# ROUTE PROFILES FOR ONE OUTER FOLD
# ============================================================

def build_route_profiles_for_fold(
    tr_idx,
    va_idx,
):

    # --------------------------------------------------------
    # Fold-specific distance bins
    # --------------------------------------------------------

    fold_distance_edges = (
        build_distance_bin_edges(
            train["Flight Distance"].iloc[
                tr_idx
            ]
        )
    )

    # --------------------------------------------------------
    # Exact distance keys
    # --------------------------------------------------------

    outer_train_keys = (
        Z_DISTANCE_KEYS.iloc[
            tr_idx
        ]
        .reset_index(drop=True)
    )

    outer_train_y = y[
        tr_idx
    ]

    # --------------------------------------------------------
    # Fold-specific bin keys
    # --------------------------------------------------------

    outer_train_bins = (
        build_distance_bins(
            train.iloc[
                tr_idx
            ],
            fold_distance_edges,
        )
        .reset_index(drop=True)
    )

    va_keys = (
        Z_DISTANCE_KEYS.iloc[
            va_idx
        ]
        .reset_index(drop=True)
    )

    te_keys = (
        U_DISTANCE_KEYS
        .reset_index(drop=True)
    )

    va_bins = (
        build_distance_bins(
            train.iloc[
                va_idx
            ],
            fold_distance_edges,
        )
        .reset_index(drop=True)
    )

    te_bins = (
        build_distance_bins(
            test,
            fold_distance_edges,
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # Cross-fitted outer-train profiles
    # --------------------------------------------------------

    print(
        "      Cross-fitting distance profiles..."
    )

    route_train = build_crossfit_profiles(
        outer_train_keys,
        outer_train_y,
        inner_folds=INNER_TE_FOLDS,
        random_state=20261010,
    )

    bin_train = build_crossfit_profiles(
        outer_train_bins,
        outer_train_y,
        inner_folds=INNER_TE_FOLDS,
        random_state=20261011,
    )


    # --------------------------------------------------------
    # Validation / test profiles from complete outer train
    # --------------------------------------------------------

    print(
        "      Mapping validation/test profiles..."
    )

    route_va, route_te = (
        build_full_profiles(
            outer_train_keys,
            outer_train_y,
            va_keys,
            te_keys,
        )
    )

    bin_va, bin_te = (
        build_full_profiles(
            outer_train_bins,
            outer_train_y,
            va_bins,
            te_bins,
        )
    )


    # --------------------------------------------------------
    # Restore original indexes
    # --------------------------------------------------------

    route_train.index = (
        X_BASE.iloc[
            tr_idx
        ].index
    )

    route_va.index = (
        X_BASE.iloc[
            va_idx
        ].index
    )

    route_te.index = (
        X_TEST_BASE.index
    )

    bin_train.index = (
        X_BASE.iloc[
            tr_idx
        ].index
    )

    bin_va.index = (
        X_BASE.iloc[
            va_idx
        ].index
    )

    bin_te.index = (
        X_TEST_BASE.index
    )


    # --------------------------------------------------------
    # Merge exact-distance + distance-bin profiles
    # --------------------------------------------------------

    route_train = pd.concat(
        [
            route_train,
            bin_train,
        ],
        axis=1,
    )

    route_va = pd.concat(
        [
            route_va,
            bin_va,
        ],
        axis=1,
    )

    route_te = pd.concat(
        [
            route_te,
            bin_te,
        ],
        axis=1,
    )


    # --------------------------------------------------------
    # Each of 7 key types has 4 statistics:
    #
    # 7 × 4 = 28
    # --------------------------------------------------------

    expected_route_features = 7 * 4

    assert route_train.shape[1] == expected_route_features, (
        f"Expected {expected_route_features} route features, "
        f"got {route_train.shape[1]}"
    )

    assert route_va.shape[1] == expected_route_features
    assert route_te.shape[1] == expected_route_features

    return (
        route_train.astype("float32"),
        route_va.astype("float32"),
        route_te.astype("float32"),
    )


# ============================================================
# PREPARE LIGHTGBM
# ============================================================

def prepare_lgbm(
    X_tr,
    X_va,
    X_te,
):

    X_tr = X_tr.copy()
    X_va = X_va.copy()
    X_te = X_te.copy()

    cat_cols = []

    for c in X_tr.columns:

        if (
            X_tr[c].dtype == "object"
            or str(
                X_tr[c].dtype
            ).startswith("category")
        ):

            cat_cols.append(c)


    for c in cat_cols:

        combined = pd.concat(
            [
                X_tr[c].astype(str),
                X_va[c].astype(str),
                X_te[c].astype(str),
            ],
            ignore_index=True,
        )

        categories = pd.Index(
            combined.unique()
        )

        dtype = pd.api.types.CategoricalDtype(
            categories=categories
        )

        X_tr[c] = X_tr[c].astype(dtype)
        X_va[c] = X_va[c].astype(dtype)
        X_te[c] = X_te[c].astype(dtype)


    return (
        X_tr,
        X_va,
        X_te,
        cat_cols,
    )


# ============================================================
# TRAIN LIGHTGBM
# ============================================================

def train_lgbm(
    X_tr,
    y_tr,
    X_va,
    y_va,
    params,
):

    (
        X_tr,
        X_va,
        _,
        cat_cols,
    ) = prepare_lgbm(
        X_tr,
        X_va,
        X_va,
    )

    model = lgb.LGBMClassifier(
        **params
    )

    model.fit(
        X_tr,
        y_tr,

        eval_set=[
            (
                X_va,
                y_va,
            )
        ],

        eval_metric="auc",

        categorical_feature=(
            cat_cols
            if len(cat_cols) > 0
            else "auto"
        ),

        callbacks=[
            lgb.early_stopping(
                100,
                verbose=False,
            )
        ],
    )

    pred = (
        model
        .predict_proba(
            X_va
        )[:, 1]
    )

    return (
        pred,
        model.best_iteration_,
    )


# ============================================================
# PHASE 1 — BUILD 3-FOLD SCREENING FEATURES
# ============================================================

print("\n" + "=" * 100)
print("PHASE 1 — BUILD 3-FOLD SCREENING FEATURES")
print("=" * 100)

screen_skf = StratifiedKFold(
    n_splits=SCREEN_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

screen_fold_data = []


for fold, (
    tr_idx,
    va_idx,
) in enumerate(
    screen_skf.split(
        np.zeros(len(y)),
        y,
    ),
    1,
):

    print("\n" + "-" * 100)
    print(
        f"BUILD SCREEN FOLD "
        f"{fold}/{SCREEN_FOLDS}"
    )
    print("-" * 100)

    # --------------------------------------------------------
    # Existing 78 features
    # --------------------------------------------------------

    (
        X_tr_base,
        X_va_base,
        X_te_base,
    ) = build_base_78_fold(
        tr_idx,
        va_idx,
    )

    print(
        f"      Base features = "
        f"{X_tr_base.shape[1]}"
    )

    # --------------------------------------------------------
    # New route profiles
    # --------------------------------------------------------

    (
        route_tr,
        route_va,
        route_te,
    ) = build_route_profiles_for_fold(
        tr_idx,
        va_idx,
    )

    print(
        f"      Route features = "
        f"{route_tr.shape[1]}"
    )

    screen_fold_data.append(
        {
            "tr_idx": tr_idx,
            "va_idx": va_idx,

            "X_tr_base": X_tr_base,
            "X_va_base": X_va_base,
            "X_te_base": X_te_base,

            "route_tr": route_tr,
            "route_va": route_va,
            "route_te": route_te,
        }
    )


# ============================================================
# ABLATION DEFINITIONS
# ============================================================

ABLATIONS = {

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    "baseline_78": [],

    # --------------------------------------------------------
    # Individual groups
    # --------------------------------------------------------

    "route_exact": [
        "route_exact",
    ],

    "route_class": [
        "route_class",
    ],

    "route_travel": [
        "route_travel",
    ],

    "route_customer": [
        "route_customer",
    ],

    "route_class_travel": [
        "route_class_travel",
    ],

    "distance_bin": [
        "distance_bin",
    ],

    "distance_bin_class_travel": [
        "distance_bin_class_travel",
    ],

    # --------------------------------------------------------
    # Logical combinations
    # --------------------------------------------------------

    "route_exact_plus_class": [
        "route_exact",
        "route_class",
    ],

    "route_exact_plus_travel": [
        "route_exact",
        "route_travel",
    ],

    "route_exact_plus_customer": [
        "route_exact",
        "route_customer",
    ],

    "route_exact_plus_class_travel": [
        "route_exact",
        "route_class_travel",
    ],

    "all_exact_route_profiles": [
        "route_exact",
        "route_class",
        "route_travel",
        "route_customer",
        "route_class_travel",
    ],

    "all_distance_bins": [
        "distance_bin",
        "distance_bin_class_travel",
    ],

    # --------------------------------------------------------
    # Everything
    # --------------------------------------------------------

    "FULL_ROUTE_PROFILE": FULL_PROFILE_GROUPS,
}


# ============================================================
# PHASE 1 — ABLATION SCREENING
# ============================================================

print("\n" + "=" * 100)
print("PHASE 1 — 3-FOLD ABLATION SCREEN")
print("=" * 100)

screen_results = []

baseline_auc = None


for experiment_name, groups in ABLATIONS.items():

    print("\n" + "-" * 100)
    print(
        f"EXPERIMENT: "
        f"{experiment_name}"
    )
    print("-" * 100)

    print(
        "Groups:",
        groups
        if groups
        else "[BASELINE 78]",
    )

    oof = np.zeros(
        len(train),
        dtype=np.float32,
    )

    fold_scores = []
    fold_iters = []

    start_time = time.time()

    for fold_no, fold_data in enumerate(
        screen_fold_data,
        1,
    ):

        X_tr = (
            fold_data[
                "X_tr_base"
            ].copy()
        )

        X_va = (
            fold_data[
                "X_va_base"
            ].copy()
        )

        # ----------------------------------------------------
        # Add selected profile groups
        # ----------------------------------------------------

        for group in groups:

            columns = PROFILE_GROUPS[
                group
            ]

            X_tr = pd.concat(
                [
                    X_tr,
                    fold_data[
                        "route_tr"
                    ][columns],
                ],
                axis=1,
            )

            X_va = pd.concat(
                [
                    X_va,
                    fold_data[
                        "route_va"
                    ][columns],
                ],
                axis=1,
            )

        # ----------------------------------------------------
        # Identical columns
        # ----------------------------------------------------

        X_va = X_va[
            X_tr.columns
        ]

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        pred, best_iter = train_lgbm(
            X_tr,
            y[
                fold_data[
                    "tr_idx"
                ]
            ],
            X_va,
            y[
                fold_data[
                    "va_idx"
                ]
            ],
            SCREEN_LGB_PARAMS,
        )

        oof[
            fold_data[
                "va_idx"
            ]
        ] = pred

        auc = roc_auc_score(
            y[
                fold_data[
                    "va_idx"
                ]
            ],
            pred,
        )

        fold_scores.append(
            auc
        )

        fold_iters.append(
            best_iter
        )

        print(
            f"Fold {fold_no}: "
            f"AUC={auc:.6f} "
            f"iter={best_iter}"
        )

    # --------------------------------------------------------
    # Overall OOF
    # --------------------------------------------------------

    overall_auc = roc_auc_score(
        y,
        oof,
    )

    mean_auc = np.mean(
        fold_scores
    )

    elapsed = (
        time.time()
        - start_time
    )

    if baseline_auc is None:

        baseline_auc = overall_auc

        delta_vs_baseline = 0.0

    else:

        delta_vs_baseline = (
            overall_auc
            - baseline_auc
        )

    print(
        f"\n{experiment_name}"
    )

    print(
        f"OOF AUC      = "
        f"{overall_auc:.6f}"
    )

    print(
        f"Fold mean    = "
        f"{mean_auc:.6f}"
    )

    print(
        f"Delta vs 78  = "
        f"{delta_vs_baseline:+.6f}"
    )

    print(
        f"Time         = "
        f"{elapsed / 60:.1f} min"
    )

    screen_results.append(
        {
            "experiment":
                experiment_name,

            "auc":
                overall_auc,

            "fold_mean":
                mean_auc,

            "fold_std":
                np.std(
                    fold_scores
                ),

            "gain_vs_baseline_78":
                delta_vs_baseline,

            "gain_vs_current_lgbm":
                (
                    overall_auc
                    - CURRENT_LGBM_OOF
                ),

            "gain_vs_best_blend":
                (
                    overall_auc
                    - CURRENT_BEST_BLEND_OOF
                ),

            "best_iterations":
                str(
                    fold_iters
                ),

            "n_profile_groups":
                len(groups),
        }
    )


# ============================================================
# SCREEN SUMMARY
# ============================================================

screen_df = (
    pd.DataFrame(
        screen_results
    )
    .sort_values(
        "auc",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


print("\n" + "=" * 100)
print("ABLATION RANKING")
print("=" * 100)

print(
    screen_df[
        [
            "experiment",
            "auc",
            "gain_vs_baseline_78",
            "gain_vs_current_lgbm",
            "gain_vs_best_blend",
            "fold_std",
        ]
    ].to_string(
        index=False
    )
)


screen_path = os.path.join(
    OUTPUT_DIR,
    "route_profile_ablation_screen.csv",
)

screen_df.to_csv(
    screen_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SELECT BEST FEATURE SET
# ============================================================

best_row = screen_df.iloc[0]

BEST_EXPERIMENT = (
    best_row["experiment"]
)

BEST_SCREEN_AUC = (
    best_row["auc"]
)

BEST_GROUPS = ABLATIONS[
    BEST_EXPERIMENT
]


print("\n" + "=" * 100)
print("SCREEN WINNER")
print("=" * 100)

print(
    f"Best experiment = "
    f"{BEST_EXPERIMENT}"
)

print(
    f"Screen OOF      = "
    f"{BEST_SCREEN_AUC:.6f}"
)

print(
    f"Gain vs 78      = "
    f"{BEST_SCREEN_AUC - baseline_auc:+.6f}"
)

print(
    f"Gain vs LGBM    = "
    f"{BEST_SCREEN_AUC - CURRENT_LGBM_OOF:+.6f}"
)

print(
    f"Gain vs blend   = "
    f"{BEST_SCREEN_AUC - CURRENT_BEST_BLEND_OOF:+.6f}"
)

print(
    "\nSelected groups:"
)

for group in BEST_GROUPS:

    print(
        f"  + {group}"
    )


# ============================================================
# PHASE 2 — FINAL 5-FOLD
# ============================================================

print("\n" + "=" * 100)
print("PHASE 2 — FINAL 5-FOLD")
print("=" * 100)

final_skf = StratifiedKFold(
    n_splits=FINAL_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


final_oof = np.zeros(
    len(train),
    dtype=np.float32,
)

final_test = np.zeros(
    len(test),
    dtype=np.float64,
)

final_fold_scores = []
final_iters = []


for fold, (
    tr_idx,
    va_idx,
) in enumerate(
    final_skf.split(
        np.zeros(len(y)),
        y,
    ),
    1,
):

    print("\n" + "-" * 100)
    print(
        f"FINAL FOLD "
        f"{fold}/{FINAL_FOLDS}"
    )
    print("-" * 100)

    # --------------------------------------------------------
    # 78 base features
    # --------------------------------------------------------

    (
        X_tr,
        X_va,
        X_te,
    ) = build_base_78_fold(
        tr_idx,
        va_idx,
    )

    print(
        f"Base features = "
        f"{X_tr.shape[1]}"
    )

    # --------------------------------------------------------
    # Route profiles
    # --------------------------------------------------------

    (
        route_tr,
        route_va,
        route_te,
    ) = build_route_profiles_for_fold(
        tr_idx,
        va_idx,
    )

    # --------------------------------------------------------
    # Add selected route groups
    # --------------------------------------------------------

    for group in BEST_GROUPS:

        columns = PROFILE_GROUPS[
            group
        ]

        X_tr = pd.concat(
            [
                X_tr,
                route_tr[columns],
            ],
            axis=1,
        )

        X_va = pd.concat(
            [
                X_va,
                route_va[columns],
            ],
            axis=1,
        )

        X_te = pd.concat(
            [
                X_te,
                route_te[columns],
            ],
            axis=1,
        )

    # --------------------------------------------------------
    # Same columns
    # --------------------------------------------------------

    X_va = X_va[
        X_tr.columns
    ]

    X_te = X_te[
        X_tr.columns
    ]

    final_expected_features = (
        78
        + sum(
            len(
                PROFILE_GROUPS[g]
            )
            for g in BEST_GROUPS
        )
    )

    assert X_tr.shape[1] == final_expected_features, (
        f"Final feature count mismatch: "
        f"expected={final_expected_features}, "
        f"got={X_tr.shape[1]}"
    )

    print(
        f"Final feature count = "
        f"{X_tr.shape[1]}"
    )

    # --------------------------------------------------------
    # LGBM categorical preparation
    # --------------------------------------------------------

    (
        X_tr_lgb,
        X_va_lgb,
        X_te_lgb,
        cat_cols,
    ) = prepare_lgbm(
        X_tr,
        X_va,
        X_te,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "Training final LightGBM..."
    )

    model = lgb.LGBMClassifier(
        **FINAL_LGB_PARAMS
    )

    model.fit(
        X_tr_lgb,
        y[tr_idx],

        eval_set=[
            (
                X_va_lgb,
                y[va_idx],
            )
        ],

        eval_metric="auc",

        categorical_feature=(
            cat_cols
            if len(cat_cols) > 0
            else "auto"
        ),

        callbacks=[
            lgb.early_stopping(
                150,
                verbose=False,
            )
        ],
    )

    best_iter = (
        model.best_iteration_
        if model.best_iteration_
        else FINAL_LGB_PARAMS[
            "n_estimators"
        ]
    )

    final_iters.append(
        best_iter
    )

    # --------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------

    p_va = (
        model
        .predict_proba(
            X_va_lgb
        )[:, 1]
    )

    # --------------------------------------------------------
    # Test prediction
    # --------------------------------------------------------

    p_te = (
        model
        .predict_proba(
            X_te_lgb
        )[:, 1]
    )

    final_oof[
        va_idx
    ] = p_va

    final_test += (
        p_te
        / FINAL_FOLDS
    )

    fold_auc = roc_auc_score(
        y[va_idx],
        p_va,
    )

    final_fold_scores.append(
        fold_auc
    )

    print(
        f"Fold {fold} AUC = "
        f"{fold_auc:.6f}"
    )

    print(
        f"Best iteration = "
        f"{best_iter}"
    )


# ============================================================
# FINAL RESULT
# ============================================================

final_auc = roc_auc_score(
    y,
    final_oof,
)

final_mean = np.mean(
    final_fold_scores
)

final_std = np.std(
    final_fold_scores
)


print("\n" + "=" * 100)
print("FINAL ROUTE PROFILE RESULT")
print("=" * 100)

for i, score in enumerate(
    final_fold_scores,
    1,
):

    print(
        f"Fold {i}: "
        f"{score:.6f}"
    )

print(
    f"\nFold mean = "
    f"{final_mean:.6f}"
)

print(
    f"Fold std  = "
    f"{final_std:.6f}"
)

print(
    f"OOF AUC   = "
    f"{final_auc:.6f}"
)

print(
    f"\nCurrent LGBM       = "
    f"{CURRENT_LGBM_OOF:.6f}"
)

print(
    f"Current best blend = "
    f"{CURRENT_BEST_BLEND_OOF:.6f}"
)

print(
    f"\nGain vs current LGBM = "
    f"{final_auc - CURRENT_LGBM_OOF:+.6f}"
)

print(
    f"Gain vs current best = "
    f"{final_auc - CURRENT_BEST_BLEND_OOF:+.6f}"
)

print(
    "\nBest iterations:"
)

print(
    final_iters
)


# ============================================================
# SAVE OOF
# ============================================================

oof_output = pd.DataFrame({
    "id":
        train[ID_COL],

    "satisfaction":
        y,

    "prediction":
        final_oof,
})

oof_path = os.path.join(
    OUTPUT_DIR,
    "route_profile_final_oof.csv",
)

oof_output.to_csv(
    oof_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE TEST PREDICTION
# ============================================================

test_output = pd.DataFrame({
    "id":
        test[ID_COL],

    "prediction":
        final_test,
})

test_pred_path = os.path.join(
    OUTPUT_DIR,
    "route_profile_final_test.csv",
)

test_output.to_csv(
    test_pred_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SUBMISSION
# ============================================================

sample = pd.read_csv(
    SAMPLE_PATH
)

submission_target_candidates = [
    c
    for c in sample.columns
    if c != ID_COL
]

assert len(
    submission_target_candidates
) >= 1, (
    "Could not identify submission target column."
)

submission_target = (
    submission_target_candidates[0]
)

submission = pd.DataFrame({
    ID_COL:
        test[ID_COL],

    submission_target:
        final_test,
})

submission_path = os.path.join(
    OUTPUT_DIR,
    "submission_route_profile.csv",
)

submission.to_csv(
    submission_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = pd.DataFrame([
    {
        "metric":
            "current_lgbm_oof",

        "value":
            CURRENT_LGBM_OOF,
    },

    {
        "metric":
            "current_best_blend_oof",

        "value":
            CURRENT_BEST_BLEND_OOF,
    },

    {
        "metric":
            "best_screen_oof",

        "value":
            BEST_SCREEN_AUC,
    },

    {
        "metric":
            "final_route_profile_oof",

        "value":
            final_auc,
    },

    {
        "metric":
            "gain_vs_current_lgbm",

        "value":
            final_auc
            - CURRENT_LGBM_OOF,
    },

    {
        "metric":
            "gain_vs_current_best_blend",

        "value":
            final_auc
            - CURRENT_BEST_BLEND_OOF,
    },

    {
        "metric":
            "selected_experiment",

        "value":
            BEST_EXPERIMENT,
    },

    {
        "metric":
            "selected_groups",

        "value":
            "|".join(
                BEST_GROUPS
            ),
    },

    {
        "metric":
            "final_feature_count",

        "value":
            78
            + sum(
                len(
                    PROFILE_GROUPS[g]
                )
                for g in BEST_GROUPS
            ),
    },
])


summary_path = os.path.join(
    OUTPUT_DIR,
    "route_profile_summary.csv",
)

summary.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# TEST STATISTICS
# ============================================================

print("\n" + "=" * 100)
print("TEST PREDICTION")
print("=" * 100)

print(
    f"min  = "
    f"{final_test.min():.6f}"
)

print(
    f"max  = "
    f"{final_test.max():.6f}"
)

print(
    f"mean = "
    f"{final_test.mean():.6f}"
)

print(
    f"std  = "
    f"{final_test.std():.6f}"
)


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n" + "=" * 100)
print("OUTPUT FILES")
print("=" * 100)

print(
    f"\nAblation results:\n"
    f"{screen_path}"
)

print(
    f"\nSummary:\n"
    f"{summary_path}"
)

print(
    f"\nOOF:\n"
    f"{oof_path}"
)

print(
    f"\nTest prediction:\n"
    f"{test_pred_path}"
)

print(
    f"\nSubmission:\n"
    f"{submission_path}"
)

print("\nDone.")