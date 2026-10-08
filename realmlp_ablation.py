import os
import gc
import time
import warnings
import inspect

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# ============================================================
# REALMLP ABLATION SEARCH
# ============================================================
#
# Existing system:
#
#   78 TE / Conditional-TE / Frequency features
#                    ↓
#              Tuned LGBM
#                    ↓
#              0.960664306
#
#                    +
#
#              RealMLP
#                    ↓
#              0.960313482
#
#                    ↓
#          LGBM + RealMLP
#                    ↓
#              0.960878324
#
#
# This script:
#
#   1. Keeps the LGBM completely unchanged.
#   2. Builds the 78 features ONCE PER OUTER FOLD.
#   3. Reuses the exact same fold features for all 10 RealMLPs.
#   4. Finds the strongest RealMLP configuration.
#   5. Evaluates its ensemble value with the existing LGBM.
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = (
    r"D:\kaggle\airline\data\train.csv"
)

TEST_PATH = (
    r"D:\kaggle\airline\data\test.csv"
)

LGBM_OOF_PATH = (
    r"D:\kaggle\airline\lgbm_realmlp_blend"
    r"\lgbm_realmlp_blend_oof.csv"
)

OUTPUT_DIR = (
    r"D:\kaggle\airline\realmlp_ablation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# BASIC SETTINGS
# ============================================================

RANDOM_STATE = 42

# Screening stage:
# use 3 folds to make the 10-model search practical.
#
# After identifying the winner:
# rerun ONLY the winner with 5 folds.
N_FOLDS = 3

INNER_TE_FOLDS = 5

TARGET = "satisfaction"

ID_COL = "id"


# ============================================================
# CURRENT BENCHMARKS
# ============================================================

CURRENT_LGBM_OOF = 0.960664306

CURRENT_REALMLP_OOF = 0.960313482

CURRENT_BEST_BLEND_OOF = 0.960878324


# ============================================================
# BLEND SEARCH
# ============================================================

COARSE_STEP = 0.01

FINE_HALF_WIDTH = 0.02

FINE_STEP = 0.001


# ============================================================
# RANK BLEND
# ============================================================

RUN_RANK_BLEND = True

RANK_BLEND_TOP_K = 3


# ============================================================
# RESUME
# ============================================================

RESUME = True


# ============================================================
# EXPERIMENT GRID
# ============================================================
#
# E00:
#   Exact current baseline
#
# E01:
#   128 → 256 epochs
#
# E02:
#   AUC-oriented validation metric
#
# E03/E04:
#   learning-rate change
#
# E05/E06:
#   batch-size change
#
# E07/E08:
#   network width
#
# E09:
#   wider + deeper network
#
# ============================================================

EXPERIMENTS = [

    {
        "name": "E00_baseline",

        "n_epochs": 128,

        "batch_size": 2048,

        "hidden_sizes": [
            256,
            256,
            256,
        ],

        "lr": 0.04,

        "val_metric_name": (
            "cross_entropy"
        ),

        "use_ls": False,

        "n_ens": 1,
    },


    {
        "name": "E01_epoch256_CE",

        "n_epochs": 256,

        "batch_size": 2048,

        "hidden_sizes": [
            256,
            256,
            256,
        ],

        "lr": 0.04,

        "val_metric_name": (
            "cross_entropy"
        ),

        "use_ls": False,

        "n_ens": 1,
    },


    {
        "name": "E02_epoch256_AUC",

        "n_epochs": 256,

        "batch_size": 2048,

        "hidden_sizes": [
            256,
            256,
            256,
        ],

        "lr": 0.04,

        "val_metric_name": (
            "1-auc_ovr"
        ),

        "use_ls": False,

        "n_ens": 1,
    },


    {
        "name": "E03_lr003_AUC",

        "n_epochs": 256,

        "batch_size": 2048,

        "hidden_sizes": [
            256,
            256,
            256,
        ],

        "lr": 0.03,

        "val_metric_name": (
            "1-auc_ovr"
        ),

        "use_ls": False,

        "n_ens": 1,
    },


    {
        "name": "E04_lr005_AUC",

        "n_epochs": 256,

        "batch_size": 2048,

        "hidden_sizes": [
            256,
            256,
            256,
        ],

        "lr": 0.05,

        "val_metric_name": (
            "1-auc_ovr"
        ),

        "use_ls": False,

        "n_ens": 1,
    },


    {
        "name": "E05_batch1024_AUC",

        "n_epochs": 256,

        "batch_size": 1024,

        "hidden_sizes": [
            256,
            256,
            256,
        ],

        "lr": 0.04,

        "val_metric_name": (
            "1-auc_ovr"
        ),

        "use_ls": False,

        "n_ens": 1,
    },


    {
        "name": "E06_batch4096_AUC",

        "n_epochs": 256,

        "batch_size": 4096,

        "hidden_sizes": [
            256,
            256,
            256,
        ],

        "lr": 0.04,

        "val_metric_name": (
            "1-auc_ovr"
        ),

        "use_ls": False,

        "n_ens": 1,
    },


    {
        "name": "E07_width128_AUC",

        "n_epochs": 256,

        "batch_size": 2048,

        "hidden_sizes": [
            128,
            128,
            128,
        ],

        "lr": 0.04,

        "val_metric_name": (
            "1-auc_ovr"
        ),

        "use_ls": False,

        "n_ens": 1,
    },


    {
        "name": "E08_width384_AUC",

        "n_epochs": 256,

        "batch_size": 2048,

        "hidden_sizes": [
            384,
            384,
            384,
        ],

        "lr": 0.04,

        "val_metric_name": (
            "1-auc_ovr"
        ),

        "use_ls": False,

        "n_ens": 1,
    },


    {
        "name": "E09_width512_depth4_AUC",

        "n_epochs": 256,

        "batch_size": 2048,

        "hidden_sizes": [
            512,
            512,
            512,
            512,
        ],

        "lr": 0.04,

        "val_metric_name": (
            "1-auc_ovr"
        ),

        "use_ls": False,

        "n_ens": 1,
    },

]


# ============================================================
# IMPORT TORCH
# ============================================================

try:

    import torch

except ImportError:

    raise ImportError(
        "\nPyTorch is not installed."
    )


# ============================================================
# IMPORT PYTABKIT
# ============================================================

try:

    import pytabkit

    from pytabkit import (
        RealMLP_TD_Classifier
    )

except ImportError:

    raise ImportError(
        "\nPyTabKit is not installed.\n"
        "Run:\n"
        "pip install -U pytabkit"
    )


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = "cuda"

    print(
        "CUDA available: YES"
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    DEVICE = "cpu"

    print(
        "CUDA available: NO"
    )

    print(
        "Using CPU."
    )

    print(
        "\nWARNING:"
    )

    print(
        "This experiment contains "
        f"{N_FOLDS} × {len(EXPERIMENTS)} "
        "RealMLP fits."
    )

    print(
        "CPU runtime may be long."
    )


# ============================================================
# PYTABKIT VERSION
# ============================================================

try:

    PYTABKIT_VERSION = (
        pytabkit.__version__
    )

except Exception:

    PYTABKIT_VERSION = (
        "unknown"
    )


print(
    "PyTabKit version:",
    PYTABKIT_VERSION,
)


# ============================================================
# REALMLP API CHECK
# ============================================================

try:

    signature = inspect.signature(
        RealMLP_TD_Classifier
    )

    REALMLP_PARAMS = set(
        signature.parameters.keys()
    )

except Exception:

    REALMLP_PARAMS = set()


print(
    "RealMLP supports n_ens:",
    "n_ens" in REALMLP_PARAMS,
)


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\n" + "=" * 100
)

print(
    "REALMLP ABLATION SEARCH"
)

print(
    "=" * 100
)


train = pd.read_csv(
    TRAIN_PATH
)

test = pd.read_csv(
    TEST_PATH
)


y = (
    train[TARGET]
    .astype(np.int8)
    .to_numpy()
)


print(
    "\nTrain shape:",
    train.shape
)

print(
    "Test shape:",
    test.shape
)

print(
    "Target mean:",
    f"{y.mean():.6f}"
)


# ============================================================
# COLUMN DEFINITIONS
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


# ============================================================
# BUILD BASE
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

        df[
            "Type of Travel"
        ]
        .astype(str)

        + "|"

        + df[
            "Customer Type"
        ]
        .astype(str)

        + "|"

        + df[
            "Class"
        ]
        .astype(str)

    )


    X["arrival_missing"] = (

        df[
            "Arrival Delay in Minutes"
        ]
        .isna()
        .astype("int8")

    )


    return X


X_BASE = build_base(
    train
)

X_TEST_BASE = build_base(
    test
)


# ============================================================
# BASE PREPROCESSING
# ============================================================

BASE_NUMERIC = [

    c

    for c in ORIGINAL_FEATURES

    if c not in CATS

]


BASE_CATEGORICAL = (

    CATS

    + [
        "traveler_segment"
    ]

)


for c in BASE_NUMERIC:

    median_value = (
        train[c].median()
    )


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


    dtype = (

        pd.api.types
        .CategoricalDtype(
            categories=categories
        )

    )


    X_BASE[c] = (
        X_BASE[c].astype(dtype)
    )

    X_TEST_BASE[c] = (
        X_TEST_BASE[c].astype(dtype)
    )


# ============================================================
# TE INPUT
# ============================================================

Z_BASE = train[
    ORIGINAL_FEATURES
].copy()

U_BASE = test[
    ORIGINAL_FEATURES
].copy()


for c in ORIGINAL_FEATURES:

    Z_BASE[c] = (
        Z_BASE[c].astype(str)
    )

    U_BASE[c] = (
        U_BASE[c].astype(str)
    )


# ============================================================
# CONDITIONAL FEATURES
# ============================================================

def build_conditional(df):

    traveler_segment = (

        df[
            "Type of Travel"
        ].astype(str)

        + "|"

        + df[
            "Customer Type"
        ].astype(str)

        + "|"

        + df[
            "Class"
        ].astype(str)

    )


    type_customer_group = (

        df[
            "Type of Travel"
        ].astype(str)

        + "|"

        + df[
            "Customer Type"
        ].astype(str)

    )


    Z = pd.DataFrame(
        index=df.index
    )


    # --------------------------------------------------------
    # Service × traveler segment
    # --------------------------------------------------------

    for c in SERVICES:

        Z[
            c + "__segment"
        ] = (

            df[c].astype(str)

            + "|"

            + traveler_segment

        )


    # --------------------------------------------------------
    # Service pair × type/customer
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


Z_COND = build_conditional(
    train
)

U_COND = build_conditional(
    test
)


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

                df[
                    numeric_col
                ]
                .astype(str)

                + "|"

                + df[
                    group_col
                ].astype(str)

            )


    return Z


Z_NUM_GROUP = (

    build_numeric_groups(
        train
    )

)


U_NUM_GROUP = (

    build_numeric_groups(
        test
    )

)


# ============================================================
# FREQUENCY FEATURES
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
            .replace(
                " ",
                "_",
            )
            .replace(
                "/",
                "_",
            )
            .replace(
                "-",
                "_",
            )

        )


        tr_freq = (

            train_df[c]
            .astype(str)
            .map(freq_map)
            .fillna(0)
            .to_numpy()

        )


        va_freq = (

            val_df[c]
            .astype(str)
            .map(freq_map)
            .fillna(0)
            .to_numpy()

        )


        te_freq = (

            test_df[c]
            .astype(str)
            .map(freq_map)
            .fillna(0)
            .to_numpy()

        )


        train_out[
            feature_name
        ] = (

            np.log1p(
                tr_freq * 1_000_000
            )
            .astype("float32")

        )


        val_out[
            feature_name
        ] = (

            np.log1p(
                va_freq * 1_000_000
            )
            .astype("float32")

        )


        test_out[
            feature_name
        ] = (

            np.log1p(
                te_freq * 1_000_000
            )
            .astype("float32")

        )


    return (

        train_out,

        val_out,

        test_out,

    )


# ============================================================
# BUILD THE 78 FEATURES
# ============================================================

def build_enriched_fold(

    tr_idx,

    va_idx,

):

    X_tr = (

        X_BASE
        .iloc[tr_idx]
        .copy()

    )


    X_va = (

        X_BASE
        .iloc[va_idx]
        .copy()

    )


    X_te = (
        X_TEST_BASE.copy()
    )


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


    tr_te = (

        enc_base
        .fit_transform(

            Z_BASE.iloc[tr_idx],

            y[tr_idx],

        )

        .astype("float32")

    )


    va_te = (

        enc_base
        .transform(

            Z_BASE.iloc[va_idx]

        )

        .astype("float32")

    )


    te_te = (

        enc_base
        .transform(

            U_BASE

        )

        .astype("float32")

    )


    for j, c in enumerate(
        ORIGINAL_FEATURES
    ):

        name = (

            "te_"

            + c
            .replace(
                " ",
                "_",
            )
            .replace(
                "/",
                "_",
            )
            .replace(
                "-",
                "_",
            )

        )


        X_tr[name] = (
            tr_te[:, j]
        )

        X_va[name] = (
            va_te[:, j]
        )

        X_te[name] = (
            te_te[:, j]
        )


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


    tr_cond = (

        enc_cond
        .fit_transform(

            Z_COND.iloc[tr_idx],

            y[tr_idx],

        )

        .astype("float32")

    )


    va_cond = (

        enc_cond
        .transform(

            Z_COND.iloc[va_idx]

        )

        .astype("float32")

    )


    te_cond = (

        enc_cond
        .transform(

            U_COND

        )

        .astype("float32")

    )


    for j in range(
        Z_COND.shape[1]
    ):

        name = (
            "cond_te_" + str(j)
        )


        X_tr[name] = (
            tr_cond[:, j]
        )

        X_va[name] = (
            va_cond[:, j]
        )

        X_te[name] = (
            te_cond[:, j]
        )


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


    tr_num = (

        enc_num
        .fit_transform(

            Z_NUM_GROUP.iloc[tr_idx],

            y[tr_idx],

        )

        .astype("float32")

    )


    va_num = (

        enc_num
        .transform(

            Z_NUM_GROUP.iloc[va_idx]

        )

        .astype("float32")

    )


    te_num = (

        enc_num
        .transform(

            U_NUM_GROUP

        )

        .astype("float32")

    )


    for j in range(
        Z_NUM_GROUP.shape[1]
    ):

        name = (
            "ng_te_" + str(j)
        )


        X_tr[name] = (
            tr_num[:, j]
        )

        X_va[name] = (
            va_num[:, j]
        )

        X_te[name] = (
            te_num[:, j]
        )


    # ========================================================
    # 4. FREQUENCY FEATURES
    # ========================================================

    freq_tr_input = pd.concat(

        [

            (
                Z_BASE
                .iloc[tr_idx]
                [NUMERIC]
                .reset_index(
                    drop=True
                )
            ),

            (
                Z_NUM_GROUP
                .iloc[tr_idx]
                .reset_index(
                    drop=True
                )
            ),

        ],

        axis=1,

    )


    freq_va_input = pd.concat(

        [

            (
                Z_BASE
                .iloc[va_idx]
                [NUMERIC]
                .reset_index(
                    drop=True
                )
            ),

            (
                Z_NUM_GROUP
                .iloc[va_idx]
                .reset_index(
                    drop=True
                )
            ),

        ],

        axis=1,

    )


    freq_te_input = pd.concat(

        [

            (
                U_BASE
                [NUMERIC]
                .reset_index(
                    drop=True
                )
            ),

            (
                U_NUM_GROUP
                .reset_index(
                    drop=True
                )
            ),

        ],

        axis=1,

    )


    (
        freq_tr,
        freq_va,
        freq_te,
    ) = build_frequency_features(

        freq_tr_input,

        freq_va_input,

        freq_te_input,

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

            [
                np.inf,
                -np.inf,
            ],

            np.nan,

            inplace=True,

        )


    numeric_cols = (

        X_tr
        .select_dtypes(

            include=[

                "float32",
                "float64",
                "int8",
                "int16",
                "int32",
                "int64",

            ]

        )
        .columns

    )


    for c in numeric_cols:

        median_value = (
            X_tr[c].median()
        )


        if pd.isna(
            median_value
        ):

            median_value = 0.0


        X_tr[c] = (

            X_tr[c]
            .fillna(median_value)
            .astype("float32")

        )


        X_va[c] = (

            X_va[c]
            .fillna(median_value)
            .astype("float32")

        )


        X_te[c] = (

            X_te[c]
            .fillna(median_value)
            .astype("float32")

        )


    # ========================================================
    # COLUMN CONSISTENCY
    # ========================================================

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


    if X_tr.shape[1] != 78:

        raise RuntimeError(

            "Expected 78 features, "
            f"got {X_tr.shape[1]}"

        )


    return (

        X_tr,

        X_va,

        X_te,

    )


# ============================================================
# PREPARE REALMLP DATA
# ============================================================

def prepare_realmlp(

    X_tr,

    X_va,

    X_te,

):

    X_tr = X_tr.copy()

    X_va = X_va.copy()

    X_te = X_te.copy()


    categorical_cols = []


    # --------------------------------------------------------
    # Detect categoricals
    # --------------------------------------------------------

    for c in X_tr.columns:

        dtype_string = (
            str(X_tr[c].dtype)
        )


        if (

            X_tr[c].dtype
            == "object"

            or

            dtype_string.startswith(
                "category"
            )

        ):

            categorical_cols.append(
                c
            )


    # --------------------------------------------------------
    # Align categories
    # --------------------------------------------------------

    for c in categorical_cols:

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


        dtype = (

            pd.api.types
            .CategoricalDtype(
                categories=categories
            )

        )


        X_tr[c] = (
            X_tr[c].astype(dtype)
        )

        X_va[c] = (
            X_va[c].astype(dtype)
        )

        X_te[c] = (
            X_te[c].astype(dtype)
        )


    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for c in X_tr.columns:

        if c in categorical_cols:
            continue


        X_tr[c] = pd.to_numeric(

            X_tr[c],

            errors="coerce",

        )


        X_va[c] = pd.to_numeric(

            X_va[c],

            errors="coerce",

        )


        X_te[c] = pd.to_numeric(

            X_te[c],

            errors="coerce",

        )


    # --------------------------------------------------------
    # Numeric imputation
    # --------------------------------------------------------

    for c in X_tr.columns:

        if c in categorical_cols:
            continue


        median_value = (
            X_tr[c].median()
        )


        if pd.isna(
            median_value
        ):

            median_value = 0.0


        X_tr[c] = (

            X_tr[c]
            .fillna(median_value)
            .astype("float32")

        )


        X_va[c] = (

            X_va[c]
            .fillna(median_value)
            .astype("float32")

        )


        X_te[c] = (

            X_te[c]
            .fillna(median_value)
            .astype("float32")

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


    return (

        X_tr,

        X_va,

        X_te,

        categorical_cols,

    )


# ============================================================
# LOAD EXISTING LGBM OOF
# ============================================================

def load_lgbm_oof():

    if not os.path.exists(
        LGBM_OOF_PATH
    ):

        raise FileNotFoundError(

            "\nLGBM OOF file not found:\n"

            f"{LGBM_OOF_PATH}"

        )


    df = pd.read_csv(
        LGBM_OOF_PATH
    )


    print(
        "\nLoaded existing OOF:"
    )

    print(
        LGBM_OOF_PATH
    )


    print(
        "Columns:",
        list(df.columns)
    )


    # ========================================================
    # ROW COUNT
    # ========================================================

    if len(df) != len(train):

        raise RuntimeError(

            "\nLGBM OOF row count mismatch.\n"

            f"Expected: {len(train)}\n"

            f"Got: {len(df)}"

        )


    # ========================================================
    # REQUIRED COLUMN
    # ========================================================

    # Your actual file uses:
    #
    # ['id',
    #  'satisfaction',
    #  'lgbm',
    #  'realmlp',
    #  'blend_prediction']
    #
    # We ONLY use "lgbm".
    #
    # ========================================================

    if "lgbm" not in df.columns:

        raise RuntimeError(

            "\nThe OOF file does not contain "
            "'lgbm'.\n"

            f"Available columns: "
            f"{list(df.columns)}"

        )


    # ========================================================
    # ID ALIGNMENT
    # ========================================================

    if ID_COL not in df.columns:

        raise RuntimeError(
            "OOF file does not contain id."
        )


    existing_ids = (

        df[ID_COL]
        .astype(str)
        .to_numpy()

    )


    train_ids = (

        train[ID_COL]
        .astype(str)
        .to_numpy()

    )


    if np.array_equal(
        existing_ids,
        train_ids,
    ):

        lgbm_pred = (

            pd.to_numeric(

                df["lgbm"],

                errors="coerce",

            )

            .to_numpy(
                dtype=np.float64
            )

        )


    else:

        print(
            "\nID order differs."
        )

        print(
            "Aligning LGBM OOF by ID..."
        )


        mapping = pd.DataFrame({

            ID_COL:
                existing_ids,

            "_lgbm":
                pd.to_numeric(

                    df["lgbm"],

                    errors="coerce",

                ),

        })


        aligned = (

            train[
                [ID_COL]
            ]
            .astype(str)

            .merge(

                mapping,

                on=ID_COL,

                how="left",

                sort=False,

            )

        )


        lgbm_pred = (

            aligned["_lgbm"]

            .to_numpy(
                dtype=np.float64
            )

        )


    # ========================================================
    # VALIDATE
    # ========================================================

    if not np.isfinite(
        lgbm_pred
    ).all():

        raise RuntimeError(
            "LGBM OOF contains NaN/inf."
        )


    # ========================================================
    # VERIFY SCORE
    # ========================================================

    lgbm_auc = (
        roc_auc_score(
            y,
            lgbm_pred,
        )
    )


    print(
        "\nVerified LGBM OOF AUC:",
        f"{lgbm_auc:.9f}"
    )


    print(
        "Expected LGBM OOF AUC :",
        f"{CURRENT_LGBM_OOF:.9f}"
    )


    print(
        "Difference            :",
        f"{lgbm_auc - CURRENT_LGBM_OOF:+.9f}"
    )


    return lgbm_pred


lgbm_oof = load_lgbm_oof()


# ============================================================
# BLEND SEARCH
# ============================================================

def search_best_raw_blend(

    lgbm_pred,

    mlp_pred,

):

    # --------------------------------------------------------
    # Coarse search
    # --------------------------------------------------------

    coarse_weights = np.arange(

        0.0,

        1.000001,

        COARSE_STEP,

    )


    coarse_scores = []


    for w in coarse_weights:

        blend = (

            w * lgbm_pred

            +

            (1.0 - w)
            * mlp_pred

        )


        auc = (
            roc_auc_score(
                y,
                blend,
            )
        )


        coarse_scores.append(
            auc
        )


    coarse_scores = np.asarray(
        coarse_scores
    )


    best_idx = int(
        np.argmax(
            coarse_scores
        )
    )


    coarse_best_w = float(

        coarse_weights[
            best_idx
        ]

    )


    # --------------------------------------------------------
    # Fine search
    # --------------------------------------------------------

    low = max(

        0.0,

        coarse_best_w
        - FINE_HALF_WIDTH,

    )


    high = min(

        1.0,

        coarse_best_w
        + FINE_HALF_WIDTH,

    )


    fine_weights = np.arange(

        low,

        high
        + FINE_STEP / 2.0,

        FINE_STEP,

    )


    fine_scores = []


    for w in fine_weights:

        blend = (

            w * lgbm_pred

            +

            (1.0 - w)
            * mlp_pred

        )


        auc = (
            roc_auc_score(
                y,
                blend,
            )
        )


        fine_scores.append(
            auc
        )


    fine_scores = np.asarray(
        fine_scores
    )


    best_idx = int(
        np.argmax(
            fine_scores
        )
    )


    best_w_lgbm = float(

        fine_weights[
            best_idx
        ]

    )


    best_w_mlp = (

        1.0
        - best_w_lgbm

    )


    best_auc = float(

        fine_scores[
            best_idx
        ]

    )


    return (

        best_w_lgbm,

        best_w_mlp,

        best_auc,

    )


# ============================================================
# RANK TRANSFORM
# ============================================================

def rank_transform(
    pred
):

    return (

        pd.Series(
            pred
        )

        .rank(
            method="average"
        )

        .to_numpy(
            dtype=np.float64
        )

        / len(pred)

    )


def search_best_rank_blend(

    lgbm_pred,

    mlp_pred,

):

    lgbm_rank = (
        rank_transform(
            lgbm_pred
        )
    )


    mlp_rank = (
        rank_transform(
            mlp_pred
        )
    )


    return (
        search_best_raw_blend(

            lgbm_rank,

            mlp_rank,

        )
    )


# ============================================================
# RESULT FILE
# ============================================================

RESULTS_PATH = os.path.join(

    OUTPUT_DIR,

    "realmlp_ablation_results.csv",

)


# ============================================================
# LOAD PREVIOUS RESULTS
# ============================================================

if (

    RESUME

    and

    os.path.exists(
        RESULTS_PATH
    )

):

    previous_results = (
        pd.read_csv(
            RESULTS_PATH
        )
    )

else:

    previous_results = (
        pd.DataFrame()
    )


completed_names = set()


if (

    not previous_results.empty

    and

    "name"
    in previous_results.columns

):

    completed_names = set(

        previous_results[
            "name"
        ]
        .astype(str)

    )


print(
    "\nCompleted experiments:",
    len(completed_names)
)


# ============================================================
# OUTER CV
# ============================================================

skf = StratifiedKFold(

    n_splits=N_FOLDS,

    shuffle=True,

    random_state=RANDOM_STATE,

)


fold_indices = list(

    skf.split(

        np.zeros(
            len(y)
        ),

        y,

    )

)


# ============================================================
# PREDICTION STORAGE
# ============================================================

n_train = len(train)

n_test = len(test)


oof_predictions = {

    exp["name"]:
        np.zeros(
            n_train,
            dtype=np.float32
        )

    for exp in EXPERIMENTS

}


test_predictions = {

    exp["name"]:
        np.zeros(
            n_test,
            dtype=np.float32
        )

    for exp in EXPERIMENTS

}


# ============================================================
# LOAD COMPLETED PREDICTIONS WHEN RESUMING
# ============================================================

def load_saved_prediction(
    name,
    kind,
):

    if kind == "oof":

        path = os.path.join(

            OUTPUT_DIR,

            f"{name}_oof.csv",

        )

    else:

        path = os.path.join(

            OUTPUT_DIR,

            f"{name}_test.csv",

        )


    if not os.path.exists(path):

        return None


    try:

        df = pd.read_csv(
            path
        )


        if len(df) != (

            n_train
            if kind == "oof"
            else n_test

        ):

            return None


        if "prediction" not in df.columns:

            return None


        pred = (

            pd.to_numeric(

                df[
                    "prediction"
                ],

                errors="coerce",

            )

            .to_numpy(
                dtype=np.float32
            )

        )


        if not np.isfinite(
            pred
        ).all():

            return None


        return pred


    except Exception:

        return None


for exp in EXPERIMENTS:

    name = exp["name"]


    if name in completed_names:

        oof_saved = (
            load_saved_prediction(
                name,
                "oof"
            )
        )


        test_saved = (
            load_saved_prediction(
                name,
                "test"
            )
        )


        if (

            oof_saved is not None

            and

            test_saved is not None

        ):

            oof_predictions[
                name
            ] = oof_saved


            test_predictions[
                name
            ] = test_saved


# ============================================================
# MODEL CONSTRUCTOR
# ============================================================

def make_realmlp(

    params,

    fold,

):

    kwargs = {

        "device":
            DEVICE,

        # Same seed across experiments
        # within the same fold so the ablation is fair.
        "random_state":
            RANDOM_STATE
            + fold,

        # Outer CV is handled manually.
        "n_cv":
            1,

        "n_refit":
            0,

        "n_epochs":
            params[
                "n_epochs"
            ],

        "batch_size":
            params[
                "batch_size"
            ],

        "hidden_sizes":
            params[
                "hidden_sizes"
            ],

        "lr":
            params[
                "lr"
            ],

        "val_metric_name":
            params[
                "val_metric_name"
            ],

        "use_ls":
            params[
                "use_ls"
            ],

        "verbosity":
            1,

    }


    # n_ens exists in your installed version.
    if (
        "n_ens"
        in REALMLP_PARAMS
    ):

        kwargs[
            "n_ens"
        ] = params[
            "n_ens"
        ]


    return (
        RealMLP_TD_Classifier(
            **kwargs
        )
    )


# ============================================================
# MAIN TRAINING LOOP
# ============================================================
#
# VERY IMPORTANT:
#
# Fold 1:
#   build 78 features ONCE
#       ↓
#   E00
#   E01
#   E02
#   ...
#   E09
#
# Fold 2:
#   build 78 features ONCE
#       ↓
#   E00
#   ...
#
# ============================================================

global_start = time.time()

completed_model_count = (
    len(completed_names)
)

total_model_count = (
    N_FOLDS
    * len(EXPERIMENTS)
)


for fold, (
    tr_idx,
    va_idx,
) in enumerate(

    fold_indices,

    1,

):

    print(
        "\n\n" + "=" * 110
    )

    print(
        f"BUILDING CACHED 78 FEATURES "
        f"— FOLD {fold}/{N_FOLDS}"
    )

    print(
        "=" * 110
    )


    # ========================================================
    # BUILD FEATURES ONCE
    # ========================================================

    feature_start = time.time()


    (

        X_tr_raw,

        X_va_raw,

        X_te_raw,

    ) = build_enriched_fold(

        tr_idx,

        va_idx,

    )


    print(
        "\n78 features built."
    )


    print(
        "Shape:",
        X_tr_raw.shape,
    )


    # ========================================================
    # PREPARE ONCE
    # ========================================================

    (

        X_tr,

        X_va,

        X_te,

        cat_cols,

    ) = prepare_realmlp(

        X_tr_raw,

        X_va_raw,

        X_te_raw,

    )


    feature_time = (

        time.time()
        - feature_start

    ) / 60.0


    print(
        "Feature preparation time:",
        f"{feature_time:.2f} min"
    )


    print(
        "Categorical columns:",
        cat_cols
    )


    # ========================================================
    # FREE RAW COPIES
    # ========================================================

    del X_tr_raw

    del X_va_raw

    del X_te_raw

    gc.collect()


    # ========================================================
    # RUN ALL REALMLP EXPERIMENTS
    # ========================================================

    for exp_id, params in enumerate(

        EXPERIMENTS,

        1,

    ):

        name = params["name"]


        # ----------------------------------------------------
        # RESUME SKIP
        # ----------------------------------------------------

        if name in completed_names:

            print(

                f"\n[{exp_id}/"
                f"{len(EXPERIMENTS)}] "
                f"Skipping completed: "
                f"{name}"

            )

            continue


        model_global_start = time.time()


        print(
            "\n" + "-" * 110
        )


        print(
            f"EXPERIMENT "
            f"{exp_id}/{len(EXPERIMENTS)}"
        )


        print(
            f"FOLD "
            f"{fold}/{N_FOLDS}"
        )


        print(
            name
        )


        print(
            "-" * 110
        )


        print(
            "epochs      :",
            params["n_epochs"]
        )


        print(
            "batch_size  :",
            params["batch_size"]
        )


        print(
            "hidden      :",
            params["hidden_sizes"]
        )


        print(
            "lr          :",
            params["lr"]
        )


        print(
            "metric      :",
            params["val_metric_name"]
        )


        # ====================================================
        # MODEL
        # ====================================================

        model = make_realmlp(

            params,

            fold,

        )


        # ====================================================
        # TRAIN
        # ====================================================

        print(
            "\nTraining..."
        )


        model.fit(

            X_tr,

            y[
                tr_idx
            ],

            X_va,

            y[
                va_idx
            ],

            cat_col_names=cat_cols,

        )


        # ====================================================
        # VALIDATION
        # ====================================================

        val_pred = (

            model
            .predict_proba(
                X_va
            )[:, 1]

            .astype(
                np.float32
            )

        )


        # ====================================================
        # TEST
        # ====================================================

        fold_test_pred = (

            model
            .predict_proba(
                X_te
            )[:, 1]

            .astype(
                np.float32
            )

        )


        # ====================================================
        # STORE
        # ====================================================

        oof_predictions[
            name
        ][
            va_idx
        ] = val_pred


        test_predictions[
            name
        ] += (

            fold_test_pred

            / N_FOLDS

        )


        # ====================================================
        # FOLD AUC
        # ====================================================

        fold_auc = (

            roc_auc_score(

                y[
                    va_idx
                ],

                val_pred,

            )

        )


        print(
            "\nFold AUC:",
            f"{fold_auc:.9f}"
        )


        # ====================================================
        # CLEAN MODEL
        # ====================================================

        del model

        gc.collect()


        if DEVICE == "cuda":

            torch.cuda.empty_cache()


        # ====================================================
        # RUNTIME ESTIMATE
        # ====================================================

        completed_model_count += 1


        model_elapsed_min = (

            time.time()
            - model_global_start

        ) / 60.0


        global_elapsed_min = (

            time.time()
            - global_start

        ) / 60.0


        remaining_models = max(

            0,

            total_model_count
            - completed_model_count,

        )


        if completed_model_count > 0:

            avg_min = (

                global_elapsed_min
                / completed_model_count

            )


            estimated_remaining = (

                avg_min
                * remaining_models

            )

        else:

            estimated_remaining = 0


        print(
            "\nThis model time:",
            f"{model_elapsed_min:.2f} min"
        )


        print(
            "Total elapsed:",
            f"{global_elapsed_min:.2f} min"
        )


        print(
            "Estimated remaining:",
            f"{estimated_remaining:.2f} min"
        )


    # ========================================================
    # FREE CURRENT FOLD FEATURES
    # ========================================================

    del X_tr

    del X_va

    del X_te

    gc.collect()


    if DEVICE == "cuda":

        torch.cuda.empty_cache()


    print(
        "\nFinished fold",
        fold
    )


# ============================================================
# CALCULATE FINAL OOF RESULTS
# ============================================================

print(
    "\n\n" + "=" * 110
)

print(
    "CALCULATING FINAL OOF RESULTS"
)

print(
    "=" * 110
)


all_results = []


for params in EXPERIMENTS:

    name = params["name"]


    # --------------------------------------------------------
    # Get OOF
    # --------------------------------------------------------

    if (
        not np.isfinite(
            oof_predictions[
                name
            ]
        ).all()
    ):

        print(
            f"WARNING: {name} has invalid predictions."
        )

        continue


    exp_oof = (
        oof_predictions[
            name
        ]
    )


    # --------------------------------------------------------
    # Standalone AUC
    # --------------------------------------------------------

    mlp_auc = (

        roc_auc_score(

            y,

            exp_oof,

        )

    )


    # --------------------------------------------------------
    # Fold-independent prediction correlation
    # --------------------------------------------------------

    prediction_corr = (

        np.corrcoef(

            lgbm_oof,

            exp_oof,

        )[0, 1]

    )


    # --------------------------------------------------------
    # Raw blend
    # --------------------------------------------------------

    (

        best_w_lgbm,

        best_w_mlp,

        best_blend_auc,

    ) = search_best_raw_blend(

        lgbm_oof,

        exp_oof,

    )


    # --------------------------------------------------------
    # Gains
    # --------------------------------------------------------

    gain_vs_lgbm = (

        best_blend_auc

        - lgbm_auc

    )


    gain_vs_old_blend = (

        best_blend_auc

        - CURRENT_BEST_BLEND_OOF

    )


    gain_vs_current_realmlp = (

        mlp_auc

        - CURRENT_REALMLP_OOF

    )


    all_results.append({

        "name":
            name,

        "n_epochs":
            params["n_epochs"],

        "batch_size":
            params["batch_size"],

        "hidden_sizes":
            str(
                params[
                    "hidden_sizes"
                ]
            ),

        "lr":
            params["lr"],

        "val_metric_name":
            params[
                "val_metric_name"
            ],

        "n_ens":
            params["n_ens"],

        "realmlp_oof_auc":
            mlp_auc,

        "gain_realmlp_vs_old":
            gain_vs_current_realmlp,

        "prediction_corr_lgbm":
            prediction_corr,

        "best_lgbm_weight":
            best_w_lgbm,

        "best_realmlp_weight":
            best_w_mlp,

        "best_raw_blend_auc":
            best_blend_auc,

        "gain_vs_lgbm":
            gain_vs_lgbm,

        "gain_vs_old_blend":
            gain_vs_old_blend,

    })


# ============================================================
# SORT RESULTS
# ============================================================

results = (

    pd.DataFrame(
        all_results
    )

    .sort_values(

        "best_raw_blend_auc",

        ascending=False,

    )

    .reset_index(
        drop=True
    )

)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(

    RESULTS_PATH,

    index=False,

    encoding="utf-8-sig",

)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n" + "=" * 110
)

print(
    "FINAL REALMLP ABLATION RANKING"
)

print(
    "=" * 110
)


print(

    results[

        [

            "name",

            "realmlp_oof_auc",

            "gain_realmlp_vs_old",

            "prediction_corr_lgbm",

            "best_lgbm_weight",

            "best_realmlp_weight",

            "best_raw_blend_auc",

            "gain_vs_lgbm",

            "gain_vs_old_blend",

        ]

    ].to_string(
        index=False
    )

)


# ============================================================
# SAVE ALL OOF PREDICTIONS
# ============================================================

print(
    "\nSaving OOF prediction matrix..."
)


oof_matrix = pd.DataFrame({

    ID_COL:
        train[
            ID_COL
        ],

    TARGET:
        y,

    "lgbm":
        lgbm_oof,

})


for params in EXPERIMENTS:

    name = params["name"]


    if name in oof_predictions:

        oof_matrix[
            name
        ] = oof_predictions[
            name
        ]


oof_matrix_path = os.path.join(

    OUTPUT_DIR,

    "all_realmlp_oof_predictions.csv",

)


oof_matrix.to_csv(

    oof_matrix_path,

    index=False,

    encoding="utf-8-sig",

)


print(
    "Saved:",
    oof_matrix_path
)


# ============================================================
# CORRELATION MATRIX
# ============================================================

print(
    "\nCalculating prediction correlation matrix..."
)


prediction_columns = [
    "lgbm"
]


prediction_columns += [

    exp["name"]

    for exp in EXPERIMENTS

]


corr_matrix = (

    oof_matrix[
        prediction_columns
    ]

    .corr()

)


corr_path = os.path.join(

    OUTPUT_DIR,

    "realmlp_prediction_correlation.csv",

)


corr_matrix.to_csv(

    corr_path,

    encoding="utf-8-sig",

)


print(
    "Saved:",
    corr_path
)


# ============================================================
# RANK BLEND FOR TOP K
# ============================================================

if RUN_RANK_BLEND:

    print(
        "\n" + "=" * 110
    )

    print(
        f"RANK BLEND — TOP {RANK_BLEND_TOP_K}"
    )

    print(
        "=" * 110
    )


    top_names = (

        results
        .head(
            RANK_BLEND_TOP_K
        )["name"]
        .astype(str)
        .tolist()

    )


    rank_results = []


    for name in top_names:

        mlp_pred = (
            oof_predictions[
                name
            ]
        )


        (

            rank_w_lgbm,

            rank_w_mlp,

            rank_auc,

        ) = search_best_rank_blend(

            lgbm_oof,

            mlp_pred,

        )


        raw_row = (

            results[
                results["name"]
                == name
            ]

            .iloc[0]

        )


        raw_auc = float(

            raw_row[
                "best_raw_blend_auc"
            ]

        )


        rank_results.append({

            "name":
                name,

            "rank_lgbm_weight":
                rank_w_lgbm,

            "rank_realmlp_weight":
                rank_w_mlp,

            "rank_blend_auc":
                rank_auc,

            "raw_blend_auc":
                raw_auc,

            "rank_minus_raw":
                rank_auc - raw_auc,

        })


    rank_df = (

        pd.DataFrame(
            rank_results
        )

        .sort_values(

            "rank_blend_auc",

            ascending=False,

        )

        .reset_index(
            drop=True
        )

    )


    rank_path = os.path.join(

        OUTPUT_DIR,

        "rank_blend_results.csv",

    )


    rank_df.to_csv(

        rank_path,

        index=False,

        encoding="utf-8-sig",

    )


    print(

        rank_df.to_string(
            index=False
        )

    )


# ============================================================
# FIND BEST MODEL
# ============================================================

best_row = (
    results.iloc[0]
)

best_name = str(
    best_row["name"]
)


best_oof = (
    oof_predictions[
        best_name
    ]
)


best_test = (
    test_predictions[
        best_name
    ]
)


best_w_lgbm = float(

    best_row[
        "best_lgbm_weight"
    ]

)


best_w_mlp = float(

    best_row[
        "best_realmlp_weight"
    ]

)


# ============================================================
# BEST RAW OOF BLEND
# ============================================================

best_blend_oof = (

    best_w_lgbm
    * lgbm_oof

    +

    best_w_mlp
    * best_oof

)


best_blend_auc = (

    roc_auc_score(

        y,

        best_blend_oof,

    )

)


best_oof_path = os.path.join(

    OUTPUT_DIR,

    "BEST_LGBM_REALMLP_OOF.csv",

)


pd.DataFrame({

    ID_COL:
        train[
            ID_COL
        ],

    TARGET:
        y,

    "lgbm":
        lgbm_oof,

    "realmlp":
        best_oof,

    "blend_prediction":
        best_blend_oof,

}).to_csv(

    best_oof_path,

    index=False,

    encoding="utf-8-sig",

)


# ============================================================
# BEST REALMLP TEST PREDICTION
# ============================================================

best_test_path = os.path.join(

    OUTPUT_DIR,

    "BEST_REALMLP_test.csv",

)


pd.DataFrame({

    ID_COL:
        test[
            ID_COL
        ],

    "prediction":
        best_test,

}).to_csv(

    best_test_path,

    index=False,

    encoding="utf-8-sig",

)


# ============================================================
# FINAL SUMMARY
# ============================================================

total_runtime_min = (

    time.time()
    - global_start

) / 60.0


print(
    "\n" + "=" * 110
)

print(
    "FINAL BEST RESULT"
)

print(
    "=" * 110
)


print(
    "\nBest RealMLP:",
    best_name
)


print(
    "RealMLP OOF AUC:",
    f"{best_row['realmlp_oof_auc']:.9f}"
)


print(
    "LGBM OOF AUC:",
    f"{lgbm_auc:.9f}"
)


print(
    "Prediction corr:",
    f"{best_row['prediction_corr_lgbm']:.9f}"
)


print(
    "Best LGBM weight:",
    f"{best_w_lgbm:.3f}"
)


print(
    "Best RealMLP weight:",
    f"{best_w_mlp:.3f}"
)


print(
    "Best blend OOF AUC:",
    f"{best_blend_auc:.9f}"
)


print(
    "Gain vs LGBM:",
    f"{best_blend_auc - lgbm_auc:+.9f}"
)


print(
    "Gain vs old 0.960878324:",
    f"{best_blend_auc - CURRENT_BEST_BLEND_OOF:+.9f}"
)


print(
    "\nTotal runtime:",
    f"{total_runtime_min:.2f} min"
)


print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)


print(
    "\nMain ranking:"
)

print(
    RESULTS_PATH
)


print(
    "\nAll OOF predictions:"
)

print(
    oof_matrix_path
)


print(
    "\nCorrelation matrix:"
)

print(
    corr_path
)


print(
    "\nBest OOF blend:"
)

print(
    best_oof_path
)


print(
    "\nDone."
)