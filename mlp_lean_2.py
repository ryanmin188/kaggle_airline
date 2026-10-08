import os
import gc
import io
import time
import contextlib
import warnings

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr

from pytabkit import RealMLP_TD_Classifier


warnings.filterwarnings("ignore")


# ============================================================
# LEAN REALMLP V2
# INTERACTION-ORIENTED REPRESENTATION
#
# Competition-only.
#
# NO external 129k dataset.
# NO Route Profile.
# NO XGBoost.
#
# V1:
#   raw 21
#   + numeric -> categorical
#   + 5 counts
#   + 3 interactions
#   + 3 TE
#
# V2:
#   raw 21
#   + numeric -> categorical
#   + 5 counts
#   + broader interaction representation
#   + NO target encoding
#
# Goal:
#   Make V2 structurally different from V1.
#
# Final:
#   Existing LGBM
#   + Existing RealMLP V1
#   + New RealMLP V2
#
# Main purpose:
#   Test whether V2 adds ensemble diversity.
# ============================================================


# ============================================================
# PATHS
# ============================================================

DATA_DIR = r"D:\kaggle\airline\data"

TRAIN_PATH = os.path.join(
    DATA_DIR,
    "train.csv",
)

TEST_PATH = os.path.join(
    DATA_DIR,
    "test.csv",
)

SAMPLE_PATH = os.path.join(
    DATA_DIR,
    "sample_submission.csv",
)


# ============================================================
# CURRENT LGBM
# ============================================================

LGBM_OOF_PATH = (
    r"D:\kaggle\airline\conditional_te_experiment"
    r"\conditional_te_lgbm_oof.csv"
)

LGBM_TEST_PATH = (
    r"D:\kaggle\airline\conditional_te_experiment"
    r"\conditional_te_lgbm_test.csv"
)


# ============================================================
# EXISTING LEAN REALMLP V1
# ============================================================

V1_OOF_PATH = (
    r"D:\kaggle\airline\lean_realmlp_blend"
    r"\lean_realmlp_oof.csv"
)

V1_TEST_PATH = (
    r"D:\kaggle\airline\lean_realmlp_blend"
    r"\lean_realmlp_test.csv"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    r"D:\kaggle\airline\lean_realmlp_v2"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

N_FOLDS = 5


# ============================================================
# REALMLP CONFIG
#
# Keep the same core configuration as V1.
#
# We deliberately change representation, not model family.
# ============================================================

REALMLP_PARAMS = {

    "random_state":
        SEED,

    "verbosity":
        0,

    "val_metric_name":
        "1-auc_ovr",

    "n_ens":
        8,

    "n_epochs":
        3,

    "batch_size":
        256,

    "use_early_stopping":
        False,

    "early_stopping_additive_patience":
        10,

    "early_stopping_multiplicative_patience":
        1,

    "lr":
        0.053,

    "wd":
        0.015,

    "sq_mom":
        0.988,

    "lr_sched":
        "flat_anneal",

    "wd_sched":
        "cos_log_15",

    "first_layer_lr_factor":
        0.25,

    "embedding_size":
        5,

    "max_one_hot_cat_size":
        18,

    "hidden_sizes":
        [
            512,
            256,
            128,
        ],

    "act":
        "silu",

    "p_drop":
        0.05,

    "p_drop_sched":
        "invsqrtp1e-3",

    "plr_hidden_1":
        16,

    "plr_hidden_2":
        8,

    "plr_act_name":
        "gelu",

    "plr_lr_factor":
        0.1151,

    "plr_sigma":
        2.33,

    "ls_eps":
        0.01,

    "ls_eps_sched":
        "sqrt_cos",

    "add_front_scale":
        False,

    "bias_init_mode":
        "neg-uniform-dynamic-2",

    "tfms":
        [
            "one_hot",
            "median_center",
            "robust_scale",
            "smooth_clip",
            "embedding",
            "l2_normalize",
        ],
}


# ============================================================
# PRIMARY 3-MODEL BLEND
#
# Deliberately conservative.
#
#   LGBM 50%
#   V1   25%
#   V2   25%
#
# We also evaluate other coarse candidates.
# ============================================================

PRIMARY_LGBM_WEIGHT = 0.50
PRIMARY_V1_WEIGHT = 0.25
PRIMARY_V2_WEIGHT = 0.25


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("LEAN REALMLP V2 - INTERACTION ORIENTED")
print("=" * 100)

print(
    f"PyTorch device = {DEVICE}"
)

if DEVICE == "cuda":

    print(
        f"GPU = "
        f"{torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(
    TRAIN_PATH
)

test = pd.read_csv(
    TEST_PATH
)

sample = pd.read_csv(
    SAMPLE_PATH
)


TARGET = "satisfaction"
ID_COL = "id"


# ============================================================
# BASIC VALIDATION
# ============================================================

if ID_COL not in train.columns:
    raise ValueError(
        "Train missing id."
    )

if TARGET not in train.columns:
    raise ValueError(
        "Train missing satisfaction."
    )

if ID_COL not in test.columns:
    raise ValueError(
        "Test missing id."
    )

if ID_COL not in sample.columns:
    raise ValueError(
        "Sample submission missing id."
    )


train_ids = pd.to_numeric(
    train[ID_COL],
    errors="raise",
)

test_ids = pd.to_numeric(
    test[ID_COL],
    errors="raise",
)

sample_ids = pd.to_numeric(
    sample[ID_COL],
    errors="raise",
)


if train_ids.duplicated().any():
    raise ValueError(
        "Train IDs are duplicated."
    )

if test_ids.duplicated().any():
    raise ValueError(
        "Test IDs are duplicated."
    )

if sample_ids.duplicated().any():
    raise ValueError(
        "Sample submission IDs are duplicated."
    )


if set(test_ids) != set(sample_ids):
    raise ValueError(
        "Test IDs and sample_submission IDs do not match."
    )


y = (
    train[TARGET]
    .astype("int8")
    .to_numpy()
    .reshape(-1)
)


print(
    f"\nTrain shape = {train.shape}"
)

print(
    f"Test shape  = {test.shape}"
)

print(
    f"Target mean = {y.mean():.6f}"
)


# ============================================================
# ORIGINAL FEATURES
# ============================================================

ORIGINAL_FEATURES = [
    c
    for c in train.columns
    if c not in [
        ID_COL,
        TARGET,
    ]
]


if len(ORIGINAL_FEATURES) != 21:

    raise ValueError(
        "Expected exactly 21 original predictors, "
        f"got {len(ORIGINAL_FEATURES)}."
    )


# ============================================================
# COLUMN TYPES
# ============================================================

CATEGORICAL_RAW = [
    c
    for c in ORIGINAL_FEATURES
    if not pd.api.types.is_numeric_dtype(
        train[c]
    )
]


NUMERIC_RAW = [
    c
    for c in ORIGINAL_FEATURES
    if c not in CATEGORICAL_RAW
]


EXPECTED_CATEGORICAL = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]


if set(CATEGORICAL_RAW) != set(
    EXPECTED_CATEGORICAL
):

    raise ValueError(
        "Unexpected categorical columns.\n"
        f"Detected: {CATEGORICAL_RAW}"
    )


print(
    "\nCategorical columns:"
)

for c in CATEGORICAL_RAW:

    print(
        f"  - {c}"
    )


print(
    "\nNumeric columns:"
)

for c in NUMERIC_RAW:

    print(
        f"  - {c}"
    )


# ============================================================
# COUNT FEATURES
# ============================================================

COUNT_COLUMNS = [

    "Gender",

    "Customer Type",

    "Type of Travel",

    "Class",

    "Leg room service",

]


# ============================================================
# V2 INTERACTION REPRESENTATION
#
# IMPORTANT:
#
# We intentionally DO NOT reproduce V1's exact 3 TE inputs.
#
# V1 had:
#
#   Class × Type of Travel × Gender
#   Class × Type of Travel × Age
#   Flight Distance × Departure/Arrival time convenient
#
# V2 instead emphasizes broad interaction structure.
# ============================================================

INTERACTION_DEFINITIONS = [

    # --------------------------------------------------------
    # Passenger context
    # --------------------------------------------------------

    (
        "Class",
        "Type of Travel",
    ),

    (
        "Class",
        "Customer Type",
    ),

    (
        "Type of Travel",
        "Customer Type",
    ),

    (
        "Class",
        "Gender",
    ),

    (
        "Type of Travel",
        "Gender",
    ),


    # --------------------------------------------------------
    # Service interactions
    # --------------------------------------------------------

    (
        "Inflight wifi service",
        "Ease of Online booking",
    ),

    (
        "Inflight wifi service",
        "Online boarding",
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
        "Food and drink",
        "Cleanliness",
    ),

    (
        "Inflight entertainment",
        "Cleanliness",
    ),

    (
        "On-board service",
        "Leg room service",
    ),

    (
        "Baggage handling",
        "Checkin service",
    ),


    # --------------------------------------------------------
    # Numeric / group interactions
    # --------------------------------------------------------

    (
        "Age",
        "Class",
    ),

    (
        "Age",
        "Type of Travel",
    ),

    (
        "Flight Distance",
        "Class",
    ),

    (
        "Flight Distance",
        "Type of Travel",
    ),

    (
        "Flight Distance",
        "Class",
        "Type of Travel",
    ),

]


# ============================================================
# INTERACTION NAME
# ============================================================

def make_interaction_name(
    columns,
):

    return (
        "_".join(columns)
        + "__int"
    )


INTERACTION_NAMES = [
    make_interaction_name(
        cols
    )
    for cols
    in INTERACTION_DEFINITIONS
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_v2_features(
    df,
):
    """
    Build V2 features.

    Components:

        21 raw features

        +
        every numeric raw feature duplicated
        as categorical/string representation

        +
        5 count features

        +
        18 interaction categorical features

    No target encoding.
    """

    X = df.copy()


    # --------------------------------------------------------
    # 1. Categorical raw values
    # --------------------------------------------------------

    for c in CATEGORICAL_RAW:

        X[c] = (
            X[c]
            .fillna("missing")
            .astype(str)
        )


    # --------------------------------------------------------
    # 2. Numeric raw values
    # --------------------------------------------------------

    for c in NUMERIC_RAW:

        X[c] = (
            X[c]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(0.0)
        )


    # --------------------------------------------------------
    # 3. Numeric -> categorical copies
    #
    # IMPORTANT:
    #
    # Keep as strings first.
    # Later we explicitly convert all categorical
    # columns to pandas CategoricalDtype.
    # --------------------------------------------------------

    for c in NUMERIC_RAW:

        cat_name = (
            f"{c}_cat_"
        )

        X[cat_name] = (
            X[c]
            .astype(str)
        )


    # --------------------------------------------------------
    # 4. Five count features
    #
    # Unsupervised.
    # No target information.
    # --------------------------------------------------------

    for c in COUNT_COLUMNS:

        count_name = (
            f"_{c}_count"
        )

        counts = (
            X[c]
            .astype(str)
            .value_counts(
                dropna=False
            )
        )

        X[count_name] = (
            X[c]
            .astype(str)
            .map(counts)
            .fillna(0)
            .astype("float32")
        )


    # --------------------------------------------------------
    # 5. Interaction categorical features
    # --------------------------------------------------------

    for columns in INTERACTION_DEFINITIONS:

        name = (
            make_interaction_name(
                columns
            )
        )

        combo = (
            X[
                columns[0]
            ]
            .astype(str)
        )


        for c in columns[1:]:

            combo = (
                combo
                + "|"
                + X[c].astype(str)
            )


        X[name] = (
            combo.astype(str)
        )


    return X


# ============================================================
# BUILD TRAIN / TEST FEATURES
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "BUILDING V2 INTERACTION REPRESENTATION"
)

print(
    "=" * 100
)


X_all = build_v2_features(
    train.drop(
        columns=[
            ID_COL,
            TARGET,
        ]
    )
)


X_test_all = build_v2_features(
    test.drop(
        columns=[
            ID_COL,
        ]
    )
)


# ============================================================
# EXPECTED FEATURE COUNT
# ============================================================

EXPECTED_FEATURE_COUNT = (
    len(ORIGINAL_FEATURES)
    +
    len(NUMERIC_RAW)
    +
    len(COUNT_COLUMNS)
    +
    len(INTERACTION_DEFINITIONS)
)


if X_all.shape[1] != (
    EXPECTED_FEATURE_COUNT
):

    raise ValueError(
        "Unexpected V2 feature count.\n"
        f"Expected = {EXPECTED_FEATURE_COUNT}\n"
        f"Actual   = {X_all.shape[1]}"
    )


if X_test_all.shape[1] != (
    EXPECTED_FEATURE_COUNT
):

    raise ValueError(
        "Unexpected V2 test feature count.\n"
        f"Expected = {EXPECTED_FEATURE_COUNT}\n"
        f"Actual   = {X_test_all.shape[1]}"
    )


print(
    f"V2 feature count = "
    f"{X_all.shape[1]}"
)


print(
    f"Interaction count = "
    f"{len(INTERACTION_DEFINITIONS)}"
)


# ============================================================
# DETECT ALL CATEGORICAL COLUMNS
# ============================================================

MODEL_CAT_COLS = []


# Original categoricals
for c in CATEGORICAL_RAW:

    MODEL_CAT_COLS.append(c)


# Numeric categorical copies
for c in NUMERIC_RAW:

    MODEL_CAT_COLS.append(
        f"{c}_cat_"
    )


# Interaction categoricals
for name in INTERACTION_NAMES:

    MODEL_CAT_COLS.append(
        name
    )


# ------------------------------------------------------------
# Safety check
# ------------------------------------------------------------

if len(
    MODEL_CAT_COLS
) != (
    len(CATEGORICAL_RAW)
    +
    len(NUMERIC_RAW)
    +
    len(INTERACTION_DEFINITIONS)
):

    raise ValueError(
        "Unexpected categorical column count."
    )


# ============================================================
# UNIFY TRAIN / TEST CATEGORIES
#
# No target is involved here.
# We only make category domains compatible.
# ============================================================

for c in MODEL_CAT_COLS:

    train_cat = (
        X_all[c]
        .astype(str)
    )

    test_cat = (
        X_test_all[c]
        .astype(str)
    )


    combined = pd.concat(
        [
            train_cat,
            test_cat,
        ],
        ignore_index=True,
    )


    categories = pd.Index(
        combined.unique()
    )


    dtype = (
        pd.api.types.CategoricalDtype(
            categories=categories
        )
    )


    X_all[c] = (
        train_cat
        .astype(dtype)
    )

    X_test_all[c] = (
        test_cat
        .astype(dtype)
    )


# ============================================================
# VERIFY CATEGORIES
# ============================================================

for c in MODEL_CAT_COLS:

    if str(
        X_all[c].dtype
    ) != "category":

        raise ValueError(
            f"Train column {c} "
            "is not categorical."
        )


    if str(
        X_test_all[c].dtype
    ) != "category":

        raise ValueError(
            f"Test column {c} "
            "is not categorical."
        )


# ============================================================
# CHECK NUMERIC DATA
# ============================================================

numeric_v2_cols = [
    c
    for c in X_all.columns
    if c not in MODEL_CAT_COLS
]


for c in numeric_v2_cols:

    if not pd.api.types.is_numeric_dtype(
        X_all[c]
    ):

        raise ValueError(
            f"V2 numeric column "
            f"{c} has dtype "
            f"{X_all[c].dtype}"
        )


    if X_all[c].isna().any():

        raise ValueError(
            f"V2 train column {c} contains NaN."
        )


    if X_test_all[c].isna().any():

        raise ValueError(
            f"V2 test column {c} contains NaN."
        )


print(
    f"V2 categorical columns = "
    f"{len(MODEL_CAT_COLS)}"
)

print(
    f"V2 numeric columns     = "
    f"{len(numeric_v2_cols)}"
)


# ============================================================
# PREDICTION FILE LOADER
# ============================================================

def load_prediction_file(
    path,
    name,
):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )


    df = pd.read_csv(
        path
    )


    if "id" not in df.columns:

        raise ValueError(
            f"{name} has no id column.\n"
            f"Columns = {df.columns.tolist()}"
        )


    prediction_candidates = [
        "prediction",
        "pred",
        "probability",
        "prob",
        "satisfaction",
        "target",
        "lgbm",
    ]


    pred_col = None


    for c in prediction_candidates:

        if c in df.columns:

            pred_col = c
            break


    if pred_col is None:

        numeric_cols = [
            c
            for c in df.columns
            if c != "id"
            and
            pd.api.types.is_numeric_dtype(
                df[c]
            )
        ]


        if len(numeric_cols) != 1:

            raise ValueError(
                f"Cannot identify prediction "
                f"column in {name}.\n"
                f"Columns = {df.columns.tolist()}"
            )


        pred_col = numeric_cols[0]


    out = df[
        [
            "id",
            pred_col,
        ]
    ].copy()


    out.columns = [
        "id",
        "prediction",
    ]


    out["id"] = pd.to_numeric(
        out["id"],
        errors="raise",
    )


    out["prediction"] = pd.to_numeric(
        out["prediction"],
        errors="raise",
    )


    if out["id"].duplicated().any():

        raise ValueError(
            f"{name} contains duplicated IDs."
        )


    if out["prediction"].isna().any():

        raise ValueError(
            f"{name} contains NaN predictions."
        )


    if not np.isfinite(
        out["prediction"].to_numpy()
    ).all():

        raise ValueError(
            f"{name} contains non-finite predictions."
        )


    return out


# ============================================================
# LOAD LGBM / V1
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "LOADING EXISTING LGBM + REALMLP V1"
)

print(
    "=" * 100
)


lgbm_oof_df = load_prediction_file(
    LGBM_OOF_PATH,
    "Current LGBM OOF",
)


lgbm_test_df = load_prediction_file(
    LGBM_TEST_PATH,
    "Current LGBM TEST",
)


v1_oof_df = load_prediction_file(
    V1_OOF_PATH,
    "RealMLP V1 OOF",
)


v1_test_df = load_prediction_file(
    V1_TEST_PATH,
    "RealMLP V1 TEST",
)


# ============================================================
# ALIGN LGBM + V1 OOF
# ============================================================

train_index_df = pd.DataFrame({
    "id": train_ids,
})


def align_prediction(
    index_df,
    prediction_df,
    name,
):

    aligned = (
        index_df
        .merge(
            prediction_df,
            on="id",
            how="left",
            validate="one_to_one",
        )
    )


    if len(aligned) != len(
        index_df
    ):

        raise ValueError(
            f"{name} row count mismatch."
        )


    if aligned["prediction"].isna().any():

        raise ValueError(
            f"{name} has missing predictions "
            "after ID alignment."
        )


    pred = (
        aligned["prediction"]
        .to_numpy(
            dtype=np.float64
        )
        .reshape(-1)
    )


    if not np.isfinite(
        pred
    ).all():

        raise ValueError(
            f"{name} contains non-finite "
            "aligned predictions."
        )


    return pred


lgbm_oof = align_prediction(
    train_index_df,
    lgbm_oof_df,
    "LGBM OOF",
)


v1_oof = align_prediction(
    train_index_df,
    v1_oof_df,
    "RealMLP V1 OOF",
)


# ============================================================
# ALIGN LGBM + V1 TEST
# ============================================================

test_index_df = pd.DataFrame({
    "id": test_ids,
})


lgbm_test = align_prediction(
    test_index_df,
    lgbm_test_df,
    "LGBM TEST",
)


v1_test = align_prediction(
    test_index_df,
    v1_test_df,
    "RealMLP V1 TEST",
)


# ============================================================
# VERIFY EXISTING MODEL AUC
# ============================================================

lgbm_auc = roc_auc_score(
    y,
    lgbm_oof,
)


v1_auc = roc_auc_score(
    y,
    v1_oof,
)


print(
    f"\nCurrent LGBM OOF = "
    f"{lgbm_auc:.9f}"
)


print(
    f"RealMLP V1 OOF   = "
    f"{v1_auc:.9f}"
)


# ============================================================
# RANK NORMALIZATION
# ============================================================

def rank_normalize(
    predictions,
):

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    ).reshape(-1)


    return (
        rankdata(
            predictions,
            method="average",
        )
        /
        len(predictions)
    )


# ============================================================
# EXISTING BASELINE:
# LGBM + V1 50/50
# ============================================================

lgbm_rank = rank_normalize(
    lgbm_oof
)

v1_rank = rank_normalize(
    v1_oof
)


baseline_5050_oof = (
    0.50 * lgbm_rank
    +
    0.50 * v1_rank
)


baseline_5050_auc = roc_auc_score(
    y,
    baseline_5050_oof,
)


print(
    f"\nExisting LGBM + V1 50/50 OOF = "
    f"{baseline_5050_auc:.9f}"
)


# ============================================================
# CHECK BASELINE AGAINST KNOWN RESULT
# ============================================================

KNOWN_BASELINE_5050 = 0.961276208


print(
    f"Known previous 50/50 OOF      = "
    f"{KNOWN_BASELINE_5050:.9f}"
)


print(
    f"Recomputed difference          = "
    f"{baseline_5050_auc - KNOWN_BASELINE_5050:+.9f}"
)


if abs(
    baseline_5050_auc
    -
    KNOWN_BASELINE_5050
) > 0.00001:

    print(
        "\nWARNING:"
    )

    print(
        "The recomputed V1 + LGBM baseline "
        "differs from the previous saved "
        "benchmark by more than 1e-5."
    )

    print(
        "The experiment will continue, "
        "but check the V1 prediction files."
    )


# ============================================================
# PREPARE CV
# ============================================================

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED,
)


# ============================================================
# STORAGE
# ============================================================

v2_oof = np.zeros(
    len(train),
    dtype=np.float64,
)


v2_test = np.zeros(
    len(test),
    dtype=np.float64,
)


fold_scores = []

fold_times = []


# ============================================================
# REALMLP TRAINING FUNCTION
# ============================================================

def train_realmlp_v2(
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    seed,
):

    params = REALMLP_PARAMS.copy()

    params[
        "random_state"
    ] = seed


    model = (
        RealMLP_TD_Classifier(
            device=DEVICE,
            **params,
        )
    )


    with contextlib.redirect_stdout(
        io.StringIO()
    ):

        model.fit(
            X_train,
            y_train,
            X_valid,
            y_valid,
            cat_col_names=MODEL_CAT_COLS,
        )


    p_valid = (
        model
        .predict_proba(
            X_valid
        )[:, 1]
    )


    p_test = (
        model
        .predict_proba(
            X_test
        )[:, 1]
    )


    p_valid = np.asarray(
        p_valid,
        dtype=np.float64,
    ).reshape(-1)


    p_test = np.asarray(
        p_test,
        dtype=np.float64,
    ).reshape(-1)


    del model

    gc.collect()


    if DEVICE == "cuda":

        torch.cuda.empty_cache()


    return (
        p_valid,
        p_test,
    )


# ============================================================
# OUTER 5-FOLD
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "REALMLP V2 - 5 FOLD OOF"
)

print(
    "=" * 100
)


start_total = time.time()


for fold, (
    tr_idx,
    va_idx,
) in enumerate(
    skf.split(
        X_all,
        y,
    ),
    start=1,
):

    fold_start = time.time()


    print(
        "\n"
        + "-" * 100
    )


    print(
        f"REALMLP V2 FOLD "
        f"{fold}/{N_FOLDS}"
    )


    print(
        "-" * 100
    )


    # --------------------------------------------------------
    # Outer split
    # --------------------------------------------------------

    X_tr = (
        X_all
        .iloc[
            tr_idx
        ]
        .copy()
    )


    X_va = (
        X_all
        .iloc[
            va_idx
        ]
        .copy()
    )


    X_te = (
        X_test_all
        .copy()
    )


    y_tr = y[
        tr_idx
    ]


    y_va = y[
        va_idx
    ]


    # --------------------------------------------------------
    # Numeric cleanup
    #
    # No TE in V2.
    # No target-based transformation at all.
    # --------------------------------------------------------

    numeric_cols = [
        c
        for c in X_tr.columns
        if c not in MODEL_CAT_COLS
    ]


    for c in numeric_cols:

        median_value = (
            X_tr[c]
            .median()
        )


        if pd.isna(
            median_value
        ):

            median_value = 0.0


        X_tr[c] = (
            X_tr[c]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(
                median_value
            )
            .astype("float32")
        )


        X_va[c] = (
            X_va[c]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(
                median_value
            )
            .astype("float32")
        )


        X_te[c] = (
            X_te[c]
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(
                median_value
            )
            .astype("float32")
        )


    # --------------------------------------------------------
    # Categorical checks
    # --------------------------------------------------------

    for c in MODEL_CAT_COLS:

        if str(
            X_tr[c].dtype
        ) != "category":

            raise ValueError(
                f"X_tr column {c} "
                "is not categorical."
            )


        if str(
            X_va[c].dtype
        ) != "category":

            raise ValueError(
                f"X_va column {c} "
                "is not categorical."
            )


        if str(
            X_te[c].dtype
        ) != "category":

            raise ValueError(
                f"X_te column {c} "
                "is not categorical."
            )


    # --------------------------------------------------------
    # Numeric NaN safety
    # --------------------------------------------------------

    for c in numeric_cols:

        if X_tr[c].isna().any():

            raise ValueError(
                f"NaN found in X_tr[{c}]"
            )


        if X_va[c].isna().any():

            raise ValueError(
                f"NaN found in X_va[{c}]"
            )


        if X_te[c].isna().any():

            raise ValueError(
                f"NaN found in X_te[{c}]"
            )


    # --------------------------------------------------------
    # Feature shape
    # --------------------------------------------------------

    if (
        X_tr.shape[1]
        != EXPECTED_FEATURE_COUNT
    ):

        raise ValueError(
            f"Unexpected X_tr shape: "
            f"{X_tr.shape}"
        )


    if (
        X_va.shape[1]
        != EXPECTED_FEATURE_COUNT
    ):

        raise ValueError(
            f"Unexpected X_va shape: "
            f"{X_va.shape}"
        )


    if (
        X_te.shape[1]
        != EXPECTED_FEATURE_COUNT
    ):

        raise ValueError(
            f"Unexpected X_te shape: "
            f"{X_te.shape}"
        )


    print(
        f"Feature shape = "
        f"{X_tr.shape}"
    )


    print(
        f"Categorical columns = "
        f"{len(MODEL_CAT_COLS)}"
    )


    # --------------------------------------------------------
    # Train V2
    # --------------------------------------------------------

    p_va, p_te = (
        train_realmlp_v2(
            X_tr,
            y_tr,
            X_va,
            y_va,
            X_te,
            seed=SEED + fold,
        )
    )


    if len(p_va) != len(
        va_idx
    ):

        raise ValueError(
            "V2 validation prediction "
            "length mismatch."
        )


    if len(p_te) != len(test):

        raise ValueError(
            "V2 test prediction "
            "length mismatch."
        )


    if not np.isfinite(
        p_va
    ).all():

        raise ValueError(
            "V2 validation prediction "
            "contains non-finite values."
        )


    if not np.isfinite(
        p_te
    ).all():

        raise ValueError(
            "V2 test prediction "
            "contains non-finite values."
        )


    v2_oof[
        va_idx
    ] = p_va


    v2_test += (
        p_te
        /
        N_FOLDS
    )


    fold_auc = roc_auc_score(
        y_va,
        p_va,
    )


    fold_scores.append(
        fold_auc
    )


    fold_time = (
        time.time()
        -
        fold_start
    )


    fold_times.append(
        fold_time
    )


    print(
        f"Fold {fold} AUC = "
        f"{fold_auc:.9f}"
    )


    print(
        f"Time = "
        f"{fold_time / 60:.2f} min"
    )


    del (
        X_tr,
        X_va,
        X_te,
        p_va,
        p_te,
    )


    gc.collect()


    if DEVICE == "cuda":

        torch.cuda.empty_cache()


# ============================================================
# V2 RESULT
# ============================================================

v2_auc = roc_auc_score(
    y,
    v2_oof,
)


v2_mean = np.mean(
    fold_scores
)


v2_std = np.std(
    fold_scores
)


print(
    "\n"
    + "=" * 100
)

print(
    "REALMLP V2 RESULT"
)

print(
    "=" * 100
)


for i, score in enumerate(
    fold_scores,
    start=1,
):

    print(
        f"Fold {i}: "
        f"{score:.9f}"
    )


print(
    f"\nV2 fold mean = "
    f"{v2_mean:.9f}"
)


print(
    f"V2 fold std  = "
    f"{v2_std:.9f}"
)


print(
    f"V2 OOF AUC   = "
    f"{v2_auc:.9f}"
)


# ============================================================
# RANK REPRESENTATIONS
# ============================================================

v2_rank = rank_normalize(
    v2_oof
)


# ============================================================
# RANK CORRELATION
#
# This is one of the most important outputs.
# ============================================================

corr_lgbm_v1 = spearmanr(
    lgbm_oof,
    v1_oof,
).statistic


corr_lgbm_v2 = spearmanr(
    lgbm_oof,
    v2_oof,
).statistic


corr_v1_v2 = spearmanr(
    v1_oof,
    v2_oof,
).statistic


print(
    "\n"
    + "=" * 100
)

print(
    "MODEL RANK CORRELATION"
)

print(
    "=" * 100
)


print(
    f"LGBM vs V1 = "
    f"{corr_lgbm_v1:.9f}"
)


print(
    f"LGBM vs V2 = "
    f"{corr_lgbm_v2:.9f}"
)


print(
    f"V1   vs V2 = "
    f"{corr_v1_v2:.9f}"
)


# ============================================================
# THREE-MODEL BLEND CANDIDATES
#
# Deliberately coarse.
# No dense weight search.
# ============================================================

blend_candidates = [

    # Existing baseline
    {
        "name":
            "LGBM50_V1_50",

        "lgbm":
            0.50,

        "v1":
            0.50,

        "v2":
            0.00,
    },

    # Main
    {
        "name":
            "LGBM50_V1_25_V2_25",

        "lgbm":
            0.50,

        "v1":
            0.25,

        "v2":
            0.25,
    },

    # More conservative V2
    {
        "name":
            "LGBM40_V1_40_V2_20",

        "lgbm":
            0.40,

        "v1":
            0.40,

        "v2":
            0.20,
    },

    # More V2
    {
        "name":
            "LGBM40_V1_30_V2_30",

        "lgbm":
            0.40,

        "v1":
            0.30,

        "v2":
            0.30,
    },

    # Keep stronger LGBM
    {
        "name":
            "LGBM50_V1_30_V2_20",

        "lgbm":
            0.50,

        "v1":
            0.30,

        "v2":
            0.20,
    },

]


blend_results = []


for candidate in blend_candidates:

    lgbm_w = candidate[
        "lgbm"
    ]

    v1_w = candidate[
        "v1"
    ]

    v2_w = candidate[
        "v2"
    ]


    total_weight = (
        lgbm_w
        +
        v1_w
        +
        v2_w
    )


    if abs(
        total_weight - 1.0
    ) > 1e-10:

        raise ValueError(
            f"Invalid weights for "
            f"{candidate['name']}"
        )


    blend_oof = (
        lgbm_w * lgbm_rank
        +
        v1_w * v1_rank
        +
        v2_w * v2_rank
    )


    auc = roc_auc_score(
        y,
        blend_oof,
    )


    blend_results.append({

        "name":
            candidate["name"],

        "lgbm_weight":
            lgbm_w,

        "v1_weight":
            v1_w,

        "v2_weight":
            v2_w,

        "auc":
            auc,

        "gain_vs_lgbm":
            auc - lgbm_auc,

        "gain_vs_v1":
            auc - v1_auc,

        "gain_vs_v2":
            auc - v2_auc,

        "gain_vs_old_5050":
            auc - baseline_5050_auc,
    })


blend_df = (
    pd.DataFrame(
        blend_results
    )
    .sort_values(
        "auc",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# PRIMARY BLEND
# ============================================================

primary_blend_oof = (
    PRIMARY_LGBM_WEIGHT
    * lgbm_rank
    +
    PRIMARY_V1_WEIGHT
    * v1_rank
    +
    PRIMARY_V2_WEIGHT
    * v2_rank
)


primary_blend_auc = roc_auc_score(
    y,
    primary_blend_oof,
)


print(
    "\n"
    + "=" * 100
)

print(
    "THREE-MODEL BLEND RESULTS"
)

print(
    "=" * 100
)


print(
    blend_df.to_string(
        index=False,
        formatters={

            "lgbm_weight":
                "{:.2f}".format,

            "v1_weight":
                "{:.2f}".format,

            "v2_weight":
                "{:.2f}".format,

            "auc":
                "{:.9f}".format,

            "gain_vs_lgbm":
                "{:+.9f}".format,

            "gain_vs_v1":
                "{:+.9f}".format,

            "gain_vs_v2":
                "{:+.9f}".format,

            "gain_vs_old_5050":
                "{:+.9f}".format,
        }
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "PRIMARY 50 / 25 / 25"
)

print(
    "=" * 100
)


print(
    f"LGBM = "
    f"{PRIMARY_LGBM_WEIGHT:.2f}"
)


print(
    f"V1   = "
    f"{PRIMARY_V1_WEIGHT:.2f}"
)


print(
    f"V2   = "
    f"{PRIMARY_V2_WEIGHT:.2f}"
)


print(
    f"\nPrimary OOF = "
    f"{primary_blend_auc:.9f}"
)


print(
    f"Gain vs LGBM = "
    f"{primary_blend_auc - lgbm_auc:+.9f}"
)


print(
    f"Gain vs existing 50/50 = "
    f"{primary_blend_auc - baseline_5050_auc:+.9f}"
)


# ============================================================
# TEST RANK BLENDS
# ============================================================

lgbm_test_rank = rank_normalize(
    lgbm_test
)


v1_test_rank = rank_normalize(
    v1_test
)


v2_test_rank = rank_normalize(
    v2_test
)


primary_blend_test = (
    PRIMARY_LGBM_WEIGHT
    * lgbm_test_rank
    +
    PRIMARY_V1_WEIGHT
    * v1_test_rank
    +
    PRIMARY_V2_WEIGHT
    * v2_test_rank
)


if not np.isfinite(
    primary_blend_test
).all():

    raise ValueError(
        "Primary test blend contains "
        "non-finite values."
    )


# ============================================================
# TEST PREDICTION STATISTICS
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "TEST PREDICTIONS"
)

print(
    "=" * 100
)


print(
    f"LGBM test mean = "
    f"{lgbm_test.mean():.9f}"
)


print(
    f"V1 test mean   = "
    f"{v1_test.mean():.9f}"
)


print(
    f"V2 test mean   = "
    f"{v2_test.mean():.9f}"
)


print(
    "\nPrimary 50/25/25 rank blend:"
)


print(
    f"min  = "
    f"{primary_blend_test.min():.9f}"
)


print(
    f"max  = "
    f"{primary_blend_test.max():.9f}"
)


print(
    f"mean = "
    f"{primary_blend_test.mean():.9f}"
)


print(
    f"std  = "
    f"{primary_blend_test.std():.9f}"
)


# ============================================================
# SAVE V2 OOF
# ============================================================

v2_oof_path = os.path.join(
    OUTPUT_DIR,
    "realmlp_v2_oof.csv",
)


pd.DataFrame({

    "id":
        train[ID_COL],

    "satisfaction":
        y,

    "prediction":
        v2_oof,

}).to_csv(
    v2_oof_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE V2 TEST
# ============================================================

v2_test_path = os.path.join(
    OUTPUT_DIR,
    "realmlp_v2_test.csv",
)


pd.DataFrame({

    "id":
        test[ID_COL],

    "prediction":
        v2_test,

}).to_csv(
    v2_test_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

model_comparison_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv",
)


model_comparison = pd.DataFrame([

    {
        "model":
            "LGBM",

        "oof_auc":
            lgbm_auc,
    },

    {
        "model":
            "RealMLP_V1",

        "oof_auc":
            v1_auc,
    },

    {
        "model":
            "RealMLP_V2",

        "oof_auc":
            v2_auc,
    },

    {
        "model":
            "LGBM_V1_50_50",

        "oof_auc":
            baseline_5050_auc,
    },

    {
        "model":
            "LGBM_V1_V2_50_25_25",

        "oof_auc":
            primary_blend_auc,
    },
])


model_comparison.to_csv(
    model_comparison_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE CORRELATION
# ============================================================

correlation_path = os.path.join(
    OUTPUT_DIR,
    "rank_correlation.csv",
)


correlation_df = pd.DataFrame([

    {
        "model_1":
            "LGBM",

        "model_2":
            "RealMLP_V1",

        "spearman":
            corr_lgbm_v1,
    },

    {
        "model_1":
            "LGBM",

        "model_2":
            "RealMLP_V2",

        "spearman":
            corr_lgbm_v2,
    },

    {
        "model_1":
            "RealMLP_V1",

        "model_2":
            "RealMLP_V2",

        "spearman":
            corr_v1_v2,
    },
])


correlation_df.to_csv(
    correlation_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE BLEND TABLE
# ============================================================

blend_table_path = os.path.join(
    OUTPUT_DIR,
    "blend_candidates.csv",
)


blend_df.to_csv(
    blend_table_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE PRIMARY BLEND OOF
# ============================================================

primary_oof_path = os.path.join(
    OUTPUT_DIR,
    "lgbm_v1_v2_502525_oof.csv",
)


pd.DataFrame({

    "id":
        train[ID_COL],

    "satisfaction":
        y,

    "lgbm_prediction":
        lgbm_oof,

    "v1_prediction":
        v1_oof,

    "v2_prediction":
        v2_oof,

    "lgbm_rank":
        lgbm_rank,

    "v1_rank":
        v1_rank,

    "v2_rank":
        v2_rank,

    "blend_502525":
        primary_blend_oof,

}).to_csv(
    primary_oof_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE PRIMARY BLEND TEST
# ============================================================

primary_test_path = os.path.join(
    OUTPUT_DIR,
    "lgbm_v1_v2_502525_test.csv",
)


pd.DataFrame({

    "id":
        test[ID_COL],

    "lgbm_prediction":
        lgbm_test,

    "v1_prediction":
        v1_test,

    "v2_prediction":
        v2_test,

    "lgbm_rank":
        lgbm_test_rank,

    "v1_rank":
        v1_test_rank,

    "v2_rank":
        v2_test_rank,

    "blend_502525":
        primary_blend_test,

}).to_csv(
    primary_test_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# CREATE PRIMARY SUBMISSION
# ============================================================

target_candidates = [
    c
    for c in sample.columns
    if c != ID_COL
]


if len(
    target_candidates
) != 1:

    raise ValueError(
        "Expected exactly one target "
        "column in sample_submission."
    )


submission_target = (
    target_candidates[0]
)


submission = sample[
    [ID_COL]
].copy()


submission["id"] = pd.to_numeric(
    submission["id"],
    errors="raise",
)


prediction_df = pd.DataFrame({

    "id":
        test_ids,

    "prediction":
        primary_blend_test,
})


submission = (
    submission
    .merge(
        prediction_df,
        on="id",
        how="left",
        validate="one_to_one",
    )
)


if len(
    submission
) != len(sample):

    raise ValueError(
        "Submission row count changed."
    )


if submission[
    "prediction"
].isna().any():

    raise ValueError(
        "Submission contains missing predictions."
    )


submission[
    submission_target
] = (
    submission[
        "prediction"
    ]
    .astype(float)
)


submission = submission[
    [
        "id",
        submission_target,
    ]
]


if len(
    submission
) != len(test):

    raise ValueError(
        "Final submission row count "
        "does not match test."
    )


if submission[
    "id"
].duplicated().any():

    raise ValueError(
        "Final submission IDs duplicated."
    )


if submission[
    submission_target
].isna().any():

    raise ValueError(
        "Final submission contains NaN."
    )


# ============================================================
# SAVE SUBMISSION
# ============================================================

submission_path = os.path.join(
    OUTPUT_DIR,
    "submission_lgbm_v1_v2_502525.csv",
)


submission.to_csv(
    submission_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "realmlp_v2_summary.csv",
)


summary = pd.DataFrame([

    {
        "metric":
            "lgbm_oof_auc",

        "value":
            lgbm_auc,
    },

    {
        "metric":
            "realmlp_v1_oof_auc",

        "value":
            v1_auc,
    },

    {
        "metric":
            "realmlp_v2_oof_auc",

        "value":
            v2_auc,
    },

    {
        "metric":
            "existing_lgbm_v1_5050_oof_auc",

        "value":
            baseline_5050_auc,
    },

    {
        "metric":
            "new_lgbm_v1_v2_502525_oof_auc",

        "value":
            primary_blend_auc,
    },

    {
        "metric":
            "gain_v2_vs_existing_5050",

        "value":
            primary_blend_auc
            -
            baseline_5050_auc,
    },

    {
        "metric":
            "spearman_lgbm_v1",

        "value":
            corr_lgbm_v1,
    },

    {
        "metric":
            "spearman_lgbm_v2",

        "value":
            corr_lgbm_v2,
    },

    {
        "metric":
            "spearman_v1_v2",

        "value":
            corr_v1_v2,
    },

    {
        "metric":
            "v2_fold_mean",

        "value":
            v2_mean,
    },

    {
        "metric":
            "v2_fold_std",

        "value":
            v2_std,
    },

    {
        "metric":
            "v2_feature_count",

        "value":
            X_all.shape[1],
    },

    {
        "metric":
            "v2_categorical_count",

        "value":
            len(MODEL_CAT_COLS),
    },

    {
        "metric":
            "v2_numeric_count",

        "value":
            len(numeric_v2_cols),
    },

    {
        "metric":
            "n_folds",

        "value":
            N_FOLDS,
    },

    {
        "metric":
            "n_ens",

        "value":
            REALMLP_PARAMS[
                "n_ens"
            ],
    },

    {
        "metric":
            "n_epochs",

        "value":
            REALMLP_PARAMS[
                "n_epochs"
            ],
    },

])


summary.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# FINAL REPORT
# ============================================================

total_time = (
    time.time()
    -
    start_total
)


print(
    "\n"
    + "=" * 100
)

print(
    "FILES SAVED"
)

print(
    "=" * 100
)


print(
    f"\nV2 OOF:"
    f"\n{v2_oof_path}"
)


print(
    f"\nV2 TEST:"
    f"\n{v2_test_path}"
)


print(
    f"\nModel comparison:"
    f"\n{model_comparison_path}"
)


print(
    f"\nRank correlation:"
    f"\n{correlation_path}"
)


print(
    f"\nBlend candidates:"
    f"\n{blend_table_path}"
)


print(
    f"\nPrimary OOF:"
    f"\n{primary_oof_path}"
)


print(
    f"\nPrimary TEST:"
    f"\n{primary_test_path}"
)


print(
    f"\nSubmission:"
    f"\n{submission_path}"
)


print(
    f"\nSummary:"
    f"\n{summary_path}"
)


print(
    "\n"
    + "=" * 100
)

print(
    "DONE"
)

print(
    f"Total time = "
    f"{total_time / 60:.2f} min"
)

print(
    "=" * 100
)