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
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

from pytabkit import RealMLP_TD_Classifier


warnings.filterwarnings("ignore")


# ============================================================
# LEAN REALMLP + CURRENT LGBM
#
# Competition-only.
#
# NO external 129k Airline Passenger Satisfaction dataset.
# NO Route Profile.
# NO Route Teacher.
# NO XGBoost.
#
# Main idea:
#
#   Raw 21 features
#       +
#   Numeric -> categorical dual representation
#       +
#   5 count features
#       +
#   3 selected categorical interactions
#       +
#   Target Encoding for those 3 interactions only
#       ↓
#     Lean RealMLP
#
# Then:
#
#   Current Conditional-TE LGBM
#              +
#      Lean RealMLP
#              ↓
#       50/50 rank blend
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


# ------------------------------------------------------------
# Existing strong Conditional-TE LGBM
# ------------------------------------------------------------

LGBM_OOF_PATH = (
    r"D:\kaggle\airline\conditional_te_experiment"
    r"\conditional_te_lgbm_oof.csv"
)

LGBM_TEST_PATH = (
    r"D:\kaggle\airline\conditional_te_experiment"
    r"\conditional_te_lgbm_test.csv"
)


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

OUTPUT_DIR = (
    r"D:\kaggle\airline\lean_realmlp_blend"
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

INNER_TE_FOLDS = 5


# ------------------------------------------------------------
# Previous benchmark
# ------------------------------------------------------------

PREVIOUS_BEST_BLEND_OOF = 0.960878324


# ------------------------------------------------------------
# RealMLP configuration
#
# Lean configuration inspired by the public S6E10
# RealMLP setup that motivated this experiment.
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Final blend
#
# Main submission = 50% rank LGBM + 50% rank RealMLP.
# ------------------------------------------------------------

PRIMARY_LGBM_WEIGHT = 0.50
PRIMARY_REALMLP_WEIGHT = 0.50


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 100)
print("LEAN REALMLP + CURRENT LGBM")
print("=" * 100)

print(
    f"PyTorch device = {DEVICE}"
)

if DEVICE == "cuda":

    print(
        f"GPU = "
        f"{torch.cuda.get_device_name(0)}"
    )


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
# VALIDATION
# ============================================================

required_train = [
    ID_COL,
    TARGET,
]

for c in required_train:

    if c not in train.columns:

        raise ValueError(
            f"Missing train column: {c}"
        )


if ID_COL not in test.columns:

    raise ValueError(
        "Missing test id column."
    )


if ID_COL not in sample.columns:

    raise ValueError(
        "Missing sample_submission id column."
    )


y = (
    train[TARGET]
    .astype("int8")
    .to_numpy()
    .reshape(-1)
)


assert len(y) == len(train), (
    "Target length does not match train length."
)


print(
    f"\nTrain shape = "
    f"{train.shape}"
)

print(
    f"Test shape  = "
    f"{test.shape}"
)

print(
    f"Target mean = "
    f"{y.mean():.6f}"
)


# ============================================================
# RAW FEATURE SET
# ============================================================

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


if len(ORIGINAL_FEATURES) != 21:

    raise ValueError(
        "Expected 21 original predictors, "
        f"got {len(ORIGINAL_FEATURES)}"
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


print(
    "\nOriginal categorical columns:"
)

for c in CATEGORICAL_RAW:

    print(
        f"  - {c}"
    )


print(
    "\nOriginal numeric columns:"
)

for c in NUMERIC_RAW:

    print(
        f"  - {c}"
    )


# ============================================================
# EXPECTED ORIGINAL COLUMNS
# ============================================================

EXPECTED_CATEGORICAL = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]


EXPECTED_NUMERIC = [
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


if set(CATEGORICAL_RAW) != set(
    EXPECTED_CATEGORICAL
):

    raise ValueError(
        "Unexpected categorical columns.\n"
        f"Detected = {CATEGORICAL_RAW}\n"
        f"Expected = {EXPECTED_CATEGORICAL}"
    )


if set(NUMERIC_RAW) != set(
    EXPECTED_NUMERIC
):

    raise ValueError(
        "Unexpected numeric columns."
    )


# ============================================================
# SELECTED INTERACTIONS
#
# Only these 3 get Target Encoding.
# ============================================================

IMPORTANT_COMBOS = [

    (
        "Class",
        "Type of Travel",
        "Gender",
    ),

    (
        "Class",
        "Type of Travel",
        "Age",
    ),

    (
        "Flight Distance",
        "Departure/Arrival time convenient",
    ),
]


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
# FEATURE ENGINEERING STATE
# ============================================================

count_map = {}

combo_map = {}


# ============================================================
# HELPER: SAFE COMBO NAME
# ============================================================

def make_combo_name(cols):

    return (
        "_".join(cols)
        + "_"
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def lean_feature_engineering(
    df,
    fit=False,
):
    """
    Build the lean RealMLP representation.

    Features:

    1. Raw original 21 features
    2. Numeric columns -> categorical copies
    3. Five count features
    4. Three selected categorical interactions

    The categorical representations are stored as pandas
    categorical columns.

    IMPORTANT:
    Count maps and combo maps are learned from TRAIN only.
    """

    X = df.copy()


    # --------------------------------------------------------
    # 1. Fill categorical missing values
    # --------------------------------------------------------

    for c in CATEGORICAL_RAW:

        X[c] = (
            X[c]
            .fillna("missing")
            .astype(str)
        )


    # --------------------------------------------------------
    # 2. Fill numeric missing values
    #
    # RealMLP currently requires numerical values to be
    # non-missing.
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
    # 3. Numeric -> categorical dual representation
    #
    # IMPORTANT FIX:
    #
    # Do NOT use:
    #
    #   ndarray.astype("category")
    #
    # because NumPy does not support pandas "category".
    #
    # Instead:
    #
    #   Series -> string -> pandas category later
    #
    # We retain exact numeric values as category labels.
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
    # 4. Count features
    #
    # Train-only frequency/count maps.
    # --------------------------------------------------------

    for c in COUNT_COLUMNS:

        count_name = (
            f"_{c}_count"
        )


        if fit:

            counts = (
                X[c]
                .astype(str)
                .value_counts(
                    dropna=False
                )
            )

            count_map[
                count_name
            ] = counts

        else:

            if count_name not in count_map:

                raise RuntimeError(
                    f"Missing count map "
                    f"for {count_name}"
                )

            counts = (
                count_map[
                    count_name
                ]
            )


        X[count_name] = (
            X[c]
            .astype(str)
            .map(counts)
            .fillna(0)
            .astype("float32")
        )


    # --------------------------------------------------------
    # 5. Three selected interactions
    #
    # Build them as strings.
    #
    # Train:
    #   store observed combinations
    #
    # Test:
    #   unseen combinations are retained as strings
    #   and later represented as a consistent category.
    # --------------------------------------------------------

    combo_names = []


    for cols in IMPORTANT_COMBOS:

        combo_name = (
            make_combo_name(cols)
        )

        combo_names.append(
            combo_name
        )


        combo_series = (
            X[
                cols[0]
            ]
            .astype(str)
        )


        for c in cols[1:]:

            combo_series = (
                combo_series
                + "_"
                + X[c].astype(str)
            )


        combo_series = (
            combo_series
            .astype(str)
        )


        if fit:

            combo_map[
                combo_name
            ] = (
                pd.Index(
                    combo_series.unique()
                )
            )

        else:

            if combo_name not in combo_map:

                raise RuntimeError(
                    f"Missing combo map "
                    f"for {combo_name}"
                )


        X[combo_name] = (
            combo_series
        )


    return (
        X,
        combo_names,
    )


# ============================================================
# FIT FEATURE MAPS ON TRAIN
# ============================================================

print(
    "\nBuilding lean feature representation..."
)


X_all, combo_names = (
    lean_feature_engineering(
        train.drop(
            columns=[
                TARGET,
                ID_COL,
            ]
        ).copy(),
        fit=True,
    )
)


X_test_all, _ = (
    lean_feature_engineering(
        test.drop(
            columns=[
                ID_COL,
            ],
            errors="ignore",
        ).copy(),
        fit=False,
    )
)


# ============================================================
# UNIFY CATEGORICAL COLUMNS
#
# This step is deliberately done after feature creation.
#
# IMPORTANT:
#
# We convert BOTH train and test values to strings before
# constructing CategoricalDtype.
#
# This avoids the subtle bug:
#
#   integer category values
#       ->
#   string category labels
#
# where direct astype(dtype) can otherwise create NaNs.
# ============================================================

MODEL_CAT_COLS = []


for c in X_all.columns:

    if (
        str(X_all[c].dtype)
        == "category"
        or
        X_all[c].dtype == object
    ):

        MODEL_CAT_COLS.append(c)


# All raw categorical columns should be included.
for c in CATEGORICAL_RAW:

    if c not in MODEL_CAT_COLS:

        MODEL_CAT_COLS.append(c)


# All numeric categorical copies should be included.
for c in NUMERIC_RAW:

    cat_name = (
        f"{c}_cat_"
    )

    if cat_name not in MODEL_CAT_COLS:

        MODEL_CAT_COLS.append(
            cat_name
        )


# All selected interaction columns should be included.
for c in combo_names:

    if c not in MODEL_CAT_COLS:

        MODEL_CAT_COLS.append(c)


# ------------------------------------------------------------
# Build common pandas categorical dtype for train/test.
# ------------------------------------------------------------

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


    categories = (
        pd.Index(
            combined.unique()
        )
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


print(
    f"Lean feature count before TE = "
    f"{X_all.shape[1]}"
)


print(
    f"Categorical feature count = "
    f"{len(MODEL_CAT_COLS)}"
)


print(
    "Selected TE combinations:"
)

for c in combo_names:

    print(
        f"  - {c}"
    )


# ============================================================
# FEATURE SHAPE CHECK
# ============================================================

EXPECTED_BEFORE_TE = (
    21
    + len(NUMERIC_RAW)
    + len(COUNT_COLUMNS)
    + len(IMPORTANT_COMBOS)
)


if X_all.shape[1] != EXPECTED_BEFORE_TE:

    raise ValueError(
        "Unexpected lean feature count.\n"
        f"Expected = {EXPECTED_BEFORE_TE}\n"
        f"Actual   = {X_all.shape[1]}"
    )


# ============================================================
# TARGET-ENCODING INPUT
#
# ONLY the 3 selected combinations are TE encoded.
# ============================================================

TE_INPUT_ALL = (
    X_all[
        combo_names
    ].copy()
)

TE_INPUT_TEST = (
    X_test_all[
        combo_names
    ].copy()
)


# ============================================================
# LOAD CURRENT LGBM
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "LOADING CURRENT CONDITIONAL-TE LGBM"
)

print(
    "=" * 100
)


def load_prediction_file(
    path,
    name,
):
    """
    Load id + prediction from an existing prediction file.
    """

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
            f"Columns: {df.columns.tolist()}"
        )


    candidates = [
        "prediction",
        "pred",
        "probability",
        "prob",
        "satisfaction",
        "target",
        "lgbm",
    ]


    pred_col = None


    for c in candidates:

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


        if len(numeric_cols) == 1:

            pred_col = (
                numeric_cols[0]
            )

        else:

            raise ValueError(
                f"Cannot identify prediction "
                f"column in {name}.\n"
                f"Columns: "
                f"{df.columns.tolist()}"
            )


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


    out["prediction"] = (
        pd.to_numeric(
            out["prediction"],
            errors="raise",
        )
    )


    out["id"] = (
        pd.to_numeric(
            out["id"],
            errors="raise",
        )
    )


    if (
        out["prediction"]
        .isna()
        .any()
    ):

        raise ValueError(
            f"{name} contains NaN predictions."
        )


    if (
        out["id"]
        .duplicated()
        .any()
    ):

        raise ValueError(
            f"{name} contains duplicated IDs."
        )


    return out


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

lgbm_oof_df = (
    load_prediction_file(
        LGBM_OOF_PATH,
        "Current LGBM OOF",
    )
)


lgbm_test_df = (
    load_prediction_file(
        LGBM_TEST_PATH,
        "Current LGBM TEST",
    )
)


# ============================================================
# ALIGN CURRENT LGBM OOF BY ID
# ============================================================

train_id_df = (
    train[
        [
            ID_COL,
            TARGET,
        ]
    ].copy()
)


train_id_df["id"] = (
    pd.to_numeric(
        train_id_df["id"],
        errors="raise",
    )
)


if (
    train_id_df["id"]
    .duplicated()
    .any()
):

    raise ValueError(
        "Train contains duplicated IDs."
    )


lgbm_oof_aligned = (
    train_id_df
    .merge(
        lgbm_oof_df,
        on="id",
        how="left",
        validate="one_to_one",
    )
)


if len(
    lgbm_oof_aligned
) != len(train):

    raise ValueError(
        "LGBM OOF row count mismatch."
    )


if (
    lgbm_oof_aligned["prediction"]
    .isna()
    .any()
):

    raise ValueError(
        "Missing current LGBM OOF predictions."
    )


lgbm_oof = (
    lgbm_oof_aligned[
        "prediction"
    ]
    .to_numpy(
        dtype=np.float64
    )
)


lgbm_auc = roc_auc_score(
    y,
    lgbm_oof,
)


print(
    f"Current LGBM OOF AUC = "
    f"{lgbm_auc:.9f}"
)


# ============================================================
# ALIGN CURRENT LGBM TEST BY ID
# ============================================================

test_id_df = (
    test[
        [
            ID_COL,
        ]
    ].copy()
)


test_id_df["id"] = (
    pd.to_numeric(
        test_id_df["id"],
        errors="raise",
    )
)


if (
    test_id_df["id"]
    .duplicated()
    .any()
):

    raise ValueError(
        "Test contains duplicated IDs."
    )


lgbm_test_aligned = (
    test_id_df
    .merge(
        lgbm_test_df,
        on="id",
        how="left",
        validate="one_to_one",
    )
)


if len(
    lgbm_test_aligned
) != len(test):

    raise ValueError(
        "LGBM test row count mismatch."
    )


if (
    lgbm_test_aligned["prediction"]
    .isna()
    .any()
):

    raise ValueError(
        "Missing current LGBM test predictions."
    )


lgbm_test = (
    lgbm_test_aligned[
        "prediction"
    ]
    .to_numpy(
        dtype=np.float64
    )
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

realmlp_oof = np.zeros(
    len(train),
    dtype=np.float64,
)


realmlp_test = np.zeros(
    len(test),
    dtype=np.float64,
)


fold_scores = []

fold_times = []


# ============================================================
# REALMLP TRAINING FUNCTION
# ============================================================

def train_realmlp(
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    seed,
):
    """
    Train one lean RealMLP ensemble.
    """

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


    # --------------------------------------------------------
    # Suppress verbose model output.
    # --------------------------------------------------------

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
#
# For every fold:
#
#   raw lean features
#          ↓
#   inner 5-fold TE on ONLY 3 combos
#          ↓
#       RealMLP
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "REALMLP 5-FOLD OOF"
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
        f"REALMLP FOLD "
        f"{fold}/{N_FOLDS}"
    )

    print(
        "-" * 100
    )


    # --------------------------------------------------------
    # Copy raw lean features
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
    # Target Encoder
    #
    # sklearn TargetEncoder.fit_transform performs internal
    # cross-fitting on the training data.
    #
    # Validation and test are transformed only after the
    # encoder has been fitted on the outer training split.
    # --------------------------------------------------------

    encoder = (
        TargetEncoder(
            target_type="binary",
            smooth="auto",
            cv=INNER_TE_FOLDS,
            shuffle=True,
            random_state=(
                SEED
                + fold
            ),
        )
    )


    te_tr = (
        encoder
        .fit_transform(
            TE_INPUT_ALL
            .iloc[
                tr_idx
            ],
            y_tr,
        )
        .astype("float32")
    )


    te_va = (
        encoder
        .transform(
            TE_INPUT_ALL
            .iloc[
                va_idx
            ]
        )
        .astype("float32")
    )


    te_te = (
        encoder
        .transform(
            TE_INPUT_TEST
        )
        .astype("float32")
    )


    # --------------------------------------------------------
    # Add only 3 TE columns.
    # --------------------------------------------------------

    for j, name in enumerate(
        combo_names
    ):

        te_name = (
            f"_{name}TE"
        )


        X_tr[
            te_name
        ] = te_tr[
            :,
            j,
        ]


        X_va[
            te_name
        ] = te_va[
            :,
            j,
        ]


        X_te[
            te_name
        ] = te_te[
            :,
            j,
        ]


    # --------------------------------------------------------
    # Final numeric cleanup
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Numeric columns:
    #
    # Everything that is not a pandas category is numeric.
    # --------------------------------------------------------

    numeric_cols = [
        c
        for c in X_tr.columns
        if str(
            X_tr[c].dtype
        ) != "category"
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
            .fillna(
                median_value
            )
            .astype("float32")
        )


        X_va[c] = (
            X_va[c]
            .fillna(
                median_value
            )
            .astype("float32")
        )


        X_te[c] = (
            X_te[c]
            .fillna(
                median_value
            )
            .astype("float32")
        )


    # --------------------------------------------------------
    # Safety checks before RealMLP
    # --------------------------------------------------------

    for name, df in [
        ("X_tr", X_tr),
        ("X_va", X_va),
        ("X_te", X_te),
    ]:

        if len(df.columns) != (
            X_all.shape[1]
            + len(combo_names)
        ):

            raise ValueError(
                f"{name} unexpected feature "
                f"count: {df.shape}"
            )


        missing_numeric = [
            c
            for c in df.columns
            if (
                str(df[c].dtype) != "category"
                and
                df[c].isna().any()
            )
        ]


        if missing_numeric:

            raise ValueError(
                f"{name} contains NaNs in "
                f"numeric columns: "
                f"{missing_numeric}"
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
    # Train RealMLP
    #
    # Fold-specific random state creates ensemble diversity.
    # --------------------------------------------------------

    p_va, p_te = (
        train_realmlp(
            X_tr,
            y_tr,
            X_va,
            y_va,
            X_te,
            seed=SEED + fold,
        )
    )


    if len(p_va) != len(va_idx):

        raise ValueError(
            "Validation prediction length "
            "does not match validation rows."
        )


    if len(p_te) != len(test):

        raise ValueError(
            "Test prediction length "
            "does not match test rows."
        )


    realmlp_oof[
        va_idx
    ] = p_va


    realmlp_test += (
        p_te
        / N_FOLDS
    )


    fold_auc = (
        roc_auc_score(
            y_va,
            p_va,
        )
    )


    fold_scores.append(
        fold_auc
    )


    fold_time = (
        time.time()
        - fold_start
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
        te_tr,
        te_va,
        te_te,
        encoder,
        p_va,
        p_te,
    )


    gc.collect()


    if DEVICE == "cuda":

        torch.cuda.empty_cache()


# ============================================================
# REALMLP OOF RESULT
# ============================================================

realmlp_auc = (
    roc_auc_score(
        y,
        realmlp_oof,
    )
)


realmlp_mean = (
    np.mean(
        fold_scores
    )
)


realmlp_std = (
    np.std(
        fold_scores
    )
)


# ============================================================
# RANK FUNCTION
# ============================================================

def rank_normalize(
    predictions
):
    """
    Convert predictions into normalized ranks.
    """

    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    ).reshape(-1)


    return (
        rankdata(
            predictions,
            method="average",
        )
        / len(predictions)
    )


# ============================================================
# OOF RANK BLENDS
#
# Only three conservative candidates.
# NO dense weight optimization.
# ============================================================

lgbm_rank = (
    rank_normalize(
        lgbm_oof
    )
)


realmlp_rank = (
    rank_normalize(
        realmlp_oof
    )
)


blend_results = []


for lgbm_weight in [
    0.40,
    0.50,
    0.60,
]:

    realmlp_weight = (
        1.0
        - lgbm_weight
    )


    blend_oof = (
        lgbm_weight
        * lgbm_rank

        +

        realmlp_weight
        * realmlp_rank
    )


    auc = (
        roc_auc_score(
            y,
            blend_oof,
        )
    )


    blend_results.append(
        {
            "lgbm_weight":
                lgbm_weight,

            "realmlp_weight":
                realmlp_weight,

            "auc":
                auc,

            "gain_vs_lgbm":
                auc
                - lgbm_auc,

            "gain_vs_realmlp":
                auc
                - realmlp_auc,
        }
    )


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
# PRIMARY 50/50 BLEND
#
# Final submission remains 50/50 even if 40/60 or 60/40
# happens to have a slightly higher OOF score.
# ============================================================

primary_blend_oof = (
    PRIMARY_LGBM_WEIGHT
    * lgbm_rank

    +

    PRIMARY_REALMLP_WEIGHT
    * realmlp_rank
)


primary_blend_auc = (
    roc_auc_score(
        y,
        primary_blend_oof,
    )
)


# ============================================================
# TEST RANK BLEND
# ============================================================

lgbm_test_rank = (
    rank_normalize(
        lgbm_test
    )
)


realmlp_test_rank = (
    rank_normalize(
        realmlp_test
    )
)


primary_blend_test = (
    PRIMARY_LGBM_WEIGHT
    * lgbm_test_rank

    +

    PRIMARY_REALMLP_WEIGHT
    * realmlp_test_rank
)


# ============================================================
# SANITY CHECKS
# ============================================================

if not np.isfinite(
    realmlp_oof
).all():

    raise ValueError(
        "RealMLP OOF contains non-finite values."
    )


if not np.isfinite(
    realmlp_test
).all():

    raise ValueError(
        "RealMLP test predictions contain "
        "non-finite values."
    )


if not np.isfinite(
    primary_blend_oof
).all():

    raise ValueError(
        "Blend OOF contains non-finite values."
    )


if not np.isfinite(
    primary_blend_test
).all():

    raise ValueError(
        "Blend test contains non-finite values."
    )


# ============================================================
# FINAL REPORT
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "LEAN REALMLP RESULT"
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
    f"\nRealMLP fold mean = "
    f"{realmlp_mean:.9f}"
)


print(
    f"RealMLP fold std  = "
    f"{realmlp_std:.9f}"
)


print(
    f"RealMLP OOF AUC   = "
    f"{realmlp_auc:.9f}"
)


print(
    f"\nCurrent LGBM OOF  = "
    f"{lgbm_auc:.9f}"
)


print(
    f"Previous best blend OOF = "
    f"{PREVIOUS_BEST_BLEND_OOF:.9f}"
)


print(
    "\n"
    + "-" * 100
)


print(
    "CONSERVATIVE BLEND CANDIDATES"
)


print(
    "-" * 100
)


print(
    blend_df.to_string(
        index=False,
        formatters={
            "lgbm_weight":
                "{:.2f}".format,

            "realmlp_weight":
                "{:.2f}".format,

            "auc":
                "{:.9f}".format,

            "gain_vs_lgbm":
                "{:+.9f}".format,

            "gain_vs_realmlp":
                "{:+.9f}".format,
        }
    )
)


print(
    "\n"
    + "=" * 100
)


print(
    "PRIMARY 50/50 RANK BLEND"
)


print(
    "=" * 100
)


print(
    f"LGBM weight    = "
    f"{PRIMARY_LGBM_WEIGHT:.2f}"
)


print(
    f"RealMLP weight = "
    f"{PRIMARY_REALMLP_WEIGHT:.2f}"
)


print(
    f"50/50 OOF AUC  = "
    f"{primary_blend_auc:.9f}"
)


print(
    f"Gain vs LGBM   = "
    f"{primary_blend_auc - lgbm_auc:+.9f}"
)


print(
    f"Gain vs previous best = "
    f"{primary_blend_auc - PREVIOUS_BEST_BLEND_OOF:+.9f}"
)


# ============================================================
# PREDICTION STATISTICS
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
    f"Lean RealMLP min  = "
    f"{realmlp_test.min():.9f}"
)


print(
    f"Lean RealMLP max  = "
    f"{realmlp_test.max():.9f}"
)


print(
    f"Lean RealMLP mean = "
    f"{realmlp_test.mean():.9f}"
)


print(
    f"Lean RealMLP std  = "
    f"{realmlp_test.std():.9f}"
)


print(
    "\n50/50 rank blend:"
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
# SAVE REALMLP OOF
# ============================================================

realmlp_oof_path = os.path.join(
    OUTPUT_DIR,
    "lean_realmlp_oof.csv",
)


pd.DataFrame({
    "id":
        train[ID_COL],

    "satisfaction":
        y,

    "prediction":
        realmlp_oof,
}).to_csv(
    realmlp_oof_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE REALMLP TEST
# ============================================================

realmlp_test_path = os.path.join(
    OUTPUT_DIR,
    "lean_realmlp_test.csv",
)


pd.DataFrame({
    "id":
        test[ID_COL],

    "prediction":
        realmlp_test,
}).to_csv(
    realmlp_test_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE BLEND OOF
# ============================================================

blend_oof_path = os.path.join(
    OUTPUT_DIR,
    "lean_realmlp_lgbm_5050_oof.csv",
)


pd.DataFrame({
    "id":
        train[ID_COL],

    "satisfaction":
        y,

    "lgbm_prediction":
        lgbm_oof,

    "realmlp_prediction":
        realmlp_oof,

    "lgbm_rank":
        lgbm_rank,

    "realmlp_rank":
        realmlp_rank,

    "blend_5050":
        primary_blend_oof,
}).to_csv(
    blend_oof_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE BLEND TEST
# ============================================================

blend_test_path = os.path.join(
    OUTPUT_DIR,
    "lean_realmlp_lgbm_5050_test.csv",
)


pd.DataFrame({
    "id":
        test[ID_COL],

    "lgbm_prediction":
        lgbm_test,

    "realmlp_prediction":
        realmlp_test,

    "lgbm_rank":
        lgbm_test_rank,

    "realmlp_rank":
        realmlp_test_rank,

    "blend_5050":
        primary_blend_test,
}).to_csv(
    blend_test_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE WEIGHT CHECK
# ============================================================

weight_check_path = os.path.join(
    OUTPUT_DIR,
    "conservative_blend_check.csv",
)


blend_df.to_csv(
    weight_check_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# CREATE SUBMISSION
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
        "sample_submission.csv should have exactly "
        "one target column besides id.\n"
        f"Found: {target_candidates}"
    )


submission_target = (
    target_candidates[0]
)


# ------------------------------------------------------------
# Align prediction explicitly to sample order by ID.
# ------------------------------------------------------------

test_rank_df = pd.DataFrame({

    "id":
        pd.to_numeric(
            test[ID_COL],
            errors="raise",
        ),

    "prediction":
        primary_blend_test,
})


submission = (
    sample[
        [ID_COL]
    ]
    .copy()
)


submission["id"] = (
    pd.to_numeric(
        submission["id"],
        errors="raise",
    )
)


if (
    submission["id"]
    .duplicated()
    .any()
):

    raise ValueError(
        "sample_submission contains duplicated IDs."
    )


submission = (
    submission
    .merge(
        test_rank_df,
        on="id",
        how="left",
        validate="one_to_one",
    )
)


if (
    len(submission)
    != len(sample)
):

    raise ValueError(
        "Submission row count changed after merge."
    )


if (
    submission["prediction"]
    .isna()
    .any()
):

    missing_n = (
        submission["prediction"]
        .isna()
        .sum()
    )


    raise ValueError(
        f"{missing_n} sample IDs "
        "have no prediction."
    )


submission[
    submission_target
] = (
    submission["prediction"]
    .astype(float)
)


submission = submission[
    [
        "id",
        submission_target,
    ]
]


# ------------------------------------------------------------
# Final submission integrity checks
# ------------------------------------------------------------

assert len(
    submission
) == len(test)


assert not (
    submission["id"]
    .duplicated()
    .any()
)


assert not (
    submission[
        submission_target
    ].isna()
    .any()
)


# ------------------------------------------------------------
# Check that the sample IDs are exactly the test IDs.
# ------------------------------------------------------------

sample_ids = set(
    pd.to_numeric(
        sample[ID_COL],
        errors="raise",
    )
)


test_ids = set(
    pd.to_numeric(
        test[ID_COL],
        errors="raise",
    )
)


if sample_ids != test_ids:

    raise ValueError(
        "sample_submission IDs do not exactly match "
        "test IDs."
    )


# ============================================================
# SAVE SUBMISSION
# ============================================================

submission_path = os.path.join(
    OUTPUT_DIR,
    "submission_lean_realmlp_5050.csv",
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
    "lean_realmlp_summary.csv",
)


summary = pd.DataFrame([
    {
        "metric":
            "current_lgbm_oof",

        "value":
            lgbm_auc,
    },

    {
        "metric":
            "lean_realmlp_oof",

        "value":
            realmlp_auc,
    },

    {
        "metric":
            "lean_realmlp_fold_mean",

        "value":
            realmlp_mean,
    },

    {
        "metric":
            "lean_realmlp_fold_std",

        "value":
            realmlp_std,
    },

    {
        "metric":
            "lean_realmlp_lgbm_5050_oof",

        "value":
            primary_blend_auc,
    },

    {
        "metric":
            "gain_5050_vs_lgbm",

        "value":
            primary_blend_auc
            - lgbm_auc,
    },

    {
        "metric":
            "previous_best_blend_oof",

        "value":
            PREVIOUS_BEST_BLEND_OOF,
    },

    {
        "metric":
            "gain_5050_vs_previous_best",

        "value":
            primary_blend_auc
            - PREVIOUS_BEST_BLEND_OOF,
    },

    {
        "metric":
            "final_lgbm_weight",

        "value":
            PRIMARY_LGBM_WEIGHT,
    },

    {
        "metric":
            "final_realmlp_weight",

        "value":
            PRIMARY_REALMLP_WEIGHT,
    },

    {
        "metric":
            "n_folds",

        "value":
            N_FOLDS,
    },

    {
        "metric":
            "realmlp_n_ens",

        "value":
            REALMLP_PARAMS[
                "n_ens"
            ],
    },

    {
        "metric":
            "realmlp_n_epochs",

        "value":
            REALMLP_PARAMS[
                "n_epochs"
            ],
    },

    {
        "metric":
            "lean_feature_count_before_te",

        "value":
            X_all.shape[1],
    },

    {
        "metric":
            "selected_te_feature_count",

        "value":
            len(combo_names),
    },

    {
        "metric":
            "final_feature_count",

        "value":
            X_all.shape[1]
            + len(combo_names),
    },
])


summary.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# FINAL FILE REPORT
# ============================================================

total_time = (
    time.time()
    - start_total
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
    f"\nRealMLP OOF:"
    f"\n{realmlp_oof_path}"
)


print(
    f"\nRealMLP TEST:"
    f"\n{realmlp_test_path}"
)


print(
    f"\nBlend OOF:"
    f"\n{blend_oof_path}"
)


print(
    f"\nBlend TEST:"
    f"\n{blend_test_path}"
)


print(
    f"\nBlend check:"
    f"\n{weight_check_path}"
)


print(
    f"\nSummary:"
    f"\n{summary_path}"
)


print(
    f"\nSubmission:"
    f"\n{submission_path}"
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