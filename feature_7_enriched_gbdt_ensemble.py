import os
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = r"D:\kaggle\airline\data\train.csv"
TEST_PATH = r"D:\kaggle\airline\data\test.csv"
SAMPLE_PATH = r"D:\kaggle\airline\data\sample_submission.csv"

OUTPUT_DIR = r"D:\kaggle\airline\enriched_gbdt_ensemble"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

SCREEN_FOLDS = 3
FINAL_FOLDS = 5
INNER_TE_FOLDS = 5

# Current best real model
CURRENT_BEST_OOF = 0.960664
CURRENT_BEST_LB = 0.96030

# Minimum screen improvement required to justify ensemble
MIN_BLEND_GAIN = 0.00003


# ============================================================
# MODEL PARAMETERS
# ============================================================

# ------------------------------------------------------------
# LightGBM
# Same core parameters as your current best Conditional-TE
# ------------------------------------------------------------

LGB_PARAMS = {
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


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",

    "learning_rate": 0.03,

    "max_depth": 6,
    "min_child_weight": 20,

    "subsample": 0.90,
    "colsample_bytree": 0.80,

    "gamma": 0,

    "reg_alpha": 0.10,
    "reg_lambda": 1.00,

    "n_estimators": 8000,

    "tree_method": "hist",

    "random_state": RANDOM_STATE,

    "n_jobs": -1,

    "verbosity": 0,

    "enable_categorical": True,

    "early_stopping_rounds": 150,
}


# ------------------------------------------------------------
# CatBoost
# ------------------------------------------------------------

CAT_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",

    "learning_rate": 0.05,

    "depth": 7,

    "l2_leaf_reg": 5,

    "iterations": 3000,

    "od_type": "Iter",
    "od_wait": 100,

    "random_seed": RANDOM_STATE,

    "thread_count": -1,

    "verbose": False,

    "allow_writing_files": False,
}


# ============================================================
# DATA
# ============================================================

print("=" * 90)
print("ENRICHED GBDT ENSEMBLE")
print("=" * 90)

print("LightGBM:", lgb.__version__)

train = pd.read_csv(
    TRAIN_PATH
)

test = pd.read_csv(
    TEST_PATH
)

TARGET = "satisfaction"
ID_COL = "id"

y = (
    train[TARGET]
    .astype(np.int8)
    .to_numpy()
)

print(
    f"\nTrain shape : {train.shape}"
)

print(
    f"Test shape  : {test.shape}"
)

print(
    f"Target mean : {y.mean():.6f}"
)


# ============================================================
# COLUMNS
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
# BASE MODEL FRAME
# ============================================================

def build_base(df):

    X = df.drop(
        columns=[
            ID_COL,
            TARGET,
        ],
        errors="ignore",
    ).copy()

    # Existing useful interaction
    X[
        "traveler_segment"
    ] = (
        df["Type of Travel"].astype(str)
        + "|"
        + df["Customer Type"].astype(str)
        + "|"
        + df["Class"].astype(str)
    )

    # Missing indicator
    X["arrival_missing"] = (
        df[
            "Arrival Delay in Minutes"
        ]
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

    dtype = (
        pd.api.types.CategoricalDtype(
            categories=categories
        )
    )

    X_BASE[c] = X_BASE[c].astype(dtype)
    X_TEST_BASE[c] = X_TEST_BASE[c].astype(dtype)


# ============================================================
# TARGET ENCODING INPUT
# ALL 21 ORIGINAL FEATURES
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


    # 13 service × traveler segment

    for c in SERVICES:

        Z[c + "__segment"] = (
            df[c].astype(str)
            + "|"
            + traveler_segment
        )


    # 5 selected service combinations

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
# NUMERIC-GROUP TE
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


Z_NUM_GROUP = build_numeric_groups(
    train
)

U_NUM_GROUP = build_numeric_groups(
    test
)


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
# FEATURE BUILDER FOR ONE OUTER FOLD
# ============================================================

def build_enriched_fold(
    tr_idx,
    va_idx,
):
    """
    Create the same 78-feature representation used by your
    current Conditional-TE model.

    IMPORTANT:
    Every target-based feature is learned ONLY from tr_idx.
    """

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
    # FINAL DATA CLEAN
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


    # Numerical columns
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
    # Remove duplicates
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


    assert X_tr.shape[1] == 78, (
        f"Expected 78 features, got "
        f"{X_tr.shape[1]}"
    )


    return (
        X_tr,
        X_va,
        X_te,
    )


# ============================================================
# MODEL-SPECIFIC TYPE PREPARATION
# ============================================================

def prepare_lgbm(
    X_tr,
    X_va,
    X_te,
):
    """
    LightGBM:
    categorical columns -> pandas category
    """

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

        dtype = (
            pd.api.types.CategoricalDtype(
                categories=categories
            )
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


def prepare_xgb(
    X_tr,
    X_va,
    X_te,
):
    """
    XGBoost:
    categorical columns -> pandas category
    """

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

        dtype = (
            pd.api.types.CategoricalDtype(
                categories=categories
            )
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


def prepare_catboost(
    X_tr,
    X_va,
    X_te,
):
    """
    CatBoost:
    categorical columns -> strings
    """

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

        X_tr[c] = (
            X_tr[c]
            .astype(str)
        )

        X_va[c] = (
            X_va[c]
            .astype(str)
        )

        X_te[c] = (
            X_te[c]
            .astype(str)
        )


    return (
        X_tr,
        X_va,
        X_te,
        cat_cols,
    )


# ============================================================
# TRAIN LGBM
# ============================================================

def train_lgbm(
    X_tr,
    y_tr,
    X_va,
    y_va,
    X_te,
):

    (
        X_tr,
        X_va,
        X_te,
        cat_cols,
    ) = prepare_lgbm(
        X_tr,
        X_va,
        X_te,
    )


    model = lgb.LGBMClassifier(
        **LGB_PARAMS
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
                150,
                verbose=False,
            )
        ],
    )


    val_pred = (
        model
        .predict_proba(
            X_va
        )[:, 1]
    )

    test_pred = (
        model
        .predict_proba(
            X_te
        )[:, 1]
    )

    return (
        val_pred,
        test_pred,
        model.best_iteration_,
    )


# ============================================================
# TRAIN XGB
# ============================================================

def train_xgb(
    X_tr,
    y_tr,
    X_va,
    y_va,
    X_te,
):

    (
        X_tr,
        X_va,
        X_te,
        cat_cols,
    ) = prepare_xgb(
        X_tr,
        X_va,
        X_te,
    )


    model = xgb.XGBClassifier(
        **XGB_PARAMS
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

        verbose=False,
    )


    val_pred = (
        model
        .predict_proba(
            X_va
        )[:, 1]
    )

    test_pred = (
        model
        .predict_proba(
            X_te
        )[:, 1]
    )


    best_iter = (
        model.best_iteration
        if hasattr(
            model,
            "best_iteration",
        )
        else model.n_estimators
    )


    return (
        val_pred,
        test_pred,
        best_iter,
    )


# ============================================================
# TRAIN CATBOOST
# ============================================================

def train_catboost(
    X_tr,
    y_tr,
    X_va,
    y_va,
    X_te,
):

    (
        X_tr,
        X_va,
        X_te,
        cat_cols,
    ) = prepare_catboost(
        X_tr,
        X_va,
        X_te,
    )


    cat_indices = [
        X_tr.columns.get_loc(c)
        for c in cat_cols
    ]


    model = CatBoostClassifier(
        **CAT_PARAMS
    )


    model.fit(
        X_tr,
        y_tr,

        eval_set=(
            X_va,
            y_va,
        ),

        cat_features=cat_indices,

        use_best_model=True,

        verbose=False,
    )


    val_pred = (
        model
        .predict_proba(
            X_va
        )[:, 1]
    )

    test_pred = (
        model
        .predict_proba(
            X_te
        )[:, 1]
    )


    best_iter = (
        model.get_best_iteration()
        + 1
    )


    return (
        val_pred,
        test_pred,
        best_iter,
    )


# ============================================================
# SCREENING
#
# 3-fold only.
# ============================================================

print("\n" + "=" * 90)
print("PHASE 1: 3-FOLD MODEL SCREENING")
print("=" * 90)

screen_skf = StratifiedKFold(
    n_splits=SCREEN_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


screen_oof = {
    "lgbm": np.zeros(
        len(train),
        dtype=np.float32,
    ),

    "xgb": np.zeros(
        len(train),
        dtype=np.float32,
    ),

    "catboost": np.zeros(
        len(train),
        dtype=np.float32,
    ),
}


screen_test = {
    "lgbm": np.zeros(
        len(test),
        dtype=np.float64,
    ),

    "xgb": np.zeros(
        len(test),
        dtype=np.float64,
    ),

    "catboost": np.zeros(
        len(test),
        dtype=np.float64,
    ),
}


screen_best_iters = {
    "lgbm": [],
    "xgb": [],
    "catboost": [],
}


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

    print("\n" + "-" * 90)
    print(
        f"SCREEN FOLD "
        f"{fold}/{SCREEN_FOLDS}"
    )
    print("-" * 90)


    # --------------------------------------------------------
    # Build EXACT SAME enriched features
    # --------------------------------------------------------

    (
        X_tr,
        X_va,
        X_te,
    ) = build_enriched_fold(
        tr_idx,
        va_idx,
    )


    print(
        f"Enriched features: "
        f"{X_tr.shape[1]}"
    )


    # ========================================================
    # LGBM
    # ========================================================

    print(
        "\n[LGBM] training..."
    )

    (
        p_va,
        p_te,
        best_iter,
    ) = train_lgbm(
        X_tr,
        y[tr_idx],
        X_va,
        y[va_idx],
        X_te,
    )

    screen_oof[
        "lgbm"
    ][va_idx] = p_va

    screen_test[
        "lgbm"
    ] += (
        p_te
        / SCREEN_FOLDS
    )

    screen_best_iters[
        "lgbm"
    ].append(
        best_iter
    )

    print(
        "LGBM fold AUC = "
        f"{roc_auc_score(y[va_idx], p_va):.6f}"
    )


    # ========================================================
    # XGB
    # ========================================================

    print(
        "\n[XGB] training..."
    )

    (
        p_va,
        p_te,
        best_iter,
    ) = train_xgb(
        X_tr,
        y[tr_idx],
        X_va,
        y[va_idx],
        X_te,
    )

    screen_oof[
        "xgb"
    ][va_idx] = p_va

    screen_test[
        "xgb"
    ] += (
        p_te
        / SCREEN_FOLDS
    )

    screen_best_iters[
        "xgb"
    ].append(
        best_iter
    )

    print(
        "XGB fold AUC = "
        f"{roc_auc_score(y[va_idx], p_va):.6f}"
    )


    # ========================================================
    # CATBOOST
    # ========================================================

    print(
        "\n[CatBoost] training..."
    )

    (
        p_va,
        p_te,
        best_iter,
    ) = train_catboost(
        X_tr,
        y[tr_idx],
        X_va,
        y[va_idx],
        X_te,
    )

    screen_oof[
        "catboost"
    ][va_idx] = p_va

    screen_test[
        "catboost"
    ] += (
        p_te
        / SCREEN_FOLDS
    )

    screen_best_iters[
        "catboost"
    ].append(
        best_iter
    )

    print(
        "CatBoost fold AUC = "
        f"{roc_auc_score(y[va_idx], p_va):.6f}"
    )


# ============================================================
# SCREEN RESULTS
# ============================================================

screen_auc = {
    name: roc_auc_score(
        y,
        pred,
    )

    for name, pred
    in screen_oof.items()
}


print("\n" + "=" * 90)
print("3-FOLD SCREEN RESULTS")
print("=" * 90)

for name, auc in sorted(
    screen_auc.items(),
    key=lambda x: x[1],
    reverse=True,
):

    print(
        f"{name:10s}: "
        f"{auc:.6f}"
    )

    print(
        f"   best iterations: "
        f"{screen_best_iters[name]}"
    )


# ============================================================
# PREDICTION CORRELATIONS
# ============================================================

print("\nPrediction correlations:")

names = [
    "lgbm",
    "xgb",
    "catboost",
]

for i in range(
    len(names)
):

    for j in range(
        i + 1,
        len(names),
    ):

        a = names[i]
        b = names[j]

        corr = np.corrcoef(
            screen_oof[a],
            screen_oof[b],
        )[0, 1]

        print(
            f"{a:10s} / "
            f"{b:10s}: "
            f"{corr:.6f}"
        )


# ============================================================
# 3-MODEL BLEND SEARCH
#
# Coarse grid.
# ============================================================

print("\n" + "=" * 90)
print("3-FOLD BLEND SEARCH")
print("=" * 90)


best_screen_blend_auc = -np.inf
best_screen_weights = None


blend_records = []


for w_lgbm in np.arange(
    0.0,
    1.0001,
    0.05,
):

    for w_xgb in np.arange(
        0.0,
        1.0001 - w_lgbm,
        0.05,
    ):

        w_cat = (
            1.0
            - w_lgbm
            - w_xgb
        )

        if w_cat < -1e-9:
            continue


        blend = (
            w_lgbm
            * screen_oof["lgbm"]

            + w_xgb
            * screen_oof["xgb"]

            + w_cat
            * screen_oof["catboost"]
        )


        auc = roc_auc_score(
            y,
            blend,
        )


        blend_records.append(
            {
                "lgbm_weight": w_lgbm,
                "xgb_weight": w_xgb,
                "catboost_weight": w_cat,
                "auc": auc,
            }
        )


        if auc > best_screen_blend_auc:

            best_screen_blend_auc = auc

            best_screen_weights = (
                w_lgbm,
                w_xgb,
                w_cat,
            )


blend_df = (
    pd.DataFrame(
        blend_records
    )
    .sort_values(
        "auc",
        ascending=False,
    )
    .reset_index(drop=True)
)


print(
    "\nTop blend candidates:"
)

print(
    blend_df.head(15).to_string(
        index=False
    )
)


print(
    "\nBest screen blend:"
)

print(
    f"LGBM      = "
    f"{best_screen_weights[0]:.2f}"
)

print(
    f"XGB       = "
    f"{best_screen_weights[1]:.2f}"
)

print(
    f"CatBoost  = "
    f"{best_screen_weights[2]:.2f}"
)

print(
    f"Blend AUC = "
    f"{best_screen_blend_auc:.6f}"
)


# ============================================================
# SAVE SCREEN RESULTS
# ============================================================

blend_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "screen_blend_results.csv",
    ),
    index=False,
    encoding="utf-8-sig",
)


screen_summary = pd.DataFrame([
    {
        "model": name,
        "screen_auc": auc,
    }

    for name, auc in screen_auc.items()
])

screen_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "screen_model_results.csv",
    ),
    index=False,
)


# ============================================================
# DECIDE WHETHER ENSEMBLE IS WORTH FINAL TRAINING
# ============================================================

best_single_name = max(
    screen_auc,
    key=screen_auc.get,
)

best_single_auc = screen_auc[
    best_single_name
]


blend_gain_vs_best = (
    best_screen_blend_auc
    - best_single_auc
)


print("\n" + "=" * 90)
print("SCREEN DECISION")
print("=" * 90)

print(
    f"Best single model : "
    f"{best_single_name}"
)

print(
    f"Best single AUC   : "
    f"{best_single_auc:.6f}"
)

print(
    f"Best blend AUC    : "
    f"{best_screen_blend_auc:.6f}"
)

print(
    f"Blend gain        : "
    f"{blend_gain_vs_best:+.6f}"
)


# ------------------------------------------------------------
# Final model selection
# ------------------------------------------------------------

if (
    blend_gain_vs_best
    >= MIN_BLEND_GAIN
):

    final_models = [
        "lgbm",
        "xgb",
        "catboost",
    ]

    final_weights = np.array(
        best_screen_weights,
        dtype=float,
    )

    # Remove effectively zero weights
    mask = final_weights > 0.02

    final_models = [
        m
        for m, keep
        in zip(
            [
                "lgbm",
                "xgb",
                "catboost",
            ],
            mask,
        )
        if keep
    ]

    final_weights = final_weights[
        mask
    ]

    final_weights = (
        final_weights
        / final_weights.sum()
    )

    final_mode = "ensemble"

else:

    # Only the best single model
    final_models = [
        best_single_name
    ]

    final_weights = np.array(
        [1.0]
    )

    final_mode = "best_single"


print(
    f"\nFinal mode = "
    f"{final_mode}"
)

print(
    "Final models:"
)

for model_name, weight in zip(
    final_models,
    final_weights,
):

    print(
        f"  {model_name:10s} "
        f"weight={weight:.4f}"
    )


# ============================================================
# FINAL 5-FOLD
# ============================================================

print("\n" + "=" * 90)
print("PHASE 2: FINAL 5-FOLD")
print("=" * 90)

final_skf = StratifiedKFold(
    n_splits=FINAL_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


final_oof = {
    name: np.zeros(
        len(train),
        dtype=np.float32,
    )

    for name in final_models
}


final_test = {
    name: np.zeros(
        len(test),
        dtype=np.float64,
    )

    for name in final_models
}


final_iters = {
    name: []

    for name in final_models
}


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

    print("\n" + "-" * 90)
    print(
        f"FINAL FOLD "
        f"{fold}/{FINAL_FOLDS}"
    )
    print("-" * 90)


    # --------------------------------------------------------
    # Build enriched features
    # --------------------------------------------------------

    (
        X_tr,
        X_va,
        X_te,
    ) = build_enriched_fold(
        tr_idx,
        va_idx,
    )


    print(
        f"Features = "
        f"{X_tr.shape[1]}"
    )


    # ========================================================
    # MODELS
    # ========================================================

    for model_name in final_models:

        print(
            f"\n[{model_name}]"
        )


        if model_name == "lgbm":

            (
                p_va,
                p_te,
                best_iter,
            ) = train_lgbm(
                X_tr,
                y[tr_idx],
                X_va,
                y[va_idx],
                X_te,
            )


        elif model_name == "xgb":

            (
                p_va,
                p_te,
                best_iter,
            ) = train_xgb(
                X_tr,
                y[tr_idx],
                X_va,
                y[va_idx],
                X_te,
            )


        elif model_name == "catboost":

            (
                p_va,
                p_te,
                best_iter,
            ) = train_catboost(
                X_tr,
                y[tr_idx],
                X_va,
                y[va_idx],
                X_te,
            )


        final_oof[
            model_name
        ][va_idx] = p_va

        final_test[
            model_name
        ] += (
            p_te
            / FINAL_FOLDS
        )

        final_iters[
            model_name
        ].append(
            best_iter
        )


        fold_auc = roc_auc_score(
            y[va_idx],
            p_va,
        )


        print(
            f"{model_name} "
            f"Fold {fold} AUC = "
            f"{fold_auc:.6f}"
        )


# ============================================================
# FINAL INDIVIDUAL OOF
# ============================================================

print("\n" + "=" * 90)
print("FINAL 5-FOLD MODEL RESULTS")
print("=" * 90)


final_auc = {}

for name in final_models:

    auc = roc_auc_score(
        y,
        final_oof[name],
    )

    final_auc[name] = auc

    print(
        f"{name:10s}: "
        f"{auc:.6f}"
    )

    print(
        f"  iterations = "
        f"{final_iters[name]}"
    )


# ============================================================
# FINAL ENSEMBLE
# ============================================================

final_blend_oof = np.zeros(
    len(train),
    dtype=np.float64,
)

final_blend_test = np.zeros(
    len(test),
    dtype=np.float64,
)


print("\nFinal weights:")

for name, weight in zip(
    final_models,
    final_weights,
):

    print(
        f"{name:10s}: "
        f"{weight:.4f}"
    )

    final_blend_oof += (
        weight
        * final_oof[name]
    )

    final_blend_test += (
        weight
        * final_test[name]
    )


final_blend_auc = roc_auc_score(
    y,
    final_blend_oof,
)


print("\n" + "=" * 90)
print("FINAL ENRICHED GBDT RESULT")
print("=" * 90)

for name, auc in final_auc.items():

    print(
        f"{name:10s}: "
        f"{auc:.6f}"
    )


print(
    f"\nFinal blend OOF = "
    f"{final_blend_auc:.6f}"
)

print(
    f"Current best OOF = "
    f"{CURRENT_BEST_OOF:.6f}"
)

print(
    f"Delta vs current best = "
    f"{final_blend_auc - CURRENT_BEST_OOF:+.6f}"
)

print(
    f"Current best LB = "
    f"{CURRENT_BEST_LB:.5f}"
)


# ============================================================
# SAVE FINAL OOF
# ============================================================

oof_output = pd.DataFrame({
    "id": train[ID_COL],
    "satisfaction": y,
})


for name in final_models:

    oof_output[
        f"{name}_oof"
    ] = final_oof[name]


oof_output[
    "final_ensemble_oof"
] = final_blend_oof


oof_path = os.path.join(
    OUTPUT_DIR,
    "enriched_gbdt_oof_predictions.csv",
)


oof_output.to_csv(
    oof_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_output = pd.DataFrame({
    "id": test[ID_COL],
})


for name in final_models:

    test_output[
        f"{name}_prediction"
    ] = final_test[name]


test_output[
    "final_ensemble_prediction"
] = final_blend_test


test_pred_path = os.path.join(
    OUTPUT_DIR,
    "enriched_gbdt_test_predictions.csv",
)


test_output.to_csv(
    test_pred_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SUBMISSION
#
# Continuous probability.
# NO threshold.
# ============================================================

sample = pd.read_csv(
    SAMPLE_PATH
)

submission_target = [
    c
    for c in sample.columns
    if c != ID_COL
][0]


submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    submission_target: final_blend_test,
})


submission_path = os.path.join(
    OUTPUT_DIR,
    "submission_14_enriched_gbdt_ensemble.csv",
)


submission.to_csv(
    submission_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# TEST STATISTICS
# ============================================================

print("\n" + "=" * 90)
print("TEST PREDICTION")
print("=" * 90)

print(
    f"min  = "
    f"{final_blend_test.min():.6f}"
)

print(
    f"max  = "
    f"{final_blend_test.max():.6f}"
)

print(
    f"mean = "
    f"{final_blend_test.mean():.6f}"
)

print(
    f"std  = "
    f"{final_blend_test.std():.6f}"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame({
    "metric": [
        "current_best_oof",
        "current_best_lb",
        "best_final_model_oof",
        "final_ensemble_oof",
        "gain_vs_current_best",
        "screen_blend_gain_vs_best_single",
    ],

    "value": [
        CURRENT_BEST_OOF,
        CURRENT_BEST_LB,
        max(final_auc.values()),
        final_blend_auc,
        final_blend_auc - CURRENT_BEST_OOF,
        blend_gain_vs_best,
    ],
})


summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "enriched_gbdt_summary.csv",
    ),
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# FILE OUTPUT
# ============================================================

print("\n" + "=" * 90)
print("FILES")
print("=" * 90)

print(
    "OOF:"
)

print(
    oof_path
)

print(
    "\nTest predictions:"
)

print(
    test_pred_path
)

print(
    "\nSubmission:"
)

print(
    submission_path
)

print("\nDone.")