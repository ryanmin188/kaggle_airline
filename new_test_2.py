
# -*- coding: utf-8 -*-

"""
REALMLP V1 FEATURE ABLATION - FIXED VERSION
============================================================

目的
----
系统比较 Lean RealMLP V1 不同 feature family 的作用。

严格限制：
    1. 只使用比赛 train.csv / test.csv
    2. 不使用原始 129k Airline Passenger Satisfaction 数据
    3. 不引入外部数据
    4. 5-fold Stratified CV
    5. 每个 variant：
        - RealMLP OOF AUC
        - 与当前 LGBM 的 rank correlation
        - LGBM 40% + RealMLP 60% rank blend OOF AUC
        - raw blend OOF AUC
        - 保存 OOF / test prediction

本版本主要修复：
    ValueError: NaN values in continuous columns are currently not allowed!

修复方式：
    - categorical 缺失 -> "__MISSING__"
    - numeric 缺失 -> 每个 CV fold 使用训练集 median 填充
    - test 使用对应 fold 的 train median
    - 不使用 target 参与缺失值填充
    - RealMLP fit 前进行最终 NaN 检查

另外：
    numeric -> categorical 的分箱使用 train+test 联合无监督 rank，
    保证 train/test 的 bin 定义一致。

当前 Champion：
    LGBM 40% + RealMLP V1 60%
    rank fusion
"""


# ============================================================
# IMPORTS
# ============================================================

import os
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from pytabkit import RealMLP_TD_Classifier

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

SEED = 42
N_SPLITS = 5

DATA_DIR = Path(r"D:\kaggle\airline\data")

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

LGBM_OOF_PATH = Path(
    r"D:\kaggle\airline\conditional_te_experiment\conditional_te_lgbm_oof.csv"
)

LGBM_TEST_PATH = Path(
    r"D:\kaggle\airline\conditional_te_experiment\conditional_te_lgbm_test.csv"
)

OUTPUT_DIR = Path(
    r"D:\kaggle\airline\realmlp_v1_ablation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Current Champion
# ------------------------------------------------------------

LGBM_WEIGHT = 0.40
MLP_WEIGHT = 0.60


# ============================================================
# ORIGINAL COLUMNS
# ============================================================

TARGET = "satisfaction"
ID_COL = "id"


CAT_COLS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]


NUM_COLS = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]


SERVICE_COLS = [
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


# ============================================================
# V1 FEATURE DEFINITIONS
# ============================================================

NUM_CAT_COLS = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]


COUNT_BASE_COLS = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]


INTERACTION_COLS = [
    "Class_Type of Travel",
    "Class_Gender",
    "Type of Travel_Gender",
]


TE_COLS = [
    "Class_Type of Travel_Gender_TE",
    "Class_Type of Travel_Age_TE",
    "Flight Distance_Departure/Arrival time convenient_TE",
]


# ============================================================
# REALMLP PARAMETERS
# ============================================================

REALMLP_PARAMS = dict(
    random_state=SEED,
    verbosity=0,

    val_metric_name="1-auc_ovr",

    n_ens=8,
    n_epochs=3,

    batch_size=256,

    use_early_stopping=False,
    early_stopping_additive_patience=10,
    early_stopping_multiplicative_patience=1,

    lr=0.053,
    wd=0.015,

    sq_mom=0.988,

    lr_sched="flat_anneal",
    wd_sched="cos_log_15",

    first_layer_lr_factor=0.25,

    embedding_size=5,
    max_one_hot_cat_size=18,

    hidden_sizes=[512, 256, 128],

    act="silu",

    p_drop=0.05,
    p_drop_sched="invsqrtp1e-3",

    plr_hidden_1=16,
    plr_hidden_2=8,

    plr_act_name="gelu",
    plr_lr_factor=0.1151,
    plr_sigma=2.33,

    ls_eps=0.01,
    ls_eps_sched="sqrt_cos",

    add_front_scale=False,

    bias_init_mode="neg-uniform-dynamic-2",

    tfms=[
        "one_hot",
        "median_center",
        "robust_scale",
        "smooth_clip",
        "embedding",
        "l2_normalize",
    ],
)


# ============================================================
# PRINT HELPERS
# ============================================================

def banner(text):
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)


# ============================================================
# LOAD DATA
# ============================================================

banner("LOAD COMPETITION DATA")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Train:", train.shape)
print("Test :", test.shape)

assert TARGET in train.columns
assert ID_COL in train.columns

y = train[TARGET].astype(int).values

print(
    f"Target mean: {y.mean():.6f}"
)


# ============================================================
# LOAD LGBM PREDICTIONS
# ============================================================

banner("LOAD CURRENT LGBM PREDICTIONS")

lgbm_oof = pd.read_csv(
    LGBM_OOF_PATH
)

lgbm_test = pd.read_csv(
    LGBM_TEST_PATH
)

print(
    "LGBM OOF :",
    lgbm_oof.shape,
)

print(
    "LGBM TEST:",
    lgbm_test.shape,
)


def detect_prediction_column(
    df,
    id_col="id",
):

    candidates = [
        "prediction",
        "pred",
        "oof_prediction",
        "ensemble_prediction",
        "conditional_te_lgbm_oof",
        "conditional_te_lgbm_test",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    numeric_candidates = [
        c
        for c in df.columns
        if c != id_col
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(numeric_candidates) == 1:
        return numeric_candidates[0]

    raise ValueError(
        "Cannot automatically identify prediction column.\n"
        f"Columns = {list(df.columns)}"
    )


lgbm_oof_pred_col = detect_prediction_column(
    lgbm_oof
)

lgbm_test_pred_col = detect_prediction_column(
    lgbm_test
)

print(
    "LGBM OOF prediction column :",
    lgbm_oof_pred_col,
)

print(
    "LGBM TEST prediction column:",
    lgbm_test_pred_col,
)


# ============================================================
# ALIGN LGBM
# ============================================================

if ID_COL in lgbm_oof.columns:

    lgbm_oof_aligned = (
        train[[ID_COL]]
        .merge(
            lgbm_oof[
                [ID_COL, lgbm_oof_pred_col]
            ],
            on=ID_COL,
            how="left",
            validate="one_to_one",
        )
    )

    lgbm_oof_pred = (
        lgbm_oof_aligned[
            lgbm_oof_pred_col
        ]
        .values
        .astype(np.float64)
    )

else:

    lgbm_oof_pred = (
        lgbm_oof[
            lgbm_oof_pred_col
        ]
        .values
        .astype(np.float64)
    )


if ID_COL in lgbm_test.columns:

    lgbm_test_aligned = (
        test[[ID_COL]]
        .merge(
            lgbm_test[
                [ID_COL, lgbm_test_pred_col]
            ],
            on=ID_COL,
            how="left",
            validate="one_to_one",
        )
    )

    lgbm_test_pred = (
        lgbm_test_aligned[
            lgbm_test_pred_col
        ]
        .values
        .astype(np.float64)
    )

else:

    lgbm_test_pred = (
        lgbm_test[
            lgbm_test_pred_col
        ]
        .values
        .astype(np.float64)
    )


assert len(lgbm_oof_pred) == len(train)
assert len(lgbm_test_pred) == len(test)

assert np.isfinite(lgbm_oof_pred).all()
assert np.isfinite(lgbm_test_pred).all()

print(
    f"\nCurrent LGBM OOF AUC: "
    f"{roc_auc_score(y, lgbm_oof_pred):.9f}"
)


# ============================================================
# BASE FEATURE ENGINEERING
# ============================================================

banner("BUILD BASE V1 FEATURES")


def build_base_features(
    df,
    combined_for_bins=None,
):
    """
    Build all non-target V1 features.

    Important:
        numeric -> categorical bins are created from a shared
        train+test unsupervised rank space.

    No target information is used.
    """

    out = pd.DataFrame(
        index=df.index
    )

    # --------------------------------------------------------
    # 1. Raw features
    # --------------------------------------------------------

    for c in CAT_COLS:

        out[c] = (
            df[c]
            .astype(str)
            .fillna("__MISSING__")
        )

    for c in NUM_COLS:

        out[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    # --------------------------------------------------------
    # 2. Numeric -> categorical copies
    # --------------------------------------------------------

    if combined_for_bins is None:
        combined_for_bins = df

    for c in NUM_CAT_COLS:

        combined_s = pd.to_numeric(
            combined_for_bins[c],
            errors="coerce",
        )

        current_s = pd.to_numeric(
            df[c],
            errors="coerce",
        )

        # rank pct over combined train + test
        combined_rank = (
            combined_s
            .rank(
                method="average",
                pct=True,
            )
        )

        current_rank = (
            current_s
            .rank(
                method="average",
                pct=True,
            )
        )

        # We actually need the rank thresholds from the
        # combined frame rather than independently cutting
        # each dataset.
        #
        # Quantile boundaries are based only on X.
        valid_values = combined_s.dropna()

        if len(valid_values) > 0:

            quantiles = valid_values.quantile(
                [
                    0.05,
                    0.10,
                    0.20,
                    0.40,
                    0.60,
                    0.80,
                    0.90,
                    0.95,
                ]
            ).values

            # Ensure strictly increasing boundaries.
            quantiles = np.unique(
                quantiles
            )

            if len(quantiles) == 0:

                bins = np.full(
                    len(current_s),
                    0,
                    dtype=np.int32,
                )

            else:

                bins = np.digitize(
                    current_s.values,
                    quantiles,
                    right=True,
                )

                bins = bins.astype(
                    np.int32
                )

                bins[
                    current_s.isna().values
                ] = -1

        else:

            bins = np.full(
                len(current_s),
                -1,
                dtype=np.int32,
            )

        out[
            c + "_cat"
        ] = bins.astype(str)

    # --------------------------------------------------------
    # 3. Frequency / count features
    # --------------------------------------------------------

    for c in COUNT_BASE_COLS:

        s = (
            df[c]
            .astype(str)
            .fillna("__MISSING__")
        )

        freq = (
            s.value_counts(
                dropna=False
            )
        )

        count_values = (
            s.map(freq)
            .astype(np.float32)
        )

        out[
            c + "_count"
        ] = np.log1p(
            count_values
        ).astype(np.float32)

    # --------------------------------------------------------
    # 4. Selected interactions
    # --------------------------------------------------------

    out[
        "Class_Type of Travel"
    ] = (
        df["Class"]
        .astype(str)
        .fillna("__MISSING__")
        + "_"
        + df["Type of Travel"]
        .astype(str)
        .fillna("__MISSING__")
    )

    out[
        "Class_Gender"
    ] = (
        df["Class"]
        .astype(str)
        .fillna("__MISSING__")
        + "_"
        + df["Gender"]
        .astype(str)
        .fillna("__MISSING__")
    )

    out[
        "Type of Travel_Gender"
    ] = (
        df["Type of Travel"]
        .astype(str)
        .fillna("__MISSING__")
        + "_"
        + df["Gender"]
        .astype(str)
        .fillna("__MISSING__")
    )

    return out


# ------------------------------------------------------------
# Shared unsupervised frame for numeric bins
# ------------------------------------------------------------

combined_numeric = pd.concat(
    [
        train.drop(columns=[TARGET]),
        test,
    ],
    axis=0,
    ignore_index=True,
)

base_train = build_base_features(
    train,
    combined_for_bins=combined_numeric,
)

base_test = build_base_features(
    test,
    combined_for_bins=combined_numeric,
)

print(
    "Base train features:",
    base_train.shape,
)

print(
    "Base test features :",
    base_test.shape,
)


# ============================================================
# TARGET ENCODING
# ============================================================

banner("BUILD TARGET ENCODING FEATURES")


def make_target_encoding(
    train_series,
    y,
    test_series,
    n_splits=5,
    seed=42,
):
    """
    OOF target encoding.

    Train:
        fold-wise OOF TE

    Test:
        full-train mapping

    Smoothing:
        alpha = 20
    """

    train_series = (
        train_series
        .astype(str)
        .fillna("__MISSING__")
    )

    test_series = (
        test_series
        .astype(str)
        .fillna("__MISSING__")
    )

    global_mean = float(
        np.mean(y)
    )

    tr_te = np.zeros(
        len(train_series),
        dtype=np.float32,
    )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    alpha = 20.0

    for tr_idx, val_idx in skf.split(
        train_series,
        y,
    ):

        tmp = pd.DataFrame(
            {
                "key": train_series.iloc[
                    tr_idx
                ].values,

                "target": y[
                    tr_idx
                ],
            }
        )

        stats = (
            tmp
            .groupby("key")["target"]
            .agg(
                ["mean", "count"]
            )
        )

        smoothed = (
            stats["mean"] * stats["count"]
            + global_mean * alpha
        ) / (
            stats["count"] + alpha
        )

        mapped = (
            train_series.iloc[
                val_idx
            ]
            .map(smoothed)
            .fillna(global_mean)
        )

        tr_te[val_idx] = (
            mapped.values
        )

    # --------------------------------------------------------
    # Full train mapping for test
    # --------------------------------------------------------

    full = pd.DataFrame(
        {
            "key": train_series.values,
            "target": y,
        }
    )

    stats = (
        full
        .groupby("key")["target"]
        .agg(
            ["mean", "count"]
        )
    )

    smoothed = (
        stats["mean"] * stats["count"]
        + global_mean * alpha
    ) / (
        stats["count"] + alpha
    )

    te_test = (
        test_series
        .map(smoothed)
        .fillna(global_mean)
        .values
        .astype(np.float32)
    )

    assert np.isfinite(
        tr_te
    ).all()

    assert np.isfinite(
        te_test
    ).all()

    return tr_te, te_test


# ------------------------------------------------------------
# TE 1
# Class × Type of Travel × Gender
# ------------------------------------------------------------

key_train_1 = (
    train["Class"].astype(str)
    + "_"
    + train["Type of Travel"].astype(str)
    + "_"
    + train["Gender"].astype(str)
)

key_test_1 = (
    test["Class"].astype(str)
    + "_"
    + test["Type of Travel"].astype(str)
    + "_"
    + test["Gender"].astype(str)
)


# ------------------------------------------------------------
# TE 2
# Class × Type of Travel × Age
# ------------------------------------------------------------

key_train_2 = (
    train["Class"].astype(str)
    + "_"
    + train["Type of Travel"].astype(str)
    + "_"
    + train["Age"].astype(str)
)

key_test_2 = (
    test["Class"].astype(str)
    + "_"
    + test["Type of Travel"].astype(str)
    + "_"
    + test["Age"].astype(str)
)


# ------------------------------------------------------------
# TE 3
# Flight Distance × Departure/Arrival time convenient
# ------------------------------------------------------------

key_train_3 = (
    train["Flight Distance"].astype(str)
    + "_"
    + train[
        "Departure/Arrival time convenient"
    ].astype(str)
)

key_test_3 = (
    test["Flight Distance"].astype(str)
    + "_"
    + test[
        "Departure/Arrival time convenient"
    ].astype(str)
)


te_train = pd.DataFrame(
    index=train.index
)

te_test = pd.DataFrame(
    index=test.index
)


tr, te = make_target_encoding(
    key_train_1,
    y,
    key_test_1,
    n_splits=5,
    seed=SEED,
)

te_train[
    TE_COLS[0]
] = tr

te_test[
    TE_COLS[0]
] = te


tr, te = make_target_encoding(
    key_train_2,
    y,
    key_test_2,
    n_splits=5,
    seed=SEED + 1,
)

te_train[
    TE_COLS[1]
] = tr

te_test[
    TE_COLS[1]
] = te


tr, te = make_target_encoding(
    key_train_3,
    y,
    key_test_3,
    n_splits=5,
    seed=SEED + 2,
)

te_train[
    TE_COLS[2]
] = tr

te_test[
    TE_COLS[2]
] = te


print(
    "TE features:",
    te_train.shape[1],
)


# ============================================================
# COMBINE
# ============================================================

all_train_features = pd.concat(
    [
        base_train,
        te_train,
    ],
    axis=1,
)

all_test_features = pd.concat(
    [
        base_test,
        te_test,
    ],
    axis=1,
)

print(
    "\nAll V1 train features:",
    all_train_features.shape,
)

print(
    "All V1 test features :",
    all_test_features.shape,
)


# ============================================================
# FEATURE FAMILY IDENTIFICATION
# ============================================================

def get_feature_families(
    df
):

    raw = []
    numeric_cat = []
    count = []
    interaction = []
    te = []
    other = []

    for c in df.columns:

        if (
            c in CAT_COLS
            or c in NUM_COLS
        ):
            raw.append(c)

        elif c.endswith(
            "_cat"
        ):
            numeric_cat.append(c)

        elif c.endswith(
            "_count"
        ):
            count.append(c)

        elif c in INTERACTION_COLS:
            interaction.append(c)

        elif c in TE_COLS:
            te.append(c)

        else:
            other.append(c)

    return {
        "raw": raw,
        "numeric_cat": numeric_cat,
        "count": count,
        "interaction": interaction,
        "te": te,
        "other": other,
    }


families = get_feature_families(
    all_train_features
)


banner("FEATURE FAMILY SUMMARY")

for k, v in families.items():

    print(
        f"{k:15s}: {len(v):3d}"
    )

    if len(v) > 0:

        print(
            " " * 17
            + ", ".join(v)
        )


# ============================================================
# ABLATION VARIANTS
# ============================================================

VARIANTS = {

    # Current V1
    "A_full_v1": [
        "raw",
        "numeric_cat",
        "count",
        "interaction",
        "te",
    ],

    # Remove TE
    "B_no_te": [
        "raw",
        "numeric_cat",
        "count",
        "interaction",
    ],

    # Remove count
    "C_no_count": [
        "raw",
        "numeric_cat",
        "interaction",
        "te",
    ],

    # Remove numeric categorical copies
    "D_no_numeric_cat": [
        "raw",
        "count",
        "interaction",
        "te",
    ],

    # Raw + interaction
    "E_raw_interaction": [
        "raw",
        "interaction",
    ],

    # Raw + interaction + count
    "F_raw_interaction_count": [
        "raw",
        "interaction",
        "count",
    ],

    # Raw + interaction + TE
    "G_raw_interaction_te": [
        "raw",
        "interaction",
        "te",
    ],

    # Raw + numeric-cat + interaction + count
    "H_raw_numericcat_interaction_count": [
        "raw",
        "numeric_cat",
        "interaction",
        "count",
    ],

    # Raw + numeric-cat + interaction + TE
    "I_raw_numericcat_interaction_te": [
        "raw",
        "numeric_cat",
        "interaction",
        "te",
    ],

    # Full V1 again as consistency check
    "J_full_check": [
        "raw",
        "numeric_cat",
        "count",
        "interaction",
        "te",
    ],
}


# ============================================================
# SELECT VARIANT FEATURES
# ============================================================

def select_variant_features(
    df,
    selected_families,
):

    cols = []

    for family in selected_families:

        cols.extend(
            families[family]
        )

    cols = list(
        dict.fromkeys(cols)
    )

    result = df[
        cols
    ].copy()

    return result


# ============================================================
# REALMLP DATA PREPARATION
# ============================================================

def prepare_categorical_columns(
    train_df,
    test_df,
):
    """
    Convert categorical columns to pandas category.

    Missing categorical values are represented as
    "__MISSING__".
    """

    tr = train_df.copy()
    te = test_df.copy()

    categorical_cols = []

    for c in tr.columns:

        if (
            tr[c].dtype == "object"
            or c.endswith("_cat")
            or c in INTERACTION_COLS
        ):

            categorical_cols.append(c)

    for c in categorical_cols:

        tr[c] = (
            tr[c]
            .astype(str)
            .replace(
                {
                    "nan": "__MISSING__",
                    "None": "__MISSING__",
                }
            )
        )

        te[c] = (
            te[c]
            .astype(str)
            .replace(
                {
                    "nan": "__MISSING__",
                    "None": "__MISSING__",
                }
            )
        )

        combined = pd.concat(
            [
                tr[c],
                te[c],
            ],
            axis=0,
        )

        categories = pd.Index(
            combined
            .dropna()
            .unique()
        )

        tr[c] = pd.Categorical(
            tr[c],
            categories=categories,
        )

        te[c] = pd.Categorical(
            te[c],
            categories=categories,
        )

    return (
        tr,
        te,
        categorical_cols,
    )


# ============================================================
# NUMERIC IMPUTATION
# ============================================================

def fit_numeric_medians(
    X_train,
    numeric_cols,
):
    """
    Fit medians using training fold only.
    """

    medians = {}

    for c in numeric_cols:

        s = pd.to_numeric(
            X_train[c],
            errors="coerce",
        )

        median = s.median()

        if pd.isna(median):

            median = 0.0

        medians[c] = float(
            median
        )

    return medians


def apply_numeric_medians(
    X,
    numeric_cols,
    medians,
):
    """
    Apply fold-specific train medians.
    """

    X = X.copy()

    for c in numeric_cols:

        X[c] = pd.to_numeric(
            X[c],
            errors="coerce",
        )

        X[c] = X[c].fillna(
            medians[c]
        )

        X[c] = X[c].astype(
            np.float32
        )

    return X


# ============================================================
# PREPARE ONE FOLD
# ============================================================

def prepare_fold_data(
    X_fold_train,
    X_fold_valid,
    X_test,
):
    """
    Prepare RealMLP input for one fold.

    Critical:
        numeric median is fitted ONLY on X_fold_train.

    This prevents validation leakage.
    """

    tr = X_fold_train.copy()
    va = X_fold_valid.copy()
    te = X_test.copy()

    # --------------------------------------------------------
    # Detect categorical columns
    # --------------------------------------------------------

    categorical_cols = []

    for c in tr.columns:

        if (
            tr[c].dtype == "object"
            or c.endswith("_cat")
            or c in INTERACTION_COLS
        ):

            categorical_cols.append(c)

    # --------------------------------------------------------
    # Categorical handling
    # --------------------------------------------------------

    for c in categorical_cols:

        tr[c] = (
            tr[c]
            .astype(str)
            .replace(
                {
                    "nan": "__MISSING__",
                    "None": "__MISSING__",
                }
            )
        )

        va[c] = (
            va[c]
            .astype(str)
            .replace(
                {
                    "nan": "__MISSING__",
                    "None": "__MISSING__",
                }
            )
        )

        te[c] = (
            te[c]
            .astype(str)
            .replace(
                {
                    "nan": "__MISSING__",
                    "None": "__MISSING__",
                }
            )
        )

        categories = pd.Index(
            pd.concat(
                [
                    tr[c],
                    va[c],
                    te[c],
                ],
                axis=0,
            )
            .dropna()
            .unique()
        )

        tr[c] = pd.Categorical(
            tr[c],
            categories=categories,
        )

        va[c] = pd.Categorical(
            va[c],
            categories=categories,
        )

        te[c] = pd.Categorical(
            te[c],
            categories=categories,
        )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_cols = [
        c
        for c in tr.columns
        if c not in categorical_cols
    ]

    # Fit medians ONLY on training fold
    medians = fit_numeric_medians(
        tr,
        numeric_cols,
    )

    tr = apply_numeric_medians(
        tr,
        numeric_cols,
        medians,
    )

    va = apply_numeric_medians(
        va,
        numeric_cols,
        medians,
    )

    te = apply_numeric_medians(
        te,
        numeric_cols,
        medians,
    )

    # --------------------------------------------------------
    # Final sanity checks
    # --------------------------------------------------------

    for name, df in [
        ("train", tr),
        ("valid", va),
        ("test", te),
    ]:

        for c in numeric_cols:

            values = df[c].values

            if not np.isfinite(
                values
            ).all():

                raise ValueError(
                    f"Non-finite values remain "
                    f"in {name}.{c}"
                )

    return tr, va, te


# ============================================================
# TRAIN ONE VARIANT
# ============================================================

def run_realmlp_variant(
    variant_name,
    X_train,
    X_test,
    y,
    lgbm_oof_pred,
):

    banner(
        f"RUN VARIANT: {variant_name}"
    )

    print(
        "Number of features:",
        X_train.shape[1],
    )

    print("\nFeatures:")

    for i, c in enumerate(
        X_train.columns,
        start=1,
    ):

        print(
            f"{i:4d}. {c}"
        )

    # --------------------------------------------------------
    # Check raw source for missing values
    # --------------------------------------------------------

    missing_summary = (
        X_train.isna()
        .sum()
    )

    missing_summary = (
        missing_summary[
            missing_summary > 0
        ]
        .sort_values(
            ascending=False
        )
    )

    if len(
        missing_summary
    ) > 0:

        print(
            "\nMissing values before "
            "fold preprocessing:"
        )

        print(
            missing_summary
            .head(20)
            .to_string()
        )

    else:

        print(
            "\nNo missing values detected "
            "before fold preprocessing."
        )

    # --------------------------------------------------------
    # CV
    # --------------------------------------------------------

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    oof = np.zeros(
        len(X_train),
        dtype=np.float32,
    )

    test_pred_folds = []

    fold_auc = []

    # --------------------------------------------------------
    # Fold loop
    # --------------------------------------------------------

    for fold, (
        tr_idx,
        va_idx,
    ) in enumerate(
        skf.split(
            X_train,
            y,
        ),
        start=1,
    ):

        print(
            "\n"
            + "-" * 90
        )

        print(
            f"{variant_name} | "
            f"Fold {fold}/{N_SPLITS}"
        )

        print(
            "-" * 90
        )

        X_fold_train_raw = (
            X_train.iloc[
                tr_idx
            ]
        )

        X_fold_valid_raw = (
            X_train.iloc[
                va_idx
            ]
        )

        y_fold_train = y[
            tr_idx
        ]

        y_fold_valid = y[
            va_idx
        ]

        # ----------------------------------------------------
        # Fold-specific preprocessing
        # ----------------------------------------------------

        (
            X_fold_train,
            X_fold_valid,
            X_fold_test,
        ) = prepare_fold_data(
            X_fold_train_raw,
            X_fold_valid_raw,
            X_test,
        )

        # ----------------------------------------------------
        # Final NaN diagnostics
        # ----------------------------------------------------

        for name, df in [
            (
                "train",
                X_fold_train,
            ),
            (
                "valid",
                X_fold_valid,
            ),
            (
                "test",
                X_fold_test,
            ),
        ]:

            numeric_columns = [
                c
                for c in df.columns
                if not (
                    pd.api.types
                    .is_categorical_dtype(
                        df[c]
                    )
                )
            ]

            for c in numeric_columns:

                arr = (
                    df[c]
                    .to_numpy()
                )

                if not np.isfinite(
                    arr
                ).all():

                    raise ValueError(
                        f"NaN/Inf remains "
                        f"in {name}.{c}"
                    )

        print(
            "Preprocessing check: OK"
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model_params = (
            REALMLP_PARAMS.copy()
        )

        model_params[
            "random_state"
        ] = SEED + fold

        model = RealMLP_TD_Classifier(
            **model_params
        )

        # ----------------------------------------------------
        # Fit
        # ----------------------------------------------------

        model.fit(
            X_fold_train,
            y_fold_train,
        )

        # ----------------------------------------------------
        # Validation prediction
        # ----------------------------------------------------

        valid_pred = (
            model
            .predict_proba(
                X_fold_valid
            )[:, 1]
        )

        # ----------------------------------------------------
        # Test prediction
        # ----------------------------------------------------

        fold_test_pred = (
            model
            .predict_proba(
                X_fold_test
            )[:, 1]
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        oof[
            va_idx
        ] = valid_pred

        test_pred_folds.append(
            fold_test_pred
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # Fold AUC
        # ----------------------------------------------------

        auc = roc_auc_score(
            y_fold_valid,
            valid_pred,
        )

        fold_auc.append(
            auc
        )

        print(
            f"Fold {fold} AUC: "
            f"{auc:.9f}"
        )

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        del model
        del X_fold_train
        del X_fold_valid
        del X_fold_test

        gc.collect()

    # ========================================================
    # TEST PREDICTION
    # ========================================================

    test_pred = (
        np.mean(
            np.vstack(
                test_pred_folds
            ),
            axis=0,
        )
        .astype(np.float32)
    )

    # ========================================================
    # STANDALONE AUC
    # ========================================================

    mlp_auc = roc_auc_score(
        y,
        oof,
    )

    # ========================================================
    # RANK CORRELATION
    # ========================================================

    lgbm_rank = (
        pd.Series(
            lgbm_oof_pred
        )
        .rank(
            method="average"
        )
        .values
    )

    mlp_rank = (
        pd.Series(
            oof
        )
        .rank(
            method="average"
        )
        .values
    )

    rank_corr = np.corrcoef(
        lgbm_rank,
        mlp_rank,
    )[0, 1]

    # ========================================================
    # NORMALIZED RANK BLEND
    # ========================================================

    lgbm_rank_norm = (
        lgbm_rank
        - lgbm_rank.min()
    ) / (
        lgbm_rank.max()
        - lgbm_rank.min()
    )

    mlp_rank_norm = (
        mlp_rank
        - mlp_rank.min()
    ) / (
        mlp_rank.max()
        - mlp_rank.min()
    )

    rank_blend = (
        LGBM_WEIGHT
        * lgbm_rank_norm
        +
        MLP_WEIGHT
        * mlp_rank_norm
    )

    rank_blend_auc = roc_auc_score(
        y,
        rank_blend,
    )

    # ========================================================
    # RAW BLEND
    # ========================================================

    raw_blend = (
        LGBM_WEIGHT
        * lgbm_oof_pred
        +
        MLP_WEIGHT
        * oof
    )

    raw_blend_auc = roc_auc_score(
        y,
        raw_blend,
    )

    # ========================================================
    # PRINT RESULT
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        f"RESULT: {variant_name}"
    )

    print(
        "=" * 90
    )

    print(
        f"MLP OOF AUC        : "
        f"{mlp_auc:.9f}"
    )

    print(
        f"Fold mean AUC      : "
        f"{np.mean(fold_auc):.9f}"
    )

    print(
        f"Fold std AUC       : "
        f"{np.std(fold_auc):.9f}"
    )

    print(
        f"LGBM-MLP rank corr : "
        f"{rank_corr:.9f}"
    )

    print(
        f"40/60 rank blend   : "
        f"{rank_blend_auc:.9f}"
    )

    print(
        f"40/60 raw blend    : "
        f"{raw_blend_auc:.9f}"
    )

    # ========================================================
    # SAVE OOF
    # ========================================================

    safe_name = (
        variant_name
        .replace(" ", "_")
    )

    oof_path = (
        OUTPUT_DIR
        / f"{safe_name}_oof.csv"
    )

    test_path = (
        OUTPUT_DIR
        / f"{safe_name}_test.csv"
    )

    oof_df = pd.DataFrame(
        {
            ID_COL:
                train[
                    ID_COL
                ].values,

            TARGET:
                y,

            "prediction":
                oof,
        }
    )

    test_pred_df = pd.DataFrame(
        {
            ID_COL:
                test[
                    ID_COL
                ].values,

            "prediction":
                test_pred,
        }
    )

    oof_df.to_csv(
        oof_path,
        index=False,
    )

    test_pred_df.to_csv(
        test_path,
        index=False,
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "variant":
            variant_name,

        "n_features":
            X_train.shape[1],

        "mlp_auc":
            mlp_auc,

        "fold_mean_auc":
            np.mean(fold_auc),

        "fold_std_auc":
            np.std(fold_auc),

        "lgbm_mlp_rank_corr":
            rank_corr,

        "rank_blend_auc":
            rank_blend_auc,

        "raw_blend_auc":
            raw_blend_auc,

        "oof_path":
            str(oof_path),

        "test_path":
            str(test_path),
    }


# ============================================================
# MAIN ABLATION
# ============================================================

banner(
    "START REALMLP V1 ABLATION"
)

print(
    "Number of variants:",
    len(VARIANTS),
)

print(
    "CV folds          :",
    N_SPLITS,
)

print(
    "LGBM weight       :",
    LGBM_WEIGHT,
)

print(
    "RealMLP weight    :",
    MLP_WEIGHT,
)


results = []


for variant_number, (
    variant_name,
    selected_families,
) in enumerate(
    VARIANTS.items(),
    start=1,
):

    print(
        "\n"
        + "#"
        * 100
    )

    print(
        f"VARIANT "
        f"{variant_number}/"
        f"{len(VARIANTS)}: "
        f"{variant_name}"
    )

    print(
        "Families:",
        selected_families,
    )

    print(
        "#"
        * 100
    )

    X_train_variant = (
        select_variant_features(
            all_train_features,
            selected_families,
        )
    )

    X_test_variant = (
        select_variant_features(
            all_test_features,
            selected_families,
        )
    )

    result = run_realmlp_variant(
        variant_name=
            variant_name,

        X_train=
            X_train_variant,

        X_test=
            X_test_variant,

        y=y,

        lgbm_oof_pred=
            lgbm_oof_pred,
    )

    results.append(
        result
    )

    # --------------------------------------------------------
    # Save progress
    # --------------------------------------------------------

    progress_df = (
        pd.DataFrame(
            results
        )
        .sort_values(
            "rank_blend_auc",
            ascending=False,
        )
    )

    progress_df.to_csv(
        OUTPUT_DIR
        / "ablation_progress.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Print ranking
    # --------------------------------------------------------

    print(
        "\nCURRENT RANKING"
    )

    ranking_cols = [
        "variant",
        "n_features",
        "mlp_auc",
        "lgbm_mlp_rank_corr",
        "rank_blend_auc",
        "raw_blend_auc",
    ]

    print(
        progress_df[
            ranking_cols
        ]
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Free memory
    # --------------------------------------------------------

    del X_train_variant
    del X_test_variant

    gc.collect()


# ============================================================
# FINAL SUMMARY
# ============================================================

banner(
    "FINAL REALMLP V1 ABLATION RESULTS"
)

results_df = (
    pd.DataFrame(
        results
    )
    .sort_values(
        "rank_blend_auc",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


summary_path = (
    OUTPUT_DIR
    / "realmlp_v1_ablation_summary.csv"
)

results_df.to_csv(
    summary_path,
    index=False,
)


print(
    results_df[
        [
            "variant",
            "n_features",
            "mlp_auc",
            "fold_mean_auc",
            "fold_std_auc",
            "lgbm_mlp_rank_corr",
            "rank_blend_auc",
            "raw_blend_auc",
        ]
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# BEST VARIANT
# ============================================================

best = results_df.iloc[0]


banner(
    "BEST ABLATION"
)

print(
    f"Best variant        : "
    f"{best['variant']}"
)

print(
    f"Number of features  : "
    f"{int(best['n_features'])}"
)

print(
    f"MLP OOF AUC         : "
    f"{best['mlp_auc']:.9f}"
)

print(
    f"Fold mean AUC       : "
    f"{best['fold_mean_auc']:.9f}"
)

print(
    f"Fold std AUC        : "
    f"{best['fold_std_auc']:.9f}"
)

print(
    f"LGBM-MLP rank corr  : "
    f"{best['lgbm_mlp_rank_corr']:.9f}"
)

print(
    f"40/60 rank blend    : "
    f"{best['rank_blend_auc']:.9f}"
)

print(
    f"40/60 raw blend     : "
    f"{best['raw_blend_auc']:.9f}"
)

print(
    "\nSaved summary:"
)

print(
    summary_path
)

print(
    "\nExperiment finished."
)
