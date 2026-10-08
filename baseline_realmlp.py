import os
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# ============================================================
# TRY IMPORTS
# ============================================================

try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch not installed.\n"
        "Run:\n"
        "pip install torch"
    )

try:
    from pytabkit import RealMLP_TD_Classifier
except ImportError:
    raise ImportError(
        "PyTabKit not installed.\n"
        "Run:\n"
        "pip install -U pytabkit"
    )


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = r"D:\kaggle\airline\data\train.csv"
TEST_PATH = r"D:\kaggle\airline\data\test.csv"
SAMPLE_PATH = r"D:\kaggle\airline\data\sample_submission.csv"

OUTPUT_DIR = (
    r"D:\kaggle\airline\realmlp_experiment"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

# Deliberately 3-fold for baseline
N_FOLDS = 3

INNER_TE_FOLDS = 5

# Current best benchmark
CURRENT_BEST_OOF = 0.960664
CURRENT_BEST_LB = 0.96030

# Baseline neural network settings
N_EPOCHS = 128
BATCH_SIZE = 2048


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = "cuda"

    print(
        f"CUDA available: YES"
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

else:

    DEVICE = "cpu"

    print(
        "CUDA available: NO"
    )

    print(
        "Using CPU."
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 90)
print("REALMLP BASELINE - ENRICHED 78 FEATURES")
print("=" * 90)

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
# BASE FEATURES
# ============================================================

def build_base(df):

    X = df.drop(
        columns=[
            ID_COL,
            TARGET,
        ],
        errors="ignore",
    ).copy()

    # Existing useful segment
    X["traveler_segment"] = (
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
# PREPROCESS BASE FEATURES
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

    median_value = (
        train[c]
        .median()
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
        pd.api.types.CategoricalDtype(
            categories=categories
        )
    )

    X_BASE[c] = (
        X_BASE[c]
        .astype(dtype)
    )

    X_TEST_BASE[c] = (
        X_TEST_BASE[c]
        .astype(dtype)
    )


# ============================================================
# BASE TE INPUT
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
# CONDITIONAL FEATURES
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
    # Service × traveler segment
    # --------------------------------------------------------

    for c in SERVICES:

        Z[c + "__segment"] = (
            df[c].astype(str)
            + "|"
            + traveler_segment
        )

    # --------------------------------------------------------
    # Selected service interactions
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
                df[numeric_col]
                .astype(str)
                + "|"
                + df[group_col]
                .astype(str)
            )

    return Z


Z_NUM_GROUP = (
    build_numeric_groups(train)
)

U_NUM_GROUP = (
    build_numeric_groups(test)
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
# BUILD 78 FEATURES
# ============================================================

def build_enriched_fold(
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

    tr_te = enc_base.fit_transform(
        Z_BASE.iloc[tr_idx],
        y[tr_idx],
    ).astype("float32")

    va_te = enc_base.transform(
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

        X_tr[name] = tr_te[:, j]
        X_va[name] = va_te[:, j]
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

    tr_cond = enc_cond.fit_transform(
        Z_COND.iloc[tr_idx],
        y[tr_idx],
    ).astype("float32")

    va_cond = enc_cond.transform(
        Z_COND.iloc[va_idx]
    ).astype("float32")

    te_cond = enc_cond.transform(
        U_COND
    ).astype("float32")


    for j in range(
        Z_COND.shape[1]
    ):

        name = (
            "cond_te_"
            + str(j)
        )

        X_tr[name] = tr_cond[:, j]
        X_va[name] = va_cond[:, j]
        X_te[name] = te_cond[:, j]


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

    tr_num = enc_num.fit_transform(
        Z_NUM_GROUP.iloc[tr_idx],
        y[tr_idx],
    ).astype("float32")

    va_num = enc_num.transform(
        Z_NUM_GROUP.iloc[va_idx]
    ).astype("float32")

    te_num = enc_num.transform(
        U_NUM_GROUP
    ).astype("float32")


    for j in range(
        Z_NUM_GROUP.shape[1]
    ):

        name = (
            "ng_te_"
            + str(j)
        )

        X_tr[name] = tr_num[:, j]
        X_va[name] = va_num[:, j]
        X_te[name] = te_num[:, j]


    # ========================================================
    # 4. FREQUENCY
    # ========================================================

    freq_tr_input = pd.concat(
        [
            Z_BASE.iloc[
                tr_idx
            ][NUMERIC]
            .reset_index(drop=True),

            Z_NUM_GROUP.iloc[
                tr_idx
            ].reset_index(drop=True),
        ],
        axis=1,
    )

    freq_va_input = pd.concat(
        [
            Z_BASE.iloc[
                va_idx
            ][NUMERIC]
            .reset_index(drop=True),

            Z_NUM_GROUP.iloc[
                va_idx
            ].reset_index(drop=True),
        ],
        axis=1,
    )

    freq_te_input = pd.concat(
        [
            U_BASE[
                NUMERIC
            ].reset_index(drop=True),

            U_NUM_GROUP
            .reset_index(drop=True),
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
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
            inplace=True,
        )


    numeric_cols = (
        X_tr.select_dtypes(
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
    # Final consistency check
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


    if X_tr.shape[1] != 78:

        raise RuntimeError(
            f"Expected 78 features, "
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
    """
    RealMLP works with pandas DataFrames and can detect
    categorical columns automatically.

    Numerical missing values are not allowed, so all numeric
    NaNs are filled using outer-training medians.
    """

    X_tr = X_tr.copy()
    X_va = X_va.copy()
    X_te = X_te.copy()


    categorical_cols = []


    # --------------------------------------------------------
    # Detect categoricals
    # --------------------------------------------------------

    for c in X_tr.columns:

        dtype_string = str(
            X_tr[c].dtype
        )

        if (
            X_tr[c].dtype == "object"
            or dtype_string.startswith(
                "category"
            )
        ):

            categorical_cols.append(c)


    # --------------------------------------------------------
    # Convert categorical variables
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
            pd.api.types.CategoricalDtype(
                categories=categories
            )
        )

        X_tr[c] = (
            X_tr[c]
            .astype(dtype)
        )

        X_va[c] = (
            X_va[c]
            .astype(dtype)
        )

        X_te[c] = (
            X_te[c]
            .astype(dtype)
        )


    # --------------------------------------------------------
    # Numeric features
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
    # Fill numeric missing
    # --------------------------------------------------------

    for c in X_tr.columns:

        if c in categorical_cols:
            continue

        median_value = (
            X_tr[c]
            .median()
        )

        if pd.isna(median_value):
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
    # Ensure identical columns
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
# OUTER CV
# ============================================================

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


oof = np.zeros(
    len(train),
    dtype=np.float32,
)

test_pred = np.zeros(
    len(test),
    dtype=np.float64,
)

fold_scores = []


# ============================================================
# TRAIN
# ============================================================

for fold, (
    tr_idx,
    va_idx,
) in enumerate(
    skf.split(
        np.zeros(len(y)),
        y,
    ),
    1,
):

    print("\n" + "=" * 90)
    print(
        f"REALMLP FOLD "
        f"{fold}/{N_FOLDS}"
    )
    print("=" * 90)


    # ========================================================
    # Build 78 features
    # ========================================================

    print(
        "\nBuilding enriched features..."
    )

    (
        X_tr,
        X_va,
        X_te,
    ) = build_enriched_fold(
        tr_idx,
        va_idx,
    )


    print(
        f"Enriched feature count = "
        f"{X_tr.shape[1]}"
    )


    # ========================================================
    # Prepare for RealMLP
    # ========================================================

    (
        X_tr,
        X_va,
        X_te,
        cat_cols,
    ) = prepare_realmlp(
        X_tr,
        X_va,
        X_te,
    )


    print(
        f"Categorical columns = "
        f"{len(cat_cols)}"
    )


    # ========================================================
    # RealMLP
    # ========================================================

    print(
        "\nTraining RealMLP..."
    )

    model = RealMLP_TD_Classifier(
        device=DEVICE,

        random_state=(
            RANDOM_STATE
            + fold
        ),

        # We are already doing our own outer CV
        n_cv=1,
        n_refit=0,

        n_epochs=N_EPOCHS,

        batch_size=BATCH_SIZE,

        hidden_sizes=[
            256,
            256,
            256,
        ],

        lr=0.04,

        val_metric_name="cross_entropy",

        use_ls=False,

        verbosity=1,
    )


    # ========================================================
    # FIT
    # ========================================================

    model.fit(
        X_tr,
        y[tr_idx],

        X_va,
        y[va_idx],

        cat_col_names=cat_cols,
    )


    # ========================================================
    # VALIDATION PREDICTION
    # ========================================================

    val_pred = (
        model
        .predict_proba(
            X_va
        )[:, 1]
    )


    # ========================================================
    # TEST PREDICTION
    # ========================================================

    fold_test_pred = (
        model
        .predict_proba(
            X_te
        )[:, 1]
    )


    oof[va_idx] = (
        val_pred
    )

    test_pred += (
        fold_test_pred
        / N_FOLDS
    )


    # ========================================================
    # SCORE
    # ========================================================

    fold_auc = roc_auc_score(
        y[va_idx],
        val_pred,
    )

    fold_scores.append(
        fold_auc
    )


    print(
        f"\nFold {fold} AUC = "
        f"{fold_auc:.6f}"
    )


# ============================================================
# FINAL OOF
# ============================================================

oof_auc = roc_auc_score(
    y,
    oof,
)

fold_mean = np.mean(
    fold_scores
)

fold_std = np.std(
    fold_scores
)


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 90)
print("REALMLP BASELINE RESULT")
print("=" * 90)

for i, score in enumerate(
    fold_scores,
    1,
):

    print(
        f"Fold {i}: "
        f"{score:.6f}"
    )


print(
    f"\nFold mean AUC = "
    f"{fold_mean:.6f}"
)

print(
    f"Fold std      = "
    f"{fold_std:.6f}"
)

print(
    f"OOF AUC       = "
    f"{oof_auc:.6f}"
)

print(
    f"\nCurrent LGBM OOF = "
    f"{CURRENT_BEST_OOF:.6f}"
)

print(
    f"Delta vs current LGBM = "
    f"{oof_auc - CURRENT_BEST_OOF:+.6f}"
)

print(
    f"\nCurrent best LB = "
    f"{CURRENT_BEST_LB:.5f}"
)


# ============================================================
# SAVE OOF
# ============================================================

oof_path = os.path.join(
    OUTPUT_DIR,
    "realmlp_enriched_oof.csv",
)

pd.DataFrame({
    "id": train[ID_COL],
    "satisfaction": y,
    "prediction": oof,
}).to_csv(
    oof_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SAVE TEST
# ============================================================

test_pred_path = os.path.join(
    OUTPUT_DIR,
    "realmlp_enriched_test.csv",
)

pd.DataFrame({
    "id": test[ID_COL],
    "prediction": test_pred,
}).to_csv(
    test_pred_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SUBMISSION
#
# IMPORTANT:
# Continuous probabilities.
# NO threshold.
# ============================================================

sample = pd.read_csv(
    SAMPLE_PATH
)

target_col = [
    c
    for c in sample.columns
    if c != ID_COL
][0]


submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    target_col: test_pred,
})


submission_path = os.path.join(
    OUTPUT_DIR,
    "submission_15_realmlp_enriched.csv",
)


submission.to_csv(
    submission_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# STATS
# ============================================================

print("\n" + "=" * 90)
print("TEST PREDICTION")
print("=" * 90)

print(
    f"min  = "
    f"{test_pred.min():.6f}"
)

print(
    f"max  = "
    f"{test_pred.max():.6f}"
)

print(
    f"mean = "
    f"{test_pred.mean():.6f}"
)

print(
    f"std  = "
    f"{test_pred.std():.6f}"
)


print("\nSaved:")

print(
    oof_path
)

print(
    test_pred_path
)

print(
    submission_path
)

print("\nDone.")