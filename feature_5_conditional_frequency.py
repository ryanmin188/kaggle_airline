import os
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

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

OUTPUT_DIR = r"D:\kaggle\airline\conditional_te_experiment"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

OUTER_FOLDS = 5
INNER_TE_FOLDS = 5

# Your current best:
# Proper Target Encoding + LightGBM
PROPER_TE_BENCHMARK = 0.960298


# ============================================================
# LIGHTGBM PARAMETERS
# ============================================================

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


# ============================================================
# LOAD
# ============================================================

print("=" * 90)
print("CONDITIONAL TARGET ENCODING + FREQUENCY + LIGHTGBM")
print("=" * 90)

print("LightGBM:", lgb.__version__)

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

TARGET = "satisfaction"
ID_COL = "id"

y = train[TARGET].astype(np.int8).to_numpy()

print(f"\nTrain shape : {train.shape}")
print(f"Test shape  : {test.shape}")
print(f"Target mean : {y.mean():.6f}")


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
    if c not in [ID_COL, TARGET]
]

print(
    f"\nOriginal features: "
    f"{len(ORIGINAL_FEATURES)}"
)


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

    # --------------------------------------------------------
    # Traveler segment
    # --------------------------------------------------------

    X["traveler_segment"] = (
        df["Type of Travel"].astype(str)
        + "|"
        + df["Customer Type"].astype(str)
        + "|"
        + df["Class"].astype(str)
    )

    # --------------------------------------------------------
    # Arrival-delay missing indicator
    # --------------------------------------------------------

    X["arrival_missing"] = (
        df["Arrival Delay in Minutes"]
        .isna()
        .astype("int8")
    )

    return X


X_base = build_base(train)
X_test_base = build_base(test)


# ============================================================
# PREPARE BASE MODEL FEATURES
# ============================================================

MODEL_CATS = CATS + [
    "traveler_segment"
]


# Numeric columns
for c in NUMERIC:

    median_value = train[c].median()

    X_base[c] = (
        X_base[c]
        .fillna(median_value)
        .astype("float32")
    )

    X_test_base[c] = (
        X_test_base[c]
        .fillna(median_value)
        .astype("float32")
    )


# Categorical columns
for c in MODEL_CATS:

    combined = pd.concat(
        [
            X_base[c],
            X_test_base[c],
        ],
        ignore_index=True,
    ).astype(str)

    categories = pd.Index(
        combined.unique()
    )

    dtype = pd.api.types.CategoricalDtype(
        categories=categories
    )

    X_base[c] = X_base[c].astype(dtype)
    X_test_base[c] = X_test_base[c].astype(dtype)


print(
    f"Base model feature count: "
    f"{X_base.shape[1]}"
)


# ============================================================
# BASE TARGET ENCODING INPUT
#
# ALL 21 original predictors are TE encoded.
#
# Numerical variables are intentionally converted to strings,
# meaning exact numeric values are treated as categories.
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


print(
    f"Base TE features: "
    f"{Z_BASE.shape[1]}"
)


# ============================================================
# CONDITIONAL TE
#
# 13 service × traveler_segment
# +
# 5 selected service × service × type/customer
# = 18
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


Z_COND = build_conditional(train)
U_COND = build_conditional(test)

print(
    f"Conditional TE features: "
    f"{Z_COND.shape[1]}"
)


# ============================================================
# NUMERIC GROUP TE
#
# Age × Type of Travel
# Age × Customer Type
# Age × Class
#
# Flight Distance × Type of Travel
# Flight Distance × Customer Type
# Flight Distance × Class
#
# = 6
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

print(
    f"Numeric-group TE features: "
    f"{Z_NUM_GROUP.shape[1]}"
)


# ============================================================
# FREQUENCY ENCODING
#
# The previous version had a bug:
# build_frequency_features() received a dataframe that had
# already been reduced to:
#
#   4 original numeric columns
#   +
#   6 numeric-group keys
#
# but then tried to access:
#
#   Type of Travel
#   Customer Type
#   Class
#
# which no longer existed in that dataframe.
#
# This corrected function simply encodes every column passed
# to it.
#
# 4 numeric exact values
# +
# 6 numeric-group values
# =
# 10 frequency features
#
# Frequency does not use y -> no target leakage.
# ============================================================

def build_frequency_features(
    train_df,
    val_df,
    test_df,
):

    results_train = pd.DataFrame(
        index=train_df.index
    )

    results_val = pd.DataFrame(
        index=val_df.index
    )

    results_test = pd.DataFrame(
        index=test_df.index
    )

    for c in train_df.columns:

        # ----------------------------------------------------
        # Frequency mapping estimated ONLY from outer training
        # ----------------------------------------------------

        freq_map = (
            train_df[c]
            .astype(str)
            .value_counts(
                normalize=True,
                dropna=False,
            )
        )

        name = (
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
            .fillna(0.0)
            .to_numpy()
        )

        va_freq = (
            val_df[c]
            .astype(str)
            .map(freq_map)
            .fillna(0.0)
            .to_numpy()
        )

        te_freq = (
            test_df[c]
            .astype(str)
            .map(freq_map)
            .fillna(0.0)
            .to_numpy()
        )

        # ----------------------------------------------------
        # Log frequency
        # ----------------------------------------------------

        results_train[name] = (
            np.log1p(
                tr_freq * 1_000_000
            ).astype("float32")
        )

        results_val[name] = (
            np.log1p(
                va_freq * 1_000_000
            ).astype("float32")
        )

        results_test[name] = (
            np.log1p(
                te_freq * 1_000_000
            ).astype("float32")
        )

    return (
        results_train,
        results_val,
        results_test,
    )


# ============================================================
# OUTER CV
# ============================================================

skf = StratifiedKFold(
    n_splits=OUTER_FOLDS,
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
best_iterations = []


# ============================================================
# OUTER LOOP
# ============================================================

for fold, (tr_idx, va_idx) in enumerate(
    skf.split(
        np.zeros(len(y)),
        y,
    ),
    1,
):

    print("\n" + "=" * 90)
    print(
        f"OUTER FOLD {fold}/{OUTER_FOLDS}"
    )
    print("=" * 90)

    # ========================================================
    # BASE MODEL FEATURES
    # ========================================================

    X_tr = X_base.iloc[
        tr_idx
    ].copy()

    X_va = X_base.iloc[
        va_idx
    ].copy()

    X_te = X_test_base.copy()


    # ========================================================
    # 1. BASE TARGET ENCODING
    # ========================================================

    print(
        "\n[1/4] Base TargetEncoder..."
    )

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
    # 2. CONDITIONAL TARGET ENCODING
    # ========================================================

    print(
        "[2/4] Conditional TargetEncoder..."
    )

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

    for j, c in enumerate(
        Z_COND.columns
    ):

        name = (
            "cond_te_"
            + str(j)
        )

        X_tr[name] = cond_tr[:, j]
        X_va[name] = cond_va[:, j]
        X_te[name] = cond_te[:, j]


    # ========================================================
    # 3. NUMERIC-GROUP TE
    # ========================================================

    print(
        "[3/4] Numeric-group TargetEncoder..."
    )

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

    for j, c in enumerate(
        Z_NUM_GROUP.columns
    ):

        name = (
            "ng_te_"
            + str(j)
        )

        X_tr[name] = num_tr[:, j]
        X_va[name] = num_va[:, j]
        X_te[name] = num_te[:, j]


    # ========================================================
    # 4. FREQUENCY ENCODING
    # ========================================================

    print(
        "[4/4] Frequency encoding..."
    )

    # --------------------------------------------------------
    # Outer-train frequency inputs
    #
    # 4 exact numeric variables
    # +
    # 6 numeric-group combinations
    # --------------------------------------------------------

    freq_tr_base = pd.concat(
        [
            Z_BASE.iloc[tr_idx][NUMERIC]
            .reset_index(drop=True),

            Z_NUM_GROUP.iloc[tr_idx]
            .reset_index(drop=True),
        ],
        axis=1,
    )

    freq_va_base = pd.concat(
        [
            Z_BASE.iloc[va_idx][NUMERIC]
            .reset_index(drop=True),

            Z_NUM_GROUP.iloc[va_idx]
            .reset_index(drop=True),
        ],
        axis=1,
    )

    freq_te_base = pd.concat(
        [
            U_BASE[NUMERIC]
            .reset_index(drop=True),

            U_NUM_GROUP
            .reset_index(drop=True),
        ],
        axis=1,
    )

    freq_tr, freq_va, freq_te = (
        build_frequency_features(
            freq_tr_base,
            freq_va_base,
            freq_te_base,
        )
    )

    # Align indexes with X matrices
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
    # CLEAN FEATURES
    # ========================================================

    X_tr = X_tr.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_va = X_va.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_te = X_te.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Fill numerical NaNs created by preprocessing
    for df in [
        X_tr,
        X_va,
        X_te,
    ]:

        numeric_cols = df.select_dtypes(
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

            if df[c].isna().any():

                df[c] = df[c].fillna(0)


    # --------------------------------------------------------
    # Remove duplicated columns if any
    # --------------------------------------------------------

    keep_columns = (
        ~X_tr.columns.duplicated()
    )

    X_tr = X_tr.loc[
        :,
        keep_columns
    ]

    X_va = X_va[
        X_tr.columns
    ]

    X_te = X_te[
        X_tr.columns
    ]


    print(
        f"\nFinal feature count = "
        f"{X_tr.shape[1]}"
    )


    # ========================================================
    # LIGHTGBM
    # ========================================================

    print(
        "Training LightGBM..."
    )

    model = lgb.LGBMClassifier(
        **LGB_PARAMS
    )

    model.fit(
        X_tr,
        y[tr_idx],
        eval_set=[
            (
                X_va,
                y[va_idx],
            )
        ],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(
                150,
                verbose=False,
            )
        ],
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    val_pred = model.predict_proba(
        X_va
    )[:, 1]

    fold_test_pred = model.predict_proba(
        X_te
    )[:, 1]

    oof[va_idx] = val_pred

    test_pred += (
        fold_test_pred
        / OUTER_FOLDS
    )


    # ========================================================
    # FOLD SCORE
    # ========================================================

    fold_auc = roc_auc_score(
        y[va_idx],
        val_pred,
    )

    fold_scores.append(
        fold_auc
    )

    best_iter = (
        model.best_iteration_
        if model.best_iteration_
        else LGB_PARAMS["n_estimators"]
    )

    best_iterations.append(
        best_iter
    )

    print(
        f"\nFold {fold} AUC = "
        f"{fold_auc:.6f}"
    )

    print(
        f"Best iteration = "
        f"{best_iter}"
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
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("FINAL CONDITIONAL TE RESULT")
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
    f"\nProper-TE benchmark = "
    f"{PROPER_TE_BENCHMARK:.6f}"
)

print(
    f"Delta vs Proper-TE = "
    f"{oof_auc - PROPER_TE_BENCHMARK:+.6f}"
)

print(
    "\nBest iterations:"
)

print(
    best_iterations
)


# ============================================================
# SAVE OOF
# ============================================================

oof_path = os.path.join(
    OUTPUT_DIR,
    "conditional_te_lgbm_oof.csv",
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
# SAVE TEST PREDICTION
# ============================================================

test_pred_path = os.path.join(
    OUTPUT_DIR,
    "conditional_te_lgbm_test.csv",
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
# Continuous probabilities ONLY.
# No thresholding.
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
    submission_target: test_pred,
})

submission_path = os.path.join(
    OUTPUT_DIR,
    "submission_11_conditional_te.csv",
)

submission.to_csv(
    submission_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# TEST PREDICTION STATS
# ============================================================

print("\n" + "=" * 90)
print("OUTPUT")
print("=" * 90)

print(
    f"Test prediction min  = "
    f"{test_pred.min():.6f}"
)

print(
    f"Test prediction max  = "
    f"{test_pred.max():.6f}"
)

print(
    f"Test prediction mean = "
    f"{test_pred.mean():.6f}"
)

print(
    f"Test prediction std  = "
    f"{test_pred.std():.6f}"
)

print(
    f"\nOOF file:\n{oof_path}"
)

print(
    f"\nTest prediction file:\n"
    f"{test_pred_path}"
)

print(
    f"\nSubmission:\n"
    f"{submission_path}"
)

print("\nDone.")