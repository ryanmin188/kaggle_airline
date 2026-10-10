# ============================================================
# TABM AIRLINE SATISFACTION EXPERIMENT
#
# Goal:
#   Train an independent TabM model using competition data only.
#   Generate 5-fold OOF predictions.
#   Evaluate rank fusion with the current Champion.
#
# Current Champion:
#   Conditional-TE LGBM 40%
#   Lean RealMLP V1     60%
#   OOF AUC = 0.961278687
#   Public LB = 0.96097
#
# Constraints:
#   - No original 129K dataset
#   - No external predictions/features
#   - No V2
#   - No Route Profile
#   - No model weight dense search
#
# Preprocessing:
#   - 17 numeric features
#   - 4 categorical features
#   - Fold-specific median imputation
#   - Fold-specific categorical mappings
#   - Piecewise Linear Embeddings for numerical features
#
# Validation:
#   - 5-fold Stratified OOF
#   - Inner validation selects the epoch
#   - Refit on the complete outer training fold
#
# Output:
#   D:\kaggle\airline\tabm_experiment\
# ============================================================

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

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)

from sklearn.metrics import roc_auc_score

from scipy.stats import rankdata

from tabm import TabM

from rtdl_num_embeddings import (
    PiecewiseLinearEmbeddings,
    compute_bins,
)


warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
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


# Current Conditional-TE LGBM
CURRENT_LGBM_OOF_PATH = (
    r"D:\kaggle\airline\conditional_te_experiment"
    r"\conditional_te_lgbm_oof.csv"
)

CURRENT_LGBM_TEST_PATH = (
    r"D:\kaggle\airline\conditional_te_experiment"
    r"\conditional_te_lgbm_test.csv"
)


# Current Lean RealMLP V1
V1_OOF_PATH = (
    r"D:\kaggle\airline\lean_realmlp_blend"
    r"\lean_realmlp_oof.csv"
)

V1_TEST_PATH = (
    r"D:\kaggle\airline\lean_realmlp_blend"
    r"\lean_realmlp_test.csv"
)


# Output directory
OUTPUT_DIR = (
    r"D:\kaggle\airline\tabm_experiment"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# 2. SETTINGS
# ============================================================

ID_COL = "id"
TARGET = "satisfaction"

SEED = 42
N_FOLDS = 5

# Inner validation
INNER_VALID_SIZE = 0.10

# Training
MAX_EPOCHS = 25
MIN_EPOCHS = 5
PATIENCE = 4

LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0003

# TabM architecture
TABM_K = 16
TABM_BLOCKS = 2
TABM_WIDTH = 256
TABM_DROPOUT = 0.10

# PLE numerical embeddings
PLE_N_BINS = 32
PLE_D_EMBEDDING = 12

# For initial experimentation, use a moderate batch size.
BATCH_SIZE = 2048

# Limit the data used to calculate bin boundaries.
# The boundaries are computed from each fitting fold only.
MAX_BIN_SAMPLE = 100_000

# Known benchmark
KNOWN_CHAMPION_OOF = 0.961278687

# Conservative TabM blend weights.
# These weights are fractions of the final blend assigned to TabM.
TABM_BLEND_WEIGHTS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
]


# ============================================================
# 3. FEATURE DEFINITIONS
# ============================================================

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

CATEGORICAL_COLS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

ALL_FEATURES = (
    NUMERIC_COLS
    + CATEGORICAL_COLS
)


# ============================================================
# 4. DEVICE AND RANDOM SEED
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


print("=" * 100)
print("TABM AIRLINE EXPERIMENT")
print("=" * 100)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

print("Folds:", N_FOLDS)
print("Max epochs:", MAX_EPOCHS)
print("Batch size:", BATCH_SIZE)
print("TabM k:", TABM_K)
print("TabM width:", TABM_WIDTH)
print("TabM blocks:", TABM_BLOCKS)


# ============================================================
# 5. GENERAL VALIDATION HELPERS
# ============================================================

def assert_file_exists(path, name):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )


def check_same_ids(
    reference_ids,
    candidate_ids,
    name,
):

    reference_ids = set(
        pd.to_numeric(
            reference_ids,
            errors="raise",
        )
    )

    candidate_ids = set(
        pd.to_numeric(
            candidate_ids,
            errors="raise",
        )
    )

    if reference_ids != candidate_ids:

        missing = (
            reference_ids
            - candidate_ids
        )

        extra = (
            candidate_ids
            - reference_ids
        )

        raise ValueError(
            f"{name}: ID mismatch.\n"
            f"Missing IDs: {len(missing)}\n"
            f"Extra IDs: {len(extra)}"
        )


def normalized_rank(values):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return (
        rankdata(
            values,
            method="average",
        )
        / len(values)
    )


# ============================================================
# 6. LOAD DATA
# ============================================================

print("\n" + "=" * 100)
print("LOADING DATA")
print("=" * 100)

assert_file_exists(
    TRAIN_PATH,
    "train.csv",
)

assert_file_exists(
    TEST_PATH,
    "test.csv",
)

assert_file_exists(
    SAMPLE_PATH,
    "sample_submission.csv",
)


train = pd.read_csv(
    TRAIN_PATH,
)

test = pd.read_csv(
    TEST_PATH,
)

sample = pd.read_csv(
    SAMPLE_PATH,
)


for col in (
    [ID_COL, TARGET]
    + ALL_FEATURES
):

    if col not in train.columns:
        raise ValueError(
            f"Missing training column: {col}"
        )


for col in (
    [ID_COL]
    + ALL_FEATURES
):

    if col not in test.columns:
        raise ValueError(
            f"Missing test column: {col}"
        )


if ID_COL not in sample.columns:
    raise ValueError(
        "sample_submission missing id."
    )


train_ids = pd.to_numeric(
    train[ID_COL],
    errors="raise",
).to_numpy()


test_ids = pd.to_numeric(
    test[ID_COL],
    errors="raise",
).to_numpy()


y = (
    train[TARGET]
    .astype(np.int8)
    .to_numpy()
)


if len(np.unique(y)) != 2:
    raise ValueError(
        "Expected a binary classification target."
    )


if len(np.unique(train_ids)) != len(train_ids):
    raise ValueError(
        "Duplicate train IDs detected."
    )


if len(np.unique(test_ids)) != len(test_ids):
    raise ValueError(
        "Duplicate test IDs detected."
    )


check_same_ids(
    test_ids,
    sample[ID_COL],
    "test vs sample_submission",
)


print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Target mean:", f"{y.mean():.6f}")
print("Numeric features:", len(NUMERIC_COLS))
print("Categorical features:", len(CATEGORICAL_COLS))


# ============================================================
# 7. FOLD-SPECIFIC PREPROCESSING
# ============================================================

def clean_category(series):

    return (
        series
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
    )


def fit_preprocessor(
    fit_df,
):
    """
    Fit preprocessing on the fitting partition only.

    Numeric:
        Median imputation.

    Categorical:
        Fit mappings on fitting partition.
        Code 0 is reserved for unseen categories.
        Known categories start from code 1.
    """

    medians = {}

    for col in NUMERIC_COLS:

        values = pd.to_numeric(
            fit_df[col],
            errors="coerce",
        )

        values = values.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        median = values.median()

        if not np.isfinite(median):
            median = 0.0

        medians[col] = float(median)


    category_maps = {}

    for col in CATEGORICAL_COLS:

        values = clean_category(
            fit_df[col]
        )

        levels = sorted(
            values.unique().tolist()
        )

        category_maps[col] = {
            value: index + 1
            for index, value in enumerate(levels)
        }


    return {
        "medians": medians,
        "category_maps": category_maps,
    }


def transform_features(
    df,
    preprocessor,
):
    """
    Return:
        X_num: float32, shape (n, 17)
        X_cat: int64, shape (n, 4)
    """

    n = len(df)

    X_num = np.zeros(
        (n, len(NUMERIC_COLS)),
        dtype=np.float32,
    )

    X_cat = np.zeros(
        (n, len(CATEGORICAL_COLS)),
        dtype=np.int64,
    )


    # Numeric
    for j, col in enumerate(NUMERIC_COLS):

        values = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        values = values.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        values = values.fillna(
            preprocessor["medians"][col]
        )

        X_num[:, j] = values.to_numpy(
            dtype=np.float32
        )


    # Categorical
    for j, col in enumerate(CATEGORICAL_COLS):

        values = clean_category(
            df[col]
        )

        mapper = preprocessor[
            "category_maps"
        ][col]

        codes = (
            values
            .map(mapper)
            .fillna(0)
            .astype(np.int64)
            .to_numpy()
        )

        X_cat[:, j] = codes


    if not np.isfinite(X_num).all():
        raise ValueError(
            "Non-finite numerical features after preprocessing."
        )


    if X_cat.min() < 0:
        raise ValueError(
            "Categorical codes must be non-negative."
        )


    return X_num, X_cat


def get_cat_cardinalities(preprocessor):

    return [
        len(
            preprocessor["category_maps"][col]
        ) + 1
        for col in CATEGORICAL_COLS
    ]


# ============================================================
# 8. CREATE TABM MODEL
# ============================================================

def create_tabm_model(
    X_num_fit,
    cat_cardinalities,
    seed,
):
    """
    Build TabM with Piecewise Linear Embeddings.

    PLE bin boundaries are estimated only from the
    current fitting partition.
    """

    set_seed(seed)

    n_fit = len(X_num_fit)

    sample_size = min(
        MAX_BIN_SAMPLE,
        n_fit,
    )

    rng = np.random.default_rng(
        seed
    )

    sample_indices = rng.choice(
        n_fit,
        size=sample_size,
        replace=False,
    )

    bin_source = torch.from_numpy(
        X_num_fit[sample_indices]
    ).float()


    bins = compute_bins(
        bin_source,
        n_bins=PLE_N_BINS,
    )


    num_embeddings = (
        PiecewiseLinearEmbeddings(
            bins,
            d_embedding=PLE_D_EMBEDDING,
            activation=False,
            version="B",
        )
    )


    model = TabM.make(
        n_num_features=len(NUMERIC_COLS),
        cat_cardinalities=cat_cardinalities,
        num_embeddings=num_embeddings,
        d_out=1,
        k=TABM_K,
        n_blocks=TABM_BLOCKS,
        d_block=TABM_WIDTH,
        dropout=TABM_DROPOUT,
    )


    model = model.to(DEVICE)

    return model


# ============================================================
# 9. PREDICTION
# ============================================================

@torch.no_grad()
def predict_probability(
    model,
    X_num,
    X_cat,
    batch_size=BATCH_SIZE,
):
    """
    Return the mean probability across TabM's k members.
    """

    model.eval()

    outputs = []

    n = len(X_num)

    for start in range(
        0,
        n,
        batch_size,
    ):

        end = min(
            start + batch_size,
            n,
        )

        xb_num = torch.from_numpy(
            X_num[start:end]
        ).to(DEVICE)

        xb_cat = torch.from_numpy(
            X_cat[start:end]
        ).to(DEVICE)


        logits = model(
            xb_num,
            xb_cat,
        ).squeeze(-1)


        # Shape before mean: (batch, k)
        # Average probabilities, not logits.
        probs = torch.sigmoid(
            logits
        ).mean(
            dim=1
        )


        outputs.append(
            probs.detach()
            .cpu()
            .numpy()
        )


    return np.concatenate(
        outputs
    )


# ============================================================
# 10. ONE EPOCH OF TRAINING
# ============================================================

def train_one_epoch(
    model,
    optimizer,
    X_num,
    X_cat,
    y_array,
    batch_size,
):

    model.train()

    n = len(y_array)

    permutation = np.random.permutation(
        n
    )

    total_loss = 0.0
    total_count = 0


    for start in range(
        0,
        n,
        batch_size,
    ):

        indices = permutation[
            start:start + batch_size
        ]

        xb_num = torch.from_numpy(
            X_num[indices]
        ).to(DEVICE)

        xb_cat = torch.from_numpy(
            X_cat[indices]
        ).to(DEVICE)

        yb = torch.from_numpy(
            y_array[indices].astype(
                np.float32,
                copy=False,
            )
        ).to(DEVICE)


        optimizer.zero_grad(
            set_to_none=True
        )


        logits = model(
            xb_num,
            xb_cat,
        ).squeeze(-1)


        # Train each TabM member independently.
        targets = yb.unsqueeze(1).expand_as(
            logits
        )


        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
        )


        loss.backward()


        # A little protection against unstable gradients.
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )


        optimizer.step()


        batch_n = len(indices)

        total_loss += (
            loss.item()
            * batch_n
        )

        total_count += batch_n


    return (
        total_loss
        / max(total_count, 1)
    )


# ============================================================
# 11. INNER VALIDATION / EARLY STOPPING
# ============================================================

def train_with_validation(
    model,
    X_fit_num,
    X_fit_cat,
    y_fit,
    X_valid_num,
    X_valid_cat,
    y_valid,
    fold,
):

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    best_auc = -np.inf
    best_epoch = 1
    stale_epochs = 0


    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            X_num=X_fit_num,
            X_cat=X_fit_cat,
            y_array=y_fit,
            batch_size=BATCH_SIZE,
        )


        valid_probability = predict_probability(
            model,
            X_valid_num,
            X_valid_cat,
        )


        valid_auc = roc_auc_score(
            y_valid,
            valid_probability,
        )


        improved = (
            valid_auc
            > best_auc + 1e-5
        )


        if improved:

            best_auc = valid_auc
            best_epoch = epoch
            stale_epochs = 0

        else:

            stale_epochs += 1


        print(
            f"Fold {fold} | "
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Loss {train_loss:.6f} | "
            f"Inner AUC {valid_auc:.9f} | "
            f"Best {best_auc:.9f} "
            f"(epoch {best_epoch})"
        )


        if (
            epoch >= MIN_EPOCHS
            and stale_epochs >= PATIENCE
        ):

            print(
                f"Fold {fold}: early stopping "
                f"at epoch {epoch}."
            )

            break


    return best_epoch, best_auc


# ============================================================
# 12. FIXED-EPOCH REFIT
# ============================================================

def train_fixed_epochs(
    model,
    X_num,
    X_cat,
    y_array,
    n_epochs,
    fold,
):

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    for epoch in range(
        1,
        n_epochs + 1,
    ):

        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            X_num=X_num,
            X_cat=X_cat,
            y_array=y_array,
            batch_size=BATCH_SIZE,
        )


        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == n_epochs
        ):

            print(
                f"Fold {fold} | "
                f"Refit epoch {epoch:02d}/{n_epochs} | "
                f"Loss {train_loss:.6f}"
            )


# ============================================================
# 13. FIVE-FOLD OOF TRAINING
# ============================================================

print("\n" + "=" * 100)
print("STARTING 5-FOLD TABM TRAINING")
print("=" * 100)


skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED,
)


oof_prediction = np.full(
    len(train),
    np.nan,
    dtype=np.float64,
)


test_prediction_sum = np.zeros(
    len(test),
    dtype=np.float64,
)


fold_metrics = []


for fold, (
    outer_train_idx,
    outer_valid_idx,
) in enumerate(
    skf.split(train, y),
    start=1,
):

    print("\n" + "#" * 100)
    print(f"OUTER FOLD {fold}/{N_FOLDS}")
    print("#" * 100)


    # --------------------------------------------------------
    # Stage A:
    # Use an internal validation subset to select best epoch.
    # This inner validation set is not the OOF fold.
    # --------------------------------------------------------

    inner_fit_idx, inner_valid_idx = train_test_split(
        outer_train_idx,
        test_size=INNER_VALID_SIZE,
        random_state=SEED + fold,
        stratify=y[outer_train_idx],
    )


    inner_fit_df = train.iloc[
        inner_fit_idx
    ]

    inner_valid_df = train.iloc[
        inner_valid_idx
    ]


    inner_preprocessor = fit_preprocessor(
        inner_fit_df
    )


    X_inner_fit_num, X_inner_fit_cat = transform_features(
        inner_fit_df,
        inner_preprocessor,
    )


    X_inner_valid_num, X_inner_valid_cat = transform_features(
        inner_valid_df,
        inner_preprocessor,
    )


    y_inner_fit = y[
        inner_fit_idx
    ]

    y_inner_valid = y[
        inner_valid_idx
    ]


    inner_cards = get_cat_cardinalities(
        inner_preprocessor
    )


    inner_model = create_tabm_model(
        X_inner_fit_num,
        inner_cards,
        seed=SEED + fold,
    )


    best_epoch, inner_best_auc = train_with_validation(
        model=inner_model,
        X_fit_num=X_inner_fit_num,
        X_fit_cat=X_inner_fit_cat,
        y_fit=y_inner_fit,
        X_valid_num=X_inner_valid_num,
        X_valid_cat=X_inner_valid_cat,
        y_valid=y_inner_valid,
        fold=fold,
    )


    print(
        f"\nFold {fold}: "
        f"best inner epoch = {best_epoch}, "
        f"inner AUC = {inner_best_auc:.9f}"
    )


    # --------------------------------------------------------
    # Release inner model and memory.
    # --------------------------------------------------------

    del inner_model
    del X_inner_fit_num
    del X_inner_fit_cat
    del X_inner_valid_num
    del X_inner_valid_cat

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # --------------------------------------------------------
    # Stage B:
    # Refit from scratch on the complete outer training fold.
    #
    # Use the epoch count selected in Stage A.
    # This gives each OOF model the full outer training fold.
    # --------------------------------------------------------

    outer_train_df = train.iloc[
        outer_train_idx
    ]

    outer_valid_df = train.iloc[
        outer_valid_idx
    ]


    full_preprocessor = fit_preprocessor(
        outer_train_df
    )


    X_outer_train_num, X_outer_train_cat = transform_features(
        outer_train_df,
        full_preprocessor,
    )


    X_outer_valid_num, X_outer_valid_cat = transform_features(
        outer_valid_df,
        full_preprocessor,
    )


    X_test_num, X_test_cat = transform_features(
        test,
        full_preprocessor,
    )


    y_outer_train = y[
        outer_train_idx
    ]

    y_outer_valid = y[
        outer_valid_idx
    ]


    full_cards = get_cat_cardinalities(
        full_preprocessor
    )


    final_model = create_tabm_model(
        X_outer_train_num,
        full_cards,
        seed=SEED + 1000 + fold,
    )


    train_fixed_epochs(
        model=final_model,
        X_num=X_outer_train_num,
        X_cat=X_outer_train_cat,
        y_array=y_outer_train,
        n_epochs=best_epoch,
        fold=fold,
    )


    # --------------------------------------------------------
    # OOF predictions for outer validation fold
    # --------------------------------------------------------

    fold_oof_pred = predict_probability(
        final_model,
        X_outer_valid_num,
        X_outer_valid_cat,
    )


    fold_auc = roc_auc_score(
        y_outer_valid,
        fold_oof_pred,
    )


    oof_prediction[
        outer_valid_idx
    ] = fold_oof_pred


    # --------------------------------------------------------
    # Test predictions from this fold's model
    # --------------------------------------------------------

    fold_test_pred = predict_probability(
        final_model,
        X_test_num,
        X_test_cat,
    )


    test_prediction_sum += (
        fold_test_pred
        / N_FOLDS
    )


    fold_metrics.append({

        "fold": fold,

        "outer_train_rows":
            len(outer_train_idx),

        "outer_valid_rows":
            len(outer_valid_idx),

        "best_epoch":
            best_epoch,

        "inner_best_auc":
            inner_best_auc,

        "outer_oof_auc":
            fold_auc,

    })


    print("\n" + "-" * 80)

    print(
        f"Fold {fold} OOF AUC = "
        f"{fold_auc:.9f}"
    )

    print(
        f"Fold {fold} selected epoch = "
        f"{best_epoch}"
    )

    print("-" * 80)


    # --------------------------------------------------------
    # Free fold memory
    # --------------------------------------------------------

    del final_model

    del X_outer_train_num
    del X_outer_train_cat

    del X_outer_valid_num
    del X_outer_valid_cat

    del X_test_num
    del X_test_cat

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 14. OOF RESULT
# ============================================================

if np.isnan(oof_prediction).any():

    raise ValueError(
        "OOF predictions contain missing rows."
    )


if not np.isfinite(oof_prediction).all():

    raise ValueError(
        "OOF predictions contain non-finite values."
    )


tabm_oof_auc = roc_auc_score(
    y,
    oof_prediction,
)


print("\n" + "=" * 100)
print("TABM OOF RESULT")
print("=" * 100)

print(
    f"TabM OOF AUC = {tabm_oof_auc:.9f}"
)


print(
    "\nFold AUC mean = "
    f"{np.mean([x['outer_oof_auc'] for x in fold_metrics]):.9f}"
)


print(
    "Fold AUC std  = "
    f"{np.std([x['outer_oof_auc'] for x in fold_metrics], ddof=1):.9f}"
)


# ============================================================
# 15. SAVE TABM PREDICTIONS
# ============================================================

tabm_oof_df = pd.DataFrame({

    ID_COL:
        train_ids,

    TARGET:
        y,

    "prediction":
        oof_prediction,

})


tabm_test_df = pd.DataFrame({

    ID_COL:
        test_ids,

    "prediction":
        test_prediction_sum,

})


tabm_oof_path = os.path.join(
    OUTPUT_DIR,
    "tabm_oof.csv",
)


tabm_test_path = os.path.join(
    OUTPUT_DIR,
    "tabm_test.csv",
)


tabm_oof_df.to_csv(
    tabm_oof_path,
    index=False,
    encoding="utf-8-sig",
)


tabm_test_df.to_csv(
    tabm_test_path,
    index=False,
    encoding="utf-8-sig",
)


fold_metrics_df = pd.DataFrame(
    fold_metrics
)


fold_metrics_path = os.path.join(
    OUTPUT_DIR,
    "tabm_fold_metrics.csv",
)


fold_metrics_df.to_csv(
    fold_metrics_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 16. LOAD EXISTING CHAMPION PREDICTIONS
# ============================================================

print("\n" + "=" * 100)
print("LOADING CURRENT CHAMPION PREDICTIONS")
print("=" * 100)


def load_reference_oof(
    path,
    name,
):

    assert_file_exists(
        path,
        name,
    )

    df = pd.read_csv(
        path
    )


    required = {
        ID_COL,
        TARGET,
        "prediction",
    }


    if not required.issubset(
        df.columns
    ):

        raise ValueError(
            f"{name}: required columns "
            f"{required}; found {df.columns.tolist()}"
        )


    if df[ID_COL].duplicated().any():
        raise ValueError(
            f"{name}: duplicate IDs."
        )


    check_same_ids(
        train_ids,
        df[ID_COL],
        name,
    )


    aligned = (
        df
        .set_index(ID_COL)
        .loc[train_ids]
        .reset_index()
    )


    if not np.array_equal(
        aligned[TARGET].to_numpy().astype(np.int8),
        y,
    ):

        raise ValueError(
            f"{name}: target labels do not align with train.csv."
        )


    pred = aligned[
        "prediction"
    ].to_numpy(
        dtype=np.float64
    )


    if not np.isfinite(pred).all():
        raise ValueError(
            f"{name}: non-finite predictions."
        )


    auc = roc_auc_score(
        y,
        pred,
    )


    print(
        f"{name:25s} AUC = {auc:.9f}"
    )


    return pred


def load_reference_test(
    path,
    name,
):

    assert_file_exists(
        path,
        name,
    )

    df = pd.read_csv(
        path
    )


    required = {
        ID_COL,
        "prediction",
    }


    if not required.issubset(
        df.columns
    ):

        raise ValueError(
            f"{name}: required columns "
            f"{required}; found {df.columns.tolist()}"
        )


    if df[ID_COL].duplicated().any():
        raise ValueError(
            f"{name}: duplicate IDs."
        )


    check_same_ids(
        test_ids,
        df[ID_COL],
        name,
    )


    aligned = (
        df
        .set_index(ID_COL)
        .loc[test_ids]
        .reset_index()
    )


    pred = aligned[
        "prediction"
    ].to_numpy(
        dtype=np.float64
    )


    if not np.isfinite(pred).all():
        raise ValueError(
            f"{name}: non-finite predictions."
        )


    return pred


current_lgbm_oof = load_reference_oof(
    CURRENT_LGBM_OOF_PATH,
    "Current Conditional-TE LGBM OOF",
)


v1_oof = load_reference_oof(
    V1_OOF_PATH,
    "Lean RealMLP V1 OOF",
)


current_lgbm_test = load_reference_test(
    CURRENT_LGBM_TEST_PATH,
    "Current Conditional-TE LGBM TEST",
)


v1_test = load_reference_test(
    V1_TEST_PATH,
    "Lean RealMLP V1 TEST",
)


# ============================================================
# 17. REBUILD CURRENT CHAMPION
# ============================================================

current_lgbm_rank_oof = normalized_rank(
    current_lgbm_oof
)


v1_rank_oof = normalized_rank(
    v1_oof
)


champion_rank_oof = (
    0.40 * current_lgbm_rank_oof
    +
    0.60 * v1_rank_oof
)


champion_oof_auc = roc_auc_score(
    y,
    champion_rank_oof,
)


print("\n" + "=" * 100)
print("CURRENT CHAMPION CHECK")
print("=" * 100)


print(
    f"Recomputed Champion OOF = "
    f"{champion_oof_auc:.9f}"
)


print(
    f"Known Champion OOF      = "
    f"{KNOWN_CHAMPION_OOF:.9f}"
)


print(
    f"Difference              = "
    f"{champion_oof_auc - KNOWN_CHAMPION_OOF:+.9f}"
)


if abs(
    champion_oof_auc
    - KNOWN_CHAMPION_OOF
) > 0.00002:

    raise ValueError(
        "Champion benchmark mismatch. "
        "Check current LGBM and RealMLP prediction files."
    )


# ============================================================
# 18. TABM RANK CORRELATION WITH CHAMPION
# ============================================================

tabm_rank_oof = normalized_rank(
    oof_prediction
)


champion_vs_tabm_corr = pd.Series(
    champion_rank_oof
).corr(
    pd.Series(tabm_rank_oof),
    method="spearman",
)


print("\n" + "=" * 100)
print("TABM DIVERSITY CHECK")
print("=" * 100)


print(
    f"Spearman(TabM rank, Champion rank) = "
    f"{champion_vs_tabm_corr:.9f}"
)


# ============================================================
# 19. TABM + CHAMPION RANK BLEND
# ============================================================

blend_results = []

blend_predictions_oof = {}


for tabm_weight in TABM_BLEND_WEIGHTS:

    blend_oof = (
        (1.0 - tabm_weight)
        * champion_rank_oof

        +

        tabm_weight
        * tabm_rank_oof
    )


    blend_auc = roc_auc_score(
        y,
        blend_oof,
    )


    blend_predictions_oof[
        tabm_weight
    ] = blend_oof


    blend_results.append({

        "tabm_weight":
            tabm_weight,

        "champion_weight":
            1.0 - tabm_weight,

        "oof_auc":
            blend_auc,

        "gain_vs_champion":
            blend_auc - champion_oof_auc,

    })


blend_results_df = (
    pd.DataFrame(
        blend_results
    )
    .sort_values(
        "oof_auc",
        ascending=False,
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 100)
print("TABM + CHAMPION RANK BLEND CANDIDATES")
print("=" * 100)


print(
    blend_results_df.to_string(
        index=False,
        formatters={
            "oof_auc": "{:.9f}".format,
            "gain_vs_champion": "{:+.9f}".format,
        },
    )
)


blend_candidates_path = os.path.join(
    OUTPUT_DIR,
    "tabm_champion_rank_blend_candidates.csv",
)


blend_results_df.to_csv(
    blend_candidates_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 20. TABM / CHAMPION RANK ON TEST
# ============================================================

current_lgbm_rank_test = normalized_rank(
    current_lgbm_test
)


v1_rank_test = normalized_rank(
    v1_test
)


champion_rank_test = (
    0.40 * current_lgbm_rank_test
    +
    0.60 * v1_rank_test
)


tabm_rank_test = normalized_rank(
    test_prediction_sum
)


# Candidate with maximum OOF AUC is used for an experimental
# submission. This does not establish that it will improve LB.
best_row = blend_results_df.iloc[0]


best_tabm_weight = float(
    best_row["tabm_weight"]
)


best_champion_weight = (
    1.0 - best_tabm_weight
)


best_oof_blend = (
    best_champion_weight
    * champion_rank_oof

    +

    best_tabm_weight
    * tabm_rank_oof
)


best_oof_auc = roc_auc_score(
    y,
    best_oof_blend,
)


best_test_blend = (
    best_champion_weight
    * champion_rank_test

    +

    best_tabm_weight
    * tabm_rank_test
)


print("\n" + "=" * 100)
print("BEST OOF CANDIDATE")
print("=" * 100)


print(
    f"Champion weight = {best_champion_weight:.3f}"
)


print(
    f"TabM weight     = {best_tabm_weight:.3f}"
)


print(
    f"OOF AUC         = {best_oof_auc:.9f}"
)


print(
    f"Gain vs Champion = "
    f"{best_oof_auc - champion_oof_auc:+.9f}"
)


MIN_MEANINGFUL_GAIN = 0.00002


if (
    best_oof_auc - champion_oof_auc
    < MIN_MEANINGFUL_GAIN
):

    print(
        "\nDECISION: No convincing OOF gain."
    )

    print(
        "Keep the current Champion as the official best model."
    )

    print(
        "The experimental blend should not automatically "
        "replace the current Champion."
    )

else:

    print(
        "\nDECISION: Promising OOF gain."
    )

    print(
        "The blend is worth considering for one controlled "
        "Public LB test, but OOF gain alone is not proof."
    )


# ============================================================
# 21. SAVE SUBMISSIONS
# ============================================================

target_columns = [
    col
    for col in sample.columns
    if col != ID_COL
]


if len(target_columns) != 1:

    raise ValueError(
        "Expected exactly one target column "
        "besides id in sample_submission."
    )


submission_target = target_columns[0]


def make_submission(
    prediction,
    filename,
):

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )


    if len(prediction) != len(test):
        raise ValueError(
            "Submission prediction length mismatch."
        )


    if not np.isfinite(
        prediction
    ).all():

        raise ValueError(
            "Submission has non-finite predictions."
        )


    pred_df = pd.DataFrame({

        ID_COL:
            test_ids,

        "prediction":
            prediction,

    })


    check_same_ids(
        sample[ID_COL],
        pred_df[ID_COL],
        "sample submission vs prediction",
    )


    submission = (
        sample[
            [ID_COL]
        ]
        .copy()
        .merge(
            pred_df,
            on=ID_COL,
            how="left",
            validate="one_to_one",
        )
    )


    if submission["prediction"].isna().any():
        raise ValueError(
            "Submission has missing predictions."
        )


    submission[
        submission_target
    ] = submission[
        "prediction"
    ].astype(
        np.float64
    )


    submission = submission[
        [
            ID_COL,
            submission_target,
        ]
    ]


    if len(submission) != len(test):
        raise ValueError(
            "Submission row count mismatch."
        )


    if not np.array_equal(
        submission[ID_COL].to_numpy(),
        sample[ID_COL].to_numpy(),
    ):

        raise ValueError(
            "Submission order does not match sample_submission."
        )


    output_path = os.path.join(
        OUTPUT_DIR,
        filename,
    )


    submission.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        "Saved:",
        output_path,
    )


    return output_path


# Current champion
champion_submission = make_submission(
    champion_rank_test,
    "submission_champion_rank_4060.csv",
)


# TabM standalone
tabm_submission = make_submission(
    tabm_rank_test,
    "submission_tabm_only_rank.csv",
)


# Best OOF-selected blend: experimental only
best_blend_submission = make_submission(
    best_test_blend,
    "submission_tabm_plus_champion_rank_best_oof.csv",
)


# ============================================================
# 22. SAVE COMBINED OOF RESULTS
# ============================================================

oof_output = pd.DataFrame({

    ID_COL:
        train_ids,

    TARGET:
        y,

    "tabm_prediction":
        oof_prediction,

    "tabm_rank":
        tabm_rank_oof,

    "champion_rank":
        champion_rank_oof,

    "best_blend_rank":
        best_oof_blend,

})


oof_output_path = os.path.join(
    OUTPUT_DIR,
    "tabm_champion_oof_predictions.csv",
)


oof_output.to_csv(
    oof_output_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 23. SAVE COMBINED TEST RESULTS
# ============================================================

test_output = pd.DataFrame({

    ID_COL:
        test_ids,

    "tabm_prediction":
        test_prediction_sum,

    "tabm_rank":
        tabm_rank_test,

    "champion_rank":
        champion_rank_test,

    "best_blend_rank":
        best_test_blend,

})


test_output_path = os.path.join(
    OUTPUT_DIR,
    "tabm_champion_test_predictions.csv",
)


test_output.to_csv(
    test_output_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 24. FINAL SUMMARY
# ============================================================

summary_df = pd.DataFrame([

    {
        "model":
            "TabM standalone",

        "oof_auc":
            tabm_oof_auc,

        "gain_vs_champion":
            tabm_oof_auc - champion_oof_auc,

        "tabm_weight":
            1.0,

        "champion_weight":
            0.0,

        "rank_correlation_with_champion":
            champion_vs_tabm_corr,

    },

    {
        "model":
            "Current Champion",

        "oof_auc":
            champion_oof_auc,

        "gain_vs_champion":
            0.0,

        "tabm_weight":
            0.0,

        "champion_weight":
            1.0,

        "rank_correlation_with_champion":
            1.0,

    },

    {
        "model":
            "Best OOF-selected TabM + Champion",

        "oof_auc":
            best_oof_auc,

        "gain_vs_champion":
            best_oof_auc - champion_oof_auc,

        "tabm_weight":
            best_tabm_weight,

        "champion_weight":
            best_champion_weight,

        "rank_correlation_with_champion":
            champion_vs_tabm_corr,

    },

])


summary_path = os.path.join(
    OUTPUT_DIR,
    "tabm_experiment_summary.csv",
)


summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig",
)


print("\n" + "=" * 100)
print("FINAL TABM SUMMARY")
print("=" * 100)


print(
    summary_df.to_string(
        index=False,
        formatters={
            "oof_auc": "{:.9f}".format,
            "gain_vs_champion": "{:+.9f}".format,
            "rank_correlation_with_champion": "{:.6f}".format,
        },
    )
)


print("\n" + "=" * 100)
print("SAVED FILES")
print("=" * 100)


print(
    "\n1. TabM OOF:"
    f"\n{tabm_oof_path}"
)


print(
    "\n2. TabM TEST:"
    f"\n{tabm_test_path}"
)


print(
    "\n3. Fold metrics:"
    f"\n{fold_metrics_path}"
)


print(
    "\n4. Blend candidates:"
    f"\n{blend_candidates_path}"
)


print(
    "\n5. Combined OOF:"
    f"\n{oof_output_path}"
)


print(
    "\n6. Combined TEST:"
    f"\n{test_output_path}"
)


print(
    "\n7. Summary:"
    f"\n{summary_path}"
)


print(
    "\n8. Champion submission:"
    f"\n{champion_submission}"
)


print(
    "\n9. TabM standalone submission:"
    f"\n{tabm_submission}"
)


print(
    "\n10. Experimental blend submission:"
    f"\n{best_blend_submission}"
)


print("\n" + "=" * 100)
print("TABM EXPERIMENT COMPLETE")
print("=" * 100)