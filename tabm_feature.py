# ============================================================
# TABM V1 FEATURE-ENHANCED EXPERIMENT
#
# Feature recipe inspired by Lean RealMLP V1.
#
# Data:
#   Competition train.csv / test.csv only.
#
# No external 129K data, external predictions, Route Profile,
# or RealMLP V2.
#
# Features:
#   21 original predictors
#   + 17 numeric-as-categorical copies
#   + 5 count features
#   + 3 categorical interaction features
#   + 3 interaction target encodings
#
# TabM inputs:
#   Numerical:    17 + 5 + 3 = 25
#   Categorical:   4 + 17 + 3 = 24
#   Total:                         49
#
# Validation:
#   5-fold outer OOF
#   Inner validation selects training epochs
#   Training-fold target encoding is cross-fitted
#
# Fusion:
#   Current LGBM 40% + RealMLP V1 60% = Champion
#
# Output:
#   D:\kaggle\airline\tabm_v1_feature_experiment\
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
from sklearn.preprocessing import TargetEncoder

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

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

CURRENT_LGBM_OOF_PATH = (
    r"D:\kaggle\airline\conditional_te_experiment"
    r"\conditional_te_lgbm_oof.csv"
)
CURRENT_LGBM_TEST_PATH = (
    r"D:\kaggle\airline\conditional_te_experiment"
    r"\conditional_te_lgbm_test.csv"
)

V1_OOF_PATH = (
    r"D:\kaggle\airline\lean_realmlp_blend"
    r"\lean_realmlp_oof.csv"
)
V1_TEST_PATH = (
    r"D:\kaggle\airline\lean_realmlp_blend"
    r"\lean_realmlp_test.csv"
)

OUTPUT_DIR = (
    r"D:\kaggle\airline\tabm_v1_feature_experiment"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

ID_COL = "id"
TARGET = "satisfaction"

SEED = 42
N_FOLDS = 5
INNER_VALID_SIZE = 0.10

# Training
MAX_EPOCHS = 25
MIN_EPOCHS = 5
PATIENCE = 4
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0003
BATCH_SIZE = 2048

# TabM
TABM_K = 16
TABM_BLOCKS = 2
TABM_WIDTH = 256
TABM_DROPOUT = 0.10

# Piecewise Linear Embeddings
PLE_N_BINS = 32
PLE_D_EMBEDDING = 12
MAX_BIN_SAMPLE = 100_000

# Current verified Champion OOF
KNOWN_CHAMPION_OOF = 0.961278687

# Pre-registered fusion candidates
TABM_BLEND_WEIGHTS = [
    0.00,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
]

MIN_MEANINGFUL_GAIN = 0.00002

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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

MISSING_TOKEN = "__MISSING__"

# Duplicate all 17 original numerical features as categories.
NUMERIC_CAT_COLS = [
    f"catnum__{col}"
    for col in NUMERIC_COLS
]

# Five count features from the Lean RealMLP V1 recipe.
COUNT_SOURCE_MAP = {
    "count__Gender": "Gender",
    "count__Customer Type": "Customer Type",
    "count__Type of Travel": "Type of Travel",
    "count__Class": "Class",
    "count__Leg room service": "catnum__Leg room service",
}

COUNT_FEATURE_COLS = list(
    COUNT_SOURCE_MAP.keys()
)

# Three interaction features.
INTERACTION_COLS = [
    "int__Class_Travel_Gender",
    "int__Class_Travel_Age",
    "int__Distance_DepArrivalConvenience",
]

# Target encoding for the same three interactions.
TE_COLS = INTERACTION_COLS.copy()

TABM_NUMERIC_FEATURES = (
    NUMERIC_COLS
    + COUNT_FEATURE_COLS
    + [f"te__{c}" for c in TE_COLS]
)

TABM_CATEGORICAL_FEATURES = (
    CATEGORICAL_COLS
    + NUMERIC_CAT_COLS
    + INTERACTION_COLS
)

assert len(NUMERIC_COLS) == 17
assert len(CATEGORICAL_COLS) == 4
assert len(COUNT_FEATURE_COLS) == 5
assert len(INTERACTION_COLS) == 3
assert len(TE_COLS) == 3
assert len(TABM_NUMERIC_FEATURES) == 25
assert len(TABM_CATEGORICAL_FEATURES) == 24


# ============================================================
# 4. RANDOM SEED AND INITIALIZATION
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

print("=" * 100)
print("TABM V1 FEATURE-ENHANCED EXPERIMENT")
print("=" * 100)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Folds:", N_FOLDS)
print("TabM k:", TABM_K)
print("TabM width:", TABM_WIDTH)
print("TabM blocks:", TABM_BLOCKS)
print("Numerical input features:", len(TABM_NUMERIC_FEATURES))
print("Categorical input features:", len(TABM_CATEGORICAL_FEATURES))
print(
    "Total input features:",
    len(TABM_NUMERIC_FEATURES) + len(TABM_CATEGORICAL_FEATURES),
)


# ============================================================
# 5. GENERAL HELPERS
# ============================================================

def assert_file_exists(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )


def check_same_ids(reference_ids, candidate_ids, name):
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
        missing = reference_ids - candidate_ids
        extra = candidate_ids - reference_ids

        raise ValueError(
            f"{name}: ID mismatch.\n"
            f"Missing: {len(missing)}\n"
            f"Extra: {len(extra)}"
        )


def normalized_rank(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Cannot rank non-finite predictions."
        )

    return (
        rankdata(values, method="average")
        / len(values)
    )


def clean_category(series):
    return (
        series
        .astype("string")
        .fillna(MISSING_TOKEN)
        .astype(str)
        .str.strip()
    )


def numeric_to_category_values(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).astype(np.float64)

    values = values.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return values


def stable_interaction_hash(component_df):
    """
    Hash combinations using feature values only.
    No labels are used in constructing interaction IDs.
    """

    hashed = pd.util.hash_pandas_object(
        component_df,
        index=False,
    )

    return (
        hashed
        .to_numpy(dtype=np.uint64)
        .view(np.int64)
    )


# ============================================================
# 6. BUILD RAW FEATURE VIEWS
# ============================================================

def build_raw_views(df):
    """
    Create target-independent feature views.

    numeric:
        17 original numerical features

    categorical:
        4 original categorical features
        17 numerical features duplicated as categories
        3 interaction categories

    count_keys:
        source variables for the 5 count features

    te:
        three interaction columns to target-encode
    """

    # --------------------------------------------------------
    # A. Original numerical features
    # --------------------------------------------------------

    numeric_df = pd.DataFrame(index=df.index)

    for col in NUMERIC_COLS:
        values = pd.to_numeric(
            df[col],
            errors="coerce",
        ).astype(np.float64)

        values = values.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        numeric_df[col] = values


    # --------------------------------------------------------
    # B. Original categorical features
    # --------------------------------------------------------

    categorical_data = {}

    for col in CATEGORICAL_COLS:
        categorical_data[col] = clean_category(
            df[col]
        )


    # --------------------------------------------------------
    # C. Numeric-as-categorical copies
    # --------------------------------------------------------

    for col in NUMERIC_COLS:
        categorical_data[
            f"catnum__{col}"
        ] = numeric_to_category_values(
            df[col]
        )


    # --------------------------------------------------------
    # D. Class × Type of Travel × Gender
    # --------------------------------------------------------

    components_1 = pd.DataFrame({
        "Class": clean_category(df["Class"]),
        "Type of Travel": clean_category(df["Type of Travel"]),
        "Gender": clean_category(df["Gender"]),
    }, index=df.index)

    categorical_data[
        "int__Class_Travel_Gender"
    ] = stable_interaction_hash(
        components_1
    )


    # --------------------------------------------------------
    # E. Class × Type of Travel × Age
    # --------------------------------------------------------

    age_values = numeric_to_category_values(
        df["Age"]
    )

    components_2 = pd.DataFrame({
        "Class": clean_category(df["Class"]),
        "Type of Travel": clean_category(df["Type of Travel"]),
        "Age": age_values,
    }, index=df.index)

    categorical_data[
        "int__Class_Travel_Age"
    ] = stable_interaction_hash(
        components_2
    )


    # --------------------------------------------------------
    # F. Flight Distance × Departure/Arrival convenience
    # --------------------------------------------------------

    distance_values = numeric_to_category_values(
        df["Flight Distance"]
    )

    departure_convenience_values = numeric_to_category_values(
        df["Departure/Arrival time convenient"]
    )

    components_3 = pd.DataFrame({
        "Flight Distance": distance_values,
        "Departure/Arrival time convenient": (
            departure_convenience_values
        ),
    }, index=df.index)

    categorical_data[
        "int__Distance_DepArrivalConvenience"
    ] = stable_interaction_hash(
        components_3
    )


    categorical_df = pd.DataFrame(
        categorical_data,
        index=df.index,
    )


    # --------------------------------------------------------
    # G. Five count-feature sources
    #
    # Canonical strings make lookup consistent, including
    # missing numeric category values.
    # --------------------------------------------------------

    count_keys = pd.DataFrame(index=df.index)

    for output_col, source_col in COUNT_SOURCE_MAP.items():
        count_keys[output_col] = clean_category(
            categorical_df[source_col]
        )


    # --------------------------------------------------------
    # H. Three target-encoding source columns
    # --------------------------------------------------------

    te_df = categorical_df[
        TE_COLS
    ].copy()


    return {
        "numeric": numeric_df,
        "categorical": categorical_df,
        "count_keys": count_keys,
        "te": te_df,
    }


def subset_raw_views(raw_views, indices):
    return {
        key: value.iloc[indices].reset_index(drop=True)
        for key, value in raw_views.items()
    }


# ============================================================
# 7. CATEGORICAL CARDINALITIES
# ============================================================

def get_cat_cardinalities(pipeline):
    """
    Code conventions:

    Numeric-as-categorical:
        0 = unseen
        1 = missing
        2+ = observed numeric value

    Other categorical features:
        0 = unseen
        1+ = observed category
    """

    cardinalities = []

    for col in TABM_CATEGORICAL_FEATURES:

        if col in NUMERIC_CAT_COLS:
            cardinality = (
                len(
                    pipeline["numeric_cat_maps"][col]
                ) + 2
            )
        else:
            cardinality = (
                len(
                    pipeline["other_cat_maps"][col]
                ) + 1
            )

        if cardinality < 2:
            raise ValueError(
                f"Invalid category cardinality for {col}: "
                f"{cardinality}"
            )

        cardinalities.append(
            int(cardinality)
        )

    return cardinalities


# ============================================================
# 8. NUMERICAL MATRIX
# ============================================================

def make_numeric_matrix(raw, pipeline, te_values):
    """
    Numerical inputs:
        17 original numeric features
         5 count features
         3 target encodings
    """

    # --------------------------------------------------------
    # A. Original numerical inputs: fit-fold medians
    # --------------------------------------------------------

    numeric_df = raw["numeric"].copy()

    for col in NUMERIC_COLS:
        numeric_df[col] = numeric_df[col].fillna(
            pipeline["medians"][col]
        )

    X_original = numeric_df[
        NUMERIC_COLS
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )


    # --------------------------------------------------------
    # B. Count features
    # --------------------------------------------------------

    count_arrays = []

    for count_col in COUNT_FEATURE_COLS:
        mapper = pipeline["count_maps"][count_col]

        values = raw["count_keys"][count_col]

        counts = (
            values
            .map(mapper)
            .fillna(0)
            .to_numpy(
                dtype=np.float64,
                copy=True,
            )
        )

        count_values = np.log1p(
            counts
        ).astype(np.float32)

        count_arrays.append(
            count_values
        )

    X_counts = np.column_stack(
        count_arrays
    ).astype(
        np.float32,
        copy=False,
    )


    # --------------------------------------------------------
    # C. Target encodings
    # --------------------------------------------------------

    X_te = np.asarray(
        te_values,
        dtype=np.float32,
    )

    if X_te.ndim == 1:
        X_te = X_te.reshape(-1, 1)

    if X_te.shape[1] != len(TE_COLS):
        raise ValueError(
            f"Expected {len(TE_COLS)} target-encoding columns, "
            f"got {X_te.shape[1]}"
        )


    # --------------------------------------------------------
    # D. Combine
    # --------------------------------------------------------

    X_num = np.column_stack([
        X_original,
        X_counts,
        X_te,
    ]).astype(
        np.float32,
        copy=True,
    )

    if X_num.shape[1] != 25:
        raise ValueError(
            f"Expected 25 numerical features, "
            f"got {X_num.shape[1]}"
        )

    if not np.isfinite(X_num).all():
        raise ValueError(
            "Numerical matrix contains NaN or inf."
        )

    return X_num


# ============================================================
# 9. CATEGORICAL MATRIX
# ============================================================

def make_categorical_matrix(raw, pipeline):
    """
    Categorical inputs:
        4 original categorical features
       17 numeric-as-categorical features
        3 interaction categories

    FIX:
        Every NumPy array that is modified in place is created
        with copy=True, avoiding read-only array assignment errors.
    """

    categorical_df = raw["categorical"]

    encoded_columns = []

    for col in TABM_CATEGORICAL_FEATURES:
        values = categorical_df[col]

        # ----------------------------------------------------
        # Numeric-as-categorical columns
        # ----------------------------------------------------

        if col in NUMERIC_CAT_COLS:
            mapper = pipeline[
                "numeric_cat_maps"
            ][col]

            mapped = (
                values
                .map(mapper)
                .fillna(0)
                .to_numpy(
                    dtype=np.int64,
                    copy=True,  # IMPORTANT FIX
                )
            )

            # This mask is only read, but make it an independent
            # array for consistent NumPy behavior.
            missing_mask = values.isna().to_numpy(
                dtype=bool,
                copy=True,
            )

            # Code 0 = unseen; code 1 = missing;
            # codes 2+ = known numeric values.
            mapped[missing_mask] = 1

        # ----------------------------------------------------
        # Original categories and interaction features
        # ----------------------------------------------------

        else:
            mapper = pipeline[
                "other_cat_maps"
            ][col]

            mapped = (
                values
                .map(mapper)
                .fillna(0)
                .to_numpy(
                    dtype=np.int64,
                    copy=True,  # IMPORTANT FIX
                )
            )

        if mapped.ndim != 1:
            raise ValueError(
                f"Expected one-dimensional codes for {col}."
            )

        if mapped.size and mapped.min() < 0:
            raise ValueError(
                f"Negative category code detected in {col}."
            )

        encoded_columns.append(
            mapped
        )


    # column_stack creates an independent combined array.
    X_cat = np.column_stack(
        encoded_columns
    ).astype(
        np.int64,
        copy=True,
    )


    if X_cat.shape[1] != 24:
        raise ValueError(
            f"Expected 24 categorical features, "
            f"got {X_cat.shape[1]}"
        )

    if not X_cat.flags.writeable:
        raise ValueError(
            "Categorical matrix must be writable."
        )

    if X_cat.size and X_cat.min() < 0:
        raise ValueError(
            "Categorical codes must be non-negative."
        )

    return X_cat


# ============================================================
# 10. FIT FOLD-SPECIFIC FEATURE PIPELINE
# ============================================================

def fit_feature_pipeline(raw_fit, y_fit, seed):
    """
    Fit preprocessing only on the fitting partition.

    Fitted elements:
        - Numerical medians
        - Numeric category mappings
        - Other category mappings
        - Count maps
        - TargetEncoder

    TargetEncoder.fit_transform returns cross-fitted encodings
    for the training rows. Validation/test rows use transform().
    """

    y_fit = np.asarray(
        y_fit,
        dtype=np.int8,
    )


    # --------------------------------------------------------
    # A. Numerical medians
    # --------------------------------------------------------

    medians = {}

    for col in NUMERIC_COLS:
        median = raw_fit["numeric"][col].median()

        if not np.isfinite(median):
            median = 0.0

        medians[col] = float(median)


    # --------------------------------------------------------
    # B. Category mappings
    # --------------------------------------------------------

    numeric_cat_maps = {}
    other_cat_maps = {}

    for col in TABM_CATEGORICAL_FEATURES:
        values = raw_fit["categorical"][col]

        if col in NUMERIC_CAT_COLS:
            unique_values = (
                values
                .dropna()
                .unique()
                .tolist()
            )

            unique_values = sorted(
                unique_values,
                key=float,
            )

            mapper = {
                float(value): index + 2
                for index, value in enumerate(unique_values)
            }

            numeric_cat_maps[col] = mapper

        else:
            unique_values = values.unique().tolist()

            unique_values = sorted(
                unique_values,
                key=lambda value: str(value),
            )

            mapper = {
                value: index + 1
                for index, value in enumerate(unique_values)
            }

            other_cat_maps[col] = mapper


    # --------------------------------------------------------
    # C. Count maps
    # --------------------------------------------------------

    count_maps = {}

    for count_col in COUNT_FEATURE_COLS:
        counts = (
            raw_fit["count_keys"][count_col]
            .value_counts(dropna=False)
        )

        count_maps[count_col] = counts.to_dict()


    # --------------------------------------------------------
    # D. Cross-fitted Target Encoding
    # --------------------------------------------------------

    te_encoder = TargetEncoder(
        target_type="binary",
        smooth="auto",
        cv=5,
        shuffle=True,
        random_state=seed,
    )

    te_train = te_encoder.fit_transform(
        raw_fit["te"],
        y_fit,
    )

    te_train = np.asarray(
        te_train,
        dtype=np.float32,
    )

    if te_train.ndim == 1:
        te_train = te_train.reshape(-1, 1)

    if te_train.shape[1] != len(TE_COLS):
        raise ValueError(
            f"Expected {len(TE_COLS)} TE columns, "
            f"got {te_train.shape[1]}"
        )

    if not np.isfinite(te_train).all():
        raise ValueError(
            "Non-finite values in cross-fitted Target Encoding."
        )


    pipeline = {
        "medians": medians,
        "numeric_cat_maps": numeric_cat_maps,
        "other_cat_maps": other_cat_maps,
        "count_maps": count_maps,
        "te_encoder": te_encoder,
    }


    X_num_train = make_numeric_matrix(
        raw_fit,
        pipeline,
        te_train,
    )

    X_cat_train = make_categorical_matrix(
        raw_fit,
        pipeline,
    )


    if len(X_num_train) != len(y_fit):
        raise ValueError(
            "Numerical training matrix has wrong row count."
        )

    if len(X_cat_train) != len(y_fit):
        raise ValueError(
            "Categorical training matrix has wrong row count."
        )


    return (
        pipeline,
        X_num_train,
        X_cat_train,
    )


# ============================================================
# 11. TRANSFORM VALIDATION / TEST FEATURES
# ============================================================

def transform_features(raw, pipeline):
    """
    Transform validation/test data using a fitted pipeline.

    No validation/test labels are used.
    """

    te_values = pipeline[
        "te_encoder"
    ].transform(
        raw["te"]
    )

    te_values = np.asarray(
        te_values,
        dtype=np.float32,
    )

    if te_values.ndim == 1:
        te_values = te_values.reshape(-1, 1)

    X_num = make_numeric_matrix(
        raw,
        pipeline,
        te_values,
    )

    X_cat = make_categorical_matrix(
        raw,
        pipeline,
    )

    if len(X_num) != len(X_cat):
        raise ValueError(
            "Numeric and categorical row counts differ."
        )

    return X_num, X_cat


# ============================================================
# 12. CREATE TABM MODEL
# ============================================================

def create_tabm_model(X_num_fit, cat_cardinalities, seed):
    set_seed(seed)

    n_fit = len(X_num_fit)

    sample_size = min(
        MAX_BIN_SAMPLE,
        n_fit,
    )

    rng = np.random.default_rng(seed)

    sample_indices = rng.choice(
        n_fit,
        size=sample_size,
        replace=False,
    )

    bin_source = torch.from_numpy(
        X_num_fit[sample_indices].copy()
    ).float()

    bins = compute_bins(
        bin_source,
        n_bins=PLE_N_BINS,
    )

    num_embeddings = PiecewiseLinearEmbeddings(
        bins,
        d_embedding=PLE_D_EMBEDDING,
        activation=False,
        version="B",
    )

    model = TabM.make(
        n_num_features=X_num_fit.shape[1],
        cat_cardinalities=cat_cardinalities,
        num_embeddings=num_embeddings,
        d_out=1,
        k=TABM_K,
        n_blocks=TABM_BLOCKS,
        d_block=TABM_WIDTH,
        dropout=TABM_DROPOUT,
    )

    return model.to(DEVICE)


# ============================================================
# 13. PREDICTION
# ============================================================

@torch.no_grad()
def predict_probability(
    model,
    X_num,
    X_cat,
    batch_size=BATCH_SIZE,
):
    model.eval()

    predictions = []

    for start in range(0, len(X_num), batch_size):
        end = min(
            start + batch_size,
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

        # TabM output: (batch_size, K)
        # Average member probabilities.
        probs = torch.sigmoid(logits).mean(dim=1)

        predictions.append(
            probs.cpu().numpy()
        )

    result = np.concatenate(predictions)

    if not np.isfinite(result).all():
        raise ValueError(
            "Model generated non-finite predictions."
        )

    return result


# ============================================================
# 14. TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    optimizer,
    X_num,
    X_cat,
    y_array,
):
    model.train()

    n = len(y_array)
    permutation = np.random.permutation(n)

    total_loss = 0.0
    total_count = 0

    for start in range(0, n, BATCH_SIZE):
        indices = permutation[
            start:start + BATCH_SIZE
        ]

        xb_num = torch.from_numpy(
            X_num[indices].copy()
        ).to(DEVICE)

        xb_cat = torch.from_numpy(
            X_cat[indices].copy()
        ).to(DEVICE)

        yb = torch.from_numpy(
            y_array[indices].astype(
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

        targets = yb.unsqueeze(1).expand_as(logits)

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

        batch_n = len(indices)

        total_loss += loss.item() * batch_n
        total_count += batch_n

    return total_loss / max(total_count, 1)


# ============================================================
# 15. INNER TRAINING / EARLY STOPPING
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

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            optimizer,
            X_fit_num,
            X_fit_cat,
            y_fit,
        )

        valid_pred = predict_probability(
            model,
            X_valid_num,
            X_valid_cat,
        )

        valid_auc = roc_auc_score(
            y_valid,
            valid_pred,
        )

        if valid_auc > best_auc + 1e-5:
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
# 16. REFIT FOR SELECTED EPOCHS
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

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(
            model,
            optimizer,
            X_num,
            X_cat,
            y_array,
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
# 17. LOAD COMPETITION DATA
# ============================================================

print("\n" + "=" * 100)
print("LOADING DATA")
print("=" * 100)

assert_file_exists(TRAIN_PATH, "train.csv")
assert_file_exists(TEST_PATH, "test.csv")
assert_file_exists(SAMPLE_PATH, "sample_submission.csv")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

required_train_cols = (
    [ID_COL, TARGET]
    + NUMERIC_COLS
    + CATEGORICAL_COLS
)

required_test_cols = (
    [ID_COL]
    + NUMERIC_COLS
    + CATEGORICAL_COLS
)

missing_train = set(required_train_cols) - set(train.columns)
missing_test = set(required_test_cols) - set(test.columns)

if missing_train:
    raise ValueError(
        f"Missing train columns: {missing_train}"
    )

if missing_test:
    raise ValueError(
        f"Missing test columns: {missing_test}"
    )

train_ids = pd.to_numeric(
    train[ID_COL],
    errors="raise",
).to_numpy()

test_ids = pd.to_numeric(
    test[ID_COL],
    errors="raise",
).to_numpy()

y = train[TARGET].astype(np.int8).to_numpy()

if len(np.unique(train_ids)) != len(train_ids):
    raise ValueError("Duplicate train IDs.")

if len(np.unique(test_ids)) != len(test_ids):
    raise ValueError("Duplicate test IDs.")

if len(np.unique(y)) != 2:
    raise ValueError("Expected binary target.")

check_same_ids(
    test_ids,
    sample[ID_COL],
    "test vs sample_submission",
)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Target mean:", f"{y.mean():.6f}")


# ============================================================
# 18. BUILD RAW FEATURE VIEWS
# ============================================================

print("\n" + "=" * 100)
print("BUILDING FEATURE VIEWS")
print("=" * 100)

raw_train = build_raw_views(train)
raw_test = build_raw_views(test)

print("Original predictors: 21")
print("Numeric-as-categorical copies: 17")
print("Count features: 5")
print("Interaction categorical features: 3")
print("Target encodings: 3")
print("TabM numerical inputs: 25")
print("TabM categorical inputs: 24")
print("Total inputs: 49")


# ============================================================
# 19. OUTER 5-FOLD OOF TRAINING
# ============================================================

print("\n" + "=" * 100)
print("STARTING FEATURE-ENHANCED TABM 5-FOLD TRAINING")
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
    skf.split(np.zeros(len(y)), y),
    start=1,
):
    print("\n" + "#" * 100)
    print(f"OUTER FOLD {fold}/{N_FOLDS}")
    print("#" * 100)

    # --------------------------------------------------------
    # Stage A: choose the epoch using an inner split.
    # --------------------------------------------------------

    inner_fit_idx, inner_valid_idx = train_test_split(
        outer_train_idx,
        test_size=INNER_VALID_SIZE,
        random_state=SEED + fold,
        stratify=y[outer_train_idx],
    )

    raw_inner_fit = subset_raw_views(
        raw_train,
        inner_fit_idx,
    )

    raw_inner_valid = subset_raw_views(
        raw_train,
        inner_valid_idx,
    )

    y_inner_fit = y[inner_fit_idx]
    y_inner_valid = y[inner_valid_idx]

    (
        inner_pipeline,
        X_inner_fit_num,
        X_inner_fit_cat,
    ) = fit_feature_pipeline(
        raw_inner_fit,
        y_inner_fit,
        seed=SEED + fold,
    )

    (
        X_inner_valid_num,
        X_inner_valid_cat,
    ) = transform_features(
        raw_inner_valid,
        inner_pipeline,
    )

    inner_cat_cardinalities = get_cat_cardinalities(
        inner_pipeline
    )

    inner_model = create_tabm_model(
        X_inner_fit_num,
        inner_cat_cardinalities,
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
        f"best inner epoch={best_epoch}, "
        f"inner AUC={inner_best_auc:.9f}"
    )

    # Free inner model memory.
    del inner_model
    del X_inner_fit_num
    del X_inner_fit_cat
    del X_inner_valid_num
    del X_inner_valid_cat
    del inner_pipeline

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # --------------------------------------------------------
    # Stage B: refit on the complete outer training fold.
    # Training TE remains cross-fitted.
    # --------------------------------------------------------

    raw_outer_train = subset_raw_views(
        raw_train,
        outer_train_idx,
    )

    raw_outer_valid = subset_raw_views(
        raw_train,
        outer_valid_idx,
    )

    y_outer_train = y[outer_train_idx]
    y_outer_valid = y[outer_valid_idx]

    (
        outer_pipeline,
        X_outer_train_num,
        X_outer_train_cat,
    ) = fit_feature_pipeline(
        raw_outer_train,
        y_outer_train,
        seed=SEED + 1000 + fold,
    )

    (
        X_outer_valid_num,
        X_outer_valid_cat,
    ) = transform_features(
        raw_outer_valid,
        outer_pipeline,
    )

    (
        X_test_num,
        X_test_cat,
    ) = transform_features(
        raw_test,
        outer_pipeline,
    )

    outer_cat_cardinalities = get_cat_cardinalities(
        outer_pipeline
    )

    final_model = create_tabm_model(
        X_outer_train_num,
        outer_cat_cardinalities,
        seed=SEED + 2000 + fold,
    )

    train_fixed_epochs(
        model=final_model,
        X_num=X_outer_train_num,
        X_cat=X_outer_train_cat,
        y_array=y_outer_train,
        n_epochs=best_epoch,
        fold=fold,
    )

    # OOF predictions for this outer validation fold.
    fold_oof_pred = predict_probability(
        final_model,
        X_outer_valid_num,
        X_outer_valid_cat,
    )

    fold_auc = roc_auc_score(
        y_outer_valid,
        fold_oof_pred,
    )

    oof_prediction[outer_valid_idx] = fold_oof_pred

    # Average test predictions across outer-fold models.
    fold_test_pred = predict_probability(
        final_model,
        X_test_num,
        X_test_cat,
    )

    test_prediction_sum += (
        fold_test_pred / N_FOLDS
    )

    fold_metrics.append({
        "fold": fold,
        "outer_train_rows": len(outer_train_idx),
        "outer_valid_rows": len(outer_valid_idx),
        "best_epoch": best_epoch,
        "inner_best_auc": inner_best_auc,
        "outer_oof_auc": fold_auc,
    })

    print("\n" + "-" * 80)
    print(f"Fold {fold} OOF AUC = {fold_auc:.9f}")
    print(f"Fold {fold} selected epoch = {best_epoch}")
    print("-" * 80)

    # Free outer fold memory.
    del final_model
    del X_outer_train_num
    del X_outer_train_cat
    del X_outer_valid_num
    del X_outer_valid_cat
    del X_test_num
    del X_test_cat
    del outer_pipeline
    del raw_outer_train
    del raw_outer_valid

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 20. TABM OOF RESULTS
# ============================================================

if np.isnan(oof_prediction).any():
    raise ValueError("TabM OOF has missing rows.")

if not np.isfinite(oof_prediction).all():
    raise ValueError("TabM OOF contains non-finite values.")

if not np.isfinite(test_prediction_sum).all():
    raise ValueError("TabM test predictions contain non-finite values.")

tabm_oof_auc = roc_auc_score(
    y,
    oof_prediction,
)

fold_metrics_df = pd.DataFrame(fold_metrics)

fold_auc_mean = fold_metrics_df[
    "outer_oof_auc"
].mean()

fold_auc_std = fold_metrics_df[
    "outer_oof_auc"
].std(ddof=1)

print("\n" + "=" * 100)
print("FEATURE-ENHANCED TABM OOF RESULT")
print("=" * 100)

print(f"TabM OOF AUC = {tabm_oof_auc:.9f}")
print(f"Fold AUC mean = {fold_auc_mean:.9f}")
print(f"Fold AUC std  = {fold_auc_std:.9f}")


# ============================================================
# 21. SAVE TABM PREDICTIONS
# ============================================================

tabm_oof_df = pd.DataFrame({
    ID_COL: train_ids,
    TARGET: y,
    "prediction": oof_prediction,
})

tabm_test_df = pd.DataFrame({
    ID_COL: test_ids,
    "prediction": test_prediction_sum,
})

tabm_oof_path = os.path.join(
    OUTPUT_DIR,
    "tabm_v1_features_oof.csv",
)

tabm_test_path = os.path.join(
    OUTPUT_DIR,
    "tabm_v1_features_test.csv",
)

fold_metrics_path = os.path.join(
    OUTPUT_DIR,
    "tabm_v1_features_fold_metrics.csv",
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

fold_metrics_df.to_csv(
    fold_metrics_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 22. LOAD CURRENT CHAMPION PREDICTIONS
# ============================================================

print("\n" + "=" * 100)
print("LOADING CURRENT CHAMPION PREDICTIONS")
print("=" * 100)


def load_reference_oof(path, name):
    assert_file_exists(path, name)

    df = pd.read_csv(path)

    required = {ID_COL, TARGET, "prediction"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"{name}: missing required columns.\n"
            f"Found: {df.columns.tolist()}"
        )

    df[ID_COL] = pd.to_numeric(
        df[ID_COL],
        errors="raise",
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
        df.set_index(ID_COL)
        .loc[train_ids]
        .reset_index()
    )

    if not np.array_equal(
        aligned[TARGET].to_numpy().astype(np.int8),
        y,
    ):
        raise ValueError(
            f"{name}: target does not align with train.csv."
        )

    pred = aligned["prediction"].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(pred).all():
        raise ValueError(
            f"{name}: non-finite predictions."
        )

    print(
        f"{name:35s} AUC = "
        f"{roc_auc_score(y, pred):.9f}"
    )

    return pred


def load_reference_test(path, name):
    assert_file_exists(path, name)

    df = pd.read_csv(path)

    required = {ID_COL, "prediction"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"{name}: missing required columns.\n"
            f"Found: {df.columns.tolist()}"
        )

    df[ID_COL] = pd.to_numeric(
        df[ID_COL],
        errors="raise",
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
        df.set_index(ID_COL)
        .loc[test_ids]
        .reset_index()
    )

    pred = aligned["prediction"].to_numpy(
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
# 23. REBUILD CURRENT CHAMPION
# ============================================================

champion_rank_oof = (
    0.40 * normalized_rank(current_lgbm_oof)
    +
    0.60 * normalized_rank(v1_oof)
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
    champion_oof_auc - KNOWN_CHAMPION_OOF
) > 0.00002:
    raise ValueError(
        "Champion OOF mismatch. "
        "Check current LGBM and RealMLP prediction files."
    )


# ============================================================
# 24. TABM / CHAMPION DIVERSITY
# ============================================================

tabm_rank_oof = normalized_rank(
    oof_prediction
)

champion_tabm_corr = pd.Series(
    champion_rank_oof
).corr(
    pd.Series(tabm_rank_oof),
    method="spearman",
)

print("\n" + "=" * 100)
print("FEATURE-ENHANCED TABM DIVERSITY CHECK")
print("=" * 100)

print(
    "Spearman(TabM rank, Champion rank) = "
    f"{champion_tabm_corr:.9f}"
)


# ============================================================
# 25. TABM + CHAMPION RANK FUSION
# ============================================================

blend_results = []

for tabm_weight in TABM_BLEND_WEIGHTS:
    champion_weight = 1.0 - tabm_weight

    blend_oof = (
        champion_weight * champion_rank_oof
        +
        tabm_weight * tabm_rank_oof
    )

    blend_auc = roc_auc_score(
        y,
        blend_oof,
    )

    blend_results.append({
        "tabm_weight": tabm_weight,
        "champion_weight": champion_weight,
        "oof_auc": blend_auc,
        "gain_vs_champion": blend_auc - champion_oof_auc,
    })


blend_results_df = (
    pd.DataFrame(blend_results)
    .sort_values(
        "oof_auc",
        ascending=False,
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 100)
print("TABM V1 FEATURES + CHAMPION RANK BLEND")
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
    "tabm_v1_features_champion_blend_candidates.csv",
)

blend_results_df.to_csv(
    blend_candidates_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 26. SELECT BEST OOF CANDIDATE
# ============================================================

best_row = blend_results_df.iloc[0]

best_tabm_weight = float(
    best_row["tabm_weight"]
)

best_champion_weight = 1.0 - best_tabm_weight

best_oof_blend = (
    best_champion_weight * champion_rank_oof
    +
    best_tabm_weight * tabm_rank_oof
)

best_oof_auc = roc_auc_score(
    y,
    best_oof_blend,
)

gain_vs_champion = (
    best_oof_auc - champion_oof_auc
)

print("\n" + "=" * 100)
print("BEST OOF CANDIDATE")
print("=" * 100)

print(f"Champion weight = {best_champion_weight:.3f}")
print(f"TabM weight     = {best_tabm_weight:.3f}")
print(f"OOF AUC         = {best_oof_auc:.9f}")
print(f"Gain vs Champion = {gain_vs_champion:+.9f}")

if gain_vs_champion >= MIN_MEANINGFUL_GAIN:
    print("\nDecision: promising OOF improvement.")
    print(
        "Consider one controlled Public LB test, "
        "but OOF gain alone is not proof of improvement."
    )
else:
    print("\nDecision: no convincing OOF improvement.")
    print("Keep the existing Champion as the official best model.")


# ============================================================
# 27. TEST RANKS AND FUSION
# ============================================================

tabm_rank_test = normalized_rank(
    test_prediction_sum
)

champion_rank_test = (
    0.40 * normalized_rank(current_lgbm_test)
    +
    0.60 * normalized_rank(v1_test)
)

best_test_blend = (
    best_champion_weight * champion_rank_test
    +
    best_tabm_weight * tabm_rank_test
)


# ============================================================
# 28. CREATE SUBMISSIONS
# ============================================================

submission_targets = [
    col
    for col in sample.columns
    if col != ID_COL
]

if len(submission_targets) != 1:
    raise ValueError(
        "Expected exactly one submission target column."
    )

SUBMISSION_TARGET = submission_targets[0]


def make_submission(prediction, filename):
    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    )

    if len(prediction) != len(test):
        raise ValueError(
            "Submission prediction length mismatch."
        )

    if not np.isfinite(prediction).all():
        raise ValueError(
            "Submission contains non-finite predictions."
        )

    pred_df = pd.DataFrame({
        ID_COL: test_ids,
        "prediction": prediction,
    })

    check_same_ids(
        sample[ID_COL],
        pred_df[ID_COL],
        "sample_submission vs prediction",
    )

    submission = (
        sample[[ID_COL]]
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
            "Submission contains missing predictions."
        )

    submission[SUBMISSION_TARGET] = (
        submission["prediction"].astype(np.float64)
    )

    submission = submission[
        [ID_COL, SUBMISSION_TARGET]
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
            "Submission ID order differs from sample_submission."
        )

    if not np.isfinite(
        submission[SUBMISSION_TARGET].to_numpy(
            dtype=np.float64
        )
    ).all():
        raise ValueError(
            "Submission has non-finite target values."
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

    print("Saved:", output_path)

    return output_path


champion_submission_path = make_submission(
    champion_rank_test,
    "submission_champion_rank_4060.csv",
)

tabm_submission_path = make_submission(
    tabm_rank_test,
    "submission_tabm_v1_features_rank.csv",
)

best_blend_submission_path = make_submission(
    best_test_blend,
    "submission_tabm_v1_features_plus_champion_best_oof.csv",
)


# ============================================================
# 29. SAVE COMBINED OOF / TEST
# ============================================================

oof_output = pd.DataFrame({
    ID_COL: train_ids,
    TARGET: y,
    "tabm_prediction": oof_prediction,
    "tabm_rank": tabm_rank_oof,
    "champion_rank": champion_rank_oof,
    "best_blend_rank": best_oof_blend,
})

oof_output_path = os.path.join(
    OUTPUT_DIR,
    "tabm_v1_features_champion_oof.csv",
)

oof_output.to_csv(
    oof_output_path,
    index=False,
    encoding="utf-8-sig",
)


test_output = pd.DataFrame({
    ID_COL: test_ids,
    "tabm_prediction": test_prediction_sum,
    "tabm_rank": tabm_rank_test,
    "champion_rank": champion_rank_test,
    "best_blend_rank": best_test_blend,
})

test_output_path = os.path.join(
    OUTPUT_DIR,
    "tabm_v1_features_champion_test.csv",
)

test_output.to_csv(
    test_output_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 30. FINAL SUMMARY
# ============================================================

summary_df = pd.DataFrame([
    {
        "model": "Feature-enhanced TabM",
        "oof_auc": tabm_oof_auc,
        "gain_vs_champion": tabm_oof_auc - champion_oof_auc,
        "tabm_weight": 1.0,
        "champion_weight": 0.0,
        "rank_correlation_with_champion": champion_tabm_corr,
    },
    {
        "model": "Current Champion",
        "oof_auc": champion_oof_auc,
        "gain_vs_champion": 0.0,
        "tabm_weight": 0.0,
        "champion_weight": 1.0,
        "rank_correlation_with_champion": 1.0,
    },
    {
        "model": "Best OOF-selected TabM + Champion",
        "oof_auc": best_oof_auc,
        "gain_vs_champion": best_oof_auc - champion_oof_auc,
        "tabm_weight": best_tabm_weight,
        "champion_weight": best_champion_weight,
        "rank_correlation_with_champion": champion_tabm_corr,
    },
])

summary_path = os.path.join(
    OUTPUT_DIR,
    "tabm_v1_features_summary.csv",
)

summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 100)
print("FINAL FEATURE-ENHANCED TABM SUMMARY")
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


# ============================================================
# 31. SAVED FILES
# ============================================================

print("\n" + "=" * 100)
print("SAVED FILES")
print("=" * 100)

for label, path in [
    ("TabM OOF", tabm_oof_path),
    ("TabM TEST", tabm_test_path),
    ("Fold metrics", fold_metrics_path),
    ("Blend candidates", blend_candidates_path),
    ("Combined OOF", oof_output_path),
    ("Combined TEST", test_output_path),
    ("Summary", summary_path),
    ("Champion submission", champion_submission_path),
    ("TabM submission", tabm_submission_path),
    ("Experimental blend submission", best_blend_submission_path),
]:
    print(f"\n{label}:\n{path}")

print("\n" + "=" * 100)
print("TABM V1 FEATURE-ENHANCED EXPERIMENT COMPLETE")
print("=" * 100)