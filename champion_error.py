# ============================================================
# CHAMPION ERROR ANALYSIS + CANDIDATE FEATURE MINING
#
# Competition: Kaggle Airline Passenger Satisfaction
#
# Current Champion:
#   Conditional-TE LGBM 40% + RealMLP V1 60%
#   Rank fusion
#   OOF AUC = 0.961278687
#   Public LB = 0.96097
#
# This script:
#   1. Loads existing OOF predictions and aligns by ID.
#   2. Reconstructs the current Champion exactly.
#   3. Finds high-rank negatives and low-rank positives.
#   4. Reports subgroup AUC and hard-case enrichment.
#   5. Analyzes disagreement between LGBM and RealMLP.
#   6. Constructs candidate features using competition data only.
#   7. Saves diagnostics and candidate train/test feature files.
#
# No model training.
# No external 129K data.
# No external labels.
# No modification of existing Champion prediction files.
#
# Outputs:
#   D:\kaggle\airline\champion_error_analysis\
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, rankdata

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS AND SETTINGS
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

CURRENT_LGBM_OOF_PATH = (
    r"D:\kaggle\airline\conditional_te_experiment"
    r"\conditional_te_lgbm_oof.csv"
)

V1_OOF_PATH = (
    r"D:\kaggle\airline\lean_realmlp_blend"
    r"\lean_realmlp_oof.csv"
)

OUTPUT_DIR = (
    r"D:\kaggle\airline\champion_error_analysis"
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

KNOWN_CHAMPION_OOF = 0.961278687

LGBM_WEIGHT = 0.40
REALMLP_WEIGHT = 0.60

# AUC in very small subgroups is noisy.
MIN_GROUP_ROWS = 500
MIN_GROUP_PER_CLASS = 50

# Hard-case definitions based on global Champion rank.
TOP_RANK_CUTOFF = 0.90
BOTTOM_RANK_CUTOFF = 0.10


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

DIGITAL_COLS = [
    "Inflight wifi service",
    "Ease of Online booking",
    "Online boarding",
]

COMFORT_COLS = [
    "Food and drink",
    "Seat comfort",
    "Inflight entertainment",
    "Cleanliness",
]

ONBOARD_COLS = [
    "On-board service",
    "Leg room service",
    "Baggage handling",
]

AIRPORT_COLS = [
    "Departure/Arrival time convenient",
    "Gate location",
    "Checkin service",
]


# ============================================================
# 4. GENERAL HELPERS
# ============================================================

def assert_file_exists(path, name):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )


def check_same_ids(
    reference_ids,
    candidate_ids,
    name,
):
    """
    Compare IDs safely.

    Accepts:
        pandas Series
        NumPy arrays
        Python lists

    FIX:
        pd.to_numeric() may return a NumPy array when the input
        is an array. NumPy arrays do not have .to_numpy().
        Converting the inputs to Series first avoids the error.
    """

    reference_ids = np.asarray(
        pd.to_numeric(
            pd.Series(reference_ids),
            errors="raise",
        )
    )

    candidate_ids = np.asarray(
        pd.to_numeric(
            pd.Series(candidate_ids),
            errors="raise",
        )
    )

    if pd.isna(reference_ids).any():
        raise ValueError(
            f"{name}: reference IDs contain missing values."
        )

    if pd.isna(candidate_ids).any():
        raise ValueError(
            f"{name}: candidate IDs contain missing values."
        )

    if len(np.unique(reference_ids)) != len(reference_ids):
        raise ValueError(
            f"{name}: duplicate reference IDs."
        )

    if len(np.unique(candidate_ids)) != len(candidate_ids):
        raise ValueError(
            f"{name}: duplicate candidate IDs."
        )

    reference_set = set(reference_ids.tolist())
    candidate_set = set(candidate_ids.tolist())

    if reference_set != candidate_set:

        missing = reference_set - candidate_set
        extra = candidate_set - reference_set

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

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite prediction values."
        )

    if len(values) == 0:
        raise ValueError(
            "Cannot rank an empty array."
        )

    return (
        rankdata(
            values,
            method="average",
        )
        / len(values)
    )


def clean_category(series):
    return (
        series
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
        .str.strip()
    )


def numeric_series(df, col):
    return (
        pd.to_numeric(
            df[col],
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )


def safe_auc(y, score):
    """
    Calculate AUC while excluding non-finite scores.
    """

    y = np.asarray(y)
    score = np.asarray(score)

    mask = np.isfinite(score)

    y = y[mask]
    score = score[mask]

    if len(y) == 0:
        return np.nan

    if len(np.unique(y)) < 2:
        return np.nan

    return roc_auc_score(
        y,
        score,
    )


def safe_spearman(a, b):
    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    mask = (
        np.isfinite(a)
        & np.isfinite(b)
    )

    if mask.sum() < 3:
        return np.nan

    return spearmanr(
        a[mask],
        b[mask],
    ).statistic


def standardized_mean_difference(a, b):
    """
    SMD = (mean(a) - mean(b)) / pooled standard deviation.

    Positive means group a has a larger mean than group b.
    """

    a = pd.to_numeric(
        pd.Series(a),
        errors="coerce",
    ).dropna().to_numpy(
        dtype=np.float64
    )

    b = pd.to_numeric(
        pd.Series(b),
        errors="coerce",
    ).dropna().to_numpy(
        dtype=np.float64
    )

    if len(a) < 2 or len(b) < 2:
        return np.nan

    va = np.var(
        a,
        ddof=1,
    )

    vb = np.var(
        b,
        ddof=1,
    )

    pooled = np.sqrt(
        (va + vb) / 2.0
    )

    if pooled == 0 or not np.isfinite(pooled):
        return np.nan

    return (
        np.mean(a) - np.mean(b)
    ) / pooled


# ============================================================
# 5. LOAD DATA
# ============================================================

print("=" * 100)
print("CHAMPION ERROR ANALYSIS")
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
    CURRENT_LGBM_OOF_PATH,
    "Current Conditional-TE LGBM OOF",
)

assert_file_exists(
    V1_OOF_PATH,
    "Lean RealMLP V1 OOF",
)


train = pd.read_csv(
    TRAIN_PATH,
)

test = pd.read_csv(
    TEST_PATH,
)


# Validate columns.

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

missing_train = (
    set(required_train_cols)
    - set(train.columns)
)

missing_test = (
    set(required_test_cols)
    - set(test.columns)
)

if missing_train:
    raise ValueError(
        f"Training data is missing columns: {missing_train}"
    )

if missing_test:
    raise ValueError(
        f"Test data is missing columns: {missing_test}"
    )


train_ids = pd.to_numeric(
    train[ID_COL],
    errors="raise",
).to_numpy()

test_ids = pd.to_numeric(
    test[ID_COL],
    errors="raise",
).to_numpy()

y = train[TARGET].astype(
    np.int8
).to_numpy()


if len(np.unique(train_ids)) != len(train_ids):
    raise ValueError(
        "Train IDs are duplicated."
    )

if len(np.unique(test_ids)) != len(test_ids):
    raise ValueError(
        "Test IDs are duplicated."
    )

if len(np.unique(y)) != 2:
    raise ValueError(
        "Expected a binary classification target."
    )


print("Train shape:", train.shape)
print("Test shape :", test.shape)
print("Target mean:", f"{y.mean():.6f}")


# ============================================================
# 6. LOAD AND ALIGN OOF PREDICTIONS
# ============================================================

def load_oof(path, name):
    """
    Load OOF predictions and align them to train.csv by ID.

    The function does not assume that the CSV row order matches
    train.csv.
    """

    df = pd.read_csv(path)

    required = {
        ID_COL,
        TARGET,
        "prediction",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"{name}: expected columns {sorted(required)}, "
            f"found {df.columns.tolist()}"
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

    df[ID_COL] = pd.to_numeric(
        df[ID_COL],
        errors="raise",
    )

    # Align by ID, not row order.
    aligned = (
        df
        .set_index(ID_COL)
        .loc[train_ids]
        .reset_index()
    )

    aligned_y = (
        aligned[TARGET]
        .to_numpy()
        .astype(np.int8)
    )

    if not np.array_equal(
        aligned_y,
        y,
    ):
        raise ValueError(
            f"{name}: target labels do not align with train.csv."
        )

    prediction = aligned[
        "prediction"
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(prediction).all():
        raise ValueError(
            f"{name}: predictions contain non-finite values."
        )

    # Prevent accidental target leakage.
    if np.array_equal(
        prediction,
        y.astype(np.float64),
    ):
        raise ValueError(
            f"{name}: prediction exactly equals target; "
            "possible leakage."
        )

    return prediction


print("\n" + "=" * 100)
print("LOADING OOF PREDICTIONS")
print("=" * 100)


lgbm_oof = load_oof(
    CURRENT_LGBM_OOF_PATH,
    "Current Conditional-TE LGBM OOF",
)

realmlp_oof = load_oof(
    V1_OOF_PATH,
    "Lean RealMLP V1 OOF",
)


lgbm_auc = roc_auc_score(
    y,
    lgbm_oof,
)

realmlp_auc = roc_auc_score(
    y,
    realmlp_oof,
)


lgbm_rank = normalized_rank(
    lgbm_oof,
)

realmlp_rank = normalized_rank(
    realmlp_oof,
)


# Reconstruct the current Champion exactly.
champion_rank = (
    LGBM_WEIGHT * lgbm_rank
    + REALMLP_WEIGHT * realmlp_rank
)

champion_auc = roc_auc_score(
    y,
    champion_rank,
)


rank_disagreement = np.abs(
    lgbm_rank - realmlp_rank
)


print(
    f"Current LGBM OOF AUC: {lgbm_auc:.9f}"
)

print(
    f"RealMLP V1 OOF AUC : {realmlp_auc:.9f}"
)

print(
    f"Champion OOF AUC    : {champion_auc:.9f}"
)

print(
    f"Known Champion OOF  : {KNOWN_CHAMPION_OOF:.9f}"
)

print(
    f"Champion difference: "
    f"{champion_auc - KNOWN_CHAMPION_OOF:+.9f}"
)

print(
    "Spearman(LGBM, RealMLP): "
    f"{safe_spearman(lgbm_rank, realmlp_rank):.6f}"
)


if abs(
    champion_auc - KNOWN_CHAMPION_OOF
) > 0.00002:

    raise ValueError(
        "Champion mismatch. Stop before interpreting errors."
    )


# ============================================================
# 7. DEFINE HARD CASES
# ============================================================

# AUC is a ranking metric.
#
# High-rank negative:
#   satisfaction=0, but Champion rank is in the global top 10%.
#
# Low-rank positive:
#   satisfaction=1, but Champion rank is in the global bottom 10%.
#
# These are diagnostic definitions, not threshold-based
# classification errors.

top_cutoff = np.quantile(
    champion_rank,
    TOP_RANK_CUTOFF,
)

bottom_cutoff = np.quantile(
    champion_rank,
    BOTTOM_RANK_CUTOFF,
)


top_rank = champion_rank >= top_cutoff

bottom_rank = champion_rank <= bottom_cutoff


high_rank_negative = (
    (y == 0)
    & top_rank
)

high_rank_positive = (
    (y == 1)
    & top_rank
)

low_rank_positive = (
    (y == 1)
    & bottom_rank
)

low_rank_negative = (
    (y == 0)
    & bottom_rank
)


print("\n" + "=" * 100)
print("HARD-CASE COUNTS")
print("=" * 100)

print(
    "High-rank negatives:",
    int(high_rank_negative.sum()),
)

print(
    "High-rank positives:",
    int(high_rank_positive.sum()),
)

print(
    "Low-rank positives :",
    int(low_rank_positive.sum()),
)

print(
    "Low-rank negatives :",
    int(low_rank_negative.sum()),
)


# ============================================================
# 8. SAVE ROW-LEVEL OOF DIAGNOSTICS
# ============================================================

diagnostics = pd.DataFrame({
    ID_COL: train_ids,
    TARGET: y,
    "lgbm_prediction": lgbm_oof,
    "realmlp_v1_prediction": realmlp_oof,
    "lgbm_rank": lgbm_rank,
    "realmlp_v1_rank": realmlp_rank,
    "champion_rank": champion_rank,
    "rank_disagreement": rank_disagreement,
    "top_rank_10pct": top_rank,
    "bottom_rank_10pct": bottom_rank,
    "high_rank_negative": high_rank_negative,
    "high_rank_positive": high_rank_positive,
    "low_rank_positive": low_rank_positive,
    "low_rank_negative": low_rank_negative,
})


diagnostics_path = os.path.join(
    OUTPUT_DIR,
    "champion_oof_diagnostics.csv",
)

diagnostics.to_csv(
    diagnostics_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 9. FEATURE ENGINEERING
# ============================================================

def build_candidate_features(df):
    """
    Construct candidate features from X only.

    No target labels, OOF scores, or external data are used.

    Candidate groups:
        1. Service-rating aggregates
        2. Service subdomain summaries
        3. Service-rating gaps
        4. Delay transformations and missingness
        5. Age/distance/delay bins
        6. Segment interaction categories
    """

    out = pd.DataFrame(
        index=df.index
    )


    # --------------------------------------------------------
    # A. Service-rating aggregates
    # --------------------------------------------------------

    service = pd.DataFrame({
        col: numeric_series(df, col)
        for col in SERVICE_COLS
    })


    out["service_mean"] = service.mean(
        axis=1
    )

    out["service_std"] = service.std(
        axis=1,
        ddof=0,
    )

    out["service_min"] = service.min(
        axis=1
    )

    out["service_max"] = service.max(
        axis=1
    )

    out["service_range"] = (
        out["service_max"]
        - out["service_min"]
    )

    out["service_low_count_le1"] = (
        service.le(1).sum(axis=1)
    )

    out["service_low_count_le2"] = (
        service.le(2).sum(axis=1)
    )

    out["service_high_count_ge4"] = (
        service.ge(4).sum(axis=1)
    )

    out["service_perfect_count_eq5"] = (
        service.eq(5).sum(axis=1)
    )

    out["service_missing_count"] = (
        service.isna().sum(axis=1)
    )


    # --------------------------------------------------------
    # B. Service subdomains
    # --------------------------------------------------------

    digital = df[
        DIGITAL_COLS
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    comfort = df[
        COMFORT_COLS
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    onboard = df[
        ONBOARD_COLS
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    airport = df[
        AIRPORT_COLS
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )


    out["digital_rating_mean"] = (
        digital.mean(axis=1)
    )

    out["digital_rating_std"] = (
        digital.std(axis=1, ddof=0)
    )

    out["comfort_rating_mean"] = (
        comfort.mean(axis=1)
    )

    out["comfort_rating_std"] = (
        comfort.std(axis=1, ddof=0)
    )

    out["onboard_rating_mean"] = (
        onboard.mean(axis=1)
    )

    out["onboard_rating_std"] = (
        onboard.std(axis=1, ddof=0)
    )

    out["airport_rating_mean"] = (
        airport.mean(axis=1)
    )

    out["airport_rating_std"] = (
        airport.std(axis=1, ddof=0)
    )


    # --------------------------------------------------------
    # C. Within-row service inconsistencies
    # --------------------------------------------------------

    out["gap_boarding_vs_booking"] = (
        numeric_series(df, "Online boarding")
        - numeric_series(df, "Ease of Online booking")
    )

    out["gap_wifi_vs_booking"] = (
        numeric_series(df, "Inflight wifi service")
        - numeric_series(df, "Ease of Online booking")
    )

    out["gap_entertainment_vs_seat"] = (
        numeric_series(df, "Inflight entertainment")
        - numeric_series(df, "Seat comfort")
    )

    out["gap_onboard_vs_legroom"] = (
        numeric_series(df, "On-board service")
        - numeric_series(df, "Leg room service")
    )

    out["gap_baggage_vs_checkin"] = (
        numeric_series(df, "Baggage handling")
        - numeric_series(df, "Checkin service")
    )

    out["gap_digital_vs_comfort"] = (
        out["digital_rating_mean"]
        - out["comfort_rating_mean"]
    )

    out["gap_service_vs_digital"] = (
        out["service_mean"]
        - out["digital_rating_mean"]
    )

    out["gap_service_vs_onboard"] = (
        out["service_mean"]
        - out["onboard_rating_mean"]
    )


    # --------------------------------------------------------
    # D. Delays, nonlinearity, and missingness
    # --------------------------------------------------------

    dep = numeric_series(
        df,
        "Departure Delay in Minutes",
    )

    arr = numeric_series(
        df,
        "Arrival Delay in Minutes",
    )


    # Preserve arrival-delay missingness explicitly.
    out["arrival_delay_missing"] = (
        arr.isna().astype(np.int8)
    )


    # Zero-filled delay variables are used together with
    # explicit missingness indicators.
    dep0 = dep.fillna(0).clip(lower=0)
    arr0 = arr.fillna(0).clip(lower=0)


    out["departure_delay_log1p"] = np.log1p(
        dep0
    )

    out["arrival_delay_log1p"] = np.log1p(
        arr0
    )

    out["delay_sum_log1p"] = np.log1p(
        dep0 + arr0
    )

    out["delay_max_log1p"] = np.log1p(
        pd.concat(
            [dep0, arr0],
            axis=1,
        ).max(axis=1)
    )

    out["delay_abs_difference_log1p"] = np.log1p(
        (arr0 - dep0).abs()
    )

    out["arrival_minus_departure_delay_fill0"] = (
        arr0 - dep0
    )


    out["departure_delay_gt15"] = (
        dep0 > 15
    ).astype(np.int8)

    out["arrival_delay_gt15"] = (
        arr0 > 15
    ).astype(np.int8)

    out["any_delay_gt15"] = (
        (dep0 > 15)
        | (arr0 > 15)
    ).astype(np.int8)

    out["both_delays_gt15"] = (
        (dep0 > 15)
        & (arr0 > 15)
    ).astype(np.int8)

    out["arrival_missing_and_departure_delayed"] = (
        arr.isna()
        & (dep0 > 15)
    ).astype(np.int8)


    # --------------------------------------------------------
    # E. Age, distance, service, and delay bands
    # --------------------------------------------------------

    age = numeric_series(
        df,
        "Age",
    )

    distance = numeric_series(
        df,
        "Flight Distance",
    )


    out["age_band"] = pd.cut(
        age,
        bins=[
            -np.inf,
            18,
            25,
            35,
            45,
            55,
            65,
            np.inf,
        ],
        labels=[
            "under18",
            "18_25",
            "26_35",
            "36_45",
            "46_55",
            "56_65",
            "over65",
        ],
    ).astype(
        "string"
    ).fillna(
        "MISSING"
    ).astype(str)


    out["distance_band"] = pd.cut(
        distance,
        bins=[
            -np.inf,
            500,
            1000,
            1500,
            2000,
            3000,
            np.inf,
        ],
        labels=[
            "le500",
            "501_1000",
            "1001_1500",
            "1501_2000",
            "2001_3000",
            "over3000",
        ],
    ).astype(
        "string"
    ).fillna(
        "MISSING"
    ).astype(str)


    out["service_mean_band"] = pd.cut(
        out["service_mean"],
        bins=[
            -np.inf,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            np.inf,
        ],
        labels=[
            "very_low",
            "low",
            "below_average",
            "average",
            "above_average",
            "high",
            "very_high",
        ],
    ).astype(
        "string"
    ).fillna(
        "MISSING"
    ).astype(str)


    out["service_low_count_band"] = pd.cut(
        out["service_low_count_le2"],
        bins=[
            -0.5,
            0.5,
            1.5,
            2.5,
            4.5,
            7.5,
            13.5,
        ],
        labels=[
            "none",
            "one",
            "two",
            "three_four",
            "five_seven",
            "eight_plus",
        ],
    ).astype(
        "string"
    ).fillna(
        "MISSING"
    ).astype(str)


    out["departure_delay_band"] = pd.cut(
        dep0,
        bins=[
            -np.inf,
            0,
            5,
            15,
            30,
            60,
            120,
            np.inf,
        ],
        labels=[
            "none",
            "1_5",
            "6_15",
            "16_30",
            "31_60",
            "61_120",
            "over120",
        ],
    ).astype(
        "string"
    ).fillna(
        "MISSING"
    ).astype(str)


    out["arrival_delay_band"] = pd.cut(
        arr0,
        bins=[
            -np.inf,
            0,
            5,
            15,
            30,
            60,
            120,
            np.inf,
        ],
        labels=[
            "none",
            "1_5",
            "6_15",
            "16_30",
            "31_60",
            "61_120",
            "over120",
        ],
    ).astype(
        "string"
    ).fillna(
        "MISSING"
    ).astype(str)


    # --------------------------------------------------------
    # F. Segment interaction categories
    # --------------------------------------------------------

    travel = clean_category(
        df["Type of Travel"]
    )

    customer = clean_category(
        df["Customer Type"]
    )

    class_value = clean_category(
        df["Class"]
    )


    out["age_band__travel"] = (
        out["age_band"]
        + "__"
        + travel
    )

    out["age_band__class"] = (
        out["age_band"]
        + "__"
        + class_value
    )

    out["distance_band__travel"] = (
        out["distance_band"]
        + "__"
        + travel
    )

    out["distance_band__class"] = (
        out["distance_band"]
        + "__"
        + class_value
    )

    out["service_mean_band__travel"] = (
        out["service_mean_band"]
        + "__"
        + travel
    )

    out["low_service_count__travel"] = (
        out["service_low_count_band"]
        + "__"
        + travel
    )

    out["low_service_count__customer"] = (
        out["service_low_count_band"]
        + "__"
        + customer
    )


    out["delay_status"] = np.select(
        [
            (dep0 <= 15) & (arr0 <= 15),
            (dep0 > 15) & (arr0 <= 15),
            (dep0 <= 15) & (arr0 > 15),
            (dep0 > 15) & (arr0 > 15),
        ],
        [
            "neither_delayed",
            "departure_only",
            "arrival_only",
            "both_delayed",
        ],
        default="other",
    )


    # Missing arrival delay gets its own status.
    out["delay_status"] = np.where(
        arr.isna(),
        "arrival_missing",
        out["delay_status"],
    )


    out["delay_status__travel"] = (
        out["delay_status"]
        + "__"
        + travel
    )


    out["service_mean_x_customer"] = (
        out["service_mean"]
        .fillna(-1)
        .round(1)
        .astype(str)
        + "__"
        + customer
    )


    out["digital_mean_x_travel"] = (
        out["digital_rating_mean"]
        .fillna(-1)
        .round(1)
        .astype(str)
        + "__"
        + travel
    )


    out["delay_severity_x_class"] = (
        out["departure_delay_band"]
        + "__"
        + out["arrival_delay_band"]
        + "__"
        + class_value
    )


    return out


# ============================================================
# 10. BUILD CANDIDATE FEATURES
# ============================================================

print("\n" + "=" * 100)
print("BUILDING CANDIDATE FEATURES")
print("=" * 100)


candidate_train = build_candidate_features(
    train
)

candidate_test = build_candidate_features(
    test
)


if list(candidate_train.columns) != list(candidate_test.columns):
    raise ValueError(
        "Train/test candidate feature columns differ."
    )

if candidate_train.columns.duplicated().any():
    raise ValueError(
        "Duplicate candidate feature columns."
    )


print(
    "Candidate features:",
    len(candidate_train.columns),
)

print(
    "Training candidate matrix:",
    candidate_train.shape,
)

print(
    "Test candidate matrix:",
    candidate_test.shape,
)


# ============================================================
# 11. SAVE CANDIDATE FEATURE MATRICES
# ============================================================

candidate_train_out = candidate_train.copy()

candidate_train_out.insert(
    0,
    ID_COL,
    train_ids,
)

candidate_train_out.insert(
    1,
    TARGET,
    y,
)


candidate_test_out = candidate_test.copy()

candidate_test_out.insert(
    0,
    ID_COL,
    test_ids,
)


candidate_train_path = os.path.join(
    OUTPUT_DIR,
    "candidate_features_train.csv",
)

candidate_test_path = os.path.join(
    OUTPUT_DIR,
    "candidate_features_test.csv",
)


candidate_train_out.to_csv(
    candidate_train_path,
    index=False,
    encoding="utf-8-sig",
)

candidate_test_out.to_csv(
    candidate_test_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 12. NUMERIC FEATURE PROFILE OF HARD CASES
# ============================================================

print("\n" + "=" * 100)
print("NUMERIC FEATURE PROFILE OF HARD CASES")
print("=" * 100)


numeric_candidate_cols = [
    col
    for col in candidate_train.columns
    if pd.api.types.is_numeric_dtype(
        candidate_train[col]
    )
]


top_neg_idx = (
    top_rank
    & (y == 0)
)

top_pos_idx = (
    top_rank
    & (y == 1)
)

bottom_pos_idx = (
    bottom_rank
    & (y == 1)
)

bottom_neg_idx = (
    bottom_rank
    & (y == 0)
)


profile_rows = []


for col in numeric_candidate_cols:

    values = pd.to_numeric(
        candidate_train[col],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )


    top_neg = values[top_neg_idx]
    top_pos = values[top_pos_idx]

    bottom_pos = values[bottom_pos_idx]
    bottom_neg = values[bottom_neg_idx]


    if (
        np.isfinite(top_neg).sum()
        + np.isfinite(top_pos).sum()
        < 2
    ):
        continue


    valid = np.isfinite(values)


    profile_rows.append({

        "feature":
            col,

        "overall_mean":
            np.nanmean(values),

        "top10_negative_mean":
            np.nanmean(top_neg),

        "top10_positive_mean":
            np.nanmean(top_pos),

        "top10_negative_vs_positive_smd":
            standardized_mean_difference(
                top_neg,
                top_pos,
            ),

        "bottom10_positive_mean":
            np.nanmean(bottom_pos),

        "bottom10_negative_mean":
            np.nanmean(bottom_neg),

        "bottom10_positive_vs_negative_smd":
            standardized_mean_difference(
                bottom_pos,
                bottom_neg,
            ),

        "spearman_with_champion_rank":
            safe_spearman(
                values,
                champion_rank,
            ),

        "univariate_auc":
            safe_auc(
                y[valid],
                values[valid],
            ),

    })


numeric_profile_df = pd.DataFrame(
    profile_rows
)


if not numeric_profile_df.empty:

    numeric_profile_df["abs_top_smd"] = (
        numeric_profile_df[
            "top10_negative_vs_positive_smd"
        ].abs()
    )

    numeric_profile_df["abs_bottom_smd"] = (
        numeric_profile_df[
            "bottom10_positive_vs_negative_smd"
        ].abs()
    )

    numeric_profile_df = numeric_profile_df.sort_values(
        [
            "abs_top_smd",
            "abs_bottom_smd",
        ],
        ascending=False,
    )


numeric_profile_path = os.path.join(
    OUTPUT_DIR,
    "numeric_feature_hard_case_profile.csv",
)

numeric_profile_df.to_csv(
    numeric_profile_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 13. SUBGROUP AUC AND HARD-CASE ENRICHMENT
# ============================================================

print("\n" + "=" * 100)
print("SUBGROUP AUC AND HARD-CASE ENRICHMENT")
print("=" * 100)


group_analysis_df = pd.DataFrame(
    index=train.index
)


for col in CATEGORICAL_COLS:
    group_analysis_df[col] = clean_category(
        train[col]
    )


GROUP_COLUMNS = [
    "age_band",
    "distance_band",
    "service_mean_band",
    "service_low_count_band",
    "departure_delay_band",
    "arrival_delay_band",
    "age_band__travel",
    "age_band__class",
    "distance_band__travel",
    "distance_band__class",
    "service_mean_band__travel",
    "low_service_count__travel",
    "low_service_count__customer",
    "delay_status",
    "delay_status__travel",
    "delay_severity_x_class",
]


for col in GROUP_COLUMNS:
    group_analysis_df[col] = (
        candidate_train[col].astype(str)
    )


group_analysis_df["__target"] = y
group_analysis_df["__rank"] = champion_rank
group_analysis_df["__top10"] = top_rank
group_analysis_df["__bottom10"] = bottom_rank


global_top_negative_rate = (
    high_rank_negative.sum()
    / max((y == 0).sum(), 1)
)

global_bottom_positive_rate = (
    low_rank_positive.sum()
    / max((y == 1).sum(), 1)
)


group_rows = []


for group_col in CATEGORICAL_COLS + GROUP_COLUMNS:

    for group_value, group in group_analysis_df.groupby(
        group_col,
        dropna=False,
        observed=True,
    ):

        n = len(group)

        if n < MIN_GROUP_ROWS:
            continue


        group_y = group[
            "__target"
        ].to_numpy()

        group_rank = group[
            "__rank"
        ].to_numpy()


        n_pos = int(
            (group_y == 1).sum()
        )

        n_neg = int(
            (group_y == 0).sum()
        )


        if (
            n_pos < MIN_GROUP_PER_CLASS
            or n_neg < MIN_GROUP_PER_CLASS
        ):
            group_auc = np.nan

        else:
            group_auc = roc_auc_score(
                group_y,
                group_rank,
            )


        top_negative_count = int(
            (
                (group_y == 0)
                & group["__top10"].to_numpy()
            ).sum()
        )

        bottom_positive_count = int(
            (
                (group_y == 1)
                & group["__bottom10"].to_numpy()
            ).sum()
        )


        top_negative_rate = (
            top_negative_count
            / max(n_neg, 1)
        )

        bottom_positive_rate = (
            bottom_positive_count
            / max(n_pos, 1)
        )


        group_rows.append({

            "group_feature":
                group_col,

            "group_value":
                str(group_value),

            "n":
                n,

            "positive_count":
                n_pos,

            "negative_count":
                n_neg,

            "positive_rate":
                n_pos / max(n, 1),

            "within_group_auc":
                group_auc,

            "top10_negative_count":
                top_negative_count,

            "top10_negative_rate_among_negatives":
                top_negative_rate,

            "top10_negative_enrichment":
                (
                    top_negative_rate
                    / max(global_top_negative_rate, 1e-12)
                ),

            "bottom10_positive_count":
                bottom_positive_count,

            "bottom10_positive_rate_among_positives":
                bottom_positive_rate,

            "bottom10_positive_enrichment":
                (
                    bottom_positive_rate
                    / max(global_bottom_positive_rate, 1e-12)
                ),

            "mean_champion_rank":
                np.mean(group_rank),

        })


subgroup_df = pd.DataFrame(
    group_rows
)


if not subgroup_df.empty:

    subgroup_df = subgroup_df.sort_values(
        [
            "within_group_auc",
            "n",
        ],
        ascending=[
            True,
            False,
        ],
        na_position="last",
    )


subgroup_path = os.path.join(
    OUTPUT_DIR,
    "subgroup_auc_and_hard_case_enrichment.csv",
)

subgroup_df.to_csv(
    subgroup_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 14. MODEL DISAGREEMENT ANALYSIS
# ============================================================

print("\n" + "=" * 100)
print("MODEL DISAGREEMENT ANALYSIS")
print("=" * 100)


disagreement_df = pd.DataFrame({

    "rank_disagreement":
        rank_disagreement,

    "lgbm_rank":
        lgbm_rank,

    "realmlp_rank":
        realmlp_rank,

    "champion_rank":
        champion_rank,

    TARGET:
        y,

})


disagreement_df["disagreement_decile"] = pd.qcut(
    disagreement_df["rank_disagreement"],
    q=10,
    labels=False,
    duplicates="drop",
)


disagreement_rows = []


for decile, group in disagreement_df.groupby(
    "disagreement_decile",
    dropna=False,
):

    if len(group) == 0 or pd.isna(decile):
        continue


    disagreement_rows.append({

        "disagreement_decile":
            int(decile),

        "n":
            len(group),

        "mean_abs_rank_disagreement":
            group["rank_disagreement"].mean(),

        "mean_lgbm_rank":
            group["lgbm_rank"].mean(),

        "mean_realmlp_rank":
            group["realmlp_rank"].mean(),

        "mean_champion_rank":
            group["champion_rank"].mean(),

        "positive_rate":
            group[TARGET].mean(),

        "champion_auc_within_decile":
            safe_auc(
                group[TARGET],
                group["champion_rank"],
            ),

    })


disagreement_summary_df = pd.DataFrame(
    disagreement_rows
)


disagreement_summary_path = os.path.join(
    OUTPUT_DIR,
    "model_disagreement_by_decile.csv",
)


disagreement_summary_df.to_csv(
    disagreement_summary_path,
    index=False,
    encoding="utf-8-sig",
)


# Save the 50,000 observations with the largest disagreement.
largest_disagreement_path = os.path.join(
    OUTPUT_DIR,
    "largest_model_disagreement_cases.csv",
)


top_disagreement_idx = np.argsort(
    -rank_disagreement
)[:50_000]


disagreement_cases = diagnostics.iloc[
    top_disagreement_idx
].copy()


disagreement_cases.to_csv(
    largest_disagreement_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 15. SAVE ROW-LEVEL HARD CASES
# ============================================================

print("\n" + "=" * 100)
print("SAVING HARD CASES")
print("=" * 100)


hard_case_mask = (
    high_rank_negative
    | low_rank_positive
)


hard_cases = diagnostics.loc[
    hard_case_mask
].copy()


hard_cases["hard_case_type"] = np.select(
    [
        hard_cases["high_rank_negative"],
        hard_cases["low_rank_positive"],
    ],
    [
        "negative_in_top10pct",
        "positive_in_bottom10pct",
    ],
    default="other",
)


# Align by the original row index.
# Both dataframes preserve the original train index.
hard_cases = pd.concat(
    [
        hard_cases,
        candidate_train.loc[
            hard_cases.index
        ].copy(),
    ],
    axis=1,
)


hard_cases_path = os.path.join(
    OUTPUT_DIR,
    "champion_hard_cases.csv",
)


hard_cases.to_csv(
    hard_cases_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 16. FEATURE DEFINITIONS
# ============================================================

feature_definitions = [
    ("service_mean", "Mean of the 13 service ratings"),
    ("service_std", "Within-row standard deviation of service ratings"),
    ("service_min", "Minimum of the 13 service ratings"),
    ("service_max", "Maximum of the 13 service ratings"),
    ("service_range", "Maximum minus minimum service rating"),
    ("service_low_count_le1", "Number of service ratings <= 1"),
    ("service_low_count_le2", "Number of service ratings <= 2"),
    ("service_high_count_ge4", "Number of service ratings >= 4"),
    ("service_perfect_count_eq5", "Number of service ratings equal to 5"),
    ("service_missing_count", "Number of missing service ratings"),
    ("digital_rating_mean", "Mean of wifi, online booking, and online boarding ratings"),
    ("comfort_rating_mean", "Mean of food, seat comfort, entertainment, and cleanliness"),
    ("onboard_rating_mean", "Mean of onboard service, legroom, and baggage handling"),
    ("airport_rating_mean", "Mean of time convenience, gate location, and check-in"),
    ("gap_boarding_vs_booking", "Online boarding rating minus online booking rating"),
    ("gap_wifi_vs_booking", "Wifi rating minus online booking rating"),
    ("gap_entertainment_vs_seat", "Entertainment rating minus seat comfort rating"),
    ("gap_onboard_vs_legroom", "On-board service rating minus legroom rating"),
    ("gap_baggage_vs_checkin", "Baggage handling rating minus check-in rating"),
    ("gap_digital_vs_comfort", "Digital service mean minus comfort service mean"),
    ("arrival_delay_missing", "Indicator that arrival delay is missing"),
    ("delay_sum_log1p", "Log of one plus departure and zero-filled arrival delay"),
    ("delay_abs_difference_log1p", "Log of one plus absolute arrival/departure delay difference"),
    ("any_delay_gt15", "Indicator that either delay exceeds 15 minutes"),
    ("both_delays_gt15", "Indicator that both delays exceed 15 minutes"),
    ("arrival_missing_and_departure_delayed", "Arrival delay missing and departure delay exceeds 15 minutes"),
    ("age_band__travel", "Age band crossed with type of travel"),
    ("age_band__class", "Age band crossed with class"),
    ("distance_band__travel", "Flight-distance band crossed with type of travel"),
    ("distance_band__class", "Flight-distance band crossed with class"),
    ("service_mean_band__travel", "Service-mean band crossed with type of travel"),
    ("low_service_count__travel", "Low-rating-count band crossed with type of travel"),
    ("low_service_count__customer", "Low-rating-count band crossed with customer type"),
    ("delay_status__travel", "Delay-status category crossed with type of travel"),
    ("delay_severity_x_class", "Departure-delay band and arrival-delay band crossed with class"),
]


feature_definitions_df = pd.DataFrame(
    feature_definitions,
    columns=[
        "feature",
        "definition",
    ],
)


feature_definitions_path = os.path.join(
    OUTPUT_DIR,
    "candidate_feature_definitions.csv",
)


feature_definitions_df.to_csv(
    feature_definitions_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 17. PRINT MOST RELEVANT DIAGNOSTICS
# ============================================================

print("\n" + "=" * 100)
print("TOP NUMERIC FEATURES DIFFERENTIATING HARD CASES")
print("=" * 100)


if not numeric_profile_df.empty:

    preview_cols = [
        "feature",
        "top10_negative_vs_positive_smd",
        "bottom10_positive_vs_negative_smd",
        "spearman_with_champion_rank",
        "univariate_auc",
    ]

    print(
        numeric_profile_df[
            preview_cols
        ].head(25).to_string(
            index=False
        )
    )

else:
    print(
        "No numeric feature profile could be calculated."
    )


print("\n" + "=" * 100)
print("LOWEST SUBGROUP AUC (WITH MINIMUM SAMPLE SIZE)")
print("=" * 100)


if not subgroup_df.empty:

    valid_subgroups = subgroup_df.dropna(
        subset=["within_group_auc"]
    )

    if not valid_subgroups.empty:

        print(
            valid_subgroups[
                [
                    "group_feature",
                    "group_value",
                    "n",
                    "positive_rate",
                    "within_group_auc",
                    "top10_negative_enrichment",
                    "bottom10_positive_enrichment",
                ]
            ].head(30).to_string(
                index=False
            )
        )

    else:
        print(
            "No subgroup has enough examples of both labels."
        )

else:
    print("No subgroup results.")


print("\n" + "=" * 100)
print("MODEL DISAGREEMENT SUMMARY")
print("=" * 100)

print(
    disagreement_summary_df.to_string(
        index=False
    )
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

summary = pd.DataFrame([

    {
        "metric":
            "Current LGBM OOF AUC",

        "value":
            lgbm_auc,
    },

    {
        "metric":
            "RealMLP V1 OOF AUC",

        "value":
            realmlp_auc,
    },

    {
        "metric":
            "Reconstructed Champion OOF AUC",

        "value":
            champion_auc,
    },

    {
        "metric":
            "LGBM vs RealMLP rank correlation",

        "value":
            safe_spearman(
                lgbm_rank,
                realmlp_rank,
            ),
    },

    {
        "metric":
            "Champion top-decile negatives",

        "value":
            int(high_rank_negative.sum()),
    },

    {
        "metric":
            "Champion bottom-decile positives",

        "value":
            int(low_rank_positive.sum()),
    },

    {
        "metric":
            "Candidate feature count",

        "value":
            len(candidate_train.columns),
    },

    {
        "metric":
            "Number of subgroup rows",

        "value":
            len(subgroup_df),
    },

])


summary_path = os.path.join(
    OUTPUT_DIR,
    "champion_error_analysis_summary.csv",
)


summary.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig",
)


print("\n" + "=" * 100)
print("FINAL SUMMARY")
print("=" * 100)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 19. OUTPUT FILES
# ============================================================

print("\n" + "=" * 100)
print("OUTPUT FILES")
print("=" * 100)


for label, path in [
    ("Summary", summary_path),
    ("OOF diagnostics", diagnostics_path),
    ("Numeric hard-case profile", numeric_profile_path),
    ("Subgroup AUC and enrichment", subgroup_path),
    ("Model disagreement deciles", disagreement_summary_path),
    ("Largest model disagreement cases", largest_disagreement_path),
    ("Champion hard cases", hard_cases_path),
    ("Candidate feature definitions", feature_definitions_path),
    ("Candidate train features", candidate_train_path),
    ("Candidate test features", candidate_test_path),
]:
    print(f"\n{label}:\n{path}")


print("\n" + "=" * 100)
print("ERROR ANALYSIS COMPLETE")
print("=" * 100)