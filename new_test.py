import os
import numpy as np
import pandas as pd

from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


# ============================================================
# CURRENT CHAMPION + OLD ENSEMBLE
#
# Current real LB champion:
#
#   Conditional-TE LGBM 40%
#   Lean RealMLP V1       60%
#   Rank blend
#
# Current LB:
#   0.96097
#
# New experiment:
#
#   Current Champion
#         +
#   Original old ensemble
#
# NO RETRAINING.
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
# LEAN REALMLP V1
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
# OLD ORIGINAL ENSEMBLE
#
# From the previous ensemble_02 system.
# ============================================================

OLD_OOF_PATH = (
    r"D:\kaggle\airline"
    r"\ensemble_02_best_oof_predictions.csv"
)

OLD_TEST_PATH = (
    r"D:\kaggle\airline"
    r"\ensemble_02_best_test_predictions.csv"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    r"D:\kaggle\airline"
    r"\champion_old_ensemble_blend"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

# Current empirically validated champion:
CURRENT_LGBM_WEIGHT = 0.40
CURRENT_V1_WEIGHT = 0.60

# Old ensemble added on top of current champion.
#
# The final structure is:
#
#   (1-old_weight) * current_champion
#   +
#   old_weight * old_ensemble
#
# This is much cleaner than a free 3-way weight search.
OLD_WEIGHTS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
]


# ============================================================
# HEADER
# ============================================================

print("=" * 100)
print("CURRENT CHAMPION + OLD ENSEMBLE ABLATION")
print("=" * 100)

print(
    "\nCurrent champion:"
)

print(
    "  LGBM = 40%"
)

print(
    "  V1   = 60%"
)

print(
    "\nCurrent verified LB = 0.96097"
)

print(
    "\nNew experiment:"
)

print(
    "  Add original ensemble_02 prediction"
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
# TARGET
# ============================================================

y = (
    train[TARGET]
    .astype("int8")
    .to_numpy()
    .reshape(-1)
)


# ============================================================
# IDS
# ============================================================

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


# ============================================================
# BASIC ID CHECKS
# ============================================================

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
        "Test IDs and sample_submission IDs "
        "do not match."
    )


print(
    f"\nTrain rows = {len(train)}"
)

print(
    f"Test rows  = {len(test)}"
)


# ============================================================
# PREDICTION LOADER
# ============================================================

def load_prediction_file(
    path,
    name,
    preferred_columns=None,
):
    """
    Load id + prediction from a CSV.
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
            f"{name} has no 'id' column.\n"
            f"Columns = {df.columns.tolist()}"
        )


    # --------------------------------------------------------
    # Explicit preferred columns first.
    # --------------------------------------------------------

    candidates = []

    if preferred_columns is not None:

        candidates.extend(
            preferred_columns
        )


    # --------------------------------------------------------
    # Generic fallback columns.
    # --------------------------------------------------------

    candidates.extend([
        "prediction",
        "pred",
        "probability",
        "prob",
        "satisfaction",
        "target",
        "lgbm",
        "ensemble_prediction",
        "best_ensemble_oof",
        "best_ensemble_test",
        "best_ensemble_pred",
    ])


    pred_col = None


    for c in candidates:

        if c in df.columns:

            pred_col = c

            break


    # --------------------------------------------------------
    # Numeric-column fallback.
    # --------------------------------------------------------

    if pred_col is None:

        numeric_cols = [

            c

            for c in df.columns

            if (
                c != "id"
                and
                pd.api.types.is_numeric_dtype(
                    df[c]
                )
            )

        ]


        # Remove obvious non-prediction columns.
        numeric_cols = [

            c

            for c in numeric_cols

            if c not in [
                "fold",
                "target",
                "y",
                "satisfaction",
            ]

        ]


        if len(
            numeric_cols
        ) != 1:

            raise ValueError(
                f"Cannot uniquely identify "
                f"prediction column in {name}.\n"
                f"Columns = {df.columns.tolist()}\n"
                f"Numeric candidates = {numeric_cols}"
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


    print(
        f"\n{name}"
    )

    print(
        f"  rows       = {len(out)}"
    )

    print(
        f"  pred col   = {pred_col}"
    )

    print(
        f"  pred min   = {out['prediction'].min():.9f}"
    )

    print(
        f"  pred max   = {out['prediction'].max():.9f}"
    )

    print(
        f"  pred mean  = {out['prediction'].mean():.9f}"
    )

    print(
        f"  pred std   = {out['prediction'].std():.9f}"
    )


    return out


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "LOADING EXISTING PREDICTIONS"
)

print(
    "=" * 100
)


lgbm_oof_df = load_prediction_file(
    LGBM_OOF_PATH,
    "Current LGBM OOF",
    preferred_columns=[
        "prediction",
    ],
)


lgbm_test_df = load_prediction_file(
    LGBM_TEST_PATH,
    "Current LGBM TEST",
    preferred_columns=[
        "prediction",
    ],
)


v1_oof_df = load_prediction_file(
    V1_OOF_PATH,
    "Lean RealMLP V1 OOF",
    preferred_columns=[
        "prediction",
    ],
)


v1_test_df = load_prediction_file(
    V1_TEST_PATH,
    "Lean RealMLP V1 TEST",
    preferred_columns=[
        "prediction",
    ],
)


old_oof_df = load_prediction_file(
    OLD_OOF_PATH,
    "Old Ensemble OOF",
    preferred_columns=[
        "best_ensemble_oof",
        "prediction",
        "ensemble_prediction",
    ],
)


old_test_df = load_prediction_file(
    OLD_TEST_PATH,
    "Old Ensemble TEST",
    preferred_columns=[
        "ensemble_prediction",
        "best_ensemble_test",
        "prediction",
    ],
)


# ============================================================
# ALIGN PREDICTIONS BY ID
# ============================================================

train_index = pd.DataFrame({
    "id":
        train_ids,
})


test_index = pd.DataFrame({
    "id":
        test_ids,
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


    if len(
        aligned
    ) != len(
        index_df
    ):

        raise ValueError(
            f"{name}: row count mismatch."
        )


    if aligned[
        "prediction"
    ].isna().any():

        missing_ids = aligned.loc[
            aligned["prediction"].isna(),
            "id",
        ].head(10).tolist()


        raise ValueError(
            f"{name}: missing predictions "
            f"after alignment.\n"
            f"Example IDs: {missing_ids}"
        )


    pred = (
        aligned[
            "prediction"
        ]
        .to_numpy(
            dtype=np.float64
        )
        .reshape(-1)
    )


    if not np.isfinite(
        pred
    ).all():

        raise ValueError(
            f"{name}: non-finite values "
            "after alignment."
        )


    return pred


# ------------------------------------------------------------
# OOF
# ------------------------------------------------------------

lgbm_oof = align_prediction(
    train_index,
    lgbm_oof_df,
    "LGBM OOF",
)


v1_oof = align_prediction(
    train_index,
    v1_oof_df,
    "V1 OOF",
)


old_oof = align_prediction(
    train_index,
    old_oof_df,
    "Old Ensemble OOF",
)


# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------

lgbm_test = align_prediction(
    test_index,
    lgbm_test_df,
    "LGBM TEST",
)


v1_test = align_prediction(
    test_index,
    v1_test_df,
    "V1 TEST",
)


old_test = align_prediction(
    test_index,
    old_test_df,
    "Old Ensemble TEST",
)


# ============================================================
# INDIVIDUAL AUC
# ============================================================

lgbm_auc = roc_auc_score(
    y,
    lgbm_oof,
)


v1_auc = roc_auc_score(
    y,
    v1_oof,
)


old_auc = roc_auc_score(
    y,
    old_oof,
)


print(
    "\n"
    + "=" * 100
)

print(
    "INDIVIDUAL OOF PERFORMANCE"
)

print(
    "=" * 100
)


print(
    f"LGBM AUC = "
    f"{lgbm_auc:.9f}"
)


print(
    f"V1 AUC   = "
    f"{v1_auc:.9f}"
)


print(
    f"Old Ensemble AUC = "
    f"{old_auc:.9f}"
)


# ============================================================
# CURRENT CHAMPION OOF
# ============================================================

current_champion_raw_oof = (
    CURRENT_LGBM_WEIGHT
    * lgbm_oof
    +
    CURRENT_V1_WEIGHT
    * v1_oof
)


current_champion_auc = roc_auc_score(
    y,
    current_champion_raw_oof,
)


# Rank representation
def rank_normalize(
    x,
):

    x = np.asarray(
        x,
        dtype=np.float64,
    ).reshape(-1)


    return (
        rankdata(
            x,
            method="average",
        )
        /
        len(x)
    )


lgbm_rank_oof = rank_normalize(
    lgbm_oof
)

v1_rank_oof = rank_normalize(
    v1_oof
)

old_rank_oof = rank_normalize(
    old_oof
)


lgbm_rank_test = rank_normalize(
    lgbm_test
)

v1_rank_test = rank_normalize(
    v1_test
)

old_rank_test = rank_normalize(
    old_test
)


current_champion_rank_oof = (
    CURRENT_LGBM_WEIGHT
    * lgbm_rank_oof
    +
    CURRENT_V1_WEIGHT
    * v1_rank_oof
)


current_champion_rank_auc = (
    roc_auc_score(
        y,
        current_champion_rank_oof,
    )
)


print(
    "\n"
    + "=" * 100
)

print(
    "CURRENT CHAMPION RECHECK"
)

print(
    "=" * 100
)


print(
    f"Raw LGBM/V1 40/60 OOF = "
    f"{current_champion_auc:.9f}"
)


print(
    f"Rank LGBM/V1 40/60 OOF = "
    f"{current_champion_rank_auc:.9f}"
)


# ============================================================
# CORRELATIONS
# ============================================================

corr_lgbm_v1 = spearmanr(
    lgbm_oof,
    v1_oof,
).statistic


corr_lgbm_old = spearmanr(
    lgbm_oof,
    old_oof,
).statistic


corr_v1_old = spearmanr(
    v1_oof,
    old_oof,
).statistic


corr_champion_old = spearmanr(
    current_champion_rank_oof,
    old_oof,
).statistic


print(
    "\n"
    + "=" * 100
)

print(
    "RANK CORRELATION"
)

print(
    "=" * 100
)


print(
    f"LGBM vs V1       = "
    f"{corr_lgbm_v1:.9f}"
)


print(
    f"LGBM vs Old      = "
    f"{corr_lgbm_old:.9f}"
)


print(
    f"V1   vs Old      = "
    f"{corr_v1_old:.9f}"
)


print(
    f"Champion vs Old  = "
    f"{corr_champion_old:.9f}"
)


# ============================================================
# EXPERIMENT 1
#
# Add old ensemble to CURRENT CHAMPION.
#
# This is intentionally a hierarchical blend:
#
#   final =
#       (1 - old_weight) * current_champion
#       +
#       old_weight * old_ensemble
#
# This keeps the already validated LGBM/V1 ratio at 40/60.
# ============================================================

hierarchical_results = []


for old_weight in OLD_WEIGHTS:

    current_weight = (
        1.0
        -
        old_weight
    )


    # --------------------------------------------------------
    # Raw version
    # --------------------------------------------------------

    raw_oof = (
        current_weight
        * current_champion_raw_oof
        +
        old_weight
        * old_oof
    )


    raw_auc = roc_auc_score(
        y,
        raw_oof,
    )


    # --------------------------------------------------------
    # Rank version
    #
    # Important:
    #
    # current_champion_rank_oof is already a rank blend.
    # The old ensemble is converted to rank before mixing.
    # --------------------------------------------------------

    rank_oof = (
        current_weight
        * current_champion_rank_oof
        +
        old_weight
        * old_rank_oof
    )


    rank_auc = roc_auc_score(
        y,
        rank_oof,
    )


    hierarchical_results.append({

        "old_weight":
            old_weight,

        "champion_weight":
            current_weight,

        "raw_auc":
            raw_auc,

        "rank_auc":
            rank_auc,

        "raw_gain_vs_champion":
            raw_auc
            -
            current_champion_auc,

        "rank_gain_vs_champion":
            rank_auc
            -
            current_champion_rank_auc,
    })


hierarchical_df = (
    pd.DataFrame(
        hierarchical_results
    )
)


# ============================================================
# DISPLAY HIERARCHICAL RESULTS
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "ADD OLD ENSEMBLE TO CURRENT CHAMPION"
)

print(
    "=" * 100
)


print(
    hierarchical_df.to_string(
        index=False,
        formatters={

            "old_weight":
                "{:.2f}".format,

            "champion_weight":
                "{:.2f}".format,

            "raw_auc":
                "{:.9f}".format,

            "rank_auc":
                "{:.9f}".format,

            "raw_gain_vs_champion":
                "{:+.9f}".format,

            "rank_gain_vs_champion":
                "{:+.9f}".format,
        },
    )
)


# ============================================================
# EXPERIMENT 2
#
# Direct 3-way rank blend.
#
# A small number of plausible candidates only.
#
# No dense optimization.
# ============================================================

direct_rank_candidates = [

    # No old ensemble:
    # current champion
    {
        "name":
            "LGBM40_V160_Old0",

        "lgbm":
            0.40,

        "v1":
            0.60,

        "old":
            0.00,
    },


    {
        "name":
            "LGBM38_V157_Old5",

        "lgbm":
            0.38,

        "v1":
            0.57,

        "old":
            0.05,
    },


    {
        "name":
            "LGBM36_V154_Old10",

        "lgbm":
            0.36,

        "v1":
            0.54,

        "old":
            0.10,
    },


    {
        "name":
            "LGBM34_V151_Old15",

        "lgbm":
            0.34,

        "v1":
            0.51,

        "old":
            0.15,
    },


    {
        "name":
            "LGBM32_V148_Old20",

        "lgbm":
            0.32,

        "v1":
            0.48,

        "old":
            0.20,
    },


    {
        "name":
            "LGBM30_V145_Old25",

        "lgbm":
            0.30,

        "v1":
            0.45,

        "old":
            0.25,
    },


    {
        "name":
            "LGBM28_V142_Old30",

        "lgbm":
            0.28,

        "v1":
            0.42,

        "old":
            0.30,
    },

]


direct_results = []


for candidate in direct_rank_candidates:

    total = (
        candidate["lgbm"]
        +
        candidate["v1"]
        +
        candidate["old"]
    )


    if abs(
        total - 1.0
    ) > 1e-10:

        raise ValueError(
            f"Invalid weights: "
            f"{candidate['name']}"
        )


    blend_oof = (

        candidate["lgbm"]
        * lgbm_rank_oof

        +

        candidate["v1"]
        * v1_rank_oof

        +

        candidate["old"]
        * old_rank_oof

    )


    auc = roc_auc_score(
        y,
        blend_oof,
    )


    direct_results.append({

        "name":
            candidate["name"],

        "lgbm_weight":
            candidate["lgbm"],

        "v1_weight":
            candidate["v1"],

        "old_weight":
            candidate["old"],

        "oof_auc":
            auc,

        "gain_vs_current":
            auc
            -
            current_champion_rank_auc,
    })


direct_df = (
    pd.DataFrame(
        direct_results
    )
    .sort_values(
        "oof_auc",
        ascending=False,
    )
    .reset_index(drop=True)
)


print(
    "\n"
    + "=" * 100
)

print(
    "DIRECT 3-MODEL RANK BLEND"
)

print(
    "=" * 100
)


print(
    direct_df.to_string(
        index=False,
        formatters={

            "lgbm_weight":
                "{:.2f}".format,

            "v1_weight":
                "{:.2f}".format,

            "old_weight":
                "{:.2f}".format,

            "oof_auc":
                "{:.9f}".format,

            "gain_vs_current":
                "{:+.9f}".format,
        },
    )
)


# ============================================================
# SELECT BEST DIAGNOSTIC CANDIDATES
# ============================================================

best_hierarchical_rank = (
    hierarchical_df.loc[
        hierarchical_df[
            "rank_auc"
        ].idxmax()
    ]
)


best_hierarchical_raw = (
    hierarchical_df.loc[
        hierarchical_df[
            "raw_auc"
        ].idxmax()
    ]
)


best_direct = (
    direct_df.iloc[0]
)


print(
    "\n"
    + "=" * 100
)

print(
    "BEST DIAGNOSTIC CANDIDATES"
)

print(
    "=" * 100
)


print(
    "\nBest hierarchical RANK:"
)

print(
    best_hierarchical_rank.to_string()
)


print(
    "\nBest hierarchical RAW:"
)

print(
    best_hierarchical_raw.to_string()
)


print(
    "\nBest direct 3-way RANK:"
)

print(
    best_direct.to_string()
)


# ============================================================
# TEST PREDICTION GENERATION
# ============================================================

# ------------------------------------------------------------
# Current champion test rank
# ------------------------------------------------------------

current_champion_rank_test = (
    CURRENT_LGBM_WEIGHT
    * lgbm_rank_test
    +
    CURRENT_V1_WEIGHT
    * v1_rank_test
)


# ------------------------------------------------------------
# Current champion raw
# ------------------------------------------------------------

current_champion_raw_test = (
    CURRENT_LGBM_WEIGHT
    * lgbm_test
    +
    CURRENT_V1_WEIGHT
    * v1_test
)


# ============================================================
# SUBMISSION TARGET
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
        "sample_submission.csv must contain "
        "exactly one target column."
    )


submission_target = (
    target_candidates[0]
)


# ============================================================
# SUBMISSION HELPER
# ============================================================

def save_submission(
    prediction,
    filename,
):

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    ).reshape(-1)


    if len(prediction) != len(test):

        raise ValueError(
            f"{filename}: prediction length "
            "does not match test."
        )


    if not np.isfinite(
        prediction
    ).all():

        raise ValueError(
            f"{filename}: prediction contains "
            "non-finite values."
        )


    pred_df = pd.DataFrame({

        "id":
            test_ids,

        "prediction":
            prediction,
    })


    submission = (
        sample[
            [ID_COL]
        ]
        .copy()
    )


    submission["id"] = pd.to_numeric(
        submission["id"],
        errors="raise",
    )


    submission = (
        submission
        .merge(
            pred_df,
            on="id",
            how="left",
            validate="one_to_one",
        )
    )


    if len(
        submission
    ) != len(sample):

        raise ValueError(
            f"{filename}: submission row count "
            "changed after merge."
        )


    if submission[
        "prediction"
    ].isna().any():

        raise ValueError(
            f"{filename}: missing predictions."
        )


    submission[
        submission_target
    ] = (
        submission[
            "prediction"
        ].astype(float)
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
            f"{filename}: final row count mismatch."
        )


    if submission[
        "id"
    ].duplicated().any():

        raise ValueError(
            f"{filename}: duplicated IDs."
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


    return output_path


# ============================================================
# ALWAYS SAVE CURRENT CHAMPION
# ============================================================

champion_rank_submission = (
    save_submission(
        current_champion_rank_test,
        "submission_champion_lgbm40_v160_rank.csv",
    )
)


# ============================================================
# SAVE ALL HIERARCHICAL CANDIDATES
# ============================================================

saved_hierarchical = []


for old_weight in OLD_WEIGHTS:

    current_weight = (
        1.0
        -
        old_weight
    )


    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    rank_test = (
        current_weight
        * current_champion_rank_test
        +
        old_weight
        * old_rank_test
    )


    rank_filename = (
        f"submission_hier_rank_old"
        f"{int(old_weight * 100):02d}.csv"
    )


    rank_path = save_submission(
        rank_test,
        rank_filename,
    )


    # --------------------------------------------------------
    # Raw
    # --------------------------------------------------------

    raw_test = (
        current_weight
        * current_champion_raw_test
        +
        old_weight
        * old_test
    )


    raw_filename = (
        f"submission_hier_raw_old"
        f"{int(old_weight * 100):02d}.csv"
    )


    raw_path = save_submission(
        raw_test,
        raw_filename,
    )


    saved_hierarchical.append({

        "old_weight":
            old_weight,

        "rank_submission":
            rank_path,

        "raw_submission":
            raw_path,
    })


# ============================================================
# SAVE DIRECT 3-WAY RANK CANDIDATES
# ============================================================

saved_direct = []


for candidate in direct_rank_candidates:

    test_prediction = (

        candidate["lgbm"]
        * lgbm_rank_test

        +

        candidate["v1"]
        * v1_rank_test

        +

        candidate["old"]
        * old_rank_test

    )


    filename = (
        "submission_direct_"
        + candidate["name"]
        + ".csv"
    )


    path = save_submission(
        test_prediction,
        filename,
    )


    saved_direct.append({

        "name":
            candidate["name"],

        "path":
            path,
    })


# ============================================================
# SAVE RESULT TABLES
# ============================================================

hierarchical_results_path = os.path.join(
    OUTPUT_DIR,
    "hierarchical_results.csv",
)


hierarchical_df.to_csv(
    hierarchical_results_path,
    index=False,
    encoding="utf-8-sig",
)


direct_results_path = os.path.join(
    OUTPUT_DIR,
    "direct_rank_results.csv",
)


direct_df.to_csv(
    direct_results_path,
    index=False,
    encoding="utf-8-sig",
)


correlation_path = os.path.join(
    OUTPUT_DIR,
    "correlations.csv",
)


pd.DataFrame([

    {
        "pair":
            "LGBM_V1",

        "spearman":
            corr_lgbm_v1,
    },

    {
        "pair":
            "LGBM_Old",

        "spearman":
            corr_lgbm_old,
    },

    {
        "pair":
            "V1_Old",

        "spearman":
            corr_v1_old,
    },

    {
        "pair":
            "CurrentChampion_Old",

        "spearman":
            corr_champion_old,
    },

]).to_csv(
    correlation_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# SUMMARY
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "summary.csv",
)


summary = pd.DataFrame([

    {
        "metric":
            "lgbm_oof",

        "value":
            lgbm_auc,
    },

    {
        "metric":
            "v1_oof",

        "value":
            v1_auc,
    },

    {
        "metric":
            "old_ensemble_oof",

        "value":
            old_auc,
    },

    {
        "metric":
            "current_rank_champion_oof",

        "value":
            current_champion_rank_auc,
    },

    {
        "metric":
            "current_rank_champion_lb",

        "value":
            0.96097,
    },

    {
        "metric":
            "lgbm_v1_spearman",

        "value":
            corr_lgbm_v1,
    },

    {
        "metric":
            "lgbm_old_spearman",

        "value":
            corr_lgbm_old,
    },

    {
        "metric":
            "v1_old_spearman",

        "value":
            corr_v1_old,
    },

    {
        "metric":
            "champion_old_spearman",

        "value":
            corr_champion_old,
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
    f"\nCurrent champion:"
)

print(
    champion_rank_submission
)


print(
    f"\nHierarchical results:"
)

print(
    hierarchical_results_path
)


print(
    f"\nDirect 3-way results:"
)

print(
    direct_results_path
)


print(
    f"\nCorrelations:"
)

print(
    correlation_path
)


print(
    f"\nSummary:"
)

print(
    summary_path
)


print(
    "\nExample hierarchical submissions:"
)

for item in saved_hierarchical:

    print(
        f"  old={item['old_weight']:.2f}"
    )

    print(
        f"    rank = "
        f"{item['rank_submission']}"
    )

    print(
        f"    raw  = "
        f"{item['raw_submission']}"
    )


print(
    "\nExample direct submissions:"
)

for item in saved_direct:

    print(
        f"  {item['name']}"
    )

    print(
        f"    {item['path']}"
    )


print(
    "\n"
    + "=" * 100
)

print(
    "DONE"
)

print(
    "=" * 100
)