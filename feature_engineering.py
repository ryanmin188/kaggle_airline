import itertools
import time

import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# ============================================================
# 0. Configuration
# ============================================================

TRAIN_PATH = r"D:\kaggle\airline\data\train.csv"

N_SCREEN_FOLDS = 3       # Fast screening
N_FINAL_FOLDS = 5        # Final confirmation

TOP_K = 3                # Keep top 3 individual interactions

RANDOM_STATE = 42

LGB_PARAMS = {
    "objective": "binary",
    "n_estimators": 960,
    "learning_rate": 0.01,
    "num_leaves": 127,
    "max_depth": -1,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
}


# ============================================================
# 1. Load data
# ============================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

start_time = time.time()

train = pd.read_csv(TRAIN_PATH)

print(f"Train shape: {train.shape}")


# ============================================================
# 2. Basic X and y
# ============================================================

X_base = train.drop(
    columns=["satisfaction", "id"]
).copy()

y = train["satisfaction"].astype(int)


# ============================================================
# 3. Original categorical variables
# ============================================================

categorical_features = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]


# ============================================================
# 4. Define ALL categorical interactions
#
# 4 categorical variables:
# Gender
# Customer Type
# Type of Travel
# Class
#
# Total:
# 6 pairwise interactions
# 4 three-way interactions
# 1 four-way interaction
# = 11 candidates
# ============================================================

interaction_groups = {
    "Gender__Customer_Type": [
        "Gender",
        "Customer Type",
    ],

    "Gender__Type_of_Travel": [
        "Gender",
        "Type of Travel",
    ],

    "Gender__Class": [
        "Gender",
        "Class",
    ],

    "Customer_Type__Type_of_Travel": [
        "Customer Type",
        "Type of Travel",
    ],

    "Customer_Type__Class": [
        "Customer Type",
        "Class",
    ],

    "Type_of_Travel__Class": [
        "Type of Travel",
        "Class",
    ],

    "Gender__Customer_Type__Type_of_Travel": [
        "Gender",
        "Customer Type",
        "Type of Travel",
    ],

    "Gender__Customer_Type__Class": [
        "Gender",
        "Customer Type",
        "Class",
    ],

    "Gender__Type_of_Travel__Class": [
        "Gender",
        "Type of Travel",
        "Class",
    ],

    "Customer_Type__Type_of_Travel__Class": [
        "Customer Type",
        "Type of Travel",
        "Class",
    ],

    "Gender__Customer_Type__Type_of_Travel__Class": [
        "Gender",
        "Customer Type",
        "Type of Travel",
        "Class",
    ],
}


# ============================================================
# 5. Function to create one interaction
# ============================================================

def add_interaction(
    X,
    interaction_name,
    source_columns
):
    """
    Add one categorical interaction feature.

    Example:
    Type of Travel + Class
    ->
    Type_of_Travel__Class
    """

    X_new = X.copy()

    X_new[interaction_name] = (
        X_new[source_columns[0]].astype("string")
    )

    for col in source_columns[1:]:
        X_new[interaction_name] = (
            X_new[interaction_name]
            + "_"
            + X_new[col].astype("string")
        )

    X_new[interaction_name] = (
        X_new[interaction_name].astype("category")
    )

    return X_new


# ============================================================
# 6. Function to build multiple interactions
# ============================================================

def build_features(
    base_X,
    interaction_names
):
    """
    Add the requested interaction features.
    """

    X = base_X.copy()

    categorical_cols = categorical_features.copy()

    for interaction_name in interaction_names:

        source_columns = interaction_groups[
            interaction_name
        ]

        X = add_interaction(
            X,
            interaction_name,
            source_columns
        )

        categorical_cols.append(
            interaction_name
        )

    # Convert original categorical variables
    for col in categorical_cols:
        X[col] = X[col].astype("category")

    return X, categorical_cols


# ============================================================
# 7. Prepare the same CV folds
#
# Using exactly the same folds for every experiment makes
# comparisons fair.
# ============================================================

screen_cv = StratifiedKFold(
    n_splits=N_SCREEN_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

final_cv = StratifiedKFold(
    n_splits=N_FINAL_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


# ============================================================
# 8. Evaluation function
# ============================================================

def evaluate_cv(
    X,
    categorical_cols,
    cv,
    verbose=True
):
    """
    Train LightGBM under the supplied CV split and
    return mean AUC, std AUC, and fold scores.
    """

    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model = lgb.LGBMClassifier(
            **LGB_PARAMS
        )

        model.fit(
            X_train,
            y_train,
            categorical_feature=categorical_cols
        )

        y_pred = model.predict_proba(
            X_valid
        )[:, 1]

        auc = roc_auc_score(
            y_valid,
            y_pred
        )

        fold_scores.append(auc)

        if verbose:
            print(
                f"    Fold {fold}: "
                f"{auc:.6f}"
            )

    mean_auc = sum(fold_scores) / len(fold_scores)

    variance = sum(
        (score - mean_auc) ** 2
        for score in fold_scores
    ) / len(fold_scores)

    std_auc = variance ** 0.5

    return mean_auc, std_auc, fold_scores


# ============================================================
# 9. Baseline: original features
# ============================================================

print("\n" + "=" * 80)
print("STEP 1 — BASELINE")
print("=" * 80)

X_original, original_categorical = build_features(
    X_base,
    []
)

baseline_mean, baseline_std, baseline_folds = evaluate_cv(
    X_original,
    original_categorical,
    screen_cv,
    verbose=True
)

print("-" * 80)
print(f"Baseline Mean AUC: {baseline_mean:.6f}")
print(f"Baseline Std:      {baseline_std:.6f}")


# ============================================================
# 10. Screen ALL 11 individual interactions
# ============================================================

print("\n" + "=" * 80)
print("STEP 2 — SCREEN ALL 11 INDIVIDUAL INTERACTIONS")
print("=" * 80)

screening_results = []

for number, interaction_name in enumerate(
    interaction_groups.keys(),
    start=1
):

    print("\n" + "-" * 80)
    print(
        f"[{number}/11] "
        f"{interaction_name}"
    )
    print("-" * 80)

    X_candidate, candidate_categorical = build_features(
        X_base,
        [interaction_name]
    )

    mean_auc, std_auc, fold_scores = evaluate_cv(
        X_candidate,
        candidate_categorical,
        screen_cv,
        verbose=True
    )

    delta_auc = mean_auc - baseline_mean

    screening_results.append({
        "interaction": interaction_name,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "delta_auc": delta_auc,
    })

    print(
        f"Mean AUC : {mean_auc:.6f}"
    )

    print(
        f"Std AUC  : {std_auc:.6f}"
    )

    print(
        f"Delta    : {delta_auc:+.6f}"
    )


# ============================================================
# 11. Rank individual interactions
# ============================================================

screening_df = pd.DataFrame(
    screening_results
)

screening_df = screening_df.sort_values(
    "mean_auc",
    ascending=False
).reset_index(drop=True)

screening_df["rank"] = (
    screening_df.index + 1
)


print("\n" + "=" * 80)
print("INDIVIDUAL INTERACTION SCREENING RESULTS")
print("=" * 80)

print(
    screening_df[
        [
            "rank",
            "interaction",
            "mean_auc",
            "std_auc",
            "delta_auc",
        ]
    ].to_string(index=False)
)


# ============================================================
# 12. Save screening results
# ============================================================

screening_df.to_csv(
    r"D:\kaggle\airline\interaction_screening_results.csv",
    index=False
)

print(
    "\nSaved:"
    "\nD:\\kaggle\\airline\\interaction_screening_results.csv"
)


# ============================================================
# 13. Automatically select TOP_K
# ============================================================

top_interactions = (
    screening_df
    .head(TOP_K)["interaction"]
    .tolist()
)


print("\n" + "=" * 80)
print(f"STEP 3 — SELECT TOP {TOP_K}")
print("=" * 80)

for rank, name in enumerate(
    top_interactions,
    start=1
):
    row = screening_df[
        screening_df["interaction"] == name
    ].iloc[0]

    print(
        f"{rank}. {name} | "
        f"AUC={row['mean_auc']:.6f} | "
        f"Delta={row['delta_auc']:+.6f}"
    )


# ============================================================
# 14. Test all combinations of TOP_K
#
# For TOP_K=3:
#
# A
# B
# C
# A+B
# A+C
# B+C
# A+B+C
#
# This checks whether individually useful features remain
# useful together.
# ============================================================

print("\n" + "=" * 80)
print(
    f"STEP 4 — TEST COMBINATIONS OF TOP {TOP_K}"
)
print("=" * 80)


combination_results = []

for r in range(
    1,
    len(top_interactions) + 1
):

    for combo in itertools.combinations(
        top_interactions,
        r
    ):

        combo = list(combo)

        combo_name = " + ".join(combo)

        print("\n" + "-" * 80)
        print(combo_name)
        print("-" * 80)

        X_combo, combo_categorical = build_features(
            X_base,
            combo
        )

        mean_auc, std_auc, fold_scores = evaluate_cv(
            X_combo,
            combo_categorical,
            screen_cv,
            verbose=True
        )

        delta_auc = mean_auc - baseline_mean

        combination_results.append({
            "features": combo_name,
            "n_interactions": len(combo),
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "delta_auc": delta_auc,
        })

        print(
            f"Mean AUC : {mean_auc:.6f}"
        )

        print(
            f"Std AUC  : {std_auc:.6f}"
        )

        print(
            f"Delta    : {delta_auc:+.6f}"
        )


# ============================================================
# 15. Rank combinations
# ============================================================

combination_df = pd.DataFrame(
    combination_results
)

combination_df = combination_df.sort_values(
    "mean_auc",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 80)
print("COMBINATION RESULTS")
print("=" * 80)

print(
    combination_df.to_string(index=False)
)


# ============================================================
# 16. Select best combination
# ============================================================

best_combination = combination_df.iloc[0]

best_features_text = best_combination[
    "features"
]

best_interactions = [
    feature.strip()
    for feature in best_features_text.split("+")
]


print("\n" + "=" * 80)
print("CURRENT BEST FEATURE SET")
print("=" * 80)

print(
    f"Features: {best_features_text}"
)

print(
    f"3-Fold Mean AUC: "
    f"{best_combination['mean_auc']:.6f}"
)

print(
    f"3-Fold Std: "
    f"{best_combination['std_auc']:.6f}"
)

print(
    f"Improvement: "
    f"{best_combination['delta_auc']:+.6f}"
)


# ============================================================
# 17. FINAL 5-FOLD CONFIRMATION
# ============================================================

print("\n" + "=" * 80)
print("STEP 5 — FINAL 5-FOLD CONFIRMATION")
print("=" * 80)

print(
    f"Testing: {best_features_text}"
)

X_best, best_categorical = build_features(
    X_base,
    best_interactions
)

final_mean, final_std, final_folds = evaluate_cv(
    X_best,
    best_categorical,
    final_cv,
    verbose=True
)


# ============================================================
# 18. Final result
# ============================================================

final_delta = final_mean - baseline_mean

print("\n" + "=" * 80)
print("FINAL FEATURE ENGINEERING RESULT")
print("=" * 80)

print(
    f"Original baseline 5-fold AUC: "
    f"{baseline_mean:.6f}"
)

print(
    f"Best interaction 5-fold AUC: "
    f"{final_mean:.6f}"
)

print(
    f"Improvement: "
    f"{final_delta:+.6f}"
)

print(
    f"Final Std: "
    f"{final_std:.6f}"
)

print("\nFinal folds:")

for i, score in enumerate(
    final_folds,
    start=1
):
    print(
        f"Fold {i}: {score:.6f}"
)


# ============================================================
# 19. Save all results
# ============================================================

combination_df.to_csv(
    r"D:\kaggle\airline\interaction_combination_results.csv",
    index=False
)

summary = pd.DataFrame([{
    "baseline_5fold_auc": baseline_mean,
    "baseline_5fold_std": baseline_std,
    "best_interactions": best_features_text,
    "best_5fold_auc": final_mean,
    "best_5fold_std": final_std,
    "improvement": final_delta,
}])

summary.to_csv(
    r"D:\kaggle\airline\interaction_final_summary.csv",
    index=False
)


# ============================================================
# 20. Finish
# ============================================================

elapsed = time.time() - start_time

print("\n" + "=" * 80)
print("ALL EXPERIMENTS FINISHED")
print("=" * 80)

print(
    f"Total elapsed time: "
    f"{elapsed / 60:.2f} minutes"
)

print("\nGenerated files:")

print(
    "1. D:\\kaggle\\airline\\interaction_screening_results.csv"
)

print(
    "2. D:\\kaggle\\airline\\interaction_combination_results.csv"
)

print(
    "3. D:\\kaggle\\airline\\interaction_final_summary.csv"
)