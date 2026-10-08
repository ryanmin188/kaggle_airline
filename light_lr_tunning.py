import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# ==========================================
# 1. Load data
# ==========================================
train_path = r"D:\kaggle\airline\data\train.csv"

train = pd.read_csv(train_path)

X = train.drop(columns=["satisfaction", "id"])
y = train["satisfaction"].astype(int)


# ==========================================
# 2. Categorical variables
# ==========================================
categorical_features = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class"
]

for col in categorical_features:
    X[col] = X[col].astype("category")


# ==========================================
# 3. 3-Fold Cross-Validation
# ==========================================
skf = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


# ==========================================
# 4. Parameter combinations
# ==========================================
configs = [
    {"learning_rate": 0.10, "n_estimators": 300},
    {"learning_rate": 0.05, "n_estimators": 500},
    {"learning_rate": 0.03, "n_estimators": 800},
    {"learning_rate": 0.02, "n_estimators": 1200},
]


results = []


# ==========================================
# 5. Experiments
# ==========================================
for config in configs:

    learning_rate = config["learning_rate"]
    n_estimators = config["n_estimators"]

    fold_scores = []

    print("\n" + "=" * 70)
    print(
        f"learning_rate = {learning_rate}, "
        f"n_estimators = {n_estimators}"
    )
    print("=" * 70)


    for fold, (train_idx, valid_idx) in enumerate(
        skf.split(X, y),
        start=1
    ):

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]


        # ==================================
        # LightGBM
        # ==================================
        model = lgb.LGBMClassifier(
            objective="binary",

            n_estimators=n_estimators,
            learning_rate=learning_rate,

            num_leaves=127,
            max_depth=-1,

            colsample_bytree=0.8,

            random_state=42,
            n_jobs=-1,

            verbosity=-1
        )


        # ==================================
        # Train
        # ==================================
        model.fit(
            X_train,
            y_train,
            categorical_feature=categorical_features
        )


        # ==================================
        # Predict
        # ==================================
        y_pred_proba = model.predict_proba(
            X_valid
        )[:, 1]


        # ==================================
        # ROC-AUC
        # ==================================
        auc = roc_auc_score(
            y_valid,
            y_pred_proba
        )

        fold_scores.append(auc)

        print(
            f"Fold {fold}: {auc:.6f}"
        )


    # ==================================
    # Mean / Std
    # ==================================
    mean_auc = sum(fold_scores) / len(fold_scores)

    variance = sum(
        (score - mean_auc) ** 2
        for score in fold_scores
    ) / len(fold_scores)

    std_auc = variance ** 0.5


    results.append({
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "mean_auc": mean_auc,
        "std_auc": std_auc
    })


# ==========================================
# 6. Final result
# ==========================================
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "mean_auc",
    ascending=False
)


print("\n" + "=" * 70)
print("LEARNING RATE / N_ESTIMATORS RESULTS")
print("=" * 70)

print(
    results_df.to_string(index=False)
)