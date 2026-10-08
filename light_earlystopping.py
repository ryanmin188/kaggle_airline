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
# 3. 3-Fold CV
# ==========================================
skf = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)


# ==========================================
# 4. Learning rates
# ==========================================
learning_rates = [0.03, 0.02, 0.01]

results = []


# ==========================================
# 5. Experiments
# ==========================================
for learning_rate in learning_rates:

    fold_scores = []
    best_iterations = []

    print("\n" + "=" * 70)
    print(f"learning_rate = {learning_rate}")
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

            n_estimators=3000,
            learning_rate=learning_rate,

            num_leaves=127,
            max_depth=-1,

            colsample_bytree=0.8,

            random_state=42,
            n_jobs=-1,

            verbosity=-1
        )


        # ==================================
        # Train with Early Stopping
        # ==================================
        model.fit(
            X_train,
            y_train,

            categorical_feature=categorical_features,

            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",

            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=100,
                    verbose=False
                )
            ]
        )


        # ==================================
        # Best iteration
        # ==================================
        best_iteration = model.best_iteration_

        best_iterations.append(best_iteration)


        # ==================================
        # Validation prediction
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
            f"Fold {fold}: "
            f"AUC = {auc:.6f}, "
            f"best_iteration = {best_iteration}"
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

    mean_best_iteration = sum(
        best_iterations
    ) / len(best_iterations)


    results.append({
        "learning_rate": learning_rate,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_best_iteration": mean_best_iteration
    })


# ==========================================
# 6. Results
# ==========================================
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "mean_auc",
    ascending=False
)


print("\n" + "=" * 70)
print("EARLY STOPPING RESULTS")
print("=" * 70)

print(
    results_df.to_string(index=False)
)