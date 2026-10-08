import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# ==========================================
# 1. Load data
# ==========================================
train_path = r"D:\kaggle\airline\data\train.csv"

train = pd.read_csv(train_path)

print("Data shape:", train.shape)


# ==========================================
# 2. X and y
# ==========================================
X = train.drop(columns=["satisfaction", "id"])
y = train["satisfaction"].astype(int)


# ==========================================
# 3. Categorical variables
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
# 4. 5-Fold Stratified Cross-Validation
# ==========================================
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Store AUC for each fold
fold_scores = []


print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION")
print("=" * 70)


# ==========================================
# 5. Cross-validation
# ==========================================
for fold, (train_idx, valid_idx) in enumerate(
    skf.split(X, y),
    start=1
):

    print(f"\n{'-' * 60}")
    print(f"Fold {fold}")
    print(f"{'-' * 60}")

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    print("Training size:", X_train.shape)
    print("Validation size:", X_valid.shape)


    # ======================================
    # LightGBM model
    # Same parameters as our baseline
    # ======================================
    model = lgb.LGBMClassifier(
        objective="binary",

        n_estimators=500,
        learning_rate=0.05,

        num_leaves=31,
        max_depth=-1,

        colsample_bytree=0.8,

        random_state=42,
        n_jobs=-1
    )


    # ======================================
    # Train
    # ======================================
    model.fit(
        X_train,
        y_train,
        categorical_feature=categorical_features
    )


    # ======================================
    # Validation prediction
    # ======================================
    y_pred_proba = model.predict_proba(
        X_valid
    )[:, 1]


    # ======================================
    # ROC-AUC
    # ======================================
    auc = roc_auc_score(
        y_valid,
        y_pred_proba
    )

    fold_scores.append(auc)

    print(f"Fold {fold} ROC-AUC: {auc:.6f}")


# ==========================================
# 6. Final CV result
# ==========================================
mean_auc = sum(fold_scores) / len(fold_scores)

std_auc = (
    sum((score - mean_auc) ** 2 for score in fold_scores)
    / len(fold_scores)
) ** 0.5


print("\n" + "=" * 70)
print("CROSS-VALIDATION RESULT")
print("=" * 70)

for i, score in enumerate(fold_scores, start=1):
    print(f"Fold {i}: {score:.6f}")

print("-" * 70)

print(f"Mean ROC-AUC: {mean_auc:.6f}")
print(f"Std ROC-AUC:  {std_auc:.6f}")