import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# ==========================================
# 1. Load data
# ==========================================
train_path = r"D:\kaggle\airline\data\train.csv"

train = pd.read_csv(train_path)

print("Data shape:", train.shape)


# ==========================================
# 2. Define X and y
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
# 4. Train / validation split
# ==========================================
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining size:", X_train.shape)
print("Validation size:", X_valid.shape)


# ==========================================
# 5. LightGBM model
# ==========================================
model = lgb.LGBMClassifier(
    objective="binary",

    n_estimators=500,
    learning_rate=0.05,

    num_leaves=31,
    max_depth=-1,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42,
    n_jobs=-1
)


# ==========================================
# 6. Train
# ==========================================
print("\nTraining LightGBM...")

model.fit(
    X_train,
    y_train,
    categorical_feature=categorical_features
)


# ==========================================
# 7. Validation prediction
# ==========================================
y_pred_proba = model.predict_proba(X_valid)[:, 1]


# ==========================================
# 8. ROC-AUC
# ==========================================
auc = roc_auc_score(
    y_valid,
    y_pred_proba
)

print("\n" + "=" * 60)
print("LIGHTGBM BASELINE RESULT")
print("=" * 60)

print(f"Validation ROC-AUC: {auc:.6f}")