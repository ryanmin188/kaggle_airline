import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# ==========================================
# 1. Load data
# ==========================================
train_path = r"D:\kaggle\airline\data\train.csv"

train = pd.read_csv(train_path)


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
# 4. Train / validation split
# ==========================================
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. LightGBM
# ==========================================
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


# ==========================================
# 6. Train
# ==========================================
model.fit(
    X_train,
    y_train,
    categorical_feature=categorical_features
)


# ==========================================
# 7. Feature importance - Split
# ==========================================
importance_split = pd.DataFrame({
    "feature": X.columns,
    "split_importance": model.booster_.feature_importance(
        importance_type="split"
    )
})


# ==========================================
# 8. Feature importance - Gain
# ==========================================
importance_gain = pd.DataFrame({
    "feature": X.columns,
    "gain_importance": model.booster_.feature_importance(
        importance_type="gain"
    )
})


# ==========================================
# 9. Combine Split + Gain
# ==========================================
importance = importance_split.merge(
    importance_gain,
    on="feature"
)


# Split ranking
importance["split_rank"] = (
    importance["split_importance"]
    .rank(method="min", ascending=False)
    .astype(int)
)

# Gain ranking
importance["gain_rank"] = (
    importance["gain_importance"]
    .rank(method="min", ascending=False)
    .astype(int)
)


# Sort by gain importance
importance = importance.sort_values(
    "gain_importance",
    ascending=False
)


# ==========================================
# 10. Print comparison table
# ==========================================
print("\n" + "=" * 90)
print("FEATURE IMPORTANCE: SPLIT VS GAIN")
print("=" * 90)

print(
    importance[
        [
            "feature",
            "split_importance",
            "split_rank",
            "gain_importance",
            "gain_rank"
        ]
    ].to_string(index=False)
)


# ==========================================
# 11. Top 15 by Split Importance
# ==========================================
top_n = 15

top_split = (
    importance
    .sort_values("split_importance", ascending=False)
    .head(top_n)
    .sort_values("split_importance")
)


plt.figure(figsize=(10, 7))

plt.barh(
    top_split["feature"],
    top_split["split_importance"]
)

plt.xlabel("Split Importance (Number of Splits)")
plt.ylabel("Feature")
plt.title("Top 15 Features by Split Importance")

plt.tight_layout()
plt.show()


# ==========================================
# 12. Top 15 by Gain Importance
# ==========================================
top_gain = (
    importance
    .sort_values("gain_importance", ascending=False)
    .head(top_n)
    .sort_values("gain_importance")
)


plt.figure(figsize=(10, 7))

plt.barh(
    top_gain["feature"],
    top_gain["gain_importance"]
)

plt.xlabel("Gain Importance (Total Gain)")
plt.ylabel("Feature")
plt.title("Top 15 Features by Gain Importance")

plt.tight_layout()
plt.show()