import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
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
# 3. Identify feature types
# ==========================================
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


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
# 5. Preprocessing
# ==========================================

# Numerical:
# - fill missing values with median
# - standardize

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical:
# - fill missing values with most frequent category
# - one-hot encoding

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# ==========================================
# 6. Logistic Regression
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# ==========================================
# 7. Build pipeline
# ==========================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# ==========================================
# 8. Train
# ==========================================

print("\nTraining model...")

pipeline.fit(X_train, y_train)


# ==========================================
# 9. Validation prediction
# ==========================================

y_pred_proba = pipeline.predict_proba(X_valid)[:, 1]


# ==========================================
# 10. ROC-AUC
# ==========================================

auc = roc_auc_score(
    y_valid,
    y_pred_proba
)

print("\n" + "=" * 60)
print("BASELINE RESULT")
print("=" * 60)

print(f"Validation ROC-AUC: {auc:.6f}")