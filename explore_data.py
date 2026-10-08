import pandas as pd
import numpy as np

# =========================
# 1. Load data
# =========================
train_path = r"D:\kaggle\airline\data\train.csv"

train = pd.read_csv(train_path)

print("=" * 70)
print("1. BASIC INFORMATION")
print("=" * 70)

print("Shape:", train.shape)
print("\nData types:")
print(train.dtypes)


# =========================
# 2. Separate feature types
# =========================
print("\n" + "=" * 70)
print("2. FEATURE TYPES")
print("=" * 70)

numeric_cols = train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train.select_dtypes(exclude=np.number).columns.tolist()

print("\nNumeric columns:")
print(numeric_cols)

print("\nCategorical columns:")
print(categorical_cols)


# =========================
# 3. Unique values
# =========================
print("\n" + "=" * 70)
print("3. UNIQUE VALUES OF CATEGORICAL VARIABLES")
print("=" * 70)

for col in categorical_cols:
    print(f"\n{col}:")
    print(train[col].value_counts(dropna=False))


# =========================
# 4. Target distribution
# =========================
print("\n" + "=" * 70)
print("4. TARGET DISTRIBUTION")
print("=" * 70)

target_rate = train["satisfaction"].mean()

print("Mean of satisfaction:", target_rate)
print("\nCounts:")
print(train["satisfaction"].value_counts())

print("\nProportions:")
print(train["satisfaction"].value_counts(normalize=True))


# =========================
# 5. Missing values
# =========================
print("\n" + "=" * 70)
print("5. MISSING VALUES")
print("=" * 70)

missing = train.isna().sum()
missing_pct = train.isna().mean() * 100

missing_table = pd.DataFrame({
    "missing_count": missing,
    "missing_pct": missing_pct
})

print(missing_table[missing_table["missing_count"] > 0])


# =========================
# 6. Numerical variable summary
# =========================
print("\n" + "=" * 70)
print("6. NUMERICAL VARIABLES")
print("=" * 70)

print(
    train[numeric_cols]
    .describe()
    .T
    .to_string()
)


# =========================
# 7. Satisfaction rate by categorical variables
# =========================
print("\n" + "=" * 70)
print("7. SATISFACTION RATE BY CATEGORICAL VARIABLES")
print("=" * 70)

categorical_for_analysis = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class"
]

for col in categorical_for_analysis:
    print(f"\n--- {col} ---")

    result = (
        train.groupby(col)["satisfaction"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )

    result["mean"] = result["mean"] * 100

    print(result)


# =========================
# 8. Satisfaction rate by age groups
# =========================
print("\n" + "=" * 70)
print("8. SATISFACTION RATE BY AGE GROUP")
print("=" * 70)

train["Age_Group"] = pd.cut(
    train["Age"],
    bins=[0, 18, 30, 40, 50, 60, 100],
    labels=["<=18", "19-30", "31-40", "41-50", "51-60", "61+"]
)

age_result = (
    train.groupby("Age_Group", observed=False)["satisfaction"]
    .agg(["mean", "count"])
)

age_result["mean"] = age_result["mean"] * 100

print(age_result)


# =========================
# 9. Satisfaction by service ratings
# =========================
print("\n" + "=" * 70)
print("9. SERVICE RATINGS")
print("=" * 70)

service_cols = [
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
    "Cleanliness"
]

service_result = []

for col in service_cols:
    temp = (
        train.groupby("satisfaction")[col]
        .mean()
        .to_dict()
    )

    service_result.append({
        "feature": col,
        "unsatisfied_mean": temp.get(False, np.nan),
        "satisfied_mean": temp.get(True, np.nan),
        "difference": (
            temp.get(True, np.nan)
            - temp.get(False, np.nan)
        )
    })

service_result = (
    pd.DataFrame(service_result)
    .sort_values("difference", ascending=False)
)

print(service_result.to_string(index=False))


# =========================
# 10. Numerical variables by target
# =========================
print("\n" + "=" * 70)
print("10. NUMERICAL VARIABLES BY SATISFACTION")
print("=" * 70)

continuous_cols = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes"
]

for col in continuous_cols:

    result = (
        train.groupby("satisfaction")[col]
        .agg(["mean", "median"])
    )

    print(f"\n--- {col} ---")
    print(result)


# =========================
# 11. Correlation with target
# =========================
print("\n" + "=" * 70)
print("11. CORRELATION WITH TARGET")
print("=" * 70)

corr_data = train[numeric_cols].copy()
corr_data["satisfaction"] = train["satisfaction"].astype(int)

corr = (
    corr_data.corr(numeric_only=True)["satisfaction"]
    .sort_values(ascending=False)
)

print(corr)


print("\n" + "=" * 70)
print("EDA FINISHED")
print("=" * 70)