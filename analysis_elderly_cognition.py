# analysis_elderly_cognition.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from scipy.stats import zscore

# -------------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------------

csv_path = r"C:\Users\Leon Parker\Documents\Coding\Data-analysis\Cognitive health dataset\cognitive_impairment_dataset.csv"

df = pd.read_csv(csv_path)

print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# -------------------------------------------------------------------
# 2. BASIC CLEANING
# -------------------------------------------------------------------

# Check missing values
print("\nMissing values per column:")
print(df.isna().sum())

# For this dataset there are no missing values, but we keep the logic
# so the script looks generalisable / robust.

num_cols = [
    "Age", "Chronic_Diseases", "Glucose_Level",
    "BMI", "MMSE_Score", "GDS_Score",
    "Sleep_Quality_Score", "Physical_Activity_Score"
]

cat_cols = [
    "Gender", "Education_Level", "Region",
    "Marital_Status", "Smoking_Status", "Alcohol_Use"
]

# Drop ID column – not useful for modelling
if "Participant_ID" in df.columns:
    df = df.drop(columns=["Participant_ID"])

# Simple numeric imputation with median (safe even if no NaNs)
for col in num_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Simple categorical imputation with mode
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing values after simple imputation:")
print(df.isna().sum())

# -------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# -------------------------------------------------------------------

# Age groups
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[59, 69, 79, 120],
    labels=["60–69", "70–79", "80+"]
)

# Binary lifestyle encodings for convenience
df["Smoker"] = df["Smoking_Status"].map({"No": 0, "Yes": 1})
df["Alcohol_User"] = df["Alcohol_Use"].map({"No": 0, "Yes": 1})

# Combine chronic conditions into a simple risk score (already a count)
df["Chronic_Risk_Score"] = df["Chronic_Diseases"]

# Target: Cognitive_Impairment_Status (already 0/1)
target = "Cognitive_Impairment_Status"

print("\nValue counts for target:")
print(df[target].value_counts())

# -------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------------

# Create a folder for figures
fig_dir = "figures"
os.makedirs(fig_dir, exist_ok=True)

# 4.1 Age distribution
plt.figure()
df["Age"].hist(bins=20)
plt.title("Age distribution")
plt.xlabel("Age (years)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "age_distribution.png"))
plt.close()

# 4.2 Cognitive impairment counts
plt.figure()
df[target].value_counts().sort_index().plot(kind="bar")
plt.title("Cognitive Impairment Status (0 = No, 1 = Yes)")
plt.xlabel("Cognitive_Impairment_Status")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "target_counts.png"))
plt.close()

# 4.3 Impairment rate by age group
imp_rate_by_age = (
    df.groupby("Age_Group")[target]
      .mean()
      .reset_index()
)

plt.figure()
sns.barplot(data=imp_rate_by_age, x="Age_Group", y=target)
plt.title("Cognitive impairment rate by age group")
plt.ylabel("Proportion impaired")
plt.xlabel("Age group")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "impairment_by_age_group.png"))
plt.close()

# 4.4 Impairment rate by region
imp_rate_by_region = (
    df.groupby("Region")[target]
      .mean()
      .reset_index()
)

plt.figure()
sns.barplot(data=imp_rate_by_region, x="Region", y=target)
plt.title("Cognitive impairment rate by region")
plt.ylabel("Proportion impaired")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "impairment_by_region.png"))
plt.close()

# 4.5 Correlation heatmap for numeric features
numeric_for_corr = [
    "Age", "Chronic_Diseases", "Glucose_Level",
    "BMI", "MMSE_Score", "GDS_Score",
    "Sleep_Quality_Score", "Physical_Activity_Score",
    "Smoker", "Alcohol_User", target
]

corr = df[numeric_for_corr].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heatmap – numeric features")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "correlation_heatmap.png"))
plt.close()

print(f"\nEDA plots saved to folder: {fig_dir}")

# -------------------------------------------------------------------
# 5. ADVANCED MODELLING – OUTLIERS + SCALING + RANDOM OVERSAMPLING
# -------------------------------------------------------------------

# 5.1 Outlier detection on numeric features (z-score)
numeric_df = df[num_cols]  # original numeric clinical features only

z_scores = np.abs(zscore(numeric_df))
threshold = 3  # common z-score threshold for outliers
outlier_indices = np.where(z_scores > threshold)[0]

print(f"\nOutliers detected (z-score > {threshold}): {len(outlier_indices)}")
# We are NOT dropping them because the dataset is small/imbalanced,
# but this is useful to mention in an analysis or report.

# 5.2 Prepare X/y
X = df.drop(columns=[target])
y = df[target]

# Re-identify numeric and categorical columns after feature engineering
num_cols_model = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_model = X.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nNumeric features for model:", num_cols_model)
print("Categorical features for model:", cat_cols_model)

# 5.3 Preprocessing: scale numeric, one-hot encode categorical
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols_model),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_model),
    ]
)

# 5.4 Model definition
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    random_state=42
)

# 5.5 Pipeline with RandomOverSampler (fix extreme imbalance)
resampler = RandomOverSampler(random_state=42)

clf = ImbPipeline(steps=[
    ("preprocess", preprocess),
    ("resample", resampler),
    ("model", model),
])

# 5.6 Train-test split (stratified to preserve class proportion in test set)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nClass distribution in training set before oversampling:")
print(y_train.value_counts())

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\n--- Classification report (after oversampling + scaling) ---")
print(classification_report(y_test, y_pred))

try:
    auc = roc_auc_score(y_test, y_proba)
    print("ROC-AUC:", auc)
except ValueError:
    print("ROC-AUC could not be computed")

# -------------------------------------------------------------------
# 6. FEATURE IMPORTANCE (AFTER OVERSAMPLING + SCALING)
# -------------------------------------------------------------------

rf = clf.named_steps["model"]
pre = clf.named_steps["preprocess"]

cat_encoder = pre.named_transformers_["cat"]
cat_feature_names = cat_encoder.get_feature_names_out(cat_cols_model)

feature_names = np.concatenate([num_cols_model, cat_feature_names])
importances = rf.feature_importances_

feat_imp = pd.DataFrame(
    {"feature": feature_names, "importance": importances}
).sort_values("importance", ascending=False)

print("\nTop 15 important features (after oversampling):")
print(feat_imp.head(15))

plt.figure(figsize=(8, 6))
sns.barplot(data=feat_imp.head(15), x="importance", y="feature")
plt.title("Top 15 feature importances – oversampled + scaled model")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "feature_importances_oversampled.png"))
plt.close()

# -------------------------------------------------------------------
# 7. SAVE CLEANED DATA FOR POWER BI / TABLEAU
# -------------------------------------------------------------------

output_path = os.path.join(
    r"C:\Users\Leon Parker\Documents\Coding\Data-analysis\Cognitive health dataset",
    "cognitive_impairment_cleaned.csv"
)
df.to_csv(output_path, index=False)
print(f"\nCleaned dataset saved to: {output_path}")
