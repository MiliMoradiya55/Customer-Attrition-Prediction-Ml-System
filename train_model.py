import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os

# ============================================
# CONFIGURATION
# ============================================
# Path to your CSV file (change this to your actual file path)
CSV_FILE = "Telco_Customer_Churn_Dataset.csv"   # <--- UPDATE THIS

# ============================================
# DATA LOADING AND PREPROCESSING
# ============================================
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"CSV file '{CSV_FILE}' not found. Please provide the file.")

print(f"Loading data from {CSV_FILE}...")
data = pd.read_csv(CSV_FILE)

# Display basic info
print(f"Original data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# ============================================
# SELECT AND RENAME REQUIRED COLUMNS
# ============================================
# Map dataset columns to our model's expected names
column_mapping = {
    'tenure': 'tenure',
    'MonthlyCharges': 'monthly_charges',
    'TotalCharges': 'total_charges',
    'Contract': 'contract',
    'PaymentMethod': 'payment_method',
    'InternetService': 'internet_service',
    'SeniorCitizen': 'senior_citizen',
    'Partner': 'partner',
    'Dependents': 'dependents',
    'Churn': 'churn'
}

# Check if all required columns exist
required_cols = list(column_mapping.keys())
for col in required_cols:
    if col not in data.columns:
        raise ValueError(f"Required column '{col}' not found in CSV.")

# Select only the needed columns and rename
data = data[required_cols].rename(columns=column_mapping)

# ============================================
# HANDLE MISSING VALUES AND DATA TYPES
# ============================================
# Convert TotalCharges to numeric (might be object/string)
data['total_charges'] = pd.to_numeric(data['total_charges'], errors='coerce')

# Drop rows with missing values (you can also impute)
initial_len = len(data)
data = data.dropna()
print(f"Dropped {initial_len - len(data)} rows with missing values.")

# --- Map binary Yes/No columns to 0/1 ---
# Columns that contain 'Yes'/'No'
binary_columns = ['partner', 'dependents']
for col in binary_columns:
    data[col] = data[col].map({'Yes': 1, 'No': 0}).astype(int)

# Map target 'churn' to binary (assuming 'Yes'/'No')
data['churn'] = data['churn'].map({'Yes': 1, 'No': 0}).astype(int)

# 'senior_citizen' is already numeric 0/1, ensure it's int
data['senior_citizen'] = data['senior_citizen'].astype(int)

# Ensure categorical columns are strings
categorical_cols = ['contract', 'payment_method', 'internet_service']
for col in categorical_cols:
    data[col] = data[col].astype(str)

# Check class distribution
print(f"\nClass distribution:\n{data['churn'].value_counts()}")

# ============================================
# PREPARE FEATURES AND TARGET
# ============================================
X = data.drop('churn', axis=1)
y = data['churn']

# Define numeric and categorical columns (using our renamed columns)
numeric_features = ['tenure', 'monthly_charges', 'total_charges']
categorical_features = ['contract', 'payment_method', 'internet_service', 
                        'senior_citizen', 'partner', 'dependents']

# Create preprocessor
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create pipeline with hyperparameters to reduce overfitting
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train pipeline
pipeline.fit(X_train, y_train)

# Predictions
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# Metrics for train set (as percentages)
train_acc = accuracy_score(y_train, y_train_pred) * 100
train_precision = precision_score(y_train, y_train_pred) * 100
train_recall = recall_score(y_train, y_train_pred) * 100
train_f1 = f1_score(y_train, y_train_pred) * 100

# Metrics for test set (as percentages)
test_acc = accuracy_score(y_test, y_test_pred) * 100
test_precision = precision_score(y_test, y_test_pred) * 100
test_recall = recall_score(y_test, y_test_pred) * 100
test_f1 = f1_score(y_test, y_test_pred) * 100

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# Cross-validation score
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
cv_mean = cv_scores.mean() * 100
cv_std = cv_scores.std() * 100

# Overfitting check
accuracy_gap = train_acc - test_acc

# Display results
print("\n" + "=" * 70)
print("MODEL PERFORMANCE REPORT (ALL METRICS IN %)")
print("=" * 70)
print(f"\n📊 Training Set Metrics:")
print(f"  Accuracy   : {train_acc:.2f}%")
print(f"  Precision  : {train_precision:.2f}%")
print(f"  Recall     : {train_recall:.2f}%")
print(f"  F1-Score   : {train_f1:.2f}%")

print(f"\n📈 Test Set Metrics:")
print(f"  Accuracy   : {test_acc:.2f}%")
print(f"  Precision  : {test_precision:.2f}%")
print(f"  Recall     : {test_recall:.2f}%")
print(f"  F1-Score   : {test_f1:.2f}%")

print(f"\n🔍 Cross-Validation (5-fold on training set):")
print(f"  Mean Accuracy: {cv_mean:.2f}% (±{cv_std:.2f}%)")

print(f"\n📉 Confusion Matrix (Test Set):")
print(cm)

print(f"\n📋 Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=['No Churn', 'Churn']))

print("\n" + "=" * 70)
print("OVERFITTING ANALYSIS")
print("=" * 70)
print(f"Train Accuracy - Test Accuracy = {accuracy_gap:.2f}%")
if accuracy_gap > 5:
    print("⚠️  Warning: Possible overfitting detected (gap > 5%).")
    print("   Consider further regularization or more data.")
elif accuracy_gap < -5:
    print("⚠️  Warning: Test accuracy is higher than train accuracy (unusual). Check data distribution.")
else:
    print("✅ Model is not overfitting (gap within acceptable range).")

if abs(test_acc - cv_mean) > 5:
    print("⚠️  Test accuracy deviates significantly from cross-validation mean.")
else:
    print("✅ Test accuracy aligns with cross-validation estimates.")
print("=" * 70)

# Save the pipeline
joblib.dump(pipeline, 'churn_model.pkl')
print("\n💾 Model saved as churn_model.pkl")