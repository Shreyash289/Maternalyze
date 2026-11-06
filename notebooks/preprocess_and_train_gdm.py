import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from imblearn.over_sampling import RandomOverSampler
import lightgbm as lgb
import joblib
import os

RAW_CSV_PATH = '../data/raw/gdm_dataset.csv'
MODEL_SAVE_PATH = '../models/gdm_classifier_small.joblib'

print("Loading raw dataset...")
df = pd.read_csv(RAW_CSV_PATH, low_memory=False)
df.columns = df.columns.str.strip()

df['GDM'] = df['Gestational Diabetes'].map({'Yes': 1, 'No': 0})

# Define only the selected features
selected_features = [
    'AgeAtStartOfSpell', 'WeightMeasured', 'Height',
    'Body Mass Index at Booking', 'Obese?', 'Ethnicity', 'Glucoselevelblood'
]

# Numeric conversion for numeric features
for col in ['AgeAtStartOfSpell', 'WeightMeasured', 'Height', 'Body Mass Index at Booking', 'Glucoselevelblood']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# Encode categoricals
for col in ['Obese?', 'Ethnicity']:
    df[col] = df[col].fillna('Unknown')
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

X = df[selected_features]
y = df['GDM']

# Balance dataset
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

X_train, X_valid, y_train, y_valid = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

params = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'random_state': 42,
}

print("Training LightGBM model on selected features...")
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, valid_data],
    num_boost_round=2000,
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]
)

# Metrics
y_pred_prob = model.predict(X_valid, num_iteration=model.best_iteration)
y_pred = (y_pred_prob >= 0.5).astype(int)

print(f"Validation Accuracy: {accuracy_score(y_valid, y_pred):.4f}")
print(f"Validation AUC: {roc_auc_score(y_valid, y_pred_prob):.4f}")
print(classification_report(y_valid, y_pred, zero_division=0))

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
joblib.dump(model, MODEL_SAVE_PATH)
print(f"Model saved as {MODEL_SAVE_PATH}")