import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import lightgbm as lgb
import joblib
import os

RAW_CSV_PATH = "../data/raw/gdm_dataset.csv"
MODEL_SAVE_PATH = "../models/child_outcome_model_full.joblib"

print("Loading raw dataset...")
df = pd.read_csv(RAW_CSV_PATH, low_memory=False)

# Define target
df['SCBU_adm'] = df['SCBU admission (Yes/NO)'].map({'Yes':1, 'No':0})

# Use full 43 features except target and UID (not a feature)
features = [
    "Index of Multiple Deprivation Rank",
    "IMD Decile",
    "AgeAtStartOfSpell",
    "WeightMeasured",
    "Height",
    "Body Mass Index at Booking",
    "Obese?",
    "Ethnicity",
    "Risk Factors",
    "AntenatalMedicalFactors",
    "PreviousObstetricHistory",
    "Parity",
    "Gravida",
    "Glucoselevelblood",
    "GlucoseToleranceTest",
    "Glucoselevel0minblood",
    "Glucoselevel120minblood",
    "FolicAcidDose",
    "SystolicBloodPressureCuff",
    "Diastolic Blood Pressure",
    "VitaminDlevelblood",
    "O_Thyroidfunctionblood",
    "Delivery_Outcome",
    "OnsetofLabourMethod",
    "Contraction frequency prior to delivery",
    "PrimaryIndicationforCaesarean",
    "Category Caesarean Section",
    "Perineal care",
    "EstimatedTotalBloodLoss",
    "Gestation",
    "Severely Premature?",
    "Gestation (Days)",
    "Gestation at booking (Weeks)",
    "No_Of_previous_Csections",
    "BabyBirthWeight",
    "Presence of meconium",
    "BW Centile",
    "Shoulder Dystocia",
    "LOS mother after delivery",
    "Sex",
    "Still_Birth",
    "TotalApgarScoreat1minutes",
    "APGAR_Score_5",
    "TotalApgarScoreat10minutes",
    "Maternity_Month"
]

df = df.dropna(subset=['SCBU_adm'])  # drop missing target rows

# Map categorical variables with sample strategies
# You may want more advanced encoding for fields with complex categories like Risk Factors

df['Obese?'] = df['Obese?'].map({'Yes':1, 'No':0}).fillna(0).astype(int)
df['Ethnicity'] = df['Ethnicity'].fillna('Unknown')
ethnicity_map = {eth: idx for idx, eth in enumerate(df['Ethnicity'].unique())}
df['Ethnicity'] = df['Ethnicity'].map(ethnicity_map)

# For multiple complex categorical features, convert to numerical by label encoding or dummy variables
# Here, using label encoding for demo purpose
for col in ['Risk Factors', 'AntenatalMedicalFactors', 'PreviousObstetricHistory', 
            'GlucoseToleranceTest', 'FolicAcidDose', 'PrimaryIndicationforCaesarean',
            'Category Caesarean Section', 'Perineal care', 'Delivery_Outcome',
            'OnsetofLabourMethod', 'Still_Birth', 'Maternity_Month']:
    df[col] = df[col].fillna('Unknown')
    df[col] = df[col].astype(str)
    df[col] = pd.factorize(df[col])[0]

# Map boolean categorical
df['Presence of meconium'] = df['Presence of meconium'].map({'Yes':1, 'No':0}).fillna(0).astype(int)
df['Shoulder Dystocia'] = df['Shoulder Dystocia'].map({'Yes':1, 'No':0}).fillna(0).astype(int)

# Map Sex
sex_map = {'Male':0, 'Female':1}
df['Sex'] = df['Sex'].map(sex_map).fillna(2).astype(int)

# Map Severely Premature?
df['Severely Premature?'] = df['Severely Premature?'].map({'Yes':1, 'No':0}).fillna(0).astype(int)

# Convert numeric columns with coercion and fill nan with median
numeric_cols = list(set(features) - {
    'Obese?', 'Ethnicity', 'Risk Factors', 'AntenatalMedicalFactors', 'PreviousObstetricHistory',
    'GlucoseToleranceTest', 'FolicAcidDose', 'PrimaryIndicationforCaesarean', 'Category Caesarean Section',
    'Perineal care', 'Delivery_Outcome', 'OnsetofLabourMethod', 'Presence of meconium',
    'Shoulder Dystocia', 'Sex', 'Still_Birth', 'Severely Premature?', 'Maternity_Month'})

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

X = df[features]
y = df['SCBU_adm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'seed': 42
}

model = lgb.train(params, train_data, valid_sets=[valid_data], callbacks=[lgb.early_stopping(stopping_rounds=20)])

y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(classification_report(y_test, y_pred))

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
joblib.dump(model, MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")