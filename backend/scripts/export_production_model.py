import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
import json

print("Membangun Model Production Xgboost...")

# 1. Atur Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Kita asumsikan CSV lu dipindah ke folder backend/data/
DATA_PATH = os.path.join(BASE_DIR, 'data', 'dataset_hipertensi_clean.csv') 
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. Load Data & Preprocessing (Sama Persis dengan Lab)
print("-> Membaca dan membersihkan data...")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['label_hypertension', 'bp_systolic', 'bp_diastolic'], errors='ignore')

if 'ak05' in X.columns:
    X['ak05'] = pd.to_numeric(X['ak05'], errors='coerce')

X = X.select_dtypes(include=[np.number])

# Simpan nilai median untuk dipakai di API Backend nanti
imputation_values = {}

if 'waist_cm' in X.columns:
    median_waist = float(X['waist_cm'].median())
    imputation_values['waist_cm'] = median_waist
    X['waist_cm'] = X['waist_cm'].fillna(median_waist)

ps_cols = ['ps_A', 'ps_B', 'ps_C', 'ps_E', 'ps_F']
for col in ps_cols:
    if col in X.columns:
        med_val = float(X[col].median())
        imputation_values[col] = med_val
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(med_val)

X = X.fillna(0)
y = df['label_hypertension']

# 3. Training XGBoost Final (Menggunakan Seluruh Data agar maksimal)
print("-> Melatih model XGBoost Production...")
ratio = int((y == 0).sum()) / int(y.sum())
model = xgb.XGBClassifier(
    scale_pos_weight=ratio,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    eval_metric='logloss',
    random_state=42
)

# KITA FIT MENGGUNAKAN RAW DATA X (TIDAK DI-SCALE)
model.fit(X, y)

# 4. Saving Artifacts ke Backend
print("-> Menyimpan Artifacts ke backend/models/saved_models/ ...")
model_path = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
imputation_path = os.path.join(MODEL_DIR, 'imputation_values.json')

joblib.dump(model, model_path)

with open(imputation_path, 'w') as f:
    json.dump(imputation_values, f, indent=4)

print("\n BERHASIL!")
print(f"1. Model tersimpan di: {model_path}")
print(f"2. Nilai Imputasi API tersimpan di: {imputation_path}")
print("\n[PERHATIAN UNTUK BACKEND]:")
print("- JANGAN gunakan StandardScaler di ml_service.py")
print("- Pastikan lu ngambil threshold optimal dari hasil train_model_v3.py lu sebelumnya untuk dipakai di API.")

