import pandas as pd
import numpy as np
import os
import joblib  # <-- INI PUSTAKA BUAT NGE-SAVE MODEL
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score

print("--- FASE 1: TRAINING & SAVING MODEL ---")

# 1. ATUR PATH FOLDER OTOMATIS
# Mengambil path folder backend secara dinamis (terhindar dari error FileNotFoundError)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'dataset_hipertensi_clean.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved_models')

# Bikin folder models/saved_models/ kalau belum ada
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. LOAD DATA
print(f"-> Membaca data dari: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['label_hypertension', 'bp_systolic', 'bp_diastolic'], errors='ignore')
X = X.select_dtypes(include=[np.number]).fillna(0)
y = df['label_hypertension']

# 3. SPLIT & SCALING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. TRAINING MODEL (Logistic Regression Juara Sementara)
print("-> Melatih model Logistic Regression...")
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluasi singkat
y_pred = model.predict(X_test_scaled)
print(f"\n[PERFORMA MODEL]")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.2%}")
print(f"Recall   : {recall_score(y_test, y_pred):.2%}")
print(f"F1-Score : {f1_score(y_test, y_pred):.2%}")

# 5. PROSES PEMBEKUAN (SAVING OBJECTS)
print("\n-> Menyimpan model dan scaler...")
model_path = os.path.join(MODEL_DIR, 'logistic_regression_model.pkl')
scaler_path = os.path.join(MODEL_DIR, 'standard_scaler.pkl')

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print("\n--- BERHASIL ---")
print(f" Model tersimpan di  : {model_path}")
print(f" Scaler tersimpan di : {scaler_path}")