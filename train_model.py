import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
import os

# Konfigurasi Grafik
sns.set(style="white")
OUTPUT_DIR = "hasil_model"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("--- MEMULAI FASE 3: TRAINING MODEL AI (Fokus: Recall & F1) ---")

# 1. LOAD DATA
df = pd.read_csv('dataset_hipertensi_clean.csv')

# 2. PEMISAHAN FITUR (X) DAN TARGET (y)
# Drop kolom target & kolom 'bocoran' (tensi)
# KOREKSI: Tambahkan error='ignore' agar tidak error kalau kolom tidak ada
X = df.drop(columns=['label_hypertension', 'bp_systolic', 'bp_diastolic'], errors='ignore')
y = df['label_hypertension']

# -------------------------------------------------------------
# LANGKAH PERBAIKAN: HANYA AMBIL KOLOM ANGKA
# Ini akan membuang kolom teks seperti 'father_health' (AB, BCD) secara otomatis
# -------------------------------------------------------------
X = X.select_dtypes(include=[np.number])

# Cek apakah masih ada missing value (NaN) yang lolos
# Jika ada, isi dengan 0 (Safe Fallback)
X = X.fillna(0)

print(f"Fitur Input Final: {len(X.columns)} variabel")
print(f"Daftar Fitur: {X.columns.tolist()}")
print(f"Proporsi Target: {y.mean():.2%} Hipertensi")

# 3. SPLIT DATA (80% Train - 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. INISIALISASI MODEL
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "XGBoost": xgb.XGBClassifier(scale_pos_weight=(len(y)-sum(y))/sum(y), use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# 6. TRAINING & EVALUASI
results = {}

print("\n--- HASIL EVALUASI ---")
print(f"{'Algoritma':<20} | {'Akurasi':<8} | {'Recall (Sakit)':<15} | {'F1-Score':<8}")
print("-" * 60)

for name, model in models.items():
    # Latih model
    model.fit(X_train_scaled, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test_scaled)
    
    # Hitung Metrik
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred) # Fokus deteksi kelas 1
    f1 = f1_score(y_test, y_pred)
    
    # Simpan hasil
    results[name] = {'model': model, 'recall': rec, 'f1': f1, 'acc': acc}
    
    print(f"{name:<20} | {acc:.2%}   | {rec:.2%}           | {f1:.2%}")
    
    # Simpan Visualisasi Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
    plt.title(f'CM - {name}\nRecall: {rec:.2%}')
    plt.xlabel('Prediksi AI')
    plt.ylabel('Fakta Medis')
    plt.savefig(f"{OUTPUT_DIR}/cm_{name.replace(' ', '_')}.png")
    plt.close()

# 7. KESIMPULAN
# Mencari model dengan Recall terbaik
best_model_name = max(results, key=lambda x: results[x]['recall'])
print("\n============================================")
print(f"JUARA ALGORITMA (RECALL TERTINGGI): {best_model_name}")
print(f"Recall   : {results[best_model_name]['recall']:.2%}")
print(f"F1-Score : {results[best_model_name]['f1']:.2%}")
print("============================================")

# Simpan Feature Importance dari Juara
best_model = results[best_model_name]['model']
plt.figure(figsize=(10, 6))
plt.title(f"Faktor Risiko Terpenting ({best_model_name})")

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
elif hasattr(best_model, 'coef_'): # Untuk Logistic Regression
    importances = abs(best_model.coef_[0])
    indices = np.argsort(importances)[::-1]
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_best.png")
print("Grafik Feature Importance tersimpan.")

print(f"Cek folder '{OUTPUT_DIR}' untuk hasil lengkap.")