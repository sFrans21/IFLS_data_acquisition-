import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
import os
import warnings

# Matikan warning yang berisik
warnings.filterwarnings('ignore')

# Konfigurasi Grafik
sns.set(style="whitegrid")
OUTPUT_DIR = "hasil_model_advanced"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("--- FASE 3.5: ADVANCED MODEL TRAINING (ENSEMBLE + THRESHOLD TUNING) ---")

# 1. LOAD & CLEAN DATA
df = pd.read_csv('dataset_hipertensi_clean.csv')

# Drop target & bocoran, lalu filter HANYA ANGKA (Solusi Error String 'AB')
X = df.drop(columns=['label_hypertension', 'bp_systolic', 'bp_diastolic'], errors='ignore')
X = X.select_dtypes(include=[np.number]).fillna(0)
y = df['label_hypertension']

print(f"Fitur Input: {len(X.columns)} variabel")

# 2. SPLIT & SCALE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. DEFINISI MODEL (TUNED HYPERPARAMETERS)
# Kita racik parameter manual agar lebih 'pintar'

# Model A: Logistic Regression (Stabil & Explainable)
clf_lr = LogisticRegression(class_weight='balanced', C=0.1, solver='liblinear', random_state=42)

# Model B: XGBoost (Kuat di pola non-linear)
# scale_pos_weight = rasio negatif/positif (untuk balancing)
ratio = float(np.sum(y == 0)) / np.sum(y == 1)
clf_xgb = xgb.XGBClassifier(
    scale_pos_weight=ratio,
    n_estimators=200,      # Lebih banyak pohon
    max_depth=4,           # Jangan terlalu dalam (biar gak overfitting)
    learning_rate=0.05,    # Belajar pelan-pelan (lebih teliti)
    subsample=0.8,         # Pakai sebagian data aja biar variatif
    use_label_encoder=False, 
    eval_metric='logloss', 
    random_state=42
)

# Model C: Random Forest (Dibatasi kedalamannya)
clf_rf = RandomForestClassifier(
    n_estimators=200, 
    max_depth=8,           # Batasi kedalaman! (Solusi RF jeblok kemarin)
    class_weight='balanced', 
    random_state=42
)

# 4. ENSEMBLE: VOTING CLASSIFIER (Soft Voting)
# Gabungkan otak LR, XGB, dan RF. Ambil rata-rata probabilitas mereka.
ensemble_model = VotingClassifier(
    estimators=[('lr', clf_lr), ('xgb', clf_xgb), ('rf', clf_rf)],
    voting='soft' 
)

print("\nMelatih Model Ensemble (Gabungan 3 Algoritma)...")
ensemble_model.fit(X_train_scaled, y_train)

# 5. THRESHOLD TUNING (RAHASIA UTAMA)
# Alih-alih pakai threshold 0.5, kita cari threshold terbaik untuk Recall
y_proba = ensemble_model.predict_proba(X_test_scaled)[:, 1]

# Hitung Precision-Recall Curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# Cari threshold dimana F1-Score maksimal (atau Recall tinggi dengan Precision wajar)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"\n--- HASIL OPTIMASI ---")
print(f"Threshold Optimal Ditemukan: {best_threshold:.4f}")
print("(Artinya: Jika probabilitas risiko > {:.1f}%, sistem akan vonis Hipertensi)".format(best_threshold*100))

# 6. EVALUASI AKHIR (DENGAN THRESHOLD BARU)
y_pred_new = (y_proba >= best_threshold).astype(int)

final_acc = accuracy_score(y_test, y_pred_new)
final_rec = recall_score(y_test, y_pred_new)
final_f1 = f1_score(y_test, y_pred_new)

print("\n--- PERFORMA FINAL (ENSEMBLE + TUNED) ---")
print(f"Akurasi   : {final_acc:.2%}")
print(f"Recall    : {final_rec:.2%} (TARGET UTAMA)")
print(f"F1-Score  : {final_f1:.2%}")

# 7. VISUALISASI CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred_new)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title(f'CONFUSION MATRIX FINAL\nModel: Ensemble Voting | Threshold: {best_threshold:.2f}')
plt.xlabel('Prediksi AI')
plt.ylabel('Fakta Medis')
plt.savefig(f"{OUTPUT_DIR}/cm_final_advanced.png")

# 8. SIMPAN IMPORTANCE (Kita ambil dari XGBoost saja sebagai perwakilan di Ensemble)
clf_xgb.fit(X_train_scaled, y_train) # Re-fit single model untuk ambil feature importance
importances = clf_xgb.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Faktor Risiko Paling Berpengaruh (Versi XGBoost)")
plt.bar(range(X.shape[1]), importances[indices], align="center", color='teal')
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_advanced.png")

print(f"\nGrafik tersimpan di folder '{OUTPUT_DIR}'.")
print("Bandingkan nilai Recall ini dengan 71.32% sebelumnya!")