import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os

# Konfigurasi Grafik
plt.style.use('default') # Reset style agar kompatibel dengan SHAP
OUTPUT_DIR = "hasil_xai"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("--- FASE 4: EXPLAINABLE AI (MENGAPA MODEL MEMPREDIKSI ITU?) ---")

# 1. LOAD & PREPARE DATA (Sama seperti Training)
df = pd.read_csv('dataset_hipertensi_clean.csv')
X = df.drop(columns=['label_hypertension', 'bp_systolic', 'bp_diastolic'], errors='ignore')
X = X.select_dtypes(include=[np.number]).fillna(0)
y = df['label_hypertension']

# Split & Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Kembalikan ke DataFrame agar SHAP bisa baca nama kolom
X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# 2. LATIH ULANG MODEL JUARA (LOGISTIC REGRESSION)
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)
print("Model Juara (Logistic Regression) siap diaudit.")

# 3. MENGHITUNG SHAP VALUES
# Menggunakan LinearExplainer karena ini model linear (Sangat Cepat)
explainer = shap.LinearExplainer(model, X_train_df)
shap_values = explainer.shap_values(X_test_df)

print("\n--- GENERATING EXPLANATION PLOTS ---")

# A. SUMMARY PLOT (Paling Penting untuk Tesis)
# Menunjukkan fitur mana yang paling kuat pengaruhnya secara global
plt.figure()
shap.summary_plot(shap_values, X_test_df, show=False)
plt.title("Faktor Risiko Paling Dominan (Global Importance)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/1_shap_summary.png")
plt.close()
print(f"[1/3] Summary Plot disimpan ke {OUTPUT_DIR}/1_shap_summary.png")

# B. DEPENDENCE PLOT (Hubungan Detail)
# Contoh: Bagaimana pengaruh Usia (Age) terhadap risiko?
plt.figure()
shap.dependence_plot("age", shap_values, X_test_df, show=False)
plt.title("Pengaruh Usia terhadap Risiko Hipertensi", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/2_shap_dependence_age.png")
plt.close()
print(f"[2/3] Dependence Plot (Age) disimpan ke {OUTPUT_DIR}/2_shap_dependence_age.png")

# C. INDIVIDUAL FORCE PLOT (Contoh Kasus Pasien Tertentu)
# Kita ambil 1 pasien yang diprediksi POSITIF HIPERTENSI
# Cari indeks pasien yang probabilitasnya tinggi
probs = model.predict_proba(X_test_scaled)[:, 1]
high_risk_idx = np.argmax(probs) # Pasien paling berisiko

print(f"\n[3/3] Menganalisis Pasien ID-{high_risk_idx} (Risiko: {probs[high_risk_idx]:.2%})...")
print("Faktor Penyebab:")

# Tampilkan nilai aslinya
pasien_data = X_test.iloc[high_risk_idx]
print(pasien_data)

# Buat Waterfall Plot (Sangat bagus untuk UI nanti)
plt.figure()
# shap.explainer.expected_value adalah base value
# shap_values[high_risk_idx] adalah kontribusi fitur
shap.plots.waterfall(shap.Explanation(values=shap_values[high_risk_idx], 
                                      base_values=explainer.expected_value, 
                                      data=pasien_data,
                                      feature_names=X.columns),
                     show=False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/3_shap_waterfall_patient.png")
plt.close()
print(f"Waterfall Plot Pasien disimpan ke {OUTPUT_DIR}/3_shap_waterfall_patient.png")

print("\n--- FASE 4 SELESAI ---")
print("Buka folder 'hasil_xai' dan lihat '1_shap_summary.png'.")
print("Itu adalah jawaban kunci untuk Bab Pembahasan Tesis Anda.")