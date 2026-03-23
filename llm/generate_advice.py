import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

print("--- FASE 5: GENERATIVE AI (OTOMASI SARAN KESEHATAN) ---")

# 1. PERSIAPAN DATA & MODEL (Sama seperti sebelumnya)
df = pd.read_csv('../dataset_hipertensi_clean.csv')
X = df.drop(columns=['label_hypertension', 'bp_systolic', 'bp_diastolic'], errors='ignore')
X = X.select_dtypes(include=[np.number]).fillna(0)
y = df['label_hypertension']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Kembalikan ke DataFrame agar nama kolom terbaca
X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# Train Model
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# 2. CARI PASIEN DENGAN RISIKO TERTINGGI (Simulasi Real-World)
probs = model.predict_proba(X_test_scaled)[:, 1]
high_risk_idx = np.argmax(probs) # Ambil indeks pasien paling parah
risk_score = probs[high_risk_idx]

# Ambil data asli pasien (sebelum di-scale) untuk ditampilkan di teks
patient_data = X_test.iloc[high_risk_idx]

print(f"\n[TARGET PASIEN DETEKSI]")
print(f"ID Pasien : {high_risk_idx}")
print(f"Prediksi  : {risk_score:.2%} (RISIKO TINGGI)")

# 3. ANALISIS PENYEBAB (XAI)
# Kita hitung SHAP hanya untuk 1 pasien ini (agar cepat)
explainer = shap.LinearExplainer(model, X_train_scaled)
shap_values_single = explainer.shap_values(X_test_scaled[high_risk_idx].reshape(1, -1))

# Gabungkan nama fitur dengan nilai SHAP-nya
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'shap_value': shap_values_single[0],
    'actual_value': patient_data.values
})

# Urutkan dari yang paling memicu risiko (SHAP positif terbesar)
top_risk_factors = feature_importance.sort_values(by='shap_value', ascending=False).head(3)

print("\n[3 FAKTOR PENYEBAB UTAMA]")
print(top_risk_factors[['feature', 'actual_value', 'shap_value']])

# 4. PROMPT ENGINEERING (INTI FASE 5)
# Kita susun kalimat untuk dikirim ke LLM
prompt = f"""
BERTINDAKLAH SEBAGAI DOKTER JANTUNG PROFESIONAL.

Tugas Anda: Berikan penjelasan medis dan saran gaya hidup yang empatik namun tegas kepada pasien berikut.

[PROFIL PASIEN]
- Risiko Hipertensi: {risk_score:.1%} (Kategori: BAHAYA)
- Faktor Risiko Utama 1: {top_risk_factors.iloc[0]['feature']} (Nilai: {top_risk_factors.iloc[0]['actual_value']:.1f})
- Faktor Risiko Utama 2: {top_risk_factors.iloc[1]['feature']} (Nilai: {top_risk_factors.iloc[1]['actual_value']:.1f})
- Faktor Risiko Utama 3: {top_risk_factors.iloc[2]['feature']} (Nilai: {top_risk_factors.iloc[2]['actual_value']:.1f})

[INSTRUKSI]
1. Jelaskan mengapa {top_risk_factors.iloc[0]['feature']} dan {top_risk_factors.iloc[1]['feature']} meningkatkan risiko hipertensi.
2. Berikan 3 langkah konkret yang bisa dilakukan besok pagi untuk menurunkan risiko.
3. Gunakan bahasa Indonesia yang mudah dipahami orang awam, hindari istilah medis rumit tanpa penjelasan.
4. Akhiri dengan kalimat motivasi.
"""

print("\n" + "="*50)
print("GENERATED PROMPT (Kirim teks di bawah ini ke LLM)")
print("="*50)
print(prompt)
print("="*50)

# 5. SIMULASI RESPON (Mockup Output)
print("\n[SIMULASI RESPON LLM (CONTOH HASIL)]")
print("-" * 20)
print("Halo Bapak/Ibu, berdasarkan analisis kesehatan Anda, risiko hipertensi Anda saat ini mencapai 98%.")
print(f"Hal ini terutama dipicu oleh faktor usia ({patient_data['age']:.0f} tahun) dan lingkar pinggang ({patient_data['waist_cm']:.0f} cm).")
print("Seiring bertambahnya usia, pembuluh darah cenderung menjadi lebih kaku, dan lemak di area perut memperberat kerja jantung.")
print("\nSaran saya untuk besok pagi:")
print("1. Kurangi Garam: Hindari makanan asin/gurih saat sarapan.")
print("2. Jalan Kaki Ringan: Lakukan selama 15 menit di sekitar rumah.")
print("3. Cek Tensi: Segera kunjungi puskesmas terdekat untuk validasi.")
print("\nKesehatan adalah investasi terbaik. Mari mulai perubahan kecil hari ini!")
print("-" * 20)