import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Konfigurasi agar grafik rapi
sns.set(style="whitegrid")
OUTPUT_DIR = "grafik_laporan"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("--- MEMULAI VISUALISASI DATA (EDA) ---")
df = pd.read_csv('dataset_hipertensi_clean.csv')

# 1. DISTRIBUSI TARGET (Pie Chart)
# Untuk melihat seberapa seimbang data kita
plt.figure(figsize=(6, 6))
counts = df['label_hypertension'].value_counts()
plt.pie(counts, labels=['Normal (0)', 'Hipertensi (1)'], autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
plt.title('Proporsi Kasus Hipertensi di Dataset')
plt.savefig(f"{OUTPUT_DIR}/1_proporsi_target.png")
print(f"[1/4] Pie chart disimpan ke {OUTPUT_DIR}/1_proporsi_target.png")

# 2. BOXPLOT: HUBUNGAN USIA & HIPERTENSI
# Hipotesis: Orang hipertensi rata-rata lebih tua
plt.figure(figsize=(8, 6))
sns.boxplot(x='label_hypertension', y='age', data=df, palette='Set2')
plt.title('Distribusi Usia: Normal vs Hipertensi')
plt.xlabel('Status (0=Normal, 1=Hipertensi)')
plt.ylabel('Usia (Tahun)')
plt.savefig(f"{OUTPUT_DIR}/2_usia_vs_hipertensi.png")
print(f"[2/4] Boxplot Usia disimpan ke {OUTPUT_DIR}/2_usia_vs_hipertensi.png")

# 3. BOXPLOT: HUBUNGAN BMI & HIPERTENSI
# Hipotesis: Orang hipertensi punya BMI lebih tinggi
plt.figure(figsize=(8, 6))
sns.boxplot(x='label_hypertension', y='bmi', data=df, palette='Set3')
plt.title('Distribusi BMI: Normal vs Hipertensi')
plt.xlabel('Status (0=Normal, 1=Hipertensi)')
plt.ylabel('BMI (kg/m^2)')
# Limit Y axis biar outlier ekstrim tidak merusak grafik
plt.ylim(10, 50) 
plt.savefig(f"{OUTPUT_DIR}/3_bmi_vs_hipertensi.png")
print(f"[3/4] Boxplot BMI disimpan ke {OUTPUT_DIR}/3_bmi_vs_hipertensi.png")

# 4. KORELASI MATRIX (Heatmap)
# Untuk melihat faktor mana yang paling merah (berkorelasi kuat) dengan Tensi
plt.figure(figsize=(12, 10))
# Ambil variabel numerik utama
cols_corr = ['age', 'bmi', 'waist_cm', 'freq_instant_noodle', 'genetic_risk_score', 'bp_systolic', 'bp_diastolic']
corr = df[cols_corr].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriks Korelasi Faktor Risiko')
plt.savefig(f"{OUTPUT_DIR}/4_heatmap_korelasi.png")
print(f"[4/4] Heatmap disimpan ke {OUTPUT_DIR}/4_heatmap_korelasi.png")

print("\n--- VISUALISASI SELESAI ---")
print(f"Cek folder '{OUTPUT_DIR}' untuk melihat hasil grafik.")