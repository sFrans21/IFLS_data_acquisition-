"""
============================================================================
SCRIPT DATA PREPARATION — PREDIKSI HIPERTENSI
Mengikuti urutan diagram alur yang telah dikoreksi berdasarkan hasil EDA:

  1. Pembersihan kode tersamar (998, 8, 9  -> NaN)
  2. Koreksi nilai mustahil tekanan darah (-> NaN)
  3. Pembentukan label target hipertensi (sistolik>=140 ATAU diastolik>=90)
  4. Penghapusan tekanan darah dari himpunan fitur (cegah kebocoran data)
  5. Encoding & feature engineering (remap 1/3->1/0, fitur turunan, seleksi fitur)
  6. Pembagian data latih-uji, lalu imputasi (di-fit pada data latih saja)

Catatan penting:
  - Pembersihan deterministik (langkah 1-5) diterapkan pada seluruh data.
  - Imputasi (langkah 6) HANYA di-fit pada data latih untuk mencegah kebocoran.
============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

PATH_DATA = "master_dataset_raw_final.csv"   # sesuaikan lokasi berkas Anda

def garis(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

# ---------------------------------------------------------------------------
print("--- MEMULAI DATA PREPARATION ---")
df = pd.read_csv(PATH_DATA, low_memory=False)
print(f"Populasi awal: {len(df)} responden, {df.shape[1]} kolom")

# LANGKAH 1 — PENGHAPUSAN NILAI DUPLIKAT AKIBAT JOIN

# 1. Cari baris yang memiliki 'pidlink' kembar
# keep=False artinya semua baris yang kembar akan ditandai dan diambil
duplikat_df = df[df.duplicated(subset='pidlink', keep=False)]

# 2. Hitung jumlah total baris duplikat dan jumlah pidlink unik yang bermasalah
total_baris_duplikat = len(duplikat_df)
jumlah_id_bermasalah = duplikat_df['pidlink'].nunique()

print(f" Ditemukan {total_baris_duplikat} baris duplikat dari {jumlah_id_bermasalah} 'pidlink' yang unik.")

if total_baris_duplikat > 0:
    # 3. Urutkan berdasarkan 'pidlink' agar data yang kembar berdampingan saat dilihat
    duplikat_df = duplikat_df.sort_values(by='pidlink')
    
    # 4. Tampilkan contoh beberapa pidlink yang duplikat di konsol
    print("\nContoh data duplikat (3 pidlink pertama):")
    contoh_id = duplikat_df['pidlink'].unique()[:3]
    print(duplikat_df[duplikat_df['pidlink'].isin(contoh_id)][['pidlink'] + df.columns[:3].tolist()])
    
    # 5. Ekspor data duplikat ke CSV terpisah untuk Anda evaluasi secara manual
    duplikat_df.to_csv('evaluasi_duplikat_pidlink.csv', index=False)
    print("\n File duplikat disimpan ke: 'evaluasi_duplikat_pidlink.csv'")
    print("Silakan buka file tersebut untuk melihat mengapa data Anda bisa mengganda.")
else:
    print(" Aman! Tidak ditemukan duplikat pada kolom 'pidlink'.")
    
    
    
# ===========================================================================
# LANGKAH 2 — PEMBERSIHAN KODE TERSAMAR (SENTINEL) -> NaN
# Dilakukan PALING AWAL agar tidak terhitung sebagai nilai nyata saat imputasi.
# ===========================================================================
garis("LANGKAH 1: Pembersihan kode tersamar")

# 1a. Umur: 998/999 = "tidak tahu". Ganti nilai tidak wajar (>150) menjadi NaN.
n_age = (df["age"] > 150).sum()
df.loc[df["age"] > 150, "age"] = np.nan

# 1b. Kode sentinel 8 & 9 ("tidak tahu"/"menolak") pada variabel berkode.
for col in ["is_diabetes", "ak02", "ak07", "km01a"]:
    if col in df.columns:
        df[col] = df[col].replace({8: np.nan, 9: np.nan})

# 1c. Nilai 9 nyasar pada kelompok ps_*.
for col in ["ps_A", "ps_B", "ps_C", "ps_E", "ps_F"]:
    if col in df.columns:
        df[col] = df[col].replace({9: np.nan})

print(f"  age > 150 (kode 998) diubah ke NaN : {int(n_age)} baris")
print("  Kode 8/9 pada is_diabetes/ak02/ak07/km01a -> NaN")
print("  Kode 9 pada ps_A..ps_F -> NaN")

# ===========================================================================
# LANGKAH 3 — KOREKSI NILAI MUSTAHIL TEKANAN DARAH -> NaN
# Wajib sebelum pelabelan karena target dibentuk dari tekanan darah.
# ===========================================================================
garis("LANGKAH 2: Koreksi nilai mustahil tekanan darah")
bad_bp = (
    (df["bp_diastolic"] >= df["bp_systolic"]) |   # diastolik tidak boleh >= sistolik
    (df["bp_systolic"] < 60) | (df["bp_systolic"] > 260) |
    (df["bp_diastolic"] < 30) | (df["bp_diastolic"] > 200)
)
print(f"  Nilai tekanan darah mustahil ditemukan : {int(bad_bp.sum())} baris -> NaN")
df.loc[bad_bp, ["bp_systolic", "bp_diastolic"]] = np.nan

# ===========================================================================
# LANGKAH 4 — PEMBENTUKAN LABEL TARGET HIPERTENSI
# Hanya bisa dibentuk jika tekanan darah tersedia & sudah bersih.
# ===========================================================================
garis("LANGKAH 3: Pembentukan label target hipertensi")
df = df.dropna(subset=["bp_systolic", "bp_diastolic"]).copy()
df["label_hypertension"] = (
    (df["bp_systolic"] >= 140) | (df["bp_diastolic"] >= 90)
).astype(int)
print(f"  Responden dengan tekanan darah valid : {len(df)}")
prop = df["label_hypertension"].mean()
print(f"  Proporsi hipertensi : {prop:.1%}  (tidak hipertensi {1-prop:.1%})")

# ===========================================================================
# LANGKAH 5 — PENGHAPUSAN TEKANAN DARAH DARI HIMPUNAN FITUR
# Mencegah kebocoran data: label adalah fungsi langsung dari bp_*.
# (Kolom disimpan sementara untuk referensi, tetapi TIDAK masuk fitur final.)
# ===========================================================================
garis("LANGKAH 4: Penghapusan tekanan darah dari fitur")
LEAKAGE_COLS = ["bp_systolic", "bp_diastolic"]
print(f"  Variabel dikeluarkan dari fitur : {LEAKAGE_COLS}")



# ===========================================================================
# LANGKAH 6 — ENCODING & FEATURE ENGINEERING & SELEKSI FITUR
# ===========================================================================
garis("LANGKAH 5: Encoding & feature engineering")

# 5a. Remap biner 1/3 -> 1/0 (NaN dipertahankan untuk diimputasi nanti).
def to_binary(s, satu=1):
    """1 -> 1 ; 3 -> 0 ; NaN tetap NaN."""
    return s.map({satu: 1, 3: 0})

df["is_female"]    = (df["sex"] == 3).astype(float)            # 3=perempuan ->1
df.loc[df["sex"].isna(), "is_female"] = np.nan
df["is_smoker"]    = to_binary(df["km01a"])                    # km01a 1=ya
df["has_diabetes"] = to_binary(df["is_diabetes"])             # 1=ya
for col in ["ps_A", "ps_B", "ps_C", "ps_E", "ps_F"]:
    df[col] = to_binary(df[col])

# 5b. Feature engineering: skor risiko genetik dari riwayat orang tua.
#     ASUMSI (perlu verifikasi buku kode): kode 'A' = sehat, selain itu = ada risiko;
#     data kosong diperlakukan sebagai tidak ada riwayat (risiko 0).
def risiko_ortu(v):
    if pd.isna(v):
        return 0
    return 0 if str(v).upper() == "A" else 1

df["risk_father"] = df["father_health"].apply(risiko_ortu)
df["risk_mother"] = df["mother_health"].apply(risiko_ortu)
df["genetic_risk_score"] = df["risk_father"] + df["risk_mother"]   # 0,1,2

# 5c. Seleksi fitur berdasar temuan EDA:
#   - Buang variabel pengganti & ID (sudah diturunkan): sex, km01a, is_diabetes, pidlink
#   - Buang redundansi ukuran tubuh: weight_kg, height_cm (diwakili bmi)
#   - Buang waist_cm: 64% kosong & sangat berkorelasi dgn bmi
#   - Buang kolom kosong-parah & hubungan lemah: ak05, ak07, ak02, freq_instant_noodle, km04, km08
#   - Buang teks asli riwayat orang tua (sudah diringkas jadi genetic_risk_score)
#   - Buang tekanan darah (kebocoran data)
FITUR = [
    "age", "is_female", "bmi",                         # demografi & antropometri
    "is_smoker", "has_diabetes", "genetic_risk_score", # gaya hidup & riwayat
    "ps_A", "ps_B", "ps_C", "ps_E", "ps_F",            # indikator stres/psikososial
]
TARGET = "label_hypertension"
df_model = df[FITUR + [TARGET]].copy()
print(f"  Jumlah fitur terpilih : {len(FITUR)}")
print(f"  Fitur: {FITUR}")

# # ===========================================================================
# # LANGKAH 7 — PEMBAGIAN DATA LATIH-UJI, LALU IMPUTASI (FIT DI DATA LATIH SAJA)
# # ===========================================================================
# garis("LANGKAH 6: Split data latih-uji lalu imputasi")
# X = df_model[FITUR]
# y = df_model[TARGET]
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.20, random_state=42, stratify=y
# )
# print(f"  Data latih : {len(X_train)}  |  Data uji : {len(X_test)}")

# # Imputasi: median untuk numerik kontinu, modus untuk biner/diskrit.
# num_cols = ["age", "bmi"]
# cat_cols = ["is_female", "is_smoker", "has_diabetes", "genetic_risk_score",
#             "ps_A", "ps_B", "ps_C", "ps_E", "ps_F"]

# imp_num = SimpleImputer(strategy="median")
# imp_cat = SimpleImputer(strategy="most_frequent")

# # .fit HANYA pada data latih, lalu transform keduanya (cegah kebocoran).
# X_train_num = pd.DataFrame(imp_num.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
# X_test_num  = pd.DataFrame(imp_num.transform(X_test[num_cols]),      columns=num_cols, index=X_test.index)
# X_train_cat = pd.DataFrame(imp_cat.fit_transform(X_train[cat_cols]), columns=cat_cols, index=X_train.index)
# X_test_cat  = pd.DataFrame(imp_cat.transform(X_test[cat_cols]),      columns=cat_cols, index=X_test.index)

# X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)[FITUR]
# X_test_final  = pd.concat([X_test_num,  X_test_cat],  axis=1)[FITUR]
# print(f"  Sisa nilai kosong pada data latih : {int(X_train_final.isna().sum().sum())}")
# print(f"  Sisa nilai kosong pada data uji   : {int(X_test_final.isna().sum().sum())}")

# ===========================================================================
# FINALISASI — SIMPAN HASIL
# ===========================================================================
garis("FINALISASI")
df_model.to_csv("dataset_hipertensi_prepared.csv", index=False)   # sebelum imputasi
# X_train_final.assign(label_hypertension=y_train.values).to_csv("train_hipertensi.csv", index=False)
# X_test_final.assign(label_hypertension=y_test.values).to_csv("test_hipertensi.csv", index=False)
print("  Tersimpan:")
print("    - dataset_hipertensi_prepared.csv  (fitur+target, sebelum imputasi)")
# print("    - train_hipertensi.csv             (data latih, sudah diimputasi)")
# print("    - test_hipertensi.csv              (data uji, sudah diimputasi)")
# print(f"\nDimensi akhir fitur : {X_train_final.shape[1]} fitur")
print("--- DATA PREPARATION SELESAI. Siap untuk pemodelan. ---")

