# """
# ============================================================================
# SCRIPT DATA PREPARATION — PREDIKSI HIPERTENSI
# Mengikuti urutan diagram alur yang telah dikoreksi berdasarkan hasil EDA:

#   1. Pembersihan kode tersamar (998, 8, 9  -> NaN)
#   2. Koreksi nilai mustahil tekanan darah (-> NaN)
#   3. Pembentukan label target hipertensi (sistolik>=140 ATAU diastolik>=90)
#   4. Penghapusan tekanan darah dari himpunan fitur (cegah kebocoran data)
#   5. Encoding & feature engineering (remap 1/3->1/0, fitur turunan, seleksi fitur)
#   6. Pembagian data latih-uji, lalu imputasi (di-fit pada data latih saja)

# Catatan penting:
#   - Pembersihan deterministik (langkah 1-5) diterapkan pada seluruh data.
#   - Imputasi (langkah 6) HANYA di-fit pada data latih untuk mencegah kebocoran.
# ============================================================================
# """

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer

# PATH_DATA = "master_dataset_raw_final.csv"   # sesuaikan lokasi berkas Anda

# def garis(t):
#     print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

# # ---------------------------------------------------------------------------
# print("--- MEMULAI DATA PREPARATION ---")
# df = pd.read_csv(PATH_DATA, low_memory=False)
# print(f"Populasi awal: {len(df)} responden, {df.shape[1]} kolom")

# # LANGKAH 1 — PEMASTIAN TIDAK ADA NILAI DUPLIKAT AKIBAT JOIN

# # 1. Cari baris yang memiliki 'pidlink' kembar
# # keep=False artinya semua baris yang kembar akan ditandai dan diambil
# duplikat_df = df[df.duplicated(subset='pidlink', keep=False)]

# # 2. Hitung jumlah total baris duplikat dan jumlah pidlink unik yang bermasalah
# total_baris_duplikat = len(duplikat_df)
# jumlah_id_bermasalah = duplikat_df['pidlink'].nunique()

# print(f" Ditemukan {total_baris_duplikat} baris duplikat dari {jumlah_id_bermasalah} 'pidlink' yang unik.")

# if total_baris_duplikat > 0:
#     # 3. Urutkan berdasarkan 'pidlink' agar data yang kembar berdampingan saat dilihat
#     duplikat_df = duplikat_df.sort_values(by='pidlink')
    
#     # 4. Tampilkan contoh beberapa pidlink yang duplikat di konsol
#     print("\nContoh data duplikat (3 pidlink pertama):")
#     contoh_id = duplikat_df['pidlink'].unique()[:3]
#     print(duplikat_df[duplikat_df['pidlink'].isin(contoh_id)][['pidlink'] + df.columns[:3].tolist()])
    
#     # 5. Ekspor data duplikat ke CSV terpisah untuk Anda evaluasi secara manual
#     duplikat_df.to_csv('evaluasi_duplikat_pidlink.csv', index=False)
#     print("\n File duplikat disimpan ke: 'evaluasi_duplikat_pidlink.csv'")
#     print("Silakan buka file tersebut untuk melihat mengapa data Anda bisa mengganda.")
# else:
#     print(" Aman! Tidak ditemukan duplikat pada kolom 'pidlink'.")
    
    
    
# # ===========================================================================
# # LANGKAH 2 — PEMBERSIHAN KODE TERSAMAR (SENTINEL) -> NaN
# # Dilakukan PALING AWAL agar tidak terhitung sebagai nilai nyata saat imputasi.
# # ===========================================================================
# garis("LANGKAH 1: Pembersihan kode tersamar")

# # 1a. Umur: 998/999 = "tidak tahu". Ganti nilai tidak wajar (>150) menjadi NaN.
# n_age = (df["age"] > 150).sum()
# df.loc[df["age"] > 150, "age"] = np.nan

# # 1b. Kode sentinel 8 & 9 ("tidak tahu"/"menolak") pada variabel berkode.
# for col in ["is_diabetes", "ak02", "ak07", "km01a"]:
#     if col in df.columns:
#         df[col] = df[col].replace({8: np.nan, 9: np.nan})

# # 1c. Nilai 9 nyasar pada kelompok ps_*.
# for col in ["ps_A", "ps_B", "ps_C", "ps_E", "ps_F"]:
#     if col in df.columns:
#         df[col] = df[col].replace({9: np.nan})

# print(f"  age > 150 (kode 998) diubah ke NaN : {int(n_age)} baris")
# print("  Kode 8/9 pada is_diabetes/ak02/ak07/km01a -> NaN")
# print("  Kode 9 pada ps_A..ps_F -> NaN")

# # ===========================================================================
# # LANGKAH 3 — KOREKSI NILAI MUSTAHIL TEKANAN DARAH -> NaN
# # Wajib sebelum pelabelan karena target dibentuk dari tekanan darah.
# # ===========================================================================
# garis("LANGKAH 2: Koreksi nilai mustahil tekanan darah")
# bad_bp = (
#     (df["bp_diastolic"] >= df["bp_systolic"]) |   # diastolik tidak boleh >= sistolik
#     (df["bp_systolic"] < 60) | (df["bp_systolic"] > 260) |
#     (df["bp_diastolic"] < 30) | (df["bp_diastolic"] > 200)
# )
# print(f"  Nilai tekanan darah mustahil ditemukan : {int(bad_bp.sum())} baris -> NaN")
# df.loc[bad_bp, ["bp_systolic", "bp_diastolic"]] = np.nan

# # ===========================================================================
# # LANGKAH 4 — PEMBENTUKAN LABEL TARGET HIPERTENSI
# # Hanya bisa dibentuk jika tekanan darah tersedia & sudah bersih.
# # ===========================================================================
# garis("LANGKAH 3: Pembentukan label target hipertensi")
# df = df.dropna(subset=["bp_systolic", "bp_diastolic"]).copy()
# df["label_hypertension"] = (
#     (df["bp_systolic"] >= 140) | (df["bp_diastolic"] >= 90)
# ).astype(int)
# print(f"  Responden dengan tekanan darah valid : {len(df)}")
# prop = df["label_hypertension"].mean()
# print(f"  Proporsi hipertensi : {prop:.1%}  (tidak hipertensi {1-prop:.1%})")

# # ===========================================================================
# # LANGKAH 5 — PENGHAPUSAN TEKANAN DARAH DARI HIMPUNAN FITUR
# # Mencegah kebocoran data: label adalah fungsi langsung dari bp_*.
# # (Kolom disimpan sementara untuk referensi, tetapi TIDAK masuk fitur final.)
# # ===========================================================================
# garis("LANGKAH 4: Penghapusan tekanan darah dari fitur")
# LEAKAGE_COLS = ["bp_systolic", "bp_diastolic"]
# print(f"  Variabel dikeluarkan dari fitur : {LEAKAGE_COLS}")



# # ===========================================================================
# # LANGKAH 6 — ENCODING & FEATURE ENGINEERING & SELEKSI FITUR
# # ===========================================================================
# garis("LANGKAH 5: Encoding & feature engineering")

# # 5a. Remap biner 1/3 -> 1/0 (NaN dipertahankan untuk diimputasi nanti).
# def to_binary(s, satu=1):
#     """1 -> 1 ; 3 -> 0 ; NaN tetap NaN."""
#     return s.map({satu: 1, 3: 0})

# df["is_female"]    = (df["sex"] == 3).astype(float)            # 3=perempuan ->1
# df.loc[df["sex"].isna(), "is_female"] = np.nan
# df["is_smoker"]    = to_binary(df["km01a"])                    # km01a 1=ya
# df["has_diabetes"] = to_binary(df["is_diabetes"])             # 1=ya
# for col in ["ps_A", "ps_B", "ps_C", "ps_E", "ps_F"]:
#     df[col] = to_binary(df[col])

# # 5b. Feature engineering: skor risiko genetik dari riwayat orang tua.
# #     ASUMSI (perlu verifikasi buku kode): kode 'A' = sehat, selain itu = ada risiko;
# #     data kosong diperlakukan sebagai tidak ada riwayat (risiko 0).
# def risiko_ortu(v):
#     if pd.isna(v):
#         return 0
#     return 0 if str(v).upper() == "A" else 1

# df["risk_father"] = df["father_health"].apply(risiko_ortu)
# df["risk_mother"] = df["mother_health"].apply(risiko_ortu)
# df["genetic_risk_score"] = df["risk_father"] + df["risk_mother"]   # 0,1,2

# # 5c. Seleksi fitur berdasar temuan EDA:
# #   - Buang variabel pengganti & ID (sudah diturunkan): sex, km01a, is_diabetes, pidlink
# #   - Buang redundansi ukuran tubuh: weight_kg, height_cm (diwakili bmi)
# #   - Buang waist_cm: 64% kosong & sangat berkorelasi dgn bmi
# #   - Buang kolom kosong-parah & hubungan lemah: ak05, ak07, ak02, freq_instant_noodle, km04, km08
# #   - Buang teks asli riwayat orang tua (sudah diringkas jadi genetic_risk_score)
# #   - Buang tekanan darah (kebocoran data)
# FITUR = [
#     "age", "is_female", "bmi",                         # demografi & antropometri
#     "is_smoker", "has_diabetes", "genetic_risk_score", # gaya hidup & riwayat
#     "ps_A", "ps_B", "ps_C", "ps_E", "ps_F",            # indikator stres/psikososial
# ]
# TARGET = "label_hypertension"
# df_model = df[FITUR + [TARGET]].copy()
# print(f"  Jumlah fitur terpilih : {len(FITUR)}")
# print(f"  Fitur: {FITUR}")

# # # ===========================================================================
# # # LANGKAH 7 — PEMBAGIAN DATA LATIH-UJI, LALU IMPUTASI (FIT DI DATA LATIH SAJA)
# # # ===========================================================================
# # garis("LANGKAH 6: Split data latih-uji lalu imputasi")
# # X = df_model[FITUR]
# # y = df_model[TARGET]
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X, y, test_size=0.20, random_state=42, stratify=y
# # )
# # print(f"  Data latih : {len(X_train)}  |  Data uji : {len(X_test)}")

# # # Imputasi: median untuk numerik kontinu, modus untuk biner/diskrit.
# # num_cols = ["age", "bmi"]
# # cat_cols = ["is_female", "is_smoker", "has_diabetes", "genetic_risk_score",
# #             "ps_A", "ps_B", "ps_C", "ps_E", "ps_F"]

# # imp_num = SimpleImputer(strategy="median")
# # imp_cat = SimpleImputer(strategy="most_frequent")

# # # .fit HANYA pada data latih, lalu transform keduanya (cegah kebocoran).
# # X_train_num = pd.DataFrame(imp_num.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
# # X_test_num  = pd.DataFrame(imp_num.transform(X_test[num_cols]),      columns=num_cols, index=X_test.index)
# # X_train_cat = pd.DataFrame(imp_cat.fit_transform(X_train[cat_cols]), columns=cat_cols, index=X_train.index)
# # X_test_cat  = pd.DataFrame(imp_cat.transform(X_test[cat_cols]),      columns=cat_cols, index=X_test.index)

# # X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)[FITUR]
# # X_test_final  = pd.concat([X_test_num,  X_test_cat],  axis=1)[FITUR]
# # print(f"  Sisa nilai kosong pada data latih : {int(X_train_final.isna().sum().sum())}")
# # print(f"  Sisa nilai kosong pada data uji   : {int(X_test_final.isna().sum().sum())}")

# # ===========================================================================
# # FINALISASI — SIMPAN HASIL
# # ===========================================================================
# garis("FINALISASI")
# df_model.to_csv("dataset_hipertensi_prepared.csv", index=False)   # sebelum imputasi
# # X_train_final.assign(label_hypertension=y_train.values).to_csv("train_hipertensi.csv", index=False)
# # X_test_final.assign(label_hypertension=y_test.values).to_csv("test_hipertensi.csv", index=False)
# print("  Tersimpan:")
# print("    - dataset_hipertensi_prepared.csv  (fitur+target, sebelum imputasi)")
# # print("    - train_hipertensi.csv             (data latih, sudah diimputasi)")
# # print("    - test_hipertensi.csv              (data uji, sudah diimputasi)")
# # print(f"\nDimensi akhir fitur : {X_train_final.shape[1]} fitur")
# print("--- DATA PREPARATION SELESAI. Siap untuk pemodelan. ---")




"""
============================================================================
SCRIPT DATA PREPARATION — PREDIKSI HIPERTENSI   [DATASET 19 KOLOM]
Mengikuti urutan diagram alur yang dikoreksi berdasarkan hasil EDA:

  0. Pemeriksaan duplikat pidlink
  1. Pembersihan kode tersamar (998, 8, 9 -> NaN)
  2. filtering usia menjadi 18 tahun keatas
  2. Koreksi nilai mustahil tekanan darah (-> NaN)
  3. Pembentukan label target hipertensi (sistolik>=140 ATAU diastolik>=90)
  4. Penghapusan tekanan darah dari himpunan fitur (cegah kebocoran data)
  5. Hitung BMI, encoding, feature engineering, seleksi fitur
  6. (opsional) Split data latih-uji lalu imputasi (di-fit pada data latih saja)

Catatan: imputasi sebaiknya dilakukan di tahap pemodelan di dalam pipeline
cross-validation. Karena itu Langkah 6 dibiarkan nonaktif di sini, dan script
ini hanya menghasilkan dataset bersih sebelum imputasi.
============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

PATH_DATA = "master_dataset_raw_final.csv"   # sesuaikan lokasi berkas Anda

def garis(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

print("--- MEMULAI DATA PREPARATION ---")
df = pd.read_csv(PATH_DATA, low_memory=False)
print(f"Populasi awal: {len(df)} responden, {df.shape[1]} kolom")

# ===========================================================================
# LANGKAH 0 — PEMERIKSAAN DUPLIKAT pidlink
# ===========================================================================
garis("LANGKAH 0: Pemeriksaan duplikat pidlink")
dup = df[df.duplicated(subset="pidlink", keep=False)]
print(f"  Ditemukan {len(dup)} baris duplikat dari {dup['pidlink'].nunique()} pidlink unik.")
if len(dup) > 0:
    dup.sort_values("pidlink").to_csv("evaluasi_duplikat_pidlink.csv", index=False)
    print("  Disimpan ke evaluasi_duplikat_pidlink.csv untuk evaluasi manual.")
else:
    print("  Aman, tidak ada duplikat pidlink.")

# ===========================================================================
# LANGKAH 1 — PEMBERSIHAN KODE TERSAMAR (SENTINEL) -> NaN
# Dilakukan PALING AWAL agar tidak terhitung sebagai nilai nyata.
# ===========================================================================
garis("LANGKAH 1: Pembersihan kode tersamar")

# 1a. Umur: 998 = "tidak tahu" -> NaN
n_age = int((df["age"] > 150).sum())
n_dbt = int(df["is_diabetes"].isin([8, 9]).sum())
n_cls = int(df["is_high_cholesterol"].isin([8, 9]).sum())
n_sleep_quality = int(df["sleep_quality"].isin([8, 9]).sum())
n_sleep_disturbance = int(df["sleep_disturbance"].isin([8, 9]).sum())

# Tampilkan jumlah data yang akan dikoreksi
print(f"Age > 150                : {n_age}")
print(f"is_diabetes (8/9)        : {n_dbt}")
print(f"is_high_cholesterol (8/9): {n_cls}")
print(f"sleep_quality (8/9)      : {n_sleep_quality}")
print(f"sleep_disturbance (8/9)  : {n_sleep_disturbance}")

df.loc[df["age"] > 150, "age"] = np.nan

# 1b. [REVISI Titik 1] Kode 8 & 9 pada variabel kondisi kronis (1/3).
#     ak02/ak07/km01a DIHAPUS (tidak ada); is_high_cholesterol DITAMBAH.
for col in ["is_diabetes", "is_high_cholesterol"]:
    df[col] = df[col].replace({8: np.nan, 9: np.nan})

# 1c. [REVISI Titik 2] ps_* DIHAPUS. Ganti dengan kode 9 pada skala tidur 1-5.
for col in ["sleep_quality", "sleep_disturbance"]:
    df[col] = df[col].replace({8: np.nan, 9: np.nan})

print(f"  age > 150 (kode 998) -> NaN : {n_age} baris")
print("  Kode 8/9 pada is_diabetes & is_high_cholesterol -> NaN")
print("  Kode 8/9 pada sleep_quality & sleep_disturbance -> NaN")


# ===========================================================================
# FILTER TAMBAHAN — HANYA RESPONDEN USIA ≥ 18 TAHUN
# Dilakukan setelah pembersihan umur agar nilai 998 yang sudah menjadi NaN
# tidak ikut dihitung sebagai usia valid.
# ===========================================================================
garis("FILTER: Responden usia >= 18 tahun")

n_sebelum = len(df)

# Buang responden dengan usia < 18 tahun
df = df[df["age"] >= 18].copy()

n_sesudah = len(df)
n_terhapus = n_sebelum - n_sesudah

print(f"Responden sebelum filter : {n_sebelum}")
print(f"Responden setelah filter : {n_sesudah}")
print(f"Responden usia < 18 tahun yang dihapus : {n_terhapus}")

# ===========================================================================
# LANGKAH 2 — KOREKSI NILAI MUSTAHIL TEKANAN DARAH -> NaN
# ===========================================================================
garis("LANGKAH 2: Koreksi nilai mustahil tekanan darah")
bad_bp = (
    (df["bp_diastolic"] >= df["bp_systolic"])
)

print(f"  Tekanan darah mustahil : {int(bad_bp.sum())} baris -> NaN")
# Tampilkan data yang bermasalah
if bad_bp.sum() > 0:
    print("\n=== Data dengan tekanan darah mustahil ===")
    print(
        df.loc[
            bad_bp,
            ["pidlink", "bp_systolic", "bp_diastolic"]  # tambahkan kolom lain jika perlu
        ].to_string(index=True)
    )

df.loc[bad_bp, ["bp_systolic", "bp_diastolic"]] = np.nan


n_awal = len(df)

# Missing sejak awal
missing_awal = (
    df["bp_systolic"].isna() |
    df["bp_diastolic"].isna()
)

print(f"Missing BP sejak awal : {missing_awal.sum()}")

# Nilai mustahil
print(f"BP mustahil          : {bad_bp.sum()}")

# Setelah dikoreksi
df.loc[bad_bp, ["bp_systolic","bp_diastolic"]] = np.nan

missing_total = (
    df["bp_systolic"].isna() |
    df["bp_diastolic"].isna()
)

print(f"Total BP tidak bisa dipakai : {missing_total.sum()}")

print(f"Responden tersisa : {n_awal - missing_total.sum()}")


# ===========================================================================
# LANGKAH 3 — PEMBENTUKAN LABEL TARGET HIPERTENSI
# ===========================================================================
garis("LANGKAH 3: Pembentukan label target hipertensi")
df = df.dropna(subset=["bp_systolic", "bp_diastolic"]).copy()
df["label_hypertension"] = (
    (df["bp_systolic"] >= 140) | (df["bp_diastolic"] >= 90)
).astype(int)
print(f"  Responden tekanan darah valid : {len(df)}")
prop = df["label_hypertension"].mean()
print(f"  Proporsi hipertensi : {prop:.1%} (tidak hipertensi {1-prop:.1%})")

# ===========================================================================
# LANGKAH 4 — PENGHAPUSAN TEKANAN DARAH DARI HIMPUNAN FITUR
# ===========================================================================
garis("LANGKAH 4: Penghapusan tekanan darah dari fitur")
LEAKAGE_COLS = ["bp_systolic", "bp_diastolic"]
print(f"  Dikeluarkan dari fitur : {LEAKAGE_COLS}")

# ===========================================================================
# LANGKAH 5 — HITUNG BMI, ENCODING, FEATURE ENGINEERING, SELEKSI FITUR
# ===========================================================================
garis("LANGKAH 5: BMI, encoding & seleksi fitur")

# [REVISI Titik 6] BMI tidak ada di dataset -> dihitung sendiri.
df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2

# 5a. Remap biner 1/3 -> 1/0 (NaN dipertahankan untuk diimputasi nanti).
def to_binary(s, satu=1):
    """1 -> 1 ; 3 -> 0 ; NaN tetap NaN."""
    return s.map({satu: 1, 3: 0})

df["is_female"]           = (df["sex"] == 3).astype(float)   # 3=perempuan ->1
df.loc[df["sex"].isna(), "is_female"] = np.nan
print(df.columns.tolist())
df["is_smoker"]           = to_binary(df["is_smoking"])      # [REVISI Titik 3] km01a -> is_smoking
df["has_diabetes"]        = to_binary(df["is_diabetes"])
df["has_high_cholesterol"] = to_binary(df["is_high_cholesterol"])  # [REVISI] fitur baru

# [REVISI Titik 4 & 5] loop ps_* dan blok genetic_risk_score DIHAPUS seluruhnya.

# 5c. [REVISI Titik 7] Seleksi fitur untuk dataset baru.
#   Dipertahankan (kekosongan rendah, relevan): age, is_female, bmi, is_smoker,
#   has_diabetes, has_high_cholesterol, sleep_quality, sleep_disturbance.
#   Dibuang (kosong-parah / hubungan lemah pada EDA):
#     waist_cm (64%), freq_hard_act (81%), freq_fast_food (90%), freq_soda (83%),
#     freq_moderate_act (52%), freq_walking (40%), freq_fried_food (44%).
#   -> Untuk memakainya, cukup tambahkan ke daftar FITUR & ke num_cols pada Langkah 6.
FITUR = [
    "age", "is_female", "bmi",                       # demografi & antropometri
    "is_smoker", "has_diabetes", "has_high_cholesterol",  # gaya hidup & kondisi kronis
    "sleep_quality", "sleep_disturbance",            # kualitas tidur
]
TARGET = "label_hypertension"
df_model = df[FITUR + [TARGET]].copy()
print(f"  Jumlah fitur terpilih : {len(FITUR)}")
print(f"  Fitur: {FITUR}")

# ===========================================================================
# LANGKAH 6 — (OPSIONAL) SPLIT & IMPUTASI (FIT DI DATA LATIH SAJA)
# Disarankan dilakukan di tahap pemodelan via Pipeline + cross-validation.
# Aktifkan dengan menghapus tanda komentar bila ingin split statis.
# ===========================================================================
garis("LANGKAH 6: Split data latih-uji lalu imputasi")
X, y = df_model[FITUR], df_model[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print("\n=== HASIL DATA SPLIT ===")
print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_train : {y_train.shape}")
print(f"y_test  : {y_test.shape}")

num_cols = ["age", "bmi"]  
cat_cols = ["is_female", "is_smoker", "has_diabetes", "has_high_cholesterol", "sleep_disturbance", "sleep_quality"]
imp_num = SimpleImputer(strategy="median")
imp_cat = SimpleImputer(strategy="most_frequent")
Xtr = X_train.copy(); Xte = X_test.copy()
Xtr[num_cols] = imp_num.fit_transform(X_train[num_cols]); Xte[num_cols] = imp_num.transform(X_test[num_cols])
Xtr[cat_cols] = imp_cat.fit_transform(X_train[cat_cols]); Xte[cat_cols] = imp_cat.transform(X_test[cat_cols])
Xtr.assign(label_hypertension=y_train.values).to_csv("train_hipertensi.csv", index=False)
Xte.assign(label_hypertension=y_test.values).to_csv("test_hipertensi.csv", index=False)


# ===========================================================================
# FINALISASI
# ===========================================================================
garis("FINALISASI")
df_model.to_csv("dataset_hipertensi_prepared.csv", index=False)
print("  Tersimpan: dataset_hipertensi_prepared.csv (fitur+target, sebelum imputasi)")
print(f"  Sisa nilai kosong per fitur:\n{df_model[FITUR].isna().sum().to_string()}")


# ===========================================================================
# INFORMASI DATA TRAIN & TEST
# ===========================================================================
garis("RINGKASAN DATA TRAIN & TEST")

print(f"Jumlah data keseluruhan : {len(df_model)}")
print(f"Data latih             : {len(X_train)} ({len(X_train)/len(df_model):.1%})")
print(f"Data uji               : {len(X_test)} ({len(X_test)/len(df_model):.1%})")

print("\nDistribusi label hipertensi")
print("- Train")
print(y_train.value_counts().rename({0: "Tidak Hipertensi", 1: "Hipertensi"}))
print((y_train.value_counts(normalize=True)*100).round(2).rename({0: "Tidak Hipertensi (%)", 1: "Hipertensi (%)"}))

print("\n- Test")
print(y_test.value_counts().rename({0: "Tidak Hipertensi", 1: "Hipertensi"}))
print((y_test.value_counts(normalize=True)*100).round(2).rename({0: "Tidak Hipertensi (%)", 1: "Hipertensi (%)"}))

print("\nMissing value sebelum imputasi")
print(f"Train : {X_train.isna().sum().sum()} sel")
print(f"Test  : {X_test.isna().sum().sum()} sel")

print("\nMissing value setelah imputasi")
print(f"Train : {Xtr.isna().sum().sum()} sel")
print(f"Test  : {Xte.isna().sum().sum()} sel")

print("\nBerkas yang disimpan:")
print(" - train_hipertensi.csv")
print(" - test_hipertensi.csv")

print("--- DATA PREPARATION SELESAI. ---")