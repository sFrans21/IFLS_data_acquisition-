# import pandas as pd

# # 1. Tentukan nama file input (.dta) dan output (.txt)
# file_path = "data/hh14_b3b_dta/b3b_kk2.dta" # Ganti dengan path file IFLS kamu
# output_txt = "hasil_eksplorasi_fisik.txt"

# # 2. Kata kunci yang ingin dicari (huruf kecil semua untuk pencarian)
# keywords = ["aktivitas", "fisik", "physical", "activity", "olahraga", "jalan", "berat", "sepeda"]

# # 3. Membaca metadata (label variabel) dari file .dta
# try:
#     reader = pd.io.stata.StataReader(file_path)
#     var_labels = reader.variable_labels()
    
#     # 4. Membuka file .txt untuk menulis hasil
#     with open(output_txt, "w", encoding="utf-8") as f:
#         f.write(f"Hasil Eksplorasi Variabel untuk file: {file_path}\n")
#         f.write("="*60 + "\n\n")
        
#         found = False
#         for var_name, label in var_labels.items():
#             label_lower = str(label).lower()
            
#             # Cek apakah ada salah satu kata kunci di dalam label pertanyaan
#             if any(kw in label_lower for kw in keywords):
#                 f.write(f"Kode Variabel : {var_name}\n")
#                 f.write(f"Label         : {label}\n")
#                 f.write("-" * 60 + "\n")
#                 found = True
                
#         if not found:
#             f.write("Tidak ditemukan variabel yang cocok dengan kata kunci.\n")
            
#     print(f"Eksplorasi selesai! Silakan buka file '{output_txt}'")

# except FileNotFoundError:
#     print(f"Error: File {file_path} tidak ditemukan. Pastikan nama dan lokasinya benar.")



import pandas as pd

# 1. Memuat dataset (Ganti dengan nama file kamu, misalnya 'b3b_kk.dta')
# Catatan: convert_categoricals=False digunakan agar value label Stata 
# (seperti 1=Yes, 3=Never) tetap terbaca sebagai angka aslinya, sehingga lebih mudah diolah.
file_path = 'data/hh14_b3b_dta/b3b_kk2.dta'
df = pd.read_stata(file_path, convert_categoricals=False)

# 2. Melihat Informasi Umum & Tipe Data
print("=== INFORMASI DATAFRAME & TIPE DATA ===")
df.info()
print("\n")

# 3. Melihat 5 Baris Pertama (Isi Dataframe)
print("=== 5 BARIS PERTAMA DATAFRAME ===")
print(df.head()) # Gunakan print(df.head()) jika tidak di Jupyter Notebook
print("\n")

# 4. Eksplorasi Khusus Berdasarkan Metadata yang Kamu Berikan
print("=== EKSPLORASI KOLOM KUNCI ===")

# a. Cek jumlah unik responden (Grain/ID)
# Asumsi kita gabungkan hhid14_9 dan pid14 untuk membuat 'pidlink' 
# (atau gunakan kolom pidlink jika sudah ada di datamu)
if 'pidlink' in df.columns:
    unik_orang = df['pidlink'].nunique()
else:
    # Menggabungkan hhid dan pid jika pidlink belum dibuat
    df['pidlink'] = df['hhid14_9'].astype(str) + df['pid14'].astype(str).str.zfill(2)
    unik_orang = df['pidlink'].nunique()

print(f"Total baris data    : {len(df)}")
print(f"Total individu unik : {unik_orang}")
print("-" * 30)

# b. Cek isi dari jenis aktivitas (kktype)
print("Distribusi Tipe Aktivitas Fisik (kktype):")
print(df['kktype'].value_counts(dropna=False))
print("-" * 30)

# c. Cek frekuensi jawaban pada kk02m (During last 7 days)
# Harusnya berisi 1 (Yes), 3 (Never), 9 (Missing) seperti metadatamu
print("Distribusi Jawaban Aktivitas 7 Hari Terakhir (kk02m):")
print(df['kk02m'].value_counts(dropna=False))

string_unik = df['kk02n1'].unique()
print("1. Daftar nilai string unik yang dimasukkan pada kk02n1:")
# Diubah ke list agar tampilannya lebih rapi saat dicetak
print(string_unik.tolist()) 
print("-" * 50)

