import pandas as pd
import pyreadstat
import os

DATA_DIR = 'data' # Sesuaikan path folder data Anda

def intip_file(subfolder, filename, col_check=None):
    path = os.path.join(DATA_DIR, subfolder, filename)
    if os.path.exists(path):
        df, meta = pyreadstat.read_dta(path)
        df.columns = df.columns.str.lower()
        print(f"\n--- ISI FILE: {filename} ---")
        print("Daftar Kolom:", df.columns.tolist())
        
        if col_check and col_check in df.columns:
            print(f"Nilai unik di kolom '{col_check}':", df[col_check].unique())
    else:
        print(f"File {filename} tidak ada.")

# 1.  Cek dulu kode penyakit di CD2 buat nyari tahu kenapa angka 3 tidak ada
# Lihat isi kolom 'cd01type' apa aja
intip_file('hh14_b3b_dta', 'b3b_cd2.dta', 'cd01type')

# 2.cek nama kolom makanan (variabel nya tipe makanannya)
intip_file('hh14_b3b_dta', 'b3b_fm1.dta')

# 3. Cek nama kolom stres (kenapa cuma ps01, 02 aja? sisanya namanya apa?)
intip_file('hh14_b3b_dta', 'b3b_ps.dta')


