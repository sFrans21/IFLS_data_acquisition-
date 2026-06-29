import pandas as pd
import pyreadstat
import os

DATA_DIR = 'data'

def inspect_file(filename):
    path = os.path.join(DATA_DIR, 'hh14_b3b_dta', filename)
    print(f"\n>>> INSPEKSI FILE: {filename} <<<")
    
    if not os.path.exists(path):
        print("[SKIP] File tidak ditemukan.")
        return

    try:
        df, meta = pyreadstat.read_dta(path)
        df.columns = df.columns.str.lower()
        
        print(f"Kolom tersedia: {df.columns.tolist()}")
        
        # Cek Label Metadata untuk mengetahui konteks file
        # Kita ambil label dari kolom pertama yang memiliki label (biasanya fm01 atau sejenisnya)
        if meta.column_names_to_labels:
            print("Deskripsi Variabel:")
            for col, desc in list(meta.column_names_to_labels.items())[:3]:
                print(f"   - {col}: {desc}")
        
        # Cek Value Labels (Arti dari angka 1, 3, 95)
        if meta.variable_value_labels:
            print("Arti Nilai (Value Labels):")
            # Ambil sampel label pertama
            first_key = list(meta.variable_value_labels.keys())[0]
            print(f"   - {first_key}: {list(meta.variable_value_labels[first_key].items())[:5]}")

    except Exception as e:
        print(f"[ERROR] Gagal membaca file: {e}")

if __name__ == "__main__":
    # Kita cek fm1 (yang Anda punya) dan fm2 (yang Anda punya)
    # Tujuannya: Membuktikan bahwa file ini spesifik per makanan, bukan gabungan.
    inspect_file('b3b_fm1.dta')
    inspect_file('b3b_fm2.dta')