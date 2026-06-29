import pandas as pd
import pyreadstat
import os

DATA_DIR = 'data'

def probe_metadata(subfolder, filename, search_keywords):
    """
    Mencari nama kolom yang benar berdasarkan kata kunci pada label variabel.
    """
    path = os.path.join(DATA_DIR, subfolder, filename)
    print(f"\n[{filename.upper()}] MEMINDAI METADATA...")
    
    if not os.path.exists(path):
        print(f"[ERROR] File tidak ditemukan: {path}")
        return

    try:
        # Hanya baca metadata (header), tidak baca baris data (Cepat & Hemat Memori)
        _, meta = pyreadstat.read_dta(path, metadataonly=True)
        
        # 1. Cari berdasarkan Keyword di Label
        print(f"   > Mencari variabel dengan kata kunci: {search_keywords}")
        found_count = 0
        
        if meta.column_names_to_labels:
            for col_name, label in meta.column_names_to_labels.items():
                # Normalisasi ke lowercase untuk pencarian
                label_lower = label.lower()
                if any(kw in label_lower for kw in search_keywords):
                    print(f"     [DITEMUKAN] {col_name} : {label}")
                    found_count += 1
        
        if found_count == 0:
            print("     [INFO] Tidak ada label yang cocok dengan keyword.")
            
        # 2. Cetak 20 Kolom Pertama (Untuk inspeksi manual jika pencarian gagal)
        print("   > 15 Kolom Pertama (Raw List):")
        cols = list(meta.column_names_to_labels.keys())[:15]
        print(f"     {cols}")

    except Exception as e:
        print(f"[ERROR] Gagal membaca metadata: {e}")

if __name__ == "__main__":
    print("--- MULAI INSPEKSI SCHEMA IFLS ---")

    # MISI 1: Cari Berat, Tinggi, dan Tensi yang BENAR di Buku US
    # Masalah sebelumnya: Kita mungkin salah ambil kolom Systolic ke-2 sebagai Diastolik
    # Keyword mencakup Inggris dan Indonesia (karena IFLS tidak konsisten)
    keywords_fisik = [
        'weight', 'berat',         # Untuk BB
        'height', 'tinggi',        # Untuk TB
        'waist', 'lingkar',        # Untuk Pinggang
        'systol', 'diastol',       # Untuk Tensi (Target)
        'blood', 'pressure'        # Keyword umum
    ]
    probe_metadata('hh14_bus_dta', 'bus_us.dta', keywords_fisik)

    # MISI 2: Cari Riwayat Kesehatan Orang Tua di Buku 3B (Modul BA)
    # Masalah sebelumnya: Kolom ba15/ba42 tidak ditemukan
    keywords_ortu = [
        'father', 'ayah', 'dad',
        'mother', 'ibu', 'mom',
        'health', 'sehat', 'sakit'
    ]
    probe_metadata('hh14_b3b_dta', 'b3b_ba1.dta', keywords_ortu)