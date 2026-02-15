# # SCRIPT EKSTRAKSI AWAL
# import pandas as pd
# import pyreadstat
# import os

# DATA_DIR = 'data'

# def load_ifls_data(subfolder, filename):
#       path = os.path.join(DATA_DIR, subfolder, filename)
#       if not os.path.exists(path):
#             print(f"File tidak ditemukan: {path}")
#             return None, None
#       df, meta = pyreadstat.read_dta(path)
#       # Buat semua kolom menjadi lowercase agar menjaga konsistensi
#       df.columns = df.columns.str.lower()
#       return df, meta

# # 1. Load Data Tracking (Ptrack) sebagai Master ID
# print("Memuat data tracking...")
# df_ptrack, _ = load_ifls_data('hh14_trk_dta', 'ptrack.dta')

# # 2. Load Data Demografi (Book 3A Cover) --> Sumber Usia dan Jenis Kelamin
# print("Memuat data demografi (Book 3A)...")
# df_b3a_cov, _ = load_ifls_data('hh14_b3a_dta', 'b3a_cov.dta')

# # 3. Ekstraksi Fitur Terpilih
# # Di b3a_cov, variabelnya biasanya adalah sex dan age
# if df_b3a_cov is not None:
#       #pastikan pidlink tersedia untuk merging
#       df_demografi = df_b3a_cov[['pidlink', 'sex', 'age']]
      
#       #  4. Merging Data
#       # Gabungkan ptrack dengan b3a_cov berdasarkan pidlink
#       # ptrack memastikan kita hanya mengambil responden yang valid di wave 5
#       final_df = pd.merge(df_ptrack[['pidlink']], df_demografi, on='pidlink', how='inner')
      
#       print("\n--- Hasil akuisisi fitur pertama (usia & jenis kelamin) ---")
#       print(final_df.head())
#       print(f"\nTotal responden dewasa berhasil diakuisisi: {len(final_df)} orang.")
      
#       #Simpan progress ke CSV sementara jika perlu
#       final_df.to_csv('master_dataset_temp.csv', index=False)
      
# else:
#       print("Gagal memproses data demografi.")
      

# print("Memuat data kebiasaan merokok (Book 3A)...")
# df_smoke, _ = load_ifls_data('hh14_b3a_dta', 'b3a_pk1.dta')

# # 8. Load Data Aktivitas Fisik (Book 3B - Section AK)
# print("Memuat data aktivitas fisik (Book 3B)...")
# df_activity, _ = load_ifls_data('hh14_b3b_dta', 'b3b_ak1.dta')

# if df_smoke is not None and df_activity is not None:
#     # --- PROSES DATA MEROKOK ---
#     # pk01: Apakah pernah merokok?
#     # pk02: Apakah masih merokok sampai sekarang?
#     df_smoke_clean = df_smoke[['pidlink', 'pk01', 'pk02']]
#     master_df = pd.merge(master_df, df_smoke_clean, on='pidlink', how='left')

#     # --- PROSES DATA AKTIVITAS FISIK ---
#     # IFLS menggunakan standar IPAQ. 
#     # ak02: Hari dalam seminggu melakukan aktivitas berat.
#     # ak05: Hari dalam seminggu melakukan aktivitas sedang.
#     # ak08: Hari dalam seminggu melakukan jalan kaki.
#     df_act_clean = df_activity[['pidlink', 'ak02', 'ak05', 'ak08']]
#     master_df = pd.merge(master_df, df_act_clean, on='pidlink', how='left')

#     print("\n--- Hasil Akuisisi Gaya Hidup (Merokok & Aktivitas) ---")
#     print(master_df[['pidlink', 'pk01', 'ak02']].head())
#     print(f"\nTotal kolom saat ini: {len(master_df.columns)}")
    
#     # Simpan progress terbaru ke CSV
#     master_df.to_csv('master_dataset_phase3.csv', index=False)

# else:
#     print("Gagal memproses data gaya hidup.")




# import pandas as pd
# import pyreadstat
# import os

# DATA_DIR = 'data'

# def load_ifls_data(subfolder, filename):
#     path = os.path.join(DATA_DIR, subfolder, filename)
#     if not os.path.exists(path):
#         print(f"File tidak ditemukan: {path}")
#         return None, None
#     df, meta = pyreadstat.read_dta(path)
#     df.columns = df.columns.str.lower()
#     return df, meta

# # --- PHASE 1 & 2 RECAP (Sudah Berhasil) ---
# print("--- MEMULAI FASE 1 & 2 ---")
# df_ptrack, _ = load_ifls_data('hh14_trk_dta', 'ptrack.dta')
# df_b3a_cov, _ = load_ifls_data('hh14_b3a_dta', 'b3a_cov.dta')

# if df_ptrack is not None and df_b3a_cov is not None:
#     df_demografi = df_b3a_cov[['pidlink', 'sex', 'age']]
#     # Inisialisasi master_df di sini
#     master_df = pd.merge(df_ptrack[['pidlink']], df_demografi, on='pidlink', how='inner')
#     print(f"Base data loaded: {len(master_df)} rows.")
# else:
#     print("Gagal load base data.")
#     exit()


# print("\n--- FASE 3: GAYA HIDUP (TARGET BARU) ---")

# # ==========================================
# # 1. AKUISISI MEROKOK (Target: Book 3B Modul KM)
# # ==========================================
# print("1. Mencari data merokok di b3b_km.dta...")
# df_smoke, _ = load_ifls_data('hh14_b3b_dta', 'b3b_km.dta')

# if df_smoke is not None:
#     # Variabel Merokok di IFLS-5 KM biasanya:
#     # km01a: Apakah pernah merokok/mengunyah tembakau? (1=Ya, 3=Tidak)
#     # km04: Apakah masih merokok sekarang?
#     # km08: Rata-rata batang rokok per hari
    
#     # Cek ketersediaan kolom
#     cols_check = ['km01a', 'km04', 'km08']
#     available_km = [c for c in cols_check if c in df_smoke.columns]
    
#     if available_km:
#         print(f"   DITEMUKAN: {available_km}")
#         df_smoke_clean = df_smoke[['pidlink'] + available_km]
#         master_df = pd.merge(master_df, df_smoke_clean, on='pidlink', how='left')
#     else:
#         print(f"   GAGAL: Kolom {cols_check} tidak ada di b3b_km.dta")
#         print("   Isi kolom:", df_smoke.columns.tolist()[:10], "...")
# else:
#     print("   File b3b_km.dta tidak ditemukan.")


# # ==========================================
# # 2. AKUISISI AKTIVITAS FISIK (Target: Gabung AK1 + AK2)
# # ==========================================
# print("\n2. Menggabungkan Aktivitas Fisik (AK1 + AK2)...")
# df_ak1, _ = load_ifls_data('hh14_b3b_dta', 'b3b_ak1.dta') # Berat & Sedang
# df_ak2, _ = load_ifls_data('hh14_b3b_dta', 'b3b_ak2.dta') # Jalan Kaki

# if df_ak1 is not None and df_ak2 is not None:
#     # Ambil bagian 1: Berat (ak02) & Sedang (ak05)
#     # Catatan: ak02 = hari aktivitas berat, ak05 = hari aktivitas sedang
#     part1 = df_ak1[['pidlink', 'ak02', 'ak05']]
    
#     # Ambil bagian 2: Jalan Kaki (ak07)
#     # Di output Anda tadi terlihat ada 'ak07' di b3b_ak2.dta
#     if 'ak07' in df_ak2.columns:
#         part2 = df_ak2[['pidlink', 'ak07']]
        
#         # Gabungkan dulu AK1 dan AK2
#         full_activity = pd.merge(part1, part2, on='pidlink', how='inner')
        
#         # Gabungkan ke Master
#         master_df = pd.merge(master_df, full_activity, on='pidlink', how='left')
#         print("   SUKSES: Data Aktivitas Fisik Lengkap (Berat, Sedang, Jalan).")
#     else:
#         print("   WARNING: ak07 tidak ditemukan di AK2. Menggunakan AK1 saja.")
#         master_df = pd.merge(master_df, part1, on='pidlink', how='left')
# else:
#     print("   Gagal memuat file AK1 atau AK2.")

# print("\n--- STATUS AKHIR DATASET ---")
# print(master_df.head())
# print(f"Jumlah Kolom: {len(master_df.columns)}")
# print("Kolom tersedia:", master_df.columns.tolist())


# # ==========================================
# # FASE 4 FINAL FIX V2: KOMORBID, STRES, & MAKANAN
# # ==========================================

# print("\n--- FASE 4: KOMORBID, STRES, & MAKANAN (FINAL FIX V2) ---")

# # ---------------------------------------------------------
# # 1. AKUISISI DIABETES (Target: b3b_cd2 - Roster Penyakit)
# # ---------------------------------------------------------
# print("1. Memproses Diabetes (Mencoba b3b_cd2.dta)...")
# df_cd, _ = load_ifls_data('hh14_b3b_dta', 'b3b_cd2.dta') 

# if df_cd is not None:
#     # DEBUG: Tampilkan kolom untuk memastikan
#     # print("   [DEBUG] Kolom di b3b_cd2:", df_cd.columns.tolist()[:10])
    
#     # PERBAIKAN 1: Tambahkan 'cd01type' ke dalam list pencarian
#     possible_col_names = ['cd01type', 'cdtype', 'cd00', 'type', 'cd_type']
#     type_col = next((col for col in possible_col_names if col in df_cd.columns), None)
    
#     if type_col:
#         print(f"   INFO: Kolom tipe penyakit ditemukan: {type_col}")
        
#         # Filter Kode Diabetes (Biasanya '3')
#         df_diabetes = df_cd[df_cd[type_col].isin([3, '3'])]
        
#         if not df_diabetes.empty:
#             # PERBAIKAN 2: Di b3b_cd2, diagnosa biasanya di kolom 'cd01' (bukan cd05)
#             # Pertanyaan: "Pernahkah dokter/perawat memberitahu Anda menderita...?"
#             diag_col = 'cd01' if 'cd01' in df_diabetes.columns else 'cd05'
            
#             if diag_col in df_diabetes.columns:
#                 print(f"   INFO: Menggunakan kolom diagnosa '{diag_col}'")
#                 df_diabetes = df_diabetes[['pidlink', diag_col]].rename(columns={diag_col: 'is_diabetes'})
                
#                 # Bersihkan duplikat (Ambil 1/Ya jika ada konflik)
#                 # IFLS: 1=Ya, 3=Tidak. Sort agar 1 di atas, lalu drop duplicates keep first
#                 df_diabetes = df_diabetes.sort_values('is_diabetes').drop_duplicates(subset='pidlink', keep='first')
                
#                 master_df = pd.merge(master_df, df_diabetes, on='pidlink', how='left')
#                 print("   [SUKSES] Data Diabetes berhasil di-merge.")
#             else:
#                 print(f"   [GAGAL] Kolom diagnosa ({diag_col}) tidak ditemukan.")
#         else:
#             print("   [WARNING] Tidak ada baris dengan kode penyakit 3 (Diabetes).")
#     else:
#         print("   [CRITICAL] Kolom cdtype/cd01type tidak ditemukan di b3b_cd2.")
# else:
#     print("   [SKIP] File b3b_cd2.dta tidak ditemukan.")


# # ---------------------------------------------------------
# # 2. AKUISISI STRES (b3b_ps)
# # ---------------------------------------------------------
# print("\n2. Memproses Tingkat Stres (b3b_ps.dta)...")
# df_ps, _ = load_ifls_data('hh14_b3b_dta', 'b3b_ps.dta')

# if df_ps is not None:
#     # PERBAIKAN LOGIKA: Jangan masukkan pidlink dulu saat pengecekan angka
#     # 1. Ambil semua kolom yang depannya 'ps' DAN belakangnya angka
#     ps_cols_only = [col for col in df_ps.columns if col.startswith('ps') and col[2:].isdigit()]
    
#     # 2. Filter hanya ps01 s.d ps10 (Pertanyaan CES-D)
#     target_ps_vars = [c for c in ps_cols_only if int(c[2:]) <= 10]
    
#     if target_ps_vars:
#         # 3. Baru gabungkan dengan pidlink untuk pengambilan data
#         cols_to_fetch = ['pidlink'] + target_ps_vars
        
#         df_ps_clean = df_ps[cols_to_fetch].drop_duplicates(subset='pidlink')
#         master_df = pd.merge(master_df, df_ps_clean, on='pidlink', how='left')
#         print(f"   [SUKSES] Data Stres ({len(target_ps_vars)} variabel) berhasil di-merge.")
#     else:
#         print("   [WARNING] Tidak ditemukan variabel ps01-ps10.")
# else:
#     print("   [SKIP] File b3b_ps.dta tidak ditemukan.")


# # ---------------------------------------------------------
# # 3. AKUISISI MAKANAN ASIN/MIE (b3b_fm1)
# # ---------------------------------------------------------
# print("\n3. Memproses Konsumsi Mie Instan (b3b_fm1.dta)...")
# df_fm, _ = load_ifls_data('hh14_b3b_dta', 'b3b_fm1.dta')

# if df_fm is not None:
#     # fmtype: Kode Makanan. fm01: Frekuensi.
#     # Cek variasi nama kolom type
#     type_col_fm = next((c for c in ['fmtype', 'fmcode', 'type', 'fm01type'] if c in df_fm.columns), None)
    
#     if type_col_fm:
#         # Kode 12 atau L = Mie Instan
#         target_codes = [12, '12', 'L', 'l'] 
#         df_noodle = df_fm[df_fm[type_col_fm].isin(target_codes)]
        
#         if not df_noodle.empty:
#             df_noodle = df_noodle[['pidlink', 'fm01']].rename(columns={'fm01': 'freq_instant_noodle'})
#             # Ambil max frekuensi jika ada duplikat
#             df_noodle = df_noodle.groupby('pidlink')['freq_instant_noodle'].max().reset_index()
            
#             master_df = pd.merge(master_df, df_noodle, on='pidlink', how='left')
#             print("   [SUKSES] Data Mie Instan berhasil di-merge.")
#         else:
#             print("   [WARNING] Tidak ada data Mie Instan (Kode 12/L).")
#     else:
#         print("   [GAGAL] Kolom tipe makanan tidak ditemukan.")
# else:
#     print("   [SKIP] File b3b_fm1.dta tidak ditemukan.")

# print("\n============================================")
# print("       STATUS AKHIR: FASE 4 SELESAI         ")
# print("============================================")
# print(master_df.head())
# print(f"Total Kolom Final: {len(master_df.columns)}")
# print("Daftar Kolom:", master_df.columns.tolist())

# # Simpan ke CSV Final
# master_df.to_csv('master_dataset_raw_complete.csv', index=False)
# print("File tersimpan: master_dataset_raw_complete.csv")





# import pandas as pd
# import pyreadstat
# import os

# DATA_DIR = 'data'

# def load_ifls_data(subfolder, filename):
#     path = os.path.join(DATA_DIR, subfolder, filename)
#     if not os.path.exists(path):
#         print(f"File tidak ditemukan: {path}")
#         return None, None
#     df, meta = pyreadstat.read_dta(path)
#     df.columns = df.columns.str.lower()
#     return df, meta

# # --- PHASE 1 & 2 RECAP (Sudah Berhasil) ---
# print("--- MEMULAI FASE 1 & 2 ---")
# df_ptrack, _ = load_ifls_data('hh14_trk_dta', 'ptrack.dta')
# df_b3a_cov, _ = load_ifls_data('hh14_b3a_dta', 'b3a_cov.dta')

# if df_ptrack is not None and df_b3a_cov is not None:
#     df_demografi = df_b3a_cov[['pidlink', 'sex', 'age']]
#     # Inisialisasi master_df di sini
#     master_df = pd.merge(df_ptrack[['pidlink']], df_demografi, on='pidlink', how='inner')
#     print(f"Base data loaded: {len(master_df)} rows.")
# else:
#     print("Gagal load base data.")
#     exit()


# print("\n--- FASE 3: GAYA HIDUP (TARGET BARU) ---")

# # ==========================================
# # 1. AKUISISI MEROKOK (Target: Book 3B Modul KM)
# # ==========================================
# print("1. Mencari data merokok di b3b_km.dta...")
# df_smoke, _ = load_ifls_data('hh14_b3b_dta', 'b3b_km.dta')

# if df_smoke is not None:
#     # Variabel Merokok di IFLS-5 KM biasanya:
#     # km01a: Apakah pernah merokok/mengunyah tembakau? (1=Ya, 3=Tidak)
#     # km04: Apakah masih merokok sekarang?
#     # km08: Rata-rata batang rokok per hari
    
#     # Cek ketersediaan kolom
#     cols_check = ['km01a', 'km04', 'km08']
#     available_km = [c for c in cols_check if c in df_smoke.columns]
    
#     if available_km:
#         print(f"   DITEMUKAN: {available_km}")
#         df_smoke_clean = df_smoke[['pidlink'] + available_km]
#         master_df = pd.merge(master_df, df_smoke_clean, on='pidlink', how='left')
#     else:
#         print(f"   GAGAL: Kolom {cols_check} tidak ada di b3b_km.dta")
#         print("   Isi kolom:", df_smoke.columns.tolist()[:10], "...")
# else:
#     print("   File b3b_km.dta tidak ditemukan.")


# # ==========================================
# # 2. AKUISISI AKTIVITAS FISIK (Target: Gabung AK1 + AK2)
# # ==========================================
# print("\n2. Menggabungkan Aktivitas Fisik (AK1 + AK2)...")
# df_ak1, _ = load_ifls_data('hh14_b3b_dta', 'b3b_ak1.dta') # Berat & Sedang
# df_ak2, _ = load_ifls_data('hh14_b3b_dta', 'b3b_ak2.dta') # Jalan Kaki

# if df_ak1 is not None and df_ak2 is not None:
#     # Ambil bagian 1: Berat (ak02) & Sedang (ak05)
#     # Catatan: ak02 = hari aktivitas berat, ak05 = hari aktivitas sedang
#     part1 = df_ak1[['pidlink', 'ak02', 'ak05']]
    
#     # Ambil bagian 2: Jalan Kaki (ak07)
#     # Di output Anda tadi terlihat ada 'ak07' di b3b_ak2.dta
#     if 'ak07' in df_ak2.columns:
#         part2 = df_ak2[['pidlink', 'ak07']]
        
#         # Gabungkan dulu AK1 dan AK2
#         full_activity = pd.merge(part1, part2, on='pidlink', how='inner')
        
#         # Gabungkan ke Master
#         master_df = pd.merge(master_df, full_activity, on='pidlink', how='left')
#         print("   SUKSES: Data Aktivitas Fisik Lengkap (Berat, Sedang, Jalan).")
#     else:
#         print("   WARNING: ak07 tidak ditemukan di AK2. Menggunakan AK1 saja.")
#         master_df = pd.merge(master_df, part1, on='pidlink', how='left')
# else:
#     print("   Gagal memuat file AK1 atau AK2.")

# print("\n--- STATUS AKHIR DATASET ---")
# print(master_df.head())
# print(f"Jumlah Kolom: {len(master_df.columns)}")
# print("Kolom tersedia:", master_df.columns.tolist())



# # ==========================================
# # FASE 4 (FINAL CORRECTED): KOMORBID, STRES, & MAKANAN
# # ==========================================

# print("\n FASE 4: KOMORBID, STRES & MAKANAN ")

# # 1. Akuisisi Diabetes
# print("1. Memproses diabetes (b3b_cd2.dta...)")
# df_cd, _ = load_ifls_data('hh14_b3b_dta', 'b3b_cd2.dta')

# if df_cd is not None:
#     # IFLS Standard Code: 'H' adalah Diabetes (Kencing Manis)
#     # Kita juga antisipasi jika ada kode angka 8 (jarang, tapi mungkin)
#     target_codes = ['H', 'h', 8, '8']
    
# # Filter baris yang kodeny adalah Diabetes
#     # Kita gunakan cd01type sesuai temuan cek_final.py
#     df_diabetes = df_cd[df_cd['cd01type'].isin(target_codes)]
    
#     if not df_diabetes.empty:
#         # cd01: "Pernahkah dokter memberitahu...?" (1=Ya, 3=Tidak)
#         df_diabetes = df_diabetes[['pidlink', 'cd01']].rename(columns={'cd01': 'is_diabetes'})
        
#         # Mapping: Ubah 3 menjadi 0, dan 1 tetap 1 (biar jadi binary 0/1)
#         # Atau biarkan raw dulu. Kita ambil prioritas: Jika pernah 1, ambil 1.
#         df_diabetes = df_diabetes.sort_values('is_diabetes').drop_duplicates(subset='pidlink', keep='first')
        
#         master_df = pd.merge(master_df, df_diabetes, on='pidlink', how='left')
#         print(f"   [SUKSES] Data Diabetes ditemukan ({len(df_diabetes)} responden).")
#     else:
#         print("   [WARNING] Tidak ditemukan responden dengan kode penyakit 'H' (Diabetes).")
#         print("   (Mungkin perlu cek b3b_cd1 atau b3b_cd3, tapi standarnya di cd2)")
# else:
#     print("   [SKIP] File b3b_cd2.dta tidak ditemukan.")


# # ---------------------------------------------------------
# # 2. AKUISISI STRES (b3b_ps - Pivot Table)
# # ---------------------------------------------------------
# print("\n2. Memproses Tingkat Stres (b3b_ps.dta - Pivot)...")
# df_ps, _ = load_ifls_data('hh14_b3b_dta', 'b3b_ps.dta')

# if df_ps is not None:
#     # Filter pertanyaan A sampai J (10 item CES-D)
#     # pstype biasanya 'A', 'B', 'C'... 'J'
#     valid_types = list('ABCDEFGHIJ')
    
#     # Ambil hanya baris yang relevan
#     df_ps_clean = df_ps[df_ps['pstype'].isin(valid_types)]
    
#     if not df_ps_clean.empty:
#         # PIVOT: Mengubah baris menjadi kolom
#         # Index: pidlink
#         # Columns: pstype
#         # Values: ps01 (Skor jawaban: 1=Jarang, 4=Sering)
#         df_ps_pivot = df_ps_clean.pivot_table(
#             index='pidlink', 
#             columns='pstype', 
#             values='ps01', 
#             aggfunc='first' # Ambil nilai pertama jika ada duplikat
#         ).reset_index()
        
#         # Rename kolom biar rapi (ps_A, ps_B, ...)
#         df_ps_pivot.columns = ['pidlink'] + [f'ps_score_{c}' for c in df_ps_pivot.columns if c != 'pidlink']
        
#         master_df = pd.merge(master_df, df_ps_pivot, on='pidlink', how='left')
#         print(f"   [SUKSES] Data Stres berhasil di-pivot & merge ({len(df_ps_pivot.columns)-1} item).")
#     else:
#         print("   [WARNING] Tidak ada data pstype A-J.")
# else:
#     print("   [SKIP] File b3b_ps.dta tidak ditemukan.")


# # ---------------------------------------------------------
# # 3. AKUISISI MAKANAN (Target: b3b_fm12 - Mie Instan)
# # ---------------------------------------------------------
# print("\n3. Memproses Makanan (Mencoba b3b_fm12.dta - Mie Instan)...")
# # PERBAIKAN: Load fm12 (Mie) bukan fm1 (Beras)
# df_mie, _ = load_ifls_data('hh14_b3b_dta', 'b3b_fm12.dta') 

# if df_mie is not None:
#     # fm01: "Berapa hari dalam seminggu lalu makan ini?"
#     df_mie = df_mie[['pidlink', 'fm01']].rename(columns={'fm01': 'freq_instant_noodle'})
    
#     # Cleaning: Pastikan satu pidlink satu baris
#     df_mie = df_mie.groupby('pidlink')['freq_instant_noodle'].max().reset_index()
    
#     master_df = pd.merge(master_df, df_mie, on='pidlink', how='left')
#     print("   [SUKSES] Data Mie Instan (fm12) berhasil di-merge.")
# else:
#     print("   [SKIP] File b3b_fm12.dta tidak ditemukan. (Cek apakah Anda mendownload semua file b3b?)")


# # ---------------------------------------------------------
# # FINALISASI
# # ---------------------------------------------------------
# print("\n============================================")
# print("       STATUS AKHIR: SEMUA FASE SELESAI     ")
# print("============================================")
# print(master_df.head())
# print(f"Total Kolom Final: {len(master_df.columns)}")
# print("Daftar Kolom:", master_df.columns.tolist())

# # Simpan ke CSV Final
# master_df.to_csv('master_dataset_raw_complete.csv', index=False)
# print("File tersimpan: master_dataset_raw_complete.csv")







# import pandas as pd
# import pyreadstat
# import os
# import sys

# # --- KONFIGURASI SISTEM ---
# DATA_DIR = 'data'

# def load_data(subfolder, filename, required=True):
#     """
#     Fungsi pemuat data yang robust (tahan banting).
#     Menangani pengecekan file dan normalisasi nama kolom.
#     """
#     path = os.path.join(DATA_DIR, subfolder, filename)
    
#     # 1. Pengecekan Eksistensi File
#     if not os.path.exists(path):
#         msg = f"[SKIP] File tidak ditemukan: {filename}"
#         if required:
#             print(f"[CRITICAL] {msg}")
#         else:
#             print(msg)
#         return None, None
    
#     # 2. Pemuatan Data Aman
#     try:
#         df, meta = pyreadstat.read_dta(path)
#         df.columns = df.columns.str.lower() # Normalisasi lowercase
#         return df, meta
#     except Exception as e:
#         print(f"[ERROR] File korup {filename}: {e}")
#         return None, None

# def main():
#     print("--- IFLS PIPELINE: INITIALIZATION ---")
    
#     # ---------------------------------------------------------
#     # 1. IDENTITAS INTI (Demografi)
#     # ---------------------------------------------------------
#     print("\n[1/5] Memproses Demografi Inti...")
#     df_ptrack, _ = load_data('hh14_trk_dta', 'ptrack.dta')
#     df_cov, _ = load_data('hh14_b3a_dta', 'b3a_cov.dta')
    
#     if df_ptrack is None or df_cov is None:
#         print("[FATAL] File inti hilang. Abort.")
#         return

#     # Merge Base Population
#     df_demo = df_cov[['pidlink', 'sex', 'age']]
#     master_df = pd.merge(df_ptrack[['pidlink']], df_demo, on='pidlink', how='inner')
#     print(f"   -> Populasi dasar: {len(master_df)} responden.")


#     # ---------------------------------------------------------
#     # 2. GAYA HIDUP (Merokok & Aktivitas)
#     # ---------------------------------------------------------
#     print("\n[2/5] Memproses Fitur Gaya Hidup...")
    
#     # Merokok
#     df_smoke, _ = load_data('hh14_b3b_dta', 'b3b_km.dta')
#     if df_smoke is not None:
#         # km01a: Pernah merokok? km04: Masih merokok? km08: Batang/hari
#         cols = [c for c in ['km01a', 'km04', 'km08'] if c in df_smoke.columns]
#         if cols:
#             df_s_clean = df_smoke[['pidlink'] + cols]
#             master_df = pd.merge(master_df, df_s_clean, on='pidlink', how='left')
#             print(f"   -> Data Merokok berhasil di-merge ({len(cols)} vars).")

#     # Aktivitas Fisik (Gabungan AK1 + AK2)
#     df_ak1, _ = load_data('hh14_b3b_dta', 'b3b_ak1.dta')
#     df_ak2, _ = load_data('hh14_b3b_dta', 'b3b_ak2.dta')
    
#     if df_ak1 is not None and df_ak2 is not None:
#         # AK1: Berat(ak02) / Sedang(ak05)
#         part1 = df_ak1[['pidlink', 'ak02', 'ak05']]
        
#         # AK2: Jalan Kaki(ak07) - Cek keberadaan kolom
#         target_col_ak2 = 'ak07' if 'ak07' in df_ak2.columns else None
        
#         if target_col_ak2:
#             part2 = df_ak2[['pidlink', target_col_ak2]]
#             full_act = pd.merge(part1, part2, on='pidlink', how='inner')
#             master_df = pd.merge(master_df, full_act, on='pidlink', how='left')
#             print("   -> Aktivitas Fisik Lengkap (Berat, Sedang, Jalan) berhasil di-merge.")
#         else:
#             print("   -> [WARN] Data Jalan Kaki (ak07) tidak ditemukan di AK2.")
#             master_df = pd.merge(master_df, part1, on='pidlink', how='left')


#     # ---------------------------------------------------------
#     # 3. KONDISI KRONIS (Diabetes - Kode H)
#     # ---------------------------------------------------------
#     print("\n[3/5] Memproses Penyakit Kronis (Diabetes)...")
#     # TARGET KOREKSI: Buku 3B, File CD3
#     df_cd3, _ = load_data('hh14_b3b_dta', 'b3b_cd3.dta')
    
#     if df_cd3 is not None:
#         # 1. Identifikasi Kolom Tipe
#         type_col = next((c for c in ['cdtype', 'cd01type'] if c in df_cd3.columns), None)
        
#         if type_col:
#             # 2. Filter Kode Diabetes ('H')
#             df_db = df_cd3[df_cd3[type_col].isin(['H', 'h'])]
            
#             # 3. Ekstraksi Diagnosa (cd05)
#             # Berdasarkan schema check, cd05 adalah kolom diagnosa utama di file ini
#             if 'cd05' in df_db.columns:
#                 df_db = df_db[['pidlink', 'cd05']].rename(columns={'cd05': 'is_diabetes'})
                
#                 # Strategi Deduplikasi: Prioritaskan '1' (Ya) daripada '3' (Tidak)
#                 # Sort ascending (1 di atas 3) -> Ambil yang pertama
#                 df_db = df_db.sort_values('is_diabetes').drop_duplicates(subset='pidlink', keep='first')
                
#                 master_df = pd.merge(master_df, df_db, on='pidlink', how='left')
                
#                 count_pos = len(df_db[df_db['is_diabetes'] == 1])
#                 print(f"   -> Data Diabetes berhasil di-merge. Kasus Positif: {count_pos}")
#             else:
#                 print("   -> [ERR] Kolom diagnosa 'cd05' tidak ditemukan di CD3.")
#         else:
#             print("   -> [ERR] Kolom 'cdtype' tidak ditemukan di CD3.")


#     # ---------------------------------------------------------
#     # 4. KESEHATAN MENTAL (Stres - Pivot)
#     # ---------------------------------------------------------
#     print("\n[4/5] Memproses Kesehatan Mental (CES-D)...")
#     df_ps, _ = load_data('hh14_b3b_dta', 'b3b_ps.dta')
    
#     if df_ps is not None:
#         # Filter hanya pertanyaan valid (A-J)
#         valid_types = list('ABCDEFGHIJ') 
#         mask = df_ps['pstype'].isin(valid_types)
#         df_ps_clean = df_ps[mask]
        
#         # Pivot: Ubah Baris menjadi Kolom
#         pivot_ps = df_ps_clean.pivot_table(
#             index='pidlink', 
#             columns='pstype', 
#             values='ps01', 
#             aggfunc='first'
#         ).reset_index()
        
#         # Ratakan nama kolom (Flatten)
#         pivot_ps.columns = ['pidlink'] + [f'ps_{c}' for c in pivot_ps.columns if c != 'pidlink']
        
#         master_df = pd.merge(master_df, pivot_ps, on='pidlink', how='left')
#         print(f"   -> Data Stres berhasil di-pivot dan merge ({len(pivot_ps.columns)-1} item).")


#     # ---------------------------------------------------------
#     # 5. DIET (Mie Instan - Proxy Garam)
#     # ---------------------------------------------------------
#     print("\n[5/5] Memproses Diet (Mie Instan)...")
#     # Target: b3b_fm12 (File spesifik Mie)
#     # Note: required=False karena mungkin user belum download
#     df_mie, _ = load_data('hh14_b3b_dta', 'b3b_fm12.dta', required=False)
    
#     if df_mie is not None:
#         if 'fm01' in df_mie.columns:
#             df_mie = df_mie[['pidlink', 'fm01']].rename(columns={'fm01': 'freq_instant_noodle'})
#             # Agregasi Max (jika ada duplikat)
#             df_mie = df_mie.groupby('pidlink')['freq_instant_noodle'].max().reset_index()
#             master_df = pd.merge(master_df, df_mie, on='pidlink', how='left')
#             print("   -> Data Diet berhasil di-merge.")
#         else:
#             print("   -> [ERR] Kolom frekuensi 'fm01' hilang di FM12.")
#     else:
#         print("   -> [INFO] Data diet dilewati (File tidak ditemukan/belum didownload).")


#     # ---------------------------------------------------------
#     # FINALISASI
#     # ---------------------------------------------------------
#     print("\n--- PIPELINE SELESAI ---")
#     print(f"Dimensi Dataset Akhir: {master_df.shape}")
#     print("Daftar Kolom:", master_df.columns.tolist())
    
#     output_filename = 'master_dataset_raw_final.csv'
#     master_df.to_csv(output_filename, index=False)
#     print(f"Dataset tersimpan di: {output_filename}")

# if __name__ == "__main__":
#     main()
    
    
    



import pandas as pd
import pyreadstat
import os
import sys

# --- KONFIGURASI SISTEM ---
DATA_DIR = 'data'

def load_data(subfolder, filename, required=True):
    path = os.path.join(DATA_DIR, subfolder, filename)
    if not os.path.exists(path):
        msg = f"[SKIP] File tidak ditemukan: {filename}"
        if required:
            print(f"[CRITICAL] {msg}")
        else:
            print(msg)
        return None, None
    
    try:
        df, meta = pyreadstat.read_dta(path)
        df.columns = df.columns.str.lower()
        return df, meta
    except Exception as e:
        print(f"[ERROR] File korup {filename}: {e}")
        return None, None

def main():
    print("--- IFLS PIPELINE: FINAL PRODUCTION BUILD ---")
    
    # ---------------------------------------------------------
    # 1. DEMOGRAFI (Usia, Jenis Kelamin)
    # ---------------------------------------------------------
    print("\n[1/7] Memproses Demografi...")
    df_ptrack, _ = load_data('hh14_trk_dta', 'ptrack.dta')
    df_cov, _ = load_data('hh14_b3a_dta', 'b3a_cov.dta')
    
    if df_ptrack is None or df_cov is None: return

    df_demo = df_cov[['pidlink', 'sex', 'age']]
    master_df = pd.merge(df_ptrack[['pidlink']], df_demo, on='pidlink', how='inner')
    print(f"   -> Populasi: {len(master_df)} responden.")

    # ---------------------------------------------------------
    # 2. GAYA HIDUP (Merokok & Aktivitas Fisik)
    # ---------------------------------------------------------
    print("\n[2/7] Memproses Gaya Hidup...")
    # Merokok
    df_smoke, _ = load_data('hh14_b3b_dta', 'b3b_km.dta')
    if df_smoke is not None:
        cols = [c for c in ['km01a', 'km04', 'km08'] if c in df_smoke.columns]
        if cols:
            master_df = pd.merge(master_df, df_smoke[['pidlink'] + cols], on='pidlink', how='left')

    # Aktivitas Fisik
    df_ak1, _ = load_data('hh14_b3b_dta', 'b3b_ak1.dta')
    df_ak2, _ = load_data('hh14_b3b_dta', 'b3b_ak2.dta')
    if df_ak1 is not None:
        part1 = df_ak1[['pidlink', 'ak02', 'ak05']]
        if df_ak2 is not None and 'ak07' in df_ak2.columns:
            part2 = df_ak2[['pidlink', 'ak07']]
            full_act = pd.merge(part1, part2, on='pidlink', how='inner')
            master_df = pd.merge(master_df, full_act, on='pidlink', how='left')
            print("   -> Gaya Hidup merged.")

    # ---------------------------------------------------------
    # 3. PENYAKIT KRONIS (Diabetes - Fixed CD3)
    # ---------------------------------------------------------
    print("\n[3/7] Memproses Diabetes (CD3)...")
    df_cd3, _ = load_data('hh14_b3b_dta', 'b3b_cd3.dta')
    if df_cd3 is not None:
        type_col = next((c for c in ['cdtype', 'cd01type'] if c in df_cd3.columns), None)
        if type_col:
            # Filter Kode 'H' (Diabetes)
            df_db = df_cd3[df_cd3[type_col].isin(['H', 'h'])]
            if 'cd05' in df_db.columns:
                df_db = df_db[['pidlink', 'cd05']].rename(columns={'cd05': 'is_diabetes'})
                df_db = df_db.sort_values('is_diabetes').drop_duplicates(subset='pidlink', keep='first')
                master_df = pd.merge(master_df, df_db, on='pidlink', how='left')
                print(f"   -> Diabetes merged. Positif: {len(df_db[df_db['is_diabetes']==1])}")

    # ---------------------------------------------------------
    # 4. KESEHATAN MENTAL (Stres - Pivot)
    # ---------------------------------------------------------
    print("\n[4/7] Memproses Stres (CES-D)...")
    df_ps, _ = load_data('hh14_b3b_dta', 'b3b_ps.dta')
    if df_ps is not None:
        valid_types = list('ABCDEFGHIJ')
        df_ps_clean = df_ps[df_ps['pstype'].isin(valid_types)]
        pivot_ps = df_ps_clean.pivot_table(index='pidlink', columns='pstype', values='ps01', aggfunc='first').reset_index()
        pivot_ps.columns = ['pidlink'] + [f'ps_{c}' for c in pivot_ps.columns if c != 'pidlink']
        master_df = pd.merge(master_df, pivot_ps, on='pidlink', how='left')
        print("   -> Stres merged.")

    # ---------------------------------------------------------
    # 5. DIET (Mie Instan - Fixed FM2)
    # ---------------------------------------------------------
    print("\n[5/7] Memproses Diet (Mie Instan - FM2)...")
    df_fm2, _ = load_data('hh14_b3b_dta', 'b3b_fm2.dta')
    if df_fm2 is not None:
        # Kode Mie Instan = 12 atau 'L'
        # fmtype: Jenis Makanan
        # fm02: Apakah makan? (1=Ya, 3=Tidak)
        # fm03: Jumlah hari (Frekuensi)
        
        # Filter Kode Mie
        target_codes = [12, '12', 'L', 'l']
        df_mie = df_fm2[df_fm2['fmtype'].isin(target_codes)]
        
        # Tentukan kolom frekuensi (fm03 biasanya hari, fm02 binary)
        val_col = 'fm03' if 'fm03' in df_mie.columns else 'fm02'
        
        if not df_mie.empty:
            df_mie = df_mie[['pidlink', val_col]].rename(columns={val_col: 'freq_instant_noodle'})
            # Ambil frekuensi tertinggi jika duplikat
            df_mie = df_mie.groupby('pidlink')['freq_instant_noodle'].max().reset_index()
            master_df = pd.merge(master_df, df_mie, on='pidlink', how='left')
            print(f"   -> Diet merged (Sumber: fm2, Kolom: {val_col}).")
        else:
            print("   -> [WARN] Tidak ada responden makan mie instan di FM2.")

# ---------------------------------------------------------
    # 6. PENGUKURAN FISIK (KOREKSI SCHEMA & HITUNG IMT)
    # ---------------------------------------------------------
    print("\n[6/7] Memproses Fisik & Target Tensi (US) - Schema Corrected...")
    df_us, _ = load_data('hh14_bus_dta', 'bus_us.dta')
    
    if df_us is not None:
        # Mapping hasil audit metadata
        rename_map = {
            'us06': 'weight_kg',      # Berat
            'us04': 'height_cm',      # Tinggi
            'us06a': 'waist_cm',      # Lingkar Pinggang
            'us07a1': 'bp_systolic',  # Sistolik 1
            'us07a2': 'bp_diastolic'  # Diastolik 1 (KOREKSI!)
        }
        
        target_cols = ['pidlink'] + list(rename_map.keys())
        existing_cols = [c for c in target_cols if c in df_us.columns]
        
        df_phys = df_us[existing_cols].rename(columns=rename_map)
        
        # --- FEATURE ENGINEERING: IMT (BMI) ---
        # Rumus: BB / (TB dalam meter)^2
        # Kita gunakan .copy() untuk menghindari SettingWithCopyWarning
        df_phys = df_phys.copy()
        
        # Validasi TB tidak nol untuk menghindari division by zero
        mask_bmi = (df_phys['height_cm'] > 0) & (df_phys['weight_kg'] > 0)
        df_phys.loc[mask_bmi, 'bmi'] = df_phys['weight_kg'] / ((df_phys['height_cm'] / 100) ** 2)
        
        df_phys = df_phys.drop_duplicates(subset='pidlink')
        master_df = pd.merge(master_df, df_phys, on='pidlink', how='left')
        print(f"   -> Fisik, Tensi, & IMT berhasil di-merge. Kolom: {list(df_phys.columns)}")

   # ---------------------------------------------------------
    # 7. RIWAYAT ORANG TUA (SMART SEARCH FALLBACK)
    # ---------------------------------------------------------
    print("\n[7/7] Memproses Riwayat Ortu (BA) - Smart Search...")
    df_ba, _ = load_data('hh14_b3b_dta', 'b3b_ba1.dta')
    
    if df_ba is not None:
        # Prioritas Pencarian Kolom:
        # 1. ba03/ba31 (Standar Baku)
        # 2. ba15_a/ba15_b (Alternatif Metadata)
        # 3. ba15 (Versi Lama)
        
        # Logika Pencarian Ayah
        col_ayah = None
        for cand in ['ba03', 'ba15_a', 'ba15']:
            if cand in df_ba.columns:
                col_ayah = cand
                break
        
        # Logika Pencarian Ibu
        col_ibu = None
        for cand in ['ba31', 'ba15_b', 'ba42']:
            if cand in df_ba.columns:
                col_ibu = cand
                break
        
        # Eksekusi Merge
        cols_to_merge = ['pidlink']
        rename_dict = {}
        
        if col_ayah:
            cols_to_merge.append(col_ayah)
            rename_dict[col_ayah] = 'father_health'
            print(f"   -> Ayah ditemukan di kolom: {col_ayah}")
            
        if col_ibu:
            cols_to_merge.append(col_ibu)
            rename_dict[col_ibu] = 'mother_health'
            print(f"   -> Ibu ditemukan di kolom: {col_ibu}")

        if len(cols_to_merge) > 1:
            df_parents = df_ba[cols_to_merge].rename(columns=rename_dict)
            df_parents = df_parents.drop_duplicates(subset='pidlink')
            master_df = pd.merge(master_df, df_parents, on='pidlink', how='left')
            print("   -> [SUKSES] Riwayat Ortu berhasil di-merge.")
        else:
            print("   -> [GAGAL] Tidak ada variabel kesehatan ortu yang cocok.")
            
            
    # ---------------------------------------------------------
    # FINALISASI
    # ---------------------------------------------------------
    print("\n--- PIPELINE SELESAI ---")
    # PENTING: Cegah ledakan data duplikat di tahap akhir
    master_df = master_df.drop_duplicates(subset='pidlink')
    
    print(f"Dimensi Akhir: {master_df.shape}")
    print("Kolom:", master_df.columns.tolist())
    master_df.to_csv('master_dataset_raw_final.csv', index=False)
    print("File tersimpan: master_dataset_raw_final.csv")

if __name__ == "__main__":
    main()