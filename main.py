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
    
#     
#     # 1. IDENTITAS INTI (Demografi)
#     
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


#     
#     # 2. GAYA HIDUP (Merokok & Aktivitas)
#     
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


#     
#     # 3. KONDISI KRONIS (Diabetes - Kode H)
#     
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


#     
#     # 4. KESEHATAN MENTAL (Stres - Pivot)
#     
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


#     
#     # 5. DIET (Mie Instan - Proxy Garam)
#     
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


#     
#     # FINALISASI
#     
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
import numpy as np
# import missingno as msno


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
    print("Mengekstraksi Data IFLS...")
    
    
    # 1. DEMOGRAFI (Usia, Jenis Kelamin)
    
    print("\n[1/7] Memproses Demografi...")
    df_ptrack, _ = load_data('hh14_trk_dta', 'ptrack.dta')
    df_cov, _ = load_data('hh14_b3a_dta', 'b3a_cov.dta')
    
    if df_ptrack is None or df_cov is None: return

    df_demo = df_cov[['pidlink', 'sex', 'age']]
    master_df = pd.merge(df_ptrack[['pidlink']], df_demo, on='pidlink', how='right')
    print(f"   -> Populasi: {len(master_df)} responden.")
    master_df['age'] = master_df['age'].replace([997, 998, 999], np.nan)    
        # Jalankan ini untuk tes sampling umur di data CD3
    df_test_cd3, _ = load_data('hh14_b3b_dta', 'b3b_cd3.dta')
    if df_test_cd3 is not None:
        # Merge sementara dengan master_df yang sudah punya kolom 'age'
        df_cek = pd.merge(master_df[['pidlink', 'age']], df_test_cd3[['pidlink']].drop_duplicates(), on='pidlink', how='inner')
        print("--- DISTRIBUSI UMUR DI B3B_CD3 ---")
        print(df_cek['age'].describe())

    # 2. GAYA HIDUP (Merokok & Aktivitas Fisik)
    
    print("\n[2/7] Memproses Kebiasaan Merokok...")
    
    # Merokok (Hanya mengekstrak km01a)

    df_smoke, _ = load_data('hh14_b3b_dta', 'b3b_km.dta')

    if df_smoke is not None:

        required_cols = ['pidlink', 'km01a']

        missing_cols = [c for c in required_cols if c not in df_smoke.columns]

        if missing_cols:
            print(f"   -> [WARN] Kolom tidak ditemukan: {missing_cols}")
        else:

            df_smoke_feat = (
                df_smoke[required_cols]
                .rename(columns={
                    'km01a': 'is_smoking'
                })
                .groupby('pidlink')
                .first()
                .reset_index()
            )

            master_df = pd.merge(
                master_df,
                df_smoke_feat,
                on='pidlink',
                how='left'
            )

            print("   -> Variabel KM merged.")


    print("\n[3/7] Memproses Aktivitas Fisik...")

    # Aktivitas Fisik (Hanya menyisakan kk02o saja)
    df_kk2, _ = load_data('hh14_b3b_dta', 'b3b_kk2.dta')
    if df_kk2 is not None:
        # Menghapus kk02m, kk02n1, dan kk02n2, hanya menyisakan kk02o
        val_cols = [c for c in ['kk02o'] if c in df_kk2.columns]
        
        if 'pidlink' in df_kk2.columns and 'kktype' in df_kk2.columns and val_cols:
            # Melakukan pivot hanya untuk kolom kk02o berdasarkan kktype (A, B, C)
            df_pivot = df_kk2.pivot_table(
                index='pidlink', 
                columns='kktype', 
                values=val_cols, 
                aggfunc='first'
            )
            
            # Merapikan nama kolom hasil pivot.
            # Hasil akhir kolom: kk02o_A, kk02o_B, dan kk02o_C
            df_pivot.columns = [f"{val}_{col}" for val, col in df_pivot.columns]
            df_pivot = df_pivot.reset_index()

            df_pivot = df_pivot.rename(columns={
                'kk02o_A': 'freq_hard_act',
                'kk02o_B': 'freq_moderate_act',
                'kk02o_C': 'freq_walking'
            })
            
            # Gabungkan ke master dataframe
            master_df = pd.merge(master_df, df_pivot, on='pidlink', how='left')
            print("   -> Gaya Hidup (Aktivitas Fisik) merged.")

    
    # 3. PENYAKIT KRONIS (Diabetes - Fixed CD3)
    print("\n[4/7] Memproses Penyakit Diabetes...")
    
    df_cd3, _ = load_data('hh14_b3b_dta', 'b3b_cd3.dta')
    
    if df_cd3 is not None:
        if 'cdtype' in df_cd3.columns and 'cd05' in df_cd3.columns:
            
            # 1. Filter hanya untuk Diabetes (Kode 'B' atau 'b')
            target_cd = ['B', 'b']
            df_diabetes = df_cd3[df_cd3['cdtype'].isin(target_cd)]
            df_chol = df_cd3[df_cd3['cdtype'].isin(['M', 'm'])]
            
            if not df_diabetes.empty:
                # 2. Ambil kolom pidlink dan nilai cd05
                df_diab_feat = df_diabetes[['pidlink', 'cd05']].copy()
                
                # 3. Ubah nama kolom agar lebih mudah dipahami di master dataset
                # Nilai asli IFLS: 1 (Ya), 3 (Tidak)
                df_diab_feat = df_diab_feat.rename(columns={'cd05': 'is_diabetes'})
                
                # 4. Tangani kemungkinan anomali duplikasi pidlink (ambil data pertama)
                df_diab_feat = df_diab_feat.groupby('pidlink')['is_diabetes'].first().reset_index()
                
                # 5. Gabungkan ke master dataframe
                master_df = pd.merge(master_df, df_diab_feat, on='pidlink', how='left')
                print("   -> Kondisi Kronis (Diabetes - CD05) merged.")
            else:
                print("   -> [WARN] Tidak ada responden dengan CDTYPE == B (Diabetes).")
        
            if not df_chol.empty:
                df_chol = (df_chol[['pidlink', 'cd05']]
                        .rename(columns={'cd05': 'is_high_cholesterol'})
                        .groupby('pidlink')['is_high_cholesterol'].first().reset_index())
                master_df = pd.merge(master_df, df_chol, on='pidlink', how='left')
                print("   -> Kondisi Kronis (Kolesterol - CD05) merged.")
            else:
                print("   -> [WARN] Tidak ada responden dengan CDTYPE == M (Kolesterol).")

        else:
            print("   -> [WARN] Kolom 'cdtype' atau 'cd05' tidak ditemukan di dataset CD3.")



    
    # 4. DIET (Fast Food - FM2)
    
    print("\n[5/7] Memproses Diet (Modul FM2 - Fast Food)...")
    df_fm2, _ = load_data('hh14_b3b_dta', 'b3b_fm2.dta')
    
    if df_fm2 is not None:
        # Filter hanya kode L (Fast food / makanan cepat saji)
        target_codes = ['L', 'l']
        df_diet = df_fm2[df_fm2['fmtype'].isin(target_codes)]
        
        if not df_diet.empty and 'fm03' in df_diet.columns:
            # Ambil kolom pidlink dan fm03 (mengabaikan fm02)
            df_fast_food = df_diet[['pidlink', 'fm03']].copy()
            
            # Rename fm03 menjadi nama fitur yang spesifik
            df_fast_food = df_fast_food.rename(columns={'fm03': 'freq_fast_food'})
            
            # Jika terdapat anomali duplikasi baris pada pidlink yang sama, ambil data pertama
            df_fast_food = df_fast_food.groupby('pidlink')['freq_fast_food'].first().reset_index()
            
            # Gabungkan dengan master_df
            master_df = pd.merge(master_df, df_fast_food, on='pidlink', how='left')
            print("   -> Diet (Fast Food fm03) merged.")
        else:
            print("   -> [WARN] Data Fast Food atau kolom fm03 tidak ditemukan di FM2.")
            
    # 4b. DIET TAMBAHAN (Soda & Gorengan - FM2, kolom fm03)
    if df_fm2 is not None and 'fm03' in df_fm2.columns:
        fm_targets = {'M': 'freq_soda', 'O': 'freq_fried_food'}
        for kode, nama_fitur in fm_targets.items():
            df_item = df_fm2[df_fm2['fmtype'].isin([kode, kode.lower()])]
            if not df_item.empty:
                df_item = (df_item[['pidlink', 'fm03']]
                           .rename(columns={'fm03': nama_fitur})
                           .groupby('pidlink')[nama_fitur].first().reset_index())
                master_df = pd.merge(master_df, df_item, on='pidlink', how='left')
                print(f"   -> Diet ({nama_fitur}) merged.")
            else:
                print(f"   -> [WARN] Tidak ada data fmtype={kode}.")


    # 5. PENGUKURAN FISIK (KOREKSI SCHEMA & HITUNG IMT)
    
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
        

        # Gabungkan ke master dataframe
        master_df = pd.merge(master_df, df_phys, on='pidlink', how='left')

        print(f"   -> Fisik & Tensi berhasil di-merge. Kolom: {list(df_phys.columns)}")

   
    # 6. KUALITAS TIDUR (Modul TDR)
    
    print("\n[7/7] Memproses Kualitas Tidur (Modul TDR)...")
    
    # Asumsi nama berkas IFLS untuk modul TDR adalah 'b3b_tdr.dta'
    # Sesuaikan nama file jika berbeda di folder data Anda
    df_tdr, _ = load_data('hh14_b3b_dta', 'b3b_tdr.dta')
    
    if df_tdr is not None:
        # Memastikan kolom yang dibutuhkan ada di dalam dataset
        if 'tdrtype' in df_tdr.columns and 'tdr01' in df_tdr.columns:
            
            # Filter hanya untuk baris pertanyaan nomor 2 (Kualitas Tidur)
            # Menggunakan list [2, '2'] untuk mengantisipasi format integer maupun string
            target_tdr = [2, '2']
            df_sleep = df_tdr[df_tdr['tdrtype'].isin(target_tdr)]
            df_dist = df_tdr[df_tdr['tdrtype'].isin([1, '1'])]
            
            if not df_sleep.empty:
                # Ambil kolom pidlink dan nilai tdr01
                df_sleep_qual = df_sleep[['pidlink', 'tdr01']].copy()
                
                # Ubah nama kolom agar deskriptif di dataset akhir
                df_sleep_qual = df_sleep_qual.rename(columns={'tdr01': 'sleep_quality'})
                
                # Mengantisipasi duplikasi pidlink (ambil nilai pertama jika ada)
                df_sleep_qual = df_sleep_qual.groupby('pidlink')['sleep_quality'].first().reset_index()
                
                # Gabungkan dengan master dataframe utama
                master_df = pd.merge(master_df, df_sleep_qual, on='pidlink', how='left')
                print("   -> Kualitas Tidur (TDRTYPE 2) merged.")
            else:
                print("   -> [WARN] Tidak ada responden dengan TDRTYPE == 2.")
        
        
            if not df_dist.empty:
                df_dist = (df_dist[['pidlink', 'tdr01']]
                        .rename(columns={'tdr01': 'sleep_disturbance'})
                        .groupby('pidlink')['sleep_disturbance'].first().reset_index())
                master_df = pd.merge(master_df, df_dist, on='pidlink', how='left')
                print("   -> Gangguan Tidur (TDRTYPE 1) merged.")
            else:
                print("   -> [WARN] Tidak ada responden dengan TDRTYPE == 1.")
        else:
            print("   -> [WARN] Kolom 'tdrtype' atau 'tdr01' tidak ditemukan di dataset TDR.")
    
    # FINALISASI
    
    print("\n--- PIPELINE SELESAI ---")
    # PENTING: Cegah ledakan data duplikat di tahap akhir
    # master_df = master_df.drop_duplicates(subset='pidlink')
    
    print(f"Dimensi Akhir: {master_df.shape}")
    print("Kolom:", master_df.columns.tolist())
    master_df.to_csv('master_dataset_raw_final.csv', index=False)
    print("File tersimpan: master_dataset_raw_final.csv")



    # print("\nTAHAP EVALUASI DUPLIKAT (SEBELUM SIMPAN)")

    # # 1. Cari baris yang memiliki 'pidlink' kembar
    # # keep=False artinya semua baris yang kembar akan ditandai dan diambil
    # duplikat_df = master_df[master_df.duplicated(subset='pidlink', keep=False)]

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
    #     print(duplikat_df[duplikat_df['pidlink'].isin(contoh_id)][['pidlink'] + master_df.columns[:3].tolist()])
        
    #     # 5. Ekspor data duplikat ke CSV terpisah untuk Anda evaluasi secara manual
    #     duplikat_df.to_csv('evaluasi_duplikat_pidlink.csv', index=False)
    #     print("\n File duplikat disimpan ke: 'evaluasi_duplikat_pidlink.csv'")
    #     print("Silakan buka file tersebut untuk melihat mengapa data Anda bisa mengganda.")
    # else:
    #     print(" Aman! Tidak ditemukan duplikat pada kolom 'pidlink'.")

    print("\n--- PIPELINE SELESAI ---")
    print(f"Dimensi Akhir Master (Belum Dihapus): {master_df.shape}")
    master_df.to_csv('master_dataset_raw_final.csv', index=False)
    print("File master tersimpan: master_dataset_raw_final.csv")
    cols = ['km08', 'km10', 'km05aa']


    # msno.matrix(master_df)
if __name__ == "__main__":
    main()