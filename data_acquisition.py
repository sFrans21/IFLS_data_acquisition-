import pandas as pd
import pyreadstat
import os
 
DATA_DIR = 'data'
OUTPUT_RAW = 'master_dataset_raw_final.csv'
 
 
def load_data(subfolder, filename, required=True):
    path = os.path.join(DATA_DIR, subfolder, filename)
    if not os.path.exists(path):
        msg = f"[SKIP] File tidak ditemukan: {filename}"
        print(f"[CRITICAL] {msg}" if required else msg)
        return None, None
    try:
        df, meta = pyreadstat.read_dta(path)
        df.columns = df.columns.str.lower()
        return df, meta
    except Exception as e:
        print(f"[ERROR] File korup {filename}: {e}")
        return None, None
 
 
def merge_module(master_df, df_module, cols_needed, rename_map, module_name,
                  filter_col=None, filter_vals=None):
    """
    Merge satu modul ke master_df TANPA resolusi duplikat apa pun.
 
    filter_col/filter_vals dipakai untuk memilih 'tipe baris' pada skema
    long-format (contoh: cdtype == 'B' untuk diabetes). Ini seleksi baris
    berdasarkan tipe pertanyaan, bukan perubahan isi data.
    """
    if df_module is None:
        print(f"   -> [SKIP] Modul {module_name} tidak tersedia.")
        return master_df
 
    df = df_module
    if filter_col is not None:
        if filter_col not in df.columns:
            print(f"   -> [WARN] Kolom filter '{filter_col}' tidak ditemukan di {module_name}.")
            return master_df
        df = df[df[filter_col].isin(filter_vals)]
        if df.empty:
            print(f"   -> [WARN] Tidak ada baris dengan {filter_col} in {filter_vals} di {module_name}.")
            return master_df
 
    missing_cols = [c for c in cols_needed if c not in df.columns]
    if missing_cols:
        print(f"   -> [WARN] Kolom tidak ditemukan di {module_name}: {missing_cols}")
        return master_df
 
    df_sel = df[cols_needed].rename(columns=rename_map)
 
    n_before = len(master_df)
    master_df = pd.merge(master_df, df_sel, on=['hhid14', 'pid14'], how='left')
    n_after = len(master_df)
 
    if n_after != n_before:
        print(f"   -> [PERHATIAN] Baris master_df berubah dari {n_before} -> {n_after} "
              f"setelah merge {module_name} (indikasi pidlink duplikat di modul ini).")
    print(f"   -> {module_name} merged mentah (kolom: {list(rename_map.values())}).")
    return master_df
 
 
def main():
    print("Mengekstraksi Data IFLS (TAHAP: DATA UNDERSTANDING) ...")
 
    # 1. DEMOGRAFI
    print("\n[1/7] Memproses Demografi...")
    # df_ptrack, _ = load_data('hh14_trk_dta', 'ptrack.dta')
    df_cov, _ = load_data('hh14_b3a_dta', 'b3a_cov.dta')
    if df_cov is None:
        return
 
    df_demo = df_cov[['hhid14', 'pid14', 'sex', 'age']]
    # master_df = pd.merge(df_ptrack[['hhid14', 'pid14']], df_demo, on='hhid14', 'pid14', how='right')
    master_df = df_demo
    print(f"   -> Populasi awal: {len(master_df)} baris.")
    
    # # 2. SUKU BANGSA
    # print("\n[2/7] Memproses Suku Bangsa...")
    # df_ar1, _ = load_data("hh14_bk_dta", "bk_ar1.dta")

    # master_df = merge_module(
    #     master_df,
    #     df_ar1,
    #     cols_needed=["pidlink", "ar15d"],
    #     rename_map={
    #         "ar15d": "ethnicity"
    #     },
    #     module_name="AR1 (Suku Bangsa)"
    # )
 
    # 2. MEROKOK
    print("\n[3/7] Memproses Kebiasaan Merokok...")
    df_smoke, _ = load_data('hh14_b3b_dta', 'b3b_km.dta')
    master_df = merge_module(
        master_df, df_smoke,
        cols_needed=['hhid14', 'pid14', 'km01a'],
        rename_map={'km01a': 'has_tobacco'},
        module_name='KM (Merokok)'
    )
 
    # 3. AKTIVITAS FISIK (KK2)
    # Diganti dari pivot_table(aggfunc='first') menjadi 3 filter terpisah,
    # konsisten dengan pola modul lain, dan tanpa aggregasi/pemilihan nilai.
    print("\n[4/7] Memproses Aktivitas Fisik...")
    df_kk2, _ = load_data('hh14_b3b_dta', 'b3b_kk2.dta')
    

    kk2_map = {
        'A': ('hard_act_last7d'),
        'B': ('moderate_act_last7d'),
        'C': ('walking_last7d'),
    }
    for kode, kk02m_name in kk2_map.items():
        master_df = merge_module(
            master_df, df_kk2,
            cols_needed=['hhid14', 'pid14', 'kk02m'],
            rename_map={'kk02m': kk02m_name},
            module_name=f'KK2 ({kk02m_name})',
            filter_col='kktype', filter_vals=[kode, kode.lower()]
        )
 
    # 4. DIABETES & KOLESTEROL & PENYAKIT GINJAL & STROKE (CD3)
    print("\n[5/7] Memproses Penyakit Kronis (CD3)...")
    df_cd3, _ = load_data('hh14_b3b_dta', 'b3b_cd3.dta')
    master_df = merge_module(
        master_df, df_cd3,
        cols_needed=['hhid14', 'pid14', 'cd05'],
        rename_map={'cd05': 'is_diabetes'},
        module_name='CD3 (Diabetes)',
        filter_col='cdtype', filter_vals=['B', 'b']
    )
    master_df = merge_module(
        master_df, df_cd3,
        cols_needed=['hhid14', 'pid14', 'cd05'],
        rename_map={'cd05': 'is_high_cholesterol'},
        module_name='CD3 (Kolesterol)',
        filter_col='cdtype', filter_vals=['M', 'm']
    )
    master_df = merge_module(
        master_df, df_cd3,
        cols_needed=['hhid14', 'pid14', 'cd05'],
        rename_map={'cd05': 'is_kidney_disease'},
        module_name='CD3 (Penyakit Ginjal)',
        filter_col='cdtype', filter_vals=['O', 'o']
    )
    master_df = merge_module(
        master_df, df_cd3,
        cols_needed=['hhid14', 'pid14', 'cd05'],
        rename_map={'cd05': 'is_stroke'},
        module_name='CD3 (Stroke)',
        filter_col='cdtype', filter_vals=['H', 'h']
    )
 
    # 5. DIET (FM2)
    print("\n[6/7] Memproses Diet (FM2)...")
    df_fm2, _ = load_data('hh14_b3b_dta', 'b3b_fm2.dta')
    fm2_map = {
        'K': ('has_noodles'),
        'L': ('has_fast_food'),
        'O': ('has_fried_food')
    }
    for kode, (fm02_name) in fm2_map.items():
        master_df = merge_module(
            master_df,
            df_fm2,
            cols_needed=['hhid14', 'pid14', 'fm02'],
            rename_map={
                'fm02': fm02_name
            },
            module_name=f'FM2 ({fm02_name})',
            filter_col='fmtype',
            filter_vals=[kode, kode.lower()]
        )
 
    # 6. FISIK & TENSI (US)
    print("\n[7/7] Memproses Fisik & Tensi (US)...")
    df_us, _ = load_data('hh14_bus_dta', 'bus_us.dta')
    rename_map_us = {
        'us06': 'weight_kg',
        'us04': 'height_cm',
        'us06a': 'waist_cm',
        'us07a1': 'bp_systolic',
        'us07a2': 'bp_diastolic',
    }
    if df_us is not None:
        cols_needed = ['hhid14', 'pid14'] + [c for c in rename_map_us if c in df_us.columns]
        master_df = merge_module(
            master_df, df_us,
            cols_needed=cols_needed,
            rename_map=rename_map_us,
            module_name='US (Fisik & Tensi)'
        )
 
 
    # FINALISASI -- TANPA drop_duplicates, TANPA imputasi apa pun
    print("\n--- AKUISISI SELESAI (DATA MASIH MENTAH) ---")
    print(f"Dimensi Akhir: {master_df.shape}")
    print(f"Jumlah pidlink unik: {master_df[['hhid14', 'pid14']].drop_duplicates()}")

    print("Kolom:", master_df.columns.tolist())
 
    master_df.to_csv(OUTPUT_RAW, index=False)
    print(f"File tersimpan: {OUTPUT_RAW}")
 
 
if __name__ == "__main__":
    main()