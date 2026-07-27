"""
============================================================================
SCRIPT EKSPLORASI DATA AWAL (EDA) — VERSI HIPERTENSI  [DATASET BARU 15 KOLOM]
Disesuaikan dengan skema hasil akuisisi ulang:
  pidlink, sex, age, has_tobacco, freq_hard_act, freq_moderate_act,
  freq_walking, is_diabetes, weight_kg, height_cm,
  , bp_systolic, bp_diastolic

'pidlink', 'sex', 'age', 'has_tobacco', 'hard_act_last7d', 'moderate_act_last7d', 'walking_last7d', 'is_diabetes', 'is_high_cholesterol', 'is_kidney_disease', 'is_stroke', 'has_noodles', 'fm03_x', 'has_fast_food', 'fm03_y', 'has_fried_food', 'fm03', 'weight_kg', 'height_cm', '', 'bp_systolic', 'bp_diastolic'

Target  : hipertensi (diturunkan dari tekanan darah, JNC-7/WHO)
Catatan : kolom bmi TIDAK ada di dataset -> dihitung sendiri.
Keluaran: statistik di konsol + 9 berkas PNG (g1..g9), satu grafik per berkas.
============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap

sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

PATH_DATA = "master_dataset_raw_final.csv"   # sesuaikan lokasi berkas Anda
H, N = "#C44E52", "#4C72B0"                   # warna: hipertensi & non-hipertensi

def garis(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)

def simpan(nama):
    plt.tight_layout()
    plt.savefig(nama, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  [tersimpan] {nama}")


# ===========================================================================
# PERSIAPAN DATA + PEMBENTUKAN TARGET HIPERTENSI
# ===========================================================================
def siapkan_data(path):
    df = pd.read_csv(path, low_memory=False)
    print(f"Populasi awal: {len(df)} responden, {df.shape[1]} kolom")

    # --- Numerikkan kolom (jaga-jaga terbaca sebagai teks) ---
    for c in ["age", "weight_kg", "height_cm", "bp_systolic", "bp_diastolic"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

#     # --- (1) Pembersihan kode tersamar -> NaN ---
    df.loc[df["age"] > 150, "age"] = np.nan                                       # 998 = tidak tahu
    df["is_diabetes"]   = df["is_diabetes"].where(~df["is_diabetes"].isin([8, 9]))      # 8/9 -> NaN
    df["is_high_cholesterol"]   = df["is_high_cholesterol"].where(~df["is_high_cholesterol"].isin([8, 9]))      # 8/9 -> NaN
    df["is_stroke"]   = df["is_stroke"].where(~df["is_stroke"].isin([8, 9]))      # 8/9 -> NaN
    df["moderate_act_last7d"]   = df["moderate_act_last7d"].where(~df["moderate_act_last7d"].isin([8, 9]))      # 8/9 -> NaN
    df["walking_last7d"]   = df["walking_last7d"].where(~df["walking_last7d"].isin([8, 9]))      # 8/9 -> NaN
    df["hard_act_last7d"]   = df["hard_act_last7d"].where(~df["hard_act_last7d"].isin([8, 9]))      # 8/9 -> NaN


#     --- (2) BMI di filter nilai outlier nya, baru dihitung sendiri ---
    df.loc[
        (df["height_cm"] < 100) | (df["height_cm"] > 200),
        "height_cm"
    ] = np.nan

    df.loc[
        (df["weight_kg"] < 25) | (df["weight_kg"] > 200),
        "weight_kg"
    ] = np.nan
    
    df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2


    # --- (3) Koreksi tekanan darah mustahil SEBELUM membuat label ---
    bad_bp = df["bp_diastolic"] >= df["bp_systolic"].fillna(False)
    print(f"Tekanan darah mustahil disisihkan: {int(bad_bp.sum())} baris")
    df.loc[bad_bp, ["bp_systolic", "bp_diastolic"]] = np.nan

    # --- (4) Pembentukan label hipertensi ---
    df = df.dropna(subset=["bp_systolic", "bp_diastolic"]).copy()
    df["hipertensi"] = ((df["bp_systolic"] >= 140) | (df["bp_diastolic"] >= 90)).astype(int)

    # --- (5) Fitur biner untuk analisis bivariat (kode IFLS 1/3 -> 1/0) ---
    df["is_female"] = df["sex"].map({1.0: 0, 3.0: 1})            # 3 = perempuan
    df["has_tobacco"]    = df["has_tobacco"].map({1.0: 1, 3.0: 0})     # 1 = ya  
    df["is_diabetes"]  = df["is_diabetes"].map({1.0: 1, 3.0: 0})    # 1 = ya
    df["is_high_cholesterol"]  = df["is_high_cholesterol"].map({1.0: 1, 3.0: 0})
    df["is_kidney_disease"]  = df["is_kidney_disease"].map({1.0: 1, 3.0: 0})
    df["is_stroke"]  = df["is_stroke"].map({1.0: 1, 3.0: 0})
    df["has_noodles"]  = df["has_noodles"].map({1.0: 1, 3.0: 0})
    df["has_fast_food"]  = df["has_fast_food"].map({1.0: 1, 3.0: 0})
    df["has_fried_food"]  = df["has_fried_food"].map({1.0: 1, 3.0: 0})
    df["hard_act_last7d"]  = df["hard_act_last7d"].map({1.0: 1, 3.0: 0})
    df["moderate_act_last7d"]  = df["moderate_act_last7d"].map({1.0: 1, 3.0: 0})
    df["walking_last7d"]  = df["walking_last7d"].map({1.0: 1, 3.0: 0})


    # --- (6) Pengelompokan untuk analisis bivariat ---
    df["kel_umur"] = pd.cut(df["age"], [0, 30, 40, 50, 60, 70, 200],
                            labels=["<30", "30-39", "40-49", "50-59", "60-69", "70+"])
    df["kat_bmi"] = pd.cut(df["bmi"], [0, 18.5, 24.9, 30, 100],
                           labels=["Underweight", "Normal", "Overweight", "Obesitas"])
    return df


# ===========================================================================
# A. STRUKTUR DATA
# ===========================================================================
def bagian_A(path):
    garis("A. STRUKTUR & DIMENSI DATA")
    raw = pd.read_csv(path, low_memory=False)
    print("Dimensi (baris, kolom):", raw.shape)
    print("hhid14 dan pid14 unik :", raw[['hhid14', 'pid14']].drop_duplicates().shape[0], "dari", len(raw))
    print("Baris duplikat penuh  :", raw.duplicated().sum())
    # print("\nPersentase nilai hilang per kolom:")
    # print((raw.isna().mean()*100).round(1).sort_values(ascending=False).to_string())

    # =========================
    # Missing Value Report
    # =========================
    missing = (raw.isna().mean() * 100).round().sort_values(ascending=False)

    print("\nPersentase nilai hilang per kolom:")
    print(missing.to_string())

    # =========================
    # Missing Value Plot
    # =========================
    plt.figure(figsize=(10,5))

    bars = plt.bar(
        missing.index,
        missing.values,
        color="steelblue"
    )

    for bar, val in zip(bars, missing.values):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            val,
            f"{int(val)}%",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.title("Persentase Missing Value per Fitur")
    plt.xlabel("Fitur")
    plt.ylabel("Missing Value (%)")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, max(missing.max()*1.15, 5))

    simpan("missing_values.png")

# ===========================================================================
# B. TARGET HIPERTENSI -> Gambar g1..g4
# ===========================================================================
def bagian_B(df):
    garis("B. PENENTUAN & KARAKTERISTIK TARGET HIPERTENSI")
    n = len(df); vc = df["hipertensi"].value_counts().sort_index()
    print(f"Responden bertensi valid : {n}")
    print(f"  Tidak hipertensi (0)   : {vc[0]} ({vc[0]/n:.1%})")
    print(f"  Hipertensi      (1)    : {vc[1]} ({vc[1]/n:.1%})")

    plt.figure(figsize=(7, 4.5))
    b = plt.bar(["Tidak Hipertensi (0)", "Hipertensi (1)"], vc.values, color=[N, H], width=.6)
    for bar, v in zip(b, vc.values):
        plt.text(bar.get_x()+bar.get_width()/2, v, f"{v}\n({v/n:.1%})", ha="center", va="bottom")
    plt.title("Komposisi Kelas Target Hipertensi"); plt.ylabel("Jumlah responden")
    plt.ylim(0, vc.max()*1.15); simpan("g1_target_balance.png")

    plt.figure(figsize=(7, 4.5))
    plt.hist(df["bp_systolic"], bins=60, color=N, edgecolor="white")
    plt.axvline(140, color=H, ls="--", lw=2, label="ambang hipertensi (140)")
    plt.title("Distribusi Tekanan Darah Sistolik")
    plt.xlabel("Sistolik (mmHg)"); plt.ylabel("Frekuensi"); plt.legend()
    simpan("g2_systolic.png")

    plt.figure(figsize=(7, 4.5))
    plt.hist(df["bp_diastolic"], bins=60, color=N, edgecolor="white")
    plt.axvline(90, color=H, ls="--", lw=2, label="ambang hipertensi (90)")
    plt.title("Distribusi Tekanan Darah Diastolik")
    plt.xlabel("Diastolik (mmHg)"); plt.ylabel("Frekuensi"); plt.legend()
    simpan("g3_diastolic.png")

    plt.figure(figsize=(7, 5.2))
    s = df.sample(min(6000, len(df)), random_state=1)
    plt.scatter(s["bp_systolic"], s["bp_diastolic"], c=s["hipertensi"].map({0: N, 1: H}), s=7, alpha=.35)
    plt.axvline(140, color="gray", ls="--"); plt.axhline(90, color="gray", ls="--")
    plt.xlabel("Sistolik (mmHg)"); plt.ylabel("Diastolik (mmHg)")
    plt.title("Sistolik vs Diastolik menurut Status Hipertensi (merah = hipertensi)")
    simpan("g4_scatter.png")


# ===========================================================================
# C. STATISTIK DESKRIPTIF + KORELASI -> Gambar g5
# ===========================================================================
def bagian_C(df):
    garis("C. STATISTIK DESKRIPTIF VARIABEL NUMERIK")
    num = ["age", "weight_kg", "height_cm", "bmi", "bp_systolic", "bp_diastolic"]
    print(df[num].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(2).to_string())

    plt.figure(figsize=(7, 5.8))
    sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": .7}, annot_kws={"size": 9})
    plt.title("Korelasi antar Variabel Numerik")
    simpan("g5_corr.png")


# ===========================================================================
# D. ANALISIS BIVARIAT -> Gambar g6..g9
# ===========================================================================
def bagian_D(df):
    garis("D. ANALISIS BIVARIAT: HUBUNGAN FAKTOR DENGAN HIPERTENSI")

    # Korelasi point-biserial fitur numerik dengan target
    print("Korelasi point-biserial dengan hipertensi:")
    kand = ["age", "bmi", ]
    for c in kand:
        if c in df.columns:
            d = df[[c, "hipertensi"]].dropna()
            r = d.corr().iloc[0, 1]
            print(f"  {c:20s}: r={r:+.3f}  (n={len(d)})")

    gu = df.groupby("kel_umur", observed=True)["hipertensi"].mean()*100
    gb = df.groupby("kat_bmi", observed=True)["hipertensi"].mean()*100
    print("\nTingkat hipertensi per kelompok umur (%):\n" + gu.round(1).to_string())
    print("\nTingkat hipertensi per kategori BMI (%):\n" + gb.round(1).to_string())

    # g6: per kelompok umur
    plt.figure(figsize=(7, 4.5))
    b = plt.bar(gu.index.astype(str), gu.values, color=plt.cm.Reds(np.linspace(.35, .9, len(gu))))
    for bar, v in zip(b, gu.values):
        plt.text(bar.get_x()+bar.get_width()/2, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    plt.title("Tingkat Hipertensi per Kelompok Umur")
    plt.ylabel("% hipertensi"); plt.xlabel("Kelompok umur (tahun)")
    plt.ylim(0, gu.max()*1.15); simpan("g6_umur.png")

    # g7: per kategori BMI
    plt.figure(figsize=(7, 4.5))
    b = plt.bar(gb.index.astype(str), gb.values, color=plt.cm.Oranges(np.linspace(.35, .9, len(gb))))
    for bar, v in zip(b, gb.values):
        plt.text(bar.get_x()+bar.get_width()/2, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    plt.title("Tingkat Hipertensi per Kategori BMI")
    plt.ylabel("% hipertensi"); plt.xlabel("Kategori BMI (WHO)")
    plt.ylim(0, gb.max()*1.15); simpan("g7_bmi.png")
    
    
#     # g8: Jumlah Responden Hipertensi berdasarkan Suku Bangsa
#     # Mapping kode AR15d -> Nama Suku Bangsa
#     ethnicity_map = {
#         1: "Jawa",
#         2: "Sunda",
#         3: "Bali",
#         4: "Batak",
#         5: "Bugis",
#         6: "Tionghoa",
#         7: "Madura",
#         8: "Sasak",
#         10: "Banjar",
#         11: "Bima-Dompu",
#         12: "Makassar",
#         13: "Nias",
#         14: "Palembang",
#         15: "Sumbawa",
#         16: "Toraja",
#         17: "Betawi",
#         18: "Dayak",
#         19: "Melayu",
#         20: "Komering",
#         21: "Ambon",
#         22: "Manado",
#         23: "Aceh",
#         25: "SumBagSel Lain",
#         26: "Banten",
#         27: "Cirebon",
#         28: "Gorontalo",
#         29: "Kutai",
#         95: "Lainnya",
#         99: "Tidak Diketahui"
#     }

#     df["ethnicity_name"] = df["ethnicity"].map(ethnicity_map)

#     ge = (
#         df["ethnicity_name"]
#         .value_counts()
#         .sort_values(ascending=False)
#     )

#     plt.figure(figsize=(14,6))

#     bars = plt.bar(
#         ge.index,
#         ge.values,
#         color=plt.cm.Blues(np.linspace(.35, .9, len(ge)))
#     )

#     for bar, v in zip(bars, ge.values):
#         plt.text(
#             bar.get_x()+bar.get_width()/2,
#             v,
#             f"{v}",
#             ha="center",
#             va="bottom",
#             fontsize=8
#         )

#     plt.title("Jumlah Responden berdasarkan Suku Bangsa")
#     plt.xlabel("Suku Bangsa")
#     plt.ylabel("Jumlah Responden")
#     plt.xticks(rotation=45, ha="right")
#     plt.ylim(0, ge.max()*1.15)

#     plt.tight_layout()

#     simpan("g8_ethnicity_total.png")


# # Jumlah yang hipertensi berdasarkan suku bangsa
#     ge = (
#         df[df["hipertensi"] == 1]["ethnicity_name"]
#         .value_counts()
#         .sort_values(ascending=False)
#     )

#     plt.figure(figsize=(14,6))

#     bars = plt.bar(
#         ge.index,
#         ge.values,
#         color=plt.cm.Oranges(np.linspace(.35, .9, len(ge)))
#     )

#     for bar, v in zip(bars, ge.values):
#         plt.text(
#             bar.get_x()+bar.get_width()/2,
#             v,
#             f"{v}",
#             ha="center",
#             va="bottom",
#             fontsize=8
#         )

#     plt.title("Jumlah Responden Hipertensi berdasarkan Suku Bangsa")
#     plt.xlabel("Suku Bangsa")
#     plt.ylabel("Jumlah Responden Hipertensi")
#     plt.xticks(rotation=45, ha="right")
#     plt.ylim(0, ge.max()*1.15)

#     plt.tight_layout()

#     simpan("g8_ethnicity_hypertension.png")


    # g9: faktor biner Ya vs Tidak
    feats = [("is_diabetes", "Diabetes"), ("has_tobacco", "Perokok"), ("is_female", "Apakah Perempuan"), ("is_high_cholesterol", "Kolesterol"), ("is_kidney_disease", "Penyakit Ginjal")]
    ya  = [df[df[c] == 1]["hipertensi"].mean()*100 for c, _ in feats]
    tdk = [df[df[c] == 0]["hipertensi"].mean()*100 for c, _ in feats]
    print("\nTingkat hipertensi menurut faktor biner (Ya vs Tidak):")
    for (c, lbl), a, t in zip(feats, ya, tdk):
        print(f"  {lbl:10s}: Ya={a:.1f}%  Tidak={t:.1f}%  (selisih {a-t:+.1f}%)")
    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(feats)); w = .38
    plt.bar(x-w/2, ya, w, label="Ya", color=H); plt.bar(x+w/2, tdk, w, label="Tidak", color=N)
    plt.xticks(x, [l for _, l in feats]); plt.legend()
    plt.ylabel("% hipertensi"); plt.title("Tingkat Hipertensi menurut Faktor Biner (Ya vs Tidak)")
    for i, (a, t) in enumerate(zip(ya, tdk)):
        plt.text(i-w/2, a, f"{a:.0f}%", ha="center", va="bottom", fontsize=8)
        plt.text(i+w/2, t, f"{t:.0f}%", ha="center", va="bottom", fontsize=8)
    simpan("g8_biner_part1.png")
    
    
    feats = [("moderate_act_last7d", "Kegiatan fisik sedang"), ("hard_act_last7d", "Kegiatan fisik berat"), ("walking_last7d", "Jalan Kaki"), ("is_stroke", "Stroke")]
    ya  = [df[df[c] == 1]["hipertensi"].mean()*100 for c, _ in feats]
    tdk = [df[df[c] == 0]["hipertensi"].mean()*100 for c, _ in feats]
    print("\nTingkat hipertensi menurut faktor biner (Ya vs Tidak):")
    for (c, lbl), a, t in zip(feats, ya, tdk):
        print(f"  {lbl:10s}: Ya={a:.1f}%  Tidak={t:.1f}%  (selisih {a-t:+.1f}%)")
    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(feats)); w = .38
    plt.bar(x-w/2, ya, w, label="Ya", color=H); plt.bar(x+w/2, tdk, w, label="Tidak", color=N)
    plt.xticks(x, [l for _, l in feats]); plt.legend()
    plt.ylabel("% hipertensi"); plt.title("Tingkat Hipertensi menurut Faktor Biner (Ya vs Tidak)")
    for i, (a, t) in enumerate(zip(ya, tdk)):
        plt.text(i-w/2, a, f"{a:.0f}%", ha="center", va="bottom", fontsize=8)
        plt.text(i+w/2, t, f"{t:.0f}%", ha="center", va="bottom", fontsize=8)
    simpan("g8_biner_part2.png")
    

    feats = [
        ("has_fast_food", "konsumsi fast-food 7 hari terakhir"),
        ("has_fried_food", "konsumsi gorengan 7 hari terakhir"),
        ("has_noodles", "konsumsi mie instan 7 hari terakhir")
    ]

    ya  = [df[df[c] == 1]["hipertensi"].mean() * 100 for c, _ in feats]
    tdk = [df[df[c] == 0]["hipertensi"].mean() * 100 for c, _ in feats]

    print("\nTingkat hipertensi menurut faktor biner (Ya vs Tidak):")
    for (c, lbl), a, t in zip(feats, ya, tdk):
        print(f"  {lbl:35s}: Ya={a:.1f}%  Tidak={t:.1f}%  (selisih {a-t:+.1f}%)")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(feats))
    w = 0.38

    ax.bar(x - w/2, ya, w, label="Ya", color=H)
    ax.bar(x + w/2, tdk, w, label="Tidak", color=N)

    # bungkus label supaya tidak saling tabrakan
    labels = [textwrap.fill(l, width=18) for _, l in feats]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)

    ax.set_ylabel("% hipertensi")
    ax.set_title("Tingkat Hipertensi menurut Faktor Biner (Ya vs Tidak)")
    ax.legend()

    for i, (a, t) in enumerate(zip(ya, tdk)):
        ax.text(i - w/2, a, f"{a:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, t, f"{t:.0f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)  # penting untuk label 2 baris
    simpan("g8_biner_part3.png")

    # # g9: per tingkat kualitas tidur (FITUR BARU)
    # gs = df.dropna(subset=["sleep_quality"]).groupby("sleep_quality", observed=True)["hipertensi"].mean()*100
    # print("\nTingkat hipertensi per skor kualitas tidur (%):\n" + gs.round(1).to_string())
    # plt.figure(figsize=(7, 4.5))
    # b = plt.bar(gs.index.astype(int).astype(str), gs.values, color=plt.cm.Purples(np.linspace(.4, .9, len(gs))))
    # for bar, v in zip(b, gs.values):
    #     plt.text(bar.get_x()+bar.get_width()/2, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    # plt.title("Tingkat Hipertensi per Skor Kualitas Tidur")
    # plt.ylabel("% hipertensi"); plt.xlabel("Skor kualitas tidur")
    # plt.ylim(0, gs.max()*1.15); simpan("g9_sleep_quality.png")
    
    #     # g10: per tingkat gangguan tidur (FITUR BARU)
    # gs = df.dropna(subset=["sleep_disturbance"]).groupby("sleep_disturbance", observed=True)["hipertensi"].mean()*100
    # print("\nTingkat hipertensi per Frekuensi gangguan tidur (%):\n" + gs.round(1).to_string())
    # plt.figure(figsize=(7, 4.5))
    # b = plt.bar(gs.index.astype(int).astype(str), gs.values, color=plt.cm.Purples(np.linspace(.4, .9, len(gs))))
    # for bar, v in zip(b, gs.values):
    #     plt.text(bar.get_x()+bar.get_width()/2, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    # plt.title("Tingkat Hipertensi per Frekuensi Gangguan Tidur")
    # plt.ylabel("% hipertensi"); plt.xlabel("Frekuensi gangguan tidur")
    # plt.ylim(0, gs.max()*1.15); simpan("g10_sleep_frequency.png")


def main():
    print("--- MEMULAI EDA HIPERTENSI (DATASET BARU) ---")
    df = siapkan_data(PATH_DATA)
    bagian_A(PATH_DATA)
    bagian_B(df)
    bagian_C(df)
    bagian_D(df)
    garis("EDA SELESAI")
    print("11 grafik dihasilkan: g1_target_balance.png - g8_biner_part3.png ")


if __name__ == "__main__":
    main()