"""
EDA pada POPULASI DEWASA (age >= 18) — untuk menyelaraskan eksplorasi data
dengan populasi pemodelan final (30.251 responden). Urutan filter dibuat identik
dengan data_preparation_hipertensi.py:
  bersihkan kode age (>150 -> NaN) -> filter age>=18 -> koreksi BP mustahil ->
  drop BP kosong -> bentuk label.
Keluaran: statistik di konsol + figur PNG (populasi dewasa) ke folder tujuan.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

PATH_DATA = "master_dataset_raw_final.csv"
OUTDIR = "eda_adult_output"
os.makedirs(OUTDIR, exist_ok=True)
H, N = "#C44E52", "#4C72B0"

def garis(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)

def simpan(nama):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, nama), dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  [tersimpan] {os.path.join(OUTDIR, nama)}")


def siapkan_data(path):
    df = pd.read_csv(path, low_memory=False)
    print(f"Populasi awal: {len(df)} responden, {df.shape[1]} kolom")

    for c in ["age", "weight_kg", "height_cm", "waist_cm", "bp_systolic", "bp_diastolic"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # (1) Pembersihan kode tersamar -> NaN
    df.loc[df["age"] > 150, "age"] = np.nan
    df["is_diabetes"] = df["is_diabetes"].where(~df["is_diabetes"].isin([8, 9]))
    df["is_high_cholesterol"] = df["is_high_cholesterol"].where(~df["is_high_cholesterol"].isin([8, 9]))
    df["sleep_quality"] = df["sleep_quality"].where(~df["sleep_quality"].isin([8, 9]))
    df["sleep_disturbance"] = df["sleep_disturbance"].where(~df["sleep_disturbance"].isin([8, 9]))

    # (2) BMI
    df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2

    # (*) FILTER USIA DEWASA >= 18 (identik data_preparation)
    n_before = len(df)
    df = df[df["age"] >= 18].copy()
    print(f"Filter age>=18: {n_before} -> {len(df)} (dihapus {n_before - len(df)})")

    # (3) Koreksi tekanan darah mustahil
    bad_bp = (
        (df["bp_diastolic"] >= df["bp_systolic"]) |
        (df["bp_systolic"] < 60) | (df["bp_systolic"] > 260) |
        (df["bp_diastolic"] < 30) | (df["bp_diastolic"] > 200)
    ).fillna(False)
    print(f"Tekanan darah mustahil disisihkan: {int(bad_bp.sum())} baris")
    df.loc[bad_bp, ["bp_systolic", "bp_diastolic"]] = np.nan

    # simpan salinan adult sebelum drop BP untuk laporan missing value
    df_adult_preBP = df.copy()

    # (4) label
    df = df.dropna(subset=["bp_systolic", "bp_diastolic"]).copy()
    df["hipertensi"] = ((df["bp_systolic"] >= 140) | (df["bp_diastolic"] >= 90)).astype(int)

    # (5) biner
    df["is_female"] = df["sex"].map({1.0: 0, 3.0: 1})
    df["is_smoker"] = df["is_smoking"].map({1.0: 1, 3.0: 0})
    df["diabetes"] = df["is_diabetes"].map({1.0: 1, 3.0: 0})
    df["is_high_cholesterol"] = df["is_high_cholesterol"].map({1.0: 1, 3.0: 0})

    # (6) kelompok
    df["kel_umur"] = pd.cut(df["age"], [0, 30, 40, 50, 60, 70, 200],
                            labels=["<30", "30-39", "40-49", "50-59", "60-69", "70+"])
    df["kat_bmi"] = pd.cut(df["bmi"], [0, 18.5, 24.9, 30, 100],
                           labels=["Underweight", "Normal", "Overweight", "Obesitas"])
    return df, df_adult_preBP


def laporan_missing(df_adult_preBP):
    garis("MISSING VALUE (populasi dewasa, sebelum drop BP)")
    cols = ["pidlink", "sex", "age", "is_smoking", "freq_hard_act", "freq_moderate_act",
            "freq_walking", "is_diabetes", "is_high_cholesterol", "freq_fast_food",
            "freq_soda", "freq_fried_food", "weight_kg", "height_cm", "waist_cm",
            "bp_systolic", "bp_diastolic", "sleep_quality", "sleep_disturbance"]
    cols = [c for c in cols if c in df_adult_preBP.columns]
    miss = (df_adult_preBP[cols].isna().mean() * 100).round(2).sort_values(ascending=False)
    print(f"N populasi dewasa (pra-drop BP): {len(df_adult_preBP)}")
    print(miss.to_string())

    plt.figure(figsize=(10, 5))
    bars = plt.bar(miss.index, miss.values, color="steelblue")
    for bar, val in zip(bars, miss.values):
        plt.text(bar.get_x() + bar.get_width()/2, val, f"{val:.1f}%",
                 ha="center", va="bottom", fontsize=8)
    plt.title("Persentase Missing Value per Fitur (Populasi Dewasa)")
    plt.xlabel("Fitur"); plt.ylabel("Missing Value (%)")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, max(miss.max()*1.15, 5))
    simpan("missing_values.png")


def bagian_B(df):
    garis("B. TARGET HIPERTENSI (dewasa)")
    n = len(df); vc = df["hipertensi"].value_counts().sort_index()
    print(f"Responden bertensi valid : {n}")
    print(f"  Tidak hipertensi (0)   : {vc[0]} ({vc[0]/n:.4%})")
    print(f"  Hipertensi      (1)    : {vc[1]} ({vc[1]/n:.4%})")
    print(f"  Rasio negatif:positif  : {vc[0]/vc[1]:.3f} : 1")

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


def bagian_C(df):
    garis("C. STATISTIK DESKRIPTIF (dewasa)")
    num = ["age", "weight_kg", "height_cm", "waist_cm", "bmi", "bp_systolic", "bp_diastolic"]
    print(df[num].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(2).to_string())

    plt.figure(figsize=(7, 5.8))
    sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": .7}, annot_kws={"size": 9})
    plt.title("Korelasi antar Variabel Numerik")
    simpan("g5_corr.png")


def bagian_D(df):
    garis("D. ANALISIS BIVARIAT (dewasa)")
    print("Korelasi point-biserial dengan hipertensi:")
    kand = ["age", "bmi", "waist_cm", "sleep_quality", "sleep_disturbance",
            "freq_walking", "freq_moderate_act", "freq_hard_act", "freq_fast_food", "freq_soda", "freq_fried_food"]
    rows = []
    for c in kand:
        if c in df.columns:
            d = df[[c, "hipertensi"]].dropna()
            r = d.corr().iloc[0, 1]
            rows.append((c, r, len(d)))
    for c, r, nn in sorted(rows, key=lambda x: -abs(x[1])):
        print(f"  {c:20s}: r={r:+.3f}  (n={nn})")

    gu = df.groupby("kel_umur", observed=True)["hipertensi"].mean()*100
    gb = df.groupby("kat_bmi", observed=True)["hipertensi"].mean()*100
    print("\nTingkat hipertensi per kelompok umur (%):\n" + gu.round(1).to_string())
    print("\nTingkat hipertensi per kategori BMI (%):\n" + gb.round(1).to_string())

    plt.figure(figsize=(7, 4.5))
    b = plt.bar(gu.index.astype(str), gu.values, color=plt.cm.Reds(np.linspace(.35, .9, len(gu))))
    for bar, v in zip(b, gu.values):
        plt.text(bar.get_x()+bar.get_width()/2, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    plt.title("Tingkat Hipertensi per Kelompok Umur")
    plt.ylabel("% hipertensi"); plt.xlabel("Kelompok umur (tahun)")
    plt.ylim(0, gu.max()*1.15); simpan("g6_umur.png")

    plt.figure(figsize=(7, 4.5))
    b = plt.bar(gb.index.astype(str), gb.values, color=plt.cm.Oranges(np.linspace(.35, .9, len(gb))))
    for bar, v in zip(b, gb.values):
        plt.text(bar.get_x()+bar.get_width()/2, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    plt.title("Tingkat Hipertensi per Kategori BMI")
    plt.ylabel("% hipertensi"); plt.xlabel("Kategori BMI (WHO)")
    plt.ylim(0, gb.max()*1.15); simpan("g7_bmi.png")

    feats = [("diabetes", "Diabetes"), ("is_smoker", "Perokok"), ("is_female", "Apakah Perempuan"), ("is_high_cholesterol", "Kolesterol")]
    ya  = [df[df[c] == 1]["hipertensi"].mean()*100 for c, _ in feats]
    tdk = [df[df[c] == 0]["hipertensi"].mean()*100 for c, _ in feats]
    print("\nTingkat hipertensi menurut faktor biner (Ya vs Tidak):")
    for (c, lbl), a, t in zip(feats, ya, tdk):
        print(f"  {lbl:16s}: Ya={a:.1f}%  Tidak={t:.1f}%  (selisih {a-t:+.1f}%)")
    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(feats)); w = .38
    plt.bar(x-w/2, ya, w, label="Ya", color=H); plt.bar(x+w/2, tdk, w, label="Tidak", color=N)
    plt.xticks(x, [l for _, l in feats]); plt.legend()
    plt.ylabel("% hipertensi"); plt.title("Tingkat Hipertensi menurut Faktor Biner (Ya vs Tidak)")
    for i, (a, t) in enumerate(zip(ya, tdk)):
        plt.text(i-w/2, a, f"{a:.0f}%", ha="center", va="bottom", fontsize=8)
        plt.text(i+w/2, t, f"{t:.0f}%", ha="center", va="bottom", fontsize=8)
    simpan("g8_biner.png")

    gs = df.dropna(subset=["sleep_quality"]).groupby("sleep_quality", observed=True)["hipertensi"].mean()*100
    print("\nTingkat hipertensi per skor kualitas tidur (%):\n" + gs.round(1).to_string())
    plt.figure(figsize=(7, 4.5))
    b = plt.bar(gs.index.astype(int).astype(str), gs.values, color=plt.cm.Purples(np.linspace(.4, .9, len(gs))))
    for bar, v in zip(b, gs.values):
        plt.text(bar.get_x()+bar.get_width()/2, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    plt.title("Tingkat Hipertensi per Skor Kualitas Tidur")
    plt.ylabel("% hipertensi"); plt.xlabel("Skor kualitas tidur (1=terbaik, 5=terburuk)")
    plt.ylim(0, gs.max()*1.15); simpan("g9_sleep.png")

    gsd = df.dropna(subset=["sleep_disturbance"]).groupby("sleep_disturbance", observed=True)["hipertensi"].mean()*100
    print("\nTingkat hipertensi per Frekuensi gangguan tidur (%):\n" + gsd.round(1).to_string())


def main():
    print("--- EDA HIPERTENSI (POPULASI DEWASA >=18) ---")
    df, df_adult_preBP = siapkan_data(PATH_DATA)
    laporan_missing(df_adult_preBP)
    bagian_B(df)
    bagian_C(df)
    bagian_D(df)
    garis("EDA DEWASA SELESAI")


if __name__ == "__main__":
    main()
