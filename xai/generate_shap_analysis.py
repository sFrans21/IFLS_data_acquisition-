"""
generate_shap_analysis.py
=========================
Analisis SHAP untuk model XGBoost prediksi hipertensi.

Menghasilkan bahan untuk dokumen TA :
  - Analisis GLOBAL : beeswarm + bar mean(|SHAP|)
  - Analisis PER-FITUR : dependence/scatter plot untuk tiap fitur
  - Analisis LOKAL : waterfall untuk satu individu + nilai base value & f(x)
  - Perbandingan TreeExplainer vs KernelExplainer (opsional) -> tabel waktu & selisih


Cara pakai:
    1. Sesuaikan blok KONFIGURASI di bawah (path model & dataset).
    2. Letakkan xgboost_hipertensi_model.pkl & dataset di folder yang sama.
    3. Jalankan:  python generate_shap_analysis.py
    4. Semua gambar + ringkasan angka (summary.json) tersimpan di folder ./shap_output/
"""

import os
import json
import time
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non-interaktif: langsung tulis ke file, tanpa jendela
import matplotlib.pyplot as plt
import shap

# ===================== KONFIGURASI =====================
MODEL_PATH   = "xgboost_hipertensi_model.pkl"      # path model terlatih Anda
DATASET_PATH = "dataset_hipertensi_prepared.csv"   # path dataset siap-pakai
TARGET_COL   = "label_hypertension"                # kolom target (dibuang dari fitur)
OUTPUT_DIR   = "shap_output"                        # folder keluaran
LOCAL_INDEX  = 0                                    # indeks sampel untuk analisis lokal
RANDOM_STATE = 42

RUN_KERNEL_COMPARISON = True   # set False bila ingin lewati (KernelExplainer lambat)
COMPARE_N    = 50              # jumlah sampel yg dibandingkan Tree vs Kernel
BACKGROUND_N = 100            # ukuran background dataset untuk Kernel & probability
# =======================================================


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def save_current_fig(fig_name):
    """Simpan figure matplotlib yang sedang aktif lalu tutup."""
    plt.savefig(os.path.join(OUTPUT_DIR, fig_name), bbox_inches="tight", dpi=150)
    plt.close()


def resolve_feature_order(model, df_features):
    """
    Tentukan urutan fitur SESUAI saat training.
    Ini krusial: SHAP memetakan nilai ke fitur berdasarkan posisi kolom, sehingga
    urutan yang salah akan menghasilkan interpretasi yang salah TANPA error.
    Jangan hanya mengandalkan urutan kolom CSV.
    """
    for getter in (
        lambda: list(model.feature_names_in_),            # XGBClassifier (sklearn API)
        lambda: list(model.get_booster().feature_names),  # Booster di balik wrapper
    ):
        try:
            names = getter()
            if names:
                return names
        except Exception:
            continue
    print("[PERINGATAN] Nama fitur tidak ditemukan di model; memakai urutan kolom CSV.")
    return list(df_features.columns)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- 1. Muat model & data ----------
    with open(MODEL_PATH, "rb") as f:
        loaded_data = pickle.load(f)
        
    # Cek apakah yang dimuat adalah dictionary
    if isinstance(loaded_data, dict):
        print(f"Isi dictionary: {loaded_data.keys()}")
        # GANTI 'model' dengan key yang tepat berdasarkan output print di atas.
        # Biasanya key-nya bernama 'model', 'xgb_model', atau 'classifier'
        model = loaded_data.get('model', loaded_data) 
    else:
        model = loaded_data

    df = pd.read_csv(DATASET_PATH)
    X_all = df.drop(columns=[TARGET_COL])

    feature_names = resolve_feature_order(model, X_all)
    X = X_all[feature_names].copy()          # susun ulang sesuai urutan training
    print("Urutan fitur dipakai:", feature_names)
    print("Jumlah sampel:", X.shape[0])

    # ---------- 2. TreeExplainer (default: ruang log-odds) ----------
    explainer = shap.TreeExplainer(model)
    t0 = time.perf_counter()
    sv = explainer(X)                        # objek Explanation: .values .base_values .data
    tree_time = time.perf_counter() - t0
    print(f"TreeExplainer selesai untuk {X.shape[0]} sampel dalam {tree_time:.4f} s")

    # ---------- 3. Analisis GLOBAL ----------
    # Beeswarm: sebaran kontribusi tiap fitur di seluruh data (warna = nilai fitur)
    shap.plots.beeswarm(sv, max_display=len(feature_names), show=False)
    save_current_fig("global_beeswarm.png")

    # Bar: ranking fitur berdasarkan rata-rata |SHAP| (kepentingan global)
    shap.plots.bar(sv, max_display=len(feature_names), show=False)
    save_current_fig("global_bar_mean_abs.png")

    # ---------- 4. Analisis PER-FITUR (dependence/scatter) ----------
    for feat in feature_names:
        shap.plots.scatter(sv[:, feat], show=False)
        save_current_fig(f"dependence_{feat}.png")

    # ---------- 5. Analisis LOKAL (satu individu) ----------
    shap.plots.waterfall(sv[LOCAL_INDEX], max_display=len(feature_names), show=False)
    save_current_fig(f"local_waterfall_idx{LOCAL_INDEX}.png")

    bvals = np.array(sv.base_values).reshape(-1)
    base_value = float(bvals[LOCAL_INDEX]) if bvals.size > 1 else float(bvals[0])
    local_contrib = sv.values[LOCAL_INDEX]
    fx_logodds = base_value + float(np.sum(local_contrib))


# ---------- 5b. Analisis Ambang Batas Netral (SHAP ≈ 0) untuk SEMUA Fitur ----------
    print("\n" + "="*50)
    print("ANALISIS AMBANG BATAS NETRAL (SHAP = 0) SEMUA FITUR")
    print("="*50)
    
    all_thresholds = {}

    for i, feat in enumerate(feature_names):
        feat_vals = X[feat].values
        shap_vals = sv.values[:, i]
        df_feat = pd.DataFrame({"val": feat_vals, "shap": shap_vals})
        
        # Cek jumlah nilai unik untuk membedakan fitur kontinu vs biner/kategorikal
        unique_vals = np.sort(df_feat["val"].unique())
        
        if len(unique_vals) <= 5:
            # --- KASUS 1: FITUR KATEGORIKAL / BINER ---
            cat_summary = {}
            for u in unique_vals:
                subset = df_feat[df_feat["val"] == u]
                mean_s = float(subset["shap"].mean())
                cat_summary[str(u)] = round(mean_s, 4)
            
            all_thresholds[feat] = {
                "tipe_fitur": "ordinal/biner",
                "mean_shap_per_kategori": cat_summary
            }
            print(f"[{feat}] (Kategorikal): Rata-rata SHAP per kelas -> {cat_summary}")
            
        else:
            # --- KASUS 2: FITUR KONTINU ---
            # Cara paling stabil untuk TA: Ambil 5% sampel dengan nilai |SHAP| paling dekat dengan 0
            df_feat["abs_shap"] = np.abs(df_feat["shap"])
            n_neutral = max(5, int(len(df_feat) * 0.05))  # minimal 5 sampel
            neutral_zone = df_feat.nsmallest(n_neutral, "abs_shap")
            
            neutral_mean = float(neutral_zone["val"].mean())
            neutral_min = float(neutral_zone["val"].min())
            neutral_max = float(neutral_zone["val"].max())
            
            # Cek korelasi untuk mengetahui arah risiko
            corr = np.corrcoef(df_feat["val"], df_feat["shap"])[0, 1]
            arah = "Positif (Semakin tinggi fitur -> Risiko naik)" if corr > 0 else "Negatif (Semakin tinggi fitur -> Risiko turun)"
            
            all_thresholds[feat] = {
                "tipe_fitur": "kontinu",
                "arah_korelasi": arah,
                "est_titik_netral_mean": round(neutral_mean, 2),
                "rentang_zona_netral": [round(neutral_min, 2), round(neutral_max, 2)]
            }
            print(f"[{feat}] (Kontinu): Titik netral ≈ {neutral_mean:.2f} (Zona Netral: {neutral_min:.2f} s.d. {neutral_max:.2f}) | Tren: {arah}")

    print("="*50 + "\n")

    # ---------- 6. Ringkasan angka untuk narasi dokumen ----------
    # Disediakan dalam log-odds (asli TreeExplainer) DAN probabilitas (sigmoid),
    # supaya Anda bisa memilih penyajian mana yang dipakai di dokumen.
    summary = {
        "n_samples": int(X.shape[0]),
        "feature_names": feature_names,
        "tree_time_s": round(tree_time, 6),
        "global_mean_abs_shap": {
            f: float(np.mean(np.abs(sv.values[:, i])))
            for i, f in enumerate(feature_names)
        },
        "local_index": LOCAL_INDEX,
        "base_value_logodds": base_value,
        "fx_logodds": fx_logodds,
        "base_value_prob": float(sigmoid(base_value)),
        "fx_prob": float(sigmoid(fx_logodds)),
        "local_shap_logodds": {
            f: float(local_contrib[i]) for i, f in enumerate(feature_names)
        },
    }
    
    summary["analisis_ambang_batas_semua_fitur"] = all_thresholds

    # ---------- 7. (Opsional) Perbandingan Tree vs Kernel ----------
    # Dibandingkan di RUANG PROBABILITAS agar adil (KernelExplainer bekerja di
    # predict_proba). TreeExplainer di-set probability + interventional + background.
    if RUN_KERNEL_COMPARISON:
        try:
            bg = shap.sample(X, BACKGROUND_N, random_state=RANDOM_STATE)
            Xc = X.iloc[:COMPARE_N]

            expl_tree_p = shap.TreeExplainer(
                model, data=bg,
                feature_perturbation="interventional",
                model_output="probability",
            )
            t0 = time.perf_counter()
            sv_tree_p = np.array(expl_tree_p.shap_values(Xc))
            tree_p_time = time.perf_counter() - t0

            f_prob = lambda data: model.predict_proba(data)[:, 1]
            expl_kernel = shap.KernelExplainer(f_prob, bg)
            t0 = time.perf_counter()
            sv_kernel = np.array(expl_kernel.shap_values(Xc, nsamples="auto"))
            kernel_time = time.perf_counter() - t0

            # samakan bentuk bila ada dimensi kelas tambahan
            if sv_tree_p.ndim == 3:
                sv_tree_p = sv_tree_p[..., -1]
            if sv_kernel.ndim == 3:
                sv_kernel = sv_kernel[..., -1]

            mean_abs_diff = float(np.mean(np.abs(sv_tree_p - sv_kernel)))
            denom = np.sum(np.abs(sv_kernel), axis=1, keepdims=True)
            denom[denom == 0] = np.nan
            contrib_diff_pct = float(
                np.nanmean(np.abs(sv_tree_p - sv_kernel) / denom) * 100
            )

            summary["explainer_comparison"] = {
                "compare_n": COMPARE_N,
                "tree_time_s": round(tree_p_time, 6),
                "kernel_time_s": round(kernel_time, 6),
                "mean_abs_diff_shap_prob": mean_abs_diff,
                "mean_contrib_diff_pct": contrib_diff_pct,
            }
            print("Perbandingan explainer:",
                  json.dumps(summary["explainer_comparison"], indent=2))
        except Exception as e:
            summary["explainer_comparison_error"] = str(e)
            print("[INFO] Perbandingan Kernel dilewati:", e)

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSelesai. Semua keluaran ada di folder: {OUTPUT_DIR}/")
    print("Kirim balik isi summary.json + gambar-gambarnya untuk penulisan narasi dokumen.")


if __name__ == "__main__":
    main()