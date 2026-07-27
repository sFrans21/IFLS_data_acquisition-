"""
============================================================================
EKSPERIMEN — PENGARUH ENCODING SUKU BANGSA (versi MURNI FREKUENSI)
Tahap: MODELLING (CRISP-DM), dijalankan setelah data_preparation.

Aturan grouping (UNSUPERVISED — hanya frekuensi responden, TIDAK menyentuh target):
    Suku DIPERTAHANKAN jika jumlah responden di TRAIN >= threshold * len(train).
    Default threshold = 1%. Selebihnya digabung ke "Lainnya".

Pemetaan diturunkan dari DATA LATIH saja lalu diterapkan ke data uji, untuk
menghindari bias optimistik pada estimasi performa
(Moscovich & Rosset, 2022; Kapoor & Narayanan, 2023).

Tiga lengan dibandingkan pada split & hyperparameter IDENTIK:
    Arm 0 : NO_SUKU   -> tanpa fitur suku (acuan; = pipeline saat ini)
    Arm A : RAW       -> suku one-hot penuh (~28 kategori)
    Arm B : GROUPED   -> suku one-hot setelah pooling 1% (~13 kategori)

Metrik: ROC-AUC (objektif tuning Anda) DAN PR-AUC (lensa kelas minoritas).
PR-AUC memakai auc(recall, precision), SAMA dengan train_model.py.
============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------
TARGET_COL = "label_hypertension"
KOL_SUKU   = "ethnicity"        # nama kolom suku di CSV (dari ar15d)
LABEL_LAIN = "Lainnya"

THRESHOLD_PROPORSI = 0.01       # aturan 1% (murni frekuensi responden)
SEED = 42

# Hyperparameter XGBoost IDENTIK untuk semua lengan (isolasi efek encoding).
XGB_PARAMS = dict(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    subsample=1.0, eval_metric="logloss", random_state=SEED, n_jobs=4,
)

# Robustness opsional: repeated Stratified K-Fold pada data gabungan.
ROBUSTNESS_CV = False
CV_SPLITS, CV_REPEATS = 5, 3


def garis(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def hitung_metrik(y_true, y_proba):
    precision, rec, _ = precision_recall_curve(y_true, y_proba)
    return {"ROC-AUC": roc_auc_score(y_true, y_proba),
            "PR-AUC":  auc(rec, precision)}


# ---------------------------------------------------------------------------
# MAPPING GROUPING — MURNI FREKUENSI, diturunkan dari TRAIN
# ---------------------------------------------------------------------------
def buat_mapping_frekuensi(suku_train, threshold_proporsi=THRESHOLD_PROPORSI,
                           label_lain=LABEL_LAIN):
    n = len(suku_train)
    batas = threshold_proporsi * n
    total = suku_train.value_counts()

    dipertahankan = [s for s in total.index if total[s] >= batas]
    dibuang       = [s for s in total.index if total[s] <  batas]
    mapping = {s: (s if s in dipertahankan else label_lain) for s in total.index}
    info = dict(batas=batas, total=total,
                dipertahankan=dipertahankan, dibuang=dibuang)
    return mapping, info


def siapkan_fitur(train_df, test_df, arm, mapping=None):
    base_cols = [c for c in train_df.columns if c not in (TARGET_COL, KOL_SUKU)]
    Xtr = train_df[base_cols].copy()
    Xte = test_df[base_cols].copy()
    if arm == "NO_SUKU":
        return Xtr, Xte

    suku_tr = train_df[KOL_SUKU].astype("object")
    suku_te = test_df[KOL_SUKU].astype("object")
    if arm == "GROUPED":
        suku_tr = suku_tr.map(mapping).fillna(LABEL_LAIN)
        suku_te = suku_te.map(mapping).fillna(LABEL_LAIN)   # kategori baru -> Lainnya

    dtr = pd.get_dummies(suku_tr, prefix="suku")
    dte = pd.get_dummies(suku_te, prefix="suku").reindex(columns=dtr.columns, fill_value=0)
    Xtr = pd.concat([Xtr.reset_index(drop=True), dtr.reset_index(drop=True)], axis=1)
    Xte = pd.concat([Xte.reset_index(drop=True), dte.reset_index(drop=True)], axis=1)
    return Xtr, Xte


def latih_dan_evaluasi(Xtr, ytr, Xte, yte):
    neg, pos = np.bincount(ytr)
    model = XGBClassifier(scale_pos_weight=neg / pos, **XGB_PARAMS)
    model.fit(Xtr, ytr)
    return hitung_metrik(yte, model.predict_proba(Xte)[:, 1])


# ===========================================================================
def main():
    garis("MEMBACA DATA HASIL DATA PREPARATION")
    train_df = pd.read_csv("train_hipertensi.csv")
    test_df  = pd.read_csv("test_hipertensi.csv")
    print(f"  Train: {train_df.shape} | Test: {test_df.shape}")
    if KOL_SUKU not in train_df.columns:
        raise SystemExit(f"[ERROR] Kolom '{KOL_SUKU}' tidak ada di CSV. "
                         f"Terapkan dulu patch data_preparation.")

    y_train, y_test = train_df[TARGET_COL], test_df[TARGET_COL]

    # --- Mapping frekuensi dari TRAIN ---
    garis("KEPUTUSAN GROUPING (murni frekuensi, dari TRAIN)")
    mapping, info = buat_mapping_frekuensi(train_df[KOL_SUKU].astype("object"))
    print(f"  n_train = {len(train_df)} | batas 1% = {info['batas']:.1f} responden\n")

    tabel = pd.DataFrame(
        [(s, int(info["total"][s]),
          "DIPERTAHANKAN" if s in info["dipertahankan"] else "-> Lainnya")
         for s in info["total"].index],
        columns=["Suku", "N_resp(train)", "Status"])
    print(tabel.to_string(index=False))
    tabel.to_csv("eksperimen_suku_keputusan_grouping.csv", index=False)
    print(f"\n  Kategori: RAW={train_df[KOL_SUKU].nunique()} -> "
          f"GROUPED={len(set(mapping.values()))}")
    print(f"  Dipertahankan ({len(info['dipertahankan'])}): {info['dipertahankan']}")
    print(f"  Digabung ({len(info['dibuang'])}): {info['dibuang']}")

    # --- Tiga lengan pada split yang sama ---
    garis("PERBANDINGAN PERFORMA (held-out test set)")
    hasil = []
    for nama, arm, mp in [("0. Tanpa Suku", "NO_SUKU", None),
                          ("A. Suku RAW",   "RAW",     None),
                          ("B. Suku GROUPED", "GROUPED", mapping)]:
        Xtr, Xte = siapkan_fitur(train_df, test_df, arm, mapping=mp)
        m = latih_dan_evaluasi(Xtr, y_train, Xte, y_test)
        hasil.append({"Lengan": nama, "n_fitur": Xtr.shape[1], **m})

    df_hasil = pd.DataFrame(hasil).set_index("Lengan")[["n_fitur", "ROC-AUC", "PR-AUC"]].round(4)
    print(df_hasil.to_string())

    base = df_hasil.loc["0. Tanpa Suku"]
    print("\n  Selisih terhadap 'Tanpa Suku':")
    for lengan in ["A. Suku RAW", "B. Suku GROUPED"]:
        print(f"    {lengan:<18} dROC-AUC={df_hasil.loc[lengan,'ROC-AUC']-base['ROC-AUC']:+.4f}"
              f"  dPR-AUC={df_hasil.loc[lengan,'PR-AUC']-base['PR-AUC']:+.4f}")
    print("\n  Selisih GROUPED - RAW (inti pertanyaan Anda):")
    print(f"    dROC-AUC={df_hasil.loc['B. Suku GROUPED','ROC-AUC']-df_hasil.loc['A. Suku RAW','ROC-AUC']:+.4f}"
          f"  dPR-AUC={df_hasil.loc['B. Suku GROUPED','PR-AUC']-df_hasil.loc['A. Suku RAW','PR-AUC']:+.4f}")

    df_hasil.to_csv("eksperimen_suku_hasil.csv")
    with open("eksperimen_suku_hasil.tex", "w", encoding="utf-8") as f:
        f.write(df_hasil.to_latex(float_format="%.4f",
                caption="Perbandingan performa berdasarkan perlakuan encoding suku bangsa",
                label="tab:eksperimen_suku", position="htbp"))

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(df_hasil)); w = 0.35
    ax.bar(x - w/2, df_hasil["ROC-AUC"], w, label="ROC-AUC")
    ax.bar(x + w/2, df_hasil["PR-AUC"],  w, label="PR-AUC")
    for i, (r, p) in enumerate(zip(df_hasil["ROC-AUC"], df_hasil["PR-AUC"])):
        ax.text(i - w/2, r + 0.005, f"{r:.3f}", ha="center", fontsize=9)
        ax.text(i + w/2, p + 0.005, f"{p:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(df_hasil.index, rotation=10)
    ax.set_ylim(0, 1.0); ax.set_ylabel("Skor")
    ax.set_title("Pengaruh encoding suku bangsa terhadap performa (test set)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig("eksperimen_suku_barchart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n  -> tersimpan: eksperimen_suku_hasil.csv/.tex, _barchart.png, _keputusan_grouping.csv")

    # Robustness opsional
    if ROBUSTNESS_CV:
        garis("ROBUSTNESS: Repeated Stratified K-Fold (mapping per-fold dari train-fold)")
        full = pd.concat([train_df, test_df], ignore_index=True)
        yfull = full[TARGET_COL].values
        agg = {"NO_SUKU": [], "RAW": [], "GROUPED": []}
        for rep in range(CV_REPEATS):
            skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED + rep)
            for tr_idx, va_idx in skf.split(full, yfull):
                tr, va = full.iloc[tr_idx], full.iloc[va_idx]
                mp, _ = buat_mapping_frekuensi(tr[KOL_SUKU].astype("object"))
                for arm in agg:
                    Xtr, Xva = siapkan_fitur(tr, va, arm, mapping=mp if arm == "GROUPED" else None)
                    agg[arm].append(latih_dan_evaluasi(Xtr, tr[TARGET_COL], Xva, va[TARGET_COL]))
        print(f"  (rata-rata atas {CV_SPLITS*CV_REPEATS} fold)")
        for arm in ["NO_SUKU", "RAW", "GROUPED"]:
            roc = np.array([m["ROC-AUC"] for m in agg[arm]])
            pr  = np.array([m["PR-AUC"]  for m in agg[arm]])
            print(f"    {arm:<9} ROC-AUC {roc.mean():.4f}±{roc.std():.4f} | "
                  f"PR-AUC {pr.mean():.4f}±{pr.std():.4f}")

    garis("SELESAI")


if __name__ == "__main__":
    main()