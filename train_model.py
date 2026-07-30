

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import (
#     roc_auc_score, recall_score, f1_score, confusion_matrix, accuracy_score,
#     precision_recall_curve, auc, roc_curve, ConfusionMatrixDisplay
# )

# # Import Models
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier

# # SIMPAN MODEL
# import joblib

# # Visualisasi
# import matplotlib
# matplotlib.use('Agg')   # backend non-interaktif: hanya simpan file, tidak buka window
# import matplotlib.pyplot as plt

# import warnings
# warnings.filterwarnings('ignore')

# import os

# OUTPUT_DIR = "hasil_modelling"
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# def out_path(filename):
#     return os.path.join(OUTPUT_DIR, filename)


# # ---------------------------------------------------------
# # FASE 1: MEMBACA DATA HASIL PREPROCESSING
# #  (Flowchart: "Baca data train dan data test dari preprocessing",
# #              "Pisahkan fitur non-labeling dan status hipertensi",
# #              "Hitung rasio kelas (scale_pos_weight) untuk XGBoost")
# #  Split & imputasi sudah dilakukan di tahap data preparation.
# # ---------------------------------------------------------
# TARGET_COL = "label_hypertension"

# train_df = pd.read_csv("train_hipertensi.csv")
# test_df = pd.read_csv("test_hipertensi.csv")

# print("=== DATA TRAIN ===", train_df.shape)
# print("=== DATA TEST  ===", test_df.shape)

# X_train = train_df.drop(columns=[TARGET_COL])
# y_train = train_df[TARGET_COL]
# X_test = test_df.drop(columns=[TARGET_COL])
# y_test = test_df[TARGET_COL]

# print("\nJumlah fitur :", X_train.shape[1])
# print("\nDistribusi kelas TRAIN")
# print((y_train.value_counts(normalize=True) * 100).round(2))
# print("\nDistribusi kelas TEST")
# print((y_test.value_counts(normalize=True) * 100).round(2))
# print("\nMissing value TRAIN :", X_train.isna().sum().sum())
# print("Missing value TEST  :", X_test.isna().sum().sum())

# # Rasio kelas untuk XGBoost (class weighting)
# neg, pos = np.bincount(y_train)
# scale_pos_weight = neg / pos


# # =========================================================
# # FUNGSI BANTU EVALUASI
# #  (Flowchart: "Evaluasi Model" -> ROC-AUC, PR-AUC, F1, Recall,
# #   Specificity, Accuracy)
# # =========================================================
# def evaluate(y_true, y_pred, y_proba):
#     """Kembalikan dict metrik standar."""
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     specificity = tn / (tn + fp)
#     precision, rec, _ = precision_recall_curve(y_true, y_proba)
#     return {
#         'ROC-AUC': roc_auc_score(y_true, y_proba),
#         'Recall (Sens)': recall_score(y_true, y_pred),
#         'Specificity': specificity,
#         'PR-AUC': auc(rec, precision),
#         'F1-Score': f1_score(y_true, y_pred),
#         'Akurasi': accuracy_score(y_true, y_pred),
#     }


# weighted_models = {}
# weighted_predictions = {}

# # ---------------------------------------------------------
# # FASE 2 & 3: ABLATION STUDY (2 SKENARIO)
# #  (Flowchart: "Jalankan skenario" ->
# #     1. Pure Baseline (tanpa class weighting)
# #     2. Baseline dengan class weighting
# #   lalu loop antar algoritma & antar skenario, tanpa tuning)
# # ---------------------------------------------------------
# print("\nMEMULAI ABLATION STUDY (2 SKENARIO)...\n" + "-" * 50)

# # CATATAN (ketidaksesuaian flowchart #4):
# # Kotak "Pilih algoritma" pada flowchart menuliskan:
# #   Logistic Regression, CatBoost, XGBoost, CatBoost
# # ("CatBoost" muncul dua kali, tanpa Random Forest). Ini kemungkinan
# # besar salah ketik pada diagram -- melatih CatBoost dua kali tidak
# # bermakna. Di sini dipakai 4 algoritma distinct (LR, RF, XGB, CatBoost).
# # >> Jika diagram memang dimaksud tanpa Random Forest, hapus entri
# #    'Random Forest' di bawah dan sesuaikan diagram/laporan.
# configs = {
#     'Logistic Regression': {
#         'base': LogisticRegression(random_state=42, max_iter=1000),
#         'base_weighted': LogisticRegression(
#             random_state=42, max_iter=1000, class_weight='balanced'
#         ),
#     },
#     'Random Forest': {
#         'base': RandomForestClassifier(random_state=42),
#         'base_weighted': RandomForestClassifier(
#             random_state=42, class_weight='balanced'
#         ),
#     },
#     'XGBoost': {
#         'base': XGBClassifier(random_state=42, eval_metric='logloss'),
#         'base_weighted': XGBClassifier(
#             random_state=42,
#             eval_metric='logloss',
#             scale_pos_weight=scale_pos_weight,
#         ),
#     },
#     'CatBoost': {
#         'base': CatBoostClassifier(random_state=42, verbose=0),
#         'base_weighted': CatBoostClassifier(
#             random_state=42,
#             verbose=0,
#             auto_class_weights='Balanced',
#         ),
#     }
# }

# # Label skenario dibuat SATU sumber kebenaran agar konsisten
# # di seluruh skrip (perbaikan ketidaksesuaian #3).
# SC_BASELINE = '1. Pure Baseline'
# SC_WEIGHTED = '2. Baseline + Class Weighting'

# results_ablation = []

# for algo_name, cfg in configs.items():
#     print(f"\nMengeksekusi {algo_name}...")

#     # Skenario 1: Pure Baseline (tanpa class weighting)
#     model_s1 = cfg['base']
#     model_s1.fit(X_train, y_train)
#     res_s1 = evaluate(
#         y_test,
#         model_s1.predict(X_test),
#         model_s1.predict_proba(X_test)[:, 1],
#     )
#     res_s1.update({'Algoritma': algo_name, 'Skenario': SC_BASELINE})
#     results_ablation.append(res_s1)

#     # Skenario 2: Baseline + Class Weighting
#     model_s2 = cfg['base_weighted']
#     model_s2.fit(X_train, y_train)

#     # Prediksi cukup sekali
#     y_pred = model_s2.predict(X_test)
#     y_proba = model_s2.predict_proba(X_test)[:, 1]

#     res_s2 = evaluate(
#         y_test,
#         y_pred,
#         y_proba,
#     )
#     res_s2.update({'Algoritma': algo_name, 'Skenario': SC_WEIGHTED})
#     results_ablation.append(res_s2)

#     # Hanya model Skenario 2 yang disimpan (kandidat champion)
#     weighted_models[algo_name] = model_s2
#     weighted_predictions[algo_name] = {
#         "y_pred": y_pred,
#         "y_proba": y_proba,
#     }


# # --- TABEL HASIL ABLATION STUDY ---
# df_ablation = pd.DataFrame(results_ablation)

# # Reorder columns agar rapi
# cols = ['Algoritma', 'Skenario', 'ROC-AUC', 'Recall (Sens)', 'Specificity',
#         'F1-Score', 'Akurasi', 'PR-AUC']
# df_ablation = df_ablation[cols]

# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)

# print("\nTABEL HASIL ABLATION STUDY (2 SKENARIO)")
# print(df_ablation.round(4).to_string(index=False))

# # --- VISUALISASI ABLATION STUDY ---
# print("\nMembuat Visualisasi Ablation Study...")

# METRIC_COLS = ['ROC-AUC', 'Recall (Sens)', 'Specificity', 'F1-Score',
#                'Akurasi', 'PR-AUC']
# SCENARIOS = [SC_BASELINE, SC_WEIGHTED]


# def render_tabel_png(tbl, judul, fname):
#     """Render DataFrame jadi PNG tabel siap tempel ke laporan."""
#     fig, ax = plt.subplots(figsize=(1.55 * len(tbl.columns) + 2.2, 0.6 * len(tbl) + 1.4))
#     ax.axis('off')
#     mtable = ax.table(
#         cellText=[[f"{v:.4f}" for v in row] for row in tbl.values],
#         rowLabels=tbl.index,
#         colLabels=tbl.columns,
#         cellLoc='center',
#         rowLoc='left',
#         loc='center',
#     )
#     mtable.auto_set_font_size(False)
#     mtable.set_fontsize(10)
#     mtable.scale(1, 1.6)

#     # Header styling
#     for j in range(len(tbl.columns)):
#         cell = mtable[0, j]
#         cell.set_facecolor('#2f4b7c')
#         cell.set_text_props(color='white', weight='bold')

#     for i in range(len(tbl)):
#         mtable[i + 1, -1].set_text_props(weight='bold')

#     # Tandai nilai terbaik tiap kolom (semua metrik: makin tinggi makin baik)
#     for j, col in enumerate(tbl.columns):
#         i_best = int(np.argmax(tbl[col].values))
#         mtable[i_best + 1, j].set_facecolor('#c9e4b4')

#     ax.set_title(judul, fontsize=13, weight='bold', pad=18)
#     plt.savefig(out_path(fname), dpi=200, bbox_inches='tight')
#     plt.close(fig)


# print("\nMembuat tabel metrik per skenario ablasi...")
# tabel_per_skenario = {}

# for sc in SCENARIOS:
#     tbl = (
#         df_ablation[df_ablation['Skenario'] == sc]
#         .set_index('Algoritma')[METRIC_COLS]
#         .round(4)
#     )
#     tabel_per_skenario[sc] = tbl

#     print(f"\n=== TABEL SKENARIO {sc} ===")
#     print(tbl.to_string())

#     slug = sc.split('. ')[1].lower().replace(' + ', '_').replace(' ', '_')
#     render_tabel_png(tbl, f"Skenario {sc}", f"tabel_ablasi_{sc[0]}_{slug}.png")

#     # Ekspor LaTeX siap \input{} ke Bab IV
#     with open(out_path(f"tabel_ablasi_{sc[0]}_{slug}.tex"), "w", encoding="utf-8") as f:
#         f.write(
#             tbl.to_latex(
#                 float_format="%.4f",
#                 caption=f"Metrik evaluasi Skenario {sc}",
#                 label=f"tab:ablasi{sc[0]}",
#                 position="htbp",
#             )
#         )

# print("-> 2 tabel tersimpan (.png, .tex) per skenario")


# # ---------------------------------------------------------
# # PEMILIHAN MODEL TERBAIK
# #  (Flowchart: "Model terbaik dipilih dari skenario 2 (baseline dengan
# #   class weighting) dengan target metrik ROC-AUC")
# # ---------------------------------------------------------
# weighted_df = df_ablation[df_ablation["Skenario"] == SC_WEIGHTED]

# best_algorithm = weighted_df.loc[
#     weighted_df["ROC-AUC"].idxmax(),
#     "Algoritma",
# ]

# print(f"\nModel terbaik (champion): {best_algorithm}")

# best_model = weighted_models[best_algorithm]
# best_prediction = weighted_predictions[best_algorithm]


# # ---------------------------------------------------------
# # FASE 5: AMBANG OPTIMAL VIA YOUDEN'S J
# #  (Flowchart, berurutan:
# #    14) "Hitung Probabilitas pada Data Train menggunakan model terbaik"
# #    15) "Cari nilai ambang (Threshold) optimal menggunakan Youden's J
# #         (J = TPR - FPR) dari hasil Data Train"
# #    16) "Aplikasikan ambang tersebut pada probabilitas Data Test"
# #    17) "Evaluasi ulang metrik kinerja pada Data Test dengan ambang baru")
# #
# #  Threshold DICARI di TRAIN lalu DIPAKAI di TEST -> mencegah leakage.
# # ---------------------------------------------------------
# print("\nFASE 5: MENCARI AMBANG OPTIMAL (YOUDEN'S J)\n" + "-" * 50)


# def find_youden(y_true, y_proba):
#     """Cari threshold yang memaksimalkan J = TPR - FPR."""
#     fpr, tpr, thr = roc_curve(y_true, y_proba)
#     J = tpr - fpr
#     ix = int(np.argmax(J))
#     return {'threshold': float(thr[ix]), 'J': float(J[ix])}


# # (14) Probabilitas pada DATA TRAIN memakai model terbaik
# y_train_proba = best_model.predict_proba(X_train)[:, 1]

# # (15) Threshold optimal dari DATA TRAIN
# info_train = find_youden(y_train, y_train_proba)
# youden_thr = info_train['threshold']

# # (16) Aplikasikan threshold ke probabilitas DATA TEST
# y_test_proba = best_prediction['y_proba']
# y_pred_default = best_prediction['y_pred']                    # ambang 0.5 (default)
# predictions_youden = (y_test_proba >= youden_thr).astype(int)  # ambang Youden

# # Koordinat titik Youden pada kurva ROC DATA TEST (dipakai di FASE 6B)
# tn, fp, fn, tp = confusion_matrix(y_test, predictions_youden).ravel()
# youden_info = {
#     'threshold': youden_thr,
#     'fpr': fp / (fp + tn),
#     'tpr': tp / (tp + fn),
# }

# # (17) Evaluasi ulang pada DATA TEST: default 0.5 vs Youden
# m_default = evaluate(y_test, y_pred_default, y_test_proba)
# m_youden = evaluate(y_test, predictions_youden, y_test_proba)

# df_youden = pd.DataFrame(
#     [
#         {
#             'Ambang': '0.5 (default)',
#             'Threshold': 0.5,
#             'J (Train)': np.nan,
#             **{k: round(v, 4) for k, v in m_default.items()},
#         },
#         {
#             'Ambang': 'Youden',
#             'Threshold': round(youden_thr, 4),
#             'J (Train)': round(info_train['J'], 4),
#             **{k: round(v, 4) for k, v in m_youden.items()},
#         },
#     ]
# ).set_index('Ambang')

# print(f"Model terbaik (champion): {best_algorithm}")
# print(df_youden.to_string())
# print("\nCatatan:")
# print(" - Threshold Youden dihitung dari DATA TRAIN (mencegah leakage).")
# print(" - Metrik akhir dievaluasi pada DATA TEST dengan ambang tersebut.")
# print(" - ROC-AUC & PR-AUC identik di kedua baris (dihitung dari probabilitas,")
# print("   bukan dari label), yang bergeser hanya Recall/Specificity/F1/Akurasi.")


# # ---------------------------------------------------------
# # FASE 6A: CONFUSION MATRIX (pada ambang Youden, Data Test)
# #  (Flowchart: "Visualisasi Evaluasi -> buat Confusion Matrix")
# # ---------------------------------------------------------
# print("\nMEMBUAT CONFUSION MATRIX...\n" + "-" * 50)

# class_labels = ['Normal', 'Hipertensi']

# fig, ax = plt.subplots(figsize=(5, 4.5))

# cm = confusion_matrix(y_test, predictions_youden)

# ConfusionMatrixDisplay(
#     confusion_matrix=cm,
#     display_labels=class_labels,
# ).plot(
#     ax=ax,
#     cmap="Blues",
#     colorbar=False,
# )

# ax.set_title(f"{best_algorithm}\n(thr={youden_thr:.3f})")
# ax.grid(False)

# plt.tight_layout()
# plt.savefig(out_path("confusion_matrix_youden.png"), dpi=150, bbox_inches="tight")
# plt.close()

# print("-> confusion_matrix_youden.png tersimpan")


# # ---------------------------------------------------------
# # FASE 6B: ROC CURVE (beserta titik Youden, Data Test)
# #  (Flowchart: "Visualisasi Evaluasi -> Kurva ROC (beserta titik Youden)")
# # ---------------------------------------------------------
# print("\nMEMBUAT ROC CURVE...\n" + "-" * 50)

# plt.figure(figsize=(8, 6))

# fpr, tpr, _ = roc_curve(y_test, best_prediction["y_proba"])
# auc_val = roc_auc_score(y_test, best_prediction["y_proba"])

# line, = plt.plot(
#     fpr,
#     tpr,
#     linewidth=2,
#     label=f'{best_algorithm} (AUC = {auc_val:.3f})',
# )

# plt.scatter(
#     youden_info["fpr"],
#     youden_info["tpr"],
#     color=line.get_color(),
#     edgecolor="black",
#     zorder=5,
#     s=70,
#     label="Youden Threshold",
# )

# plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")

# plt.xlabel("False Positive Rate (1 - Specificity)")
# plt.ylabel("True Positive Rate (Sensitivity / Recall)")
# plt.title(f"ROC Curve - {best_algorithm}")
# plt.legend(loc="lower right")
# plt.grid(alpha=0.3)

# plt.tight_layout()
# plt.savefig(out_path("roc_curve_youden.png"), dpi=150, bbox_inches="tight")
# plt.close()

# print("-> roc_curve_youden.png tersimpan")


# # ---------------------------------------------------------
# # FASE 7: EKSPOR ARTIFACT MODEL
# #  (Flowchart: "Ekspor Artifact Model -> menyimpan Champion Model,
# #   nilai Threshold Youden, dan urutan fitur (Feature Names) ke file .pkl")
# # ---------------------------------------------------------
# artifact = {
#     "model": best_model,
#     "threshold": youden_thr,
#     "feature_names": list(X_train.columns),  # jaga urutan kolom saat inference
# }

# model_filename = out_path('best_hipertensi_model.pkl')
# joblib.dump(artifact, model_filename)
# print(f"\nModel + threshold + skema fitur disimpan: {model_filename}")
# print(f"   Champion  = {best_algorithm}")
# print(f"   Threshold = {youden_thr:.4f}")

# # --- Cara pakai di backend ---
# # art = joblib.load('best_hipertensi_model.pkl')
# # X_new = df_pasien[art['feature_names']]              # jaga urutan kolom
# # proba = art['model'].predict_proba(X_new)[:, 1]
# # pred  = (proba >= art['threshold']).astype(int)      # pakai ambang Youden, bukan 0.5







"""
============================================================================
SCRIPT DATA PREPARATION — PREDIKSI HIPERTENSI
Urutan (mengikuti flowchart sebagai source of truth):

  0. Pemeriksaan + penghapusan duplikat kombinasi (hhid14, pid14)
  1. Pembersihan kode tersamar (sentinel 8, 9, dan umur > 150) -> NaN
  2. Filter responden usia >= 18 tahun
  3. Koreksi nilai mustahil tekanan darah + buang baris tanpa tekanan darah
  4. Pembentukan label target hipertensi (sistolik>=140 ATAU diastolik>=90)
  5. Tekanan darah TIDAK dijadikan fitur (anti target-leakage) + hitung BMI
  6. Encoding biner & Feature Engineering active_status
  7. Split 80/20 -> lalu seleksi fitur & imputasi HANYA belajar dari TRAIN

CATATAN PENTING (anti-kebocoran / leakage):
  - Split dilakukan SEBELUM seleksi fitur & imputasi.
  - Ambang seleksi fitur (missing 40%) dihitung dari TRAIN saja (perbaikan B3).
  - Imputer di-fit HANYA di train, lalu di-transform ke test.
============================================================================
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

PATH_DATA = "master_dataset_raw_final.csv"


def garis(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


print("--- MEMULAI DATA PREPARATION ---")
df = pd.read_csv(PATH_DATA, low_memory=False)
print(f"Populasi awal: {len(df)} responden, {df.shape[1]} kolom")

# ===========================================================================
# LANGKAH 0 — PEMERIKSAAN + PENGHAPUSAN DUPLIKAT (hhid14, pid14)   [A1]
#  Duplikat WAJIB dibuang di sini, sebelum split, agar individu yang sama
#  tidak bocor ke train sekaligus test.
# ===========================================================================
garis("LANGKAH 0: Pemeriksaan & penghapusan duplikat (hhid14, pid14)")
dup = df[df.duplicated(subset=['hhid14', 'pid14'], keep=False)]
if len(dup) == 0:
    print("  Tidak ditemukan duplikat pada kombinasi (hhid14, pid14).")
else:
    print(
        f"  Ditemukan {len(dup)} baris duplikat "
        f"dari {dup[['hhid14', 'pid14']].drop_duplicates().shape[0]} "
        "kombinasi (hhid14, pid14). Baris duplikat dibuang (keep='first')."
    )
df = df.drop_duplicates(subset=['hhid14', 'pid14'], keep='first').copy()

# ===========================================================================
# LANGKAH 1 — PEMBERSIHAN KODE TERSAMAR (SENTINEL) -> NaN
# ===========================================================================
garis("LANGKAH 1: Pembersihan kode tersamar (sentinel)")

df.loc[df["age"] > 150, "age"] = np.nan   # 998 = 'tidak tahu' -> NaN

# CATATAN (B2): pada titik ini kolom penyakit kronis MASIH bernama 'is_*'
# (rename ke 'has_*' baru terjadi di LANGKAH 6). Maka sentinel dibersihkan
# memakai nama MENTAHnya (is_*), bukan nama pasca-rename. Kalau salah nama,
# penjaga 'if col in df.columns' membuat pembersihan diam-diam TERLEWAT.
feature_cols = [
    'has_tobacco',
    'is_diabetes', 'is_high_cholesterol', 'is_kidney_disease', 'is_stroke',
    'hard_act_last7d', 'moderate_act_last7d', 'walking_last7d',
]
for col in feature_cols:
    if col in df.columns:
        df[col] = df[col].replace({8: np.nan, 9: np.nan})

print("  Kode sentinel (umur>150, 8, 9) pada umur & kondisi kronis -> NaN")

# ===========================================================================
# FILTER — HANYA RESPONDEN USIA >= 18 TAHUN
# ===========================================================================
garis("FILTER: Responden usia >= 18 tahun")
n_sebelum = len(df)
df = df[df["age"] >= 18].copy()
print(f"  Responden usia < 18 tahun yang dihapus : {n_sebelum - len(df)}")

# ===========================================================================
# LANGKAH 2 — KOREKSI NILAI MUSTAHIL TEKANAN DARAH + BUANG BARIS KOSONG
# ===========================================================================
garis("LANGKAH 2: Koreksi nilai mustahil tekanan darah")
# B4: diastolik >= sistolik = tidak masuk akal secara medis.
# Perbandingan dengan NaN otomatis menghasilkan False, jadi tanpa .fillna().
bad_bp = df["bp_diastolic"] >= df["bp_systolic"]
print(f"  Tekanan darah mustahil disisihkan : {int(bad_bp.sum())} baris -> NaN")
df.loc[bad_bp, ["bp_systolic", "bp_diastolic"]] = np.nan

# Buang baris tanpa tekanan darah (label tidak dapat dibentuk tanpa ini).
df = df.dropna(subset=["bp_systolic", "bp_diastolic"]).copy()

# Buang outlier antropometri (batas medis tetap) sebelum hitung BMI.
df.loc[(df["height_cm"] < 100) | (df["height_cm"] > 200), "height_cm"] = np.nan
df.loc[(df["weight_kg"] < 25) | (df["weight_kg"] > 200), "weight_kg"] = np.nan
df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2

# ===========================================================================
# LANGKAH 3 — PEMBENTUKAN LABEL TARGET HIPERTENSI
# ===========================================================================
garis("LANGKAH 3: Pembentukan label target hipertensi")
df["label_hypertension"] = (
    (df["bp_systolic"] >= 140) | (df["bp_diastolic"] >= 90)
).astype(int)
prop = df["label_hypertension"].mean()
print(f"  Proporsi hipertensi : {prop:.1%} (tidak hipertensi {1-prop:.1%})")

# ===========================================================================
# LANGKAH 4 — TEKANAN DARAH TIDAK DIJADIKAN FITUR (ANTI TARGET-LEAKAGE)
#  Label berasal dari bp_systolic/bp_diastolic, maka keduanya SENGAJA tidak
#  dicantumkan di 'fitur_kandidat' (LANGKAH 7). weight/height juga tidak
#  dipakai langsung; hanya turunannya, yaitu 'bmi'.
# ===========================================================================

# ===========================================================================
# LANGKAH 5 & 6 — ENCODING BINER & FEATURE ENGINEERING active_status
# ===========================================================================
garis("LANGKAH 5-6: Encoding biner & fitur active_status")


def to_binary(s, satu=1):
    """Petakan kode IFLS {1 -> 1 (ya), 3 -> 0 (tidak)}; sisanya -> NaN."""
    return s.map({satu: 1, 3: 0})


# Jenis kelamin: 3 = perempuan -> is_female = 1
df["is_female"] = (df["sex"] == 3).astype(float)
df.loc[df["sex"].isna(), "is_female"] = np.nan

biner_map = [
    "has_tobacco", "is_diabetes", "is_high_cholesterol", "is_kidney_disease",
    "is_stroke", "hard_act_last7d", "moderate_act_last7d", "walking_last7d",
]
for col in biner_map:
    if col in df.columns:
        df[col] = to_binary(df[col])

# Samakan penamaan penyakit kronis: is_* -> has_*
df = df.rename(columns={
    "is_diabetes": "has_diabetes",
    "is_high_cholesterol": "has_high_cholesterol",
    "is_kidney_disease": "has_kidney_disease",
    "is_stroke": "has_stroke",
})

# Feature Engineering active_status (mengacu Riskesdas 2013).
print("  Membuat fitur 'active_status' dari hard_act, moderate_act, walking...")
df["active_status"] = np.nan   # tetap NaN bila data aktivitas tidak lengkap

# Kurang aktif (0): tidak melakukan aktivitas berat DAN sedang
#  (status walking harus diketahui: 0 atau 1, bukan NaN)
kondisi_tidak_aktif = (
    (
        (df["hard_act_last7d"] == 0)
        & (df["moderate_act_last7d"] == 0)
        & (df["walking_last7d"] == 1)
    )
    | (
        (df["hard_act_last7d"] == 0)
        & (df["moderate_act_last7d"] == 0)
        & (df["walking_last7d"] == 0)
    )
)
df.loc[kondisi_tidak_aktif, "active_status"] = 0

# Aktif (1): melakukan aktivitas berat ATAU sedang
kondisi_aktif = (df["hard_act_last7d"] == 1) | (df["moderate_act_last7d"] == 1)
df.loc[kondisi_aktif, "active_status"] = 1

# Buang fitur aktivitas mentah agar tidak redundan/multikolinear.
df = df.drop(
    columns=["hard_act_last7d", "moderate_act_last7d", "walking_last7d"],
    errors="ignore",
)

# ===========================================================================
# LANGKAH 7 — SPLIT DULU, LALU SELEKSI FITUR & IMPUTASI (belajar dari TRAIN)
# ===========================================================================
fitur_kandidat = [
    "age", "is_female", "bmi",
    "has_tobacco", "has_diabetes", "has_high_cholesterol",
    "has_kidney_disease", "has_stroke", "active_status",
]
TARGET = "label_hypertension"
df_model = df[fitur_kandidat + [TARGET]].copy()

# --- LANGKAH 7a: SPLIT 80/20 (SEBELUM seleksi & imputasi) ---
garis("LANGKAH 7a: Split data latih-uji (80/20, stratified)")
X_all, y = df_model[fitur_kandidat], df_model[TARGET]
X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train = {len(X_train_all)} baris | Test = {len(X_test_all)} baris")

# --- LANGKAH 7b: SELEKSI FITUR (ambang missing 40%) DIHITUNG DARI TRAIN [B3] ---
#  Kriteria hanya berbasis persentase nilai hilang (unsupervised: tidak
#  menyentuh target). Tetap dihitung dari TRAIN agar konsisten & bebas bocor.
garis("LANGKAH 7b: Seleksi fitur via missing value (dihitung dari TRAIN)")
mcar_train = (X_train_all.isna().mean() * 100).round(2)
print("Persentase nilai hilang per fitur kandidat (TRAIN):")
print(mcar_train.to_string())

THRESHOLD = 40.0
FITUR         = [c for c in fitur_kandidat if mcar_train[c] <= THRESHOLD]
FITUR_DROPPED = [c for c in fitur_kandidat if mcar_train[c] >  THRESHOLD]
print(f"\n  [LOLOS] sisa NaN <= {THRESHOLD}% : {FITUR}")
print(f"  [DROP]  sisa NaN >  {THRESHOLD}% : {FITUR_DROPPED}")

# Terapkan daftar fitur (hasil dari train) ke train & test.
X_train = X_train_all[FITUR].copy()
X_test  = X_test_all[FITUR].copy()
print(f"  Jumlah fitur terpilih: {len(FITUR)}")

# --- LANGKAH 7c: IMPUTASI (imputer di-FIT HANYA di train) ---
garis("LANGKAH 7c: Imputasi (median numerik, modus biner) — fit di TRAIN saja")
# 'if c in FITUR' membuat imputasi tahan bila suatu fitur ter-drop di 7b.
num_cols = [c for c in ["age", "bmi"] if c in FITUR]
cat_cols = [
    c for c in [
        "is_female", "has_tobacco", "has_diabetes", "has_high_cholesterol",
        "has_kidney_disease", "has_stroke", "active_status",
    ] if c in FITUR
]

imp_num = SimpleImputer(strategy="median")
imp_cat = SimpleImputer(strategy="most_frequent")

Xtr = X_train.copy()
Xte = X_test.copy()
Xtr[num_cols] = imp_num.fit_transform(X_train[num_cols])
Xte[num_cols] = imp_num.transform(X_test[num_cols])
Xtr[cat_cols] = imp_cat.fit_transform(X_train[cat_cols])
Xte[cat_cols] = imp_cat.transform(X_test[cat_cols])

# ===========================================================================
# FINALISASI
# ===========================================================================
garis("FINALISASI")
assert list(Xtr.columns) == list(Xte.columns), "Kolom train & test tidak identik!"
print(f"  Total fitur akhir: {Xtr.shape[1]} (train) / {Xte.shape[1]} (test)")

Xtr.assign(label_hypertension=y_train.values).to_csv("train_hipertensi.csv", index=False)
Xte.assign(label_hypertension=y_test.values).to_csv("test_hipertensi.csv", index=False)

print("\nMissing value setelah imputasi:")
print(f"Train : {Xtr.isna().sum().sum()} sel")
print(f"Test  : {Xte.isna().sum().sum()} sel")
print("--- DATA PREPARATION SELESAI. ---")