# """
# local_waterfall_probability.py
# ==============================
# Membuat ulang analisis LOKAL (waterfall) dalam satuan PROBABILITAS,
# sesuai keputusan penyajian hybrid (global log-odds, lokal probabilitas).

# Jalankan di mesin Anda (environment yang punya shap==0.51.0 + xgboost),
# di folder yang sama dengan model & dataset. Hasil:
#   - shap_output/local_waterfall_prob_idx0.png
#   - cetak base value, f(x), dan kontribusi per fitur dalam probabilitas.

# Catatan: di ruang probabilitas, TreeExplainer butuh background dataset dan
# mode 'interventional'. base value = rata-rata keluaran model atas background
# (mendekati 0,5 karena penyeimbangan kelas), bukan prevalensi asli.
# """

# import os
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import shap

# MODEL_PATH   = "xgboost_hipertensi_model.pkl"
# DATASET_PATH = "dataset_hipertensi_imputed.csv"
# TARGET_COL   = "label_hypertension"
# OUTPUT_DIR   = "shap_output"
# LOCAL_INDEX  = 0          # samakan dengan analisis lokal di skrip utama
# BACKGROUND_N = 200
# RANDOM_STATE = 42

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ---------- 1. Muat model & data ----------
# with open(MODEL_PATH, "rb") as f:
#     loaded_data = pickle.load(f)
        
# # Cek apakah yang dimuat adalah dictionary
# if isinstance(loaded_data, dict):
#     print(f"Isi dictionary: {loaded_data.keys()}")
#     # GANTI 'model' dengan key yang tepat berdasarkan output print di atas.
#     # Biasanya key-nya bernama 'model', 'xgb_model', atau 'classifier'
#     model = loaded_data.get('model', loaded_data) 
# else:
#     model = loaded_data

# df = pd.read_csv(DATASET_PATH)
# X = df.drop(columns=[TARGET_COL])
# try:
#     feats = list(model.feature_names_in_)
# except Exception:
#     feats = list(X.columns)
# X = X[feats]

# bg = shap.sample(X, BACKGROUND_N, random_state=RANDOM_STATE)
# explainer = shap.TreeExplainer(
#     model, data=bg,
#     feature_perturbation="interventional",
#     model_output="probability",
# )
# sv = explainer(X.iloc[[LOCAL_INDEX]])

# shap.plots.waterfall(sv[0], max_display=len(feats), show=False)
# plt.savefig(os.path.join(OUTPUT_DIR, "local_waterfall_prob_idx0.png"),
#             bbox_inches="tight", dpi=150)
# plt.close()

# base = float(np.array(sv.base_values).reshape(-1)[0])
# fx = base + float(np.sum(sv.values[0]))
# print(f"base value (prob): {base:.4f}")
# print(f"f(x) (prob)      : {fx:.4f}")
# print("kontribusi per fitur (prob):")
# for f_, v in zip(feats, sv.values[0]):
#     print(f"  {f_:<22} {float(v):+.4f}")



"""
local_waterfall_probability.py
==============================
Membuat ulang analisis LOKAL (waterfall) dalam satuan PROBABILITAS,
sesuai keputusan penyajian hybrid (global log-odds, lokal probabilitas).

Jalankan di mesin Anda (environment yang punya shap==0.51.0 + xgboost),
di folder yang sama dengan model & dataset. Hasil:
  - shap_output/local_waterfall_prob_idx0.png
  - cetak base value, f(x), dan kontribusi per fitur dalam probabilitas.

PEMISAHAN PERAN DATA (harus identik dengan generate_shap_analysis.py):
  - Individu yang dijelaskan  -> test_hipertensi.csv  (EXPLAIN_ON)
  - Background dataset        -> train_hipertensi.csv

Catatan: di ruang probabilitas, TreeExplainer butuh background dataset dan
mode 'interventional'. base value = rata-rata keluaran model atas background
(mendekati 0,5 karena penyeimbangan kelas), bukan prevalensi asli.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

MODEL_PATH   = "xgboost_hipertensi_model.pkl"
TRAIN_PATH   = "train_hipertensi.csv"     # sumber background
TEST_PATH    = "test_hipertensi.csv"      # sumber individu yang dijelaskan
TARGET_COL   = "label_hypertension"
OUTPUT_DIR   = "shap_output"

EXPLAIN_ON   = "test"     # WAJIB sama dengan generate_shap_analysis.py
LOCAL_INDEX  = 0          # WAJIB sama dengan generate_shap_analysis.py
BACKGROUND_N = 200
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 1. Muat model & data ----------
with open(MODEL_PATH, "rb") as f:
    loaded_data = pickle.load(f)

# Artifact FASE 7 train_model.py: {'model', 'threshold', 'feature_names'}
if isinstance(loaded_data, dict):
    print(f"Artifact berupa dict dengan key: {list(loaded_data.keys())}")
    model = loaded_data.get("model", loaded_data)
    artifact_features = loaded_data.get("feature_names")
    threshold = loaded_data.get("threshold")
else:
    model = loaded_data
    artifact_features = None
    threshold = None

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

X_train_all = train_df.drop(columns=[TARGET_COL])
X_test_all  = test_df.drop(columns=[TARGET_COL])

# Urutan fitur: artifact -> atribut model -> kolom CSV.
# Urutan yang salah menyebabkan salah interpretasi TANPA error.
if artifact_features:
    feats = list(artifact_features)
else:
    try:
        feats = list(model.feature_names_in_)
    except Exception:
        feats = list(X_train_all.columns)

X_train = X_train_all[feats]
X_test  = X_test_all[feats]
X_explain = X_test if EXPLAIN_ON == "test" else X_train

# ---------- 2. Explainer di ruang probabilitas ----------
# Background dari TRAIN: distribusi rujukan tidak boleh berasal dari data uji.
bg = shap.sample(X_train, BACKGROUND_N, random_state=RANDOM_STATE)
explainer = shap.TreeExplainer(
    model, data=bg,
    feature_perturbation="interventional",
    model_output="probability",
)
sv = explainer(X_explain.iloc[[LOCAL_INDEX]])

shap.plots.waterfall(sv[0], max_display=len(feats), show=False)
plt.savefig(os.path.join(OUTPUT_DIR, f"local_waterfall_prob_idx{LOCAL_INDEX}.png"),
            bbox_inches="tight", dpi=150)
plt.close()

base = float(np.array(sv.base_values).reshape(-1)[0])
fx = base + float(np.sum(sv.values[0]))

print(f"Explanation set  : {EXPLAIN_ON} (indeks {LOCAL_INDEX})")
print(f"Background       : train, n={BACKGROUND_N}")
print(f"base value (prob): {base:.4f}")
print(f"f(x) (prob)      : {fx:.4f}")

# Verifikasi silang: f(x) rekonstruksi harus mendekati predict_proba asli.
# Selisih besar = tanda urutan fitur atau background salah.
p_asli = float(model.predict_proba(X_explain.iloc[[LOCAL_INDEX]])[:, 1][0])
print(f"predict_proba    : {p_asli:.4f}  (selisih {abs(p_asli - fx):.6f})")

if threshold is not None:
    print(f"threshold Youden : {threshold:.4f} -> prediksi: "
          f"{'Hipertensi' if p_asli >= threshold else 'Normal'}")

print("kontribusi per fitur (prob):")
for f_, v in zip(feats, sv.values[0]):
    print(f"  {f_:<22} {float(v):+.4f}")