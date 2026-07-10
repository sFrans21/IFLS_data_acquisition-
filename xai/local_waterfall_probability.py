"""
local_waterfall_probability.py
==============================
Membuat ulang analisis LOKAL (waterfall) dalam satuan PROBABILITAS,
sesuai keputusan penyajian hybrid (global log-odds, lokal probabilitas).

Jalankan di mesin Anda (environment yang punya shap==0.51.0 + xgboost),
di folder yang sama dengan model & dataset. Hasil:
  - shap_output/local_waterfall_prob_idx0.png
  - cetak base value, f(x), dan kontribusi per fitur dalam probabilitas.

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
DATASET_PATH = "dataset_hipertensi_imputed.csv"
TARGET_COL   = "label_hypertension"
OUTPUT_DIR   = "shap_output"
LOCAL_INDEX  = 0          # samakan dengan analisis lokal di skrip utama
BACKGROUND_N = 200
RANDOM_STATE = 42

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
X = df.drop(columns=[TARGET_COL])
try:
    feats = list(model.feature_names_in_)
except Exception:
    feats = list(X.columns)
X = X[feats]

bg = shap.sample(X, BACKGROUND_N, random_state=RANDOM_STATE)
explainer = shap.TreeExplainer(
    model, data=bg,
    feature_perturbation="interventional",
    model_output="probability",
)
sv = explainer(X.iloc[[LOCAL_INDEX]])

shap.plots.waterfall(sv[0], max_display=len(feats), show=False)
plt.savefig(os.path.join(OUTPUT_DIR, "local_waterfall_prob_idx0.png"),
            bbox_inches="tight", dpi=150)
plt.close()

base = float(np.array(sv.base_values).reshape(-1)[0])
fx = base + float(np.sum(sv.values[0]))
print(f"base value (prob): {base:.4f}")
print(f"f(x) (prob)      : {fx:.4f}")
print("kontribusi per fitur (prob):")
for f_, v in zip(feats, sv.values[0]):
    print(f"  {f_:<22} {float(v):+.4f}")