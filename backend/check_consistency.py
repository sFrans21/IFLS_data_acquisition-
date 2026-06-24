import numpy as np
import pandas as pd
from backend.services.ml_service import model as ml_model, predict_risk
from backend.services.xai_service import model as xai_model, explainer

base = {
    "age": 40, "is_female": 0, "bmi": 24, "waist_cm": 85, "is_smoker": 0,
    "freq_instant_noodle": 2, "ak02": 2, "ak05": 3, "ak07": 30,
    "has_diabetes": 0, "genetic_risk_score": 0,
    "ps_A": 1, "ps_B": 1, "ps_C": 1, "ps_E": 1, "ps_F": 1,
}

# 1. Apakah ml_service & xai_service memuat OBJEK model yang sama isinya?
print("Jumlah fitur ml_model :", len(ml_model.feature_names_in_))
print("Jumlah fitur xai_model:", len(xai_model.feature_names_in_))
print("Urutan fitur sama?    :", list(ml_model.feature_names_in_) == list(xai_model.feature_names_in_))

# 2. Hubungan SHAP <-> prediksi untuk 1 kasus
feats = list(xai_model.feature_names_in_)
df = pd.DataFrame([base])[feats]

risk = predict_risk(base)
margin = float(xai_model.predict(df.to_numpy(), output_margin=True)[0])  # skor mentah (log-odds)

sv = explainer.shap_values(df.to_numpy())
arr = sv[0] if not isinstance(sv, list) else sv[1][0]
base_val = explainer.expected_value
if isinstance(base_val, (list, np.ndarray)):
    base_val = float(np.ravel(base_val)[-1])

print("\nrisk (predict_risk, probabilitas):", round(risk, 4))
print("margin model (log-odds)          :", round(margin, 4))
print("base_value SHAP                  :", round(float(base_val), 4))
print("base_value + sum(SHAP)           :", round(float(base_val) + float(arr.sum()), 4))
print("  -> dua baris terakhir harus ~sama jika SHAP & model konsisten")