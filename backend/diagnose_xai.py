import pandas as pd
from backend.services.xai_service import model, explainer, imputation_values
from backend.services.ml_service import predict_risk

base = {
    "age": 40, "is_female": 0, "bmi": 24, "waist_cm": 85, "is_smoker": 0,
    "freq_instant_noodle": 2, "ak02": 2, "ak05": 3, "ak07": 30,
    "has_diabetes": 0, "genetic_risk_score": 0,
    "ps_A": 1, "ps_B": 1, "ps_C": 1, "ps_E": 1, "ps_F": 1,
}

def signed_shap(d):
    feats = model.feature_names_in_
    df = pd.DataFrame([d])
    for c in feats:
        if c not in df.columns:
            df[c] = imputation_values.get(c, 0.0)
    df = df[feats]
    sv = explainer.shap_values(df.to_numpy())
    arr = sv[1][0] if isinstance(sv, list) else sv[0]
    return dict(zip(feats, [float(x) for x in arr]))

print("model.classes_ =", getattr(model, "classes_", "?"))

for label, ch in [("NON-PEROKOK", {"is_smoker": 0}), ("PEROKOK", {"is_smoker": 1})]:
    d = {**base, **ch}
    print(f"[{label}] risk={predict_risk(d):.3f}  SHAP is_smoker={signed_shap(d)['is_smoker']:+.4f}")

for label, ch in [("TANPA RIWAYAT", {"genetic_risk_score": 0}), ("ADA RIWAYAT", {"genetic_risk_score": 2})]:
    d = {**base, **ch}
    print(f"[{label}] risk={predict_risk(d):.3f}  SHAP genetic={signed_shap(d)['genetic_risk_score']:+.4f}")