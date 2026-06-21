import os
import joblib
import pandas as pd
import shap # Pastikan lu udah pip install shap
from backend.services.ml_service import predict_risk
from backend.services.llm_service import generate_clinical_narrative
from backend.services.xai_service import get_top_risk_factors
# ==========================================
# 1. LOAD MODEL & SCALER (Saat Server Menyala)
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved_models")

SCALER_PATH = os.path.join(MODELS_DIR, "standard_scaler.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_model.pkl") # Ganti logistic_regression.pkl kalau lu pakai itu

try:
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    print(" Model & Scaler berhasil dimuat!")
except Exception as e:
    print(f" Gagal memuat model: {e}")

# Inisialisasi XAI (SHAP) on-the-fly
# Catatan: TreeExplainer untuk XGBoost. Jika pakai Logistic Regression, gunakan shap.LinearExplainer
try:
    explainer = shap.TreeExplainer(model) 
except:
    explainer = None

# ==========================================
# 2. FUNGSI ORKESTRATOR (PIPA INTEGRASI)
# ==========================================
def run_full_clinical_analysis(patient_data_dict: dict):
    # # 1. Ubah input dari Frontend menjadi DataFrame
    # df_raw = pd.DataFrame([patient_data_dict])
    
    # # --- TAHAP A: MELEWATI SCALER (15 KOLOM) ---
    # if hasattr(scaler, 'feature_names_in_'):
    #     scaler_cols = scaler.feature_names_in_
    # else:
    #     scaler_cols = df_raw.columns
        
    # df_for_scaler = df_raw.reindex(columns=scaler_cols, fill_value=0)
    # scaled_values = scaler.transform(df_for_scaler)
    # df_scaled = pd.DataFrame(scaled_values, columns=scaler_cols)

    # # --- TAHAP B: MENYESUAIKAN DENGAN XGBOOST (16 KOLOM) ---
    # if hasattr(model, 'feature_names_in_'):
    #     model_cols = list(model.feature_names_in_)
    # elif hasattr(model, 'get_booster'):
    #     model_cols = model.get_booster().feature_names
    # else:
    #     model_cols = list(df_scaled.columns)
        
    # # Suntikkan kolom yang diminta XGBoost tapi tidak ada di Scaler (contoh: ak05)
    # for col in model_cols:
    #     if col not in df_scaled.columns:
    #         # Ambil langsung dari data raw/dict, kasih 0 kalau kosong
    #         df_scaled[col] = patient_data_dict.get(col, 0)
            
    # # Urutkan final sesuai kemauan mutlak XGBoost
    # df_final = df_scaled[model_cols]

    # # 3. PREDIKSI (MACHINE LEARNING)
    # risk_score = float(model.predict_proba(df_final)[0][1])
    # risk_status = "High Risk" if risk_score > 0.5 else "Low Risk"




    # 1. PREDIKSI (MACHINE LEARNING) - Didelegasikan ke ML Service
    try:
        risk_score = predict_risk(patient_data_dict)
        risk_status = "High Risk" if risk_score > 0.5 else "Low Risk"
    except Exception as e:
        raise ValueError(f"ML Prediction Error: {e}")

    # # 4. EXPLAINABLE AI (SHAP)
    # top_features = {}
    # if explainer:
    #     shap_values = explainer.shap_values(df_final) # Pastikan SHAP juga pakai df_final
    #     feature_importance = dict(zip(df_final.columns, shap_values[0]))
        
    #     top_features = dict(sorted(
    #         {k: float(v) for k, v in feature_importance.items() if v > 0}.items(),
    #         key=lambda item: item[1], 
    #         reverse=True
    #     )[:3])
    
    # 4. EXPLAINABLE AI (SHAP) - DIDELEGASIKAN KE XAI SERVICE
    try:
        # Langsung panggil service yang sudah Anda buat dengan benar
        top_features = get_top_risk_factors(patient_data_dict, top_n=8)
    except Exception as e:
        print(f"XAI Error: {e}")
        top_features = {}

    # 5. GENERASI NARASI MEDIS (RAG + LLM)
    clinical_narrative = generate_clinical_narrative(patient_data_dict, top_features, risk_score, risk_status)

    # 6. BUNGKUS JSON
    return {
        "status": "success",
        "data": {
            "prediction": {
                "risk_score": round(risk_score, 4),
                "risk_status": risk_status
            },
            "xai_analysis": top_features,
            "clinical_narrative": clinical_narrative
        }
    }

# ==========================================
# BLOK TESTING LOKAL
# ==========================================
if __name__ == "__main__":
    # Ini simulasi JSON payload yang akan dikirim dari React/Frontend lu nanti
    mock_patient_payload = {
        "age": 45,
        "is_female": 0,          # 0 = Laki-laki, 1 = Perempuan
        "bmi": 28.5,
        "waist_cm": 95,
        "is_smoker": 1,
        "freq_instant_noodle": 5,
        "ak02": 1,               # Angka dummy untuk testing
        "ak05": 0,
        "ak07": 0,               # Angka dummy untuk testing
        "has_diabetes": 0,
        "genetic_risk_score": 1.2, # Angka dummy untuk testing
        "ps_A": 0,               # Angka dummy untuk testing
        "ps_B": 1,               # Angka dummy untuk testing
        "ps_C": 0,               # Angka dummy untuk testing
        "ps_E": 0,               # Angka dummy untuk testing
        "ps_F": 1                # Angka dummy untuk testing
    }
    
    print("\n--- MENJALANKAN INTEGRASI FULL BACKEND ---")
    final_response = run_full_clinical_analysis(mock_patient_payload)
    
    import json
    print("\n--- HASIL FINAL JSON RESPONSE ---")
    print(json.dumps(final_response, indent=4))