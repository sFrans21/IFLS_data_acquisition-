# import os
# import joblib
# import pandas as pd
# import numpy as np
# import shap

# # 1. Tentukan Path ke file .pkl
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # PASTIKAN INI MENGARAH KE MODEL XGBOOST LU, BUKAN LOGREG
# MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models", "xgboost_hipertensi_model.pkl") 
# SCALER_PATH = os.path.join(BASE_DIR, "models", "saved_models", "standard_scaler.pkl")

# # 2. Load Model & Scaler ke Memory
# try:
#     model = joblib.load(MODEL_PATH)
#     scaler = joblib.load(SCALER_PATH)
    
#     # Inisialisasi SHAP Explainer khusus untuk Tree-based models (XGBoost, Random Forest, dll)
#     explainer = shap.TreeExplainer(model)
    
#     print(" [XAI Service] SHAP TreeExplainer (XGBoost) berhasil dimuat!")
# except Exception as e:
#     print(f" [XAI Service] GAGAL memuat SHAP: {e}")
#     model = None
#     scaler = None
#     explainer = None

# # 3. Fungsi Utama XAI
# def get_top_risk_factors(input_dict: dict, top_n: int = 3) -> dict:
#     """
#     Menghitung SHAP values untuk 1 pasien dan mengembalikan N fitur
#     yang paling berkontribusi terhadap risiko hipertensinya.
#     """
#     if explainer is None or scaler is None:
#         raise ValueError("Internal Server Error: XAI Service belum siap.")

#     # A. Preprocessing input
#     df_input = pd.DataFrame([input_dict])
#     training_features = scaler.feature_names_in_
    
#     for col in training_features:
#         if col not in df_input.columns:
#             df_input[col] = 0.0
            
#     df_input = df_input[training_features]
#     scaled_input = scaler.transform(df_input)

#     # B. Kalkulasi SHAP Values
#     shap_values = explainer.shap_values(scaled_input)
    
#     # C. Ekstraksi dan Pengurutan Fitur
#     # Perhatikan: Output shap_values dari TreeExplainer XGBoost biasanya sedikit berbeda formatnya
#     # Jika shap_values adalah list (klasifikasi multi-class), ambil index [1] untuk kelas positif
#     if isinstance(shap_values, list):
#         feature_impacts = np.abs(shap_values[1][0])
#     else:
#         feature_impacts = np.abs(shap_values[0])
    
#     # Gabungkan nama fitur dengan nilai SHAP-nya
#     impact_dict = {feature: float(impact) for feature, impact in zip(training_features, feature_impacts)}
    
#     # Urutkan dari dampak paling besar
#     sorted_impacts = dict(sorted(impact_dict.items(), key=lambda item: item[1], reverse=True))
#     top_features = dict(list(sorted_impacts.items())[:top_n])
    
#     return top_features


import os
import joblib
import pandas as pd
import numpy as np
import shap
import json

# 1. Tentukan Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models", "xgboost_hipertensi_model.pkl")
IMPUTE_PATH = os.path.join(BASE_DIR, "models", "saved_models", "imputation_values.json")

# 2. Load Model & JSON Imputasi 
try:
    model = joblib.load(MODEL_PATH)
    with open(IMPUTE_PATH, 'r') as f:
        imputation_values = json.load(f)
        
    explainer = shap.TreeExplainer(model)
    print(" [XAI Service] SHAP TreeExplainer berhasil dimuat!")
except Exception as e:
    print(f"[XAI Service] GAGAL memuat SHAP: {e}")
    model = None
    explainer = None
    imputation_values = {}

# 3. Fungsi Utama XAI
def get_top_risk_factors(input_dict: dict, top_n: int = 3) -> dict:
    if explainer is None:
        raise ValueError("Internal Server Error: XAI Service belum siap.")

    # A. Preprocessing input
    df_input = pd.DataFrame([input_dict])
    training_features = model.feature_names_in_ # Ambil nama kolom dari model langsung
    
    # B. Imputasi Pintar (Menggunakan Median dari JSON, bukan 0.0)
    for col in training_features:
        if col not in df_input.columns:
            df_input[col] = imputation_values.get(col, 0.0)
            
    # Pastikan urutan kolom sesuai 100% dengan saat training
    df_input = df_input[training_features]
    
    # C. Kalkulasi SHAP Values
    # [FIX UTAMA]: Gunakan .to_numpy() agar C++ XGBoost tidak error membaca dimensi DataFrame
    shap_values = explainer.shap_values(df_input.to_numpy())
    
    # D. Ekstraksi dan Pengurutan Fitur
    # Ambil nilai SHAP BERTANDA (jangan di-abs):
    # positif = menaikkan risiko, negatif = menurunkan risiko.
    if isinstance(shap_values, list):
        feature_impacts = shap_values[1][0]
    else:
        feature_impacts = shap_values[0]
    
    impact_dict = {feature: float(impact) for feature, impact in zip(training_features, feature_impacts)}
    
    # Urutkan berdasarkan MAGNITUDO (nilai absolut), tetapi simpan nilai bertandanya.
    sorted_impacts = dict(sorted(impact_dict.items(), key=lambda item: abs(item[1]), reverse=True))
    top_features = dict(list(sorted_impacts.items())[:top_n])
    
    return top_features