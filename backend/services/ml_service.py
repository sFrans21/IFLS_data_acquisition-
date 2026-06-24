# import os
# import joblib
# import pandas as pd

# # 1. Tentukan Path ke file .pkl
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models", "logistic_regression_model.pkl")
# MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models", "xgboost_model.pkl")

# SCALER_PATH = os.path.join(BASE_DIR, "models", "saved_models", "standard_scaler.pkl")

# # 2. Load Model & Scaler ke Memory (Global Variables)
# try:
#     model = joblib.load(MODEL_PATH)
#     scaler = joblib.load(SCALER_PATH)
#     print("[ML Service] Model dan Scaler berhasil dimuat ke memory!")
# except Exception as e:
#     print(f" [ML Service] GAGAL memuat model: {e}")
#     model = None
#     scaler = None

# def predict_risk(input_dict: dict) -> float:
#     """
#     Menerima dictionary data pasien dari FastAPI, mengubahnya jadi format yang 
#     dimengerti model, dan mengembalikan probabilitas risiko hipertensi.
#     """
#     if model is None or scaler is None:
#         raise ValueError("Internal Server Error: Model AI belum siap.")

#     # 1. Ubah input dictionary menjadi Pandas DataFrame (1 baris)
#     df_input = pd.DataFrame([input_dict])
    
#     # --- MULAI DARI SINI ADALAH KODE FIX-NYA ---
    
#     # 2. Ambil "memori" urutan dan nama kolom yang dipelajari model saat training
#     training_features = scaler.feature_names_in_
    
#     # 3. Filter dan Urutkan df_input agar PERSIS dengan training_features
#     # Jika ada kolom dari Flutter yang tidak dikenal model (seperti ak05 yang terbuang), abaikan.
#     # Jika ada kolom yang ditunggu model tapi Flutter tidak kirim, isi dengan 0.
#     for col in training_features:
#         if col not in df_input.columns:
#             df_input[col] = 0.0 # Nilai default aman
            
#     # Potong dan urutkan dataframe (Ini bikin sistem lu kebal/bulletproof!)
#     df_input = df_input[training_features] 
    
#     # --- SELESAI KODE FIX ---

#     # 4. Lakukan normalisasi data (Scaling)
#     scaled_input = scaler.transform(df_input)
    
#     # 5. Prediksi probabilitas
#     probability = model.predict_proba(scaled_input)[0][1]
    
#     return float(probability)



import os
import joblib
import json
import pandas as pd

# 1. Tentukan Path (SAMA seperti di xai_service.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models", "xgboost_model.pkl")
IMPUTE_PATH = os.path.join(BASE_DIR, "models", "saved_models", "imputation_values.json")

# 2. Load Model & nilai imputasi (TANPA scaler)
try:
    model = joblib.load(MODEL_PATH)
    with open(IMPUTE_PATH, "r") as f:
        imputation_values = json.load(f)
    print("[ML Service] Model XGBoost & imputasi berhasil dimuat!")
except Exception as e:
    print(f"[ML Service] GAGAL memuat model: {e}")
    model = None
    imputation_values = {}

def predict_risk(input_dict: dict) -> float:
    if model is None:
        raise ValueError("Internal Server Error: Model AI belum siap.")

    # A. Ubah input jadi DataFrame
    df_input = pd.DataFrame([input_dict])

    # B. Ambil nama kolom dari MODEL (sama seperti xai_service.py)
    training_features = model.feature_names_in_

    # C. Isi kolom kosong dengan MEDIAN (bukan 0.0) — sama seperti xai_service.py
    for col in training_features:
        if col not in df_input.columns:
            df_input[col] = imputation_values.get(col, 0.0)

    # D. Urutkan kolom persis seperti saat training
    df_input = df_input[training_features]

    # E. Prediksi (pakai .to_numpy(), TANPA scaler — sama seperti xai_service.py)
    probability = model.predict_proba(df_input.to_numpy())[0][1]
    return float(probability)