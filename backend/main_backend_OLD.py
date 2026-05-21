# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from services.ml_service import predict_risk
# import pandas as pd
# import numpy as np
# import joblib
# import os

# # Inisialisasi Aplikasi FastAPI
# app = FastAPI(
#     title="HypertensAI Backend",
#     description="API untuk Prediksi Risiko Hipertensi, XAI, dan LLM Narrative",
#     version="1.0.0"
# )

# # ==========================================
# # 1. SCHEMAS (KONTRAK API)
# # ==========================================

# # Request Payload (Dari Flutter)
# class PredictionRequest(BaseModel):
#     age: float
#     systolic_bp: float
#     diastolic_bp: float
#     height_cm: float
#     weight_kg: float
#     bmi: float
#     is_smoker: int
#     has_diabetes: int
#     is_female: int
#     freq_instant_noodle: int
#     ak02: int
#     ak05: int
#     ak07: int
#     ps_A: int
#     ps_B: int
#     ps_C: int
#     ps_E: int
#     ps_F: int
#     genetic_risk_score: int

# # Response Payload (Ke Flutter)
# class ShapValues(BaseModel):
#     # Dictionary fleksibel untuk menampung top features dari SHAP
#     pass 

# class PredictionResponse(BaseModel):
#     prediction_score: float
#     risk_level: str
#     shap_values: dict
#     narrative: str


# # Nanti kita load beneran pakai joblib.load() saat server startup
# # model = joblib.load(MODEL_PATH)
# # scaler = joblib.load(SCALER_PATH)

# # ==========================================
# # 3. ENDPOINT UTAMA
# # ==========================================

# @app.post("/predict", response_model=PredictionResponse)
# async def predict_hypertension(payload: PredictionRequest):
#     try:
#         # --- A. PREPROCESSING (Sesuai Syarat Tim Frontend) ---
        
#         # 1. Konversi Pydantic Object ke Dictionary lalu ke Pandas DataFrame
#         input_dict = payload.model_dump()
        
#         # Simpan nilai Tensi untuk keperluan Prompt LLM nanti sebelum di-drop
#         sys_bp = input_dict['systolic_bp']
#         dia_bp = input_dict['diastolic_bp']
        
#         # 2. DROP fitur yang menyebabkan Data Leakage atau tidak dipakai model AI
#         columns_to_drop = ['systolic_bp', 'diastolic_bp', 'height_cm', 'weight_kg']
#         for col in columns_to_drop:
#             input_dict.pop(col, None)
            
#         # 3. INJECT nilai dummy untuk waist_cm (karena alasan UX Flutter)
#         # Asumsi: Kita pakai median nasional atau rata-rata berdasarkan gender (misal 90.0)
#         # Di tahap optimasi nanti, ini bisa kita ganti dengan rumus regresi sederhana
#         input_dict['waist_cm'] = 90.0 
        
#         # Ubah jadi DataFrame satu baris siap masuk Scaler
#         df_input = pd.DataFrame([input_dict])
        
#     # --- B. PREDIKSI MACHINE LEARNING (REAL) ---
#         # Panggil fungsi dari ml_service.py
#         real_score = predict_risk(input_dict)

#         # Tentukan label berdasarkan threshold (sementara kita pakai 0.5)
#         risk_label = "Risiko Tinggi" if real_score > 0.5 else "Risiko Rendah"

#         # --- C. EXPLAINABLE AI / SHAP (Placeholder) ---
#         # Nanti kita panggil fungsi dari xai_service.py
#         dummy_shap = {
#             "age": 0.45,
#             "bmi": 0.20,
#             "is_smoker": 0.15
#         }
        
#         # --- D. LLM NARRATIVE / RAG (Placeholder) ---
#         # Nanti kita panggil fungsi dari llm_service.py (Objective 2)
#         # Kita akan nge-passing sys_bp dan dia_bp ke sini
#         dummy_narrative = f"Berdasarkan analisis, usia dan BMI Anda memicu risiko hipertensi. Tekanan darah Anda yang diinputkan ({sys_bp}/{dia_bp}) juga menjadi catatan penting. Kurangi garam sesuai pedoman JNC-7."
        
#         # --- E. KEMBALIKAN RESPONSE KE FLUTTER ---
#         return PredictionResponse(
#             prediction_score=real_score,
#             risk_level=risk_label,
#             shap_values=dummy_shap,
#             narrative=dummy_narrative
#         )

#     except Exception as e:
#         # Error Handling yang baik agar Flutter tidak crash
#         raise HTTPException(status_code=500, detail=str(e))



# @app.get("/")
# async def root():
#     return {
#         "message": "Welkam ke HyppertensAI backend API!",
#         "status": "Running",
#         "docs": "Silahkan buka http://127/0.0.1:8000/docs untk melihat dokumentasi API"
#     }

# # Entry point buat testing jalankan file ini langsung
# if __name__ == "__main__":
#     import uvicorn
#     # Jalankan server di localhost port 8000
#     uvicorn.run("main_backend:app", host="127.0.0.1", port=8000, reload=True)



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from services.ml_service import predict_risk
from services.xai_service import get_top_risk_factors # <-- [FIX 1] Import XAI Service
import pandas as pd
import numpy as np
import os

# Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="HypertensAI Backend",
    description="API untuk Prediksi Risiko Hipertensi, XAI, dan LLM Narrative",
    version="1.0.0"
)

# ==========================================
# 1. SCHEMAS (KONTRAK API)
# ==========================================

class PredictionRequest(BaseModel):
    age: float
    systolic_bp: float
    diastolic_bp: float
    height_cm: float
    weight_kg: float
    bmi: float
    is_smoker: int
    has_diabetes: int
    is_female: int
    freq_instant_noodle: int
    ak02: int
    ak05: int
    ak07: int
    ps_A: int
    ps_B: int
    ps_C: int
    ps_E: int
    ps_F: int
    genetic_risk_score: int

class ShapValues(BaseModel):
    pass 

class PredictionResponse(BaseModel):
    prediction_score: float
    risk_level: str
    shap_values: dict
    narrative: str

# ==========================================
# 2. ENDPOINT UTAMA
# ==========================================

@app.post("/predict", response_model=PredictionResponse)
async def predict_hypertension(payload: PredictionRequest):
    try:
        # --- A. PREPROCESSING ---
        input_dict = payload.model_dump()
        
        # Simpan nilai Tensi untuk keperluan Prompt LLM nanti sebelum di-drop
        sys_bp = input_dict['systolic_bp']
        dia_bp = input_dict['diastolic_bp']
        
        # DROP fitur yang menyebabkan Data Leakage atau tidak dipakai model AI
        columns_to_drop = ['systolic_bp', 'diastolic_bp', 'height_cm', 'weight_kg']
        for col in columns_to_drop:
            input_dict.pop(col, None)
            
        # [FIX 2] Baris inject dummy waist_cm dihapus! Biarkan Service Layer yang handle via JSON Imputasi
        
        # --- B. PREDIKSI MACHINE LEARNING (REAL) ---
        real_score = predict_risk(input_dict)

        # [FIX 3] Gunakan threshold optimal XGBoost lu
        risk_label = "Risiko Tinggi" if real_score >= 0.2806 else "Risiko Rendah"

        # --- C. EXPLAINABLE AI / SHAP (REAL) ---
        # [FIX 4] Panggil XAI Service untuk dapetin nilai SHAP asli
        real_shap_values = get_top_risk_factors(input_dict, top_n=3)
        
        # --- D. LLM NARRATIVE / RAG (Placeholder) ---
        dummy_narrative = f"Berdasarkan analisis, faktor-faktor di atas memicu risiko hipertensi Anda. Tekanan darah Anda yang diinputkan ({sys_bp}/{dia_bp}) juga menjadi catatan penting. Kurangi garam sesuai pedoman JNC-7."
        
        # --- E. KEMBALIKAN RESPONSE KE FLUTTER ---
        return PredictionResponse(
            prediction_score=real_score,
            risk_level=risk_label,
            shap_values=real_shap_values, # <-- [FIX 5] Oper variabel aslinya, bukan dummy
            narrative=dummy_narrative
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Welcome ke HypertensAI backend API!",
        "status": "Running",
        "docs": "Silahkan buka http://127.0.0.1:8000/docs untuk melihat dokumentasi API"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_backend:app", host="127.0.0.1", port=8000, reload=True)