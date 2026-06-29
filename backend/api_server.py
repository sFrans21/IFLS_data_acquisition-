from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# Import fungsi sakti lu yang barusan kelar
from backend.services.analysis_service import run_full_clinical_analysis

# Inisialisasi Aplikasi API
app = FastAPI(
    title="Hypertension CDSS API",
    description="Sistem Pendukung Keputusan Klinis untuk Deteksi Hipertensi berbasis XAI dan RAG",
    version="1.0.0"
)


# Buka gerbang CORS biar Frontend (React/Next.js) lu nanti bisa nembak API ini tanpa diblokir
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# DEFINISI PAYLOAD (Validasi Data Otomatis)
# ==========================================
class PatientPayload(BaseModel):
    age: float
    is_female: int
    bmi: float
    waist_cm: float
    is_smoker: int
    freq_instant_noodle: float
    ak02: int
    ak05: int
    ak07: int
    has_diabetes: int
    genetic_risk_score: float
    ps_A: int
    ps_B: int
    ps_C: int
    ps_E: int
    ps_F: int

# ==========================================
# ENDPOINT UTAMA
# ==========================================
@app.post("/api/v1/analyze", tags=["Clinical Analysis"])
async def analyze_patient(patient_data: PatientPayload):
    """
    Endpoint ini menerima 16 fitur pasien dari Frontend, 
    lalu menjalankan Prediksi ML, Analisis SHAP, dan Narasi Groq LLM.
    """
    try:
        # Ubah data dari Pydantic Model menjadi Dictionary biasa
        patient_dict = patient_data.dict()
        
        # Eksekusi pipa integrasi lu!
        result = run_full_clinical_analysis(patient_dict)
        
        return result
        
    except Exception as e:
        # Kalau ada error di dalam fungsi, langsung lempar HTTP 500
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint buat ngecek server nyala atau nggak
@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Server CDSS Hipertensi Menyala!"}