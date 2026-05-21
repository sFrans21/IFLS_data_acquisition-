import joblib

# Sesuaikan dengan lokasi persis file scaler lu
SCALER_PATH = "backend/models/saved_models/standard_scaler.pkl"
XGBOOST_PATH = "backend/models/saved_models/xgboost_model.pkl"

print("\n MENGEKSTRAK INGATAN MODEL...\n")

try:
    scaler = joblib.load(SCALER_PATH)
    
    # Cek apakah scaler menyimpan nama fitur
    if hasattr(scaler, 'feature_names_in_'):
        kolom_asli = list(scaler.feature_names_in_)
        print("=== DAFTAR KOLOM DARI STANDARD SCALER ===")
        print(kolom_asli)
        print(f"\nTotal fitur: {len(kolom_asli)} kolom")
    else:
        print("Scaler tidak menyimpan nama kolom. Mari kita cek model XGBoost-nya...")
        model = joblib.load(XGBOOST_PATH)
        
        # XGBoost scikit-learn API
        if hasattr(model, 'feature_names_in_'):
            kolom_asli = list(model.feature_names_in_)
        # XGBoost native API
        elif hasattr(model, 'get_booster'):
            kolom_asli = model.get_booster().feature_names
        else:
            kolom_asli = "Tidak ditemukan di model maupun scaler."
            
        print("=== DAFTAR KOLOM DARI MODEL ML ===")
        print(kolom_asli)

except Exception as e:
    print(f"Error saat membaca file .pkl: {e}")