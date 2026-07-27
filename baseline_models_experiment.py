import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

def load_and_prepare_data(filepath):
    """
    Fungsi untuk memuat dataset final IFLS-5 yang sudah dibersihkan
    (32.334 baris, 16 fitur input).
    """
    # Ganti dengan path file dataset finalmu
    df = pd.read_csv(filepath) 
    
    # Asumsi kolom target bernama 'label_hypertension'
    X = df.drop(columns=['label_hypertension'])
    y = df['label_hypertension']
    
    # Stratified split 80:20 sesuai metodologi
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def run_baseline_experiments(X_train, X_test, y_train, y_test):
    """
    Fungsi untuk menjalankan dan mengevaluasi semua baseline model
    dengan threshold default 0.5.
    """
    # Inisialisasi model
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=False)
    }
    
    print("Mulai proses training dan evaluasi...\n")
    print("-" * 40)
    
    # Loop untuk training dan testing
    for name, model in models.items():
        # Training
        model.fit(X_train, y_train)
        
        # Prediksi (threshold 0.5)
        y_pred = model.predict(X_test)
        
        # Kalkulasi Metrik
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Cetak hasil agar sesuai dengan format Tabel V.5
        print(f"Model: {name}")
        print(f"Akurasi          : {acc * 100:.2f}%")
        print(f"Recall (Kelas 1) : {recall * 100:.2f}%")
        print(f"F1-Score         : {f1 * 100:.2f}%")
        print(f"Threshold        : 0,500 (default)")
        print("-" * 40)

if __name__ == "__main__":
    # Tentukan lokasi file dataset
    DATASET_PATH = "dataset_hipertensi_clean.csv" # Sesuaikan path-nya
    
    try:
        # 1. Siapkan data
        X_train, X_test, y_train, y_test = load_and_prepare_data(DATASET_PATH)
        
        # 2. Jalankan eksperimen
        run_baseline_experiments(X_train, X_test, y_train, y_test)
        
    except FileNotFoundError:
        print(f"Error: File {DATASET_PATH} tidak ditemukan. Pastikan path dataset sudah benar.")