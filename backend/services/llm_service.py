
# import os
# import chromadb
# from chromadb.utils import embedding_functions
# import google.generativeai as genai
# from dotenv import load_dotenv

# # ==========================================
# # FIX: BACA .ENV DAN INJEKSI PAKSA
# # ==========================================
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ENV_PATH = os.path.join(BASE_DIR, ".env")

# load_dotenv(dotenv_path=ENV_PATH, override=True)
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# if not GEMINI_API_KEY:
#     raise ValueError("GEMINI_API_KEY tidak ditemukan di file .env")

# # Konfigurasi library lama
# genai.configure(api_key=GEMINI_API_KEY)

# # ==========================================
# # GUNAKAN MODEL LAMA YANG DIDUKUNG SDK INI
# # ==========================================
# llm_model = genai.GenerativeModel('gemini-pro')

# # Konfigurasi Path ChromaDB
# DB_DIR = os.path.join(BASE_DIR, "chrom_db")

# # Inisialisasi ChromaDB
# chroma_client = chromadb.PersistentClient(path=DB_DIR)
# sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# # ==========================================
# # 1. MODUL QUERY TRANSFORMER
# # ==========================================
# FEATURE_MAP = {
#     'waist_cm': 'obesitas sentral, lingkar pinggang, penurunan berat badan',
#     'freq_instant_noodle': 'diet rendah natrium, batasan konsumsi garam, mi instan',
#     'is_smoker': 'bahaya merokok, nikotin, plak pembuluh darah',
#     'bmi': 'indeks massa tubuh, obesitas, target berat badan',
#     'age': 'risiko kardiovaskular pada lansia, kekakuan pembuluh darah',
#     'has_diabetes': 'komplikasi diabetes dan hipertensi, target tensi ketat'
# }

# def transform_query(top_shap_features: dict) -> str:
#     query_parts = []
#     for feature in top_shap_features.keys():
#         if feature in FEATURE_MAP:
#             query_parts.append(FEATURE_MAP[feature])
    
#     if not query_parts:
#         return "modifikasi gaya hidup untuk hipertensi"
        
#     return " ".join(query_parts)

# # ==========================================
# # 2. FUNGSI UTAMA GENERATE NARRATIVE
# # ==========================================
# def generate_clinical_narrative(patient_profile: dict, top_shap_features: dict) -> str:
#     search_query = transform_query(top_shap_features)
    
#     try:
#         collection = chroma_client.get_collection(
#             name="pedoman_hipertensi",
#             embedding_function=sentence_transformer_ef
#         )
        
#         results = collection.query(
#             query_texts=[search_query],
#             n_results=2 
#         )
        
#         retrieved_chunks = results['documents'][0] if results['documents'] else []
#         context_text = "\n\n".join(retrieved_chunks)
        
#     except Exception as e:
#         print(f"Error saat retrieval: {e}")
#         context_text = "Gagal mengambil dokumen referensi."

#     prompt = f"""
# Anda adalah sebuah sistem Clinical Decision Support untuk deteksi hipertensi.
# Tugas Anda adalah merangkai narasi edukasi medis singkat untuk pasien.

# ATURAN KETAT:
# 1. Anda HANYA BOLEH menggunakan informasi dari DOKUMEN REFERENSI di bawah ini. Jangan mengarang saran medis di luar dokumen ini.
# 2. Jelaskan mengapa faktor risiko pasien berbahaya berdasarkan dokumen tersebut.
# 3. Gunakan bahasa Indonesia yang empatik namun tegas dan jangan menggunakan markdown formatting list, cetak tebal dll.
# 4. Maksimal 3 paragraf pendek.

# PROFIL PASIEN:
# - Usia: {patient_profile.get('age', 'Tidak diketahui')}
# - Faktor Risiko Utama (Berdasarkan Analisis AI): {list(top_shap_features.keys())}

# DOKUMEN REFERENSI DARI PEDOMAN MEDIS:
# {context_text}

# Buat narasi edukasi sekarang:
# """

#     response = llm_model.generate_content(prompt)
#     return response.text

# # ==========================================
# # BLOK TESTING LOKAL
# # ==========================================
# if __name__ == "__main__":
#     print("\n--- MENJALANKAN TESTING LLM SERVICE ---")
    
#     mock_profile = {"age": 45}
#     mock_shap = {
#         "waist_cm": 0.35, 
#         "freq_instant_noodle": 0.22
#     }
    
#     print("1. Hasil Query Transformer:")
#     print(transform_query(mock_shap))
#     print("\n2. Memanggil Gemini API... Mohon tunggu...\n")
    
#     hasil_narasi = generate_clinical_narrative(mock_profile, mock_shap)
#     print("3. Hasil Narasi Medis LLM:")
#     print(hasil_narasi)
#     print("\n--- TESTING SELESAI ---")















# import os
# import chromadb
# from chromadb.utils import embedding_functions
# from openai import OpenAI
# from dotenv import load_dotenv

# # ==========================================
# # FIX: BACA .ENV UNTUK OPENAI
# # ==========================================
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ENV_PATH = os.path.join(BASE_DIR, ".env")

# load_dotenv(dotenv_path=ENV_PATH, override=True)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# if not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY tidak ditemukan di file .env")

# # Inisialisasi OpenAI Client
# client = OpenAI(api_key=OPENAI_API_KEY)

# # Konfigurasi Path ChromaDB
# DB_DIR = os.path.join(BASE_DIR, "chrom_db")

# # Inisialisasi ChromaDB
# chroma_client = chromadb.PersistentClient(path=DB_DIR)
# sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# # ==========================================
# # 1. MODUL QUERY TRANSFORMER
# # ==========================================
# FEATURE_MAP = {
#     'waist_cm': 'obesitas sentral, lingkar pinggang, penurunan berat badan',
#     'freq_instant_noodle': 'diet rendah natrium, batasan konsumsi garam, mi instan',
#     'is_smoker': 'bahaya merokok, nikotin, plak pembuluh darah',
#     'bmi': 'indeks massa tubuh, obesitas, target berat badan',
#     'age': 'risiko kardiovaskular pada lansia, kekakuan pembuluh darah',
#     'has_diabetes': 'komplikasi diabetes dan hipertensi, target tensi ketat'
# }

# def transform_query(top_shap_features: dict) -> str:
#     query_parts = []
#     for feature in top_shap_features.keys():
#         if feature in FEATURE_MAP:
#             query_parts.append(FEATURE_MAP[feature])
    
#     if not query_parts:
#         return "modifikasi gaya hidup untuk hipertensi"
        
#     return " ".join(query_parts)

# # ==========================================
# # 2. FUNGSI UTAMA GENERATE NARRATIVE
# # ==========================================
# def generate_clinical_narrative(patient_profile: dict, top_shap_features: dict) -> str:
#     search_query = transform_query(top_shap_features)
    
#     try:
#         collection = chroma_client.get_collection(
#             name="pedoman_hipertensi",
#             embedding_function=sentence_transformer_ef
#         )
        
#         results = collection.query(
#             query_texts=[search_query],
#             n_results=2 
#         )
        
#         retrieved_chunks = results['documents'][0] if results['documents'] else []
#         context_text = "\n\n".join(retrieved_chunks)
        
#     except Exception as e:
#         print(f"Error saat retrieval: {e}")
#         context_text = "Gagal mengambil dokumen referensi."

#     prompt_content = f"""
# Anda adalah sebuah sistem Clinical Decision Support untuk deteksi hipertensi.
# Tugas Anda adalah merangkai narasi edukasi medis singkat untuk pasien.

# ATURAN KETAT:
# 1. Anda HANYA BOLEH menggunakan informasi dari DOKUMEN REFERENSI di bawah ini. Jangan mengarang saran medis di luar dokumen ini.
# 2. Jelaskan mengapa faktor risiko pasien berbahaya berdasarkan dokumen tersebut.
# 3. Gunakan bahasa Indonesia yang empatik namun tegas dan jangan menggunakan markdown formatting list, cetak tebal dll.
# 4. Maksimal 3 paragraf pendek.

# PROFIL PASIEN:
# - Usia: {patient_profile.get('age', 'Tidak diketahui')}
# - Faktor Risiko Utama (Berdasarkan Analisis AI): {list(top_shap_features.keys())}

# DOKUMEN REFERENSI DARI PEDOMAN MEDIS:
# {context_text}
# """

#     # Memanggil model gpt-3.5-turbo (Sangat murah dan cepat)
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "Anda adalah asisten AI medis."},
#             {"role": "user", "content": prompt_content}
#         ]
#     )
    
#     return response.choices[0].message.content

# # ==========================================
# # BLOK TESTING LOKAL
# # ==========================================
# if __name__ == "__main__":
#     print("\n--- MENJALANKAN TESTING LLM SERVICE ---")
    
#     mock_profile = {"age": 45}
#     mock_shap = {
#         "waist_cm": 0.35, 
#         "freq_instant_noodle": 0.22
#     }
    
#     print("1. Hasil Query Transformer:")
#     print(transform_query(mock_shap))
#     print("\n2. Memanggil OpenAI API... Mohon tunggu...\n")
    
#     hasil_narasi = generate_clinical_narrative(mock_profile, mock_shap)
#     print("3. Hasil Narasi Medis LLM:")
#     print(hasil_narasi)
#     print("\n--- TESTING SELESAI ---")




import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================
# FIX: BACA .ENV UNTUK GROQ
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(dotenv_path=ENV_PATH, override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY tidak ditemukan di file .env")

# Inisialisasi OpenAI Client TAPI dibelokkan ke server Groq
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1" # <-- Ini rahasianya
)

# if not OPENAI_API_KEY:
#     raise ValueError("OPENAI_API_KEY tidak ditemukan di file .env")

# # Inisialisasi murni OpenAI Client
# client = OpenAI(
#     api_key=OPENAI_API_KEY
# )


# Konfigurasi Path ChromaDB
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Inisialisasi ChromaDB
chroma_client = chromadb.PersistentClient(path=DB_DIR)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# ==========================================
# 1. MODUL QUERY TRANSFORMER
# ==========================================
FEATURE_MAP = {
    'age': 'risiko kardiovaskular pada lansia, penuaan pembuluh darah, target tensi lansia',
    'is_female': 'risiko hipertensi pada wanita, menopause, hormon',
    'bmi': 'indeks massa tubuh, obesitas, target berat badan ideal',
    'waist_cm': 'obesitas sentral, lingkar pinggang, penumpukan lemak perut',
    'is_smoker': 'bahaya merokok, nikotin, kerusakan pembuluh darah',
    'freq_instant_noodle': 'diet rendah natrium, batasan konsumsi garam, mi instan',
    'has_diabetes': 'komplikasi diabetes dan hipertensi',
    'genetic_risk_score': 'riwayat hipertensi keluarga, faktor keturunan genetik',
    'ak02': 'aktivitas fisik kurang, olahraga untuk penderita hipertensi',
    'ak05': 'kualitas tidur buruk, istirahat kurang',
    'ak07': 'konsumsi alkohol, pola hidup tidak sehat',
    'ps_A': 'stres psikososial, manajemen stres, pikiran',
    'ps_B': 'stres emosional, kecemasan',
    'ps_C': 'gejala depresi, pengaruh pikiran negatif',
    'ps_E': 'gangguan tidur karena stres, insomnia',
    'ps_F': 'rasa kesepian, butuh dukungan sosial'
}

# Nama tampilan tiap faktor untuk narasi ke pasien (selaras dengan label di frontend).
NAME_MAP = {
    'age': 'usia',
    'is_female': 'jenis kelamin',
    'bmi': 'indeks massa tubuh (BMI)',
    'waist_cm': 'lingkar pinggang',
    'is_smoker': 'kebiasaan merokok',
    'freq_instant_noodle': 'konsumsi mi instan',
    'ak02': 'kurangnya aktivitas fisik berat',
    'ak05': 'kurangnya aktivitas fisik sedang',
    'ak07': 'durasi aktivitas fisik',
    'has_diabetes': 'riwayat diabetes',
    'genetic_risk_score': 'riwayat hipertensi keluarga',
    'ps_A': 'mudah merasa terganggu',
    'ps_B': 'kesulitan berkonsentrasi',
    'ps_C': 'perasaan sedih atau murung',
    'ps_E': 'tingkat optimisme',
    'ps_F': 'rasa cemas',
}

def transform_query(top_shap_features: dict) -> str:
    query_parts = []
    for feature in top_shap_features.keys():
        if feature in FEATURE_MAP:
            query_parts.append(FEATURE_MAP[feature])
    
    if not query_parts:
        return "modifikasi gaya hidup untuk mencegah hipertensi"
        
    return " ".join(query_parts)

# ==========================================
# 2. FUNGSI UTAMA GENERATE NARRATIVE (TAMBAH PARAMETER RISIKO)
# ==========================================
# Perhatikan tambahan parameter risk_score dan risk_status di bawah ini:
def generate_clinical_narrative(patient_profile: dict, top_shap_features: dict, risk_score: float, risk_status: str) -> str:
    search_query = transform_query(top_shap_features)
    
    try:
        collection = chroma_client.get_collection(
            name="pedoman_hipertensi",
            embedding_function=sentence_transformer_ef
        )
        
        results = collection.query(
            query_texts=[search_query],
            n_results=2 
        )
        
        retrieved_chunks = results['documents'][0] if results['documents'] else []
        context_text = "\n\n".join(retrieved_chunks)
        
    except Exception as e:
        print(f"Error saat retrieval: {e}")
        context_text = "Gagal mengambil dokumen referensi."

# --- Klasifikasi faktor: arah (tanda SHAP) + bisa/tidak bisa diubah ---
    NON_MODIFIABLE = {"age", "is_female", "genetic_risk_score"}

    total_impact = sum(abs(v) for v in top_shap_features.values())
    ranked = sorted(top_shap_features.items(), key=lambda kv: abs(kv[1]), reverse=True)

    def pct(value):
        return (abs(value) / total_impact * 100) if total_impact > 0 else 0.0

    # Untuk NARASI: fokus 3 faktor terbesar, dipisah berdasarkan arahnya.
    naik, turun = [], []
    for key, value in ranked[:3]:
        nama = NAME_MAP.get(key, key)
        frase = f"{nama} ({pct(value):.1f}%)"
        (naik if value > 0 else turun).append(frase)

    # Untuk PROMPT: kelompokkan target saran.
    target_saran = [NAME_MAP.get(k, k) for k, v in ranked[:3] if v > 0 and k not in NON_MODIFIABLE]
    faktor_tetap = [NAME_MAP.get(k, k) for k, v in ranked[:3] if v > 0 and k in NON_MODIFIABLE]
    faktor_protektif = [NAME_MAP.get(k, k) for k, v in ranked[:3] if v < 0]

    kategori = "tinggi" if "high" in risk_status.lower() else "rendah"

    def gabung(items):
        if len(items) > 1:
            return ", ".join(items[:-1]) + ", dan " + items[-1]
        return items[0] if items else ""

    intro = (
        f"Berdasarkan analisis data, risiko hipertensi Anda adalah "
        f"{risk_score * 100:.1f}% yang termasuk dalam kategori {kategori}."
    )
    if naik:
        intro += f" Faktor yang paling menaikkan risiko Anda adalah {gabung(naik)}."
    if turun:
        intro += f" Sebaliknya, faktor yang membantu menurunkan risiko Anda adalah {gabung(turun)}."
    
    # Prompt LLM yang sekarang jauh lebih cerdas dan situasional
    prompt_content = f"""
Anda adalah dokter spesialis yang empatik. Tugas Anda HANYA menulis bagian SARAN KLINIS singkat, melanjutkan narasi yang sudah ada.

KONTEKS PASIEN:
- Kategori risiko: {risk_status}
- Faktor yang BISA diubah & menaikkan risiko (PRIORITAS SARAN): {target_saran if target_saran else "tidak ada"}
- Faktor yang TIDAK bisa diubah (hanya akui, jangan disuruh ubah): {faktor_tetap if faktor_tetap else "tidak ada"}
- Faktor yang menurunkan risiko (boleh diapresiasi agar dipertahankan): {faktor_protektif if faktor_protektif else "tidak ada"}

ATURAN WAJIB:
1. Mulai TEPAT dengan kalimat: "Berdasarkan faktor tersebut, disarankan ...".
2. Beri saran konkret HANYA untuk "Faktor yang BISA diubah & menaikkan risiko". JANGAN menyuruh pasien mengubah faktor yang tidak bisa diubah (usia, jenis kelamin, riwayat keluarga); cukup sebut sebagai hal yang perlu diwaspadai.
3. Jika daftar faktor yang bisa diubah KOSONG, alihkan ke saran pencegahan umum sesuai DOKUMEN REFERENSI (pola makan, aktivitas fisik, kontrol tekanan darah rutin).
4. Saran HANYA boleh bersumber dari DOKUMEN REFERENSI di bawah. Dilarang mengarang angka atau anjuran di luar dokumen.
5. AKURASI MEDIS (PENTING): bila menyebut natrium/garam, gunakan kata "membatasi" atau "mengurangi" (contoh: "membatasi asupan natrium hingga kurang dari 2000 mg per hari"). DILARANG menulis "meningkatkan asupan natrium".
6. Bahasa Indonesia natural, tanpa markdown (tanpa *, tanpa bullet, tanpa bold). Maksimal 2 kalimat. Jangan mengulang angka persentase atau skor risiko.

DOKUMEN REFERENSI KEMENKES:
{context_text}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Anda adalah dokter spesialis yang empatik."},
            {"role": "user", "content": prompt_content}
        ]
    )
    
    # response = client.chat.completions.create(
    #     model="gpt-4o", # <-- Ganti dengan model OpenAI
    #     messages=[
    #         {"role": "system", "content": "Anda adalah dokter spesialis yang empatik."},
    #         {"role": "user", "content": prompt_content}
    #     ]
    # )

    saran = response.choices[0].message.content.strip()
    # Paragraf 1 = fakta deterministik, Paragraf 2 = saran dari LLM
    return f"{intro}\n\n{saran}"

# ==========================================
# BLOK TESTING LOKAL
# ==========================================
if __name__ == "__main__":
    print("\n--- MENJALANKAN TESTING LLM SERVICE ---")
    
    mock_profile = {"age": 45}
    mock_shap = {
        "waist_cm": 0.35, 
        "freq_instant_noodle": 0.22
    }
    
    print("1. Hasil Query Transformer:")
    print(transform_query(mock_shap))
    print("\n2. Memanggil Groq API... Mohon tunggu...\n")
    
    hasil_narasi = generate_clinical_narrative(mock_profile, mock_shap)
    print("3. Hasil Narasi Medis LLM:")
    print(hasil_narasi)
    print("\n--- TESTING SELESAI ---")