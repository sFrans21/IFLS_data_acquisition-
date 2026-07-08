
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
















# import os
# import chromadb
# from chromadb.utils import embedding_functions
# from openai import OpenAI
# from dotenv import load_dotenv

# # ==========================================
# # FIX: BACA .ENV UNTUK GROQ
# # ==========================================
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ENV_PATH = os.path.join(BASE_DIR, ".env")

# load_dotenv(dotenv_path=ENV_PATH, override=True)
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# if not GROQ_API_KEY:
#     raise ValueError("GROQ_API_KEY tidak ditemukan di file .env")

# # Inisialisasi OpenAI Client TAPI dibelokkan ke server Groq
# client = OpenAI(
#     api_key=GROQ_API_KEY,
#     base_url="https://api.groq.com/openai/v1" # <-- Ini rahasianya
# )

# # if not OPENAI_API_KEY:
# #     raise ValueError("OPENAI_API_KEY tidak ditemukan di file .env")

# # # Inisialisasi murni OpenAI Client
# # client = OpenAI(
# #     api_key=OPENAI_API_KEY
# # )


# # Konfigurasi Path ChromaDB
# DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# # Inisialisasi ChromaDB
# chroma_client = chromadb.PersistentClient(path=DB_DIR)
# sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# # ==========================================
# # 1. MODUL QUERY TRANSFORMER
# # ==========================================
# # Query RAG per fitur — HANYA 8 fitur yang benar-benar dipakai model saat ini.
# FEATURE_MAP = {
#     'age': 'risiko kardiovaskular pada lansia, penuaan pembuluh darah, target tensi lansia',
#     'is_female': 'risiko hipertensi pada wanita, menopause, hormon',
#     'bmi': 'indeks massa tubuh, obesitas, target berat badan ideal',
#     'is_smoker': 'bahaya merokok, nikotin, kerusakan pembuluh darah',
#     'has_diabetes': 'komplikasi diabetes dan hipertensi, target tensi ketat',
#     'has_high_cholesterol': 'kolesterol tinggi, dislipidemia, plak dan penyempitan pembuluh darah, aterosklerosis',
#     'sleep_quality': 'kualitas tidur buruk dan hipertensi, istirahat tidak berkualitas, tekanan darah',
#     'sleep_disturbance': 'gangguan tidur, sulit tidur atau sering terbangun, insomnia dan tekanan darah',
# }

# # Nama tampilan tiap faktor untuk narasi ke pasien (selaras dengan label di frontend).
# NAME_MAP = {
#     'age': 'usia',
#     'is_female': 'jenis kelamin',
#     'bmi': 'indeks massa tubuh (BMI)',
#     'is_smoker': 'kebiasaan merokok',
#     'has_diabetes': 'riwayat diabetes',
#     'has_high_cholesterol': 'riwayat kolesterol tinggi',
#     'sleep_quality': 'kualitas tidur',
#     'sleep_disturbance': 'gangguan tidur',
# }

# def transform_query(top_shap_features: dict) -> str:
#     query_parts = []
#     for feature in top_shap_features.keys():
#         if feature in FEATURE_MAP:
#             query_parts.append(FEATURE_MAP[feature])
    
#     if not query_parts:
#         return "modifikasi gaya hidup untuk mencegah hipertensi"
        
#     return " ".join(query_parts)

# # ==========================================
# # 2. FUNGSI UTAMA GENERATE NARRATIVE (TAMBAH PARAMETER RISIKO)
# # ==========================================
# # Perhatikan tambahan parameter risk_score dan risk_status di bawah ini:
# def generate_clinical_narrative(patient_profile: dict, top_shap_features: dict, risk_score: float, risk_status: str) -> str:
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

# # --- Klasifikasi faktor: arah (tanda SHAP) + bisa/tidak bisa diubah ---
#     NON_MODIFIABLE = {"age", "is_female"}

#     total_impact = sum(abs(v) for v in top_shap_features.values())
#     ranked = sorted(top_shap_features.items(), key=lambda kv: abs(kv[1]), reverse=True)

#     def pct(value):
#         return (abs(value) / total_impact * 100) if total_impact > 0 else 0.0

#     # Untuk NARASI: fokus 3 faktor terbesar, dipisah berdasarkan arahnya.
#     naik, turun = [], []
#     for key, value in ranked[:3]:
#         nama = NAME_MAP.get(key, key)
#         frase = f"{nama} ({pct(value):.1f}%)"
#         (naik if value > 0 else turun).append(frase)

#     # Untuk PROMPT: kelompokkan target saran.
#     target_saran = [NAME_MAP.get(k, k) for k, v in ranked[:3] if v > 0 and k not in NON_MODIFIABLE]
#     faktor_tetap = [NAME_MAP.get(k, k) for k, v in ranked[:3] if v > 0 and k in NON_MODIFIABLE]
#     faktor_protektif = [NAME_MAP.get(k, k) for k, v in ranked[:3] if v < 0]

#     kategori = "tinggi" if "high" in risk_status.lower() else "rendah"

#     def gabung(items):
#         if len(items) > 1:
#             return ", ".join(items[:-1]) + ", dan " + items[-1]
#         return items[0] if items else ""

#     intro = (
#         f"Hasil analisis menunjukkan risiko hipertensi Anda berada di angka "
#         f"{risk_score * 100:.1f}% yang termasuk dalam kategori {kategori}."
#     )
#     if naik:
#         intro += f" Faktor yang paling menaikkan risiko Anda adalah {gabung(naik)}."
#     if turun:
#         intro += f" Sebaliknya, faktor yang membantu menurunkan risiko Anda adalah {gabung(turun)}."
    
#     # Prompt LLM yang sekarang jauh lebih cerdas dan situasional
#     prompt_content = f"""
# Anda adalah dokter spesialis yang empatik. Tugas Anda HANYA menulis bagian SARAN KLINIS singkat, melanjutkan narasi yang sudah ada.

# KONTEKS PASIEN:
# - Kategori risiko: {risk_status}
# - Faktor yang BISA diubah & menaikkan risiko (PRIORITAS SARAN): {target_saran if target_saran else "tidak ada"}
# - Faktor yang TIDAK bisa diubah (hanya akui, jangan disuruh ubah): {faktor_tetap if faktor_tetap else "tidak ada"}
# - Faktor yang menurunkan risiko (boleh diapresiasi agar dipertahankan): {faktor_protektif if faktor_protektif else "tidak ada"}

# ATURAN WAJIB:
# 1. Mulai TEPAT dengan kalimat: "Berdasarkan faktor tersebut, disarankan ...".
# 2. Beri saran konkret HANYA untuk "Faktor yang BISA diubah & menaikkan risiko". JANGAN menyuruh pasien mengubah faktor yang tidak bisa diubah (usia, jenis kelamin, riwayat keluarga); cukup sebut sebagai hal yang perlu diwaspadai.
# 3. Jika daftar faktor yang bisa diubah KOSONG, alihkan ke saran pencegahan umum sesuai DOKUMEN REFERENSI (pola makan, aktivitas fisik, kontrol tekanan darah rutin).
# 4. Saran HANYA boleh bersumber dari DOKUMEN REFERENSI di bawah. Dilarang mengarang angka atau anjuran di luar dokumen.
# 5. AKURASI MEDIS (PENTING): bila menyebut natrium/garam, gunakan kata "membatasi" atau "mengurangi" (contoh: "membatasi asupan natrium hingga kurang dari 2000 mg per hari"). DILARANG menulis "meningkatkan asupan natrium".
# 6. Bahasa Indonesia natural, tanpa markdown (tanpa *, tanpa bullet, tanpa bold). Maksimal 2 kalimat. Jangan mengulang angka persentase atau skor risiko.

# DOKUMEN REFERENSI KEMENKES:
# {context_text}
# """

#     response = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[
#             {"role": "system", "content": "Anda adalah dokter spesialis yang empatik."},
#             {"role": "user", "content": prompt_content}
#         ]
#     )
    
#     # response = client.chat.completions.create(
#     #     model="gpt-4o", # <-- Ganti dengan model OpenAI
#     #     messages=[
#     #         {"role": "system", "content": "Anda adalah dokter spesialis yang empatik."},
#     #         {"role": "user", "content": prompt_content}
#     #     ]
#     # )

#     saran = response.choices[0].message.content.strip()
#     # Paragraf 1 = fakta deterministik, Paragraf 2 = saran dari LLM
#     return f"{intro}\n\n{saran}"

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
#     print("\n2. Memanggil Groq API... Mohon tunggu...\n")
    
#     hasil_narasi = generate_clinical_narrative(mock_profile, mock_shap)
#     print("3. Hasil Narasi Medis LLM:")
#     print(hasil_narasi)
#     print("\n--- TESTING SELESAI ---")












# import os
# import chromadb
# from chromadb.utils import embedding_functions
# import google.generativeai as genai
# from dotenv import load_dotenv

# # ==========================================
# # 1. KONFIGURASI GEMINI API
# # ==========================================
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ENV_PATH = os.path.join(BASE_DIR, ".env")

# load_dotenv(dotenv_path=ENV_PATH, override=True)
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# if not GEMINI_API_KEY:
#     raise ValueError("GEMINI_API_KEY tidak ditemukan di file .env")

# # Konfigurasi SDK Gemini
# genai.configure(api_key=GEMINI_API_KEY)

# # Inisialisasi Model Gemini (Menggunakan 1.5 Flash untuk Free Tier yang cepat)
# # System instruction bisa langsung dimasukkan di sini
# llm_model = genai.GenerativeModel(
#     model_name='gemini-1.5-flash',
#     system_instruction="Anda adalah dokter spesialis yang empatik."
# )

# # ==========================================
# # 2. KONFIGURASI CHROMADB
# # ==========================================
# DB_DIR = os.path.join(BASE_DIR, "chroma_db")
# chroma_client = chromadb.PersistentClient(path=DB_DIR)
# sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# # ==========================================
# # 3. MODUL QUERY TRANSFORMER
# # ==========================================
# FEATURE_MAP = {
#     'age': 'risiko kardiovaskular pada lansia, penuaan pembuluh darah, target tensi lansia',
#     'is_female': 'risiko hipertensi pada wanita, menopause, hormon',
#     'bmi': 'indeks massa tubuh, obesitas, target berat badan ideal',
#     'waist_cm': 'obesitas sentral, lingkar pinggang, penumpukan lemak perut',
#     'is_smoker': 'bahaya merokok, nikotin, kerusakan pembuluh darah',
#     'freq_instant_noodle': 'diet rendah natrium, batasan konsumsi garam, mi instan',
#     'has_diabetes': 'komplikasi diabetes dan hipertensi',
#     'genetic_risk_score': 'riwayat hipertensi keluarga, faktor keturunan genetik',
#     'ak02': 'aktivitas fisik kurang, olahraga untuk penderita hipertensi',
#     'ak05': 'kualitas tidur buruk, istirahat kurang',
#     'ak07': 'konsumsi alkohol, pola hidup tidak sehat',
#     'ps_A': 'stres psikososial, manajemen stres, pikiran',
#     'ps_B': 'stres emosional, kecemasan',
#     'ps_C': 'gejala depresi, pengaruh pikiran negatif',
#     'ps_E': 'gangguan tidur karena stres, insomnia',
#     'ps_F': 'rasa kesepian, butuh dukungan sosial'
# }

# NAME_MAP = {
#     'age': 'usia',
#     'is_female': 'jenis kelamin',
#     'bmi': 'indeks massa tubuh (BMI)',
#     'waist_cm': 'lingkar pinggang',
#     'is_smoker': 'kebiasaan merokok',
#     'freq_instant_noodle': 'konsumsi mi instan',
#     'ak02': 'kurangnya aktivitas fisik berat',
#     'ak05': 'kurangnya aktivitas fisik sedang',
#     'ak07': 'durasi aktivitas fisik',
#     'has_diabetes': 'riwayat diabetes',
#     'genetic_risk_score': 'riwayat hipertensi keluarga',
#     'ps_A': 'mudah merasa terganggu',
#     'ps_B': 'kesulitan berkonsentrasi',
#     'ps_C': 'perasaan sedih atau murung',
#     'ps_E': 'tingkat optimisme',
#     'ps_F': 'rasa cemas',
# }

# def transform_query(top_shap_features: dict) -> str:
#     query_parts = []
#     for feature in top_shap_features.keys():
#         if feature in FEATURE_MAP:
#             query_parts.append(FEATURE_MAP[feature])
    
#     if not query_parts:
#         return "modifikasi gaya hidup untuk mencegah hipertensi"
        
#     return " ".join(query_parts)

# # ==========================================
# # 4. FUNGSI UTAMA GENERATE NARRATIVE
# # ==========================================
# def generate_clinical_narrative(patient_profile: dict, top_shap_features: dict, risk_score: float, risk_status: str) -> str:
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

#     # --- Klasifikasi faktor: arah (tanda SHAP) + bisa/tidak bisa diubah ---
#     NON_MODIFIABLE = {"age", "is_female", "genetic_risk_score"}

#     total_impact = sum(abs(v) for v in top_shap_features.values())
#     ranked = sorted(top_shap_features.items(), key=lambda kv: abs(kv[1]), reverse=True)

#     def pct(value):
#         return (abs(value) / total_impact * 100) if total_impact > 0 else 0.0

#     naik, turun = [], []
#     for key, value in ranked[:3]:
#         nama = NAME_MAP.get(key, key)
#         frase = f"{nama} ({pct(value):.1f}%)"
#         (naik if value > 0 else turun).append(frase)

#     target_saran = [NAME_MAP.get(k, k) for k, v in ranked[:3] if v > 0 and k not in NON_MODIFIABLE]
#     faktor_tetap = [NAME_MAP.get(k, k) for k, v in ranked[:3] if v > 0 and k in NON_MODIFIABLE]
#     faktor_protektif = [NAME_MAP.get(k, k) for k, v in ranked[:3] if v < 0]

#     kategori = "tinggi" if "high" in risk_status.lower() else "rendah"

#     def gabung(items):
#         if len(items) > 1:
#             return ", ".join(items[:-1]) + ", dan " + items[-1]
#         return items[0] if items else ""

#     intro = (
#         f"Berdasarkan analisis data, risiko hipertensi Anda adalah "
#         f"{risk_score * 100:.1f}% yang termasuk dalam kategori {kategori}."
#     )
#     if naik:
#         intro += f" Faktor yang paling menaikkan risiko Anda adalah {gabung(naik)}."
#     if turun:
#         intro += f" Sebaliknya, faktor yang membantu menurunkan risiko Anda adalah {gabung(turun)}."
    
#     prompt_content = f"""
# Tugas Anda HANYA menulis bagian SARAN KLINIS singkat, melanjutkan narasi yang sudah ada.

# KONTEKS PASIEN:
# - Kategori risiko: {risk_status}
# - Faktor yang BISA diubah & menaikkan risiko (PRIORITAS SARAN): {target_saran if target_saran else "tidak ada"}
# - Faktor yang TIDAK bisa diubah (hanya akui, jangan disuruh ubah): {faktor_tetap if faktor_tetap else "tidak ada"}
# - Faktor yang menurunkan risiko (boleh diapresiasi agar dipertahankan): {faktor_protektif if faktor_protektif else "tidak ada"}

# ATURAN WAJIB:
# 1. Mulai TEPAT dengan kalimat: "Berdasarkan faktor tersebut, disarankan ...".
# 2. Beri saran konkret HANYA untuk "Faktor yang BISA diubah & menaikkan risiko". JANGAN menyuruh pasien mengubah faktor yang tidak bisa diubah (usia, jenis kelamin, riwayat keluarga); cukup sebut sebagai hal yang perlu diwaspadai.
# 3. Jika daftar faktor yang bisa diubah KOSONG, alihkan ke saran pencegahan umum sesuai DOKUMEN REFERENSI (pola makan, aktivitas fisik, kontrol tekanan darah rutin).
# 4. Saran HANYA boleh bersumber dari DOKUMEN REFERENSI di bawah. Dilarang mengarang angka atau anjuran di luar dokumen.
# 5. AKURASI MEDIS (PENTING): bila menyebut natrium/garam, gunakan kata "membatasi" atau "mengurangi" (contoh: "membatasi asupan natrium hingga kurang dari 2000 mg per hari"). DILARANG menulis "meningkatkan asupan natrium".
# 6. Bahasa Indonesia natural, tanpa markdown (tanpa bintang, tanpa bullet, tanpa bold). Maksimal 2 kalimat. Jangan mengulang angka persentase atau skor risiko.

# DOKUMEN REFERENSI KEMENKES:
# {context_text}
# """
    
#     # Memanggil Gemini API
#     response = llm_model.generate_content(prompt_content)
#     saran = response.text.strip()
    
#     return f"{intro}\n\n{saran}"

# # ==========================================
# # BLOK TESTING LOKAL
# # ==========================================
# if __name__ == "__main__":
#     print("\n--- MENJALANKAN TESTING LLM SERVICE (GEMINI API) ---")
    
#     mock_profile = {"age": 45}
#     mock_shap = {
#         "waist_cm": 0.35, 
#         "freq_instant_noodle": 0.22,
#         "is_female": -0.10 # Contoh faktor protektif
#     }
#     mock_score = 0.65
#     mock_status = "High Risk"
    
#     print("1. Hasil Query Transformer:")
#     print(transform_query(mock_shap))
#     print("\n2. Memanggil Gemini API... Mohon tunggu...\n")
    
#     # FIX: Menambahkan parameter risk_score dan risk_status yang sebelumnya kurang
#     hasil_narasi = generate_clinical_narrative(mock_profile, mock_shap, mock_score, mock_status)
#     print("3. Hasil Narasi Medis LLM:")
#     print(hasil_narasi)
#     print("\n--- TESTING SELESAI ---")



























# """
# llm_service.py  -- Modul Narasi Medis Hipertensi (REVISI)

# Arsitektur tiga lapis:
#   (1) Lapisan deterministik : menerjemahkan nilai SHAP -> arah, besaran, dan
#       barometer klinis. Semua ANGKA dan ARAH dihitung di sini, bukan oleh LLM.
#   (2) Retrieval-Augmented Generation (RAG) : mengambil potongan Pedoman
#       Hipertensi Kemenkes dari ChromaDB sebagai sumber saran yang tepercaya.
#   (3) LLM (Groq / Llama-3.1) : HANYA merangkai kalimat saran menjadi bahasa
#       yang empatik; dilarang membuat angka, arah, atau anjuran baru.

# Prinsip anti-halusinasi / anti-misinformasi:
#   - Arah efek diambil dari TANDA nilai SHAP individu, bukan dari asumsi atau
#     dari korelasi fitur (yang pada ringkasan model sempat terbalik).
#   - Persentase kontribusi = share lokal |shap_i| / sum_j |shap_j| (satuan
#     log-odds, ruang aditif SHAP), dikategorikan dengan ambang 50/25/10/3.
#   - Setiap fitur diberi TINGKAT KEANDALAN (Tier A/B/C). Fitur yang polanya
#     berlawanan dengan bukti medis (is_smoker) di-override: teks fakta selalu
#     menyatakan kebenaran medis, dan arah "protektif" yang keliru TIDAK PERNAH
#     ditampilkan ke pasien.
#   - Saran gaya hidup dipicu oleh KONDISI KLINIS nyata pasien (mis. IMT >= 25,
#     perokok), bukan oleh tanda SHAP -- sehingga saran berhenti merokok tetap
#     muncul untuk perokok meski SHAP model menandainya "protektif".
# """

# import os

# # ------------------------------------------------------------------
# # Import berat (LLM & RAG) dibuat opsional agar lapisan deterministik
# # tetap dapat diimpor dan diuji tanpa dependensi eksternal.
# # ------------------------------------------------------------------
# try:
#     import chromadb
#     from chromadb.utils import embedding_functions
#     from openai import OpenAI
#     from dotenv import load_dotenv
#     _DEPS_OK = True
# except Exception:  # pragma: no cover
#     _DEPS_OK = False


# # ==================================================================
# # 0. KONFIGURASI KLIEN (Groq via SDK OpenAI + ChromaDB)
# # ==================================================================
# client = None
# chroma_client = None
# sentence_transformer_ef = None
# LLM_MODEL = "llama-3.1-8b-instant"

# if _DEPS_OK:
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     ENV_PATH = os.path.join(BASE_DIR, ".env")
#     load_dotenv(dotenv_path=ENV_PATH, override=True)

#     GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#     if GROQ_API_KEY:
#         client = OpenAI(
#             api_key=GROQ_API_KEY,
#             base_url="https://api.groq.com/openai/v1",
#         )

#     try:
#         DB_DIR = os.path.join(BASE_DIR, "chroma_db")
#         chroma_client = chromadb.PersistentClient(path=DB_DIR)
#         sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
#             model_name="all-MiniLM-L6-v2"
#         )
#     except Exception as e:  # pragma: no cover
#         print(f"[WARN] ChromaDB tidak siap: {e}")


# # ==================================================================
# # 1. KAMUS FITUR
# # ==================================================================
# # Kueri RAG per fitur (menuntun retriever ke bagian pedoman yang relevan).
# FEATURE_MAP = {
#     'age': 'risiko kardiovaskular pada lansia, penuaan pembuluh darah',
#     'is_female': 'risiko hipertensi menurut jenis kelamin, menopause, hormon',
#     'bmi': 'indeks massa tubuh, obesitas, target berat badan ideal',
#     'is_smoker': 'bahaya merokok, nikotin, kerusakan pembuluh darah, berhenti merokok',
#     'has_diabetes': 'komplikasi diabetes dan hipertensi, kontrol gula darah',
#     'has_high_cholesterol': 'kolesterol tinggi, dislipidemia, aterosklerosis, diet rendah lemak jenuh',
#     'sleep_quality': 'kualitas tidur dan tekanan darah, higiene tidur',
#     'sleep_disturbance': 'gangguan tidur, insomnia, dan tekanan darah',
# }

# # Nama tampilan untuk narasi ke pasien.
# NAME_MAP = {
#     'age': 'usia',
#     'is_female': 'jenis kelamin',
#     'bmi': 'indeks massa tubuh (IMT)',
#     'is_smoker': 'kebiasaan merokok',
#     'has_diabetes': 'riwayat diabetes',
#     'has_high_cholesterol': 'riwayat kolesterol tinggi',
#     'sleep_quality': 'kualitas tidur',
#     'sleep_disturbance': 'gangguan tidur',
# }

# # Fitur yang tidak dapat diubah (hanya diakui, tidak dijadikan target saran).
# NON_MODIFIABLE = {"age", "is_female"}

# # Ambang netral dari analisis SHAP (lihat subbab Implementasi XAI).
# AMBANG_NETRAL = {"age": 43.0, "bmi": 24.0}


# # ==================================================================
# # 2. TINGKAT KEANDALAN FITUR (Tier A / B / C)
# # ==================================================================
# #   A = arah model searah & konsisten dengan bukti medis  -> fakta penuh
# #   B = magnitudo kecil / arah spesifik-model              -> fakta di-hedge
# #   C = arah model BERLAWANAN dengan bukti medis           -> override fakta
# TIER = {
#     'age': 'A', 'bmi': 'A', 'has_diabetes': 'A', 'has_high_cholesterol': 'A',
#     'is_female': 'B', 'sleep_quality': 'B', 'sleep_disturbance': 'B',
#     'is_smoker': 'C',
# }

# # Teks #2 (alasan berbasis fakta). Dipilih oleh TANDA SHAP untuk Tier A,
# # di-hedge untuk Tier B, dan di-override (selalu benar secara medis) untuk C.
# FAKTA_MEDIS = {
#     'age': {
#         'naik': "Bertambahnya usia meningkatkan risiko hipertensi karena elastisitas dinding arteri menurun dan kekakuan pembuluh darah meningkat seiring waktu.",
#         'turun': "Usia yang lebih muda cenderung bersifat protektif; risiko hipertensi umumnya baru meningkat seiring bertambahnya usia.",
#     },
#     'bmi': {
#         'naik': "Indeks massa tubuh yang lebih tinggi meningkatkan risiko hipertensi karena menambah beban kerja jantung dan resistensi pembuluh darah.",
#         'turun': "Menjaga indeks massa tubuh pada rentang sehat membantu tekanan darah tetap terkendali.",
#     },
#     'has_diabetes': {
#         'naik': "Riwayat diabetes meningkatkan risiko hipertensi melalui gangguan pada fungsi metabolik dan pembuluh darah.",
#         'turun': "Tidak adanya riwayat diabetes merupakan kondisi yang menguntungkan bagi kendali tekanan darah.",
#     },
#     'has_high_cholesterol': {
#         'naik': "Kolesterol tinggi meningkatkan risiko hipertensi karena mendorong pengerasan dan penyempitan pembuluh darah (aterosklerosis).",
#         'turun': "Kadar kolesterol yang normal mendukung kelancaran aliran darah dan kendali tekanan darah.",
#     },
#     # Tier B -- hedge, tanpa klaim sebab-akibat yang kuat.
#     'is_female': {
#         'hedge': "Pola risiko hipertensi menurut jenis kelamin bervariasi menurut usia dan status hormonal, sehingga tidak dijadikan dasar rekomendasi.",
#     },
#     'sleep_quality': {
#         'hedge': "Kaitan kualitas tidur dengan tekanan darah pada data ini belum konsisten dengan literatur, sehingga tidak dijadikan klaim sebab-akibat.",
#     },
#     'sleep_disturbance': {
#         'hedge': "Gangguan tidur dikaitkan dengan tekanan darah pada sebagian literatur, namun pada model ini pengaruhnya kecil sehingga disampaikan secara berhati-hati.",
#     },
#     # Tier C -- override: SELALU nyatakan fakta medis yang benar.
#     'is_smoker': {
#         'override': "Menjadi perokok aktif maupun mantan perokok cenderung meningkatkan risiko hipertensi karena nikotin menyempitkan pembuluh darah serta meningkatkan tekanan dan kekakuan arteri.",
#     },
# }


# # ==================================================================
# # 3. FUNGSI BANTU DETERMINISTIK
# # ==================================================================
# def _cap(teks: str) -> str:
#     """Kapitalisasi huruf pertama saja (menjaga singkatan seperti IMT)."""
#     return teks[:1].upper() + teks[1:] if teks else teks


# def kategori_magnitudo(persen: float) -> str:
#     """Ambang 50/25/10/3 (share kontribusi lokal)."""
#     if persen >= 50:
#         return "dominan"
#     if persen >= 25:
#         return "kuat"
#     if persen >= 10:
#         return "moderat"
#     if persen >= 3:
#         return "kecil"
#     return "sangat kecil"


# def klasifikasi_bmi(bmi: float) -> str:
#     """Barometer IMT menurut standar WHO."""
#     if bmi < 18.5:
#         return "di bawah normal (kurus)"
#     if bmi < 25:
#         return "normal"
#     if bmi < 30:
#         return "berlebih (overweight)"
#     return "obesitas"


# def _posisi_netral(fitur: str, nilai: float) -> str:
#     amb = AMBANG_NETRAL.get(fitur)
#     if amb is None or nilai is None:
#         return ""
#     return "di bawah" if nilai < amb else "di atas"


# def _deskripsi_nilai(fitur: str, pf: dict) -> str:
#     """Frasa nilai + barometer untuk teks objektif tiap fitur."""
#     v = pf.get(fitur)
#     if fitur == 'age' and v is not None:
#         return f"yakni {int(round(v))} tahun yang berada {_posisi_netral('age', v)} titik netral model (sekitar 43 tahun),"
#     if fitur == 'bmi' and v is not None:
#         return f"sebesar {v:.1f} yang tergolong {klasifikasi_bmi(v)},"
#     if fitur == 'is_female':
#         return "(laki-laki)" if v == 0 else "(perempuan)" if v == 1 else ""
#     if fitur == 'is_smoker':
#         return "(perokok)" if v == 1 else "(bukan perokok)" if v == 0 else ""
#     if fitur == 'has_diabetes':
#         return "(ada riwayat)" if v == 1 else "(tidak ada riwayat)" if v == 0 else ""
#     if fitur == 'has_high_cholesterol':
#         return "(ada riwayat)" if v == 1 else "(tidak ada riwayat)" if v == 0 else ""
#     if fitur in ('sleep_quality', 'sleep_disturbance') and v is not None:
#         return f"(skor {int(round(v))} dari 5)"
#     return ""


# def _teks_fakta(fitur: str, shap_val: float, persen: float) -> str:
#     """Teks #2 sesuai tier + tanda SHAP + guardrail magnitudo."""
#     tier = TIER.get(fitur, 'B')

#     # Tier C: is_smoker -> selalu fakta benar (override), abaikan tanda SHAP.
#     if tier == 'C':
#         return FAKTA_MEDIS[fitur]['override']

#     # Guardrail: kontribusi sangat kecil -> jangan buat klaim tegas.
#     if persen < 3:
#         return "Pengaruhnya terhadap prediksi ini dapat diabaikan."

#     if tier == 'A':
#         kunci = 'naik' if shap_val > 0 else 'turun'
#         return FAKTA_MEDIS[fitur][kunci]

#     # Tier B -> hedge.
#     return FAKTA_MEDIS[fitur].get('hedge', "Pengaruhnya kecil dan disampaikan secara berhati-hati.")


# def _teks_objektif(fitur: str, shap_val: float, persen: float, pf: dict) -> str:
#     """Teks #1 objektif: nama + nilai/barometer + kategori + arah + persen."""
#     nama = NAME_MAP.get(fitur, fitur)
#     desk = _deskripsi_nilai(fitur, pf)
#     kategori = kategori_magnitudo(persen)

#     # Override arah untuk is_smoker: JANGAN pernah menyebut "menurunkan".
#     if TIER.get(fitur) == 'C' or persen < 3:
#         return (f"{_cap(nama)} Anda {desk} memiliki pengaruh yang "
#                 f"dapat diabaikan pada prediksi ini (kontribusi {persen:.1f}%).")

#     arah = "meningkatkan" if shap_val > 0 else "menurunkan"
#     return (f"{_cap(nama)} Anda {desk} memiliki pengaruh {kategori} "
#             f"dalam {arah} risiko ini, dengan kontribusi {persen:.1f}% "
#             f"dari keseluruhan faktor.")


# # ==================================================================
# # 4. SARAN GAYA HIDUP (dipicu KONDISI KLINIS, bukan tanda SHAP)
# # ==================================================================
# SARAN_SNIPPET = {
#     'is_smoker':             "berhenti merokok karena memberi manfaat langsung bagi kesehatan pembuluh darah dan tekanan darah",
#     'bmi':                   "menjaga berat badan melalui penurunan kalori bertahap serta aktivitas fisik ringan hingga sedang seperti jalan kaki atau berenang",
#     'has_high_cholesterol':  "membatasi asupan lemak jenuh dan memperbanyak serat dari sayur, buah, serta biji-bijian utuh",
#     'has_diabetes':          "mengelola kadar gula darah dengan membatasi konsumsi gula dan karbohidrat olahan",
#     'sleep':                 "memperbaiki kualitas tidur dengan jadwal tidur yang teratur serta membatasi kafein dan paparan layar sebelum tidur",
# }
# # Urutan prioritas klinis (maks. 3 dipakai agar tetap 2-3 kalimat).
# PRIORITAS_SARAN = ['is_smoker', 'bmi', 'has_high_cholesterol', 'has_diabetes', 'sleep']
# SARAN_FALLBACK = "mempertahankan pola hidup sehat saat ini berupa pola makan seimbang, aktivitas fisik teratur, dan tidur yang cukup"
# SARAN_PENUTUP = "serta berkonsultasi dengan tenaga kesehatan untuk pemeriksaan dan penanganan lebih lanjut"


# def _pemicu_saran(pf: dict) -> list:
#     """Daftar faktor termodifikasi yang benar-benar tidak sehat pada pasien ini."""
#     picu = []
#     if pf.get('is_smoker') == 1:
#         picu.append('is_smoker')
#     if pf.get('bmi') is not None and pf['bmi'] >= 25:
#         picu.append('bmi')
#     if pf.get('has_high_cholesterol') == 1:
#         picu.append('has_high_cholesterol')
#     if pf.get('has_diabetes') == 1:
#         picu.append('has_diabetes')
#     sd, sq = pf.get('sleep_disturbance'), pf.get('sleep_quality')
#     if (sd is not None and sd >= 4) or (sq is not None and sq <= 2):
#         picu.append('sleep')
#     # urutkan menurut prioritas klinis, batasi 3
#     picu = [f for f in PRIORITAS_SARAN if f in picu][:3]
#     return picu


# def _gabung(items: list) -> str:
#     if len(items) > 1:
#         return ", ".join(items[:-1]) + ", dan " + items[-1]
#     return items[0] if items else ""


# def bangun_saran_deterministik(pf: dict) -> str:
#     """Saran gaya hidup 2-3 kalimat, sepenuhnya terkendali (tanpa LLM)."""
#     picu = _pemicu_saran(pf)
#     if not picu:
#         return f"Berdasarkan faktor tersebut, disarankan pasien untuk {SARAN_FALLBACK}, {SARAN_PENUTUP}."
#     isi = _gabung([SARAN_SNIPPET[f] for f in picu])
#     return f"Berdasarkan faktor tersebut, disarankan pasien untuk {isi}, {SARAN_PENUTUP}."


# # ==================================================================
# # 5. RAG + LLM (opsional; degradasi ke deterministik bila tak tersedia)
# # ==================================================================
# def transform_query(shap_values: dict) -> str:
#     parts = [FEATURE_MAP[f] for f in shap_values if f in FEATURE_MAP]
#     return " ".join(parts) if parts else "modifikasi gaya hidup untuk mencegah hipertensi"


# def _ambil_konteks_rag(shap_values: dict) -> str:
#     if chroma_client is None or sentence_transformer_ef is None:
#         return ""
#     try:
#         collection = chroma_client.get_collection(
#             name="pedoman_hipertensi", embedding_function=sentence_transformer_ef
#         )
#         res = collection.query(query_texts=[transform_query(shap_values)], n_results=2)
#         chunks = res['documents'][0] if res.get('documents') else []
#         return "\n\n".join(chunks)
#     except Exception as e:
#         print(f"[WARN] Retrieval gagal: {e}")
#         return ""


# def _saran_llm(pf: dict, konteks: str) -> str:
#     """LLM merangkai ulang saran deterministik agar lebih luwes; TIDAK menambah fakta."""
#     picu = _pemicu_saran(pf)
#     target = [NAME_MAP[f] if f != 'sleep' else 'kualitas tidur' for f in picu]
#     materi = bangun_saran_deterministik(pf)

#     if client is None:
#         return materi  # fallback deterministik

#     prompt = f"""Tugas Anda HANYA menuliskan ulang SARAN GAYA HIDUP berikut agar mengalir dan empatik,
# TANPA menambah, mengurangi, atau mengubah anjuran maupun angka apa pun.

# MATERI SARAN (sumber kebenaran, jangan diubah isinya):
# "{materi}"

# FAKTOR YANG BOLEH DISINGGUNG: {target if target else "pola hidup sehat umum"}

# DOKUMEN REFERENSI KEMENKES (boleh dipakai untuk memperjelas, dilarang mengarang di luar ini):
# {konteks if konteks else "(tidak tersedia)"}

# ATURAN WAJIB:
# 1. Mulai TEPAT dengan: "Berdasarkan faktor tersebut, disarankan pasien untuk ...".
# 2. Dilarang menyuruh mengubah usia atau jenis kelamin.
# 3. Bila menyebut natrium/garam, gunakan kata "membatasi"/"mengurangi"; DILARANG menulis "meningkatkan asupan natrium".
# 4. Bahasa Indonesia natural tanpa markdown (tanpa bintang, bullet, atau cetak tebal). Maksimal 3 kalimat.
# """
#     try:
#         resp = client.chat.completions.create(
#             model=LLM_MODEL,
#             messages=[
#                 {"role": "system", "content": "Anda adalah dokter yang empatik dan taat pada instruksi."},
#                 {"role": "user", "content": prompt},
#             ],
#         )
#         return resp.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"[WARN] LLM gagal, memakai saran deterministik: {e}")
#         return materi


# # ==================================================================
# # 6. FUNGSI UTAMA
# # ==================================================================
# def generate_clinical_narrative(patient_features: dict,
#                                 shap_values: dict,
#                                 risk_score: float,
#                                 risk_status: str,
#                                 use_llm: bool = True) -> dict:
#     """
#     patient_features : nilai mentah fitur pasien (age, bmi, is_smoker, dst).
#     shap_values      : nilai SHAP per fitur (log-odds) untuk pasien ini.
#     Mengembalikan dict berisi tiga narasi: ringkasan+interpretasi, dan saran.
#     """
#     total = sum(abs(v) for v in shap_values.values())
#     ranked = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)

#     def pct(v):
#         return (abs(v) / total * 100) if total > 0 else 0.0

#     kategori_risiko = "tinggi" if "high" in str(risk_status).lower() else "rendah"
#     intro = (f"Hasil analisis menunjukkan risiko hipertensi Anda sebesar "
#              f"{risk_score * 100:.1f}%, yang termasuk kategori {kategori_risiko}.")

#     # Interpretasi tiga faktor teratas: teks objektif + teks fakta.
#     interpretasi = []
#     for fitur, val in ranked[:3]:
#         p = pct(val)
#         interpretasi.append({
#             "fitur": fitur,
#             "persen": round(p, 1),
#             "objektif": _teks_objektif(fitur, val, p, patient_features),
#             "fakta": _teks_fakta(fitur, val, p),
#         })

#     # Saran gaya hidup.
#     if use_llm:
#         konteks = _ambil_konteks_rag(shap_values)
#         saran = _saran_llm(patient_features, konteks)
#     else:
#         saran = bangun_saran_deterministik(patient_features)

#     return {"ringkasan": intro, "interpretasi": interpretasi, "saran": saran}


# def format_narasi(hasil: dict) -> str:
#     """Merangkai dict keluaran menjadi teks utuh untuk ditampilkan."""
#     baris = [hasil["ringkasan"], ""]
#     for item in hasil["interpretasi"]:
#         baris.append(item["objektif"])
#         baris.append(item["fakta"])
#         baris.append("")
#     baris.append(hasil["saran"])
#     return "\n".join(baris)


# # ==================================================================
# # 7. BLOK TESTING LOKAL
# # ==================================================================
# if __name__ == "__main__":
#     print("\n--- TESTING NARASI (data individu index 0) ---\n")

#     # Individu index 0 pada dataset (pria 59 th, IMT 24,1, perokok, tidur buruk).
#     fitur_pasien = {
#         'age': 59, 'is_female': 0, 'bmi': 24.135, 'is_smoker': 1,
#         'has_diabetes': 0, 'has_high_cholesterol': 0,
#         'sleep_quality': 2, 'sleep_disturbance': 1,
#     }
#     # Nilai SHAP (log-odds) dari summary.json.
#     shap = {
#         'age': 1.0392, 'is_female': 0.09318, 'bmi': 0.16150, 'is_smoker': -0.02001,
#         'has_diabetes': -0.007571, 'has_high_cholesterol': -0.011011,
#         'sleep_quality': -0.094783, 'sleep_disturbance': 0.002837,
#     }

#     hasil = generate_clinical_narrative(
#         fitur_pasien, shap, risk_score=0.7619, risk_status="High Risk", use_llm=False
#     )
#     print(format_narasi(hasil))
#     print("\n--- SELESAI ---")


# """
# llm_service.py  -- Modul Narasi Medis Hipertensi (REVISI)

# Arsitektur tiga lapis:
#   (1) Lapisan deterministik : menerjemahkan nilai SHAP -> arah, besaran, dan
#       barometer klinis. Semua ANGKA dan ARAH dihitung di sini, bukan oleh LLM.
#   (2) Retrieval-Augmented Generation (RAG) : mengambil potongan Pedoman
#       Hipertensi Kemenkes dari ChromaDB sebagai sumber saran yang tepercaya.
#   (3) LLM (Groq / Llama-3.1) : HANYA merangkai kalimat saran menjadi bahasa
#       yang empatik; dilarang membuat angka, arah, atau anjuran baru.

# Prinsip anti-halusinasi / anti-misinformasi:
#   - Arah efek diambil dari TANDA nilai SHAP individu, bukan dari asumsi atau
#     dari korelasi fitur (yang pada ringkasan model sempat terbalik).
#   - Persentase kontribusi = share lokal |shap_i| / sum_j |shap_j| (satuan
#     log-odds, ruang aditif SHAP), dikategorikan dengan ambang 50/25/10/3.
#   - Setiap fitur diberi TINGKAT KEANDALAN (Tier A/B/C). Fitur yang polanya
#     berlawanan dengan bukti medis (is_smoker) di-override: teks fakta selalu
#     menyatakan kebenaran medis, dan arah "protektif" yang keliru TIDAK PERNAH
#     ditampilkan ke pasien.
#   - Saran gaya hidup dipicu oleh KONDISI KLINIS nyata pasien (mis. IMT >= 25,
#     perokok), bukan oleh tanda SHAP -- sehingga saran berhenti merokok tetap
#     muncul untuk perokok meski SHAP model menandainya "protektif".
# """

# import os

# # ------------------------------------------------------------------
# # Import berat (LLM & RAG) dibuat opsional agar lapisan deterministik
# # tetap dapat diimpor dan diuji tanpa dependensi eksternal.
# # ------------------------------------------------------------------
# try:
#     import chromadb
#     from chromadb.utils import embedding_functions
#     from openai import OpenAI
#     from dotenv import load_dotenv
#     _DEPS_OK = True
# except Exception:  # pragma: no cover
#     _DEPS_OK = False


# # ==================================================================
# # 0. KONFIGURASI KLIEN (Groq via SDK OpenAI + ChromaDB)
# # ==================================================================
# client = None
# chroma_client = None
# sentence_transformer_ef = None
# LLM_MODEL = "llama-3.1-8b-instant"

# if _DEPS_OK:
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     ENV_PATH = os.path.join(BASE_DIR, ".env")
#     load_dotenv(dotenv_path=ENV_PATH, override=True)

#     GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#     if GROQ_API_KEY:
#         client = OpenAI(
#             api_key=GROQ_API_KEY,
#             base_url="https://api.groq.com/openai/v1",
#         )

#     try:
#         DB_DIR = os.path.join(BASE_DIR, "chroma_db")
#         chroma_client = chromadb.PersistentClient(path=DB_DIR)
#         sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
#             model_name="all-MiniLM-L6-v2"
#         )
#     except Exception as e:  # pragma: no cover
#         print(f"[WARN] ChromaDB tidak siap: {e}")


# # ==================================================================
# # 1. KAMUS FITUR
# # ==================================================================
# # Kueri RAG per fitur (menuntun retriever ke bagian pedoman yang relevan).
# FEATURE_MAP = {
#     'age': 'risiko kardiovaskular pada lansia, penuaan pembuluh darah',
#     'is_female': 'risiko hipertensi menurut jenis kelamin, menopause, hormon',
#     'bmi': 'indeks massa tubuh, obesitas, target berat badan ideal',
#     'is_smoker': 'bahaya merokok, nikotin, kerusakan pembuluh darah, berhenti merokok',
#     'has_diabetes': 'komplikasi diabetes dan hipertensi, kontrol gula darah',
#     'has_high_cholesterol': 'kolesterol tinggi, dislipidemia, aterosklerosis, diet rendah lemak jenuh',
#     'sleep_quality': 'kualitas tidur dan tekanan darah, higiene tidur',
#     'sleep_disturbance': 'gangguan tidur, insomnia, dan tekanan darah',
# }

# # Nama tampilan untuk narasi ke pasien.
# NAME_MAP = {
#     'age': 'usia',
#     'is_female': 'jenis kelamin',
#     'bmi': 'indeks massa tubuh (IMT)',
#     'is_smoker': 'kebiasaan merokok',
#     'has_diabetes': 'riwayat diabetes',
#     'has_high_cholesterol': 'riwayat kolesterol tinggi',
#     'sleep_quality': 'kualitas tidur',
#     'sleep_disturbance': 'gangguan tidur',
# }

# # Fitur yang tidak dapat diubah (hanya diakui, tidak dijadikan target saran).
# NON_MODIFIABLE = {"age", "is_female"}

# # Ambang netral dari analisis SHAP (lihat subbab Implementasi XAI).
# AMBANG_NETRAL = {"age": 43.0, "bmi": 24.0}


# # ==================================================================
# # 2. TINGKAT KEANDALAN FITUR (Tier A / B / C)
# # ==================================================================
# #   A = arah model searah & konsisten dengan bukti medis  -> fakta penuh
# #   B = magnitudo kecil / arah spesifik-model              -> fakta di-hedge
# #   C = arah model BERLAWANAN dengan bukti medis           -> override fakta
# TIER = {
#     'age': 'A', 'bmi': 'A', 'has_diabetes': 'A', 'has_high_cholesterol': 'A',
#     'is_female': 'B', 'sleep_quality': 'B', 'sleep_disturbance': 'B',
#     'is_smoker': 'C',
# }

# # Teks #2 (alasan berbasis fakta). Dipilih oleh TANDA SHAP untuk Tier A,
# # di-hedge untuk Tier B, dan di-override (selalu benar secara medis) untuk C.
# FAKTA_MEDIS = {
#     'age': {
#         'naik': "Bertambahnya usia meningkatkan risiko hipertensi karena elastisitas dinding arteri menurun dan kekakuan pembuluh darah meningkat seiring waktu.",
#         'turun': "Usia yang lebih muda cenderung bersifat protektif; risiko hipertensi umumnya baru meningkat seiring bertambahnya usia.",
#     },
#     'bmi': {
#         'naik': "Indeks massa tubuh yang lebih tinggi meningkatkan risiko hipertensi karena menambah beban kerja jantung dan resistensi pembuluh darah.",
#         'turun': "Menjaga indeks massa tubuh pada rentang sehat membantu tekanan darah tetap terkendali.",
#     },
#     'has_diabetes': {
#         'naik': "Riwayat diabetes meningkatkan risiko hipertensi melalui gangguan pada fungsi metabolik dan pembuluh darah.",
#         'turun': "Tidak adanya riwayat diabetes merupakan kondisi yang menguntungkan bagi kendali tekanan darah.",
#     },
#     'has_high_cholesterol': {
#         'naik': "Kolesterol tinggi meningkatkan risiko hipertensi karena mendorong pengerasan dan penyempitan pembuluh darah (aterosklerosis).",
#         'turun': "Kadar kolesterol yang normal mendukung kelancaran aliran darah dan kendali tekanan darah.",
#     },
#     # Tier B -- hedge, tanpa klaim sebab-akibat yang kuat.
#     'is_female': {
#         'hedge': "Pola risiko hipertensi menurut jenis kelamin bervariasi menurut usia dan status hormonal, sehingga tidak dijadikan dasar rekomendasi.",
#     },
#     'sleep_quality': {
#         'hedge': "Kaitan kualitas tidur dengan tekanan darah pada data ini belum konsisten dengan literatur, sehingga tidak dijadikan klaim sebab-akibat.",
#     },
#     'sleep_disturbance': {
#         'hedge': "Gangguan tidur dikaitkan dengan tekanan darah pada sebagian literatur, namun pada model ini pengaruhnya kecil sehingga disampaikan secara berhati-hati.",
#     },
#     # Tier C -- override: SELALU nyatakan fakta medis yang benar.
#     'is_smoker': {
#         'override': "Menjadi perokok aktif maupun mantan perokok cenderung meningkatkan risiko hipertensi karena nikotin menyempitkan pembuluh darah serta meningkatkan tekanan dan kekakuan arteri.",
#     },
# }


# # ==================================================================
# # 3. FUNGSI BANTU DETERMINISTIK
# # ==================================================================
# def _cap(teks: str) -> str:
#     """Kapitalisasi huruf pertama saja (menjaga singkatan seperti IMT)."""
#     return teks[:1].upper() + teks[1:] if teks else teks


# def kategori_magnitudo(persen: float) -> str:
#     """Ambang 50/25/10/3 (share kontribusi lokal)."""
#     if persen >= 50:
#         return "dominan"
#     if persen >= 25:
#         return "kuat"
#     if persen >= 10:
#         return "moderat"
#     if persen >= 3:
#         return "kecil"
#     return "sangat kecil"


# def klasifikasi_bmi(bmi: float) -> str:
#     """Barometer IMT menurut standar Asia-Pasifik WHO."""
#     if bmi < 18.5:
#         return "di bawah normal (kurus)"
#     if bmi < 23.0:
#         return "normal"
#     if bmi < 27.5:
#         return "berlebih (overweight)"
#     return "obesitas"



# def _posisi_netral(fitur: str, nilai: float) -> str:
#     amb = AMBANG_NETRAL.get(fitur)
#     if amb is None or nilai is None:
#         return ""
#     return "di bawah" if nilai < amb else "di atas"


# def _deskripsi_nilai(fitur: str, pf: dict) -> str:
#     """Frasa nilai + barometer untuk teks objektif tiap fitur."""
#     v = pf.get(fitur)
#     if fitur == 'age' and v is not None:
#         return f"yakni {int(round(v))} tahun yang berada {_posisi_netral('age', v)} titik netral model (sekitar 43 tahun),"
#     if fitur == 'bmi' and v is not None:
#         return f"sebesar {v:.1f} yang tergolong {klasifikasi_bmi(v)},"
#     if fitur == 'is_female':
#         return "(laki-laki)" if v == 0 else "(perempuan)" if v == 1 else ""
#     if fitur == 'is_smoker':
#         return "(perokok)" if v == 1 else "(bukan perokok)" if v == 0 else ""
#     if fitur == 'has_diabetes':
#         return "(ada riwayat)" if v == 1 else "(tidak ada riwayat)" if v == 0 else ""
#     if fitur == 'has_high_cholesterol':
#         return "(ada riwayat)" if v == 1 else "(tidak ada riwayat)" if v == 0 else ""
#     if fitur in ('sleep_quality', 'sleep_disturbance') and v is not None:
#         return f"(skor {int(round(v))} dari 5)"
#     return ""


# def _teks_fakta(fitur: str, shap_val: float, persen: float) -> str:
#     """Teks #2 sesuai tier + tanda SHAP + guardrail magnitudo."""
#     tier = TIER.get(fitur, 'B')

#     # Tier C: is_smoker -> selalu fakta benar (override), abaikan tanda SHAP.
#     if tier == 'C':
#         return FAKTA_MEDIS[fitur]['override']

#     # Guardrail: kontribusi sangat kecil -> jangan buat klaim tegas.
#     if persen < 3:
#         return "Pengaruhnya terhadap prediksi ini dapat diabaikan."

#     if tier == 'A':
#         kunci = 'naik' if shap_val > 0 else 'turun'
#         return FAKTA_MEDIS[fitur][kunci]

#     # Tier B -> hedge.
#     return FAKTA_MEDIS[fitur].get('hedge', "Pengaruhnya kecil dan disampaikan secara berhati-hati.")


# def _teks_objektif(fitur: str, shap_val: float, persen: float, pf: dict) -> str:
#     """Teks #1 objektif: nama + nilai/barometer + kategori + arah + persen."""
#     nama = NAME_MAP.get(fitur, fitur)
#     desk = _deskripsi_nilai(fitur, pf)
#     kategori = kategori_magnitudo(persen)

#     # Override arah untuk is_smoker: JANGAN pernah menyebut "menurunkan".
#     if TIER.get(fitur) == 'C' or persen < 3:
#         return (f"{_cap(nama)} Anda {desk} memiliki pengaruh yang "
#                 f"dapat diabaikan pada prediksi ini (kontribusi {persen:.1f}%).")

#     arah = "meningkatkan" if shap_val > 0 else "menurunkan"
#     return (f"{_cap(nama)} Anda {desk} memiliki pengaruh {kategori} "
#             f"dalam {arah} risiko ini, dengan kontribusi {persen:.1f}% "
#             f"dari keseluruhan faktor.")


# # ==================================================================
# # 4. SARAN GAYA HIDUP (dipicu KONDISI KLINIS, bukan tanda SHAP)
# # ==================================================================
# SARAN_SNIPPET = {
#     'is_smoker':             "berhenti merokok karena memberi manfaat langsung bagi kesehatan pembuluh darah dan tekanan darah",
#     'bmi':                   "menjaga berat badan melalui penurunan kalori bertahap serta aktivitas fisik ringan hingga sedang seperti jalan kaki atau berenang",
#     'has_high_cholesterol':  "membatasi asupan lemak jenuh dan memperbanyak serat dari sayur, buah, serta biji-bijian utuh",
#     'has_diabetes':          "mengelola kadar gula darah dengan membatasi konsumsi gula dan karbohidrat olahan",
#     'sleep':                 "memperbaiki kualitas tidur dengan jadwal tidur yang teratur serta membatasi kafein dan paparan layar sebelum tidur",
# }
# # Urutan prioritas klinis (maks. 3 dipakai agar tetap 2-3 kalimat).
# PRIORITAS_SARAN = ['is_smoker', 'bmi', 'has_high_cholesterol', 'has_diabetes', 'sleep']
# SARAN_FALLBACK = "mempertahankan pola hidup sehat saat ini berupa pola makan seimbang, aktivitas fisik teratur, dan tidur yang cukup"
# SARAN_PENUTUP = "serta berkonsultasi dengan tenaga kesehatan untuk pemeriksaan dan penanganan lebih lanjut"


# def _pemicu_saran(pf: dict) -> list:
#     """Daftar faktor termodifikasi yang benar-benar tidak sehat pada pasien ini."""
#     picu = []
#     if pf.get('is_smoker') == 1:
#         picu.append('is_smoker')
#     if pf.get('bmi') is not None and pf['bmi'] >= 25:
#         picu.append('bmi')
#     if pf.get('has_high_cholesterol') == 1:
#         picu.append('has_high_cholesterol')
#     if pf.get('has_diabetes') == 1:
#         picu.append('has_diabetes')
#     sd, sq = pf.get('sleep_disturbance'), pf.get('sleep_quality')
#     if (sd is not None and sd >= 4) or (sq is not None and sq <= 2):
#         picu.append('sleep')
#     # urutkan menurut prioritas klinis, batasi 3
#     picu = [f for f in PRIORITAS_SARAN if f in picu][:3]
#     return picu


# def _gabung(items: list) -> str:
#     if len(items) > 1:
#         return ", ".join(items[:-1]) + ", dan " + items[-1]
#     return items[0] if items else ""


# def bangun_saran_deterministik(pf: dict) -> str:
#     """Saran gaya hidup 2-3 kalimat, sepenuhnya terkendali (tanpa LLM)."""
#     picu = _pemicu_saran(pf)
#     if not picu:
#         return f"Berdasarkan faktor tersebut, disarankan pasien untuk {SARAN_FALLBACK}, {SARAN_PENUTUP}."
#     isi = _gabung([SARAN_SNIPPET[f] for f in picu])
#     return f"Berdasarkan faktor tersebut, disarankan pasien untuk {isi}, {SARAN_PENUTUP}."


# # ==================================================================
# # 5. RAG + LLM (opsional; degradasi ke deterministik bila tak tersedia)
# # ==================================================================
# def transform_query(shap_values: dict) -> str:
#     parts = [FEATURE_MAP[f] for f in shap_values if f in FEATURE_MAP]
#     return " ".join(parts) if parts else "modifikasi gaya hidup untuk mencegah hipertensi"


# def _ambil_konteks_rag(shap_values: dict) -> str:
#     if chroma_client is None or sentence_transformer_ef is None:
#         return ""
#     try:
#         collection = chroma_client.get_collection(
#             name="pedoman_hipertensi", embedding_function=sentence_transformer_ef
#         )
#         res = collection.query(query_texts=[transform_query(shap_values)], n_results=2)
#         chunks = res['documents'][0] if res.get('documents') else []
#         return "\n\n".join(chunks)
#     except Exception as e:
#         print(f"[WARN] Retrieval gagal: {e}")
#         return ""


# def _saran_llm(pf: dict, konteks: str) -> str:
#     """LLM merangkai ulang saran deterministik agar lebih luwes; TIDAK menambah fakta."""
#     picu = _pemicu_saran(pf)
#     target = [NAME_MAP[f] if f != 'sleep' else 'kualitas tidur' for f in picu]
#     materi = bangun_saran_deterministik(pf)

#     if client is None:
#         return materi  # fallback deterministik

#     prompt = f"""Tugas Anda HANYA menuliskan ulang SARAN GAYA HIDUP berikut agar mengalir dan empatik,
# TANPA menambah, mengurangi, atau mengubah anjuran maupun angka apa pun.

# MATERI SARAN (sumber kebenaran, jangan diubah isinya):
# "{materi}"

# FAKTOR YANG BOLEH DISINGGUNG: {target if target else "pola hidup sehat umum"}

# DOKUMEN REFERENSI KEMENKES (boleh dipakai untuk memperjelas, dilarang mengarang di luar ini):
# {konteks if konteks else "(tidak tersedia)"}

# ATURAN WAJIB:
# 1. Mulai TEPAT dengan: "Berdasarkan faktor tersebut, disarankan pasien untuk ...".
# 2. Dilarang menyuruh mengubah usia atau jenis kelamin.
# 3. Bila menyebut natrium/garam, gunakan kata "membatasi"/"mengurangi"; DILARANG menulis "meningkatkan asupan natrium".
# 4. Bahasa Indonesia natural tanpa markdown (tanpa bintang, bullet, atau cetak tebal). Maksimal 3 kalimat.
# """
#     try:
#         resp = client.chat.completions.create(
#             model=LLM_MODEL,
#             messages=[
#                 {"role": "system", "content": "Anda adalah dokter yang empatik dan taat pada instruksi."},
#                 {"role": "user", "content": prompt},
#             ],
#         )
#         return resp.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"[WARN] LLM gagal, memakai saran deterministik: {e}")
#         return materi


# # ==================================================================
# # 6. FUNGSI UTAMA
# # ==================================================================
# def generate_clinical_narrative(patient_features: dict,
#                                 shap_values: dict,
#                                 risk_score: float,
#                                 risk_status: str,
#                                 use_llm: bool = True,
#                                 as_dict: bool = False):
#     """
#     patient_features : nilai mentah fitur pasien (age, bmi, is_smoker, dst).
#     shap_values      : nilai SHAP per fitur (log-odds) untuk pasien ini.

#     Secara default mengembalikan STRING narasi utuh (kompatibel dengan frontend
#     lama). Setel as_dict=True untuk memperoleh dict terstruktur
#     {ringkasan, interpretasi, saran} bila ingin UI yang lebih rinci.
#     """
#     total = sum(abs(v) for v in shap_values.values())
#     ranked = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)

#     def pct(v):
#         return (abs(v) / total * 100) if total > 0 else 0.0

#     kategori_risiko = "tinggi" if "high" in str(risk_status).lower() else "rendah"
#     intro = (f"Hasil analisis menunjukkan risiko hipertensi Anda sebesar "
#              f"{risk_score * 100:.1f}%, yang termasuk kategori {kategori_risiko}.")

#     # Interpretasi tiga faktor teratas: teks objektif + teks fakta.
#     interpretasi = []
#     for fitur, val in ranked[:3]:
#         p = pct(val)
#         interpretasi.append({
#             "fitur": fitur,
#             "persen": round(p, 1),
#             "objektif": _teks_objektif(fitur, val, p, patient_features),
#             "fakta": _teks_fakta(fitur, val, p),
#         })

#     # Saran gaya hidup.
#     if use_llm:
#         konteks = _ambil_konteks_rag(shap_values)
#         saran = _saran_llm(patient_features, konteks)
#     else:
#         saran = bangun_saran_deterministik(patient_features)

#     hasil = {"ringkasan": intro, "interpretasi": interpretasi, "saran": saran}
#     return hasil if as_dict else format_narasi(hasil)


# def format_narasi(hasil: dict) -> str:
#     """Merangkai dict keluaran menjadi STRING utuh, antar-paragraf dipisah baris kosong."""
#     paragraf = [hasil["ringkasan"]]
#     for item in hasil["interpretasi"]:
#         paragraf.append(f'{item["objektif"]} {item["fakta"]}')
#     paragraf.append(hasil["saran"])
#     return "\n\n".join(paragraf)


# # ==================================================================
# # 7. BLOK TESTING LOKAL
# # ==================================================================
# if __name__ == "__main__":
#     print("\n--- TESTING NARASI (data individu index 0) ---\n")

#     # Individu index 0 pada dataset (pria 59 th, IMT 24,1, perokok, tidur buruk).
#     fitur_pasien = {
#         'age': 59, 'is_female': 0, 'bmi': 24.135, 'is_smoker': 1,
#         'has_diabetes': 0, 'has_high_cholesterol': 0,
#         'sleep_quality': 2, 'sleep_disturbance': 1,
#     }
#     # Nilai SHAP (log-odds) dari summary.json.
#     shap = {
#         'age': 1.0392, 'is_female': 0.09318, 'bmi': 0.16150, 'is_smoker': -0.02001,
#         'has_diabetes': -0.007571, 'has_high_cholesterol': -0.011011,
#         'sleep_quality': -0.094783, 'sleep_disturbance': 0.002837,
#     }

#     hasil = generate_clinical_narrative(
#         fitur_pasien, shap, risk_score=0.7619, risk_status="High Risk", use_llm=False
#     )
#     print(hasil)  # default: STRING (kompatibel dengan frontend)
#     print("\n--- SELESAI ---")



"""
llm_service.py  -- Modul Narasi Medis Hipertensi (REVISI, tanpa RAG)

Arsitektur dua lapis:
  (1) Lapisan deterministik : menerjemahkan nilai SHAP -> arah, besaran, dan
      barometer klinis. Semua ANGKA dan ARAH dihitung di sini, bukan oleh LLM.
  (2) LLM (Groq / Llama-3.1) : HANYA merangkai kalimat saran menjadi bahasa
      yang empatik (prompt engineering); dilarang membuat angka, arah, atau
      anjuran baru. Materi saran sudah ditetapkan secara deterministik.

Prinsip anti-halusinasi / anti-misinformasi:
  - Arah efek diambil dari TANDA nilai SHAP individu, bukan dari asumsi atau
    dari korelasi fitur (yang pada ringkasan model sempat terbalik).
  - Persentase kontribusi = share lokal |shap_i| / sum_j |shap_j| (satuan
    log-odds, ruang aditif SHAP), dikategorikan dengan ambang 50/25/10/3.
  - Setiap fitur diberi TINGKAT KEANDALAN (Tier A/B/C). Fitur yang polanya
    berlawanan dengan bukti medis (is_smoker) di-override: teks fakta selalu
    menyatakan kebenaran medis, dan arah "protektif" yang keliru TIDAK PERNAH
    ditampilkan ke pasien.
  - Saran gaya hidup dipicu oleh KONDISI KLINIS nyata pasien (mis. IMT >= 25,
    perokok), bukan oleh tanda SHAP -- sehingga saran berhenti merokok tetap
    muncul untuk perokok meski SHAP model menandainya "protektif".
"""

import os

# ------------------------------------------------------------------
# Import LLM dibuat opsional agar lapisan deterministik tetap dapat
# diimpor dan diuji tanpa dependensi eksternal.
# ------------------------------------------------------------------
try:
    from openai import OpenAI
    from dotenv import load_dotenv
    _DEPS_OK = True
except Exception:  # pragma: no cover
    _DEPS_OK = False


# ==================================================================
# 0. KONFIGURASI KLIEN (Groq via SDK OpenAI)
# ==================================================================
client = None
LLM_MODEL = "llama-3.1-8b-instant"

if _DEPS_OK:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ENV_PATH = os.path.join(BASE_DIR, ".env")
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if GROQ_API_KEY:
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )


# ==================================================================
# 1. KAMUS FITUR
# ==================================================================
# Nama tampilan untuk narasi ke pasien.
NAME_MAP = {
    'age': 'usia',
    'is_female': 'jenis kelamin',
    'bmi': 'indeks massa tubuh (IMT)',
    'is_smoker': 'kebiasaan merokok',
    'has_diabetes': 'riwayat diabetes',
    'has_high_cholesterol': 'riwayat kolesterol tinggi',
    'sleep_quality': 'kualitas tidur',
    'sleep_disturbance': 'gangguan tidur',
}

# Fitur yang tidak dapat diubah (hanya diakui, tidak dijadikan target saran).
NON_MODIFIABLE = {"age", "is_female"}

# Ambang netral dari analisis SHAP (lihat subbab Implementasi XAI).
AMBANG_NETRAL = {"age": 43.0, "bmi": 24.0}


# ==================================================================
# 2. TINGKAT KEANDALAN FITUR (Tier A / B / C)
# ==================================================================
#   A = arah model searah & konsisten dengan bukti medis  -> fakta penuh
#   B = magnitudo kecil / arah spesifik-model              -> fakta di-hedge
#   C = arah model BERLAWANAN dengan bukti medis           -> override fakta
TIER = {
    'age': 'A', 'bmi': 'A', 'has_diabetes': 'A', 'has_high_cholesterol': 'A',
    'is_female': 'B', 'sleep_quality': 'B', 'sleep_disturbance': 'B',
    'is_smoker': 'C',
}

# Teks #2 (alasan berbasis fakta). Dipilih oleh TANDA SHAP untuk Tier A,
# di-hedge untuk Tier B, dan di-override (selalu benar secara medis) untuk C.
FAKTA_MEDIS = {
    'age': {
        'naik': "Bertambahnya usia meningkatkan risiko hipertensi karena elastisitas dinding arteri menurun dan kekakuan pembuluh darah meningkat seiring waktu.",
        'turun': "Usia yang lebih muda cenderung bersifat protektif; risiko hipertensi umumnya baru meningkat seiring bertambahnya usia.",
    },
    'bmi': {
        'naik': "Indeks massa tubuh yang lebih tinggi meningkatkan risiko hipertensi karena menambah beban kerja jantung dan resistensi pembuluh darah.",
        'turun': "Menjaga indeks massa tubuh pada rentang sehat membantu tekanan darah tetap terkendali.",
    },
    'has_diabetes': {
        'naik': "Riwayat diabetes meningkatkan risiko hipertensi melalui gangguan pada fungsi metabolik dan pembuluh darah.",
        'turun': "Tidak adanya riwayat diabetes merupakan kondisi yang menguntungkan bagi kendali tekanan darah.",
    },
    'has_high_cholesterol': {
        'naik': "Kolesterol tinggi meningkatkan risiko hipertensi karena mendorong pengerasan dan penyempitan pembuluh darah (aterosklerosis).",
        'turun': "Kadar kolesterol yang normal mendukung kelancaran aliran darah dan kendali tekanan darah.",
    },
    # Tier B -- hedge, tanpa klaim sebab-akibat yang kuat.
    'is_female': {
        'hedge': "Pola risiko hipertensi menurut jenis kelamin bervariasi menurut usia dan status hormonal, sehingga tidak dijadikan dasar rekomendasi.",
    },
    'sleep_quality': {
        'hedge': "Kaitan kualitas tidur dengan tekanan darah pada data ini belum konsisten dengan literatur, sehingga tidak dijadikan klaim sebab-akibat.",
    },
    'sleep_disturbance': {
        'hedge': "Gangguan tidur dikaitkan dengan tekanan darah pada sebagian literatur, namun pada model ini pengaruhnya kecil sehingga disampaikan secara berhati-hati.",
    },
    # Tier C -- override: SELALU nyatakan fakta medis yang benar.
    'is_smoker': {
        'override': "Menjadi perokok aktif maupun mantan perokok cenderung meningkatkan risiko hipertensi karena nikotin menyempitkan pembuluh darah serta meningkatkan tekanan dan kekakuan arteri.",
    },
}


# ==================================================================
# 3. FUNGSI BANTU DETERMINISTIK
# ==================================================================
def _cap(teks: str) -> str:
    """Kapitalisasi huruf pertama saja (menjaga singkatan seperti IMT)."""
    return teks[:1].upper() + teks[1:] if teks else teks


def kategori_magnitudo(persen: float) -> str:
    """Ambang 50/25/10/3 (share kontribusi lokal)."""
    if persen >= 50:
        return "dominan"
    if persen >= 25:
        return "kuat"
    if persen >= 10:
        return "moderat"
    if persen >= 3:
        return "kecil"
    return "sangat kecil"


def klasifikasi_bmi(bmi: float) -> str:
    """Barometer IMT menurut standar WHO."""
    if bmi < 18.5:
        return "di bawah normal (kurus)"
    if bmi < 25:
        return "normal"
    if bmi < 30:
        return "berlebih (overweight)"
    return "obesitas"


def _posisi_netral(fitur: str, nilai: float) -> str:
    amb = AMBANG_NETRAL.get(fitur)
    if amb is None or nilai is None:
        return ""
    return "di bawah" if nilai < amb else "di atas"


def _deskripsi_nilai(fitur: str, pf: dict) -> str:
    """Frasa nilai + barometer untuk teks objektif tiap fitur."""
    v = pf.get(fitur)
    if fitur == 'age' and v is not None:
        return f"yakni {int(round(v))} tahun,"
    if fitur == 'bmi' and v is not None:
        return f"sebesar {v:.1f} yang tergolong {klasifikasi_bmi(v)},"
    if fitur == 'is_female':
        return "(laki-laki)" if v == 0 else "(perempuan)" if v == 1 else ""
    if fitur == 'is_smoker':
        return "(perokok)" if v == 1 else "(bukan perokok)" if v == 0 else ""
    if fitur == 'has_diabetes':
        return "(ada riwayat)" if v == 1 else "(tidak ada riwayat)" if v == 0 else ""
    if fitur == 'has_high_cholesterol':
        return "(ada riwayat)" if v == 1 else "(tidak ada riwayat)" if v == 0 else ""
    if fitur in ('sleep_quality', 'sleep_disturbance') and v is not None:
        return f"(skor {int(round(v))} dari 5)"
    return ""


def _teks_fakta(fitur: str, shap_val: float, persen: float) -> str:
    """Teks #2 sesuai tier + tanda SHAP + guardrail magnitudo."""
    tier = TIER.get(fitur, 'B')

    # Tier C: is_smoker -> selalu fakta benar (override), abaikan tanda SHAP.
    if tier == 'C':
        return FAKTA_MEDIS[fitur]['override']

    # Guardrail: kontribusi sangat kecil -> jangan buat klaim tegas.
    if persen < 3:
        return "Pengaruhnya terhadap prediksi ini dapat diabaikan."

    if tier == 'A':
        kunci = 'naik' if shap_val > 0 else 'turun'
        return FAKTA_MEDIS[fitur][kunci]

    # Tier B -> hedge.
    return FAKTA_MEDIS[fitur].get('hedge', "Pengaruhnya kecil dan disampaikan secara berhati-hati.")


def _teks_objektif(fitur: str, shap_val: float, persen: float, pf: dict) -> str:
    """Teks #1 objektif: nama + nilai/barometer + kategori + arah + persen."""
    nama = NAME_MAP.get(fitur, fitur)
    desk = _deskripsi_nilai(fitur, pf)
    kategori = kategori_magnitudo(persen)

    # Override arah untuk is_smoker: JANGAN pernah menyebut "menurunkan".
    if TIER.get(fitur) == 'C' or persen < 3:
        return (f"{_cap(nama)} Anda {desk} memiliki pengaruh yang "
                f"dapat diabaikan pada prediksi ini (kontribusi {persen:.1f}%).")

    arah = "meningkatkan" if shap_val > 0 else "menurunkan"
    return (f"{_cap(nama)} Anda {desk} memiliki pengaruh {kategori} "
            f"dalam {arah} risiko ini, dengan kontribusi {persen:.1f}% "
            f"dari keseluruhan faktor.")


# ==================================================================
# 4. SARAN GAYA HIDUP (dipicu KONDISI KLINIS, bukan tanda SHAP)
# ==================================================================
SARAN_SNIPPET = {
    'is_smoker':             "berhenti merokok karena memberi manfaat langsung bagi kesehatan pembuluh darah dan tekanan darah",
    'bmi':                   "menjaga berat badan melalui penurunan kalori bertahap serta aktivitas fisik ringan hingga sedang seperti jalan kaki atau berenang",
    'has_high_cholesterol':  "membatasi asupan lemak jenuh dan memperbanyak serat dari sayur, buah, serta biji-bijian utuh",
    'has_diabetes':          "mengelola kadar gula darah dengan membatasi konsumsi gula dan karbohidrat olahan",
    'sleep':                 "memperbaiki kualitas tidur dengan jadwal tidur yang teratur serta membatasi kafein dan paparan layar sebelum tidur",
}
# Urutan prioritas klinis (maks. 3 dipakai agar tetap 2-3 kalimat).
PRIORITAS_SARAN = ['is_smoker', 'bmi', 'has_high_cholesterol', 'has_diabetes', 'sleep']
SARAN_FALLBACK = "mempertahankan pola hidup sehat saat ini berupa pola makan seimbang, aktivitas fisik teratur, dan tidur yang cukup"
SARAN_PENUTUP = "serta berkonsultasi dengan tenaga kesehatan untuk pemeriksaan dan penanganan lebih lanjut"


def _pemicu_saran(pf: dict) -> list:
    """Daftar faktor termodifikasi yang benar-benar tidak sehat pada pasien ini."""
    picu = []
    if pf.get('is_smoker') == 1:
        picu.append('is_smoker')
    if pf.get('bmi') is not None and pf['bmi'] >= 25:
        picu.append('bmi')
    if pf.get('has_high_cholesterol') == 1:
        picu.append('has_high_cholesterol')
    if pf.get('has_diabetes') == 1:
        picu.append('has_diabetes')
    sd, sq = pf.get('sleep_disturbance'), pf.get('sleep_quality')
    if (sd is not None and sd >= 4) or (sq is not None and sq <= 2):
        picu.append('sleep')
    # urutkan menurut prioritas klinis, batasi 3
    picu = [f for f in PRIORITAS_SARAN if f in picu][:3]
    return picu


def _gabung(items: list) -> str:
    if len(items) > 1:
        return ", ".join(items[:-1]) + ", dan " + items[-1]
    return items[0] if items else ""


def bangun_saran_deterministik(pf: dict) -> str:
    """Saran gaya hidup 2-3 kalimat, sepenuhnya terkendali (tanpa LLM)."""
    picu = _pemicu_saran(pf)
    if not picu:
        return f"Berdasarkan faktor tersebut, disarankan pasien untuk {SARAN_FALLBACK}, {SARAN_PENUTUP}."
    isi = _gabung([SARAN_SNIPPET[f] for f in picu])
    return f"Berdasarkan faktor tersebut, disarankan pasien untuk {isi}, {SARAN_PENUTUP}."


# ==================================================================
# 5. LLM untuk penghalusan saran (opsional; degradasi ke deterministik)
# ==================================================================
def _saran_llm(pf: dict) -> str:
    """LLM merangkai ulang saran deterministik agar lebih luwes; TIDAK menambah fakta."""
    picu = _pemicu_saran(pf)
    target = [NAME_MAP[f] if f != 'sleep' else 'kualitas tidur' for f in picu]
    materi = bangun_saran_deterministik(pf)

    if client is None:
        return materi  # fallback deterministik

    prompt = f"""Tugas Anda HANYA menuliskan ulang SARAN GAYA HIDUP berikut agar mengalir dan empatik,
TANPA menambah, mengurangi, atau mengubah anjuran maupun angka apa pun.

MATERI SARAN (sumber kebenaran, jangan diubah isinya):
"{materi}"

FAKTOR YANG BOLEH DISINGGUNG: {target if target else "pola hidup sehat umum"}

ATURAN WAJIB:
1. Mulai TEPAT dengan: "Berdasarkan faktor tersebut, disarankan pasien untuk ...".
2. Dilarang menyuruh mengubah usia atau jenis kelamin.
3. Bila menyebut natrium/garam, gunakan kata "membatasi"/"mengurangi"; DILARANG menulis "meningkatkan asupan natrium".
4. Bahasa Indonesia natural tanpa markdown (tanpa bintang, bullet, atau cetak tebal). Maksimal 3 kalimat.
"""
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Anda adalah dokter yang empatik dan taat pada instruksi."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[WARN] LLM gagal, memakai saran deterministik: {e}")
        return materi


# ==================================================================
# 6. FUNGSI UTAMA
# ==================================================================
def generate_clinical_narrative(patient_features: dict,
                                shap_values: dict,
                                risk_score: float,
                                risk_status: str,
                                use_llm: bool = True,
                                as_dict: bool = False):
    """
    patient_features : nilai mentah fitur pasien (age, bmi, is_smoker, dst).
    shap_values      : nilai SHAP per fitur (log-odds) untuk pasien ini.

    Secara default mengembalikan STRING narasi utuh (kompatibel dengan frontend
    lama). Setel as_dict=True untuk memperoleh dict terstruktur
    {ringkasan, interpretasi, saran} bila ingin UI yang lebih rinci.
    """
    total = sum(abs(v) for v in shap_values.values())
    ranked = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)

    def pct(v):
        return (abs(v) / total * 100) if total > 0 else 0.0

    kategori_risiko = "tinggi" if "high" in str(risk_status).lower() else "rendah"
    intro = (f"Hasil analisis menunjukkan risiko hipertensi Anda sebesar "
             f"{risk_score * 100:.1f}%, yang termasuk kategori {kategori_risiko}.")

    # Interpretasi tiga faktor teratas: teks objektif + teks fakta.
    interpretasi = []
    for fitur, val in ranked[:3]:
        p = pct(val)
        interpretasi.append({
            "fitur": fitur,
            "persen": round(p, 1),
            "objektif": _teks_objektif(fitur, val, p, patient_features),
            "fakta": _teks_fakta(fitur, val, p),
        })

    # Saran gaya hidup.
    if use_llm:
        saran = _saran_llm(patient_features)
    else:
        saran = bangun_saran_deterministik(patient_features)

    hasil = {"ringkasan": intro, "interpretasi": interpretasi, "saran": saran}
    return hasil if as_dict else format_narasi(hasil)


def format_narasi(hasil: dict) -> str:
    """Merangkai dict keluaran menjadi STRING utuh, antar-paragraf dipisah baris kosong."""
    paragraf = [hasil["ringkasan"]]
    for item in hasil["interpretasi"]:
        paragraf.append(f'{item["objektif"]} {item["fakta"]}')
    paragraf.append(hasil["saran"])
    return "\n\n".join(paragraf)


# ==================================================================
# 7. BLOK TESTING LOKAL
# ==================================================================
if __name__ == "__main__":
    print("\n--- TESTING NARASI (data individu index 0) ---\n")

    # Individu index 0 pada dataset (pria 59 th, IMT 24,1, perokok, tidur buruk).
    fitur_pasien = {
        'age': 59, 'is_female': 0, 'bmi': 24.135, 'is_smoker': 1,
        'has_diabetes': 0, 'has_high_cholesterol': 0,
        'sleep_quality': 2, 'sleep_disturbance': 1,
    }
    # Nilai SHAP (log-odds) dari summary.json.
    shap = {
        'age': 1.0392, 'is_female': 0.09318, 'bmi': 0.16150, 'is_smoker': -0.02001,
        'has_diabetes': -0.007571, 'has_high_cholesterol': -0.011011,
        'sleep_quality': -0.094783, 'sleep_disturbance': 0.002837,
    }

    hasil = generate_clinical_narrative(
        fitur_pasien, shap, risk_score=0.7619, risk_status="High Risk", use_llm=False
    )
    print(hasil)  # default: STRING (kompatibel dengan frontend)
    print("\n--- SELESAI ---")