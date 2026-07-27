import os
import chromadb
from chromadb.utils import embedding_functions

#Config path direktori
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(BASE_DIR, "data", "knowledge_base")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

print("Memulai proses Indexing Knowledge Base...")

# inisialisasi chromaDB client dan moel embedding
chroma_client = chromadb.PersistentClient(path=DB_DIR)
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Buat atau timpa koleksi database lama
collection = chroma_client.get_or_create_collection(
      name="pedoman_hipertensi",
      embedding_function=sentence_transformer_ef
)

# proses baca file dan potong teks
documents = []
metadatas = []
ids = []

#iterasi smeua file txt di dalam folder knowledge_base
for filename in os.listdir(KB_DIR):
      if filename.endswith(".txt"):
            file_path = os.path.join(KB_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                  content = file.read()
                  
                  #pemotongan sederhana berdasarkan spasi ganda antar paragraf
                  chunks = content.split("\n\n")
                  
                  for i, chunk in enumerate(chunks):
                        chunk = chunk.strip()
                        if not chunk:
                              continue

                        #ekstraksi metadata sederhaan dari baris pertama teks
                        topic_meta = "UMUM"
                        source_meta = "TIDAK_DIKETAHUI"
                        
                        if chunk.startswith("[METADATA:"):
                              lines = chunk.split("\n")
                              meta_line = lines[0]
                              #menghapus tag metadata dari teks utama agar tidak membingungkan LLM
                              clean_chunk = "\n".join(lines[1:]).strip()
                              
                              if "SUMBER=" in meta_line:
                                    source_meta =  meta_line.split("SUMBER=")[1].replace("]", "").strip()
                                    
                              documents.append(clean_chunk)
                              metadatas.append({"source": source_meta, "file": filename})
                              ids.append(f"{filename}_chunk_{i}")

                              
#memasukkan semua potongan ke dalam vector database
if documents:
      collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
      )
      print(f"Berhasil memasukkan {len(documents)} chunk teks ke dalam Vector Database!")
      print(f"Database tersimpan di: {DB_DIR}")
else:
      print("Tidak ada teks yang ditemukan utnuk diproses.")
      