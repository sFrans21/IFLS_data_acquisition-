# Struktur Dataset

print("="*60)
print("DIMENSI DATASET")
print("="*60)

print(f"Jumlah Baris : {df.shape[0]}")
print(f"Jumlah Kolom : {df.shape[1]}")

print("\nInformasi Dataset:")
print(df.info())

print("\nTipe Data:")
print(df.dtypes.value_counts())
