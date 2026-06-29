# import pandas as pd
# # Membaca dataset
# master_df = pd.read_csv("master_dataset_raw_final.csv")
# # Variabel biner yang akan dicek
# binary_cols = [
#     "is_smoking",
#     "is_diabetes",
#     "is_high_cholesterol"
# ]

# for col in binary_cols:
#     print("=" * 60)
#     print(f"Variabel: {col}")
#     print("=" * 60)

#     # Jumlah tiap kategori (termasuk NaN)
#     print("\nJumlah setiap kategori:")
#     print(master_df[col].value_counts(dropna=False).sort_index())

#     # Persentase tiap kategori (termasuk NaN)
#     print("\nPersentase setiap kategori:")
#     print((master_df[col].value_counts(dropna=False, normalize=True)
#             .sort_index() * 100).round(2).astype(str) + " %")

#     # Kategori dominan (mengabaikan NaN)
#     dominant = master_df[col].mode(dropna=True)[0]
#     dominant_pct = (
#         master_df[col]
#         .value_counts(normalize=True, dropna=True)
#         .loc[dominant] * 100
#     )

#     print(f"\nKategori dominan (tanpa NaN): {dominant}")
#     print(f"Proporsi kategori dominan    : {dominant_pct:.2f} %")
#     print("\n")




import pandas as pd
from scipy.stats import chi2_contingency

# Membaca dataset
master_df = pd.read_csv("master_dataset_raw_final.csv")

# Membuat indikator missing
master_df["smoking_missing"] = master_df["is_smoking"].isna()

# Tabel kontingensi
table = pd.crosstab(
    master_df["smoking_missing"],
    master_df["hypertension"]
)

print("Tabel Kontingensi")
print(table)

# Uji Chi-Square
chi2, p, dof, expected = chi2_contingency(table)

print(f"\nChi-square statistic : {chi2:.4f}")
print(f"p-value              : {p:.6f}")

if p < 0.05:
    print("\nKesimpulan: Missing value BERHUBUNGAN dengan status hipertensi.")
else:
    print("\nKesimpulan: Tidak ada bukti bahwa missing value berhubungan dengan status hipertensi.")