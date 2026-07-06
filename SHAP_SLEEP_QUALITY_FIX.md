# Perbaikan Arah SHAP — Skala `sleep_quality`

**Tanggal:** 2026-07-06
**Status:** Selesai (fix diterapkan di repo frontend `hypertensai`)

---

## 1. Gejala (Symptom)

Pada kartu **Pengaruh Faktor Risiko**, faktor **"Kualitas tidur: Sangat buruk"** ditampilkan **hijau (mengurangi risiko)**. Ini tidak masuk akal secara klinis — tidur sangat buruk seharusnya *menaikkan* risiko hipertensi (merah).

Persentase pada UI dihitung dari `|SHAP| fitur / Σ|SHAP| semua fitur`, dan warna dari tanda nilai SHAP (positif = merah/naik, negatif = hijau/turun). Perhitungan ini **sudah benar**; masalahnya ada di data yang masuk ke model.

---

## 2. Akar Masalah (Root Cause)

**Skala `sleep_quality` di frontend terbalik terhadap skala yang dipakai saat training model.**

### Skala training (IFLS-5 `tdr01`, PSQI, `tdrtype=2`)
Model dilatih pada nilai mentah IFLS. Hubungan yang dipelajari bersifat monotonik — **nilai lebih tinggi = risiko hipertensi lebih tinggi**:

| sleep_quality | Rate hipertensi |
|:-:|:-:|
| 1 | 22.7% (terendah) |
| 2 | 23.8% |
| 3 | 27.1% |
| 4 | 30.9% (tertinggi) |
| 5 | 28.8% |

Agar koheren secara klinis, ini berarti pada skala IFLS: **1 = tidur terbaik, 5 = tidur terburuk**.

### Skala frontend (SEBELUM fix) — TERBALIK
`fields.ts` menetapkan `1 = "Sangat buruk"` … `5 = "Sangat baik"`. Jadi saat pengguna memilih **"Sangat buruk"**, yang terkirim ke model adalah **nilai 1**, yang oleh model dibaca sebagai *kategori tidur TERBAIK* → mendorong risiko **turun** → bar hijau.

> Catatan: `sleep_disturbance` (`tdrtype=1`) TIDAK terbalik (1 = jarang/tidak pernah … 5 = selalu, sesuai IFLS). Itu sebabnya hanya "Kualitas tidur" yang tampak salah.

---

## 3. Bukti (Reproduksi Numerik)

Model + SHAP dijalankan ulang pada pasien di screenshot (usia 42, laki-laki, BMI 21.6, perokok/diabetes/kolesterol tinggi = Ya). Hasilnya **identik** dengan UI:

| Fitur | SHAP | Share | Warna |
|---|:-:|:-:|:-:|
| bmi | −0.2666 | 27.3% | GREEN |
| has_high_cholesterol | +0.2456 | 25.1% | RED |
| has_diabetes | +0.1768 | 18.1% | RED |
| age | −0.0906 | 9.3% | GREEN |
| is_female | +0.0901 | 9.2% | RED |
| **sleep_quality** | **−0.0594** | **6.1%** | **GREEN** |
| is_smoker | −0.0434 | 4.4% | GREEN |
| sleep_disturbance | +0.0041 | 0.4% | RED |

`risk = 0.514` (= 51.4%, cocok dengan UI).

### Sweep membuktikan pembalikan arah
Menahan semua fitur lain tetap, hanya `sleep_quality` diubah:

| Nilai dikirim | SHAP | Arah |
|:-:|:-:|:-:|
| 1 | −0.059 | GREEN (turun) ← yang dikirim "Sangat buruk" SEBELUM fix |
| 2 | +0.005 | ~netral |
| 3 | −0.035 | GREEN |
| 4 | +0.039 | RED (naik) |
| 5 | +0.039 | RED (naik) ← yang SEHARUSNYA dikirim "Sangat buruk" |

Kesimpulan: setelah fix, "Sangat buruk" mengirim nilai 5 → SHAP **+0.039 (merah/naik)** — arah yang benar secara klinis.

---

## 4. Perbaikan yang Diterapkan

Tidak perlu retrain model atau ubah backend. Model setia pada data mentah IFLS; hanya kontrak label→angka di UI yang salah.

### File 1 — `hypertensai/src/lib/fields.ts`
Membalik nilai opsi `sleepQuality` agar label selaras dengan skala IFLS.

**Sebelum:**
```ts
const sleepQuality: SelectOption[] = [
  { value: 1, label: "1 — Sangat buruk" },
  { value: 2, label: "2 — Buruk" },
  { value: 3, label: "3 — Cukup" },
  { value: 4, label: "4 — Baik" },
  { value: 5, label: "5 — Sangat baik" },
];
```

**Sesudah:**
```ts
// Skala IFLS-5 `tdr01` (PSQI, tdrtype=2): 1 = kualitas terbaik ... 5 = terburuk.
// Model dilatih pada skala mentah ini (nilai tinggi = tidur makin buruk = risiko
// hipertensi makin tinggi). Label WAJIB mengikuti arah tersebut, jika terbalik
// maka "Sangat buruk" justru terkirim sebagai nilai 1 dan SHAP salah arah.
const sleepQuality: SelectOption[] = [
  { value: 1, label: "1 — Sangat baik" },
  { value: 2, label: "2 — Baik" },
  { value: 3, label: "3 — Cukup" },
  { value: 4, label: "4 — Buruk" },
  { value: 5, label: "5 — Sangat buruk" },
];
```

### File 2 — `hypertensai/src/lib/api.ts`
Heuristik mode-demo memuat asumsi terbalik yang sama.

**Sebelum:**
```ts
if (p.sleep_quality <= 2) score += 0.08;
```

**Sesudah:**
```ts
if (p.sleep_quality >= 4) score += 0.08; // skala IFLS: nilai tinggi = tidur buruk
```

---

## 5. Yang BUKAN Bug

**"Merokok: Ya" tampil hijau (−0.043, 4.4%)** adalah perilaku model yang wajar, bukan kesalahan encoding:
- `is_smoker` (0/1) sudah cocok dengan training.
- Sinyal global sangat lemah (korelasi dengan label hanya +0.017).
- Untuk pasien spesifik ini (laki-laki kurus 42 tahun) ini efek interaksi minor.

Kalau ingin merokok berperilaku lebih intuitif, itu ranah pemodelan/data — bukan perbaikan mapping.
