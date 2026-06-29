#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
 AUDIT METADATA IFLS (.dta)
==============================================================================
Tujuan: buka setiap file .dta dan cetak fakta yang dibutuhkan untuk MEMVERIFIKASI
semua tebakan semantik di pipeline, langsung dari data Anda sendiri:

    1. Bentuk file (jumlah baris & kolom)
    2. Cek "grain" (unit analisis): apakah pidlink UNIK (1 baris/orang) atau
       file LOOP (banyak baris/orang -> wajib di-collapse sebelum merge)
    3. Daftar SEMUA kolom + label variabel
    4. Label nilai (value labels), mis. 1='Laki-laki', 3='Perempuan'
    5. Distribusi nilai (value_counts) untuk kolom fokus + semua kolom '*type',
       supaya Anda bisa MELIHAT kode yang benar-benar ada:
         - cdtype : apakah benar ada kode 'H' = Diabetes?
         - fmtype : kode mie instan yang benar (12? 'L'? lainnya?)
         - pstype : apakah benar A..J? Apakah section ini memang CES-D?
         - ba15_a / ba15_b : sebenarnya variabel apa (orang tua atau anak)?
         - us07* : mana sistolik, mana diastolik?

Cara pakai:
    python audit_metadata_ifls.py                      # audit seluruh folder 'data'
    python audit_metadata_ifls.py data/hh14_bus_dta/bus_us.dta   # satu file
    python audit_metadata_ifls.py data/hh14_b3b_dta             # satu folder
    python audit_metadata_ifls.py --no-values         # metadata saja (lebih cepat)
    python audit_metadata_ifls.py --out audit.txt     # ganti nama file output

Hasil dicetak ke layar DAN disimpan ke file teks (default: metadata_audit.txt).

Butuh: pip install pyreadstat
==============================================================================
"""
import os
import sys
import argparse

try:
    import pyreadstat
except ImportError:
    sys.exit("[FATAL] pyreadstat belum terpasang. Jalankan: pip install pyreadstat pandas")

# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------
DATA_DIR = "data"

# Kolom yang nilainya ingin diperiksa mendalam (value_counts), per nama file.
# Nama boleh huruf kecil; pencocokan dilakukan case-insensitive.
FOCUS_COLUMNS = {
    "ptrack.dta":  ["pidlink"],
    "b3a_cov.dta": ["sex", "age"],
    "bk_ar1.dta":  ["ar07", "ar09", "ar10", "ar11"],  # roster Book K: sex, umur, pid ayah, pid ibu
    "b3b_km.dta":  ["km01a", "km04", "km08"],
    "b3b_ak1.dta": ["ak02", "ak05"],
    "b3b_ak2.dta": ["ak07"],
    "b3b_cd3.dta": ["cdtype", "cd01type", "cd05"],
    "b3b_ps.dta":  ["pstype", "ps01"],
    "b3b_fm2.dta": ["fmtype", "fm02", "fm03"],
    "bus_us.dta":  ["us04", "us06", "us06a", "us07a1", "us07a2"],
    "b3b_ba1.dta": ["ba03", "ba15", "ba15_a", "ba15_b", "ba31", "ba42"],
}

ID_CANDIDATES = ["pidlink", "hhid14", "pid14", "hhid", "pid"]

MAX_LABELS = 40        # batas jumlah label nilai yang ditampilkan per kolom
MAX_VALUE_COUNTS = 25  # batas jumlah baris value_counts per kolom fokus


class Tee:
    """Tulis baris ke layar DAN ke file sekaligus."""
    def __init__(self, fh):
        self.fh = fh

    def __call__(self, line=""):
        print(line)
        self.fh.write(line + "\n")


def find_dta_files(targets):
    files = []
    for t in targets:
        if os.path.isdir(t):
            for root, _, names in os.walk(t):
                for n in sorted(names):
                    if n.lower().endswith(".dta"):
                        files.append(os.path.join(root, n))
        elif os.path.isfile(t) and t.lower().endswith(".dta"):
            files.append(t)
        else:
            print(f"[SKIP] Bukan .dta / tidak ditemukan: {t}")
    return files


def read_meta(path):
    """Baca metadata saja (cepat), dengan fallback encoding latin1."""
    try:
        _, meta = pyreadstat.read_dta(path, metadataonly=True)
    except Exception:
        _, meta = pyreadstat.read_dta(path, metadataonly=True, encoding="latin1")
    return meta


def read_columns(path, cols):
    """Baca sebagian kolom saja (cepat), dengan fallback encoding latin1."""
    try:
        df, _ = pyreadstat.read_dta(path, usecols=cols)
    except Exception:
        df, _ = pyreadstat.read_dta(path, usecols=cols, encoding="latin1")
    return df


def actual_name(meta, wanted):
    """Cari nama kolom asli secara case-insensitive; None jika tidak ada."""
    low = {c.lower(): c for c in meta.column_names}
    return low.get(wanted.lower())


def audit_file(path, out, show_values=True):
    base = os.path.basename(path).lower()
    out("")
    out("=" * 78)
    out(f"FILE: {path}")
    out("=" * 78)

    try:
        meta = read_meta(path)
    except Exception as e:
        out(f"  [ERROR] Gagal membaca metadata: {e}")
        return

    out(f"  Bentuk : {meta.number_rows} baris x {meta.number_columns} kolom")

    # --- Cek grain / unit analisis ---
    id_col = next((actual_name(meta, c) for c in ID_CANDIDATES if actual_name(meta, c)), None)
    if id_col:
        try:
            df_id = read_columns(path, [id_col])
            n_unique = df_id[id_col].nunique(dropna=True)
            n_rows = len(df_id)
            if n_unique == n_rows:
                grain = "PERSON-LEVEL (1 baris/orang) -> aman di-LEFT JOIN langsung"
            else:
                grain = "LOOP/MULTI-ROW -> WAJIB di-collapse ke 1 baris/orang SEBELUM merge"
            out(f"  Grain  : id='{id_col}'  unik={n_unique}  baris={n_rows}  ->  {grain}")
        except Exception as e:
            out(f"  Grain  : gagal dihitung ({e})")
    else:
        out("  Grain  : tidak ada kolom id (pidlink/hhid/pid) terdeteksi")

    # --- Daftar kolom + label variabel + label nilai ---
    labels = meta.column_names_to_labels or {}
    vlabels = getattr(meta, "variable_value_labels", {}) or {}
    types = getattr(meta, "readstat_variable_types", {}) or {}

    out("")
    out("  KOLOM | [tipe] LABEL VARIABEL")
    out("  " + "-" * 74)
    for col in meta.column_names:
        out(f"  {col:<14} [{types.get(col, '')}] {labels.get(col, '') or ''}")
        if col in vlabels:
            items = list(vlabels[col].items())
            decoded = ", ".join(f"{k}={v}" for k, v in items[:MAX_LABELS])
            more = "" if len(items) <= MAX_LABELS else f"  ...(+{len(items) - MAX_LABELS} lagi)"
            out(f"                 nilai: {decoded}{more}")

    # --- Distribusi nilai untuk kolom fokus + semua kolom '*type' ---
    if not show_values:
        return

    focus = list(FOCUS_COLUMNS.get(base, []))
    for c in meta.column_names:
        if c.lower().endswith("type"):
            focus.append(c)

    # resolusi ke nama asli + buang duplikat, pertahankan urutan
    focus_actual = []
    for f in focus:
        a = actual_name(meta, f)
        if a and a not in focus_actual:
            focus_actual.append(a)

    if not focus_actual:
        return

    out("")
    out("  DISTRIBUSI NILAI (untuk verifikasi kode yang benar-benar ada)")
    out("  " + "-" * 74)
    try:
        df = read_columns(path, focus_actual)
    except Exception as e:
        out(f"  [ERROR] gagal membaca kolom fokus: {e}")
        return

    for col in focus_actual:
        out(f"  >> {col}   (label: {labels.get(col, '') or ''})")
        vc = df[col].value_counts(dropna=False).head(MAX_VALUE_COUNTS)
        code_map = vlabels.get(col, {})
        for val, cnt in vc.items():
            lbl = code_map.get(val, "")
            lbl = f" ({lbl})" if lbl else ""
            out(f"        {repr(val)}{lbl}: {cnt}")
        out("")


def main():
    ap = argparse.ArgumentParser(description="Audit metadata file .dta IFLS")
    ap.add_argument("targets", nargs="*",
                    help="File .dta atau folder (default: folder 'data')")
    ap.add_argument("--out", default="metadata_audit.txt",
                    help="File output teks (default: metadata_audit.txt)")
    ap.add_argument("--no-values", action="store_true",
                    help="Lewati value_counts (metadata saja, lebih cepat)")
    args = ap.parse_args()

    targets = args.targets if args.targets else [DATA_DIR]
    files = find_dta_files(targets)
    if not files:
        sys.exit(f"[FATAL] Tidak ada file .dta ditemukan di: {targets}")

    with open(args.out, "w", encoding="utf-8") as fh:
        out = Tee(fh)
        out(f"AUDIT METADATA IFLS - {len(files)} file ditemukan")
        for p in files:
            audit_file(p, out, show_values=not args.no_values)
        out("")
        out("=" * 78)
        out("CHECKLIST VERIFIKASI:")
        out("  [ ] Grain tiap file  -> yang 'LOOP' wajib di-collapse sebelum merge")
        out("  [ ] cdtype           -> apakah benar ada kode 'H' = Diabetes?")
        out("  [ ] cd05             -> apa labelnya? (diagnosis? masih sakit? umur?)")
        out("  [ ] fmtype           -> kode mie instan yang benar")
        out("  [ ] pstype + ps01    -> benarkah A..J & section ini CES-D? (CES-D resmi: section KP)")
        out("  [ ] b3b_ba1 ba15_*   -> sebenarnya tentang orang tua atau anak?")
        out("  [ ] bus_us us07*     -> mana sistolik, mana diastolik?")
        out("  [ ] bk_ar1 ar07/10/11-> sumber sex/umur & pid ayah(ar10)/ibu(ar11)")
        out("=" * 78)

    print(f"\n[OK] Audit lengkap tersimpan ke: {args.out}")


if __name__ == "__main__":
    main()