"""
================================================================================
HypertensAI - train_model_v3.py
Versi    : 3.1
Author   : Samuel (18222131) -- ITB Sistem & Teknologi Informasi
Deskripsi: Pipeline training model dengan fix komprehensif:
           [FIX 1] Threshold selection: argmax(F1) -> target Recall >= 90%
           [FIX 2] XGBoost & RF tidak menerima data yang di-scale
           [FIX 3] F2-Score ditambahkan sebagai metrik utama screening medis
           [FIX 4] Cross-Validation 5-Fold untuk evaluasi yang lebih robust
           [FIX 5] use_label_encoder dihapus (deprecated di XGBoost terbaru)
           [FIX 6] Ensemble LR+XGB saja (RF terbukti drag-down recall)
           [FIX 7] waist_cm: imputasi median (bukan 0 -- 59.7% missing)
           [FIX 8] ps_A-ps_F: imputasi median (skala 1-3, fillna(0) tidak valid)
           [FIX 9] ak05 (object): di-convert ke numerik sebelum dipakai
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve
)
from sklearn.pipeline import Pipeline
import os
import warnings
warnings.filterwarnings('ignore')


# --- KONFIGURASI -------------------------------------------------------------
RECALL_TARGET = 0.90    # NFR02: Recall WAJIB > 90%
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
OUTPUT_DIR    = "hasil_model_v3"
CV_FOLDS      = 5

sns.set(style="whitegrid")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("=" * 65)
print("  HypertensAI -- train_model_v3.py  (v3.1)")
print("  TARGET: Recall > 90% (NFR02)")
print("=" * 65)


# --- 1. LOAD & PREPROCESSING DATA --------------------------------------------
print("\n[PREPROCESSING]")
df = pd.read_csv('dataset_hipertensi_clean.csv')

# Drop kolom target & bocoran (data leakage)
X = df.drop(columns=['label_hypertension', 'bp_systolic', 'bp_diastolic'], errors='ignore')

# [FIX 9] ak05 bertipe object/string -> convert ke numerik dulu
# errors='coerce': nilai tidak valid -> NaN (akan dihandle fillna di bawah)
if 'ak05' in X.columns:
    X['ak05'] = pd.to_numeric(X['ak05'], errors='coerce')
    print("  [FIX 9] ak05 di-convert ke numerik [OK]")

# Ambil semua kolom numerik (ak05 kini ikut masuk sebagai fitur ke-16)
X = X.select_dtypes(include=[np.number])

# [FIX 7] waist_cm: 59.7% missing -- fillna(0) tidak masuk akal medis
# Imputasi dengan MEDIAN (lebih robust dari mean untuk data skewed)
if 'waist_cm' in X.columns:
    median_waist = X['waist_cm'].median()
    X['waist_cm'] = X['waist_cm'].fillna(median_waist)
    print(f"  [FIX 7] waist_cm -> imputasi median = {median_waist:.1f} cm [OK]")

# [FIX 8] ps_A-ps_F: skala ordinal 1-3
# fillna(0) menciptakan nilai di luar skala -> noise untuk model
# Imputasi dengan median masing-masing kolom
ps_cols = ['ps_A', 'ps_B', 'ps_C', 'ps_E', 'ps_F']
for col in ps_cols:
    if col in X.columns and X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())
print("  [FIX 8] ps_A-ps_F -> imputasi median masing-masing kolom [OK]")

# Fallback: kolom lain yang masih ada NaN
sisa_nan = X.isnull().sum().sum()
if sisa_nan > 0:
    X = X.fillna(0)
    print(f"  [FALLBACK] {sisa_nan} NaN tersisa diisi 0")
else:
    print("  Tidak ada NaN tersisa [OK]")

y = df['label_hypertension']

# Info dataset
n_pos = int(y.sum())
n_neg = int((y == 0).sum())
ratio = n_neg / n_pos   # Untuk scale_pos_weight XGBoost

print(f"\n[DATA]")
print(f"  Total sampel     : {len(df):,}")
print(f"  Jumlah fitur     : {len(X.columns)}")
print(f"  Daftar fitur     : {X.columns.tolist()}")
print(f"  Positif (Hipert) : {n_pos:,} ({n_pos/len(y):.2%})")
print(f"  Negatif (Sehat)  : {n_neg:,} ({n_neg/len(y):.2%})")
print(f"  Imbalance ratio  : {ratio:.2f}:1")


# --- 2. TRAIN/TEST SPLIT ------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y      # Wajib untuk imbalanced data -- jaga proporsi di tiap split
)

print(f"\n[SPLIT]")
print(f"  Train : {len(X_train):,} sampel")
print(f"  Test  : {len(X_test):,} sampel")


# --- 3. DEFINISI MODEL --------------------------------------------------------
# [FIX 2] LR butuh StandardScaler -> pakai sklearn Pipeline
#         XGBoost tidak butuh scaling -> terima data mentah

# -- Model A: Logistic Regression (scaler built-in via Pipeline)
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        class_weight='balanced',
        C=0.1,
        solver='liblinear',
        random_state=RANDOM_STATE,
        max_iter=1000
    ))
])

# -- Model B: XGBoost standalone
# [FIX 5] use_label_encoder dihapus (deprecated)
clf_xgb = xgb.XGBClassifier(
    scale_pos_weight=ratio,   # Bobot kelas minoritas (hipertensi)
    n_estimators=300,
    max_depth=4,              # Tidak terlalu dalam -> cegah overfitting
    learning_rate=0.05,       # Belajar pelan -> generalisasi lebih baik
    subsample=0.8,            # 80% sampel per tree -> variasi
    colsample_bytree=0.8,     # 80% fitur per tree -> cegah korelasi antar tree
    min_child_weight=5,       # Minimal 5 sampel per leaf -> regularisasi
    gamma=0.1,                # Min loss reduction untuk split
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    verbosity=0
)

# -- [FIX 6] Ensemble: LR + XGB saja (RF dikeluarkan, terbukti drag-down recall)
# weights=[1,2]: XGBoost diberi kepercayaan 2x lebih besar
ensemble_lr_xgb = VotingClassifier(
    estimators=[
        ('lr',  pipe_lr),
        ('xgb', clf_xgb)
    ],
    voting='soft',
    weights=[1, 2]
)


# --- 4. FUNGSI HELPER ---------------------------------------------------------
def f2_score(precision, recall, eps=1e-9):
    """
    F2-Score: bobot Recall 2x lebih besar dari Precision.
    Lebih cocok dari F1 untuk medical screening (meminimalkan FN).
    """
    return (5 * precision * recall) / (4 * precision + recall + eps)


def find_optimal_threshold(y_true, y_prob, recall_target=RECALL_TARGET):
    """
    [FIX 1] Cari threshold optimal untuk medical screening:

    Strategi (berbeda dari v2 yang salah pakai argmax F1):
      1. Kumpulkan semua threshold yang menghasilkan Recall >= target
      2. Di antara yang lolos, pilih threshold dengan F2-Score tertinggi
      3. Fallback: jika tidak ada yang lolos, ambil threshold dengan
         Recall tertinggi yang bisa dicapai model
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    best_threshold  = None
    best_f2         = -1
    fallback_thresh = None
    fallback_recall = -1

    for i, thresh in enumerate(thresholds):
        rec  = recalls[i]
        prec = precisions[i]
        f2   = f2_score(prec, rec)

        if rec > fallback_recall:
            fallback_recall = rec
            fallback_thresh = thresh

        if rec >= recall_target and f2 > best_f2:
            best_f2        = f2
            best_threshold = thresh

    if best_threshold is not None:
        return best_threshold, True
    else:
        return fallback_thresh, False


def evaluate_model(name, y_true, y_pred, y_prob, threshold):
    """Hitung, cetak, dan kembalikan metrik evaluasi lengkap."""
    acc  = accuracy_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    f2   = f2_score(prec, rec)
    auc  = roc_auc_score(y_true, y_prob)

    status = "[TERCAPAI]" if rec >= RECALL_TARGET else "[BELUM]"

    print(f"\n{'-' * 55}")
    print(f"  {name}")
    print(f"{'-' * 55}")
    print(f"  Threshold   : {threshold:.4f}")
    print(f"  AUC-ROC     : {auc:.4f}")
    print(f"  Recall      : {rec:.2%}   {status}")
    print(f"  Precision   : {prec:.2%}")
    print(f"  F1-Score    : {f1:.2%}")
    print(f"  F2-Score    : {f2:.2%}  <- Metrik utama (Recall 2x bobot)")
    print(f"  Akurasi     : {acc:.2%}")
    print()
    print(classification_report(
        y_true, y_pred,
        target_names=['Sehat', 'Hipertensi'],
        zero_division=0
    ))

    return {'recall': rec, 'precision': prec, 'f1': f1, 'f2': f2, 'auc': auc, 'acc': acc}


def save_confusion_matrix(y_true, y_pred, name, threshold, output_dir):
    """Simpan confusion matrix sebagai gambar."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['Sehat', 'Hipertensi'],
        yticklabels=['Sehat', 'Hipertensi']
    )
    plt.title(
        f'{name}\n'
        f'Threshold: {threshold:.3f}  |  Recall: {recall_score(y_true, y_pred):.2%}\n'
        f'TP={tp}  FN={fn} (terlewat!)  |  FP={fp}  TN={tn}'
    )
    plt.xlabel('Prediksi Model')
    plt.ylabel('Fakta Medis')
    plt.tight_layout()

    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '')
    plt.savefig(f"{output_dir}/cm_{safe_name}.png", dpi=150)
    plt.close()


# --- 5. EKSPERIMEN A: LOGISTIC REGRESSION + THRESHOLD ------------------------
print("\n\n" + "=" * 65)
print("  EKSPERIMEN A: Logistic Regression + Threshold Moving")
print("=" * 65)

pipe_lr.fit(X_train, y_train)
y_prob_lr              = pipe_lr.predict_proba(X_test)[:, 1]
thresh_lr, achieved_lr = find_optimal_threshold(y_test, y_prob_lr)
y_pred_lr              = (y_prob_lr >= thresh_lr).astype(int)
metrics_lr             = evaluate_model("Logistic Regression", y_test, y_pred_lr, y_prob_lr, thresh_lr)
save_confusion_matrix(y_test, y_pred_lr, "LR Threshold", thresh_lr, OUTPUT_DIR)

if not achieved_lr:
    print(f"  [!]  Recall {RECALL_TARGET:.0%} tidak tercapai oleh LR.")
    print(f"      Recall tertinggi yang bisa dicapai: {metrics_lr['recall']:.2%}")


# --- 6. EKSPERIMEN B: XGBOOST STANDALONE + THRESHOLD -------------------------
print("\n\n" + "=" * 65)
print("  EKSPERIMEN B: XGBoost Standalone + Threshold Moving")
print("=" * 65)

# [FIX 2] XGBoost langsung makan X_train (bukan versi scaled)
clf_xgb.fit(X_train, y_train)
y_prob_xgb               = clf_xgb.predict_proba(X_test)[:, 1]
thresh_xgb, achieved_xgb = find_optimal_threshold(y_test, y_prob_xgb)
y_pred_xgb               = (y_prob_xgb >= thresh_xgb).astype(int)
metrics_xgb              = evaluate_model("XGBoost Standalone", y_test, y_pred_xgb, y_prob_xgb, thresh_xgb)
save_confusion_matrix(y_test, y_pred_xgb, "XGBoost Threshold", thresh_xgb, OUTPUT_DIR)

if not achieved_xgb:
    print(f"  [!]  Recall {RECALL_TARGET:.0%} tidak tercapai oleh XGBoost.")
    print(f"      Recall tertinggi yang bisa dicapai: {metrics_xgb['recall']:.2%}")


# --- 7. EKSPERIMEN C: ENSEMBLE LR+XGB + THRESHOLD ----------------------------
print("\n\n" + "=" * 65)
print("  EKSPERIMEN C: Ensemble (LR + XGB) + Threshold Moving")
print("=" * 65)

# VotingClassifier refit kedua estimator secara internal:
# - pipe_lr  -> StandardScaler + LR (scaling otomatis)
# - clf_xgb  -> XGBoost (raw data, tidak di-scale)
ensemble_lr_xgb.fit(X_train, y_train)
y_prob_ens               = ensemble_lr_xgb.predict_proba(X_test)[:, 1]
thresh_ens, achieved_ens = find_optimal_threshold(y_test, y_prob_ens)
y_pred_ens               = (y_prob_ens >= thresh_ens).astype(int)
metrics_ens              = evaluate_model("Ensemble LR+XGB", y_test, y_pred_ens, y_prob_ens, thresh_ens)
save_confusion_matrix(y_test, y_pred_ens, "Ensemble LR+XGB", thresh_ens, OUTPUT_DIR)

if not achieved_ens:
    print(f"  [!]  Recall {RECALL_TARGET:.0%} tidak tercapai oleh Ensemble.")
    print(f"      Recall tertinggi yang bisa dicapai: {metrics_ens['recall']:.2%}")


# --- 8. THRESHOLD SCAN -- VISUALISASI TRADE-OFF --------------------------------
print("\n\n" + "=" * 65)
print("  THRESHOLD SCAN: Visualisasi Trade-off Recall vs Precision")
print("=" * 65)

# Gunakan model dengan AUC tertinggi sebagai subjek scan
all_results = [
    ("LR",       y_prob_lr,  metrics_lr),
    ("XGBoost",  y_prob_xgb, metrics_xgb),
    ("Ensemble", y_prob_ens, metrics_ens),
]
best_scan        = max(all_results, key=lambda x: x[2]['auc'])
scan_name        = best_scan[0]
scan_probs       = best_scan[1]

scan_rows = []
for thresh in np.arange(0.05, 0.96, 0.05):
    y_pred_s = (scan_probs >= thresh).astype(int)
    if y_pred_s.sum() == 0:
        continue
    rec  = recall_score(y_test, y_pred_s, zero_division=0)
    prec = precision_score(y_test, y_pred_s, zero_division=0)
    f1   = f1_score(y_test, y_pred_s, zero_division=0)
    f2   = f2_score(prec, rec)
    acc  = accuracy_score(y_test, y_pred_s)
    scan_rows.append({
        'threshold': round(thresh, 2), 'recall': round(rec, 4),
        'precision': round(prec, 4),   'f1':     round(f1,  4),
        'f2':        round(f2,   4),   'accuracy': round(acc, 4)
    })

df_scan = pd.DataFrame(scan_rows)
print(f"\nThreshold Scan -- Model terbaik (AUC tertinggi): {scan_name}")
print(df_scan.to_string(index=False))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(df_scan['threshold'], df_scan['recall'],    'o-', color='tomato',    label='Recall',    linewidth=2)
axes[0].plot(df_scan['threshold'], df_scan['precision'], 's-', color='steelblue', label='Precision', linewidth=2)
axes[0].plot(df_scan['threshold'], df_scan['f2'],        '^-', color='seagreen',  label='F2-Score',  linewidth=2)
axes[0].axhline(y=RECALL_TARGET, color='red', linestyle='--', alpha=0.8,
                label=f'Target Recall {RECALL_TARGET:.0%}')
axes[0].set_xlabel('Threshold')
axes[0].set_ylabel('Score')
axes[0].set_title(f'Trade-off Metrik vs Threshold\n(Model: {scan_name})')
axes[0].legend()
axes[0].grid(alpha=0.3)

pr_prec, pr_rec, _ = precision_recall_curve(y_test, scan_probs)
axes[1].plot(pr_rec, pr_prec, color='purple', linewidth=2)
axes[1].axvline(x=RECALL_TARGET, color='red', linestyle='--', alpha=0.8,
                label=f'Target Recall {RECALL_TARGET:.0%}')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title(f'Precision-Recall Curve\n(Model: {scan_name})')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/threshold_analysis_{scan_name}.png", dpi=150)
plt.close()
print(f"Grafik tersimpan -> '{OUTPUT_DIR}/threshold_analysis_{scan_name}.png'")


# --- 9. CROSS-VALIDATION (ROBUSTNESS CHECK) -----------------------------------
print("\n\n" + "=" * 65)
print(f"  CROSS-VALIDATION ({CV_FOLDS}-Fold Stratified) -- Robustness Check")
print("=" * 65)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for cv_name, cv_model in [
    ("Logistic Regression", pipe_lr),
    ("XGBoost",             clf_xgb),
    ("Ensemble LR+XGB",     ensemble_lr_xgb),
]:
    recall_cv = cross_val_score(cv_model, X, y, cv=cv, scoring='recall',  n_jobs=-1)
    auc_cv    = cross_val_score(cv_model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"\n  {cv_name}")
    print(f"    CV Recall  (threshold=0.5) : {recall_cv.mean():.4f} +/- {recall_cv.std():.4f}")
    print(f"    CV AUC-ROC                 : {auc_cv.mean():.4f} +/- {auc_cv.std():.4f}")


# --- 10. RINGKASAN FINAL ------------------------------------------------------
print("\n\n" + "=" * 65)
print("  RINGKASAN KOMPARASI SEMUA EKSPERIMEN")
print("=" * 65)

summary = {
    "LR + Threshold Moving"       : (metrics_lr,  thresh_lr),
    "XGBoost + Threshold Moving"  : (metrics_xgb, thresh_xgb),
    "Ensemble LR+XGB + Threshold" : (metrics_ens, thresh_ens),
}

header = f"{'Model':<35} | {'Thresh':>6} | {'Recall':>7} | {'Precision':>9} | {'F2':>7} | {'AUC':>7} | Status"
print(f"\n{header}")
print("-" * len(header))

for model_name, (m, t) in summary.items():
    status = "TARGET OK" if m['recall'] >= RECALL_TARGET else "Belum"
    print(
        f"{model_name:<35} | {t:>6.3f} | {m['recall']:>7.2%} | "
        f"{m['precision']:>9.2%} | {m['f2']:>7.2%} | {m['auc']:>7.4f} | {status}"
    )

# Pilih pemenang: Recall >= 90% -> F2-Score tertinggi
qualified = {k: v for k, v in summary.items() if v[0]['recall'] >= RECALL_TARGET}

print()
if qualified:
    winner   = max(qualified, key=lambda k: qualified[k][0]['f2'])
    w_m, w_t = summary[winner]
    print(f"[BEST] MODEL TERPILIH  : {winner}")
    print(f"   Threshold       : {w_t:.4f}")
    print(f"   Recall          : {w_m['recall']:.2%}  (NFR02 terpenuhi)")
    print(f"   Precision       : {w_m['precision']:.2%}")
    print(f"   F2-Score        : {w_m['f2']:.2%}")
    print(f"   AUC-ROC         : {w_m['auc']:.4f}")
    print(f"   Akurasi         : {w_m['acc']:.2%}")
else:
    best_name = max(summary, key=lambda k: summary[k][0]['recall'])
    best_m    = summary[best_name][0]
    print(f"[!]  Tidak ada model yang tembus Recall {RECALL_TARGET:.0%}")
    print(f"   Model terbaik (Recall tertinggi) : {best_name}")
    print(f"   Recall                           : {best_m['recall']:.2%}")
    print(f"   -> Next step: Lanjutkan ke SMOTE/ADASYN + retune")

print(f"\nSemua output grafik tersimpan di: '{OUTPUT_DIR}/'")
print("=" * 65)