



# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_auc_score, recall_score, f1_score, confusion_matrix, accuracy_score, precision_recall_curve, auc

# # Import Models
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier

# # SIMPAN MODEL
# import joblib

# import warnings
# warnings.filterwarnings('ignore')

# # ---------------------------------------------------------
# # FASE 1: PERSIAPAN DATA
# # ---------------------------------------------------------
# TARGET_COL = 'label_hypertension' 

# # # Load dataset
# # df = pd.read_csv('dataset_hipertensi_prepared.csv')

# # # Mengubah semua kolom yang berisi teks (kategorikal) menjadi angka (0 dan 1)
# # df = pd.get_dummies(df, drop_first=True)

# # # Pisahkan Fitur (X) dan Target (y)
# # X = df.drop(columns=['label_hypertension'], errors='ignore')
# # X = X.select_dtypes(include=[np.number]).fillna(0)
# # y = df[TARGET_COL]

# # # Cek rasio kelas untuk melihat apakah data imbalance
# # print("Distribusi Kelas Target:")
# # print(y.value_counts(normalize=True) * 100)
# # print("-" * 50)

# # # Train-Test Split (80% Train, 20% Test)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import matplotlib
# matplotlib.use('Agg')   # backend non-interaktif: tidak menampilkan window, hanya simpan file
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

# # ---------------------------------------------------------
# # FASE 1: MEMBACA DATA HASIL PREPROCESSING
# # ---------------------------------------------------------

# TARGET_COL = "label_hypertension"

# # Load data yang SUDAH di-split & diimputasi
# train_df = pd.read_csv("train_hipertensi.csv")
# test_df  = pd.read_csv("test_hipertensi.csv")

# print("=== DATA TRAIN ===")
# print(train_df.shape)

# print("=== DATA TEST ===")
# print(test_df.shape)

# # Pisahkan fitur dan target
# X_train = train_df.drop(columns=[TARGET_COL])
# y_train = train_df[TARGET_COL]

# X_test = test_df.drop(columns=[TARGET_COL])
# y_test = test_df[TARGET_COL]

# print("\nJumlah fitur :", X_train.shape[1])

# print("\nDistribusi kelas TRAIN")
# print(y_train.value_counts())
# print((y_train.value_counts(normalize=True)*100).round(2))

# print("\nDistribusi kelas TEST")
# print(y_test.value_counts())
# print((y_test.value_counts(normalize=True)*100).round(2))

# print("\nMissing value TRAIN :", X_train.isna().sum().sum())
# print("Missing value TEST  :", X_test.isna().sum().sum())



# # Standarisasi fitur numerik (Sangat penting untuk Logistic Regression)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Hitung rasio kelas untuk XGBoost
# neg, pos = np.bincount(y_train)
# scale_pos_weight = neg / pos

# # ---------------------------------------------------------
# # FASE 2 & 3: INISIALISASI & EVALUASI MODEL BASELINE
# # ---------------------------------------------------------
# print("\nMEMULAI EVALUASI BASELINE...\n" + "-"*50)
# models = {
#     'Logistic Regression': LogisticRegression(random_state=42),
#     'Random Forest': RandomForestClassifier(random_state=42),
#     'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
#     'CatBoost': CatBoostClassifier(random_state=42, verbose=0)
# }

# results_baseline = []

# for name, model in models.items():
#     X_tr = X_train_scaled if name == 'Logistic Regression' else X_train
#     X_te = X_test_scaled if name == 'Logistic Regression' else X_test
    
#     model.fit(X_tr, y_train)
#     y_pred = model.predict(X_te)
#     y_proba = model.predict_proba(X_te)[:, 1]
    
#     # Kalkulasi Metrik
#     roc_auc = roc_auc_score(y_test, y_proba)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     acc = accuracy_score(y_test, y_pred)
    
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#     specificity = tn / (tn + fp)
    
#     precision, rec, _ = precision_recall_curve(y_test, y_proba)
#     pr_auc = auc(rec, precision)
    
#     results_baseline.append({
#         'Model': name,
#         'ROC-AUC': round(roc_auc, 4),
#         'Recall (Sens)': round(recall, 4),
#         'Specificity': round(specificity, 4),
#         'PR-AUC': round(pr_auc, 4),
#         'F1-Score': round(f1, 4),
#         'Akurasi': round(acc, 4)
#     })

# df_baseline = pd.DataFrame(results_baseline).set_index('Model')
# # pd.set_option('display.float_format', lambda x: '%.4f' % x)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)

# print("TABEL KINERJA AWAL MODEL (BASELINE)")
# # print(df_baseline)
# print(df_baseline.map(lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x))

# # ---------------------------------------------------------
# # FASE 4: HYPERPARAMETER TUNING UNTUK SEMUA MODEL
# # ---------------------------------------------------------
# print("\nMEMULAI HYPERPARAMETER TUNING (Fokus pada ROC-AUC)...\n" + "-"*50)

# # Konfigurasi Model & Parameter Grid
# # Catatan: Class weights ditambahkan untuk menangani imbalance data secara native
# tuning_configs = {
#     'Logistic Regression': {
#         'model': LogisticRegression(class_weight='balanced', random_state=42),
#         'params': {'C': [0.01, 0.1, 1, 10]},
#         'use_scaled': True
#     },
#     'Random Forest': {
#         'model': RandomForestClassifier(class_weight='balanced', random_state=42),
#         'params': {'max_depth': [5, 10, None], 'n_estimators': [100, 200]},
#         'use_scaled': False
#     },
#     'XGBoost': {
#         'model': XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight),
#         'params': {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]},
#         'use_scaled': False
#     },
#     'CatBoost': {
#         'model': CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced'),
#         'params': {'depth': [4, 6], 'learning_rate': [0.01, 0.1], 'iterations': [100, 200]},
#         'use_scaled': False
#     }
# }

# results_tuned = []
# best_models = {} # Menyimpan model terbaik untuk masing-masing algoritma
# predictions = {} # menyimpan y_pred & y_proba untuk setiap model

# for name, config in tuning_configs.items():
#     print(f"Tuning {name}...")
    
#     # Pilih data yang tepat (scaled untuk Regresi Logistik)
#     X_tr = X_train_scaled if config['use_scaled'] else X_train
#     X_te = X_test_scaled if config['use_scaled'] else X_test
    
#     # Setup GridSearchCV
#     grid_search = GridSearchCV(
#         estimator=config['model'], 
#         param_grid=config['params'], 
#         scoring='roc_auc', 
#         cv=3, 
#         verbose=1, 
#         n_jobs=-1 
#     )
    
#     # Fit Model
#     grid_search.fit(X_tr, y_train)
    
#     # Ekstrak model terbaik
#     best_model = grid_search.best_estimator_
#     best_models[name] = best_model
    
#     # Prediksi
#     y_pred = best_model.predict(X_te)
#     y_proba = best_model.predict_proba(X_te)[:, 1]
#     predictions[name] = {'y_pred': y_pred, 'y_proba': y_proba}   # <-- TAMBAHKAN

    
#     # Kalkulasi Metrik
#     roc_auc = roc_auc_score(y_test, y_proba)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     acc = accuracy_score(y_test, y_pred)
    
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#     specificity = tn / (tn + fp)
    
#     precision, rec, _ = precision_recall_curve(y_test, y_proba)
#     pr_auc = auc(rec, precision)
    
#     # Simpan Hasil
#     results_tuned.append({
#         'Model': name,
#         'ROC-AUC': round(roc_auc, 2),
#         'Recall (Sens)': round(recall, 2),
#         'Specificity': round(specificity, 2),
#         'PR-AUC': round(pr_auc, 2),
#         'F1-Score': round(f1, 2),
#         'Akurasi': round(acc, 2)
#     })
    
#     print(f"Best Params {name}: {grid_search.best_params_}\n")

# # ---------------------------------------------------------
# # FASE 5: KOMPARASI & SIMPAN MODEL
# # ---------------------------------------------------------
# df_tuned = pd.DataFrame(results_tuned).set_index('Model')

# print("\nTABEL KINERJA MODEL SETELAH TUNING")
# print(df_tuned)

# print("\nmembuat visuasliasi eval...\n" + "-"*50)

# class_labels = ['Normal','Hipertensi']

# # ---- 5a. CONFUSION MATRIX (satu figure untuk semua model) ----
# n_models = len(best_models)
# fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
# if n_models == 1:
#     axes = [axes]

# for ax, name in zip(axes, best_models):
#     cm = confusion_matrix(y_test, predictions[name]['y_pred'])
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
#     disp.plot(ax=ax, cmap='Blues', colorbar=False)
#     ax.set_title(name)
#     ax.grid(False)

# plt.suptitle('Confusion Matrix - Model Setelah Tuning (Test Set)', fontsize=13)
# plt.tight_layout()
# plt.savefig('confusion_matrix_all.png', dpi=150, bbox_inches='tight')
# plt.close(fig)
# print("-> confusion_matrix_all.png tersimpan")


# # ---- 5b. KURVA ROC + PERHITUNGAN AUC (komparasi dalam satu grafik) ----
# plt.figure(figsize=(8, 6))

# for name in best_models:
#     y_proba = predictions[name]['y_proba']
#     fpr, tpr, _ = roc_curve(y_test, y_proba)          # titik-titik kurva
#     auc_val = roc_auc_score(y_test, y_proba)          # perhitungan AUC
#     plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc_val:.3f})')

# # Garis referensi random classifier
# plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')

# plt.xlabel('False Positive Rate (1 - Specificity)')
# plt.ylabel('True Positive Rate (Sensitivity / Recall)')
# plt.title('Kurva ROC - Komparasi Model (Test Set)')
# plt.legend(loc='lower right')
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig('roc_curve_all.png', dpi=150, bbox_inches='tight')
# plt.close()
# print("-> roc_curve_all.png tersimpan")


# # ---- 5c. RINGKASAN AUC (tabel) ----
# auc_summary = pd.DataFrame({
#     'Model': list(best_models.keys()),
#     'ROC-AUC': [round(roc_auc_score(y_test, predictions[n]['y_proba']), 5)
#                 for n in best_models]
# }).set_index('Model').sort_values('ROC-AUC', ascending=False)

# print("RINGKASAN ROC-AUC (diurutkan):")
# print(auc_summary)


# # Fase 6: Menyimpan model XGBoost terbaik ke dalam file .pkl (Sesuai kode awal)
# # Mengambil dari dictionary best_models
# print(best_models)
# best_xgb = best_models['XGBoost']
# model_filename = 'xgboost_hipertensi_model.pkl'
# joblib.dump(best_xgb, model_filename)

# print(f"\nModel XGBoost berhasil disimpan dengan nama: {model_filename}")










import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score, confusion_matrix, accuracy_score,
    precision_recall_curve, auc, roc_curve, ConfusionMatrixDisplay
)

# Import Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import fbeta_score, make_scorer

# SIMPAN MODEL
import joblib

# Visualisasi
import matplotlib
matplotlib.use('Agg')   # backend non-interaktif: hanya simpan file, tidak buka window
import matplotlib.pyplot as plt

import warnings
# CATATAN: Saat menjalankan eksperimen baseline LR (scaled vs no-scaling),
# sebaiknya KOMENTARI baris di bawah ini agar ConvergenceWarning terlihat.
# Warning itu adalah bukti kunci bahwa versi unscaled tidak konvergen.
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# FASE 1: MEMBACA DATA HASIL PREPROCESSING
# (Split & imputasi sudah dilakukan di tahap data preparation)
# ---------------------------------------------------------
TARGET_COL = "label_hypertension"

train_df = pd.read_csv("train_hipertensi.csv")
test_df  = pd.read_csv("test_hipertensi.csv")

print("=== DATA TRAIN ===", train_df.shape)
print("=== DATA TEST  ===", test_df.shape)

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
X_test  = test_df.drop(columns=[TARGET_COL])
y_test  = test_df[TARGET_COL]

print("\nJumlah fitur :", X_train.shape[1])
print("\nDistribusi kelas TRAIN")
print((y_train.value_counts(normalize=True) * 100).round(2))
print("\nDistribusi kelas TEST")
print((y_test.value_counts(normalize=True) * 100).round(2))
print("\nMissing value TRAIN :", X_train.isna().sum().sum())
print("Missing value TEST  :", X_test.isna().sum().sum())

# Standarisasi fitur (untuk Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Rasio kelas untuk XGBoost
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos


# =========================================================
# FUNGSI BANTU EVALUASI
# =========================================================
def evaluate(y_true, y_pred, y_proba):
    """Kembalikan dict metrik standar."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    precision, rec, _ = precision_recall_curve(y_true, y_proba)
    return {
        'ROC-AUC': roc_auc_score(y_true, y_proba),
        'Recall (Sens)': recall_score(y_true, y_pred),
        'Specificity': specificity,
        'PR-AUC': auc(rec, precision),
        'F1-Score': f1_score(y_true, y_pred),
        'Akurasi': accuracy_score(y_true, y_pred),
    }


# ---------------------------------------------------------
# FASE 2 & 3: BASELINE
#  -> PERMINTAAN #1: LR diuji DUA kali (tanpa scaling & dengan scaling)
#     sebagai bukti pengaruh StandardScaler pada Logistic Regression.
# ---------------------------------------------------------
print("\nMEMULAI EVALUASI BASELINE...\n" + "-" * 50)

# Format: 'nama': (model, use_scaled)
baseline_models = {
    'Logistic Regression (Tanpa Scaling)': (LogisticRegression(random_state=42), False),
    'Logistic Regression (Scaled)':        (LogisticRegression(random_state=42), True),
    'Random Forest':                       (RandomForestClassifier(random_state=42), False),
    'XGBoost':                             (XGBClassifier(random_state=42, eval_metric='logloss'), False),
    'CatBoost':                            (CatBoostClassifier(random_state=42, verbose=0), False),
}

results_baseline = []
for name, (model, use_scaled) in baseline_models.items():
    X_tr = X_train_scaled if use_scaled else X_train
    X_te = X_test_scaled  if use_scaled else X_test

    model.fit(X_tr, y_train)
    y_pred  = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    row = {'Model': name}
    row.update({k: round(v, 4) for k, v in evaluate(y_test, y_pred, y_proba).items()})
    results_baseline.append(row)

df_baseline = pd.DataFrame(results_baseline).set_index('Model')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("TABEL KINERJA AWAL MODEL (BASELINE)")
print(df_baseline)

# Ringkas selisih LR untuk memudahkan penulisan laporan
lr_no  = df_baseline.loc['Logistic Regression (Tanpa Scaling)']
lr_yes = df_baseline.loc['Logistic Regression (Scaled)']
print("\n>> Pengaruh StandardScaler pada Logistic Regression (Scaled - Tanpa Scaling):")
print((lr_yes - lr_no).round(4))
print("   (Selisih positif = scaling meningkatkan metrik tsb. Bila selisih ~0, "
      "artinya solver tetap konvergen tanpa scaling.)")


# ---------------------------------------------------------
# FASE 4: HYPERPARAMETER TUNING (fokus ROC-AUC)
# ---------------------------------------------------------
print("\nMEMULAI HYPERPARAMETER TUNING (Fokus pada ROC-AUC)...\n" + "-" * 50)

tuning_configs = {
    'Logistic Regression': {
        'model': LogisticRegression(class_weight='balanced', random_state=42),
        'params': {'C': [0.01, 0.1, 1, 10]},
        'use_scaled': True
    },
    'Random Forest': {
        'model': RandomForestClassifier(class_weight='balanced', random_state=42),
        'params': {'max_depth': [5, 10, None], 'n_estimators': [100, 200]},
        'use_scaled': False
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight, n_jobs=1),
        'params': {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]},
        'use_scaled': False
    },
    'CatBoost': {
        'model': CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced'),
        'params': {'depth': [4, 6], 'learning_rate': [0.01, 0.1], 'iterations': [100, 200]},
        'use_scaled': False
    }
}

results_tuned = []
best_models = {}
predictions = {}   # y_pred (threshold 0.5) & y_proba per model

for name, config in tuning_configs.items():
    print(f"Tuning {name}...")
    X_tr = X_train_scaled if config['use_scaled'] else X_train
    X_te = X_test_scaled  if config['use_scaled'] else X_test

    grid_search = GridSearchCV(
        estimator=config['model'], param_grid=config['params'],
        scoring='roc_auc', cv=3, verbose=1, n_jobs=2
    )
    grid_search.fit(X_tr, y_train)

    best_model = grid_search.best_estimator_
    best_models[name] = best_model

    y_pred  = best_model.predict(X_te)
    y_proba = best_model.predict_proba(X_te)[:, 1]
    predictions[name] = {'y_pred': y_pred, 'y_proba': y_proba}

    row = {'Model': name}
    row.update({k: round(v, 4) for k, v in evaluate(y_test, y_pred, y_proba).items()})
    results_tuned.append(row)

    print(f"Best Params {name}: {grid_search.best_params_}\n")

df_tuned = pd.DataFrame(results_tuned).set_index('Model')
print("\nTABEL KINERJA MODEL SETELAH TUNING (threshold default 0.5)")
print(df_tuned)


# ---------------------------------------------------------
# FASE 5: AMBANG OPTIMAL VIA YOUDEN'S J
#  -> PERMINTAAN #2
#  Youden J = TPR - FPR  (jarak vertikal terjauh kurva ROC ke garis diagonal).
#  Threshold dengan J tertinggi = titik yang paling menyeimbangkan
#  false positive rate dan false negative rate.
# ---------------------------------------------------------
print("\nFASE 5: MENCARI AMBANG OPTIMAL (YOUDEN'S J)\n" + "-" * 50)

def find_youden(y_true, y_proba):
    """Cari threshold yang memaksimalkan J = TPR - FPR."""
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    J = tpr - fpr
    ix = int(np.argmax(J))
    # thr[0] pada sklearn baru bisa bernilai inf; argmax J tidak akan memilihnya
    # karena di titik itu TPR=FPR=0 sehingga J=0.
    return {'threshold': float(thr[ix]), 'J': float(J[ix]),
            'fpr': float(fpr[ix]), 'tpr': float(tpr[ix])}

# --- (Alternatif untuk konteks skrining medis) ---
# Youden membobot sensitivity & specificity SAMA. Jika false negative dianggap
# jauh lebih mahal (pasien hipertensi lolos), pilih threshold terendah yang
# masih memenuhi target recall, misal >= 0.90:
#
# def threshold_for_recall(y_true, y_proba, target=0.90):
#     prec, rec, thr = precision_recall_curve(y_true, y_proba)
#     # thr sepanjang len(rec)-1; sejajarkan dengan rec[:-1]
#     ok = np.where(rec[:-1] >= target)[0]
#     return float(thr[ok[-1]]) if len(ok) else 0.5
# ---------------------------------------------------

youden_info = {}
predictions_youden = {}
rows = []

for name in best_models:
    y_proba = predictions[name]['y_proba']
    info = find_youden(y_test, y_proba)
    youden_info[name] = info

    thr = info['threshold']
    y_pred_def = predictions[name]['y_pred']       # prediksi @ threshold 0.5
    y_pred_opt = (y_proba >= thr).astype(int)      # roc_curve memakai konvensi score >= thr
    predictions_youden[name] = y_pred_opt

    # Metrik LENGKAP @0.5 (default) vs @Youden -> dua baris per model
    m_def = evaluate(y_test, y_pred_def, y_proba)
    m_opt = evaluate(y_test, y_pred_opt, y_proba)

    row_def = {'Model': name, 'Ambang': '0.5 (default)', 'Threshold': 0.5, 'J': np.nan}
    row_def.update({k: round(v, 4) for k, v in m_def.items()})
    rows.append(row_def)

    row_opt = {'Model': name, 'Ambang': 'Youden', 'Threshold': round(thr, 4), 'J': round(info['J'], 4)}
    row_opt.update({k: round(v, 4) for k, v in m_opt.items()})
    rows.append(row_opt)

df_youden = pd.DataFrame(rows).set_index(['Model', 'Ambang'])
print(df_youden)
print("\nCatatan:")
print(" - ROC-AUC & PR-AUC dihitung dari probabilitas (ranking), BUKAN dari label,")
print("   sehingga nilainya IDENTIK di baris 0.5 dan Youden (tidak dipengaruhi threshold).")
print(" - Yang bergeser hanya Recall, Specificity, F1-Score, dan Akurasi.")
print(" - J = Youden's index (TPR - FPR) pada ambang terpilih.")


# ---------------------------------------------------------
# FASE 6: VISUALISASI EVALUASI
# ---------------------------------------------------------
print("\nMEMBUAT VISUALISASI EVAL...\n" + "-" * 50)
class_labels = ['Normal', 'Hipertensi']

# 6a. Confusion matrix pada AMBANG YOUDEN
n_models = len(best_models)
fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
if n_models == 1:
    axes = [axes]
for ax, name in zip(axes, best_models):
    cm = confusion_matrix(y_test, predictions_youden[name])
    ConfusionMatrixDisplay(cm, display_labels=class_labels).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f"{name}\n(thr={youden_info[name]['threshold']:.3f})")
    ax.grid(False)
plt.suptitle("Confusion Matrix @ Ambang Youden (Test Set)", fontsize=13)
plt.tight_layout()
plt.savefig('confusion_matrix_youden.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("-> confusion_matrix_youden.png tersimpan")

# 6b. Kurva ROC + tanda titik Youden tiap model
plt.figure(figsize=(8, 6))
for name in best_models:
    y_proba = predictions[name]['y_proba']
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_val = roc_auc_score(y_test, y_proba)
    line, = plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc_val:.3f})')
    # tandai titik Youden pada warna garis yang sama
    plt.scatter(youden_info[name]['fpr'], youden_info[name]['tpr'],
                color=line.get_color(), edgecolor='black', zorder=5, s=70)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity / Recall)')
plt.title("Kurva ROC + Titik Youden (Test Set)")
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_youden.png', dpi=150, bbox_inches='tight')
plt.close()
print("-> roc_curve_youden.png tersimpan (titik hitam = ambang Youden tiap model)")


# ---------------------------------------------------------
# FASE 7: SIMPAN MODEL + THRESHOLD + SKEMA FITUR
#  Menyimpan threshold Youden bersama model agar backend memprediksi
#  dengan ambang yang benar (bukan 0.5 default).
# ---------------------------------------------------------
best_xgb = best_models['XGBoost']
youden_thr_xgb = youden_info['XGBoost']['threshold']

artifact = {
    'model': best_xgb,
    'threshold': youden_thr_xgb,
    'feature_names': list(X_train.columns),  # untuk validasi urutan kolom saat inference
}
model_filename = 'xgboost_hipertensi_model.pkl'
joblib.dump(artifact, model_filename)
print(f"\nModel + threshold + skema fitur disimpan: {model_filename}")
print(f"   threshold Youden XGBoost = {youden_thr_xgb:.4f}")

# --- Cara pakai di backend ---
# art = joblib.load('xgboost_hipertensi_model.pkl')
# X_new = df_pasien[art['feature_names']]              # jaga urutan kolom
# proba = art['model'].predict_proba(X_new)[:, 1]
# pred  = (proba >= art['threshold']).astype(int)      # pakai ambang Youden, bukan 0.5