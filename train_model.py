



# import pandas as pd
# import numpy as np
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import (
#     roc_auc_score, recall_score, f1_score, confusion_matrix, accuracy_score,
#     precision_recall_curve, auc, roc_curve, ConfusionMatrixDisplay
# )

# # Import Models
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
# from sklearn.metrics import fbeta_score, make_scorer
# import seaborn as sns

# # SIMPAN MODEL
# import joblib

# # Visualisasi
# import matplotlib
# matplotlib.use('Agg')   # backend non-interaktif: hanya simpan file, tidak buka window
# import matplotlib.pyplot as plt

# import warnings
# # CATATAN: Saat menjalankan eksperimen baseline LR (scaled vs no-scaling),
# # sebaiknya KOMENTARI baris di bawah ini agar ConvergenceWarning terlihat.
# # Warning itu adalah bukti kunci bahwa versi unscaled tidak konvergen.
# warnings.filterwarnings('ignore')

# import os

# OUTPUT_DIR = "hasil_modelling"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# def out_path(filename):
#     return os.path.join(OUTPUT_DIR, filename)





# # ---------------------------------------------------------
# # FASE 1: MEMBACA DATA HASIL PREPROCESSING
# # (Split & imputasi sudah dilakukan di tahap data preparation)
# # ---------------------------------------------------------
# TARGET_COL = "label_hypertension"

# train_df = pd.read_csv("train_hipertensi.csv")
# test_df  = pd.read_csv("test_hipertensi.csv")

# print("=== DATA TRAIN ===", train_df.shape)
# print("=== DATA TEST  ===", test_df.shape)

# X_train = train_df.drop(columns=[TARGET_COL])
# y_train = train_df[TARGET_COL]
# X_test  = test_df.drop(columns=[TARGET_COL])
# y_test  = test_df[TARGET_COL]

# print("\nJumlah fitur :", X_train.shape[1])
# print("\nDistribusi kelas TRAIN")
# print((y_train.value_counts(normalize=True) * 100).round(2))
# print("\nDistribusi kelas TEST")
# print((y_test.value_counts(normalize=True) * 100).round(2))
# print("\nMissing value TRAIN :", X_train.isna().sum().sum())
# print("Missing value TEST  :", X_test.isna().sum().sum())


# # Rasio kelas untuk XGBoost
# neg, pos = np.bincount(y_train)
# scale_pos_weight = neg / pos


# # =========================================================
# # FUNGSI BANTU EVALUASI
# # =========================================================
# def evaluate(y_true, y_pred, y_proba):
#     """Kembalikan dict metrik standar."""
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     specificity = tn / (tn + fp)
#     precision, rec, _ = precision_recall_curve(y_true, y_proba)
#     return {
#         'ROC-AUC': roc_auc_score(y_true, y_proba),
#         'Recall (Sens)': recall_score(y_true, y_pred),
#         'Specificity': specificity,
#         'PR-AUC': auc(rec, precision),
#         'F1-Score': f1_score(y_true, y_pred),
#         'Akurasi': accuracy_score(y_true, y_pred),
#     }



# # ---------------------------------------------------------
# # FASE 2, 3, 4: ABLATION STUDY (4 SKENARIO EKSPERIMEN)
# # ---------------------------------------------------------
# print("\nMEMULAI ABLATION STUDY (4 SKENARIO)...\n" + "-" * 50)

# # Konfigurasi dasar dan hyperparameter untuk tiap algoritma
# configs = {
#     'Logistic Regression': {
#         'base': LogisticRegression(random_state=42, max_iter=1000),
#         'base_weighted': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
#         'params': {'C': [0.01, 0.1, 1, 10]}
#     },
#     'Random Forest': {
#         'base': RandomForestClassifier(random_state=42),
#         'base_weighted': RandomForestClassifier(random_state=42, class_weight='balanced'),
#         'params': {'max_depth': [5, 10, None], 'n_estimators': [100, 200]}
#     },
#     'XGBoost': {
#         'base': XGBClassifier(random_state=42, eval_metric='logloss'),
#         'base_weighted': XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=scale_pos_weight),
#         'params': {'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
#     },
#     'CatBoost': {
#         'base': CatBoostClassifier(random_state=42, verbose=0),
#         'base_weighted': CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced'),
#         'params': {'depth': [4, 6], 'learning_rate': [0.01, 0.1], 'iterations': [100, 200]}
#     }
# }

# results_ablation = []
# champion_models = {}  # Hanya menyimpan model dari Skenario 4 (Tuned + Weighted)
# predictions_champion = {}

# for algo_name, cfg in configs.items():
#     print(f"\nMengeksekusi {algo_name}...")
    
#     # Skenario 1: Pure Baseline
#     model_s1 = cfg['base']
#     model_s1.fit(X_train, y_train)
#     res_s1 = evaluate(y_test, model_s1.predict(X_test), model_s1.predict_proba(X_test)[:, 1])
#     res_s1.update({'Algoritma': algo_name, 'Skenario': '1. Pure Baseline'})
#     results_ablation.append(res_s1)
    
#     # Skenario 2: Baseline + Class Weighting
#     model_s2 = cfg['base_weighted']
#     model_s2.fit(X_train, y_train)
#     res_s2 = evaluate(y_test, model_s2.predict(X_test), model_s2.predict_proba(X_test)[:, 1])
#     res_s2.update({'Algoritma': algo_name, 'Skenario': '2. Baseline + Weighting'})
#     results_ablation.append(res_s2)
    
#     # Skenario 3: Tuning Only (No Weighting)
#     gs_s3 = GridSearchCV(cfg['base'], cfg['params'], scoring='accuracy', cv=3, n_jobs=2)
#     gs_s3.fit(X_train, y_train)
#     res_s3 = evaluate(y_test, gs_s3.predict(X_test), gs_s3.predict_proba(X_test)[:, 1])
#     res_s3.update({'Algoritma': algo_name, 'Skenario': '3. Tuning Only'})
#     results_ablation.append(res_s3)
    
#     # Skenario 4: Tuning + Class Weighting (The Champion)
#     gs_s4 = GridSearchCV(cfg['base_weighted'], cfg['params'], scoring='accuracy', cv=3, n_jobs=2)
#     gs_s4.fit(X_train, y_train)
    
#     best_s4 = gs_s4.best_estimator_
#     y_pred_s4 = best_s4.predict(X_test)
#     y_proba_s4 = best_s4.predict_proba(X_test)[:, 1]
    
#     res_s4 = evaluate(y_test, y_pred_s4, y_proba_s4)
#     res_s4.update({'Algoritma': algo_name, 'Skenario': '4. Tuning + Weighting'})
#     results_ablation.append(res_s4)
    
#     # Simpan khusus Skenario 4 untuk pencarian Youden's J nanti
#     champion_models[algo_name] = best_s4
#     predictions_champion[algo_name] = {'y_pred': y_pred_s4, 'y_proba': y_proba_s4}


# # --- TABEL HASIL ABLATION STUDY ---
# df_ablation = pd.DataFrame(results_ablation)
# # Reorder columns agar rapi
# cols = ['Algoritma', 'Skenario', 'ROC-AUC', 'Recall (Sens)', 'Specificity', 'F1-Score', 'Akurasi', 'PR-AUC']
# df_ablation = df_ablation[cols]

# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
# print("\nTABEL HASIL ABLATION STUDY (4 SKENARIO)")
# print(df_ablation.round(4).to_string(index=False))


# # --- VISUALISASI ABLATION STUDY ---
# print("\nMembuat Visualisasi Ablation Study...")
# # --- TABEL PER SKENARIO ABLASI ---
# METRIC_COLS = ['ROC-AUC', 'Recall (Sens)', 'Specificity', 'F1-Score', 'Akurasi', 'PR-AUC']
# SCENARIOS = ['1. Pure Baseline', '2. Baseline + Weighting',
#              '3. Tuning Only', '4. Tuning + Weighting']

# def render_tabel_png(tbl, judul, fname):
#     """Render DataFrame jadi PNG tabel siap tempel ke laporan."""
#     fig, ax = plt.subplots(figsize=(1.55 * len(tbl.columns) + 2.2, 0.6 * len(tbl) + 1.4))
#     ax.axis('off')
#     mtable = ax.table(
#         cellText=[[f"{v:.4f}" for v in row] for row in tbl.values],
#         rowLabels=tbl.index,
#         colLabels=tbl.columns,
#         cellLoc='center', rowLoc='left', loc='center'
#     )
#     mtable.auto_set_font_size(False)
#     mtable.set_fontsize(10)
#     mtable.scale(1, 1.6)

#     # Header styling
#     for j in range(len(tbl.columns)):
#         cell = mtable[0, j]
#         cell.set_facecolor('#2f4b7c')
#         cell.set_text_props(color='white', weight='bold')
#     for i in range(len(tbl)):
#         mtable[i + 1, -1].set_text_props(weight='bold')

#     # Tandai nilai terbaik tiap kolom (semua metrik di sini: makin tinggi makin baik)
#     for j, col in enumerate(tbl.columns):
#         i_best = int(np.argmax(tbl[col].values))
#         mtable[i_best + 1, j].set_facecolor('#c9e4b4')

#     ax.set_title(judul, fontsize=13, weight='bold', pad=18)
#     plt.savefig(out_path(fname), dpi=200, bbox_inches='tight')
#     plt.close(fig)

# print("\nMembuat tabel metrik per skenario ablasi...")
# tabel_per_skenario = {}

# for sc in SCENARIOS:
#     tbl = (df_ablation[df_ablation['Skenario'] == sc]
#            .set_index('Algoritma')[METRIC_COLS]
#            .round(4))
#     tabel_per_skenario[sc] = tbl

#     print(f"\n=== TABEL SKENARIO {sc} ===")
#     print(tbl.to_string())

#     slug = sc.split('. ')[1].lower().replace(' + ', '_').replace(' ', '_')
#     render_tabel_png(tbl, f"Skenario {sc}", f"tabel_ablasi_{sc[0]}_{slug}.png")

#     # Ekspor LaTeX siap \input{} ke Bab IV
#     with open(out_path(f"tabel_ablasi_{sc[0]}_{slug}.tex"), "w", encoding="utf-8") as f:
#         f.write(tbl.to_latex(
#             float_format="%.4f",
#             caption=f"Metrik evaluasi Skenario {sc}",
#             label=f"tab:ablasi{sc[0]}",
#             position="htbp"
#         ))

# print("-> 4 tabel tersimpan (.png, .tex) per skenario")


# # # ---------------------------------------------------------
# # # FASE 5: AMBANG OPTIMAL VIA YOUDEN'S J
# # #  -> PERMINTAAN #2
# # #  Youden J = TPR - FPR  (jarak vertikal terjauh kurva ROC ke garis diagonal).
# # #  Threshold dengan J tertinggi = titik yang paling menyeimbangkan
# # #  false positive rate dan false negative rate.
# # # ---------------------------------------------------------
# # print("\nFASE 5: MENCARI AMBANG OPTIMAL (YOUDEN'S J)\n" + "-" * 50)

# # def find_youden(y_true, y_proba):
# #     """Cari threshold yang memaksimalkan J = TPR - FPR."""
# #     fpr, tpr, thr = roc_curve(y_true, y_proba)
# #     J = tpr - fpr
# #     ix = int(np.argmax(J))
# #     # thr[0] pada sklearn baru bisa bernilai inf; argmax J tidak akan memilihnya
# #     # karena di titik itu TPR=FPR=0 sehingga J=0.
# #     return {'threshold': float(thr[ix]), 'J': float(J[ix]),
# #             'fpr': float(fpr[ix]), 'tpr': float(tpr[ix])}

# # # --- (Alternatif untuk konteks skrining medis) ---
# # # Youden membobot sensitivity & specificity SAMA. Jika false negative dianggap
# # # jauh lebih mahal (pasien hipertensi lolos), pilih threshold terendah yang
# # # masih memenuhi target recall, misal >= 0.90:
# # #
# # # def threshold_for_recall(y_true, y_proba, target=0.90):
# # #     prec, rec, thr = precision_recall_curve(y_true, y_proba)
# # #     # thr sepanjang len(rec)-1; sejajarkan dengan rec[:-1]
# # #     ok = np.where(rec[:-1] >= target)[0]
# # #     return float(thr[ok[-1]]) if len(ok) else 0.5
# # # ---------------------------------------------------

# # youden_info = {}
# # predictions_youden = {}
# # rows = []

# # for name in champion_models:
# #     y_proba = predictions_champion[name]['y_proba']
# #     info = find_youden(y_test, y_proba)
# #     youden_info[name] = info

# #     thr = info['threshold']
# #     y_pred_def = predictions_champion[name]['y_pred']       # prediksi @ threshold 0.5
# #     y_pred_opt = (y_proba >= thr).astype(int)      # roc_curve memakai konvensi score >= thr
# #     predictions_youden[name] = y_pred_opt

# #     # Metrik LENGKAP @0.5 (default) vs @Youden -> dua baris per model
# #     m_def = evaluate(y_test, y_pred_def, y_proba)
# #     m_opt = evaluate(y_test, y_pred_opt, y_proba)

# #     row_def = {'Model': name, 'Ambang': '0.5 (default)', 'Threshold': 0.5, 'J': np.nan}
# #     row_def.update({k: round(v, 4) for k, v in m_def.items()})
# #     rows.append(row_def)

# #     row_opt = {'Model': name, 'Ambang': 'Youden', 'Threshold': round(thr, 4), 'J': round(info['J'], 4)}
# #     row_opt.update({k: round(v, 4) for k, v in m_opt.items()})
# #     rows.append(row_opt)

# # df_youden = pd.DataFrame(rows).set_index(['Model', 'Ambang'])
# # print(df_youden)
# # print("\nCatatan:")
# # print(" - ROC-AUC & PR-AUC dihitung dari probabilitas (ranking), BUKAN dari label,")
# # print("   sehingga nilainya IDENTIK di baris 0.5 dan Youden (tidak dipengaruhi threshold).")
# # print(" - Yang bergeser hanya Recall, Specificity, F1-Score, dan Akurasi.")
# # print(" - J = Youden's index (TPR - FPR) pada ambang terpilih.")




# # ---------------------------------------------------------
# # FASE 5: AMBANG OPTIMAL VIA YOUDEN'S J 
# #  -> Mencari threshold menggunakan TRAINING SET
# #  -> Mengevaluasi threshold tersebut pada TEST SET
# # ---------------------------------------------------------
# print("\nFASE 5: MENCARI AMBANG OPTIMAL (YOUDEN'S J)\n" + "-" * 50)

# def find_youden(y_true, y_proba):
#     """Cari threshold yang memaksimalkan J = TPR - FPR."""
#     fpr, tpr, thr = roc_curve(y_true, y_proba)
#     J = tpr - fpr
#     ix = int(np.argmax(J))
#     return {'threshold': float(thr[ix]), 'J': float(J[ix])}

# youden_info = {}
# predictions_youden = {}
# rows = []

# for name in champion_models:
#     model = champion_models[name]
    
#     # 1. CARI THRESHOLD MENGGUNAKAN DATA TRAIN (Mencegah Leakage)
#     y_train_proba = model.predict_proba(X_train)[:, 1]
#     info_train = find_youden(y_train, y_train_proba)
#     thr_opt = info_train['threshold']
    
#     # 2. APLIKASIKAN THRESHOLD KE DATA TEST
#     y_test_proba = predictions_champion[name]['y_proba']
#     y_pred_def = predictions_champion[name]['y_pred']      # Prediksi bawaan (threshold 0.5)
#     y_pred_opt = (y_test_proba >= thr_opt).astype(int)     # Prediksi dengan threshold Youden
    
#     predictions_youden[name] = y_pred_opt

#     # 3. EVALUASI METRIK PADA DATA TEST
#     m_def = evaluate(y_test, y_pred_def, y_test_proba)
#     m_opt = evaluate(y_test, y_pred_opt, y_test_proba)

#     # 4. AMBIL KOORDINAT FPR & TPR UNTUK VISUALISASI FASE 6
#     # Karena kita mencari threshold di Train, kita hitung ulang FPR dan TPR di Test 
#     # agar titik yang digambar di Fase 6 jatuh persis di garis kurva ROC Test.
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt).ravel()
#     test_tpr = tp / (tp + fn)
#     test_fpr = fp / (fp + tn)

#     youden_info[name] = {
#         'threshold': thr_opt,
#         'fpr': test_fpr,
#         'tpr': test_tpr
#     }

#     # 5. SIMPAN KE TABEL
#     row_def = {'Model': name, 'Ambang': '0.5 (default)', 'Threshold': 0.5, 'J (Train)': np.nan}
#     row_def.update({k: round(v, 4) for k, v in m_def.items()})
#     rows.append(row_def)

#     row_opt = {'Model': name, 'Ambang': 'Youden', 'Threshold': round(thr_opt, 4), 'J (Train)': round(info_train['J'], 4)}
#     row_opt.update({k: round(v, 4) for k, v in m_opt.items()})
#     rows.append(row_opt)

# df_youden = pd.DataFrame(rows).set_index(['Model', 'Ambang'])
# print(df_youden)
# print("\nCatatan:")
# print(" - Threshold Youden (J) sekarang dihitung dari Training Set untuk mencegah leakage.")
# print(" - Evaluasi F1-Score, Recall, dll tetap murni dilakukan di Test Set menggunakan threshold tersebut.")



# # ---------------------------------------------------------
# # FASE 6: VISUALISASI EVALUASI
# # ---------------------------------------------------------
# print("\nMEMBUAT VISUALISASI EVAL...\n" + "-" * 50)
# class_labels = ['Normal', 'Hipertensi']

# # 6a. Confusion matrix pada AMBANG YOUDEN
# n_models = len(champion_models)
# fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
# if n_models == 1:
#     axes = [axes]
# for ax, name in zip(axes, champion_models):
#     cm = confusion_matrix(y_test, predictions_youden[name])
#     ConfusionMatrixDisplay(cm, display_labels=class_labels).plot(ax=ax, cmap='Blues', colorbar=False)
#     ax.set_title(f"{name}\n(thr={youden_info[name]['threshold']:.3f})")
#     ax.grid(False)
# plt.suptitle("Confusion Matrix @ Ambang Youden (Test Set)", fontsize=13)
# plt.tight_layout()
# plt.savefig(out_path('confusion_matrix_youden.png'), dpi=150, bbox_inches='tight')
# plt.close(fig)
# print("-> confusion_matrix_youden.png tersimpan")

# # 6b. Kurva ROC + tanda titik Youden tiap model
# plt.figure(figsize=(8, 6))
# for name in champion_models:
#     y_proba = predictions_champion[name]['y_proba']
#     fpr, tpr, _ = roc_curve(y_test, y_proba)
#     auc_val = roc_auc_score(y_test, y_proba)
#     line, = plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc_val:.3f})')
#     # tandai titik Youden pada warna garis yang sama
#     plt.scatter(youden_info[name]['fpr'], youden_info[name]['tpr'],
#                 color=line.get_color(), edgecolor='black', zorder=5, s=70)
# plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
# plt.xlabel('False Positive Rate (1 - Specificity)')
# plt.ylabel('True Positive Rate (Sensitivity / Recall)')
# plt.title("Kurva ROC + Titik Youden (Test Set)")
# plt.legend(loc='lower right')
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig(out_path('roc_curve_youden.png'), dpi=150, bbox_inches='tight')
# plt.close()
# print("-> roc_curve_youden.png tersimpan (titik hitam = ambang Youden tiap model)")


# # ---------------------------------------------------------
# # FASE 7: SIMPAN MODEL + THRESHOLD + SKEMA FITUR
# #  Menyimpan threshold Youden bersama model agar backend memprediksi
# #  dengan ambang yang benar (bukan 0.5 default).
# # ---------------------------------------------------------
# best_xgb = champion_models['XGBoost']
# youden_thr_xgb = youden_info['XGBoost']['threshold']

# artifact = {
#     'model': best_xgb,
#     'threshold': youden_thr_xgb,
#     'feature_names': list(X_train.columns),  # untuk validasi urutan kolom saat inference
# }
# model_filename = 'xgboost_hipertensi_model.pkl'
# joblib.dump(artifact, model_filename)
# print(f"\nModel + threshold + skema fitur disimpan: {model_filename}")
# print(f"   threshold Youden XGBoost = {youden_thr_xgb:.4f}")

# # --- Cara pakai di backend ---
# # art = joblib.load('xgboost_hipertensi_model.pkl')
# # X_new = df_pasien[art['feature_names']]              # jaga urutan kolom
# # proba = art['model'].predict_proba(X_new)[:, 1]
# # pred  = (proba >= art['threshold']).astype(int)      # pakai ambang Youden, bukan 0.5







# import pandas as pd
# import numpy as np
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import (
#     roc_auc_score, recall_score, f1_score, confusion_matrix, accuracy_score,
#     precision_recall_curve, auc, roc_curve, ConfusionMatrixDisplay
# )

# # Import Models
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier
# from sklearn.metrics import fbeta_score, make_scorer
# import seaborn as sns

# # SIMPAN MODEL
# import joblib

# # Visualisasi
# import matplotlib
# matplotlib.use('Agg')   # backend non-interaktif: hanya simpan file, tidak buka window
# import matplotlib.pyplot as plt

# import warnings
# # CATATAN: Saat menjalankan eksperimen baseline LR (scaled vs no-scaling),
# # sebaiknya KOMENTARI baris di bawah ini agar ConvergenceWarning terlihat.
# # Warning itu adalah bukti kunci bahwa versi unscaled tidak konvergen.
# warnings.filterwarnings('ignore')

# import os

# OUTPUT_DIR = "hasil_modelling"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# def out_path(filename):
#     return os.path.join(OUTPUT_DIR, filename)





# # ---------------------------------------------------------
# # FASE 1: MEMBACA DATA HASIL PREPROCESSING
# # (Split & imputasi sudah dilakukan di tahap data preparation)
# # ---------------------------------------------------------
# TARGET_COL = "label_hypertension"

# train_df = pd.read_csv("train_hipertensi.csv")
# test_df  = pd.read_csv("test_hipertensi.csv")

# print("=== DATA TRAIN ===", train_df.shape)
# print("=== DATA TEST  ===", test_df.shape)

# X_train = train_df.drop(columns=[TARGET_COL])
# y_train = train_df[TARGET_COL]
# X_test  = test_df.drop(columns=[TARGET_COL])
# y_test  = test_df[TARGET_COL]

# print("\nJumlah fitur :", X_train.shape[1])
# print("\nDistribusi kelas TRAIN")
# print((y_train.value_counts(normalize=True) * 100).round(2))
# print("\nDistribusi kelas TEST")
# print((y_test.value_counts(normalize=True) * 100).round(2))
# print("\nMissing value TRAIN :", X_train.isna().sum().sum())
# print("Missing value TEST  :", X_test.isna().sum().sum())


# # Rasio kelas untuk XGBoost
# neg, pos = np.bincount(y_train)
# scale_pos_weight = neg / pos


# # =========================================================
# # FUNGSI BANTU EVALUASI
# # =========================================================
# def evaluate(y_true, y_pred, y_proba):
#     """Kembalikan dict metrik standar."""
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     specificity = tn / (tn + fp)
#     precision, rec, _ = precision_recall_curve(y_true, y_proba)
#     return {
#         'ROC-AUC': roc_auc_score(y_true, y_proba),
#         'Recall (Sens)': recall_score(y_true, y_pred),
#         'Specificity': specificity,
#         'PR-AUC': auc(rec, precision),
#         'F1-Score': f1_score(y_true, y_pred),
#         'Akurasi': accuracy_score(y_true, y_pred),
#     }


# weighted_models = {}
# weighted_predictions = {}

# # ---------------------------------------------------------
# # FASE 2 & 3: ABLATION STUDY (2 SKENARIO)
# # ---------------------------------------------------------
# print("\nMEMULAI ABLATION STUDY (2 SKENARIO)...\n" + "-" * 50)

# # Konfigurasi model untuk dua skenario:
# # 1) Pure Baseline
# # 2) Baseline + Class Weighting
# configs = {
#     'Logistic Regression': {
#         'base': LogisticRegression(random_state=42, max_iter=1000),
#         'base_weighted': LogisticRegression(
#             random_state=42, max_iter=1000, class_weight='balanced'
#         ),
#     },
#     'Random Forest': {
#         'base': RandomForestClassifier(random_state=42),
#         'base_weighted': RandomForestClassifier(
#             random_state=42, class_weight='balanced'
#         ),
#     },
#     'XGBoost': {
#         'base': XGBClassifier(random_state=42, eval_metric='logloss'),
#         'base_weighted': XGBClassifier(
#             random_state=42,
#             eval_metric='logloss',
#             scale_pos_weight=scale_pos_weight,
#         ),
#     },
#     'CatBoost': {
#         'base': CatBoostClassifier(random_state=42, verbose=0),
#         'base_weighted': CatBoostClassifier(
#             random_state=42,
#             verbose=0,
#             auto_class_weights='Balanced',
#         ),
#     }
# }

# results_ablation = []

# for algo_name, cfg in configs.items():
#     print(f"\nMengeksekusi {algo_name}...")

#     # Skenario 1: Pure Baseline
#     model_s1 = cfg['base']
#     model_s1.fit(X_train, y_train)
#     res_s1 = evaluate(
#         y_test,
#         model_s1.predict(X_test),
#         model_s1.predict_proba(X_test)[:, 1],
#     )
#     res_s1.update({'Algoritma': algo_name, 'Skenario': '1. Pure Baseline'})
#     results_ablation.append(res_s1)

#     # Skenario 2: Baseline + Class Weighting
#     model_s2 = cfg['base_weighted']
#     model_s2.fit(X_train, y_train)

#     # Prediksi cukup sekali
#     y_pred = model_s2.predict(X_test)
#     y_proba = model_s2.predict_proba(X_test)[:, 1]

#     res_s2 = evaluate(
#         y_test,
#         y_pred,
#         y_proba,
#     )
#     res_s2.update({
#         'Algoritma': algo_name,
#         'Skenario': '2. Baseline + Class Weighting'
#     })
#     results_ablation.append(res_s2)

#     weighted_models[algo_name] = model_s2
#     weighted_predictions[algo_name] = {
#         "y_pred": y_pred,
#         "y_proba": y_proba
#     }
    


# # --- TABEL HASIL ABLATION STUDY ---
# df_ablation = pd.DataFrame(results_ablation)

# # Reorder columns agar rapi
# cols = ['Algoritma', 'Skenario', 'ROC-AUC', 'Recall (Sens)', 'Specificity', 'F1-Score', 'Akurasi', 'PR-AUC']
# df_ablation = df_ablation[cols]

# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)

# print("\nTABEL HASIL ABLATION STUDY (2 SKENARIO)")
# print(df_ablation.round(4).to_string(index=False))

# # --- VISUALISASI ABLATION STUDY ---
# print("\nMembuat Visualisasi Ablation Study...")

# METRIC_COLS = ['ROC-AUC', 'Recall (Sens)', 'Specificity', 'F1-Score', 'Akurasi', 'PR-AUC']
# SCENARIOS = ['1. Pure Baseline', '2. Baseline + Weighting']


# def render_tabel_png(tbl, judul, fname):
#     """Render DataFrame jadi PNG tabel siap tempel ke laporan."""
#     fig, ax = plt.subplots(figsize=(1.55 * len(tbl.columns) + 2.2, 0.6 * len(tbl) + 1.4))
#     ax.axis('off')
#     mtable = ax.table(
#         cellText=[[f"{v:.4f}" for v in row] for row in tbl.values],
#         rowLabels=tbl.index,
#         colLabels=tbl.columns,
#         cellLoc='center',
#         rowLoc='left',
#         loc='center',
#     )
#     mtable.auto_set_font_size(False)
#     mtable.set_fontsize(10)
#     mtable.scale(1, 1.6)

#     # Header styling
#     for j in range(len(tbl.columns)):
#         cell = mtable[0, j]
#         cell.set_facecolor('#2f4b7c')
#         cell.set_text_props(color='white', weight='bold')

#     for i in range(len(tbl)):
#         mtable[i + 1, -1].set_text_props(weight='bold')

#     # Tandai nilai terbaik tiap kolom (semua metrik di sini: makin tinggi makin baik)
#     for j, col in enumerate(tbl.columns):
#         i_best = int(np.argmax(tbl[col].values))
#         mtable[i_best + 1, j].set_facecolor('#c9e4b4')

#     ax.set_title(judul, fontsize=13, weight='bold', pad=18)
#     plt.savefig(out_path(fname), dpi=200, bbox_inches='tight')
#     plt.close(fig)


# print("\nMembuat tabel metrik per skenario ablasi...")
# tabel_per_skenario = {}

# for sc in SCENARIOS:
#     tbl = (
#         df_ablation[df_ablation['Skenario'] == sc]
#         .set_index('Algoritma')[METRIC_COLS]
#         .round(4)
#     )
#     tabel_per_skenario[sc] = tbl

#     print(f"\n=== TABEL SKENARIO {sc} ===")
#     print(tbl.to_string())

#     slug = sc.split('. ')[1].lower().replace(' + ', '_').replace(' ', '_')
#     render_tabel_png(tbl, f"Skenario {sc}", f"tabel_ablasi_{sc[0]}_{slug}.png")

#     # Ekspor LaTeX siap \input{} ke Bab IV
#     with open(out_path(f"tabel_ablasi_{sc[0]}_{slug}.tex"), "w", encoding="utf-8") as f:
#         f.write(
#             tbl.to_latex(
#                 float_format="%.4f",
#                 caption=f"Metrik evaluasi Skenario {sc}",
#                 label=f"tab:ablasi{sc[0]}",
#                 position="htbp",
#             )
#         )

# print("-> 2 tabel tersimpan (.png, .tex) per skenario")


# #  cari model terbaik menggunakan ROC-AUC
# weighted_df = df_ablation[
#     df_ablation["Skenario"] == "2. Baseline + Weighting"
# ]

# best_algorithm = weighted_df.loc[
#     weighted_df["ROC-AUC"].idxmax(),
#     "Algoritma"
# ]

# print(f"Model terbaik: {best_algorithm}")


# best_model = weighted_models[best_algorithm]

# best_prediction = weighted_predictions[best_algorithm]

# model = best_model
# name = best_algorithm



# # ---------------------------------------------------------
# # FASE 6A: CONFUSION MATRIX
# # ---------------------------------------------------------
# print("\nMEMBUAT CONFUSION MATRIX...\n" + "-" * 50)

# class_labels = ['Normal', 'Hipertensi']

# fig, ax = plt.subplots(figsize=(5, 4.5))

# cm = confusion_matrix(y_test, predictions_youden)

# ConfusionMatrixDisplay(
#     confusion_matrix=cm,
#     display_labels=class_labels
# ).plot(
#     ax=ax,
#     cmap="Blues",
#     colorbar=False
# )

# ax.set_title(
#     f"{best_algorithm}\n(thr={youden_thr:.3f})"
# )
# ax.grid(False)

# plt.tight_layout()
# plt.savefig(
#     out_path("confusion_matrix_youden.png"),
#     dpi=150,
#     bbox_inches="tight"
# )
# plt.close()

# print("-> confusion_matrix_youden.png tersimpan")


# # ---------------------------------------------------------
# # FASE 6B: ROC CURVE
# # ---------------------------------------------------------
# print("\nMEMBUAT ROC CURVE...\n" + "-" * 50)

# plt.figure(figsize=(8,6))

# fpr, tpr, _ = roc_curve(y_test, best_prediction["y_proba"])
# auc_val = roc_auc_score(y_test, best_prediction["y_proba"])

# line, = plt.plot(
#     fpr,
#     tpr,
#     linewidth=2,
#     label=f'{best_algorithm} (AUC = {auc_val:.3f})'
# )

# plt.scatter(
#     youden_info["fpr"],
#     youden_info["tpr"],
#     color=line.get_color(),
#     edgecolor="black",
#     zorder=5,
#     s=70,
#     label="Youden Threshold"
# )

# plt.plot(
#     [0,1],
#     [0,1],
#     "k--",
#     linewidth=1,
#     label="Random (AUC = 0.500)"
# )

# plt.xlabel("False Positive Rate (1 - Specificity)")
# plt.ylabel("True Positive Rate (Sensitivity / Recall)")
# plt.title(f"ROC Curve - {best_algorithm}")
# plt.legend(loc="lower right")
# plt.grid(alpha=0.3)

# plt.tight_layout()
# plt.savefig(
#     out_path("roc_curve_youden.png"),
#     dpi=150,
#     bbox_inches="tight"
# )
# plt.close()

# print("-> roc_curve_youden.png tersimpan")




# artifact = {
#     "model": best_model,
#     "threshold": youden_thr,
#     "feature_names": list(X_train.columns)
# }


# model_filename = 'best_hipertensi_model.pkl'
# joblib.dump(artifact, model_filename)
# print(f"\nModel + threshold + skema fitur disimpan: {model_filename}")














import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score, confusion_matrix, accuracy_score,
    precision_recall_curve, auc, roc_curve, ConfusionMatrixDisplay
)

# Import Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# SIMPAN MODEL
import joblib

# Visualisasi
import matplotlib
matplotlib.use('Agg')   # backend non-interaktif: hanya simpan file, tidak buka window
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import os

OUTPUT_DIR = "hasil_modelling"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def out_path(filename):
    return os.path.join(OUTPUT_DIR, filename)


# ---------------------------------------------------------
# FASE 1: MEMBACA DATA HASIL PREPROCESSING
#  (Flowchart: "Baca data train dan data test dari preprocessing",
#              "Pisahkan fitur non-labeling dan status hipertensi",
#              "Hitung rasio kelas (scale_pos_weight) untuk XGBoost")
#  Split & imputasi sudah dilakukan di tahap data preparation.
# ---------------------------------------------------------
TARGET_COL = "label_hypertension"

train_df = pd.read_csv("train_hipertensi.csv")
test_df = pd.read_csv("test_hipertensi.csv")

print("=== DATA TRAIN ===", train_df.shape)
print("=== DATA TEST  ===", test_df.shape)

X_train = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL]
X_test = test_df.drop(columns=[TARGET_COL])
y_test = test_df[TARGET_COL]

print("\nJumlah fitur :", X_train.shape[1])
print("\nDistribusi kelas TRAIN")
print((y_train.value_counts(normalize=True) * 100).round(2))
print("\nDistribusi kelas TEST")
print((y_test.value_counts(normalize=True) * 100).round(2))
print("\nMissing value TRAIN :", X_train.isna().sum().sum())
print("Missing value TEST  :", X_test.isna().sum().sum())

# Rasio kelas untuk XGBoost (class weighting)
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos


# =========================================================
# FUNGSI BANTU EVALUASI
#  (Flowchart: "Evaluasi Model" -> ROC-AUC, PR-AUC, F1, Recall,
#   Specificity, Accuracy)
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


weighted_models = {}
weighted_predictions = {}

# ---------------------------------------------------------
# FASE 2 & 3: ABLATION STUDY (2 SKENARIO)
#  (Flowchart: "Jalankan skenario" ->
#     1. Pure Baseline (tanpa class weighting)
#     2. Baseline dengan class weighting
#   lalu loop antar algoritma & antar skenario, tanpa tuning)
# ---------------------------------------------------------
print("\nMEMULAI ABLATION STUDY (2 SKENARIO)...\n" + "-" * 50)

# CATATAN (ketidaksesuaian flowchart #4):
# Kotak "Pilih algoritma" pada flowchart menuliskan:
#   Logistic Regression, CatBoost, XGBoost, CatBoost
# ("CatBoost" muncul dua kali, tanpa Random Forest). Ini kemungkinan
# besar salah ketik pada diagram -- melatih CatBoost dua kali tidak
# bermakna. Di sini dipakai 4 algoritma distinct (LR, RF, XGB, CatBoost).
# >> Jika diagram memang dimaksud tanpa Random Forest, hapus entri
#    'Random Forest' di bawah dan sesuaikan diagram/laporan.
configs = {
    'Logistic Regression': {
        'base': LogisticRegression(random_state=42, max_iter=1000),
        'base_weighted': LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced'
        ),
    },
    'Random Forest': {
        'base': RandomForestClassifier(random_state=42),
        'base_weighted': RandomForestClassifier(
            random_state=42, class_weight='balanced'
        ),
    },
    'XGBoost': {
        'base': XGBClassifier(random_state=42, eval_metric='logloss'),
        'base_weighted': XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
        ),
    },
    'CatBoost': {
        'base': CatBoostClassifier(random_state=42, verbose=0),
        'base_weighted': CatBoostClassifier(
            random_state=42,
            verbose=0,
            auto_class_weights='Balanced',
        ),
    }
}

# Label skenario dibuat SATU sumber kebenaran agar konsisten
# di seluruh skrip (perbaikan ketidaksesuaian #3).
SC_BASELINE = '1. Pure Baseline'
SC_WEIGHTED = '2. Baseline + Class Weighting'

results_ablation = []

for algo_name, cfg in configs.items():
    print(f"\nMengeksekusi {algo_name}...")

    # Skenario 1: Pure Baseline (tanpa class weighting)
    model_s1 = cfg['base']
    model_s1.fit(X_train, y_train)
    res_s1 = evaluate(
        y_test,
        model_s1.predict(X_test),
        model_s1.predict_proba(X_test)[:, 1],
    )
    res_s1.update({'Algoritma': algo_name, 'Skenario': SC_BASELINE})
    results_ablation.append(res_s1)

    # Skenario 2: Baseline + Class Weighting
    model_s2 = cfg['base_weighted']
    model_s2.fit(X_train, y_train)

    # Prediksi cukup sekali
    y_pred = model_s2.predict(X_test)
    y_proba = model_s2.predict_proba(X_test)[:, 1]

    res_s2 = evaluate(
        y_test,
        y_pred,
        y_proba,
    )
    res_s2.update({'Algoritma': algo_name, 'Skenario': SC_WEIGHTED})
    results_ablation.append(res_s2)

    # Hanya model Skenario 2 yang disimpan (kandidat champion)
    weighted_models[algo_name] = model_s2
    weighted_predictions[algo_name] = {
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


# --- TABEL HASIL ABLATION STUDY ---
df_ablation = pd.DataFrame(results_ablation)

# Reorder columns agar rapi
cols = ['Algoritma', 'Skenario', 'ROC-AUC', 'Recall (Sens)', 'Specificity',
        'F1-Score', 'Akurasi', 'PR-AUC']
df_ablation = df_ablation[cols]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\nTABEL HASIL ABLATION STUDY (2 SKENARIO)")
print(df_ablation.round(4).to_string(index=False))

# --- VISUALISASI ABLATION STUDY ---
print("\nMembuat Visualisasi Ablation Study...")

METRIC_COLS = ['ROC-AUC', 'Recall (Sens)', 'Specificity', 'F1-Score',
               'Akurasi', 'PR-AUC']
SCENARIOS = [SC_BASELINE, SC_WEIGHTED]


def render_tabel_png(tbl, judul, fname):
    """Render DataFrame jadi PNG tabel siap tempel ke laporan."""
    fig, ax = plt.subplots(figsize=(1.55 * len(tbl.columns) + 2.2, 0.6 * len(tbl) + 1.4))
    ax.axis('off')
    mtable = ax.table(
        cellText=[[f"{v:.4f}" for v in row] for row in tbl.values],
        rowLabels=tbl.index,
        colLabels=tbl.columns,
        cellLoc='center',
        rowLoc='left',
        loc='center',
    )
    mtable.auto_set_font_size(False)
    mtable.set_fontsize(10)
    mtable.scale(1, 1.6)

    # Header styling
    for j in range(len(tbl.columns)):
        cell = mtable[0, j]
        cell.set_facecolor('#2f4b7c')
        cell.set_text_props(color='white', weight='bold')

    for i in range(len(tbl)):
        mtable[i + 1, -1].set_text_props(weight='bold')

    # Tandai nilai terbaik tiap kolom (semua metrik: makin tinggi makin baik)
    for j, col in enumerate(tbl.columns):
        i_best = int(np.argmax(tbl[col].values))
        mtable[i_best + 1, j].set_facecolor('#c9e4b4')

    ax.set_title(judul, fontsize=13, weight='bold', pad=18)
    plt.savefig(out_path(fname), dpi=200, bbox_inches='tight')
    plt.close(fig)


print("\nMembuat tabel metrik per skenario ablasi...")
tabel_per_skenario = {}

for sc in SCENARIOS:
    tbl = (
        df_ablation[df_ablation['Skenario'] == sc]
        .set_index('Algoritma')[METRIC_COLS]
        .round(4)
    )
    tabel_per_skenario[sc] = tbl

    print(f"\n=== TABEL SKENARIO {sc} ===")
    print(tbl.to_string())

    slug = sc.split('. ')[1].lower().replace(' + ', '_').replace(' ', '_')
    render_tabel_png(tbl, f"Skenario {sc}", f"tabel_ablasi_{sc[0]}_{slug}.png")

    # Ekspor LaTeX siap \input{} ke Bab IV
    with open(out_path(f"tabel_ablasi_{sc[0]}_{slug}.tex"), "w", encoding="utf-8") as f:
        f.write(
            tbl.to_latex(
                float_format="%.4f",
                caption=f"Metrik evaluasi Skenario {sc}",
                label=f"tab:ablasi{sc[0]}",
                position="htbp",
            )
        )

print("-> 2 tabel tersimpan (.png, .tex) per skenario")


# ---------------------------------------------------------
# PEMILIHAN MODEL TERBAIK
#  (Flowchart: "Model terbaik dipilih dari skenario 2 (baseline dengan
#   class weighting) dengan target metrik ROC-AUC")
# ---------------------------------------------------------
weighted_df = df_ablation[df_ablation["Skenario"] == SC_WEIGHTED]

best_algorithm = weighted_df.loc[
    weighted_df["ROC-AUC"].idxmax(),
    "Algoritma",
]

print(f"\nModel terbaik (champion): {best_algorithm}")

best_model = weighted_models[best_algorithm]
best_prediction = weighted_predictions[best_algorithm]


# ---------------------------------------------------------
# FASE 5: AMBANG OPTIMAL VIA YOUDEN'S J
#  (Flowchart, berurutan:
#    14) "Hitung Probabilitas pada Data Train menggunakan model terbaik"
#    15) "Cari nilai ambang (Threshold) optimal menggunakan Youden's J
#         (J = TPR - FPR) dari hasil Data Train"
#    16) "Aplikasikan ambang tersebut pada probabilitas Data Test"
#    17) "Evaluasi ulang metrik kinerja pada Data Test dengan ambang baru")
#
#  Threshold DICARI di TRAIN lalu DIPAKAI di TEST -> mencegah leakage.
# ---------------------------------------------------------
print("\nFASE 5: MENCARI AMBANG OPTIMAL (YOUDEN'S J)\n" + "-" * 50)


def find_youden(y_true, y_proba):
    """Cari threshold yang memaksimalkan J = TPR - FPR."""
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    J = tpr - fpr
    ix = int(np.argmax(J))
    return {'threshold': float(thr[ix]), 'J': float(J[ix])}


# (14) Probabilitas pada DATA TRAIN memakai model terbaik
y_train_proba = best_model.predict_proba(X_train)[:, 1]

# (15) Threshold optimal dari DATA TRAIN
info_train = find_youden(y_train, y_train_proba)
youden_thr = info_train['threshold']

# (16) Aplikasikan threshold ke probabilitas DATA TEST
y_test_proba = best_prediction['y_proba']
y_pred_default = best_prediction['y_pred']                    # ambang 0.5 (default)
predictions_youden = (y_test_proba >= youden_thr).astype(int)  # ambang Youden

# Koordinat titik Youden pada kurva ROC DATA TEST (dipakai di FASE 6B)
tn, fp, fn, tp = confusion_matrix(y_test, predictions_youden).ravel()
youden_info = {
    'threshold': youden_thr,
    'fpr': fp / (fp + tn),
    'tpr': tp / (tp + fn),
}

# (17) Evaluasi ulang pada DATA TEST: default 0.5 vs Youden
m_default = evaluate(y_test, y_pred_default, y_test_proba)
m_youden = evaluate(y_test, predictions_youden, y_test_proba)

df_youden = pd.DataFrame(
    [
        {
            'Ambang': '0.5 (default)',
            'Threshold': 0.5,
            'J (Train)': np.nan,
            **{k: round(v, 4) for k, v in m_default.items()},
        },
        {
            'Ambang': 'Youden',
            'Threshold': round(youden_thr, 4),
            'J (Train)': round(info_train['J'], 4),
            **{k: round(v, 4) for k, v in m_youden.items()},
        },
    ]
).set_index('Ambang')

print(f"Model terbaik (champion): {best_algorithm}")
print(df_youden.to_string())
print("\nCatatan:")
print(" - Threshold Youden dihitung dari DATA TRAIN (mencegah leakage).")
print(" - Metrik akhir dievaluasi pada DATA TEST dengan ambang tersebut.")
print(" - ROC-AUC & PR-AUC identik di kedua baris (dihitung dari probabilitas,")
print("   bukan dari label), yang bergeser hanya Recall/Specificity/F1/Akurasi.")


# ---------------------------------------------------------
# FASE 6A: CONFUSION MATRIX (pada ambang Youden, Data Test)
#  (Flowchart: "Visualisasi Evaluasi -> buat Confusion Matrix")
# ---------------------------------------------------------
print("\nMEMBUAT CONFUSION MATRIX...\n" + "-" * 50)

class_labels = ['Normal', 'Hipertensi']

fig, ax = plt.subplots(figsize=(5, 4.5))

cm = confusion_matrix(y_test, predictions_youden)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_labels,
).plot(
    ax=ax,
    cmap="Blues",
    colorbar=False,
)

ax.set_title(f"{best_algorithm}\n(thr={youden_thr:.3f})")
ax.grid(False)

plt.tight_layout()
plt.savefig(out_path("confusion_matrix_youden.png"), dpi=150, bbox_inches="tight")
plt.close()

print("-> confusion_matrix_youden.png tersimpan")


# ---------------------------------------------------------
# FASE 6B: ROC CURVE (beserta titik Youden, Data Test)
#  (Flowchart: "Visualisasi Evaluasi -> Kurva ROC (beserta titik Youden)")
# ---------------------------------------------------------
print("\nMEMBUAT ROC CURVE...\n" + "-" * 50)

plt.figure(figsize=(8, 6))

fpr, tpr, _ = roc_curve(y_test, best_prediction["y_proba"])
auc_val = roc_auc_score(y_test, best_prediction["y_proba"])

line, = plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f'{best_algorithm} (AUC = {auc_val:.3f})',
)

plt.scatter(
    youden_info["fpr"],
    youden_info["tpr"],
    color=line.get_color(),
    edgecolor="black",
    zorder=5,
    s=70,
    label="Youden Threshold",
)

plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")

plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity / Recall)")
plt.title(f"ROC Curve - {best_algorithm}")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(out_path("roc_curve_youden.png"), dpi=150, bbox_inches="tight")
plt.close()

print("-> roc_curve_youden.png tersimpan")


# ---------------------------------------------------------
# FASE 7: EKSPOR ARTIFACT MODEL
#  (Flowchart: "Ekspor Artifact Model -> menyimpan Champion Model,
#   nilai Threshold Youden, dan urutan fitur (Feature Names) ke file .pkl")
# ---------------------------------------------------------
artifact = {
    "model": best_model,
    "threshold": youden_thr,
    "feature_names": list(X_train.columns),  # jaga urutan kolom saat inference
}

model_filename = out_path('best_hipertensi_model.pkl')
joblib.dump(artifact, model_filename)
print(f"\nModel + threshold + skema fitur disimpan: {model_filename}")
print(f"   Champion  = {best_algorithm}")
print(f"   Threshold = {youden_thr:.4f}")

# --- Cara pakai di backend ---
# art = joblib.load('best_hipertensi_model.pkl')
# X_new = df_pasien[art['feature_names']]              # jaga urutan kolom
# proba = art['model'].predict_proba(X_new)[:, 1]
# pred  = (proba >= art['threshold']).astype(int)      # pakai ambang Youden, bukan 0.5