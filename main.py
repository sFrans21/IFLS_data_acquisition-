# SCRIPT EKSTRAKSI AWAL
import pandas as pd
import pyreadstat

# Load data Tracking sebagai basis
df_trk, meta_trk = pyreadstat.read_dta('data/hh14_trk_dta/htrack.dta')

#Ambil pidlink dan status keberadaan individu
base_df = df_trk[['pidlink', 'age', 'sex']]

# Load data klinis dari Book US
df_us, meta_us = pyreadstat.read_dta('data/hh14_bus_dta/bus_cov')

#Ambil pidlink, Tekanan Darah (Sistolik/Diastolik), dan Berat/Tinggi
#Notes: variabel tensi biasanya ada di us07a1 dan us07a2, tepatnya di pengkuran 1 2
df_klinis = df_us[['pidlink', 'us07a1', 'us07a2', 'us01', 'us02']]

print("Eksplorasi awal berhasil. Data siap untuk di merging")