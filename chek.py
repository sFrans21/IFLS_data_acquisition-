import pandas as pd
import pyreadstat
import os

# Configuration
FILE_PATH = os.path.join('data', 'hh14_b3b_dta', 'b3b_cd3.dta')

def inspect_schema():
    if not os.path.exists(FILE_PATH):
        print(f"[FATAL] File not found: {FILE_PATH}")
        return

    df, meta = pyreadstat.read_dta(FILE_PATH)
    df.columns = df.columns.str.lower()
    
    # Filter for Diabetes to see context
    df_diabetes = df[df['cdtype'] == 'H'] # We know 'cdtype' exists from your log
    
    print(f"--- SCHEMA DUMP: {os.path.basename(FILE_PATH)} ---")
    print(f"Total Columns: {len(df.columns)}")
    print("Column Registry:")
    print(df.columns.tolist())
    
    print("\n--- METADATA LABELS (Variable Descriptions) ---")
    # We look for descriptions containing 'diagnosed' or 'know'
    if meta.column_names_to_labels:
        for col, label in meta.column_names_to_labels.items():
            # Check specifically for diagnosis-like keywords
            if any(x in label.lower() for x in ['know', 'tahu', 'diagnos', 'sakit']):
                print(f"[{col}]: {label}")

if __name__ == "__main__":
    inspect_schema()