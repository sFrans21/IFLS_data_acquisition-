import pandas as pd
import pyreadstat
import os
import sys
from typing import Optional, Tuple

# Configuration
DATA_DIR = 'data'
SUBFOLDER = 'hh14_b3b_dta'
TARGET_FILENAME = 'b3b_cd3.dta'
TARGET_CODES = {'H', 'h'}  # Set for O(1) lookup

def get_file_path(subfolder: str, filename: str) -> str:
    return os.path.join(DATA_DIR, subfolder, filename)

def load_data(filepath: str) -> Tuple[Optional[pd.DataFrame], Optional[object]]:
    """
    Loads STATA file securely. Returns DataFrame and Metadata.
    """
    if not os.path.exists(filepath):
        print(f"[ERR] File not found: {filepath}")
        return None, None

    try:
        df, meta = pyreadstat.read_dta(filepath)
        df.columns = df.columns.str.lower()
        return df, meta
    except Exception as e:
        print(f"[ERR] Failed to read DTA file: {e}")
        return None, None

def analyze_diabetes_presence(df: pd.DataFrame) -> None:
    """
    Scans the DataFrame for disease type columns and verifies the existence
    of diabetes codes.
    """
    # Vectorized search for the type column
    # We prefer 'cd01type' based on IFLS5 specs, but fallback to general 'type' matching
    type_col = next((c for c in df.columns if 'type' in c), None)

    if not type_col:
        print(f"[ERR] No type column found. Available columns: {df.columns.tolist()}")
        return

    print(f"[INFO] Identified disease type column: '{type_col}'")

    # Extract unique values for inspection
    unique_codes = set(df[type_col].unique())
    print(f"[INFO] Unique codes found: {sorted(list(unique_codes))}")

    # Intersection check (Set theory) for efficiency
    found_targets = unique_codes.intersection(TARGET_CODES)

    if found_targets:
        print(f"[SUCCESS] Target codes found: {found_targets}")
        
        # Filter Logic
        mask = df[type_col].isin(TARGET_CODES)
        df_diabetes = df[mask]
        
        # Diagnostic Check for Diagnosis Column
        if 'cd01' in df_diabetes.columns:
            # Normalize: 1 (Yes) should be prioritized. 
            # We sort by cd01 (1 before 3) and keep the first occurrence per pidlink.
            df_final = df_diabetes[['pidlink', 'cd01']].rename(columns={'cd01': 'is_diabetes'})
            df_final = df_final.sort_values('is_diabetes').drop_duplicates(subset='pidlink', keep='first')

            print(f"[INFO] Extracted {len(df_final)} unique records.")
            
            positive_cases = (df_final['is_diabetes'] == 1).sum()
            print(f"[INFO] Positive Diabetes Cases (Code 1): {positive_cases}")
        else:
            print("[ERR] Diagnosis column 'cd01' missing in filtered data.")
    else:
        print("[WARN] Target codes (H/h) NOT found in this file.")

def main():
    filepath = get_file_path(SUBFOLDER, TARGET_FILENAME)
    print(f"--- EXECUTION START: {TARGET_FILENAME} ---")
    
    df, _ = load_data(filepath)
    
    if df is not None:
        analyze_diabetes_presence(df)
    
    print("--- EXECUTION END ---")

if __name__ == "__main__":
    main()