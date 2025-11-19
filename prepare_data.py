import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Tuple, List, Dict, Any

PROCESSED_DATA_FILENAME = 'processed_data.csv'
TARGET_SCALER_FILENAME = 'target_scaler.pkl'

def load_and_prepare_data(
    data_file_path: str,
    input_features: List[str],
    target_column: str,
    processed_dir: str,
    artifacts_dir: str,
    feature_bounds: Tuple[int, int] # <--- NEW ARGUMENT (Min, Max)
) -> Tuple[pd.DataFrame, StandardScaler, List[str]]:
    """
    Loads data, merges sheets, cleans, and performs Min-Max scaling on Inputs
    and Standard Scaling on Targets.
    """
    CATEGORY_FEATURE_NAME = 'Category_C'
    min_val, max_val = feature_bounds
    
    print(f"--- Data Preparation Start ---")
    print(f"Feature Scaling Range: [{min_val}, {max_val}] -> [0, 1]")
    
    # 1. Load and Merge Data
    try:
        xls = pd.ExcelFile(data_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Main dataset not found at '{data_file_path}'.")
        
    df_b = pd.read_excel(xls, 'Elips0')
    df_c = pd.read_excel(xls, 'Elips1')
    
    # 2. Create Categorical Feature
    df_b[CATEGORY_FEATURE_NAME] = 0
    df_c[CATEGORY_FEATURE_NAME] = 1
    
    df = pd.concat([df_b, df_c], ignore_index=True)
    
    # Final list of features (X columns)
    present_base_features = [col for col in input_features if col in df.columns]
    final_input_features = present_base_features + [CATEGORY_FEATURE_NAME]
    
    # 3. Data Cleaning
    df_to_process = df.dropna(subset=[target_column]).copy()
    
    # --- NEW: 4. Input Feature Scaling (Min-Max) ---
    # Formula: (x - min) / (max - min)
    # We only scale the geometric features, NOT the category
    print("Applying Min-Max scaling to input features...")
    
    # Check for out-of-bounds data issues
    if df_to_process[present_base_features].min().min() < min_val:
        print(f"WARNING: Data contains values smaller than defined Min ({min_val})!")
    if df_to_process[present_base_features].max().max() > max_val:
        print(f"WARNING: Data contains values larger than defined Max ({max_val})!")

    df_to_process[present_base_features] = (df_to_process[present_base_features] - min_val) / (max_val - min_val)

    # 5. Target Scaling (StandardScaler)
    print(f"Scaling target column ('{target_column}')...")
    target_scaler = StandardScaler()
    df_to_process[target_column] = target_scaler.fit_transform(df_to_process[[target_column]])

    # Create directories
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Save Artifacts
    joblib.dump(target_scaler, os.path.join(artifacts_dir, TARGET_SCALER_FILENAME))

    # 6. Saving Processed Data
    cols_to_save = final_input_features + [target_column]
    df_to_process[cols_to_save].to_csv(os.path.join(processed_dir, PROCESSED_DATA_FILENAME), index=False)
    print(f"Processed data saved. Inputs scaled to [0,1]. Target standardized.")
    
    return df_to_process, target_scaler, final_input_features