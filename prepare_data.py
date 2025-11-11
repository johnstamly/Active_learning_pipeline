import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Tuple, List, Dict, Any

# Filenames used for saving (now treated as constants *within* the function)
PROCESSED_DATA_FILENAME = 'processed_data.csv'
TARGET_SCALER_FILENAME = 'target_scaler.pkl'

def load_and_prepare_data(
    data_file_path: str,
    input_features: List[str],
    target_column: str,
    processed_dir: str,
    artifacts_dir: str
) -> Tuple[pd.DataFrame, StandardScaler, List[str]]:
    """
    Loads data, merges sheets, creates a numeric category, cleans, and scales
    the target feature. Saves the processed data and the fitted scaler.
    (Parameters now come from config via the orchestrator).
    """
    CATEGORY_FEATURE_NAME = 'Category_C'
    
    # ... (Rest of the function body remains the same, ensuring it uses the passed arguments)
    print(f"--- Data Preparation Start ---")
    
    # 1. Load and Merge Data
    try:
        xls = pd.ExcelFile(data_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Main dataset not found at '{data_file_path}'. Please check the path or create the file.")
    except Exception as e:
        raise Exception(f"Error reading Excel file: {e}")
        
    df_b = pd.read_excel(xls, 'Elips0')
    df_c = pd.read_excel(xls, 'Elips1')
    
    # 2. Create Categorical Feature
    df_b[CATEGORY_FEATURE_NAME] = 0
    df_c[CATEGORY_FEATURE_NAME] = 1
    
    df = pd.concat([df_b, df_c], ignore_index=True)
    print(f"Total rows loaded: {len(df)}")
    
    # Final list of features (X columns)
    present_base_features = [col for col in input_features if col in df.columns]
    
    if len(present_base_features) != len(input_features):
        missing = set(input_features) - set(df.columns)
        print(f"WARNING: Missing expected base features: {missing}. These will be skipped.")
        
    final_input_features = present_base_features + [CATEGORY_FEATURE_NAME]
    
    if len(final_input_features) <= 1:
        raise ValueError(f"No valid input features found in the dataset. Cannot proceed.")
    
    # 3. Data Cleaning
    df_to_process = df.dropna(subset=[target_column]).copy()
    print(f"Rows with valid '{target_column}' for training: {len(df_to_process)}")
    
    if len(df_to_process) == 0:
        raise ValueError(f"No valid data found for target column '{target_column}'. Cannot proceed.")

    # 4. Target Scaling
    print(f"Scaling target column ('{target_column}')...")
    
    target_scaler = StandardScaler()
    df_to_process[target_column] = target_scaler.fit_transform(df_to_process[[target_column]])

    # Create directories
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Save the target scaler (Artifact)
    target_scaler_path = os.path.join(artifacts_dir, TARGET_SCALER_FILENAME)
    joblib.dump(target_scaler, target_scaler_path)
    print(f"Target scaler saved to {target_scaler_path}")

    # 5. Saving Data
    cols_to_save = final_input_features + [target_column]
    df_to_process[cols_to_save].to_csv(os.path.join(processed_dir, PROCESSED_DATA_FILENAME), index=False)
    print(f"Processed data (X + Y) saved to {os.path.join(processed_dir, PROCESSED_DATA_FILENAME)}")
    
    print(f"--- Data Preparation Complete ---")

    return df_to_process, target_scaler, final_input_features

# The main() function (for standalone testing) is now removed, as the orchestrator is the main entry point.
# You can restore it if you need standalone testing, but it complicates the configuration flow slightly.