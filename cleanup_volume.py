import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any

# Assuming ConfigManager is available in the current directory
from config_manager import ConfigManager 

# --- Configuration Constant ---
CONFIG_FILE = "config.yaml"

# --- Calculation Constants ---
PI_CONSTANT = np.pi 

def calculate_volume(row: pd.Series, base_features: List[str], thickness: float) -> float:
    """
    Calculates the total volume based on the sum of the elliptical areas of the 6 plies.
    Area of ply_i = PI * a_ply_i * b_ply_i.
    Volume = Sum(Area_i) * Thickness.
    
    Args:
        row: A row from the DataFrame.
        base_features: List of 12 ply column names.
        thickness: The constant thickness value.
    """
    
    total_volume = 0.0
    
    # Iterate through the 6 ply pairs (a1/b1, a2/b2, ..., a6/b6)
    for i in range(1, 7):
        a_col = f'a_ply{i}'
        b_col = f'b_ply{i}'
        
        # Check that the columns are present and the values are not NaN before calculating
        if a_col in base_features and b_col in base_features:
            a_val = row.get(a_col)
            b_val = row.get(b_col)
            
            # Skip if input data is missing for this row
            if pd.isna(a_val) or pd.isna(b_val):
                continue
            
            # Area = PI * a * b
            area_i = PI_CONSTANT * a_val * b_val
            volume_i = area_i * thickness
            
            total_volume += volume_i
            
    return total_volume

def cleanup_volume_entries(config_manager: ConfigManager):
    """
    Loads the dataset (PATHS.DATA_FILE), recalculates and updates the 'Volume' column
    for all entries in the 'Elips0' and 'Elips1' sheets.
    """
    
    # Get parameters from configuration
    data_file = config_manager.get('PATHS.DATA_FILE')
    base_features = config_manager.get('COLUMNS.BASE_INPUT_FEATURES')
    volume_col = config_manager.get('COLUMNS.VOLUME')
    thickness = config_manager.get('OPTIMIZATION.THICKNESS_CONSTANT')
    
    EXCEL_SHEETS = ['Elips0', 'Elips1']
    
    print(f"\n--- Starting Volume Recalculation and Cleanup for {data_file} ---")

    # --- MODIFICATION: STEP 1: READ ALL DATA FIRST ---
    # Load all existing sheets into memory before opening any writer
    all_sheets_data = {}
    try:
        xls = pd.ExcelFile(data_file)
        for sheet_name in xls.sheet_names:
            all_sheets_data[sheet_name] = pd.read_excel(xls, sheet_name)
        xls.close() # Explicitly close the read file
        print(f"Successfully loaded {len(all_sheets_data)} sheets from {data_file}.")
        
    except FileNotFoundError:
        print(f"FATAL ERROR: Data file not found at {data_file}.")
        return
    except Exception as e:
        print(f"FATAL ERROR: Could not read Excel file. Error: {e}")
        return

    # --- MODIFICATION: STEP 2: OPEN WRITER *AFTER* READING IS COMPLETE ---
    try:
        writer = pd.ExcelWriter(data_file, engine='xlsxwriter')
    except Exception as e:
        print(f"FATAL ERROR: Could not open Excel writer. Is {data_file} open? Error: {e}")
        return

    sheets_processed = 0
    
    # --- MODIFICATION: STEP 3: Iterate over the in-memory data ---
    for sheet_name, df in all_sheets_data.items():
        
        # Check if this is one of the sheets we need to modify
        if sheet_name in EXCEL_SHEETS:
            print(f"Processing sheet: {sheet_name}")
            
            # Check if we have the necessary input columns
            if not all(feature in df.columns for feature in base_features):
                print(f"   Skipping: Missing one or more required ply features in {sheet_name}.")
                
            else:
                # Calculate the volume for ALL rows and assign it
                df[volume_col] = df.apply(
                    calculate_volume, 
                    axis=1, 
                    base_features=base_features, 
                    thickness=thickness
                )
                
                print(f"   Successfully calculated and updated {len(df)} entries.")
                sheets_processed += 1
        
        # --- MODIFICATION: STEP 4: Write ALL sheets (modified or not) back to the file ---
        # This preserves sheets you didn't intend to process
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Close the writer to save the file
    writer.close()
    
    print(f"\n--- Volume Cleanup Complete. Total sheets modified: {sheets_processed} ---")
    
def main():
    """Entry point for standalone execution."""
    try:
        # Load ConfigManager here for standalone run
        config_manager = ConfigManager(CONFIG_FILE)
        cleanup_volume_entries(config_manager)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()