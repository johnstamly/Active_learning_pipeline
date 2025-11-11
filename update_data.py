import pandas as pd
import numpy as np
import os
import shutil
from typing import List, Dict, Any

# Assuming ConfigManager is available for import
from config_manager import ConfigManager 

# --- Configuration Constant ---
CONFIG_FILE = "config.yaml"
CATEGORIES = [0, 1]

# --- Column and Calculation Constants ---
PREDICTED_MEAN_COL = 'Predicted_Strength_Mean' 
VOLUME_COL = 'Volume' 
THICKNESS_CONSTANT = 2 
PI_CONSTANT = np.pi 

# --- Filenames ---
NEW_CANDIDATE_FILENAME = 'new_candidate_point_cat_{}.csv'
PREDICTED_STRENGTH_UNSCALED_SOURCE = 'Predicted_Strength_Unscaled' 

def calculate_volume(row: pd.Series, thickness: float) -> float:
    """Calculates the total volume based on the sum of the elliptical areas of the 6 plies."""
    
    total_volume = 0.0
    
    for i in range(1, 7):
        a_col = f'a_ply{i}'
        b_col = f'b_ply{i}'
        
        if a_col in row and b_col in row:
            a_val = row[a_col]
            b_val = row[b_col]
            
            # Area = PI * a * b
            area_i = PI_CONSTANT * a_val * b_val
            volume_i = area_i * thickness
            
            total_volume += volume_i
            
    return total_volume

def update_data_file(config_manager: ConfigManager):
    """
    Loads existing data, strictly preserves column order, appends new candidates,
    calculates volume, and maps predicted strength to the correct column.
    """
    
    # 1. Get configuration paths and names
    data_file = config_manager.get('PATHS.DATA_FILE')
    processed_dir = config_manager.get('PATHS.PROCESSED_DIR')
    target_column = config_manager.get('COLUMNS.TARGET')
    base_features = config_manager.get('COLUMNS.BASE_INPUT_FEATURES')
    
    # Define the *new* columns we need to ensure are present
    new_data_cols = [target_column, VOLUME_COL, PREDICTED_MEAN_COL]

    # The original Excel sheets
    EXCEL_SHEETS = {
        0: 'Elips0',
        1: 'Elips1'
    }
    
    print("\n--- Data Update Start (Enforcing Original Column Order) ---")

    # 2. Create Backup of the Old File
    backup_file = data_file.replace('.xlsx', '_BACKUP.xlsx')
    try:
        shutil.copyfile(data_file, backup_file)
        print(f"Original file backed up to: {backup_file}")
    except FileNotFoundError:
        print(f"Error: Original data file '{data_file}' not found. Cannot create backup.")
        return
    
    # 3. Pre-load ALL existing data to capture column order
    all_existing_data = {}
    
    try:
        xls = pd.ExcelFile(data_file)
        
        # --- PHASE 1: READ EXISTING DATA AND DETERMINE FINAL COLUMN ORDER ---
        for category, sheet_name in EXCEL_SHEETS.items():
            
            if sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name)
                
                # Identify columns present in the old sheet
                existing_cols = list(df.columns)
                
                # Identify new columns that MUST be added to the sheet
                missing_new_cols = [col for col in new_data_cols if col not in existing_cols]
                
                # The FINAL column order is: [Existing Columns] + [Missing New Columns]
                final_column_order = existing_cols + missing_new_cols

                # Reindex to ensure the DataFrame has all necessary columns, preserving order
                df = df.reindex(columns=final_column_order)
                
            else:
                # If sheet doesn't exist, create a standard order
                final_column_order = base_features + new_data_cols
                df = pd.DataFrame(columns=final_column_order)
            
            all_existing_data[sheet_name] = {'df': df, 'order': final_column_order}
            
    except Exception as e:
        print(f"FATAL Error during initial Excel read: {e}. Aborting update.")
        return
        
    # 4. Open Excel Writer
    output_writer = pd.ExcelWriter(data_file, engine='xlsxwriter')

    # 5. Process and Append New Candidates
    for category in CATEGORIES:
        sheet_name = EXCEL_SHEETS[category]
        candidate_file = NEW_CANDIDATE_FILENAME.format(category)
        candidate_path = os.path.join(processed_dir, candidate_file)

        print(f"Processing sheet '{sheet_name}' (Category {category})...")
        
        data_package = all_existing_data.get(sheet_name)
        df_existing = data_package['df']
        final_column_order = data_package['order'] # Use the determined order

        try:
            df_new = pd.read_csv(candidate_path)
            
            # 6. Prepare New Data for Appending
            
            # a) Calculate and add Volume
            df_new[VOLUME_COL] = df_new.apply(
                calculate_volume, 
                axis=1, 
                thickness=THICKNESS_CONSTANT
            )
            
            # b) Rename prediction column and select columns for append
            df_new.rename(
                columns={PREDICTED_STRENGTH_UNSCALED_SOURCE: PREDICTED_MEAN_COL},
                inplace=True
            )
            
            # c) Select only the columns needed for the new row and fill others with NaN
            # We select ALL columns in the FINAL ORDER, filling inputs from df_new and others with pd.NA
            new_rows_data = {}
            for col in final_column_order:
                if col in df_new.columns:
                    # Copy ply inputs, Volume, and Predicted Mean
                    new_rows_data[col] = df_new[col].values
                elif col == target_column:
                    # Target Strength is always NA for the new experiment
                    new_rows_data[col] = [pd.NA] * len(df_new)
                else:
                    # Copy any other existing column as NA (e.g., Specimen ID)
                    new_rows_data[col] = [pd.NA] * len(df_new)

            df_new_append = pd.DataFrame(new_rows_data, columns=final_column_order)
            
            # d) Concatenate (This correctly appends to the existing data)
            df_updated = pd.concat([df_existing, df_new_append], ignore_index=True)
            
            # e) Save to the new Excel file
            df_updated.to_excel(output_writer, sheet_name=sheet_name, index=False)
            
            print(f"  {len(df_new_append)} new candidate row(s) appended to sheet '{sheet_name}'.")

        except FileNotFoundError:
            print(f"  Warning: Candidate file not found for Category {category}. Saving existing data only.")
            # If candidate not found, save the existing data to prevent loss
            df_existing.to_excel(output_writer, sheet_name=sheet_name, index=False)
        except Exception as e:
            print(f"Error processing Category {category}: {e}")
            
    # 7. Finalize the new Excel file
    try:
        output_writer.close()
        print(f"\nSuccessfully updated and saved new data file: {data_file}")
        print("--- Data Update Complete ---")
    except Exception as e:
        print(f"FATAL ERROR: Could not save the updated Excel file. Check if {data_file} is open. Error: {e}")

if __name__ == '__main__':
    try:
        # Load ConfigManager here for standalone run
        from config_manager import ConfigManager
        config_manager = ConfigManager(CONFIG_FILE)
        update_data_file(config_manager)
    except Exception as e:
        print(f"An error occurred during data update: {e}")