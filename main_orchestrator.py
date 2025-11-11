import os
import shutil
import pandas as pd
from typing import Dict, Any

# Import the ConfigManager
from config_manager import ConfigManager 

# Import main functions from the simple scripts (assuming they are in the same dir)
from prepare_data import load_and_prepare_data
from train_surrogate import train_dkgp_dual_models
from find_new_candidate_points import find_dual_candidate_points

# --- Configuration Constant ---
CONFIG_FILE = "config.yaml"

def setup_directories(config_manager: ConfigManager):
    """Creates necessary directories and optionally cleans up previous runs."""
    processed_dir = config_manager.get('PATHS.PROCESSED_DIR')
    artifacts_dir = config_manager.get('PATHS.ARTIFACTS_DIR')
    cleanup = config_manager.get('PROJECT.CLEANUP_PREVIOUS_RUNS')
    
    if cleanup:
        print("Cleaning up previous run directories...")
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)
        if os.path.exists(artifacts_dir):
            shutil.rmtree(artifacts_dir)
            
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    print(f"Directories ready: {processed_dir}, {artifacts_dir}")

def main_orchestrator():
    """Executes the full three-step workflow: Prepare -> Train -> Find Candidates."""
    
    try:
        # Load the configuration immediately
        config_manager = ConfigManager(CONFIG_FILE)
        
        print("\n" + "#"*70)
        print(f"## STARTING {config_manager.get('PROJECT.NAME')} WORKFLOW ##")
        print("#"*70)
        
        setup_directories(config_manager)
        
        # --- STEP 1: PREPARE DATA ---
        print("\n[STEP 1/3] Executing Data Preparation...")
        _, _, final_features = load_and_prepare_data(
            data_file_path=config_manager.get('PATHS.DATA_FILE'),
            input_features=config_manager.get('COLUMNS.BASE_INPUT_FEATURES'),
            target_column=config_manager.get('COLUMNS.TARGET'),
            processed_dir=config_manager.get('PATHS.PROCESSED_DIR'),
            artifacts_dir=config_manager.get('PATHS.ARTIFACTS_DIR')
        )
        print(f"Data Prepared. Total feature count: {len(final_features)}")
        
        # --- STEP 2: TRAIN SURROGATE MODELS ---
        print("\n[STEP 2/3] Executing Dual Surrogate Model Training...")
        # Pass the config manager to train_dkgp_dual_models
        _ = train_dkgp_dual_models(config_manager)
        print("Dual Models Trained and Artifacts Saved.")
        
        # --- STEP 3: FIND NEW CANDIDATE POINTS ---
        print("\n[STEP 3/3] Executing Candidate Finding (Bayesian Optimization)...")
        # Pass the config manager to find_dual_candidate_points
        candidates_found: Dict[int, pd.DataFrame] = find_dual_candidate_points(config_manager)
        
        print("\n" + "="*70)
        print("## WORKFLOW SUCCESSFUL ##")
        
        # Report the final candidates
        base_features = config_manager.get('COLUMNS.BASE_INPUT_FEATURES')
        
        if 0 in candidates_found:
            print("\nCandidate 1 (Category 0) found:")
            print(candidates_found[0][base_features + [config_manager.get('COLUMNS.CATEGORY'), 'Predicted_Strength_Unscaled']].iloc[0])
        if 1 in candidates_found:
            print("\nCandidate 2 (Category 1) found:")
            print(candidates_found[1][base_features + [config_manager.get('COLUMNS.CATEGORY'), 'Predicted_Strength_Unscaled']].iloc[0])
        
        print("="*70)

    except Exception as e:
        print("\n" + "#"*70)
        print(f"## WORKFLOW FAILED: {type(e).__name__} ##")
        print(f"Error details: {e}")
        print("#"*70)

if __name__ == '__main__':
    main_orchestrator()