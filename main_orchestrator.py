import os
import shutil
import pandas as pd
import argparse
from typing import Dict, Any

# Import the ConfigManager
from config_manager import ConfigManager 

# Import main functions from the simple scripts
from prepare_data import load_and_prepare_data
from train_surrogate import train_dkgp_dual_models
from find_new_candidate_points import find_dual_candidate_points
from visualize_patches import generate_candidate_visualizations

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
    
    # --- 0. PARSE ARGUMENTS ---
    parser = argparse.ArgumentParser(description="Deep Kernel Active Learning Orchestrator")
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['active_learning', 'pure_mean', 'robust'], 
        default='active_learning',
        help="Mode: 'active_learning' (EI), 'pure_mean' (Exploitation), 'robust' (Safe/LCB)"
    )
    args = parser.parse_args()

    try:
        # Load the configuration immediately
        config_manager = ConfigManager(CONFIG_FILE)
        
        print("\n" + "#"*70)
        print(f"## STARTING {config_manager.get('PROJECT.NAME')} WORKFLOW ##")
        print(f"## MODE: {args.mode.upper()} ##")
        print("#"*70)
        
        setup_directories(config_manager)
        
        # --- STEP 1: PREPARE DATA ---
        print("\n[STEP 1/4] Executing Data Preparation...")
        _, _, final_features = load_and_prepare_data(
            data_file_path=config_manager.get('PATHS.DATA_FILE'),
            input_features=config_manager.get('COLUMNS.BASE_INPUT_FEATURES'),
            target_column=config_manager.get('COLUMNS.TARGET'),
            processed_dir=config_manager.get('PATHS.PROCESSED_DIR'),
            artifacts_dir=config_manager.get('PATHS.ARTIFACTS_DIR'),
            feature_bounds=(config_manager.get('OPTIMIZATION.A_PLY_MIN'), config_manager.get('OPTIMIZATION.A_PLY_MAX'))
        )
        print(f"Data Prepared. Total feature count: {len(final_features)}")
        
        # --- STEP 2: TRAIN SURROGATE MODELS ---
        print("\n[STEP 2/4] Executing Dual Surrogate Model Training...")
        _ = train_dkgp_dual_models(config_manager)
        print("Dual Models Trained and Artifacts Saved.")
        
        # --- STEP 3: FIND NEW CANDIDATE POINTS ---
        print(f"\n[STEP 3/4] Executing Candidate Finding (Mode: {args.mode})...")
        candidates_found: Dict[int, pd.DataFrame] = find_dual_candidate_points(
            config_manager, 
            mode=args.mode
        )

        # --- STEP 4: VISUALIZE CANDIDATES ---
        print("\n[STEP 4/4] Generating 3D Visualizations...")
        generate_candidate_visualizations(config_manager)
        print("Visualizations generated.")
        
        print("\n" + "="*70)
        print("## WORKFLOW SUCCESSFUL ##")
        
        # Report the final candidates
        base_features = config_manager.get('COLUMNS.BASE_INPUT_FEATURES')
        target_name = config_manager.get('COLUMNS.TARGET')
        
        # Define the list of potential columns we want to see
        potential_cols = base_features + [
            f'Predicted_{target_name}_Unscaled', 
            'Predicted_Volume',
            'Multi_Obj_Score',
            'Strategy_Score_Raw',          # Generic score column
            'Predicted_Sigma_Unscaled',    # Specific to robust mode
            'Selection_Mode'
        ]

        if 0 in candidates_found:
            print("\nCandidate 1 (Category 0) found:")
            # Only select columns that actually exist in the dataframe
            valid_cols = [c for c in potential_cols if c in candidates_found[0].columns]
            print(candidates_found[0][valid_cols].iloc[0])
            
        if 1 in candidates_found:
            print("\nCandidate 2 (Category 1) found:")
            valid_cols = [c for c in potential_cols if c in candidates_found[1].columns]
            print(candidates_found[1][valid_cols].iloc[0])
        
        print("="*70)

    except Exception as e:
        print("\n" + "#"*70)
        print(f"## WORKFLOW FAILED: {type(e).__name__} ##")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()
        print("#"*70)

if __name__ == '__main__':
    main_orchestrator()