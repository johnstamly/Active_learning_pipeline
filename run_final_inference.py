import argparse
import pandas as pd
from config_manager import ConfigManager
from find_new_candidate_points import find_dual_candidate_points
from visualize_patches import generate_candidate_visualizations

def main():
    # 1. Parse Mode 
    parser = argparse.ArgumentParser(description="Run Final Inference Steps Only")
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['active_learning', 'pure_mean', 'robust'], 
        default='robust',
        help="Mode: 'robust' (Safe/LCB), 'pure_mean' (Max Strength), 'active_learning' (EI)"
    )
    args = parser.parse_args()

    # 2. Load Configuration
    config_manager = ConfigManager("config.yaml")
    
    print("\n" + "#"*60)
    print(f"## SKIPPING TRAINING -> USING EXISTING ARTIFACTS ##")
    print(f"## MODE: {args.mode.upper()} ##")
    print("#"*60)

    # 3. Execute Candidate Finding
    print(f"\n[STEP 1/2] Finding Candidates ({args.mode})...")
    candidates = find_dual_candidate_points(config_manager, mode=args.mode)

    # 4. Execute Visualization
    print(f"\n[STEP 2/2] Generating Visualizations...")
    generate_candidate_visualizations(config_manager)

    # 5. Report Results (Robust Printing)
    print("\n" + "="*60)
    print("## INFERENCE SUCCESSFUL ##")
    
    base_features = config_manager.get('COLUMNS.BASE_INPUT_FEATURES')
    target_name = config_manager.get('COLUMNS.TARGET')
    
    # Always show these core columns
    cols_to_show = base_features + [
        f'Predicted_{target_name}_Unscaled', 
        'Predicted_Volume', 
        'Multi_Obj_Score',
        'Strategy_Score_Raw'  # This is the generic name we used in the update
    ]
    
    # Add Sigma if it exists (only in robust mode)
    if 'Predicted_Sigma_Unscaled' in candidates[0].columns:
        cols_to_show.append('Predicted_Sigma_Unscaled')
    
    if 0 in candidates:
        print("\nCandidate (Category 0):")
        # Filter columns to only those that actually exist in the dataframe
        valid_cols = [c for c in cols_to_show if c in candidates[0].columns]
        print(candidates[0][valid_cols].iloc[0])
    
    if 1 in candidates:
        print("\nCandidate (Category 1):")
        valid_cols = [c for c in cols_to_show if c in candidates[1].columns]
        print(candidates[1][valid_cols].iloc[0])
        
    print("="*60)

if __name__ == "__main__":
    main()