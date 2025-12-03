import torch
import gpytorch
import pandas as pd
import numpy as np
import os
import joblib
from scipy.stats import norm
from typing import Dict, Any
from model_definitions import DeepKernel, DeepKernelGP

# --- HELPER FUNCTIONS ---

def generate_random_valid_point(config_manager: Any, category: int) -> np.ndarray:
    """
    Generates a single valid integer point with category-specific constraints.
    
    Constraints:
    1. Bounds: A_PLY_MIN <= val <= A_PLY_MAX
    2. Nesting (a): a_i < a_{i+1} (Strictly increasing)
    3. Nesting (b): b_i < b_{i+1} (Strictly increasing)
    4. Aspect Ratio: b_i <= a_i
    5. Support (Cat 1 Only): a_i <= b_{i+1} (Prevents overhangs in mixed orientation)
    """
    A_PLY_MIN = config_manager.get('OPTIMIZATION.A_PLY_MIN')
    A_PLY_MAX = config_manager.get('OPTIMIZATION.A_PLY_MAX')
    full_range = range(A_PLY_MIN, A_PLY_MAX + 1)

    def generate_valid_sequence():
        while True:
            plies = np.random.choice(full_range, size=6, replace=False)
            plies.sort()
            return plies

    while True:
        a_plies = generate_valid_sequence()
        b_plies = []
        possible = True
        
        for i in range(6):
            max_b_current = a_plies[i] # Constraint 4: b_i <= a_i
            
            # Base Lower Bound
            min_b = A_PLY_MIN
            
            if i > 0:
                # Constraint 3: b_i > b_{i-1} -> b_i >= b_{i-1} + 1
                min_b_nesting = b_plies[-1] + 1
                min_b = max(min_b, min_b_nesting)
                
                # Constraint 5 (Category 1 Only): a_{i-1} <= b_i
                if category == 1:
                    min_b_support = a_plies[i-1]
                    min_b = max(min_b, min_b_support)
            
            # Apply Upper Bound
            max_b = min(max_b_current, A_PLY_MAX)
            
            # Check Feasibility
            if min_b > max_b:
                possible = False
                break
            
            # Sample Valid Integer
            b_val = np.random.randint(min_b, max_b + 1)
            b_plies.append(b_val)
        
        if possible:
            return np.array(a_plies.tolist() + b_plies, dtype=np.float32)

def calculate_elliptical_volume(candidates_np: np.ndarray, thickness: float = 2.0) -> np.ndarray:
    """
    Calculates volume for a batch of candidates.
    Formula: Sum( pi * a * b * t )
    Input shape: (N, 12) -> First 6 are 'a', Last 6 are 'b'
    """
    # Split into a and b (first 6 columns are a, last 6 are b)
    a_plies = candidates_np[:, 0:6]
    b_plies = candidates_np[:, 6:12]
    
    # Calculate area of each ply: pi * a * b
    ply_areas = np.pi * a_plies * b_plies
    
    # Total volume = Sum of areas * thickness
    total_volume = np.sum(ply_areas, axis=1) * thickness
    return total_volume

def expected_improvement(mu: torch.Tensor, sigma: torch.Tensor, max_y: float) -> torch.Tensor:
    sigma_safe = torch.where(sigma > 1e-6, sigma, torch.ones_like(sigma) * 1e-6)
    Z = (mu - max_y) / sigma_safe
    ei = (mu - max_y) * norm.cdf(Z.detach().numpy()) + sigma_safe * norm.pdf(Z.detach().numpy())
    ei = torch.where(sigma <= 1e-6, torch.max(torch.zeros_like(mu), mu - max_y), ei)
    return ei

# --- MAIN FUNCTION ---

def find_dual_candidate_points(config_manager: Any) -> Dict[int, pd.DataFrame]:
    print("\n--- Candidate Finding Start (Multi-Objective: Strength & Volume) ---")

    artifacts_dir = config_manager.get('PATHS.ARTIFACTS_DIR')
    base_features = config_manager.get('COLUMNS.BASE_INPUT_FEATURES')
    latent_dim = config_manager.get('MODEL.LATENT_DIM')
    num_random_samples = config_manager.get('OPTIMIZATION.NUM_RANDOM_SAMPLES')
    
    # Optim bounds
    A_PLY_MIN = config_manager.get('OPTIMIZATION.A_PLY_MIN')
    A_PLY_MAX = config_manager.get('OPTIMIZATION.A_PLY_MAX')
    THICKNESS_CONST = config_manager.get('OPTIMIZATION.THICKNESS_CONSTANT')
    
    # Weighting Factor
    try:
        vol_weight = config_manager.get('OPTIMIZATION.VOLUME_WEIGHT')
    except:
        vol_weight = 0.0
    
    print(f"Optimization Weights -> Strength: {1 - vol_weight:.2f} | Minimize Volume: {vol_weight:.2f}")

    # Pre-calculate theoretical Min/Max Volume for normalization
    min_vol_theoretical = 6 * np.pi * A_PLY_MIN * A_PLY_MIN * THICKNESS_CONST
    max_vol_theoretical = 6 * np.pi * A_PLY_MAX * A_PLY_MAX * THICKNESS_CONST
    
    print(f"Volume Range for Normalization: {min_vol_theoretical:.0f} - {max_vol_theoretical:.0f} mm^3")

    categories = [0, 1]
    input_dim = len(base_features)
    target_scaler = joblib.load(os.path.join(artifacts_dir, 'target_scaler.pkl'))
    all_candidates = {}

    for category in categories:
        print(f"\n--- Optimizing for Category {category} ---")
        
        # 1. Load Model
        try:
            model_path = os.path.join(artifacts_dir, f'dkgp_model_cat_{category}.pth')
            data_path = os.path.join(artifacts_dir, f'train_data_cat_{category}.pt')
            model_state = torch.load(model_path)
            train_data = torch.load(data_path)
        except FileNotFoundError:
            print(f"Skipping Category {category}: Artifacts not found.")
            continue
            
        train_x = train_data['train_x']
        train_y = train_data['train_y']
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # Ensure mlp_depth matches your config if you added that parameter to DeepKernel, otherwise remove it
        feature_extractor = DeepKernel(input_dim=input_dim, latent_dim=latent_dim, mlp_depth=config_manager.get('MODEL.MLP_DEPTH')) 
        model = DeepKernelGP(train_x, train_y, likelihood, feature_extractor)
        model.load_state_dict(model_state)
        model.eval()
        likelihood.eval()
        
        f_max = train_y.max().item()

        # 2. Generate Candidates (Physical Integers)
        # UPDATED: We pass 'category' here to enforce specific constraints
        candidate_pool_list = [generate_random_valid_point(config_manager, category) for _ in range(num_random_samples)]
        candidate_pool_np = np.stack(candidate_pool_list)
        
        # 3. Calculate Normalized Volume Score (0 to 1)
        volumes = calculate_elliptical_volume(candidate_pool_np, thickness=THICKNESS_CONST)
        volumes_norm = (volumes - min_vol_theoretical) / (max_vol_theoretical - min_vol_theoretical)
        volumes_norm = np.clip(volumes_norm, 0, 1)

        # 4. Calculate Strength EI
        candidate_pool_scaled_np = (candidate_pool_np - A_PLY_MIN) / (A_PLY_MAX - A_PLY_MIN)
        candidate_pool_tensor = torch.tensor(candidate_pool_scaled_np, dtype=torch.float32)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictive_dist = likelihood(model(candidate_pool_tensor))
            mu = predictive_dist.mean
            sigma = predictive_dist.stddev
            ei_strength = expected_improvement(mu, sigma, f_max)
            
        # Normalize EI
        ei_min = ei_strength.min()
        ei_max = ei_strength.max()
        if ei_max > ei_min:
            ei_norm = (ei_strength - ei_min) / (ei_max - ei_min)
        else:
            ei_norm = ei_strength

        # 5. Multi-Objective Scoring
        ei_norm_np = ei_norm.numpy()
        final_score = ((1.0 - vol_weight) * ei_norm_np) - (vol_weight * volumes_norm)
        
        # 6. Select Best
        best_idx = np.argmax(final_score)
        
        # 7. Formatting
        best_candidate_physical = candidate_pool_np[best_idx]
        best_vol = volumes[best_idx]
        best_ei_raw = ei_strength[best_idx].item()
        
        predicted_y_scaled = mu[best_idx].item()
        predicted_y_unscaled = target_scaler.inverse_transform(np.array([[predicted_y_scaled]]))[0, 0]

        new_candidate_df = pd.DataFrame([best_candidate_physical], columns=base_features)
        new_candidate_df[config_manager.get('COLUMNS.CATEGORY')] = category
        new_candidate_df[f'Predicted_{config_manager.get("COLUMNS.TARGET")}_Unscaled'] = predicted_y_unscaled
        new_candidate_df['Predicted_Volume'] = best_vol
        new_candidate_df['EI_Strength_Raw'] = best_ei_raw
        new_candidate_df['Multi_Obj_Score'] = final_score[best_idx]

        save_path = os.path.join(config_manager.get('PATHS.PROCESSED_DIR'), f'new_candidate_point_cat_{category}.csv')
        new_candidate_df.to_csv(save_path, index=False)

        print(f"  Selected Design Stats:")
        print(f"    -> Strength Prediction: {predicted_y_unscaled:.2f}")
        print(f"    -> Volume: {best_vol:.0f} mm^3")
        print(f"    -> EI (Strength): {best_ei_raw:.4f}")
        print(f"    -> Multi-Objective Score: {final_score[best_idx]:.4f}")

        all_candidates[category] = new_candidate_df

    return all_candidates