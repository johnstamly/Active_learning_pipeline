import torch
import gpytorch
import pandas as pd
import numpy as np
import os
import joblib
from scipy.stats import norm
from typing import Dict, Any
from model_definitions import DeepKernel, DeepKernelGP

# --- HELPER FUNCTIONS (Unchanged) ---
# ... (Keep generate_random_valid_point and calculate_elliptical_volume as they were) ...
# I will retain them in the full block below for copy-paste convenience

def generate_random_valid_point(config_manager: Any, category: int) -> np.ndarray:
    """Generates a single valid integer point with category-specific constraints."""
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
            max_b_current = a_plies[i]
            min_b = A_PLY_MIN
            if i > 0:
                min_b = max(min_b, b_plies[-1] + 1)
                if category == 1:
                    min_b = max(min_b, a_plies[i-1])
            max_b = min(max_b_current, A_PLY_MAX)
            if min_b > max_b:
                possible = False
                break
            b_val = np.random.randint(min_b, max_b + 1)
            b_plies.append(b_val)
        
        if possible:
            return np.array(a_plies.tolist() + b_plies, dtype=np.float32)

def calculate_elliptical_volume(candidates_np: np.ndarray, thickness: float = 2.0) -> np.ndarray:
    a_plies = candidates_np[:, 0:6]
    b_plies = candidates_np[:, 6:12]
    ply_areas = np.pi * a_plies * b_plies
    total_volume = np.sum(ply_areas, axis=1) * thickness
    return total_volume

def expected_improvement(mu: torch.Tensor, sigma: torch.Tensor, max_y: float) -> torch.Tensor:
    sigma_safe = torch.where(sigma > 1e-6, sigma, torch.ones_like(sigma) * 1e-6)
    Z = (mu - max_y) / sigma_safe
    ei = (mu - max_y) * norm.cdf(Z.detach().numpy()) + sigma_safe * norm.pdf(Z.detach().numpy())
    ei = torch.where(sigma <= 1e-6, torch.max(torch.zeros_like(mu), mu - max_y), ei)
    return ei

# --- MAIN FUNCTION ---

def find_dual_candidate_points(config_manager: Any, mode: str = 'active_learning') -> Dict[int, pd.DataFrame]:
    """
    Finds candidates based on the selected mode.
    
    Args:
        mode: 
          - 'active_learning': Maximize Expected Improvement (Exploration)
          - 'pure_mean': Maximize Mean only (Risky Exploitation)
          - 'robust': Maximize Lower Confidence Bound (Mean - 1.96*Sigma) (Safe Exploitation)
    """
    print(f"\n--- Candidate Finding Start (Mode: {mode.upper()}) ---")

    artifacts_dir = config_manager.get('PATHS.ARTIFACTS_DIR')
    base_features = config_manager.get('COLUMNS.BASE_INPUT_FEATURES')
    latent_dim = config_manager.get('MODEL.LATENT_DIM')
    num_random_samples = config_manager.get('OPTIMIZATION.NUM_RANDOM_SAMPLES')
    
    A_PLY_MIN = config_manager.get('OPTIMIZATION.A_PLY_MIN')
    A_PLY_MAX = config_manager.get('OPTIMIZATION.A_PLY_MAX')
    THICKNESS_CONST = config_manager.get('OPTIMIZATION.THICKNESS_CONSTANT')
    
    try:
        vol_weight = config_manager.get('OPTIMIZATION.VOLUME_WEIGHT')
    except:
        vol_weight = 0.0
    
    print(f"Optimization Weights -> Strength: {1 - vol_weight:.2f} | Minimize Volume: {vol_weight:.2f}")

    min_vol_theoretical = 6 * np.pi * A_PLY_MIN * A_PLY_MIN * THICKNESS_CONST
    max_vol_theoretical = 6 * np.pi * A_PLY_MAX * A_PLY_MAX * THICKNESS_CONST
    
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
        feature_extractor = DeepKernel(input_dim=input_dim, latent_dim=latent_dim, mlp_depth=config_manager.get('MODEL.MLP_DEPTH')) 
        model = DeepKernelGP(train_x, train_y, likelihood, feature_extractor)
        model.load_state_dict(model_state)
        model.eval()
        likelihood.eval()
        
        f_max = train_y.max().item()

        # 2. Generate Candidates
        candidate_pool_list = [generate_random_valid_point(config_manager, category) for _ in range(num_random_samples)]
        candidate_pool_np = np.stack(candidate_pool_list)
        
        # 3. Calculate Normalized Volume (Minimize this)
        volumes = calculate_elliptical_volume(candidate_pool_np, thickness=THICKNESS_CONST)
        volumes_norm = (volumes - min_vol_theoretical) / (max_vol_theoretical - min_vol_theoretical)
        volumes_norm = np.clip(volumes_norm, 0, 1)

        # 4. Model Predictions
        candidate_pool_scaled_np = (candidate_pool_np - A_PLY_MIN) / (A_PLY_MAX - A_PLY_MIN)
        candidate_pool_tensor = torch.tensor(candidate_pool_scaled_np, dtype=torch.float32)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictive_dist = likelihood(model(candidate_pool_tensor))
            mu = predictive_dist.mean
            sigma = predictive_dist.stddev
            
        # 5. Score Calculation Branch
        if mode == 'active_learning':
            # Expected Improvement (Balance)
            raw_strength_score = expected_improvement(mu, sigma, f_max)
            print("   -> Strategy: Active Learning (Expected Improvement)")
            
        elif mode == 'pure_mean':
            # Pure Exploitation (Risky)
            raw_strength_score = mu
            print("   -> Strategy: Pure Mean (Max Strength, Ignoring Risk)")
            
        elif mode == 'robust':
            # Lower Confidence Bound (Safe)
            # 1.96 sigma covers 95% of the probability mass
            # "I am 95% confident the strength is at least this value"
            raw_strength_score = mu - (1.96 * sigma)
            print("   -> Strategy: Robust/Safe (Lower Confidence Bound: Mean - 1.96*Sigma)")

        # Normalize Strength Score (0 to 1)
        s_min = raw_strength_score.min()
        s_max = raw_strength_score.max()
        if s_max > s_min:
            strength_score_norm = (raw_strength_score - s_min) / (s_max - s_min)
        else:
            strength_score_norm = raw_strength_score

        # 6. Multi-Objective Scoring
        if isinstance(strength_score_norm, torch.Tensor):
            strength_score_norm_np = strength_score_norm.numpy()
        else:
            strength_score_norm_np = strength_score_norm

        final_score = ((1.0 - vol_weight) * strength_score_norm_np) - (vol_weight * volumes_norm)
        
        # 7. Select Best
        best_idx = np.argmax(final_score)
        
        # 8. Formatting Results
        best_candidate_physical = candidate_pool_np[best_idx]
        best_vol = volumes[best_idx]
        best_metric_raw = raw_strength_score[best_idx].item()
        
        predicted_y_scaled = mu[best_idx].item()
        predicted_sigma_scaled = sigma[best_idx].item()
        
        # Unscale Mean
        predicted_y_unscaled = target_scaler.inverse_transform(np.array([[predicted_y_scaled]]))[0, 0]
        # Unscale Sigma (approximate)
        predicted_sigma_unscaled = predicted_sigma_scaled * target_scaler.scale_[0]

        new_candidate_df = pd.DataFrame([best_candidate_physical], columns=base_features)
        new_candidate_df[config_manager.get('COLUMNS.CATEGORY')] = category
        new_candidate_df[f'Predicted_{config_manager.get("COLUMNS.TARGET")}_Unscaled'] = predicted_y_unscaled
        new_candidate_df['Predicted_Sigma_Unscaled'] = predicted_sigma_unscaled
        new_candidate_df['Predicted_Volume'] = best_vol
        
        new_candidate_df['Strategy_Score_Raw'] = best_metric_raw
        new_candidate_df['Multi_Obj_Score'] = final_score[best_idx]
        new_candidate_df['Selection_Mode'] = mode

        save_path = os.path.join(config_manager.get('PATHS.PROCESSED_DIR'), f'new_candidate_point_cat_{category}.csv')
        new_candidate_df.to_csv(save_path, index=False)

        print(f"  Selected Design Stats:")
        print(f"    -> Strength Prediction: {predicted_y_unscaled:.2f} ± {predicted_sigma_unscaled:.2f} MPa")
        print(f"    -> Volume: {best_vol:.0f} mm^3")
        print(f"    -> Multi-Objective Score: {final_score[best_idx]:.4f}")

        all_candidates[category] = new_candidate_df

    return all_candidates