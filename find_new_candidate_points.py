import torch
import gpytorch
import pandas as pd
import numpy as np
import os
import joblib
from scipy.stats import norm
from typing import Dict, Any, List, Tuple

# Filenames used for loading/saving (now treated as constants *within* the function)
TARGET_SCALER_FILENAME = 'target_scaler.pkl'
DKGP_MODEL_FILENAME = 'dkgp_model_cat_{}.pth'
TRAIN_DATA_FILENAME = 'train_data_cat_{}.pt'
NEW_CANDIDATE_FILENAME = 'new_candidate_point_cat_{}.csv'
PROCESSED_DATA_DIR_NAME = 'processed' # Subdirectory of PARENT(artifacts_dir)

# --- Model Definitions (Re-defined for loading, taking dynamic dims) ---
class DeepKernel(torch.nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, latent_dim)
    def forward(self, x):
        return self.linear1(x)

class DeepKernelGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.feature_extractor = feature_extractor
    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
# Constrain Function
def generate_random_valid_point(config_manager: Any) -> np.ndarray:
    """Generates a single, random, constraint-respecting integer point."""
    
    A_PLY_MIN = config_manager.get('OPTIMIZATION.A_PLY_MIN') # 5
    A_PLY_MAX = config_manager.get('OPTIMIZATION.A_PLY_MAX') # 35
    
    full_range = range(A_PLY_MIN, A_PLY_MAX + 1)
    
    def generate_valid_sequence():
        """Generates 6 monotonically increasing integer plies within [5, 35]."""
        while True:
            # Generate 6 distinct numbers in the range [5, 35], sorted for monotonicity
            plies = np.random.choice(full_range, size=6, replace=False)
            plies.sort()
            return plies
            
    while True:
        # 1. Generate a_plies: [a1, a2, a3, a4, a5, a6]
        a_plies = generate_valid_sequence()
        
        # 2. Generate b_plies: [b1, b2, b3, b4, b5, b6]
        b_plies = []
        possible = True
        
        for i in range(6):
            
            # --- INTRA-PLY MAX CONSTRAINT: b_ply_i <= a_ply_i ---
            # Max b_ply_i can be is the value of a_ply_i
            max_b_current = a_plies[i]
            
            # --- CROSS-PLY MIN CONSTRAINT: b_ply_i+1 > a_ply_i ---
            # For the *first* ply (i=0): min_b is only constrained by the next ply's requirement (handled below).
            # For subsequent plies (i > 0): The minimum is determined by the previous b_ply.
            min_b = b_plies[-1] + 1 if i > 0 else A_PLY_MIN
            
            # --- CRITICAL CROSS-PLY CONSTRAINT: b_ply_i > a_ply_i-1 ---
            # Ensure the current minimum is at least 1 greater than the *previous* a_ply.
            # This is simplified: b_ply_i must be > a_ply_i-1.
            if i > 0:
                min_b = max(min_b, a_plies[i-1] + 1) 
            
            # Global upper bound (35) must still be respected
            max_b = min(max_b_current, A_PLY_MAX)
            
            # Check if there is a single available integer in the range [min_b, max_b]
            if min_b > max_b:
                # If this sequence of 'a' plies makes generating 'b' plies impossible, restart.
                possible = False
                break
            
            # Sample b_val, ensuring integer type
            b_val = np.random.randint(min_b, max_b + 1)
            b_plies.append(b_val)
        
        if possible:
            break
            
    # Combine and return (12 features)
    return np.array(a_plies.tolist() + b_plies, dtype=np.float32)

# --- Expected Improvement (EI) Function (Same as before) ---
def expected_improvement(mu: torch.Tensor, sigma: torch.Tensor, max_y: float) -> torch.Tensor:
    sigma_safe = torch.where(sigma > 1e-6, sigma, torch.ones_like(sigma) * 1e-6)
    Z = (mu - max_y) / sigma_safe
    
    ei = (mu - max_y) * norm.cdf(Z.detach().numpy()) + sigma_safe * norm.pdf(Z.detach().numpy())
    
    ei = torch.where(sigma <= 1e-6, torch.max(torch.zeros_like(mu), mu - max_y), ei)
    
    return ei

# --- Candidate Finding Main Function ---
def find_dual_candidate_points(
    config_manager: Any # Using Any to avoid circular dependency on ConfigManager class
) -> Dict[int, pd.DataFrame]:
    """
    Loads models, finds the point maximizing Expected Improvement (EI) for each category,
    and saves the two new candidates.
    """
    print("\n--- Candidate Finding Start (Dual EI Optimization) ---")

    # Extract settings from config
    artifacts_dir = config_manager.get('PATHS.ARTIFACTS_DIR')
    base_features = config_manager.get('COLUMNS.BASE_INPUT_FEATURES')
    target_column = config_manager.get('COLUMNS.TARGET')
    latent_dim = config_manager.get('MODEL.LATENT_DIM')
    num_random_samples = config_manager.get('OPTIMIZATION.NUM_RANDOM_SAMPLES')
    
    categories = [0, 1]
    input_dim = len(base_features)

    # Load Scaler (common to both models)
    target_scaler = joblib.load(os.path.join(artifacts_dir, TARGET_SCALER_FILENAME))

    all_candidates = {}

    for category in categories:
        print(f"\n--- Optimizing for Category {category} ---")
        
        # 1. Load Model and Training Data
        try:
            model_state = torch.load(os.path.join(artifacts_dir, DKGP_MODEL_FILENAME.format(category)))
            train_data = torch.load(os.path.join(artifacts_dir, TRAIN_DATA_FILENAME.format(category)))
        except FileNotFoundError:
            print(f"Skipping Category {category}: Model or data not found.")
            continue
            
        train_x = train_data['train_x']
        train_y = train_data['train_y']
        
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        feature_extractor = DeepKernel(input_dim=input_dim, latent_dim=latent_dim)
        model = DeepKernelGP(train_x, train_y, likelihood, feature_extractor)
        model.load_state_dict(model_state)
        
        # Set to evaluation mode
        model.eval()
        likelihood.eval()

        # Determine f_max
        f_max = train_y.max().item()
        print(f"Max observed scaled Strength: {f_max:.4f} (from {len(train_y)} points)")

        # 2. Generate Constraint-Respecting Candidate Pool
        candidate_pool_list = [generate_random_valid_point(config_manager) for _ in range(num_random_samples)]
        candidate_pool_np = np.stack(candidate_pool_list)
        candidate_pool = torch.tensor(candidate_pool_np, dtype=torch.float32)
        
        # 3. Evaluate EI
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictive_dist = likelihood(model(candidate_pool))
            mu = predictive_dist.mean
            sigma = predictive_dist.stddev
            
            ei_values = expected_improvement(mu, sigma, f_max)
            
        # 4. Find the Best Candidate
        best_ei_index = torch.argmax(ei_values)
        best_candidate_x = candidate_pool[best_ei_index]
        best_ei_value = ei_values[best_ei_index].item()
        
        # 5. Format and Save Output
        new_candidate_df = pd.DataFrame([best_candidate_x.numpy()], columns=base_features)
        
        # Add the fixed Category_C column
        new_candidate_df[config_manager.get('COLUMNS.CATEGORY')] = category
        
        # Add prediction metrics
        predicted_y_scaled = mu[best_ei_index].item()
        predicted_y_unscaled = target_scaler.inverse_transform(np.array([[predicted_y_scaled]]))[0, 0]
        
        new_candidate_df[f'Predicted_{target_column}_Unscaled'] = predicted_y_unscaled
        new_candidate_df['Expected_Improvement'] = best_ei_value

        # 6. Save the New Candidate
        processed_dir_path = config_manager.get('PATHS.PROCESSED_DIR')
        save_path = os.path.join(processed_dir_path, NEW_CANDIDATE_FILENAME.format(category))

        # --- FIX: Ensure the directory exists ---
        os.makedirs(processed_dir_path, exist_ok=True)

        new_candidate_df.to_csv(save_path, index=False)

        print(f"  Best EI: {best_ei_value:.4f}")
        
        print(f"  Best EI: {best_ei_value:.4f}")
        print(f"  Predicted Unscaled Strength: {predicted_y_unscaled:.2f}")
        print(f"  New candidate saved to {save_path}")
        
        all_candidates[category] = new_candidate_df

    print("\n--- Candidate Finding Complete ---")
    return all_candidates