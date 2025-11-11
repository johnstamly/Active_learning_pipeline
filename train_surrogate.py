import torch
import gpytorch
import pandas as pd
import os
import joblib
from typing import Tuple, List, Dict, Any

# Filenames used for saving (now treated as constants *within* the function)
PROCESSED_DATA_FILENAME = 'processed_data.csv'
DKGP_MODEL_FILENAME = 'dkgp_model_cat_{}.pth'
TRAIN_DATA_FILENAME = 'train_data_cat_{}.pt'

# We define the model class outside the function as it's static
class DeepKernel(torch.nn.Module):
    # Constructor now takes dimensions dynamically
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, latent_dim)
    def forward(self, x):
        return self.linear1(x)

class DeepKernelGP(gpytorch.models.ExactGP):
    # ... (Model Definition remains the same)
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.feature_extractor = feature_extractor
        for param in self.feature_extractor.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# --- 2. Single Model Training Helper ---
def train_single_model(
    category: int, 
    df_cat: pd.DataFrame, 
    config_manager: Any # Using Any to avoid circular dependency on ConfigManager class
) -> Tuple[DeepKernelGP, gpytorch.likelihoods.GaussianLikelihood] | Tuple[None, None]:
    """Helper function to train and save one model for a specific category."""
    
    # Extract settings from config
    input_features = config_manager.get('COLUMNS.BASE_INPUT_FEATURES')
    target_column = config_manager.get('COLUMNS.TARGET')
    artifacts_dir = config_manager.get('PATHS.ARTIFACTS_DIR')
    
    input_dim = len(input_features)
    latent_dim = config_manager.get('MODEL.LATENT_DIM')
    num_epochs = config_manager.get('MODEL.NUM_EPOCHS')
    lr = config_manager.get('MODEL.LEARNING_RATE')
    weight_decay = config_manager.get('MODEL.WEIGHT_DECAY')
    
    print(f"\n--- Training Model for Category {category} (D_in={input_dim}, D_latent={latent_dim}) ---")
    
    # Data Tensors
    train_x = torch.tensor(df_cat[input_features].values, dtype=torch.float32)
    train_y = torch.tensor(df_cat[target_column].values, dtype=torch.float32)
    
    if len(train_x) == 0:
        print(f"WARNING: No data found for Category {category}. Skipping training.")
        return None, None

    # Initialize Model Components
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    feature_extractor = DeepKernel(input_dim=input_dim, latent_dim=latent_dim)
    model = DeepKernelGP(train_x, train_y, likelihood, feature_extractor)

    model.train()
    likelihood.train()

    # Define Optimizer and Loss
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print(f"Training on {len(train_x)} data points for {num_epochs} epochs...")
    
    # Training Loop
    for i in range(num_epochs):
        optimizer.zero_grad()
        output = model(train_x) 
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 200 == 0:
            print(f"  Cat {category} | Epoch {i+1:04d}/{num_epochs}: Loss = {loss.item():.4f}")

    # Save Model State and Training Data
    model_save_path = os.path.join(artifacts_dir, DKGP_MODEL_FILENAME.format(category))
    data_save_path = os.path.join(artifacts_dir, TRAIN_DATA_FILENAME.format(category))

    torch.save(model.state_dict(), model_save_path)
    torch.save({'train_x': train_x, 'train_y': train_y}, data_save_path)
    
    print(f"Trained model state and training data saved for Category {category}.")

    return model, likelihood

# --- 3. Main Orchestration Function ---
def train_dkgp_dual_models(
    config_manager: Any # Using Any to avoid circular dependency on ConfigManager class
) -> Dict[int, Tuple[DeepKernelGP, gpytorch.likelihoods.GaussianLikelihood] | Tuple[None, None]]:
    """
    Loads data, splits by category (0 and 1), and trains two separate DKGP models.
    """
    processed_dir = config_manager.get('PATHS.PROCESSED_DIR')
    artifacts_dir = config_manager.get('PATHS.ARTIFACTS_DIR')
    target_column = config_manager.get('COLUMNS.TARGET')
    category_feature_name = config_manager.get('COLUMNS.CATEGORY')
    
    print(f"\n--- Surrogate Model Training Start (Dual Models) ---")

    # Load Data
    processed_file_path = os.path.join(processed_dir, PROCESSED_DATA_FILENAME)
    try:
        df = pd.read_csv(processed_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Processed data not found at {processed_file_path}.")

    # 1. Split Data by Category
    df_cat_0 = df[df[category_feature_name] == 0].copy().drop(columns=[category_feature_name])
    df_cat_1 = df[df[category_feature_name] == 1].copy().drop(columns=[category_feature_name])

    # Ensure artifacts directory exists
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 2. Train Model 0
    model_0, likelihood_0 = train_single_model(0, df_cat_0, config_manager)

    # 3. Train Model 1
    model_1, likelihood_1 = train_single_model(1, df_cat_1, config_manager)

    print(f"\n--- Dual Model Training Complete ---")
    
    return {
        0: (model_0, likelihood_0),
        1: (model_1, likelihood_1)
    }