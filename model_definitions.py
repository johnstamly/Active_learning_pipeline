import torch
import gpytorch

class DeepKernel(torch.nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        # Hidden layer with more neurons than input (e.g., 2x input) to capture interactions
        hidden_dim = input_dim * 2 
        
        self.layer1 = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.Tanh() # Tanh is often smoother than ReLU for GPs
        self.layer2 = torch.nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

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