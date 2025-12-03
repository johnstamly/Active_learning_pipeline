import torch
import gpytorch

class DeepKernel(torch.nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, mlp_depth: int = 2):
        super().__init__()
        dims = [input_dim]
        dims.append(input_dim + latent_dim)
        for k in range(2, mlp_depth):
            dims.append(max(dims[-1] - 2, latent_dim))
        dims.append(latent_dim)

        self.activation = torch.nn.Tanh() # Tanh is often smoother than ReLU for GPs
        self.dropout = torch.nn.Dropout(p=0.2)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(dims[i], dims[i+1]) for i in range(mlp_depth)])
        
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
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