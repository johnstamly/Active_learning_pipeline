import torch
import gpytorch

class DeepKernel(torch.nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, mlp_depth: int = 6, 
                 skip_every: int = 2, use_dense: bool = False):
        super().__init__()
        self.skip_every = skip_every
        self.use_dense = use_dense
        
        # Calculate dimensions with growth factor
        dims = [input_dim]
        growth_rate = (latent_dim - input_dim) / max(mlp_depth - 1, 1)
        for i in range(1, mlp_depth):
            next_dim = int(input_dim + i * growth_rate)
            dims.append(max(next_dim, latent_dim // 2))
        dims[-1] = latent_dim
        
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.layers = torch.nn.ModuleList()
        self.skip_layers = torch.nn.ModuleDict()
        
        for i in range(len(dims) - 1):
            self.layers.append(torch.nn.Linear(dims[i], dims[i+1]))
        
        # Skip connections
        for i in range(0, len(dims) - skip_every, skip_every):
            if dims[i] != dims[i + skip_every]:
                self.skip_layers[f"skip_{i}"] = torch.nn.Linear(dims[i], dims[i + skip_every])
        
        # Init
        for layer in self.layers:
            torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    
    def forward(self, x):
        residual_stack = [x]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)
            residual_stack.append(x)
            
            # Apply skip connection
            if i >= self.skip_every - 1 and (i + 1) % self.skip_every == 0:
                skip_from = i - self.skip_every + 1
                skip_tensor = residual_stack[skip_from]
                skip_key = f"skip_{skip_from}"
                
                if skip_key in self.skip_layers:
                    skip_tensor = self.skip_layers[skip_key](skip_tensor)
                elif skip_tensor.shape[-1] != x.shape[-1]:
                    continue # specific dimension mismatch safety

                x = torch.cat([x, skip_tensor], dim=-1) if self.use_dense else x + skip_tensor
        return x

class DeepKernelGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        latent_dim = feature_extractor.layers[-1].out_features
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )
        self.feature_extractor = feature_extractor

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)