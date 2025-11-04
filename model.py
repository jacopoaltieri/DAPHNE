import numpy as np
import torch
import torch.nn as nn
import config

device = config.DEVICE

class InnerNormalization(nn.Module):
    """
    Normalizes each feature using precomputed mean and std,
    without altering the physical meaning of external data.
    """
    def __init__(self, mean, std):
        super(InnerNormalization, self).__init__()
        # Register as buffers so they're saved with the model but not trained
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        return (x - self.mean) / self.std


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(5, 128),  nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            # nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Xavier (Glorot) initialization for all linear layers
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        return self.net(x)

# Instantiate and move to device
pinn_model = PINN().to(device)

def source_term_Q(x, z, z_s):
  
    Q_PRIME = config.Q_PRIME
    SIGMA = config.SIGMA
    X_SOURCE_POS = config.X_SOURCE_POS
    
    rho_cp = config.RHO * config.C  # volumetric heat capacity [J/mm^3/C] 
    
    r_sq = (x - X_SOURCE_POS)**2 + (z - z_s)**2
    power_density = (Q_PRIME / (2 * torch.pi * SIGMA**2)) * torch.exp(-r_sq / (2 * SIGMA**2))
    
    Q_pinn = power_density / rho_cp
    return Q_pinn


