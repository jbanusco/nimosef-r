import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class SIREN(nn.Module):
    '''
        Implicit representation with sine nonlinearity

        Inputs;
            in_size: Input features
            out_size: Output features
            bias: if True, enable bias for the linear operation
            omega_0: SIREN parameter
    '''
    def __init__(self, in_size, out_size, bias=True, trainable=False, omega_0=30):
        super(SIREN, self).__init__()
        self.omega_0 = nn.Parameter(omega_0*torch.ones(1), requires_grad=trainable)
        self.linear = nn.Linear(in_size, out_size, bias=bias)

    """
       Weights initialization as per the SIREN paper
       is_first: True for first layer, False for hidden and last layers
    """
    def init_weights_SIREN(self, is_first=True, omega_0=30): 
        with torch.no_grad():
            if is_first:
                in_size = self.linear.weight.size(-1)
                self.linear.weight.uniform_(-1/in_size, 1/in_size)
            else:
                in_size = self.linear.weight.size(-1)
                self.linear.weight.uniform_(-np.sqrt(6/in_size)/omega_0, np.sqrt(6/in_size)/omega_0)

    def forward(self, x):
        x = torch.sin(self.omega_0*self.linear(x))
        return x 

class INR(nn.Module):
    def __init__(self, coord_size: int, embed_size: int, hidden_size: int, out_size: int, 
                 num_hidden_layers: int, use_residual: bool = True):
        super(INR, self).__init__()

        in_fts = coord_size + embed_size
        # hidden_fts = int(hidden_size / 2)  # Complex numbers
        hidden_fts = hidden_size  # Complex numbers
        # dtype = torch.cfloat
        dtype = torch.float

        self.net = nn.ModuleList()

        # Append first layer
        first_layer = SIREN(in_fts, hidden_fts, trainable=False, omega_0=30)
        first_layer.init_weights_SIREN(is_first=True)
        self.net.append(first_layer)

        # Append layers
        for _ in range(num_hidden_layers - 1):
            hidden_layer = SIREN(hidden_fts, hidden_fts, trainable=False, omega_0=30)
            hidden_layer.init_weights_SIREN(is_first=False)
            self.net.append(hidden_layer)

        # Append last layer
        final_layer = nn.Linear(hidden_fts, out_size, dtype=dtype)
        self.net.append(final_layer)

        self.use_residual = use_residual
        if not self.use_residual:
            self.net = nn.Sequential(*self.net)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
        _, x_prev = x
        x = torch.cat(x, dim=1)
        if self.use_residual:
            for layer in self.net:
                x = layer(x) + x_prev
                x_prev = x
        else:
            x = self.net(x).real
        
        return x
    
