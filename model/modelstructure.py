import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from xenonpy.model import SequentialLinear

# fixed linearly reducing pyramid shape
def neuron_vector(nL, in_neu, out_neu):
    return [int(x) for x in np.rint(np.linspace(in_neu, out_neu, nL+2))[1:-1]]

class Chi_Model(nn.Module):
    def __init__(self, sp_mdl_p, sp_mdl_s, dim_ur, temp_dim):
        super(Chi_Model, self).__init__()

        self.network1 = deepcopy(sp_mdl_p)
        self.network2 = deepcopy(sp_mdl_s)

        self.temp_dim = temp_dim
        if temp_dim == 1:
            self.out_lin = nn.Linear(dim_ur, 5)
        elif temp_dim == 2:
            self.out_lin = nn.Linear(dim_ur, 7)
        
        self.out_act = nn.Sigmoid()
        self.dim_ur = dim_ur

    def forward(self, x1, x2, temp):
        ur1 = self.network1(x1)
        ur2 = self.network2(x2)

        sp = (ur1[:, :self.dim_ur] - ur2[:, :self.dim_ur]) ** 2
        r1 = ur1[:, self.dim_ur:] ** 2
        r2 = ur2[:, self.dim_ur:] ** 2

        z0 = sp - r1 - r2
        z = self.out_lin(z0)

        z_soluble = self.out_act(z[:, 0:1])

        if self.temp_dim == 1:
            As = z[:, 1:2]
            Bs = z[:, 2:3]
            z_comp = As + Bs * temp[:, 0:1]

            At = z[:, 3:4]
            Bt = z[:, 4:5]
            z_target = At + Bt * temp[:, 0:1]

        elif self.temp_dim == 2:
            As = z[:, 1:2]
            Bs = z[:, 2:3]
            Cs = z[:, 3:4]
            z_comp = As + Bs * temp[:, 0:1] + Cs * temp[:, 1:2]

            At = z[:, 4:5]
            Bt = z[:, 5:6]
            Ct = z[:, 6:7]
            z_target = At + Bt * temp[:, 0:1] + Ct * temp[:, 1:2]

        y = torch.cat((z_soluble, z_comp, z_target, z, z0), dim=1)

        return y
    
# Example usage
if __name__ == "__main__":
    # Assuming sp_mdl_p and sp_mdl_s are already defined models
    sp_mdl_p = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
    sp_mdl_s = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
    dim_ur = 10
    temp_dim = 1  # or 2, depending on the desired temperature transformation

    model = Chi_Model(temp_dim)

    # Dummy input data
    x1 = torch.randn(5, 10)
    x2 = torch.randn(5, 10)
    temp = torch.randn(5, temp_dim)

    # Set the model to evaluation mode
    model.eval()

    # Perform a forward pass in evaluation mode
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(x1, x2, temp)
    
    print(output)
    