import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from xenonpy.model import SequentialLinear

# Define the model class based on temp_dim
class Chi_Model(nn.Module):
    def __init__(self, sp_mdl_p, sp_mdl_s, dim_ur, temp_dim):
        super(Chi_Model, self).__init__()

        self.network1 = deepcopy(sp_mdl_p)
        self.network2 = deepcopy(sp_mdl_s)

        if temp_dim == 1:
            self.out_lin = nn.Linear(dim_ur, 5)
            self.out_act = nn.Sigmoid()
            self.dim_ur = dim_ur
        elif temp_dim == 2:
            self.out_lin = nn.Linear(dim_ur, 7)
            self.out_act = nn.Sigmoid()
            self.dim_ur = dim_ur

    def forward(self, x1, x2, temp):
        ur1 = self.network1(x1)
        ur2 = self.network2(x2)

        sp = (ur1[:,:self.dim_ur] - ur2[:,:self.dim_ur])**2
        r1 = ur1[:,self.dim_ur:]**2
        r2 = ur2[:,self.dim_ur:]**2

        z0 = sp - r1 - r2
        z = self.out_lin(z0)

        z_soluble = self.out_act(z[:,0:1])

        if self.out_lin.out_features == 5:  # For temp_dim == 1
            As = z[:,1:2]
            Bs = z[:,2:3]
            z_comp = As + Bs*temp[:,0:1]

            At = z[:,3:4]
            Bt = z[:,4:5]
            z_target = At + Bt*temp[:,0:1]
        elif self.out_lin.out_features == 7:  # For temp_dim == 2
            As = z[:,1:2]
            Bs = z[:,2:3]
            Cs = z[:,3:4]
            z_comp = As + Bs*temp[:,0:1] + Cs*temp[:,1:2]

            At = z[:,4:5]
            Bt = z[:,5:6]
            Ct = z[:,6:7]
            z_target = At + Bt*temp[:,0:1] + Ct*temp[:,1:2]

        y = torch.cat((z_soluble, z_comp, z_target, z, z0), dim=1)

        return y