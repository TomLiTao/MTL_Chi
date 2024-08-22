import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from xenonpy.model import SequentialLinear
from model.modelstructure import Chi_Model

# functions to save and load the NN model
def save_NN(paras_p, paras_s, dim_out, c_mdl, file_name):
    torch.save({'model_p': paras_p, 'model_s': paras_s, 'chi': c_mdl.state_dict(), 'dim_out': dim_out}, file_name)
    
def load_NN(file_name):
    tmp_paras = torch.load(file_name)
    c_model = Chi_Model(SequentialLinear(**tmp_paras['model_p']), SequentialLinear(**tmp_paras['model_s']), tmp_paras['dim_out'],temp_dim=1)
    _ = c_model.load_state_dict(tmp_paras['chi'])
    return c_model