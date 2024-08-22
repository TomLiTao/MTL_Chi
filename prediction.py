
# basic
import pandas as pd
import numpy as np
import pickle as pk
import os, joblib
from copy import deepcopy
from model.utils import load_NN
from model.modelstructure import Chi_Model
from model.load_data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, KFold, GroupShuffleSplit


# pytorch
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

# XenonPy
from xenonpy.datatools import Splitter
from xenonpy.datatools.transform import Scaler


# plotting figures
import matplotlib.pyplot as plt
import seaborn as sns

# user-friendly printout
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import xenonpy
xenonpy.__version__

#parameters setting
cuda_opt = 'cuda:0' # change to cpu if not using GPU
cvtestidx = 1 # index of the set of randomly split test data stored in test_cv_idx.pkl
test_ratio = 1 # ratio of test data in the whole dataset
temp_dim = 1 # 1 or 2
n_CV_val = 5 # number of cross-validation folds

import pandas as pd
from copy import deepcopy
dir_load = 'sample_data'

data_PI = pd.read_csv(f'{dir_load}/data_PI.csv', index_col=0)
data_COSMO = pd.read_csv(f'{dir_load}/data_COSMO.csv', index_col=0)
data_Chi = pd.read_csv(f'{dir_load}/data_Chi.csv', index_col=0)
desc_PI = pd.read_csv(f'{dir_load}/desc_PI.csv', index_col=0)
desc_COSMO = pd.read_csv(f'{dir_load}/desc_COSMO.csv', index_col=0)
desc_Chi = pd.read_csv(f'{dir_load}/desc_Chi.csv', index_col=0)

with open(f'{dir_load}/desc_names.pkl', 'rb') as f:
    tmp = pk.load(f)
dname_p_ff = deepcopy(tmp['p_ff'])
dname_p_rd = deepcopy(tmp['p_rd'])
dname_s_ff = deepcopy(tmp['s_ff'])
dname_s_rd = deepcopy(tmp['s_rd'])

with open(f'{dir_load}/test_cv_idx.pkl', 'rb') as f:
    tmp = pk.load(f)
tmp_idx_trs = deepcopy(tmp['train'])
tmp_idx_vals = deepcopy(tmp['test'])

    

dname_rd = np.concatenate([dname_p_rd, dname_s_rd])
desc_s0_s = deepcopy(desc_PI)
desc_s_s = deepcopy(desc_COSMO)
desc_t_s = deepcopy(desc_Chi)

# Exp-Chi data splitting
idx = cvtestidx
idx_split_t = {'idx_tr': deepcopy(tmp_idx_trs[idx]), 'idx_te': deepcopy(tmp_idx_vals[idx])}
idx_split_t['idx_tr'].shape[0], idx_split_t['idx_te'].shape[0]

# COSMO data splitting (exclude test ps_pair cases first)
idx = data_COSMO['ps_pair'].apply(lambda x: x in data_Chi['ps_pair'].loc[idx_split_t['idx_te']].values)
idx.sum()

sp_s = Splitter(size=(~idx).sum(), test_size=test_ratio, random_state=0)
_, tmp_idx = sp_s.split(idx.index[~idx].values)
idx.loc[tmp_idx] = True
idx_split_s = {'idx_tr': data_COSMO.index[~idx], 'idx_te': data_COSMO.index[idx]}
idx.sum(), (~idx).sum(), idx.sum()/idx.shape[0]*100

# PI data splitting (exclude test polymer cases first)
idx = data_PI['ps_pair'].apply(lambda x: x in data_Chi['ps_pair'].loc[idx_split_t['idx_te']].values)
idx.sum()

sp_s0 = Splitter(size=(~idx).sum(), test_size=test_ratio, random_state=0)
_, tmp_idx = sp_s0.split(idx.index[~idx].values)
idx.loc[tmp_idx] = True
idx_split_s0 = {'idx_tr': data_PI.index[~idx], 'idx_te': data_PI.index[idx]}
idx.sum(), (~idx).sum(), idx.sum()/idx.shape[0]*100


tmp_desc = pd.concat([desc_s0_s.loc[idx_split_s0['idx_tr'],dname_rd], desc_s_s.loc[idx_split_s['idx_tr'],dname_rd], desc_t_s.loc[idx_split_t['idx_tr'],dname_rd]], axis=0)
# tmp_desc = desc_t_s.loc[idx_split_t['idx_tr'],dname_rd]

x_scaler = Scaler().yeo_johnson().standard()
_ = x_scaler.fit(tmp_desc.drop_duplicates(keep='first'))
desc_s0_s[dname_rd] = x_scaler.transform(desc_s0_s[dname_rd])
desc_s_s[dname_rd] = x_scaler.transform(desc_s_s[dname_rd])
desc_t_s[dname_rd] = x_scaler.transform(desc_t_s[dname_rd])

dname_rd = np.concatenate([dname_p_rd, dname_s_rd])
desc_s0_s = deepcopy(desc_PI)
desc_s_s = deepcopy(desc_COSMO)
desc_t_s = deepcopy(desc_Chi)

tmp_desc = pd.concat([desc_s0_s.loc[idx_split_s0['idx_tr'],dname_rd], desc_s_s.loc[idx_split_s['idx_tr'],dname_rd], desc_t_s.loc[idx_split_t['idx_tr'],dname_rd]], axis=0)
# tmp_desc = desc_t_s.loc[idx_split_t['idx_tr'],dname_rd]

x_scaler = Scaler().yeo_johnson().standard()
_ = x_scaler.fit(tmp_desc.drop_duplicates(keep='first'))
desc_s0_s[dname_rd] = x_scaler.transform(desc_s0_s[dname_rd])
desc_s_s[dname_rd] = x_scaler.transform(desc_s_s[dname_rd])
desc_t_s[dname_rd] = x_scaler.transform(desc_t_s[dname_rd])

# filter out constant descriptors
dname_rd = np.concatenate([dname_p_rd, dname_s_rd])
tmp_desc = pd.concat([desc_s0_s.loc[idx_split_s0['idx_tr'],dname_rd], desc_s_s.loc[idx_split_s['idx_tr'],dname_rd], desc_t_s.loc[idx_split_t['idx_tr'],dname_rd]], axis=0)
dname_rd_fil = tmp_desc.columns[tmp_desc.std() != 0]
dname_p_rd_fil = np.intersect1d(dname_rd_fil, dname_p_rd)
dname_s_rd_fil = np.intersect1d(dname_rd_fil, dname_s_rd)

dname_ff = np.concatenate([dname_p_ff, dname_s_ff])
tmp_desc = pd.concat([desc_s0_s.loc[idx_split_s0['idx_tr'],dname_ff], desc_s_s.loc[idx_split_s['idx_tr'],dname_ff], desc_t_s.loc[idx_split_t['idx_tr'],dname_ff]], axis=0)
dname_ff_fil = tmp_desc.columns[tmp_desc.std() != 0]
dname_p_ff_fil = np.intersect1d(dname_ff_fil, dname_p_ff)
dname_s_ff_fil = np.intersect1d(dname_ff_fil, dname_s_ff)

dname_p = np.concatenate([dname_p_ff_fil, dname_p_rd_fil])
dname_s = np.concatenate([dname_s_ff_fil, dname_s_rd_fil])

y_s0 = data_PI[['soluble']]
y_s0.columns = ['y']
y_s = data_COSMO[['chi']]
y_s.columns = ['y']
y_t = data_Chi[['chi']]
y_t.columns = ['y']

if temp_dim == 1:
    temp_s = 1/(data_COSMO[['temp']] + 273.15)
    temp_s.columns = ['T1']
    temp_t = 1/(data_Chi[['temp']] + 273.15)
    temp_t.columns = ['T1']
elif temp_dim == 2:
    temp_s = pd.concat([1/(data_COSMO[['temp']] + 273.15), (data_COSMO[['temp']] + 273.15)**(-2)], axis=1)
    temp_s.columns = ['T1', 'T2']
    temp_t = pd.concat([1/(data_Chi[['temp']] + 273.15), (data_Chi[['temp']] + 273.15)**(-2)], axis=1)
    temp_t.columns = ['T1', 'T2']
    
# ys_scaler = Scaler().min_max((-1,1))
ys_scaler = Scaler().standard()
_ = ys_scaler.fit(y_s.loc[idx_split_s['idx_tr']].reset_index(drop=True))
y_s_s = ys_scaler.transform(y_s)

# yt_scaler = Scaler().min_max((-1,1))
yt_scaler = Scaler().standard()
_ = yt_scaler.fit(y_t.loc[idx_split_t['idx_tr']].reset_index(drop=True))
y_t_s = yt_scaler.transform(y_t)

# temp_scaler = Scaler().min_max((-1,1))
tempS_scaler = Scaler().standard()
_ = tempS_scaler.fit(temp_s.loc[idx_split_s['idx_tr'],:])
temp_s_s = tempS_scaler.transform(temp_s)

tempT_scaler = Scaler().standard()
_ = tempT_scaler.fit(temp_t.loc[idx_split_t['idx_tr'],:])
temp_t_s = tempT_scaler.transform(temp_t)

poly_group = data_Chi.loc[idx_split_t['idx_tr'],'ps_pair']

gp_cv = GroupKFold(n_splits=n_CV_val)
idx_trs, idx_vals = [], []

np.random.seed(0)
for idx_tr, idx_val in gp_cv.split(y_t['y'].loc[idx_split_t['idx_tr']], groups=poly_group.to_list()):
    idx_trs.append(y_t['y'].loc[idx_split_t['idx_tr']].iloc[idx_tr].index.values)
    idx_vals.append(y_t['y'].loc[idx_split_t['idx_tr']].iloc[idx_val].index.values)


for iCV, (idx_tr, idx_val) in enumerate(zip(idx_trs, idx_vals)):
    XT_P_TR = torch.tensor(desc_t_s.loc[idx_tr, dname_p].values.astype("float"), dtype=torch.float32, device=cuda_opt)
    XT_S_TR = torch.tensor(desc_t_s.loc[idx_tr, dname_s].values.astype("float"), dtype=torch.float32, device=cuda_opt)
    XT_T_TR = torch.tensor(temp_t_s.loc[idx_tr, :].values.astype("float"), dtype=torch.float32, device=cuda_opt)

print(XT_P_TR.shape)
print(XT_S_TR.shape)
print(XT_T_TR.shape)

model = load_NN('/home/lulab/Projects/ml_for_polymer/MTL_Chi/final_models/MT_testset_1/model_0/best_loss_target_val.pt')
model.to(cuda_opt)
model.eval()  # Set the model to evaluation mode

py_target_train = model(XT_P_TR, XT_S_TR, XT_T_TR)
print(py_target_train.shape)
print(py_target_train)




