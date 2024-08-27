import pandas as pd
import numpy as np
import pickle as pk
import os
import sys

from rdkit import Chem

from datetime import datetime
import matplotlib.pyplot as plt


from xenonpy.descriptor import Fingerprints
import xenonpy
xenonpy.__version__

from tqdm.autonotebook import tqdm
from radonpy.core import poly, calc, const
from radonpy.ff.gaff2 import GAFF2
from radonpy.ff.descriptor import FF_descriptor
const.print_level = 1


data = pd.read_csv('sample_data/data_Chi.csv', index_col=0)
smis_poly = []
smis_solv = []
for smi in data['ps_pair'].values:
    tmp = smi.split('_')
    smis_poly.append(tmp[0])
    smis_solv.append(tmp[1])
    
    
# extract the unique SMILES of polymers and solvents
uni_poly = np.unique(smis_poly)
uni_solv = np.unique(smis_solv)

print(f'Unique number of polymer SMILES: {uni_poly.shape[0]}')
print(f'Unique number of solvent SMILES: {uni_solv.shape[0]}')

# set up a dictionary for descriptor calculation
uni_smis = {'Polymer': uni_poly, 'Solvent': uni_solv}

# set up a dictionary to store all descriptors
desc_data = {}

# set up a dictionary for descriptor calculation
uni_smis = {'Polymer': uni_poly, 'Solvent': uni_solv}

# set up a dictionary to store all descriptors
desc_data = {}

# parameters for force-field descriptors
nk = 20
sigma = 1/nk/2

for key, val in uni_smis.items():
    try:
        ff = GAFF2()
        ff_desc = FF_descriptor(ff, polar=True)
        desc_names = ff_desc.ffkm_desc_names(nk=nk)

        desc = ff_desc.ffkm_mp(list(val), nk=nk, s=sigma, cyclic=0)
            
        desc_data[f'{key}_ff'] = pd.DataFrame(desc, columns=[f'{key}_{x}' for x in desc_names], index=val)
        
        print(datetime.now())
        print(f'{key} done')
        
    except:
        print(f'{key} failed')
        pass
    
print('All done!')

print(datetime.now())
print('Program started...')
for key, val in uni_smis.items():
    mols = [Chem.MolFromSmiles(x) for x in val]
    
    desc_data[f'{key}_rdk'] = Fingerprints(featurizers = 'DescriptorFeature', input_type='mol', on_errors='nan').transform(mols)
    desc_data[f'{key}_rdk']['Ipc'] = np.log(desc_data[f'{key}_rdk']['Ipc'])
    desc_data[f'{key}_rdk'].index = val
    desc_data[f'{key}_rdk'].columns = [f'{key}_{x}' for x in desc_data[f'{key}_rdk'].columns]

    print(datetime.now())
    print(f'{key} done')



