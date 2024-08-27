import pandas as pd
import pickle as pk
from copy import deepcopy
import os
from xenonpy.datatools import Splitter

import pandas as pd
import pickle as pk
from copy import deepcopy
import os
from xenonpy.datatools import Splitter

class DataLoader:
    def __init__(self, dir_load, test_ratio=0.2, cvtestidx=1):
        """
        Initialize the DataLoader with the directory to load data from, test ratio, and cross-validation index.

        Parameters:
        dir_load (str): Directory path to load data from.
        test_ratio (float): Ratio of the test set size for splitting. Default is 0.2.
        cvtestidx (int): Index for cross-validation fold. Default is 1.
        """
        self.dir_load = dir_load
        self.data_PI = None
        self.data_COSMO = None
        self.data_Chi = None
        self.desc_PI = None
        self.desc_COSMO = None
        self.desc_Chi = None
        self.dname_p_ff = None
        self.dname_p_rd = None
        self.dname_s_ff = None
        self.dname_s_rd = None
        self.tmp_idx_trs = None
        self.tmp_idx_vals = None
        self.cvtestidx = cvtestidx
        self.test_ratio = test_ratio

    def load_data(self):
        """
        Load the data from CSV files into pandas DataFrames.
        """
        self.data_PI = pd.read_csv(f'{self.dir_load}/data_PI.csv', index_col=0)
        self.data_COSMO = pd.read_csv(f'{self.dir_load}/data_COSMO.csv', index_col=0)
        self.data_Chi = pd.read_csv(f'{self.dir_load}/data_Chi.csv', index_col=0)
        self.desc_PI = pd.read_csv(f'{self.dir_load}/desc_PI.csv', index_col=0)
        self.desc_COSMO = pd.read_csv(f'{self.dir_load}/desc_COSMO.csv', index_col=0)
        self.desc_Chi = pd.read_csv(f'{self.dir_load}/desc_Chi.csv', index_col=0)

    def load_descriptions(self):
        """
        Load the description names from a pickle file.
        """
        with open(f'{self.dir_load}/desc_names.pkl', 'rb') as f:
            tmp = pk.load(f)
        self.dname_p_ff = deepcopy(tmp['p_ff'])
        self.dname_p_rd = deepcopy(tmp['p_rd'])
        self.dname_s_ff = deepcopy(tmp['s_ff'])
        self.dname_s_rd = deepcopy(tmp['s_rd'])

    def load_indices(self):
        """
        Load the precomputed training and testing indices for cross-validation from a pickle file.
        """
        with open(f'{self.dir_load}/test_cv_idx.pkl', 'rb') as f:
            tmp = pk.load(f)
        self.tmp_idx_trs = deepcopy(tmp['train'])
        self.tmp_idx_vals = deepcopy(tmp['test'])

    def split_exp_chi(self):
        """
        Split the Exp-Chi dataset into training and testing sets based on the cross-validation index.

        Returns idx_split_t:
        dict: A dictionary with training and testing indices.
        """
        idx = self.cvtestidx
        idx_split_t = {'idx_tr': deepcopy(self.tmp_idx_trs[idx]), 'idx_te': deepcopy(self.tmp_idx_vals[idx])}
        return idx_split_t

    def split_cosmo(self, idx_split_t):
        """
        Split the COSMO dataset into training and testing sets, excluding samples in the Exp-Chi test set.

        Parameters:
        idx_split_t (dict): A dictionary with training and testing indices for the Exp-Chi split.

        Returns idx_split_s:
        dict: A dictionary with training and testing indices for the COSMO split.
        """
        idx = self.data_COSMO['ps_pair'].apply(lambda x: x in self.data_Chi['ps_pair'].loc[idx_split_t['idx_te']].values)
        sp_s = Splitter(size=(~idx).sum(), test_size=self.test_ratio, random_state=0)
        _, tmp_idx = sp_s.split(idx.index[~idx].values)
        idx.loc[tmp_idx] = True
        idx_split_s = {'idx_tr': self.data_COSMO.index[~idx], 'idx_te': self.data_COSMO.index[idx]}
        return idx_split_s

    def split_pi(self, idx_split_t):
        """
        Split the PI dataset into training and testing sets, excluding samples in the Exp-Chi test set.

        Parameters:
        idx_split_t (dict): A dictionary with training and testing indices for the Exp-Chi split.

        Returns idx_split_s0:
        dict: A dictionary with training and testing indices for the PI split.
        """
        idx = self.data_PI['ps_pair'].apply(lambda x: x in self.data_Chi['ps_pair'].loc[idx_split_t['idx_te']].values)
        sp_s0 = Splitter(size=(~idx).sum(), test_size=self.test_ratio, random_state=0)
        _, tmp_idx = sp_s0.split(idx.index[~idx].values)
        idx.loc[tmp_idx] = True
        idx_split_s0 = {'idx_tr': self.data_PI.index[~idx], 'idx_te': self.data_PI.index[idx]}
        return idx_split_s0
    
class DataLoader_pred(DataLoader):
    def __init__(self, dir_load):
        """
        Initialize the DataLoader_pred with the directory to load data from.

        Parameters:
        dir_load (str): Directory path to load data from.
        """
        super().__init__(dir_load)
        self.data = None
        self.desc = None

    def load_data(self):
        """
        Load the data from CSV files into pandas DataFrames.
        """
        self.data = pd.read_csv(f'{self.dir_load}/demo_smiles.csv', index_col=0)
        self.desc = pd.read_csv(f'{self.dir_load}/demo_desc.csv', index_col=0)

# Example usage
if __name__ == "__main__":
            dir_load = os.path.join(os.getcwd(), 'demo_data')

            data_loader_pred = DataLoader_pred(dir_load)
            
            data_loader_pred.load_data()
            data_loader_pred.load_descriptions()
            
            # Access the loaded data
            print(data_loader_pred.data.head())
            print(data_loader_pred.desc.head())
            print(data_loader_pred.dname_p_ff[:5])



