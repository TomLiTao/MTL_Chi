import numpy as np
import pandas as pd
from copy import deepcopy
from xenonpy.datatools.transform import Scaler
import os
from load_data import DataLoader

class DescriptorScaler:
    def __init__(self, data_loader):
        """
        Initialize the DescriptorScaler with an instance of DataLoader.

        Parameters:
        data_loader (DataLoader): An instance of the DataLoader class.
        """
        self.data_loader = data_loader
        # Concatenate descriptor names from two sources
        self.dname_rd = np.concatenate([data_loader.dname_p_rd, data_loader.dname_s_rd])
        # Create deep copies of the descriptor DataFrames
        self.desc_s0_s = deepcopy(data_loader.desc_PI)
        self.desc_s_s = deepcopy(data_loader.desc_COSMO)
        self.desc_t_s = deepcopy(data_loader.desc_Chi)
        # Initialize a Scaler object with Yeo-Johnson transformation and standard scaling
        self.x_scaler = Scaler().yeo_johnson().standard()

    def fit_scaler(self):
        """
        Fit the scaler on the concatenated training descriptors from PI, COSMO, and Chi datasets.
        """
        # Concatenate the training descriptors from PI, COSMO, and Chi datasets
        tmp_desc = pd.concat([
            self.desc_s0_s.loc[self.data_loader.idx_split_s0['idx_tr'], self.dname_rd],  # Training descriptors from PI dataset
            self.desc_s_s.loc[self.data_loader.idx_split_s['idx_tr'], self.dname_rd],    # Training descriptors from COSMO dataset
            self.desc_t_s.loc[self.data_loader.idx_split_t['idx_tr'], self.dname_rd]     # Training descriptors from Chi dataset
        ], axis=0)
        
        # Fit the scaler on the concatenated training descriptors, dropping duplicates
        _ = self.x_scaler.fit(tmp_desc.drop_duplicates(keep='first'))

    def transform_descriptors(self):
        """
        Transform the descriptors in the PI, COSMO, and Chi datasets using the fitted scaler.
        """
        # Transform the descriptors in the PI dataset using the fitted scaler
        self.desc_s0_s[self.dname_rd] = self.x_scaler.transform(self.desc_s0_s[self.dname_rd])
        # Transform the descriptors in the COSMO dataset using the fitted scaler
        self.desc_s_s[self.dname_rd] = self.x_scaler.transform(self.desc_s_s[self.dname_rd])
        # Transform the descriptors in the Chi dataset using the fitted scaler
        self.desc_t_s[self.dname_rd] = self.x_scaler.transform(self.desc_t_s[self.dname_rd])

# Example usage
if __name__ == "__main__":
    # Assuming DataLoader is already defined and data is loaded
    dir_load = os.path.join(os.getcwd(), 'sample_data')
    test_ratio = 0.2  # Set the desired test ratio
    cvtestidx = 1     # Set the desired cross-validation index

    data_loader = DataLoader(dir_load, test_ratio, cvtestidx)
    data_loader.load_data()
    data_loader.load_descriptions()
    data_loader.load_indices()

    # Perform data splitting
    data_loader.idx_split_t = data_loader.split_exp_chi()
    data_loader.idx_split_s = data_loader.split_cosmo(data_loader.idx_split_t)
    data_loader.idx_split_s0 = data_loader.split_pi(data_loader.idx_split_t)

    # Initialize and use DescriptorScaler
    descriptor_scaler = DescriptorScaler(data_loader)
    descriptor_scaler.fit_scaler()
    descriptor_scaler.transform_descriptors()

    # Access the transformed descriptors
    print(descriptor_scaler.desc_s0_s.head())
    print(descriptor_scaler.desc_s_s.head())
    print(descriptor_scaler.desc_t_s.head())