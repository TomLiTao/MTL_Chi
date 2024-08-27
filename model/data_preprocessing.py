import numpy as np
import pandas as pd
from copy import deepcopy
from xenonpy.datatools.transform import Scaler
from model.load_data import DataLoader, DataLoader_pred
from sklearn.model_selection import GroupKFold

class DescriptorProcessor:
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
        # Initialize attributes for filtered descriptor names
        self.dname_p = None
        self.dname_s = None

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
        
        # Filter out constant descriptors
        self.filter_constant_descriptors()

    def filter_constant_descriptors(self):
        """
        Filter out constant descriptors from the datasets.
        """
        # Filter out constant descriptors for dname_rd
        tmp_desc = pd.concat([
            self.desc_s0_s.loc[self.data_loader.idx_split_s0['idx_tr'], self.dname_rd],
            self.desc_s_s.loc[self.data_loader.idx_split_s['idx_tr'], self.dname_rd],
            self.desc_t_s.loc[self.data_loader.idx_split_t['idx_tr'], self.dname_rd]
        ], axis=0)
        dname_rd_fil = tmp_desc.columns[tmp_desc.std() != 0]
        dname_p_rd_fil = np.intersect1d(dname_rd_fil, self.data_loader.dname_p_rd)
        dname_s_rd_fil = np.intersect1d(dname_rd_fil, self.data_loader.dname_s_rd)

        # Filter out constant descriptors for dname_ff
        dname_ff = np.concatenate([self.data_loader.dname_p_ff, self.data_loader.dname_s_ff])
        tmp_desc = pd.concat([
            self.desc_s0_s.loc[self.data_loader.idx_split_s0['idx_tr'], dname_ff],
            self.desc_s_s.loc[self.data_loader.idx_split_s['idx_tr'], dname_ff],
            self.desc_t_s.loc[self.data_loader.idx_split_t['idx_tr'], dname_ff]
        ], axis=0)
        dname_ff_fil = tmp_desc.columns[tmp_desc.std() != 0]
        dname_p_ff_fil = np.intersect1d(dname_ff_fil, self.data_loader.dname_p_ff)
        dname_s_ff_fil = np.intersect1d(dname_ff_fil, self.data_loader.dname_s_ff)

        # Combine filtered descriptors
        self.dname_p = np.concatenate([dname_p_ff_fil, dname_p_rd_fil])
        self.dname_s = np.concatenate([dname_s_ff_fil, dname_s_rd_fil])

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
        
class DescriptorProcessor_pred:
            def __init__(self, data_loader_pred):
                """
                Initialize the DescriptorProcessor_pred with an instance of DataLoader_pred.

                Parameters:
                data_loader_pred (DataLoader_pred): An instance of the DataLoader_pred class.
                """
                self.data_loader_pred = data_loader_pred
                # Concatenate descriptor names from two sources
                self.dname_rd = np.concatenate([data_loader_pred.dname_p_rd, data_loader_pred.dname_s_rd])
                self.dname_ff = np.concatenate([data_loader_pred.dname_p_ff, data_loader_pred.dname_s_ff])
                self.dname_p = np.concatenate([self.data_loader_pred.dname_p_ff, self.data_loader_pred.dname_p_rd])
                self.dname_s = np.concatenate([self.data_loader_pred.dname_s_ff, self.data_loader_pred.dname_s_rd])
                # Create deep copies of the descriptor DataFrames
                self.desc = deepcopy(data_loader_pred.desc)
                # Initialize a Scaler object with Yeo-Johnson transformation and standard scaling
                self.x_scaler = Scaler().yeo_johnson().standard()

            def fit_scaler(self):
                self.temp_desc = self.desc[self.dname_rd]
                # Fit the scaler on the concatenated training descriptors
                _ = self.x_scaler.fit(self.temp_desc.drop_duplicates(keep='first'))
                

            def transform_descriptors(self):
                """
                Transform the descriptors in the demo datasets using the fitted scaler.
                """
                # Transform the descriptors in the PI dataset using the fitted scaler
                self.desc[self.dname_rd] = self.x_scaler.transform(self.desc[self.dname_rd])
  
        

class YValueProcessor:
    def __init__(self, data_loader: DataLoader):
        """
        Initialize the YValueProcessor with an instance of DataLoader.

        Parameters:
        data_loader (DataLoader): An instance of the DataLoader class.
        """
        self.data_loader = data_loader

        # Initialize attributes for scaled y values
        self.y_s0 = None
        self.y_s = None
        self.y_t = None
        self.y_s_s = None
        self.y_t_s = None

    def process_y_values(self):
        """
        Process and scale the y values.
        """
        # Extract and rename y values
        self.y_s0 = self.data_loader.data_PI[['soluble']].rename(columns={'soluble': 'y'})
        self.y_s = self.data_loader.data_COSMO[['chi']].rename(columns={'chi': 'y'})
        self.y_t = self.data_loader.data_Chi[['chi']].rename(columns={'chi': 'y'})

        # Scale the y values
        ys_scaler = Scaler().standard()
        _ = ys_scaler.fit(self.y_s.loc[self.data_loader.idx_split_s['idx_tr']].reset_index(drop=True))
        self.y_s_s = ys_scaler.transform(self.y_s)

        yt_scaler = Scaler().standard()
        _ = yt_scaler.fit(self.y_t.loc[self.data_loader.idx_split_t['idx_tr']].reset_index(drop=True))
        self.y_t_s = yt_scaler.transform(self.y_t)

class TemperatureProcessor:
    def __init__(self, data_loader: DataLoader, temp_dim: int):
        """
        Initialize the TemperatureProcessor with an instance of DataLoader and temperature dimension.

        Parameters:
        data_loader (DataLoader): An instance of the DataLoader class.
        temp_dim (int): The dimension of the temperature transformation (1 or 2).
        """
        self.data_loader = data_loader
        self.temp_dim = temp_dim

        # Initialize attributes for scaled temperature values
        self.temp_s = None
        self.temp_t = None
        self.temp_s_s = None
        self.temp_t_s = None

    def process_temperatures(self):
        """
        Process and scale the temperature values.
        """
        # Transform temperature variables based on the specified dimension
        if self.temp_dim == 1:
            self.temp_s = 1 / (self.data_loader.data_COSMO[['temp']] + 273.15)
            self.temp_s.columns = ['T1']
            self.temp_t = 1 / (self.data_loader.data_Chi[['temp']] + 273.15)
            self.temp_t.columns = ['T1']
        elif self.temp_dim == 2:
            self.temp_s = pd.concat([
                1 / (self.data_loader.data_COSMO[['temp']] + 273.15),
                (self.data_loader.data_COSMO[['temp']] + 273.15) ** -2
            ], axis=1)
            self.temp_s.columns = ['T1', 'T2']
            self.temp_t = pd.concat([
                1 / (self.data_loader.data_Chi[['temp']] + 273.15),
                (self.data_loader.data_Chi[['temp']] + 273.15) ** -2
            ], axis=1)
            self.temp_t.columns = ['T1', 'T2']

        # Scale the temperature variables
        tempS_scaler = Scaler().standard()
        _ = tempS_scaler.fit(self.temp_s.loc[self.data_loader.idx_split_s['idx_tr'], :])
        self.temp_s_s = tempS_scaler.transform(self.temp_s)

        tempT_scaler = Scaler().standard()
        _ = tempT_scaler.fit(self.temp_t.loc[self.data_loader.idx_split_t['idx_tr'], :])
        self.temp_t_s = tempT_scaler.transform(self.temp_t)
        
class TemperatureProcessor_pred(TemperatureProcessor):
    def __init__(self, data_loader_pred, temp_dim=1):
        """
        Initialize the TemperatureProcessor_pred with an instance of DataLoader_pred and temperature dimension.

        Parameters:
        data_loader_pred (DataLoader_pred): An instance of the DataLoader_pred class.
        temp_dim (int): The dimension of the temperature transformation (default is 1).
        """
        super().__init__(data_loader_pred, temp_dim)
        self.data_loader_pred = data_loader_pred

    def process_temperatures(self):
        """
        Process and scale the temperature values.
        """
        # Transform temperature variables based on the specified dimension
        if self.temp_dim == 1:
            self.temp_s = 1 / (self.data_loader_pred.data['temp'] + 273.15)
            self.temp_s.columns = ['T1']
        elif self.temp_dim == 2:
            self.temp_s = pd.concat([
                1 / (self.data_loader_pred.data['temp'] + 273.15),
                (self.data_loader_pred.data['temp'] + 273.15) ** -2
            ], axis=1)
            self.temp_s.columns = ['T1', 'T2']

        # Scale the temperature variables
        tempS_scaler = Scaler().standard()
        self.temp_s = self.temp_s.values.reshape(-1, 1)  # Reshape to 2D array
        _ = tempS_scaler.fit(self.temp_s)
        self.temp = tempS_scaler.transform(self.temp_s)
        
class GroupKFoldSplitter:
    def __init__(self, data, target_column, group_column, idx_split, n_splits, random_seed=0):
        """
        Initialize the GroupKFoldSplitter with the necessary parameters.

        Parameters:
        data (pd.DataFrame): The input data containing the target and group columns.
        target_column (str): The name of the target column.
        group_column (str): The name of the group column.
        idx_split (list): The indices for the training data.
        n_splits (int): The number of cross-validation splits.
        random_seed (int): The random seed for reproducibility (default is 0).
        """
        self.data = data
        self.target_column = target_column
        self.group_column = group_column
        self.idx_split = idx_split
        self.n_splits = n_splits
        self.random_seed = random_seed
        self.idx_trs = []
        self.idx_vals = []

    def split(self):
        """
        Perform the group-wise cross-validation split and store the indices.
        """
        # Extract the groups for the training data
        poly_group = self.data.loc[self.idx_split, self.group_column]

        # Initialize GroupKFold
        gp_cv = GroupKFold(n_splits=self.n_splits)

        # Set the random seed for reproducibility
        np.random.seed(self.random_seed)

        # Perform the split and store the indices
        for idx_tr, idx_val in gp_cv.split(self.data.loc[self.idx_split, self.target_column], groups=poly_group.to_list()):
            self.idx_trs.append(self.data.loc[self.idx_split].iloc[idx_tr].index.values)
            self.idx_vals.append(self.data.loc[self.idx_split].iloc[idx_val].index.values)

    def get_splits(self):
        """
        Get the training and validation indices for each fold.

        Returns:
        tuple: A tuple containing two lists - training indices and validation indices.
        """
        return self.idx_trs, self.idx_vals
# Example usage
if __name__ == "__main__":
    import os
    dir_load = os.path.join(os.getcwd(), 'demo_data')

    data_loader_pred = DataLoader_pred(dir_load)  

    # Load the data using the DataLoader_pred instance
    data_loader_pred.load_data()
    data_loader_pred.load_descriptions()
    data_loader_pred.load_indices()
    

    # Create an instance of DescriptorProcessor_pred
    descriptor_processor_pred = DescriptorProcessor_pred(data_loader_pred)

    # Fit the scaler on the descriptors
    descriptor_processor_pred.fit_scaler()

    # Transform the descriptors using the fitted scaler
    descriptor_processor_pred.transform_descriptors()
    
    # Create an instance of TemperatureProcessor_pred
    temperature_processor_pred = TemperatureProcessor_pred(data_loader_pred)

    # Process and scale the temperature values
    temperature_processor_pred.process_temperatures()
    print(temperature_processor_pred.temp)




