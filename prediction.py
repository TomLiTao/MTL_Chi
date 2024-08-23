import os
import torch
from model.load_data import DataLoader
from model.data_preprocessing import DescriptorProcessor, YValueProcessor, TemperatureProcessor
from model.utils import load_NN
import torch
import os

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
    
    # Initialize DescriptorProcessor with the DataLoader instance
    descriptor_processor = DescriptorProcessor(data_loader)
    descriptor_processor.filter_constant_descriptors()  # Filter descriptors

    # Initialize and use YValueProcessor
    y_processor = YValueProcessor(data_loader)
    y_processor.process_y_values()

    # Initialize and use TemperatureProcessor
    temp_dim = 1  # or 2, depending on the desired temperature transformation
    temp_processor = TemperatureProcessor(data_loader, temp_dim)
    temp_processor.process_temperatures()

    # Load the model
    model_path = '/home/lulab/Projects/ml_for_polymer/MTL_Chi/final_models/MT_testset_1/model_1/best_loss_target_val.pt'
    model = load_NN(model_path)
    cuda_opt = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(cuda_opt)
    model.eval()  # Set the model to evaluation mode

    # Prepare the input tensors
    XT_P_TE = torch.tensor(descriptor_processor.desc_t_s.loc[data_loader.idx_split_t['idx_te'], descriptor_processor.dname_p].values.astype("float"), dtype=torch.float32, device=cuda_opt)
    XT_S_TE = torch.tensor(descriptor_processor.desc_t_s.loc[data_loader.idx_split_t['idx_te'], descriptor_processor.dname_s].values.astype("float"), dtype=torch.float32, device=cuda_opt)
    XT_T_TE = torch.tensor(temp_processor.temp_t_s.loc[data_loader.idx_split_t['idx_te'], :].values.astype("float"), dtype=torch.float32, device=cuda_opt)
    
    
    # Print the first few entries of the tensors
    print("First few entries of XT_P_TE:", XT_P_TE[:5])
    print("First few entries of XT_S_TE:", XT_S_TE[:5])
    print("First few entries of XT_T_TE:", XT_T_TE[:5])

    # Perform the forward pass to get predictions
    with torch.no_grad():  # Disable gradient calculation for inference
        py_target_test = model(XT_P_TE, XT_S_TE, XT_T_TE)

    # Convert the predictions to numpy array
    tmp_mat = py_target_test.to('cpu').detach().numpy()

    # Print the predictions
    print(tmp_mat)




