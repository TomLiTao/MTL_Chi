import os
import torch
from model.load_data import DataLoader_pred
from model.data_preprocessing import DescriptorProcessor_pred, TemperatureProcessor_pred
from model.utils import load_NN
import torch
import os
import pandas as pd

# Example usage
if __name__ == "__main__":
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

    # Load the model
    model_path = '/home/lulab/Projects/ml_for_polymer/MTL_Chi/final_models/MT_testset_1/model_1/best_loss_target_val.pt'
    model = load_NN(model_path)
    cuda_opt = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(cuda_opt)
    model.eval()  # Set the model to evaluation mode
    
   # dname_p and dname_s are lists of column names
    dname_p = descriptor_processor_pred.dname_p
    dname_s = descriptor_processor_pred.dname_s


# Construct tensors from the DataFrame using valid column names
    XT_P_TE = torch.tensor(descriptor_processor_pred.desc[dname_p].values.astype("float"), dtype=torch.float32, device=cuda_opt)
    XT_S_TE = torch.tensor(descriptor_processor_pred.desc[dname_s].values.astype("float"), dtype=torch.float32, device=cuda_opt)
    XT_T_TE = torch.tensor(temperature_processor_pred.temp.astype("float"), dtype=torch.float32, device=cuda_opt)
    
    # Reshape XT_P_TE to have shape (3, 330)
    XT_P_TE = XT_P_TE[:3, :330]
    XT_S_TE = XT_S_TE[:3, :337]
    print(XT_T_TE.shape)
    # Perform the forward pass to get predictions
    with torch.no_grad():  # Disable gradient calculation for inference
        py_target_test = model(XT_P_TE, XT_S_TE, XT_T_TE)

    # Convert the predictions to numpy array
    tmp_mat = py_target_test.to('cpu').detach().numpy()

    # Convert the predictions to a pandas DataFrame
    predictions_df = pd.DataFrame(tmp_mat)

    # Save the predictions to a CSV file
    predictions_df.to_csv('/home/lulab/Projects/ml_for_polymer/MTL_Chi/demo_data/predictions.csv', index=True)




