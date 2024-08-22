import torch
import numpy as np
from model.utils import load_NN

# For this example, we're using dummy data. Replace these with your actual data.
num_samples = 1  # Number of samples in your batch
dim_polymer = 330  # Example dimension of polymer input features
dim_solvent = 337  # Example dimension of solvent input features
temp_dim = 1  # Temperature dimension (1 or 2)

# Dummy data (replace with your actual data)
x1_numpy = np.random.rand(num_samples, dim_polymer).astype(np.float32)  # Polymer input tensor
x2_numpy = np.random.rand(num_samples, dim_solvent).astype(np.float32)  # Solvent input tensor
temp_numpy = np.random.rand(num_samples, temp_dim).astype(np.float32)  # Temperature tensor

# Convert numpy arrays to PyTorch tensors
x1 = torch.tensor(x1_numpy, dtype=torch.float32)
x2 = torch.tensor(x2_numpy, dtype=torch.float32)
temp = torch.tensor(temp_numpy, dtype=torch.float32)

model = load_NN('/home/lulab/Projects/ml_for_polymer/MTL_Chi/final_models/MT_testset_1/model_0/best_loss_target_val.pt')
model.eval()  # Set the model to evaluation mode

# Make predictions
with torch.no_grad():  # Disable gradient calculation
    output = model(x1, x2, temp)

# Print output shape
print("Output shape:", output.shape)
# Process output
print("Model output:", output)
