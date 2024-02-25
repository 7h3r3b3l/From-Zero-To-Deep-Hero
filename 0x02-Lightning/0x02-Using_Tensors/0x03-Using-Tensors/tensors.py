import torch
import numpy as np


# Define a tensor
t1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
# Checking the shape of a tensor
shape = t1.shape
# Checking the number of dimentions in a tensor
dimentions = t1.ndim
type_of_tensor = t1.dtype
# Numpy array
np_arr = np.random.normal(0, 10, [2,4])
torch_tensor = torch.from_numpy(np_arr)
# Get devive
dev1 = t1.device
dev2 = torch_tensor.device
# Reshape
r1 = torch_tensor.reshape(4,2)
