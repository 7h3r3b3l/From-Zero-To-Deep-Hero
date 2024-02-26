import torch



# Define the matrix
matrix = torch.rand([10,10])
# Use of where function
rand_ones = torch.where(matrix > 0.5, 0, 1)
print(rand_ones)
