## Exercies from part 2.1 - Data Manipulation
import torch

## 1. Run the code in this section. Change the conditional statement X == Y to X < Y or X < Y and see what kind of tensor you can get

# Create the two tensors
T1 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
T2 = torch.tensor([[9,8,7],[6,5,4],[3,2,1]])
# Compare the two tensors with different operations
eq = T1 == T2
gt = T1 > T2
lt = T1 < T2
## 2. Replace the two tensors that operate by element in te broadcasting mechanismm with other shapes
b1 = torch.tensor(T1, dtype=torch.float64).reshape(-1)
b2 = torch.tensor(lt).reshape(-1)
