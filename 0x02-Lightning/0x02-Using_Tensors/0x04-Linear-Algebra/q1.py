import torch



# Q1
t1 = torch.tensor([1.2,5.1, -4.6])
t2 = torch.tensor([-2.1, 3.1, 5.5])
print(t1.dot(t2) )


# Q2

A = torch.tensor([[1,2],[3,4]])
B = torch.tensor([[5,6],[7,8]])
print(A,B)
print(A @ B)

