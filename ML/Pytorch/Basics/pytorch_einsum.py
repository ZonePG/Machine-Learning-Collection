import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Permutation of Tensors
print(torch.einsum("ij->ji", x))

# Summation
print(torch.einsum("ij->", x))

# Column Sum
print(torch.einsum("ij->j", x))

# Row Sum
print(torch.einsum("ij->i", x))

# Matrix-Vector Multiplication
v = torch.tensor([[1, 2, 3]])
print(torch.einsum("ij,kj->ik", x, v))

# Matrix-Matrix Mutiplication
print(torch.einsum("ij,kj->ik", x, x))

# Dot product first row with first row of matrix
print(torch.einsum("i,i->", x[0], x[0]))

# Dot product with matrix
print(torch.einsum("ij,ij->", x, x))

# Hadamard Product (element-wise multiplication)
print(torch.einsum("ij,ij->ij", x, x))

# Outer Product
a = torch.rand((3))
b = torch.rand((5))
print(torch.einsum("i,j->ij", a, b))

# Batch Matrix Multiplication
a = torch.rand((3, 2, 5))
b = torch.rand((3, 5, 3))
print(torch.einsum("ijk,ikl->ijl", a, b))

# Matrix Diagonal
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(torch.einsum("ii->i", x))

# Matrix Trace
print(torch.einsum("ii->", x))