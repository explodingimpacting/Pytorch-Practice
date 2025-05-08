"""
Quick pytorch tensor overview
"""

import torch
import numpy as np

#initialize a tensor
#directly from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

#from a numpy array
#tensors can be created from NumPy arrays and vice versa (look up bridging)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#from another tensor
#new tensor retains properties (shape, datatype) of arg tensor, unless explicitly overridden
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

#overrides the datatype of x_data
x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

#with random or constant values:
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")

#Attributes of a Tensor
#tensors attributes describe their shape, datatype, and device on which they are stored.
tensor = torch.rand(3, 4)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Datatype of tensor: \n {} \n")
print(f"Device tensor is stored on: \n {tensor.device} \n")

#operations on Tensors 
"""
there are 1200 tensor operations (arithmetic, linear algebra, matrix multipication, transposing
indexing, slicing)"""

if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())

#standard numpy-like indexing and slicing
tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:,0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0 #for every row set column 1 to 0
print(tensor)

#joining tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

#arithmetic operations
#computes matrix multiplication of 2 tensors. y1, y2, y3 will have same value.
#''tensor.T'' returns transpose of a tensor
#all 3 lines do the same thing

y1 = tensor @ tensor.T # using @ operator
y2 = tensor.matmul(tensor.T) # same operation as above just using .matmul
y3 = torch.rand_like(y1) # same result but stored in y3
torch.matmul(tensor, tensor.T, out=y3)

#this computes element-wise product. z1, z2, z3 will have same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

#single element tensor
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

#in place operations
print(f"{tensor}\n")
tensor.add_(5)
print(tensor)

#bridge with NumPy; tensor to numpy
#tensors on CPU and NumPy arrays can share their underlying mem locations, & changing one will change the other.
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

#numpy array to tensor
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

