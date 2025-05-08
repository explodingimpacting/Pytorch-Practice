Tensor notes: 
- Tensors are similar to NumPy's ndarrays but can also run on GPU's
- Tensors and NumPy arrays share underlying memory, elimnating the need to copy data.
- Tensors are optimized for automatic differentiation
- Tensors can be init from other tensors and will retain properties (shape, datatype) of argument tensor, unless explicitly overridden.
- shape = (2, 3, ) will output: 
    tensor([[0., 0., 0.],
        [0., 0., 0.]])
- (block of rows, rows within blocks, columns)

"""Operations on Tensors"""
There are 1200 tensor operations (arithmetic, linear algebra, matrix multipication, transposing
indexing, slicing), samling and more that can be done. These operations can be done on CPU or
accelerator such as CUDA, MPS, MTIA, or XPU
Default, tensors are created on CPU, us .to method to move tensors to accelerators
Copying large tensors across devices can be expensive in terms of time & memory

You can use torch.cat to concatenate a sequence of tensors along a given dimension. See torch.stack, another tensor joining operator that is subtly different from torch.cat

Matrix multiplication (Dot Product):
- tensor = [[a, b], [c, d]]
- tensor.T = [[a, c], [b, d]]
- Now tensor @ tensor.T is: 
[[a*a + b*b, a*c + b*d], 
 [c*a + d*b, c*c + d*d]]

Element-wise Multiplication
- tensor = [[a, b], [c, d]]
- tensor * tensor = [[a*a, b*b], [c*c, d*d]]

Single-element tensors If you have a one-element tensor, for example by aggregating all values of a tensor into one value, you can convert it to a Python numerical value using item():

in-place operations: operations that store the result into operand are called in-place. Denoted by _ suffix. For example: x.copy_(y), x.t_() will change x.

- in place operations save some memory, can be problematic when computing derivatives b/c of immediate loss of history. Hence are usually discouraged.