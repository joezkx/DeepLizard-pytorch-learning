import numpy as np
import torch

array_test = np.array(
 [
    [1,1,1],
    [1,1,1],
    [1,1,1],
    [1,1,1],
]
)
# what is rank\axes\shape of an array ?

shape = array_test.shape

rank = array_test.ndim

dim1 = shape[0]
dim2 = shape[1]

print('the array is {}'.format(array_test))
print('the shape of array_test is {}'.format(shape)) # -> []
print('the rank of array_test is {}'.format(rank)) # -> "n?"d
print('the axes of array_test：1st-axis is {},2nd-axis is {}'.format(dim1,dim2)) # -> B/C/W/H

# how to convert array into tensor ?

tensor_a = torch.Tensor(array_test)
# dtype_t = tensor_a.dtype
type_t = type(tensor_a)
# dtype_a = array_test.dtype
type_a = type(array_test)

print("array_test dtype is {},tensor_a dtype is {}".format(type_a,type_t))

# tensor rela attributes&op

## attr

tshape = tensor_a.shape

trank = len(tshape)


print('the tensor is {},device is {} '.format(tensor_a,tensor_a.device))
print('the shape of tensor_a is {}'.format(tshape))
print('the rank of tensor_a is {}'.format(trank))

## op

# 1.Reshaping operations

# (1) aixs mul (2) numel
numOftensor1 =torch.tensor(tshape).prod()
numOftensor2 =tensor_a.numel()

reshapet = tensor_a.reshape(1,numOftensor2)
print('after reshape,the shape of tensor_a is {}'.format(reshapet.shape))

# 2.Element-wise operations



# 3.Reduction operations


# 4.Access operations

# how to input the image as a tensor into CNN ?

# image->[B,C,H,W]
