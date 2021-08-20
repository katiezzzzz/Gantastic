import numpy as np
import torch
import torch.nn.functional as F

a = torch.tensor([[[1, 2, 3, 4],
                  [4, 5, 6, 7],
                  [5, 6, 7, 8],
                  [8, 9, 10, 11]],
                 [[11, 21, 31, 41],
                  [41, 51, 61, 71],
                  [51, 61, 71, 81],
                  [81, 91, 101, 111]]])
print(a[:,:,0])
print(a.shape)
b = torch.zeros((2,4,6))
for idx0 in range(a.shape[0]):
    for idx1 in range(a.shape[1]):
        dim1 = a[idx0,idx1]
        b[idx0,idx1] = torch.cat((dim1, dim1[:2]), -1)
print(b)