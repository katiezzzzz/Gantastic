import numpy as np
import torch

array1 = np.array([[[1, 1], [1, 1], [1, 1]],
                  [[2, 2], [2, 2], [2, 2]],
                  [[3, 3], [3, 3], [3, 3]]])
array2 = np.array([1, 2, 3])

shuffler = np.random.permutation(len(array1))
array1_shuffled = array1[shuffler]
array2_shuffled = array2[shuffler]

a = torch.zeros((20,5))
a = a.repeat(64, 64)
print(a.shape)