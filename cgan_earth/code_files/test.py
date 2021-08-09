import numpy as np
import torch

array1 = np.array([[[1, 1], [1, 1], [1, 1]],
                  [[2, 2], [2, 2], [2, 2]],
                  [[3, 3], [3, 3], [3, 3]]])
array2 = np.array([1, 2, 3])

shuffler = np.random.permutation(len(array1))
print(array1.shape)
array1_shuffled = array1[shuffler]
array2_shuffled = array2[shuffler]

print(array1_shuffled)
print([4]*2)