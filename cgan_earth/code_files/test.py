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
l = [[0, 0], [1, 1], [2, 2], [3, 3], [0, 1]]
l = torch.from_numpy(np.array(l))
one_hot = torch.zeros((l.shape[0], 4))
for n in range(l.shape[0]):
    if l[n, 0] == l[n, 1]:
        one_hot[n,l[n]] = 1.0
    else:
        one_hot[n,l[n, 0]] = 0.5
        one_hot[n,l[n, 1]] = 0.5

def gen_intermediate_labels(label1, label2, val, n_classes, device):
    '''
    Generate one-hot label for an intermediate image
    Params:
        label1: integer representing the first label class
        label2: integer representing the second label class
        val: value of the first label class
        n_classes: total number of classes
    Return:
        tensor of shape (n_classes)
    '''
    one_hot = torch.zeros((n_classes)).to(device)
    one_hot[label1] = val
    one_hot[label2] = 1 - val
    return one_hot

print(gen_intermediate_labels(0, 1, 0.1, 4, 'cpu'))