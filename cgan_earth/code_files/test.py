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
p1d = (2,2)
out = F.pad(a, p1d, "circular")
print(a)
print(out)