from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import tifffile as tiff
import numpy as np
import os

PATH = os.path.dirname(os.path.realpath(__file__))
numbers = np.around(np.arange(0,1.1,0.1),1)
figure(figsize=(10, 10))
i = 0
f, axarr = plt.subplots(1,11)
for n in numbers:
    im = tiff.imread(PATH + f'/cgan_microstructure_22[{str(n)}].tif')
    axarr[i].imshow(im)
    i += 1
plt.show()