from scipy.ndimage.morphology import binary_erosion, binary_dilation, distance_transform_edt
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from scipy.ndimage import label
import pandas as pd
import numpy as np
import cv2
import os

PATH = os.path.dirname(os.path.realpath(__file__))
print(PATH)

'''

img = cv2.imread(PATH+"/cgan_microstructure_24[0.8].tif", cv2.IMREAD_UNCHANGED)

# clean up specks
a_thr = binary_erosion(img, iterations = 2)
a_thr = binary_dilation(img, iterations = 2)

# do distance transform as prepartion for watershed
distances = distance_transform_edt(a_thr)

# find watershed seeds
seeds = peak_local_max(distances, exclude_border=False, indices =False, min_distance=20, footprint=np.ones((3,3)))
seeds = label(seeds)[0]

# watershed
ws = watershed(img, seeds, mask=a_thr)

# compute region properties
props = regionprops(ws)

# get the sizes for each of the remaining objects and store in dataframe
entries = []
for p in props:
    entry = [p['label'], p['area'], p['perimeter'], *p['centroid']]
    entries.append(entry)

df = pd.DataFrame(entries, columns= ['label', 'area', 'perimeter', 'y', 'x'])

print(df)

plt. figure(1)
plt.imshow(ws, cmap='tab20c')


plt.figure(2)
plt.imshow(img, cmap='gray')

#plt.figure(3)
#plt.hist(img1.flat, bins=10, range=(0,0.05))

plt.show()

'''