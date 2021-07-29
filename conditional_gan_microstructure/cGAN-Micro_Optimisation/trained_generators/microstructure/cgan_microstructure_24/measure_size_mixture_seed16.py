from scipy.ndimage.morphology import binary_erosion, binary_dilation, distance_transform_edt
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.measure import regionprops
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from scipy.ndimage import label
import pandas as pd
import numpy as np
import cv2
import os

PATH = os.path.dirname(os.path.realpath(__file__))

current_fig = 0.5
img = cv2.imread(PATH+f"/cgan_microstructure_24[{current_fig}].tif", cv2.IMREAD_UNCHANGED)

# clean up specks
a_thr = binary_erosion(img, iterations = 1)
a_thr = binary_dilation(img, iterations = 1)

# do distance transform as prepartion for watershed
distances = distance_transform_edt(a_thr)
img = np.where(img <= 0, img, 1)
img = img.astype(int)

# watershed segmentation
coords = peak_local_max(distances, exclude_border=(15,15), footprint=np.ones((3,3)), labels=img)
mask = np.zeros(distances.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
ws = watershed(-distances, markers, mask=img)

# compute region properties
props = regionprops(ws)

# get the sizes for each of the remaining objects and store in dataframe
# if two particles are too close to each other, remove them
entries = []
dist_lim = 10
for p in props:
    entry = [p['label'], p['area'], p['perimeter'], *p['centroid']]
    # calculate radius, assuming spherical particles
    entry.append(np.round(np.sqrt(entry[1]/np.pi),1))
    if entry[1] > 40 and len(entries) > 0:
        overlap = False
        overlap_store = []
        row = 0
        overlap_row = None
        for particle in entries:
            y_dist = abs(entry[3] - particle[3])
            x_dist = abs(entry[4] - particle[4])
            # additional parameter to deal with mixture
            if particle[1] < 200:
                dist_lim = 5
            else:
                dist_lim = 10
            if x_dist < dist_lim and y_dist < dist_lim:
                overlap = True
                overlap_store = particle
                overlap_row = row
            row += 1
        if overlap == False:
            entries.append(entry)
        else:
            # note (138,64)
            #print(f'x: {overlap_store[4]}, y: {overlap_store[3]}')
            #print(f'store area: {overlap_store[1]}, entry area: {entries[overlap_row][1]}')
            # compare area, the one with greater area should substitute 
            if overlap_store[1] > entries[overlap_row][1]:
                entries[overlap_row] = overlap_store
    elif entry[1] > 10 and len(entries) == 0:
        entries.append(entry)

df = pd.DataFrame(entries, columns= ['label', 'area', 'perimeter', 'y', 'x', 'radius'])

print(df)

y_pos = df['y']
x_pos = df['x']

try:
    os.mkdir(PATH+f'/[{current_fig}]')
except:
    pass

df.to_csv(PATH+f'/[{current_fig}]/[{current_fig}].csv', index=False)

plt. figure(1)
plt.imshow(ws, cmap='tab20c')
plt.title(str(current_fig))
plt.savefig(PATH+f'/[{current_fig}]/[{current_fig}]_mask.png')


plt.figure(2)
plt.imshow(img, cmap='gray')
plt.plot(x_pos, y_pos, linestyle='', marker='o')
plt.title(str(current_fig))
plt.savefig(PATH+f'/[{current_fig}]/[{current_fig}]_raw.png')

plt.figure(3)
plt.hist(df['radius'], bins=20, range=(3,11))
plt.title(str(current_fig))
plt.savefig(PATH+f'/[{current_fig}]/[{current_fig}]_hist.png')

plt.show()

