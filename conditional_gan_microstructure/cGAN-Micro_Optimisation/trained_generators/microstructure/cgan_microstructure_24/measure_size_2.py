import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops
import os 

#----------------------------------------------------------------------------------------------------------------------#
# IMAGE PRETREATMENT

PATH = os.path.dirname(os.path.realpath(__file__))

img = cv2.imread(PATH+"/cgan_microstructure_24[0.8].tif", cv2.IMREAD_UNCHANGED)
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#Gaussian_Blur = cv2.GaussianBlur(gray,(21, 21), cv2.BORDER_DEFAULT)

# Use fixed threshold to mask black areas
_, thresh = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY_INV) # _ = 30

# Morphological closing to close holes inside particles; opening to get rid of noise
img_mop1 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
img_mop = cv2.morphologyEx(img_mop1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
tiled_h = np.hstack((img_mop1, img_mop)) # stack images side-by-side

plt.figure('Pretreatment')
plt.subplot(2, 2, 1) # Figure two has subplots 2 raw, 2 columns, and this is plot 1
plt.gca().set_title('Gray')
plt.xticks([]), plt.yticks([]) # To hide axes
plt.imshow(img, cmap='gray')

plt.subplot(2, 2, 2)
plt.gca().set_title('Gaussian_Blur')
plt.xticks([]), plt.yticks([])
plt.imshow(img, cmap='gray')

plt.subplot(2, 2, 3)
plt.gca().set_title('Thresh')
plt.xticks([]), plt.yticks([])
plt.imshow(thresh, cmap='gray')

plt.subplot(2, 2, 4)
plt.gca().set_title('img_mop')
plt.xticks([]), plt.yticks([])
plt.imshow(img_mop, cmap='gray')


#----------------------------------------------------------------------------------------------------------------------#
# WTERSHED WITH SKIMAGE

distance = ndi.distance_transform_edt(img_mop) # Calculates distance of pixels from background

#Find peaks in an image as coordinate list or boolean mask.
img_mop = np.where(img_mop <= 0, img_mop, 1)
img_mop = img_mop.astype(int)
local_maxi = peak_local_max(distance, indices=False, min_distance=50, footprint=np.ones((3, 3)), labels=img_mop)
markers = ndi.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=img_mop)

plt.figure('Processing')
plt.subplot(2, 2, 1) # Figure two has subplots 2 raw, 2 columns, and this is plot 1
plt.gca().set_title('Distance trans')
plt.xticks([]), plt.yticks([]) # To hide axes
plt.imshow(distance, cmap='gray')

plt.subplot(2, 2, 2)
plt.gca().set_title('local_maxi')
plt.xticks([]), plt.yticks([])
plt.imshow(local_maxi, cmap='gray')

plt.subplot(2, 2, 3)
plt.gca().set_title('markers')
plt.xticks([]), plt.yticks([])
plt.imshow(markers, cmap='gray')

plt.figure('Watershed')
plt.gca().set_title('Watershed')
plt.xticks([]), plt.yticks([]) # To hide axes
plt.imshow(labels)

plt.show()

#----------------------------------------------------------------------------------------------------------------------#
# DATA ANALYSIS

# Regionprops: Measure properties of labeled image regions. It can give A LOT of properties, see info in:
# https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
props = regionprops(labels)

# Determine scale bar (largest object) and set the scale.
thr_size = 6000
for p in props:
    if p['area'] > thr_size:
        box = p['bbox']
        scale = box[3]-box[1]


# Remove smaller detected areas, and give area and diameter for each of the remaining particles.
for p in props:
    if p['equivalent_diameter'] > 15 and p['equivalent_diameter'] < 40:
        entry = [p['label'], p['area'], p['equivalent_diameter'], *p['centroid']]
        n = entry[0]
        y = entry[3]
        x = entry[4]-60 # so that number shows on the left of particle
        cv2.putText(img, str(n), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        print('Particle {} | Area (nm^2): {}; Equivalent diameter (nm): {}'.format(str(n),
                                            str(int(((entry[1]*40000)/(scale**2)))), str(int((entry[2])*200/scale))))

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()