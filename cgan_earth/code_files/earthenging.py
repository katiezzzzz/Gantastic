'''
https://developers.google.com/earth-engine/guides 
'''

# Authenticate to use Google Earth Engine API
import ee
ee.Authenticate()

from pathlib import Path
from FireHR.data import *

# Bounding box coordinates
left   = 35.782
right  = 36.819
bottom = 42.388
top    = 43.410

path_save   = Path('data')
products    = ["COPERNICUS/S2"]  # Product id in google earth engine
bands       = ['B4', 'B3', 'B2'] # Red, Green, Blue

R = RegionST(name         = 'TeslaGigaBerlin', 
             bbox         = [left,bottom,right,top], 
             pixel_size   = 10.0,
             time_start   = '2021-03-01', 
             time_end     = '2021-04-25')

# Download time series
# download_data_ts(R, products, bands, path_save)

time_window = R.times[0], R.times[-1]

# Download median composite of the 3 least cloudy images within the time_window
download_data(R, time_window, products, bands, path_save, use_least_cloudy=3)

#download_data_ts(R, products, bands, path_save, show_progress=True)

import numpy as np
import matplotlib.pyplot as plt
from banet.data import open_tif

brightness = 3
im = np.concatenate([open_tif(f'data/download.{b}.tif').read() for b in bands])
im = im.transpose(1,2,0).astype(np.float32)/10000
plt.imshow(brightness*im)
plt.show()