import numpy as np
from plotoptix import TkOptiX
from plotoptix.materials import make_material
from plotoptix.materials import m_flat
from plotoptix.utils import map_to_colors  # map variable to matplotlib color map
### Welcome to SliceGAN ###
####### Steve Kench #######
import tifffile
import colorcet as cc

import matplotlib.pyplot as plt

s = 1
a = 1
tort = True

print('start')
if tort:
    min_cut = 40
    min_val = 0
    max_cut = 100
    cmap = cc.cm.fire
    img = tifffile.imread('alej_cond_fm.tif')
else:
    min_cut = 0
    min_val = 20
    max_cut = 70
    cmap = cc.cm.CET_L16
    img = tifffile.imread('alej_tort_fm.tif')

crop = 10
img = img[crop:-crop, crop:-crop, crop:-crop]
inrt = 118
img[:inrt, :inrt, :inrt] = -1
n = img.shape[0]

flux = np.array(np.where(img[:,:,:,0] > min_cut)).T
c = img[:, :, :, 0].reshape(-1)
mask = c > min_cut
c[c>max_cut] = max_cut
c = c[mask]
c -= min_cut + min_val
c = c.astype(np.float)
c[c<0] = 0
c = c **0.95
c*=255/c.max()
parts = np.array(np.where(img[:,:,:,0] ==0)).T
# cr = img[:, :, :, 0].reshape(-1)
# mask = cr ==0
# cr = cr[mask]
#
# c_parts = np.zeros((cr.size, 3))
# c_parts[:, 0] = cr
# c_parts[:,-1] = cb

print('done')

optix = TkOptiX(start_now=False, devices=[0]) # no need to open the window yet
optix.set_param(min_accumulation_step=4,     # set more accumulation frames
                max_accumulation_frames=2000, # to get rid of the noise
                light_shading="Hard")        # use "Hard" light shading for the best caustics and "Soft" for fast convergence

optix.set_uint("path_seg_range", 15, 30)

alpha = np.full((1, 1, 4), 0.3, dtype=np.float32)
optix.set_texture_2d("mask", (255*alpha).astype(np.uint8))
m_diffuse_3 = make_material("TransparentDiffuse", color_tex="mask")
optix.setup_material("diffuse_1", m_diffuse_3)

if tort:
    optix.set_data("cubes_b", pos=parts, u=[s, 0, 0], v=[0, s, 0], w=[0, 0, s],
               geom="Parallelepipeds", # cubes, actually default geometry
               mat="diffuse",          # opaque, mat, default
               c = (0.2, 0.2, 0.2))
optix.set_data("cubes_flux", pos=flux, u=[s, 0, 0], v=[0, s, 0], w=[0, 0, s],
               geom="Parallelepipeds", # cubes, actually default geometry
               mat="diffuse",          # opaque, mat, default
               # c = map_to_colors(c/255, cc.cm.isoluminant_cm_70_c39))
               c = map_to_colors(c/255, cmap))
optix.setup_camera("cam1",cam_type="Pinhole", eye=[-3557.2212 , -730.4132, -1483.8723 ], target=[128, 128 , 128], up=[0,-1, 0], fov=5)
optix.set_background(10)
# optix.set_ambient(0)


optix.set_float("tonemap_exposure", 0.5)
optix.set_float("tonemap_gamma", 2.2)

optix.add_postproc("Gamma")      # apply gamma correction postprocessing stage, or
# optix.add_postproc("Denoiser")  # use AI denoiser (exposure and gamma are applied as well)

x = n/2
optix.setup_light("light1", pos=[x, x, -x*2], color=10*np.array([1.0, 1.0, 1.0]), radius=50)
optix.setup_light("light2", pos=[x, -x*2, x], color=10*np.array([1.0, 1.0, 1.0]), radius=50)
optix.setup_light("light3", pos=[-484.97705,   127,  127], color=15*np.array([1.0, 1.0, 1.0]), radius=100)
optix.start()
