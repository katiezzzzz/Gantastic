### Welcome to SliceGAN ###
####### Steve Kench #######

from slicecgan import *
import numpy as np

## Make directory


PATH = os.path.dirname(os.path.realpath(__file__))
Project_name = 'cgan_microstructure_24'
Project_dir = PATH+'/trained_generators/microstructure/'

## Data Processing
image_type = 'twophase' # threephase, twophase or colour
data_type = 'self' # png, jpg, tif, array, array2D
data_path = []
labels = []
# # # Scotts labels

wandb_name = Project_name

for r, r_lab in zip(['6', '_mix', '10'], [0, 0.5, 1]):
    file = PATH+'/training_data/r{}.tiff'.format(r)
    data_path.append(file) # path to training data.
    labels.append([r_lab])

isotropic = True
Training = 0 # Run with False to show an image during training
Project_path = mkdr(Project_name, Project_dir, Training)
print('Using project name {}'.format(Project_path))

# Network Architectures
imsize, nz,  channels, sf, lbls = 64, 32, 2, 1, len(labels[0]*2)
lays = 5
laysd = 5
dk, gk = [4]*laysd, [4]*lays                                    # kernal sizes
# gk[0]=8
ds, gs = [2]*laysd, [2]*lays                                    # strides
# gs[0] = 4
df, gf = [channels, 64, 128, 256, 512, 1], [nz, 512, 256, 128, 64, channels]  # filter sizes for hidden layers
dp, gp = [1, 1, 1, 1, 0], [2, 2, 2, 2, 3]

##Create Networks
netD, netG = slicecgan_rc_nets(Project_path, Training, lbls, dk, ds, df,dp, gk ,gs, gf, gp)

if Training:
    data = conditional_trainer(Project_path, image_type, data_path, labels, netD, netG, isotropic, channels, imsize, nz, sf, wandb_name)

else:
    numbers = np.arange(0,1.1,0.1)
    labels = []
    for n in numbers:
        labels.append([n])
    with torch.no_grad():
        imgs, raw, netG = test_2d_cgan(Project_path, labels, image_type, netG(), nz,  lf=8)
        for im in imgs:
            for ph in [0, 1]:
                print(len(im[im == ph]) / im.size)
