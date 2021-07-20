### Welcome to SliceGAN ###
####### Steve Kench #######

from slicecgan import *

## Make directory

Project_name = 'Alej_batch5'
Project_dir = 'trained_generators/NMC_Alej/'

## Data Processing
image_type = 'threephase' # threephase, twophase or colour
data_type = 'self' # png, jpg, tif, array, array2D
data_path = []
labels = []
# # # Scotts labels

wandb_name = Project_name
# for ca, ca_lab in zip(['000.10','100.00'], [0, 1]):
#     for cc, cc_lab  in zip(['000.10','100.00'], [0, 1]):
#         for por, por_lab in zip(['30','40','50'], [0, 0.5, 1]):
#             # for wt_lab, wt in enumerate(zip(['90','96']),1):
#             file = 'ds_wt0.92_ca{}_cc{}_case01_porosity0.{}_phases.npy'.format(ca, cc, por)
#             data_path.append('Examples/Scott_NMC/round1_2/'+ file) # path to training data.
#             labels.append([ca_lab, cc_lab, por_lab])

for wt, wt_lab in zip(['85', '90', '95'], [0, 0.5, 1]):
    # for psd, psd_lab in zip(['0', '0.5', '1'], [0, 0.5, 1]):
    for psd, psd_lab in zip(['1'], [1]):
        for comp, comp_lab in zip(['0','10','20'], [0, 0.5, 1]):
            # for wt_lab, wt in enumerate(zip(['90','96']),1):
            file = 'training_data/Alej_NMC/batch5/processed_1/wt{}_psd1frac{}_comp{}.tif'.format(wt, psd, comp)
            data_path.append(file) # path to training data.
            labels.append([wt_lab, psd_lab, comp_lab])

# # Alej labels
# for wt, NMC in zip(['94','95', '96'],[0, 0.5, 1]):
#     data_path.append('training_data/Alej_NMC/batch2/{}.npy'.format(lab))
#     labels.append([NMC])

isotropic = True
Training = 1 # Run with False to show an image during training
Project_path = mkdr(Project_name, Project_dir, Training)
print('Using project name {}'.format(Project_path))

# Network Architectures
imsize, nz,  channels, sf, lbls = 64, 32, 3, 1, len(labels[0]*2)
lays = 5
laysd = 5
dk, gk = [4]*laysd, [4]*lays                                    # kernal sizes
# gk[0]=8
ds, gs = [2]*laysd, [2]*lays                                    # strides
# gs[0] = 4
df, gf = [channels, 64, 128, 256, 512, 1], [nz, 512, 256, 128, 64, channels]  # filter sizes for hidden layers
dp, gp = [1, 1, 1, 1, 0], [2, 2, 2, 2, 3]

##Create Networks
netD, netG = slicecgan_rc_pc_nets(Project_path, Training, lbls, dk, ds, df,dp, gk ,gs, gf, gp)

if Training:
    data = conditional_trainer(Project_path, image_type, data_path, labels, netD, netG, isotropic, channels, imsize, nz, sf, wandb_name)

else:
    with torch.no_grad():
        imgs, raw, netG = test_img_cgan(Project_path, labels[1:3], image_type, netG(), nz,  lf=8, twoph=0)
        for im in imgs:
            for ph in [0, 1, 2]:
                print(len(im[im == ph]) / im.size)
