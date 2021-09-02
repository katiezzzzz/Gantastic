from code_files import *
import numpy as np
import os

PATH = os.path.dirname(os.path.realpath(__file__))
Project_name = 'earth_cylinder_r_1'
Project_dir = PATH + '/trained_generators/'
wandb_name = Project_name

labels = [0, 1, 2, 3, 4]

# define hyperparameters and architecture
ngpu = 1
z_dim = 64
lr = 0.0001
Training = 0
n_classes = 5
batch_size = 10
im_channels = 3
num_epochs = 600
img_length = 128 # size of training image
proj_path = mkdr(Project_name, Project_dir, Training)
device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")

# create networks
netG, netD = cgan_earth_nets(proj_path, Training, z_dim+n_classes, im_channels+n_classes)

# animate
forest_lbl = [0]
city_lbl = [1]
desert_lbl = [2]
sea_lbl = [3]
snow_lbl = [4]
lf = 16
ratio = 2

# test1: forest, then transit to sea, then roll in sea
# the speed currently must start with the lowest possible speed to make sure the noise has right dimension
imgs1, noise, netG = roll_video(proj_path, sea_lbl, netG(z_dim+n_classes, img_length), n_classes, z_dim, lf=lf, device=device, ratio=ratio, n_clips=24*2, step_size=0.25)
imgs2, noise, netG = transit_video(sea_lbl, forest_lbl, n_classes, noise, netG, lf=lf, ratio=ratio, device=device, step_size=0.25, z_step_size=0.1, l_step_size=0.1, transit_mode='uniform')
imgs3, noise, netG = roll_video(proj_path, forest_lbl, netG, n_classes, z_dim, lf=lf, device=device, ratio=ratio, n_clips=24, step_size=0.25, original_noise=noise)
imgs4, noise, netG = roll_video(proj_path, forest_lbl, netG, n_classes, z_dim, lf=lf, device=device, ratio=ratio, n_clips=24, step_size=0.25, original_noise=noise, replace=True)

# concatenante the imgs together and make video
imgs = np.vstack((imgs1, imgs2))
imgs = np.vstack((imgs, imgs3))
animate(proj_path, imgs, fps=24)