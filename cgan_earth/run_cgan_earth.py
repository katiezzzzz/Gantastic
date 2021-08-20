from code_files import *
import numpy as np

PATH = os.path.dirname(os.path.realpath(__file__))
Project_name = 'earth_cylinder'
Project_dir = PATH + '/trained_generators/'
wandb_name = Project_name

# import training images and define labels
data_path = []
labels = []

for img_path, label in zip(['snow1'], [0]):
    file = PATH + '/earth_screenshots/{}.jpg'.format(img_path)
    data_path.append(file) # path to training data
    labels.append(label)

imgs = read_img(data_path)

# define hyperparameters and architecture
ngpu = 1
z_dim = 64
lr = 0.0001
Training = 1
n_classes = 1
batch_size = 18
im_channels = 3
num_epochs = 600
img_length = 128 # size of training image
proj_path = mkdr(Project_name, Project_dir, Training)
device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")

# create networks
netG, netD = cgan_earth_nets(proj_path, Training, z_dim+n_classes, im_channels+n_classes)

# train
if Training:
    train(proj_path, netG, netD, imgs, labels, img_length, n_classes, num_epochs, z_dim, 
          batch_size, lr, device, wandb_name)
else:
    labels = [0]
    test(proj_path, labels, netG(z_dim+n_classes, img_length), n_classes, z_dim, lf=24, device=device, ratio=2)
