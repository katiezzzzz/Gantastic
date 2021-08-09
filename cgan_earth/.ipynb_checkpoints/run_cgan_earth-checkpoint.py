from code_files import *
import numpy as np

PATH = os.path.dirname(os.path.realpath(__file__))
Project_name = 'earth_cylinder'
Project_dir = PATH + '/trained_generators/'
wandb_name = Project_name

# import training images and define labels
data_path = []
labels = []

for img_path, label in zip(['city1', 'desert1', 'forest1', 'sea1', 'snow1'], [0, 1, 2, 3, 4]):
    file = PATH + '/earth_screenshots/{}.jpg'.format(img_path)
    data_path.append(file) # path to training data.
    labels.append(label)

imgs = read_img(data_path)

# define hyperparameters and architecture
ngpu = 1
z_dim = 64
lr = 0.0001
Training = 1
n_classes = 5
batch_size = 20
num_epochs = 600
img_length = 128 # size of training image
proj_path = mkdr(Project_name, Project_dir, Training)
device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")

# create networks
netG, netD = cgan_earth_nets(proj_path)

# train
if Training:
    train(proj_path, netG, netD, imgs, labels, img_length, n_classes, num_epochs, z_dim, 
          batch_size, lr, device, wandb_name)
else:
    # test
    pass