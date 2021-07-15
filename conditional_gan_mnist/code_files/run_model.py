from params import config
from networks import Generator, Discriminator
import util, training

import os
import wandb
import torch
from torch import nn

wandb.init(project="mnist-test", config=config)

PATH = os.path.dirname(os.path.realpath(__file__))
mnist_shape = (1, 28, 28)
criterion = nn.BCEWithLogitsLoss()
train = config['train']
n_epochs = config['epochs']
n_classes = config['classes']
z_dim = config['z_dim']
batch_size = config['batch_size']
lr = config['learning_rate']
device = torch.device("cuda" if(torch.cuda.is_available()) else "cpu")


generator_input_dim, discriminator_im_chan = util.get_input_dimensions(z_dim, mnist_shape, n_classes)
gen = Generator(input_dim=generator_input_dim).to(device)
disc = Discriminator(im_chan=discriminator_im_chan).to(device)
gen = gen.apply(util.weights_init)
disc = disc.apply(util.weights_init)
testloader = util.load_data(path=PATH, train=False, batch_size=batch_size)

if train == True:
    training.train(PATH, gen, disc, mnist_shape, n_classes, criterion, n_epochs, z_dim, batch_size, lr, device)
else:
    util.test(PATH, testloader, gen, disc, criterion, z_dim, device, n_classes, mnist_shape)
