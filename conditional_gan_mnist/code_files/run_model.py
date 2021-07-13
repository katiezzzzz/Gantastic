from params import config
from networks import Generator, Discriminator
import util, training

from torch import nn
import wandb

wandb.init(project="mnist-test", config=config)

mnist_shape = (1, 28, 28)
criterion = nn.BCEWithLogitsLoss()
n_epochs = config['epochs']
n_classes = config['classes']
z_dim = config['z_dim']
batch_size = config['batch_size']
lr = config['learning_rate']
device = config['device']


generator_input_dim, discriminator_im_chan = util.get_input_dimensions(z_dim, mnist_shape, n_classes)
gen = Generator(input_dim=generator_input_dim).to(device)
disc = Discriminator(im_chan=discriminator_im_chan).to(device)
gen = gen.apply(util.weights_init)
disc = disc.apply(util.weights_init)

training.train(gen, disc, mnist_shape, n_classes, criterion, n_epochs, z_dim, batch_size, lr, device)

