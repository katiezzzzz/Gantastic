import torch
import wandb
from torch import nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def get_input_dimensions(z_dim, mnist_shape, n_classes):
    '''
    Function for getting the size of the conditional input dimensions 
    from z_dim, the image shape, and number of classes.
    Returns: 
        generator_input_dim: the input dimensionality of the conditional generator, 
                          which takes the noise and class vectors
        discriminator_im_chan: the number of input channels to the discriminator
                            (e.g. C x 28 x 28 for MNIST)
    '''
    generator_input_dim = z_dim+n_classes
    discriminator_im_chan = mnist_shape[0]+n_classes
    return generator_input_dim, discriminator_im_chan

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

def get_noise(n_samples, input_dim, device='cpu'):
    return torch.randn(n_samples, input_dim, device=device)

def make_img_array(n_img, img_tensor):
    img_num = 0
    img_array = []
    for i in img_tensor:
        img_array += [wandb.Image(i)]
        img_num += 1
        if img_num >= n_img:
            break
    return img_array