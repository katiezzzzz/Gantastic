import torch.nn.functional as F
import torch.nn as nn
import pickle
import torch

def cgan_earth_nets(path):

    class Generator(nn.Module):
        '''
        Generator Class
        Values:
            z_dim: the dimension of the noise vector, a scalar
            im_chan: the number of channels in the images
            hidden_dim: the inner dimension, a scalar
        '''
        def __init__(self, g_dim, im_chan=3, hidden_dim=64):
            super(Generator, self).__init__()
            self.g_dim = g_dim
            self.final_conv = nn.Conv2d(hidden_dim, im_chan, 3, 1, 0)
            # Build the neural network
            self.gen = nn.Sequential(
                self.make_gen_block(g_dim, hidden_dim * 4),
                self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
                self.make_gen_block(hidden_dim * 2, hidden_dim),
            )

        def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2):
            '''
            Function to return a sequence of operations corresponding to a generator block of DCGAN;
            a transposed convolution, a batchnorm (except in the final layer), and an activation.
            Parameters:
                input_channels: how many channels the input feature representation has
                output_channels: how many channels the output feature representation should have
                kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
                stride: the stride of the convolution
            '''
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, noise, labels):
            '''
            Function for completing a forward pass of the generator: Given a noise tensor,
            returns generated images.
            Parameters:
                noise: a noise tensor with dimensions (n_samples, z_dim)
            '''
            x = torch.cat((noise.float(), labels.float()), 1)
            x = self.gen(x)
            up = nn.Upsample(size = 128+2)
            return torch.sigmoid(self.final_conv(up(x)))


    class Critic(nn.Module):
        '''
        Critic Class
        Values:
            im_chan: the number of channels in the images, fitted for the dataset used
            hidden_dim: the inner dimension, a scalar
        '''
        def __init__(self, d_dim, hidden_dim=64):
            super(Critic, self).__init__()
            self.crit = nn.Sequential(
                self.make_crit_block(d_dim, hidden_dim),
                self.make_crit_block(hidden_dim, hidden_dim * 2),
                self.make_crit_block(hidden_dim * 2, 1, final_layer=True),
            )

        def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
            '''
            Function to return a sequence of operations corresponding to a critic block of DCGAN;
            a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
            Parameters:
                input_channels: how many channels the input feature representation has
                output_channels: how many channels the output feature representation should have
                kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
                stride: the stride of the convolution
                final_layer: a boolean, true if it is the final layer and false otherwise 
                        (affects activation and batchnorm)
            '''
            if not final_layer:
                return nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                    nn.BatchNorm2d(output_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            else:
                return nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                )

        def forward(self, image, labels):
            '''
            Function for completing a forward pass of the critic: Given an image tensor, 
            returns a 1-dimension tensor representing fake/real.
            Parameters:
                image: a flattened image tensor with dimension (im_chan)
            '''
            x = torch.cat((image.float(), labels.float()), 1)
            crit_pred = self.crit(x)
            return crit_pred.view(len(crit_pred), -1)

    return Generator, Critic