import torch.nn.functional as F
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from torch import autograd
from torch import nn
import numpy as np
import subprocess
import torch
import wandb
import os

PATH = os.path.dirname(os.path.realpath(__file__))

def mkdr(proj, proj_dir, Training, n=0):
    '''
    Make project directories
    Params:
        proj: string, project name
        proj_dir: string, project directory
        Training: bool
    Return:
        string of where the project information will be stored
    '''
    if Training:
        proj = proj + '_' + str(n)
        pth = proj_dir + proj
        try:
            os.makedirs(pth)
            return pth + '/' + proj
        except:
            pth = mkdr(proj[:-len('_' + str(n))], proj_dir, Training, n+1)
            return pth
    else:
        pth = proj_dir + proj
        return pth + '/' + proj

def crop(img, l):
    '''
    Crop a given image
    Params:
        img: dimension (channels, x, y)
        label: label of the input image, integer
        l: length of cropped image
    Return:
        array of cropped image
    '''
    x_max, y_max = img.shape[1:]
    x = np.random.randint(0, x_max-l+1)
    y = np.random.randint(0, y_max-l+1)
    return img[:, x:x+l, y:y+l]

def read_img(img_pth):
    imgs = np.array([])
    for i in range(len(img_pth)):
        im = plt.imread(img_pth[i])
        # make sure the img_dim is (channels, width, length)
        im = np.moveaxis(im, -1, 0)
        if i == 0:
            imgs = im
        else:
            imgs = np.vstack((imgs,im))
    return imgs.reshape(len(img_pth), im.shape[0], im.shape[1], im.shape[2])

def batch(imgs, labels, batch_size, img_length, device):
    '''
    Take training images and randomly crop into small images of size (img_length, img_length),
    then put into a batch
    Params:
        imgs: image array of dimension (n_img, channels, 2679, 4800)
        labels: label array of dimension (n_img)
        batch_size: integer
        img_length: length of cropped training image
        device: cuda or cpu
    Return:
        batch of training images of dimension (batch_size, channels, img_length, img_length)
        labels of dimension (batch_size)
    '''
    n_img = imgs.shape[0]
    # how many cropped image each training image should give
    n_crop = batch_size // n_img
    data = np.array([])
    batch_labels = np.array([])
    for i in range(n_img):
        for _ in range(n_crop):
            if len(data) == 0:
                data = crop(imgs[i], img_length)
            else:
                data = np.vstack((data, crop(imgs[i], img_length)))
            batch_labels = np.append(batch_labels, int(labels[i]))
    n_excess = batch_size - n_crop*n_img
    # randomly select an image and crop n_excess number of times
    if n_excess > 0:
        random_i = np.random.randint(0, n_img)
        random_img = imgs[random_i]
        for _ in range(n_excess):
            out = crop(random_img, img_length)
            data = np.vstack((data, crop(random_img, img_length)))
            batch_labels = np.append(batch_labels, int(random_i))
    # shuffle along dim 0
    shuffler = np.random.permutation(batch_size)
    data = data.reshape(batch_size, imgs.shape[1], img_length, img_length)[shuffler]
    batch_labels = batch_labels[shuffler]
    return torch.from_numpy(data).to(device), batch_labels

def gen_labels(labels, n_classes):
    '''
    Generate one-hot labels for dataset
    Params:
        labels: array of labels, (*)
        n_classes: number of classes, integer
    Return:
        tensor of one hot labels, (*, n_classes)
    '''
    # convert labels to tensors
    labels = torch.from_numpy(np.array(labels).astype(int))
    return F.one_hot(labels, num_classes=n_classes)

def param_init(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        torch.nn.init.normal_(layer.weight, 0.0, 0.02)
    if isinstance(layer, nn.BatchNorm2d):
        torch.nn.init.normal_(layer.weight, 0.0, 0.02)
        torch.nn.init.constant_(layer.bias, 0)

def calc_gradient_penalty(netD, real_data, fake_data, batch_size, img_length, device, gp_lambda, n_channels, lables):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, n_channels, img_length, img_length)
    alpha = alpha.to(device)

    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates, lables)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty

def wandb_init(name):
    load_dotenv(os.path.join(os.path.dirname(__file__),'.env'))
    API_KEY = os.getenv('WANDB_API_KEY')
    print("Logging into W and B using API key {}".format(API_KEY))
    process = subprocess.run(["wandb", "login", API_KEY])
    print("stderr:", process.stderr)

    ENTITY = os.getenv('WANDB_ENTITY')
    PROJECT = os.getenv('WANDB_PROJECT')
    print('initing')
    wandb.init(entity=ENTITY,name=name, project=PROJECT)

    wandb_config = {
        'active': True,
        'api_key': API_KEY,
        'entity': ENTITY,
        'project': PROJECT,
        'watch_called': False,
        'no_cuda': False,
        'seed': 42,
        'log_interval': 100,
    }
    wandb.watch_called = wandb_config['watch_called']
    wandb.config.no_cuda = wandb_config['no_cuda']
    wandb.config.seed = wandb_config['seed']
    wandb.config.log_interval = wandb_config['log_interval']