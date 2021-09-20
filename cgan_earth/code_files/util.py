from numpy.core.numeric import outer
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from code_files.video import *
from moviepy.editor import *
from torch import autograd
from tqdm import tqdm
from torch import nn
import numpy as np
import subprocess
import warnings
import tifffile
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
                data = crop(imgs[i], img_length)[None, :, :, :]
            else:
                data = np.vstack((data, crop(imgs[i], img_length)[None, :, :, :]))
            batch_labels = np.append(batch_labels, int(labels[i]))
    n_excess = batch_size - n_crop*n_img
    # randomly select an image and crop n_excess number of times
    if n_excess > 0:
        warnings.warn('batch size is not a multiple of number of classes')
    # shuffle along dim 0
    shuffler = np.random.permutation(batch_size)
    data = data[shuffler]
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
    labels = torch.from_numpy(np.array(labels).astype(np.int64))
    return F.one_hot(labels, num_classes=n_classes)

def gen_intermediate_labels(label1, label2, val, n_classes, device):
    '''
    Generate one-hot label for an intermediate image
    Params:
        label1: integer representing the first label class
        label2: integer representing the second label class
        val: value of the first label class
        n_classes: total number of classes
    Return:
        tensor of shape (n_classes)
    '''
    one_hot = torch.zeros((n_classes)).to(device)
    one_hot[label1] = val
    one_hot[label2] = 1 - val
    return one_hot

def add_noise_dim(noise, new_noise, n_repeat, start_idx=0):
    for idx0 in range(noise.shape[0]):
        for idx1 in range(noise.shape[1]):
            for idx2 in range(noise.shape[2]):
                dim2 = noise[idx0, idx1, idx2]
                new_noise[idx0, idx1, idx2] = torch.cat((dim2, dim2[start_idx:n_repeat]), -1)
    return new_noise

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

def test(path, labels, netG, n_classes, z_dim=64, lf=4, device='cpu', ratio=2):
    try:
        netG.load_state_dict(torch.load(path + '_Gen.pt'))
    except:
        netG = nn.DataParallel(netG)
        netG.load_state_dict(torch.load(path + '_Gen.pt'))
    
    netG.to(device)
    names = ['forest', 'city', 'desert', 'sea', 'snow', 'star']
    tifs, raws = [], []
    # try to generate rectangular, instead of square images
    random = torch.randn(1, z_dim, lf, lf*ratio-2, device=device)
    noise = torch.zeros((1, z_dim, lf, lf*ratio)).to(device)
    for idx0 in range(random.shape[0]):
        for idx1 in range(random.shape[1]):
            for idx2 in range(random.shape[2]):
                dim2 = random[idx0, idx1, idx2]
                noise[idx0, idx1, idx2] = torch.cat((dim2, dim2[:2]), -1)
    netG.eval()
    test_labels = gen_labels(labels, n_classes)[:, :, None, None]
    for i in range(len(labels)):
        lbl = test_labels[i].repeat(1, 1, lf, lf*ratio).to(device)
        with torch.no_grad():
            img = netG(noise, lbl, Training=False, ratio=ratio).cuda()
            raws.append(img)
        print('Postprocessing')
        tif = torch.multiply(img, 255).cpu().detach().numpy()
        try:
            name = names[i]
        except:
            name = 'none'
        tifffile.imwrite(path + '_' + name + '.tif', tif)
        tifs.append(tif)
    return tifs, netG

def roll_video(path, label, netG, n_classes, z_dim=64, lf=4, device='cpu', ratio=2, n_clips=30, step_size=1, original_noise=None):
    '''
    Given an integer label, generate an array of images that roll through z and can be used to make a video
    Params:
        path: string, directory of where the generator is stored
        label: list containing one integer
        netG: torch.nn.Module generator class
        n_classes: integer
        z_dim: integer
        lf: integer, size of input seed
        device: string
        ratio: integer
        n_clips: integer
        step_size: float between 0 and lf*ratio, the size of step through z space
        original_noise: tensor, (1, z_dim, lf, lf*2), random noise vector
    Return:
        imgs: array containing generated images and can be used to make a video
        original_noise: tensor, (1, z_dim, lf, lf*2), generated random noise
        netG: trained generator class
    '''
    try:
        netG.load_state_dict(torch.load(path + '_Gen.pt'))
    except:
        netG = nn.DataParallel(netG)
        netG.load_state_dict(torch.load(path + '_Gen.pt'))
    
    netG.to(device)
    if step_size >= 1:
        max_len = lf*ratio
    else:
        max_len = lf*ratio+1
    # try to generate rectangular, instead of square images
    random = torch.randn(1, z_dim, lf, lf*ratio-2, device=device)
    #random[0][0] = torch.arange(lf*(lf*ratio-2)).view(lf, lf*ratio-2)
    if original_noise == None:
        original_noise = torch.zeros((1, z_dim, lf, max_len)).to(device)
        if step_size >= 1:
            original_noise = add_noise_dim(random, original_noise, 2)
        else:
            # generate half a z step more data in final dimension
            original_noise = add_noise_dim(random, original_noise, 3)
    else:
        max_len = original_noise.shape[-1]
        
    netG.eval()
    test_label = gen_labels(label, n_classes)[:, :, None, None]
    imgs = np.array([])
    noise = original_noise
    step = 0.0
    lbl = test_label.repeat(1, 1, lf, max_len).to(device)
    label = label[0]
    if step_size >= 1:
        num_img = 1
    else:
        num_img = int(1/step_size)

    with torch.no_grad():
        #print(noise[0][0])
        img = netG(noise, lbl, Training=False, ratio=ratio).cuda()
        img = torch.multiply(img, 255).cpu().detach().numpy()
        for _ in tqdm(range(n_clips)):
            for i in range(num_img):
                if step_size < 1:
                    # one z represents 32 pixels in the -1 dimension
                    step_idx = int(i * step_size * 32)
                    out = img[:, :, :, step_idx:img.shape[-1]-(32-step_idx)]
                else:
                    out = img[:, :, :, :img.shape[-1]-32]
                out = np.moveaxis(out, 1, -1)
                if imgs.shape[0] == 0:
                    imgs = out
                else:
                    imgs = np.vstack((imgs, out))
                step += step_size
                # avoid step growing too large
            max_step = lf*ratio-2
            if max_len == lf*ratio:
                IntStep = True
            else:
                IntStep = False
            if step > max_step:
                step -= max_step
            img = roll_pixels(img, int(step_size*32), int(max_step*32), IntStep)
            noise = roll_noise(original_noise, step, max_step, IntStep)
    return imgs, noise, netG

def transit_video(label1, label2, n_classes, original_noise, netG, lf=4, ratio=2, device='cpu', step_size=1, z_step_size=1, l_step_size=0.1, transit_mode='uniform'):
    '''
    Given all the labels and positions of original and target labels in list of labels, generate an image array for video transition
    Params:
        label1: list containing an integer, original label
        label2: list containing an integer, target label
        n_classes: integer
        original_noise: tensor of dimension (1, z_dim, lf, lf*ratio)
        netG: trained generator class
        lf: size of input seed
        ratio: integer
        device: string
        step_size: integer with value between 1 and lf*ratio
        z_step_size: float between 0 and 1
        l_step_size: float between 0 and 1
        transit_mode: uniform or scroll or circular
    Return:
        imgs: image array that contains the transition, can be used to make a video
        original_noise: tensor of dimension (1, z_dim, lf, lf*ratio)
        netG: trained generator class
    '''
    max_len = original_noise.shape[-1]

    imgs = np.array([])
    noise = original_noise
    step = 0
    prev_label = gen_labels(label1, n_classes)[:, :, None, None]
    lbl = prev_label.repeat(1, 1, lf, max_len).to(device)
    if transit_mode == 'uniform':
        n_clips = int(1 // l_step_size)
    else:
        n_clips = int((1 // z_step_size) + (1 // l_step_size) + 2)
        l_step = 0
        z_step = 0
        l_done_step = 0
        z_done_step = 0
    label1 = label1[0]
    label2 = label2[0]
    if step_size >= 1:
        num_img = 1
    else:
        num_img = int(1/step_size)
    for _ in tqdm(range(n_clips)):
        with torch.no_grad():
            img = netG(noise, lbl, Training=False, ratio=ratio).cuda()
            img = torch.multiply(img, 255).cpu().detach().numpy()
            for i in range(num_img):
                if step_size < 1:
                    # one z represents 32 pixels in the -1 dimension
                    step_idx = int(i * step_size * 32)
                    out = img[:, :, :, step_idx:img.shape[-1]-(32-step_idx)]
                else:
                    out = img[:, :, :, :img.shape[-1]-32]
                out = np.moveaxis(out, 1, -1)
                if imgs.shape[0] == 0:
                    imgs = out
                else:
                    imgs = np.vstack((imgs, out))
                step += step_size
            if transit_mode == 'uniform':
                lbl = uniform_transit(label1, label2, lbl, l_step_size)
            # avoid step growing too large
            elif transit_mode == 'scroll':
                lbl, l_step, z_step, l_done_step, z_done_step = scroll_transit(label1, label2, lbl, 
                z_step_size, l_step_size, max_len, l_step, z_step, l_done_step, z_done_step)
            elif transit_mode == 'circular':
                lbl, l_step, z_step, l_done_step, z_done_step = circular_transit(label1, label2, lbl,
                z_step_size, l_step_size, lf, max_len, l_step, z_step, l_done_step, z_done_step)
            max_step = lf*ratio-2
            if max_len == lf*ratio:
                IntStep = True
            else:
                IntStep = False
            if step > max_step:
                step -= max_step
            noise = roll_noise(original_noise, step, max_step, IntStep)
    return imgs, noise, netG

def effects(label1, label2, n_classes, original_noise, netG, lf=4, ratio=2, device='cpu', step_size=1, z_step_num=3, l_step_size=0.1, z_max_num=5, effect='circles', n_circles=3):
    '''
    Given all the labels and positions of original and target labels in list of labels, generate an image array for video transition
    Params:
        label1: list containing an integer, original label
        label2: list containing an integer, target label
        n_classes: integer
        original_noise: tensor of dimension (1, z_dim, lf, lf*ratio)
        netG: trained generator class
        lf: size of input seed
        ratio: integer
        device: string
        step_size: integer with value between 1 and lf*ratio
        z_step_num: integer between 1 and z_max_num
        l_step_size: float between 0 and 1
        z_max_num: integer between 2+z_step_num and lf
        effect: type of effects, only 'circles' at the moment
    Return:
        imgs: image array that contains the transition, can be used to make a video
        original_noise: tensor of dimension (1, z_dim, lf, lf*ratio)
        netG: trained generator class
    '''
    max_width = original_noise.shape[-2]
    max_len = original_noise.shape[-1]

    imgs = np.array([])
    noise = original_noise
    step = 0
    prev_label = gen_labels(label1, n_classes)[:, :, None, None]
    lbl = prev_label.repeat(1, 1, lf, max_len).to(device)
    if effect == 'circles':
        n_clips = int((z_max_num // z_step_num) + (1 // l_step_size) + 2)
        l_step = 0
        z_step = 0
        l_done_step = 0
        z_done_step = 0
        centres = generate_centres(n_circles, max_width, max_len, z_max_num//2)
    label1 = label1[0]
    label2 = label2[0]
    if step_size >= 1:
        num_img = 1
    else:
        num_img = int(1/step_size)
    for _ in tqdm(range(n_clips)):
        with torch.no_grad():
            img = netG(noise, lbl, Training=False, ratio=ratio).cuda()
            img = torch.multiply(img, 255).cpu().detach().numpy()
            for i in range(num_img):
                if step_size < 1:
                    # one z represents 32 pixels in the -1 dimension
                    step_idx = int(i * step_size * 32)
                    out = img[:, :, :, step_idx:img.shape[-1]-(32-step_idx)]
                else:
                    out = img[:, :, :, :img.shape[-1]-32]
                out = np.moveaxis(out, 1, -1)
                if imgs.shape[0] == 0:
                    imgs = out
                else:
                    imgs = np.vstack((imgs, out))
                step += step_size
            if effect == 'circles':
                lbl, l_step, z_step, l_done_step, z_done_step = circular_effects(label1, label2, lbl, z_step_num, 
                l_step_size, lf, max_len, l_step, z_step, l_done_step, z_done_step, z_max_num, centres)
            max_step = lf*ratio-2
            if max_len == lf*ratio:
                IntStep = True
            else:
                IntStep = False
            if step > max_step:
                step -= max_step
            noise = roll_noise(original_noise, step, max_step, IntStep)
    return imgs, noise, netG

def change_noise(label, original_noise, netG, n_classes, z_dim=64, lf=4, device='cpu', ratio=2, n_clips=30, step_size=1, value=0.01, method='add'):
    max_len = original_noise.shape[-1]

    test_label = gen_labels(label, n_classes)[:, :, None, None]

    lbl = test_label.repeat(1, 1, lf, max_len).to(device)
    imgs = np.array([])
    noise = original_noise
    step = 0.0
    if step_size >= 1:
        num_img = 1
    else:
        num_img = int(1/step_size)
    for _ in tqdm(range(n_clips)):
        with torch.no_grad():
            img = netG(noise, lbl, Training=False, ratio=ratio).cuda()
            img = torch.multiply(img, 255).cpu().detach().numpy()
            for i in range(num_img):
                if step_size < 1:
                    # one z represents 32 pixels in the -1 dimension
                    step_idx = int(i * step_size * 32)
                    out = img[:, :, :, step_idx:img.shape[-1]-(32-step_idx)]
                else:
                    out = img[:, :, :, :img.shape[-1]-32]
                out = np.moveaxis(out, 1, -1)
                if imgs.shape[0] == 0:
                    imgs = out
                else:
                    imgs = np.vstack((imgs, out))
                step += step_size
            max_step = lf*ratio-2
            if max_len == lf*ratio:
                IntStep = True
            else:
                IntStep = False
            if step_size < 1:
                step_idx = 1
            else:
                step_idx = step_size
            if step > max_step:
                step -= max_step
            noise = roll_noise(noise, step_idx, max_step, IntStep)
            if method == 'sub':
                noise = torch.sub(noise, value)
            elif method == 'add':
                noise = torch.add(noise, value)
    return imgs, noise, netG

def animate(path, imgs, fps=24):
    clip = ImageSequenceClip(list(imgs),fps=fps)
    clip.write_gif(path + '_demo1.gif')
    clip.close()
    return clip

def calc_eta(steps, time, start, i, epoch, num_epochs):
    elap = time - start
    progress = epoch * steps + i + 1
    rem = num_epochs * steps - progress
    ETA = rem / progress * elap
    hrs = int(ETA / 3600)
    mins = int((ETA / 3600 % 1) * 60)
    print('[%d/%d][%d/%d]\tETA: %d hrs %d mins'
          % (epoch, num_epochs, i, steps,
             hrs, mins))

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