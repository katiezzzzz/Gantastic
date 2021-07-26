import os
from torch import nn
import torch
from torch import autograd
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import wandb
from dotenv import load_dotenv
import subprocess
import shutil

## Training Utils
def mkdr(proj,proj_dir,Training, n=0):

    if Training:
        proj = proj + '_' + str(n)
        pth = proj_dir + proj
        try:
            os.mkdir(pth)
            return pth + '/' + proj
        except:
            pth = mkdr(proj[:-len('_' + str(n))], proj_dir, Training, n+1)
            return pth
    else:
        pth = proj_dir + proj
        return pth + '/' + proj


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def pre_proc(paths, sf):
    img_list = []
    for img in paths:
        img = tifffile.imread(img) # [30:-30]
        img = torch.tensor(img[::sf, ::sf])
        if len(img.shape) > 2:
            img = img[:, :, 0]
        h ,w = img.shape
        print(img.shape)
        phases = np.unique(img)
        # returns array([0,1])
        print(phases)
        oh_img = torch.zeros([len(phases), h, w])
        for ch, ph in enumerate(phases):
            oh_img[ch][img==ph] = 1
        img_list.append(oh_img)
    return img_list

def batch(imgs, lbls, l, bs, device):
    nlabs = len(lbls[0])
    data = np.empty([bs, 2, l, l])
    labelset = np.zeros([bs, nlabs * 2, 1, 1])
    p = 0
    nimgs = len(imgs)
    for img,lbl in zip(imgs, lbls):
        x_max, y_max = img.shape[1:]
        f = [1,2]
        np.random.shuffle(f)
        img.permute(0, f[0], f[1])
        if bs < nimgs:
            print('ERROR batch size smaller than n imgs')
            raise ValueError
        for i in range((bs//nimgs)):
            for j,lb in enumerate(lbl):
                labelset[p, j] = lb
                labelset[p, j+nlabs] = 1 - lb
            x = np.random.randint(1, x_max - l - 1)
            y = np.random.randint(1, y_max - l - 1)
            data[p] = img[:, x:x + l, y:y + l]
            p += 1
    return torch.FloatTensor(data).to(device), torch.FloatTensor(labelset).to(device)

def batchvw(imgs, lbls, l, bs, device):
    nlabs = len(lbls[0])
    data = np.empty([bs, 3, l, l])
    labelset = np.zeros([bs, nlabs * 2, 1, 1, 1])
    p = 0
    nimgs = len(imgs)
    for img, lbl in zip(imgs, lbls):
        x_max, y_max, z_max = img.shape[1:]
        if bs < nimgs:
            print('ERROR batch size smaller than n imgs')
            raise ValueError
        for i in range((bs//nimgs)):
            for j,lb in enumerate(lbl):
                labelset[p, j] = lb
                labelset[p, j+nlabs] = 1 - lb
            x = np.random.randint(x_max)
            y = np.random.randint(y_max - l)
            z = np.random.randint(z_max - l)
            data[p] = img[:, x, y:y + l, z:z + l]
            p += 1
    return torch.FloatTensor(data).to(device), torch.FloatTensor(labelset).to(device)

def cond_calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, gp_lambda, nc, labs):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, nc, l, l)
    alpha = alpha.to(device)

    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates, labs)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


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

## Plotting Utils
def post_proc(img, imtype):
    try:
        img = img.detach().cpu()
    except:
        pass
    if imtype == 'colour':
        return np.int_(255*(np.swapaxes(img[0], 0, -1)))
    if imtype == 'twophase':
        sqrs = np.zeros_like(img)
        sqrs = sqrs[:,0]
        p1 = np.array(img[:, 0])
        p2 = np.array(img[:, 1])
        sqrs[(p1 < p2)] = 1  # background, yellow
        return sqrs
    if imtype == 'threephase':
        sqrs = np.zeros_like(img)[:,0]
        p1 = np.array(img[:, 0])
        p2 = np.array(img[:, 1])
        p3 = np.array(img[:, 2])
        sqrs[(p1 > p2) & (p1 > p3)] = 0  # background, yellow
        sqrs[(p2 > p1) & (p2 > p3)] = 1  # spheres, green
        sqrs[(p3 > p2) & (p3 > p1)] = 2  # binder, purple
        return sqrs
    if imtype == 'grayscale':
        return 255*img[:][0]

def post_proc_2d(img, imtype):
    img = img.cpu().detach().numpy()[0]
    img = np.argmax(img, axis=0)
    return img

def test_plotter(sqrs, slcs, imtype, pth, wandb_flag):
    sqrs = post_proc(sqrs,imtype)[0]
    if wandb_flag:
        images = []
        for j in range(slcs):
            if imtype=='colour':
                images.append(sqrs[j, :, :, :])
                images.append(sqrs[:, j, :, :])
                images.append(sqrs[:, :, j, :])
            else:
                rescale = 255/sqrs.max()
                images.append(sqrs[j, :, :]*rescale)
                images.append(sqrs[:, j, :]*rescale)
                images.append(sqrs[:, :, j]*rescale)
        wandb.log({"examples" : [wandb.Image(i) for i in images]})
    else:
        fig, axs = plt.subplots(slcs, 3)
        if imtype == 'colour':
            for j in range(slcs):
                axs[j, 0].imshow(sqrs[j, :, :, :], vmin = 0, vmax = 255)
                axs[j, 1].imshow(sqrs[:, j, :, :],  vmin = 0, vmax = 255)
                axs[j, 2].imshow(sqrs[:, :, j, :],  vmin = 0, vmax = 255)
        elif imtype == 'grayscale':
            for j in range(slcs):
                axs[j, 0].imshow(sqrs[j, :, :], cmap = 'gray')
                axs[j, 1].imshow(sqrs[:, j, :], cmap = 'gray')
                axs[j, 2].imshow(sqrs[:, :, j], cmap = 'gray')
        else:
            for j in range(slcs):
                axs[j, 0].imshow(sqrs[j, :, :])
                axs[j, 1].imshow(sqrs[:, j, :])
                axs[j, 2].imshow(sqrs[:, :, j])
        plt.savefig(pth + '_slices.png')
        plt.close()

def graph_plot(data,labels,pth,name):
    for datum,lbl in zip(data,labels):
        plt.plot(datum, label = lbl)
    plt.legend()
    plt.savefig(pth + '_' + name)
    plt.close()


def test_img(pth, imtype, netG, nz = 64, lf = 4):
    netG.load_state_dict(torch.load(pth + '_Gen.pt'))
    netG.eval()
    noise = torch.randn(1, nz, lf, lf, lf)
    raw = netG(noise)
    print('Postprocessing')
    gb = post_proc(raw,imtype)
    tif = np.int_(gb)
    tifffile.imwrite(pth + '.tif', tif)

    return tif,raw, netG

def test_img_cgan(pth, label_list, imtype, netG, nz = 64, lf = 4, twoph = True):
    device = torch.device("cuda:0")
    try:
        netG.load_state_dict(torch.load(pth + '_Gen.pt'))
    except:
        netG = nn.DataParallel(netG)
        netG.load_state_dict(torch.load(pth + '_Gen.pt'))

    netG.to(device)
    tifs, raws = [], []
    noise = torch.randn(1, nz, lf, lf, lf, device=device)
    netG.eval()
    for lbls in label_list:
        fake_labels = torch.ones([1, len(lbls) * 2, 1, 1, 1], device=device)
        for ch, lbl in enumerate(lbls):
            fake_labels[:,ch] = lbl
            fake_labels[:, ch+len(lbls)] = 1 - lbl
        fake_labels = fake_labels.repeat(1, 1, lf,lf,lf)
        print(fake_labels[0,:,0,0,0])
        with torch.no_grad():
            raw = netG(noise, fake_labels)
        print('Postprocessing')
        gb = post_proc(raw,imtype)
        if twoph:
            gb[gb==0] = 3
            gb[gb!=3] = 0
        tif = np.int_(gb)
        tifffile.imwrite(pth + str(lbls)+ '.tif', tif)
        tifs.append(tif)
        raws.append(raw.cpu())
    return tifs, raws, netG

def test_2d_cgan(pth, label_list, imtype, netG, nz = 64, lf = 4):
    device = torch.device("cuda:0")
    try:
        netG.load_state_dict(torch.load(pth + '_Gen.pt'))
    except:
        netG = nn.DataParallel(netG)
        netG.load_state_dict(torch.load(pth + '_Gen.pt'))
    
    netG.to(device)
    tifs, raws = [], []
    noise = torch.randn(1, nz, lf, lf, device=device)
    netG.eval()
    for lbls in label_list:
        fake_labels = torch.ones([1, len(lbls) * 2, 1, 1], device=device)
        for ch, lbl in enumerate(lbls):
            fake_labels[:, ch] = lbl
            fake_labels[:, ch+len(lbls)] = 1 - lbl
        fake_labels = fake_labels.repeat(1, 1, lf, lf)
        print(fake_labels[0, :, 0, 0])
        with torch.no_grad():
            raw = netG(noise, fake_labels)
        print('Postprocessing')
        gb = post_proc_2d(raw,imtype)
        tif = np.int_(gb)
        print(tif.shape)
        tifffile.imwrite(pth + str(lbls)+ '.tif', tif)
        tifs.append(tif)
        raws.append(raw.cpu())
    return tifs, raws, netG    


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

def wandb_save_models(pth, fn):
    shutil.copy(pth+fn, os.path.join(wandb.run.dir, fn))
    wandb.save(fn)
