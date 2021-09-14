from code_files.util import *
from torch import nn
import numpy as np
import torch
import time

def train(pth, gen, disc, imgs, labels, img_length, n_classes, num_epochs, z_dim, batch_size, lr, device, wandb_name):

    rt = 0
    lz = 6
    beta1 = 0.5
    beta2 = 0.999
    c_lambda = 10
    crit_iter = 5
    log_iter = 200
    n_channels = 3
    iters = 30000 // batch_size
    print(device, " will be used.\n")

    netG = gen(z_dim+n_classes, img_length).to(device)
    netD = disc(n_channels+n_classes).to(device)

    if rt:
        netG.load_state_dict(torch.load('cgan_earth/trained_generators/earth_cylinder_r_3/earth_cylinder_r_3_Gen.pt'))
        netD.load_state_dict(torch.load('cgan_earth/trained_generators/earth_cylinder_r_3/earth_cylinder_r_3_Disc.pt'))
    else:
        netG = netG.apply(param_init)
        netD = netD.apply(param_init)

    optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
    optD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))

    if wandb_name:
        wandb_init(wandb_name)
        wandb.watch(netD)
        wandb.watch(netG)
    print("Starting Training Loop...")

    start = time.time()
    for epoch in range(num_epochs):
        for i in range(iters):
            real_data, b_labels = batch(imgs, labels, batch_size, img_length, device)
            # normalise real_data
            real_data = torch.div(real_data, 255)
            one_hot_labels = gen_labels(b_labels, n_classes)[:, :, None, None]
            G_labels = one_hot_labels.repeat(1, 1, lz, lz).to(device)
            D_labels = one_hot_labels.repeat(1, 1, img_length, img_length).to(device)

            ### Discriminator
            # generate fake images
            noise = torch.randn(batch_size, z_dim, lz, lz, device=device)
            fake_data = netG(noise, G_labels).detach()
            # train discriminator
            start_disc = time.time()
            netD.zero_grad()
            # real_data.shape (batch_size, 3, img_length, img_length)
            # fake_data.shape (batch_size, 3, img_length, img_length)
            # D_labels.shape (batch_size, n_classes, img_length, img_length)
            disc_real = netD(real_data, D_labels).mean()
            disc_fake = netD(fake_data, D_labels).mean()
            # calculate gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data, batch_size,
                                                     img_length, device, c_lambda, n_channels,
                                                     D_labels)
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_cost.backward()
            optD.step()
            if wandb_name:
                wandb.log({'Gradient penalty': gradient_penalty.item()})
                wandb.log({'Wass': disc_real.item() - disc_fake.item()})
                wandb.log({'Discriminator real': disc_real.item()})
                wandb.log({'Discriminator fake': disc_fake.item()})
            
            ### Generator
            start_gen = time.time()
            if (i % crit_iter) == 0:
                netG.zero_grad()
                errG = 0
                noise = torch.randn(batch_size, z_dim, lz, lz, device=device)
                fake = netG(noise, G_labels)
                output = netD(fake, D_labels)
                errG -= output.mean()
                # Calculate gradients for G
                errG.backward()
                optG.step()
                
            # Output training stats & show imgs
            if i % log_iter == 0:
                torch.save(netG.state_dict(), pth + '_Gen.pt')
                torch.save(netD.state_dict(), pth + '_Disc.pt')
                noise = torch.randn(1, z_dim, lz+2, lz+2, device=device)
                netG.eval()
                test_labels = gen_labels(labels, n_classes)[:, :, None, None]
                for tst_lbl in test_labels:
                    lbl = tst_lbl.repeat(1, 1, lz+2, lz+2).to(device)
                    with torch.no_grad():
                        img = torch.multiply(netG(noise, lbl), 255)
                    img = img.cpu().detach().numpy()
                    img = np.moveaxis(img, 1, -1)
                    wandb.log({"fake" : wandb.Image(img)})
                netG.train()
                calc_eta(iters, time.time(), start, i, epoch, num_epochs)



    
