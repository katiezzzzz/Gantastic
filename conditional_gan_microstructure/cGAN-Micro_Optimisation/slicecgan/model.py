from slicecgan.util import *
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import time
import tifffile
import torch
import torch.nn as nn


def conditional_trainer(pth, imtype, real_data, labels, Disc, Gen, isotropic, nc, l, nz, sf, wandb_name):
    print('Loading Dataset...')
    Training = 0
    ## Constants for NNs
    ngpu = 1
    nlabels = len(labels[0])
    D_batch_size = 8
    num_epochs = 600
    iters = 30000//D_batch_size
    lrg = 0.0004
    lr = 0.0001
    beta1 = 0.9
    beta2 = 0.999
    Lambda = 10
    critic_iters = 5
    cudnn.benchmark = True
    workers = 0
    lz = 4
    ##Dataloaders for each orientation
    device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device, " will be used.\n")

    training_imgs = pre_proc(real_data, sf)

    # Create the Genetator network
    netG = Gen().to(device)
    rt = 0
    
    if Training == False:
        rt = 1
        D_batch_size = 1

    if rt:
        netG.load_state_dict(torch.load('trained_generators/microstructure/cgan_microstructure_27/cgan_microstructure_27_Gen.pt'))
    optG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, beta2))

    # Define 1 discriminator and optimizer
    netD = Disc().to(device)
    if rt:
        netD.load_state_dict(torch.load('trained_generators/microstructure/cgan_microstructure_27/cgan_microstructure_27_Disc.pt'))
    optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))

    disc_real_log = []
    disc_fake_log = []
    gp_log = []
    Wass_log = []
    if wandb_name:
        wandb_init(wandb_name)
        wandb.watch(netD)
        wandb.watch(netG)
    print("Starting Training Loop...")
    # For each epoch
    start = time.time()
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i in range(iters):
            real_data, lbl = batch(training_imgs, labels, l, D_batch_size, device)
            G_labels = lbl.repeat(1, 1, lz, lz).to(device)
            D_labels = lbl.repeat(1, 1, l, l).to(device)
            ### Discriminator
            ## Generate fake image batch with G
            noise = torch.randn(D_batch_size, nz, lz, lz, device=device)
            fake_data = netG(noise, G_labels).detach()
            # For each dimension
            start_disc = time.time()
            ##train on real images
            netD.zero_grad()
            # Forward pass real batch through D
            out_real = netD(real_data, D_labels).view(-1).mean()
            # train on fake images
            out_fake = netD(fake_data, D_labels).mean()
            #grad calc
            gradient_penalty = cond_calc_gradient_penalty(netD, real_data, fake_data,
                                                          D_batch_size, l,
                                                          device, Lambda, nc, D_labels)

            disc_cost = out_fake - out_real + gradient_penalty
            disc_cost.backward()
            optD.step()

            disc_real_log.append(out_real.item())
            disc_fake_log.append(out_fake.item())
            Wass_log.append(out_real.item() - out_fake.item())
            gp_log.append(gradient_penalty.item())
            if wandb_name:
                wandb.log({'Gradient penalty': gradient_penalty.item()})
                wandb.log({'Wass': out_real.item() - out_fake.item()})
                wandb.log({'Discriminator real': out_real.item()})
                wandb.log({'Discriminator fake': out_fake.item()})
            ### Generator Training
            start_gen = time.time()
            if i % int(critic_iters) == 0:
                netG.zero_grad()
                errG = 0
                noise = torch.randn(D_batch_size, nz, lz, lz, device=device)
                fake = netG(noise, G_labels)
                # For each plane
                output = netD(fake, D_labels)
                errG -= output.mean()
                # Calculate gradients for G
                errG.backward()
                optG.step()
            # Output training stats & show imgs
            if i % 200 == 0:
                torch.save(netG.state_dict(), pth + '_Gen.pt')
                torch.save(netD.state_dict(), pth + '_Disc.pt')
                if wandb_name:
                    wandb_save_models(pth, '_Gen.pt')
                    wandb_save_models(pth, '_Disc.pt')
                noise = torch.randn(1, nz, lz + 2, lz + 2, device=device)
                netG.eval()
                for tst_lbls in labels:
                    lbl = torch.zeros(1, nlabels * 2, lz + 2, lz + 2)
                    lbl_str = ''
                    for lb in range(nlabels):
                        lbl[:, lb] = tst_lbls[lb]
                        lbl[:, lb + nlabels] = 1- tst_lbls[lb]
                        lbl_str += '_' + str(tst_lbls[lb])
                    with torch.no_grad():
                        img = netG(noise, lbl.type(torch.FloatTensor).cuda())
                    img = img.cpu().detach().numpy()[0]
                    img = np.argmax(img, axis=0)
                    wandb.log({"fake" : wandb.Image(img)})
                    # img: list of tensors of shape [1,2,64,64]
                    # turn into shape into shapes [64,64]
                    # take argmax over 1st dimension
                netG.train()

            ###Print progress
                ## calc ETA
                calc_eta(iters, time.time(), start, i, epoch, num_epochs)
                ###save example slices
                # plotting graphs
                if not wandb_name:
                    graph_plot([disc_real_log, disc_fake_log], ['real', 'fake'], pth, 'LossGraph')
                    graph_plot([Wass_log], ['Wass Distance'], pth, 'WassGraph')
                    graph_plot([gp_log], ['Gradient Penalty'], pth, 'GpGraph')

