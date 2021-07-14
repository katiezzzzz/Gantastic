import preprocessing
import util

import os
import torch
import wandb
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

def train(gen, disc, mnist_shape, n_classes, criterion, n_epochs, z_dim, batch_size, lr, device):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        ])

    dataloader = DataLoader(
        MNIST(os.path.dirname(os.path.realpath(__file__))+'/data', download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True)

    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

    cur_step = 0
    generator_losses = []
    discriminator_losses = []
    display_step = 500

    for epoch in range(n_epochs):
        for real, labels in tqdm(dataloader):
            cur_batch_size = len(real)
            # Flatten the batch of real images from the dataset
            real = real.to(device)

            one_hot_labels = preprocessing.get_one_hot_labels(labels.to(device), n_classes)
            image_one_hot_labels = one_hot_labels[:, :, None, None]
            image_one_hot_labels = image_one_hot_labels.repeat(1, 1, mnist_shape[1], mnist_shape[2])

            disc_opt.zero_grad()
            # Get noise corresponding to the current batch_size 
            fake_noise = util.get_noise(cur_batch_size, z_dim, device=device)
        
            noise_and_labels = preprocessing.combine_vectors(fake_noise, one_hot_labels)
            fake = gen(noise_and_labels)
            
            # Make sure that enough images were generated
            assert len(fake) == len(real)
            # Check that correct tensors were combined
            assert tuple(noise_and_labels.shape) == (cur_batch_size, fake_noise.shape[1] + one_hot_labels.shape[1])
            # It comes from the correct generator
            assert tuple(fake.shape) == (len(real), 1, 28, 28)

            # get the predictions from the discriminator
            fake_image_and_labels = preprocessing.combine_vectors(fake, image_one_hot_labels).detach()
            real_image_and_labels = preprocessing.combine_vectors(real, image_one_hot_labels)
            disc_fake_pred = disc(fake_image_and_labels)
            disc_real_pred = disc(real_image_and_labels)
            
            # Make sure shapes are correct 
            assert tuple(fake_image_and_labels.shape) == (len(real), fake.detach().shape[1] + image_one_hot_labels.shape[1], 28 ,28)
            assert tuple(real_image_and_labels.shape) == (len(real), real.shape[1] + image_one_hot_labels.shape[1], 28 ,28)
            # Make sure that enough predictions were made
            assert len(disc_real_pred) == len(real)
            # Make sure that the inputs are different
            assert torch.any(fake_image_and_labels != real_image_and_labels)
            # Shapes must match
            assert tuple(fake_image_and_labels.shape) == tuple(real_image_and_labels.shape)
            assert tuple(disc_fake_pred.shape) == tuple(disc_real_pred.shape)
            
            disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward(retain_graph=True)
            disc_opt.step() 

            # Keep track of the average discriminator loss
            discriminator_losses += [disc_loss.item()]

            ### Update generator ###
            gen_opt.zero_grad()

            fake_image_and_labels = preprocessing.combine_vectors(fake, image_one_hot_labels)
            disc_fake_pred = disc(fake_image_and_labels)
            gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
            gen_loss.backward()
            gen_opt.step()

            # Keep track of the generator losses
            generator_losses += [gen_loss.item()]

            wandb.log({"gen loss":gen_loss.item()})
            wandb.log({"disc loss":disc_loss.item()})
            wandb.log({"fake":util.make_img_array(8,fake)})
            wandb.log({"real":util.make_img_array(8,real)})

            if cur_step % display_step == 0 and cur_step > 0:
                gen_mean = sum(generator_losses[-display_step:]) / display_step
                disc_mean = sum(discriminator_losses[-display_step:]) / display_step
                print(f"Step {cur_step}: Generator loss: {gen_mean}, discriminator loss: {disc_mean}")
            cur_step += 1
