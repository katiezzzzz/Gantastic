
import torch
import wandb
from torch import nn
import preprocessing
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

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


def load_data(path, train, batch_size):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    if train == True:
        trainloader = DataLoader(
            MNIST(path+'/data', train=True, download=True, transform=transform),
            batch_size=batch_size,
            shuffle=True)
        return trainloader
    else:
        testloader = DataLoader(
            MNIST(path+'/data', train=False, download=True, transform=transform),
            batch_size=batch_size,
            shuffle=True)
        return testloader


def make_img_array(n_img, img_tensor):
    img_num = 0
    img_array = []
    for i in img_tensor:
        img_array += [wandb.Image(i)]
        img_num += 1
        if img_num >= n_img:
            break
    return img_array


def test(path, testloader, gen, disc, criterion, z_dim, device, n_classes, mnist_shape):
    gen.load_state_dict(torch.load(path+'/store/gen.pkl'))
    disc.load_state_dict(torch.load(path+'/store/disc.pkl'))
    gen.eval()

    cur_step = 0

    for real, labels in tqdm(testloader):
        real = real.to(device)
        one_hot_labels = preprocessing.get_one_hot_labels(labels.to(device), n_classes)
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, mnist_shape[1], mnist_shape[2])

        noise = get_noise(len(real), z_dim, device=device)
        noise_and_labels = preprocessing.combine_vectors(noise, one_hot_labels)
        fake = gen(noise_and_labels)

        fake_image_and_labels = preprocessing.combine_vectors(fake, image_one_hot_labels).detach()
        real_image_and_labels = preprocessing.combine_vectors(real, image_one_hot_labels)
        disc_fake_pred = disc(fake_image_and_labels)
        disc_real_pred = disc(real_image_and_labels)

        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2

        fake_image_and_labels = preprocessing.combine_vectors(fake, image_one_hot_labels)
        disc_fake_pred = disc(fake_image_and_labels)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))

        wandb.log({"gen loss":gen_loss.item()})
        wandb.log({"disc loss":disc_loss.item()})
        wandb.log({"fake":make_img_array(8,fake)})
        wandb.log({"real":make_img_array(8,real)})


        
