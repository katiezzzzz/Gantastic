# Gantastic

## UROP Project with TLDR Group (Dyson School of Engineering, Imperial College London)
The project was done from 06/2021 to 09/2021 and the research was focused around Generative Adversarial Networks (GAN).

### Exploring MNIST
Created a basic GAN network and trained using MNIST dataset
- Network architecture in `conditional_gan_mnist/code_files/networks.py`
- Network hyper-parameters in `conditional_gan_mnist/code_files/params.py`
- Data preprocessing functions in `conditional_gan_mnist/code_files/preprocessing.py`
- Training function in `conditional_gan_mnist/code_files/training.py`
- Function that runs the model `conditional_gan_mnist/code_files/run_model.py`
- Utility functions used in the model `conditional_gan_mnist/code_files/util.py`

### Exploring a microstructure dataset
Created a dataset of particles (circles / eggs / mixture of them, of different radius)
- Data generation done in `conditional_gan_microstructure/code_files/data_generator.py`
- Circle generation class in `conditional_gan_microstructure/code_files/data_class.py`
- Egg generation class in `conditional_gan_microstructure/code_files/egg_class.py`
- Mixture (of circles and eggs) generation class in `conditional_gan_microstructure/code_files/mixture_class.py`
- cGAN used as a reference
