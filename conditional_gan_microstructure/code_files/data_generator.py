import os
from data_class import CircleGenerator

PATH = os.path.dirname(os.path.realpath(__file__))

# make small circles, image size 800 x 800
image_length = 2000
small_radius = 6
big_radius = 10
n_images = 1
ratio = 0.4

SmallGenerator = CircleGenerator(image_length, small_radius, ratio)
BigGenerator = CircleGenerator(image_length, big_radius, ratio)

SmallGenerator.make_images(n_images, PATH+'/data/SmallCircles.pkl')
BigGenerator.make_images(n_images, PATH+'/data/BigCircles.pkl')
