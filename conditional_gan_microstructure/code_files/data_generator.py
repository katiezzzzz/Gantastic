import os
from mixture_class import CircleMixtureGenerator
from data_class import CircleGenerator
from egg_class import EggGenerator

PATH = os.path.dirname(os.path.realpath(__file__))

# make small circles, image size 800 x 800
image_length = 2000
medium_radius = 8
small_radius = 6
big_radius = 10
ratio = 0.4

#SmallGenerator = CircleGenerator(image_length, small_radius, ratio)
#BigGenerator = CircleGenerator(image_length, big_radius, ratio)
#MediumGenerator = CircleGenerator(image_length, medium_radius, ratio)
#MixtureGenerator = CircleMixtureGenerator(image_length, small_radius, big_radius, 0.4)
EggsGenerator = EggGenerator(image_length, medium_radius, ratio)

#SmallGenerator.make_image(PATH+'/data/SmallCircles.tiff')
#BigGenerator.make_image(PATH+'/data/BigCircles.tiff')
#MediumGenerator.make_image(PATH+'/data/r8.tiff')
#MixtureGenerator.make_image(PATH+'/data/r_mix.tiff')
EggsGenerator.make_image(PATH+'/data/r_egg.tiff')
