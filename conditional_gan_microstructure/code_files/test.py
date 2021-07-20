import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

PATH = os.path.dirname(os.path.realpath(__file__))
print(PATH)

with open(PATH+'/data/BigCircles.pkl', 'rb') as pickle_file:
    content = pickle.load(pickle_file)

figure(figsize=(10,10))
for image in content:
    plt.imshow(image)
    plt.show()