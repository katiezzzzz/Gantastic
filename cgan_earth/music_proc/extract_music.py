import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import os

PATH = os.path.dirname(os.path.realpath(__file__))
# extract volume
data, samplerate = sf.read(PATH+'/Crazy-SimplePlan.wav')
print(data.shape)

print(data[:,0])
plt.plot(data[:,0])
plt.show()