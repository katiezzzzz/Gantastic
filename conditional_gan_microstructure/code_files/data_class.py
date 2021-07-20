from time import sleep
from tqdm import tqdm
import numpy as np
import pickle

class CircleGenerator():
    def __init__(self, length, radius, ratio):
        '''
        Parameters:
            length: side length of the image generated, integer
            radius: radius of the generated circles, integer
            ratio: pixel to image ratio, float
        '''
        self.length = length
        self.radius = radius
        self.ratio = ratio

    def make_circles(self, centre):
        '''
        Create a single circle with the specified centre
        Parameters:
            centre: numpy array of shape (2,)
        Return:
            mask: an array of dimension (length, length) filled with boolean
        '''
        Y, X = np.ogrid[:self.length, :self.length]
        dist_from_center = np.sqrt((X - centre[1])**2 + (Y-centre[0])**2)
        mask = dist_from_center < self.radius

        return mask

    def check_centre_distance(self, centre1, centre2):
        '''
        Check if two circles are overlapping
        Parameters:
            centre1: numpy array of shape (2,)
            centre2: numpy array of shape (2,)
        Return:
            boolean of whether the two circles overlap 
        '''
        dist_apart = np.sqrt((centre1[0]-centre2[0])**2 + (centre1[1]-centre2[1])**2)
        if dist_apart >= 2*self.radius:
            return True
        else:
            return False

    def generate_centres(self):
        '''
        Generate centres of circles
        Return:
            numpy array of shape (n, 2)
        '''
        self.centres = np.array([])
        # the number of centres generated depends on the pixel to image ratio
        total = int(self.ratio * (self.length/(2*self.radius))**2)
        for n in tqdm(range(total)):
            if len(self.centres) == 0:
                self.centres = np.random.randint(self.radius, self.length-self.radius, 2)
            else:
                new_centre = np.random.randint(self.radius, self.length-self.radius, 2)
                if self.centres.ndim == 1:
                    while self.check_centre_distance(self.centres, new_centre) == False:
                        new_centre = np.random.randint(self.radius, self.length-self.radius, 2)
                else:
                    overlap = True
                    count = 0
                    while overlap == True:
                        for old_centre in self.centres:
                            if self.check_centre_distance(old_centre, new_centre) == False:
                                count += 1
                        if count > 0:
                            new_centre = np.random.randint(self.radius, self.length-self.radius, 2)
                            count = 0
                        else:
                            overlap = False
                self.centres = np.vstack((self.centres,new_centre))
            sleep(0.001)
        return self.centres
    
    def make_images(self, n_images, path):
        '''
        Make images of circles
        Parameters:
            n_images: number of images to be generated
            path: system path of saving the images
        Return:
            array of size (n_images, self.length, self.length)
        '''
        image_array = np.array([])
        for n in range(n_images):
            x = np.zeros((self.length, self.length))
            centres = self.generate_centres()
            for centre in centres:
                x[self.make_circles(centre)] = 1
            if len(image_array) == 0:
                image_array = np.copy(x)
            else:
                image_array = np.vstack((image_array,x))
        image_array = np.reshape(image_array, (n_images, self.length, self.length))
        with open(path, 'wb') as f:
            pickle.dump(image_array, f)
        
        return image_array
            