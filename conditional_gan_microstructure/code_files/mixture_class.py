from PIL import Image
from time import sleep
from tqdm import tqdm
import numpy as np
import pickle

class CircleMixtureGenerator():
    def __init__(self, length, radius1, radius2, ratio):
        '''
        Generate 50:50 mixture of circles
        Parameters:
            length: side length of the image generated, integer
            radius: radius of the generated circles, integer
            ratio: pixel to image ratio, float
        '''
        self.length = length
        self.radius1 = radius1
        self.radius2 = radius2
        self.ratio = ratio

    def make_circles(self, centre, radius):
        '''
        Create a single circle with the specified centre
        Parameters:
            centre: numpy array of shape (2,)
            radius: radius of circle
        Return:
            mask: an array of dimension (length, length) filled with boolean
        '''
        Y, X = np.ogrid[:self.length, :self.length]
        dist_from_center = np.sqrt((X - centre[1])**2 + (Y-centre[0])**2)
        mask = dist_from_center < radius

        return mask

    def check_centre_distance(self, centre1, centre2, min_distance):
        '''
        Check if two circles are overlapping
        Parameters:
            centre1: numpy array of shape (2,)
            centre2: numpy array of shape (2,)
            min_distance: minimum distance of two circles apart
        Return:
            boolean of whether the two circles overlap 
        '''
        dist_apart = np.sqrt((centre1[0]-centre2[0])**2 + (centre1[1]-centre2[1])**2)
        if dist_apart >= min_distance:
            return True
        else:
            return False

    def count_centre_overlap(self, centre, other_centres):
        '''
        Count the number of times the created centre overlap with existing other_centres
        (different radius)
        Parameters:
            centre: numpy array of shape (2,)
            other_centres: numpy array of shape (n, 2)
        Return:
            number of times they have overlapped
        '''
        count = 0
        for other_centre in other_centres:
            if self.check_centre_distance(centre, other_centre, self.radius1+self.radius2) == False:
                count += 1
        return count

    def generate_centres(self, radius, other_centres=None, first=True):
        '''
        Generate centres of circles
        Return:
            numpy array of shape (n, 2)
        '''
        centres = np.array([])
        # the number of centres generated depends on the pixel to image ratio
        total = int((self.ratio * (self.length/(2*radius))**2)/2)
        for n in tqdm(range(total)):
            # case when centres is empty, make sure it does not overlap with other_centres
            if len(centres) == 0:
                centres = np.random.randint(radius, self.length-radius, 2)
                if first == False:
                    overlap = True
                    count = 0
                    while overlap == True:
                        count = self.count_centre_overlap(centres, other_centres)
                        if count > 0:
                            centres = np.random.randint(radius, self.length-radius, 2)
                            count = 0
                        else:
                            overlap = False
            else:
                new_centre = np.random.randint(radius, self.length-radius, 2)
                # the case when centres is only occupied by one centre
                if centres.ndim == 1:
                    overlap = True
                    count = 0
                    while overlap == True:
                        if self.check_centre_distance(centres, new_centre, 2*radius) == False:
                            count += 1
                        if first == False:
                            count += self.count_centre_overlap(new_centre, other_centres)
                        if count > 0:
                            new_centre = np.random.randint(radius, self.length-radius, 2)
                            count = 0
                        else:
                            overlap = False
                # when centres is filled with 2 or more centres, check the arrays one by one
                else:
                    overlap = True
                    count = 0
                    while overlap == True:
                        for old_centre in centres:
                            if self.check_centre_distance(old_centre, new_centre, 2*radius) == False:
                                count += 1
                        if first == False:
                            count += self.count_centre_overlap(new_centre, other_centres)
                        if count > 0:
                            new_centre = np.random.randint(radius, self.length-radius, 2)
                            count = 0
                        else:
                            overlap = False
                centres = np.vstack((centres,new_centre))
            sleep(0.001)
        return centres
    
    def make_image(self, path):
        '''
        Make images of circles
        Parameters:
            n_images: number of images to be generated
            path: system path of saving the images
        Return:
            array of size (self.length, self.length)
        '''
        image_array = np.array([])
        x = np.zeros((self.length, self.length))
        centres1 = self.generate_centres(self.radius1)
        centres2 = self.generate_centres(radius=self.radius2, other_centres=centres1, first=False)
        for centre in centres1:
            x[self.make_circles(centre, self.radius1)] = 1
        for centre in centres2:
            x[self.make_circles(centre, self.radius2)] = 1
        if len(image_array) == 0:
            image_array = np.copy(x)
        else:
            image_array = np.vstack((image_array,x))
        image_array = np.reshape(image_array, (self.length, self.length)).astype(np.uint8)
        im = Image.fromarray(image_array)
        im.save(path)
        return image_array
            