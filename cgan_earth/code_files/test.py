from os import XATTR_REPLACE
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

labels = torch.tensor([0, 1, 2])
n_classes = 3
cur_label = F.one_hot(labels, num_classes=n_classes)[:, :, None, None]
cur_label = cur_label[0].repeat(1, 1, 6, 12)

def make_circle(max_len, radius, lf, centre=None):
    '''
    Create a single circle around the image centre
    Parameters:
        max_len: maximum diameter of the generated circle
        radius: float between 0 and 1
        ratio: image width / image height
    Return:
        mask: an array of dimension (max_len/ratio, max_len) filled with boolean
    '''
    Y, X = np.ogrid[:lf, :max_len]
    if np.any(centre) == None:
        dist_from_center = np.sqrt((X - (max_len//2))**2 + (Y - (lf//2))**2)
    else:
        dist_from_center = np.sqrt((X - centre[1])**2 + (Y - centre[0])**2)
    if radius != None:
        mask = dist_from_center < max_len*radius
    else:
        mask = dist_from_center < max_len + 100
    return mask[None, None, :, :]

def check_centre_distance(centre1, centre2, radius):
    '''
    Check if two circles are overlapping
    Parameters:
        centre1: numpy array of shape (2,)
        centre2: numpy array of shape (2,)
        radius: integer
    Return:
        boolean of whether the two circles overlap 
    '''
    dist_apart = np.sqrt((centre1[0]-centre2[0])**2 + (centre1[1]-centre2[1])**2)
    if dist_apart >= 2*radius:
        return True
    else:
        return False

def generate_centres(n_circles, img_width, img_len, radius):
    '''
    Generate non-overlapping centres of circles
    Params:
        n_circles: integer, number of centres to be generated
        img_width: integer, maximum width of the image (in this case, lf)
        img_len: integer, maximum length of the image (in this case, lf*ratio)
        radius: integer, radius of generated circle
    Return:
        numpy array of shape (n, 2)
    '''
    centres = np.array([])
    for n in range(n_circles):
        if len(centres) == 0:
            centres = np.array([np.random.randint(radius, img_width-radius, 1)[0], np.random.randint(radius, img_len-radius, 1)[0]])[None, :]
        else:
            new_centre = np.array([np.random.randint(radius, img_width-radius, 1)[0], np.random.randint(radius, img_len-radius, 1)[0]])
            if centres.ndim == 1:
                while check_centre_distance(centres, new_centre, radius) == False:
                    new_centre = np.array([np.random.randint(radius, img_width-radius, 1)[0], np.random.randint(radius, img_len-radius, 1)[0]])
            else:
                overlap = True
                count = 0
                while overlap == True:
                    for old_centre in centres:
                        if check_centre_distance(old_centre, new_centre, radius) == False:
                            count += 1
                    if count > 0:
                        new_centre = np.array([np.random.randint(radius, img_width-radius, 1)[0], np.random.randint(radius, img_len-radius, 1)[0]])
                        count = 0
                    else:
                        overlap = False
            centres = np.vstack((centres,new_centre))
    return centres

def make_multiple_circles(radius, lf, max_len, centres):
    mask = np.array([])
    for centre in centres:
        if len(mask) == 0:
            mask = make_circle(max_len, radius/max_len, lf, centre=centre)
        else:
            mask += make_circle(max_len, radius/max_len, lf, centre=centre)
    return mask

def circular_effects(label1_channel, label2_channel, cur_label, z_step_num, l_step_size, lf, max_len, l_step, z_step,
                 l_done_step, z_done_step, z_max_num, centres):
    # make sure the numbers are in the right range
    if z_step_num < 2:
        z_step_num = 2
    if z_max_num < z_step_num + 2:
        z_max_num = z_step_num + 2

    # make multiple circles
    # first generate centres
    z_step_radius = z_step_num // 2
    z_max_radius = z_max_num // 2

    new_label = cur_label.float()
    step_radius = z_step + z_step_radius
    l_done_radius = l_done_step + z_step_radius
    z_done_radius = z_done_step + z_step_radius
    l_stop_step = 1 // l_step_size
    # the size of circles cannot be larger than z_max_radius
    if z_done_radius >= z_max_radius or l_done_radius >= z_max_radius:
        new_label[:, label1_channel, :, :] = torch.ones_like(new_label[:, label1_channel, :, :])
        new_label[:, label2_channel, :, :] = torch.zeros_like(new_label[:, label2_channel, :, :])
    else:
        if step_radius > z_max_radius:
            step_radius = z_max_radius
        if l_step <= l_stop_step:
            mask = make_multiple_circles(step_radius, lf, max_len, centres)
            l_step += 1
            z_step += z_step_radius
        elif l_step == l_stop_step and z_step < z_max_radius:
            # prevent label from getting larger than 1
            mask_s = make_multiple_circles(l_done_radius, lf, max_len, centres)
            mask_b = make_multiple_circles(step_radius, lf, max_len, centres)
            mask = np.invert(np.invert(mask_b) + mask_s)
            l_done_step += z_step_radius
            z_step += z_step_radius
        else:
            if l_done_radius != 0:
                z_done_radius = l_done_radius
                l_done_step += z_step_radius
            mask_s = make_multiple_circles(z_done_radius, lf, max_len, centres)
            mask_b = make_multiple_circles(step_radius, lf, max_len, centres)
            mask = np.invert(np.invert(mask_b) + mask_s)
            z_done_step += z_step_radius 
        new_label[:, label1_channel, :, :][mask] = torch.sub(new_label[:, label1_channel, :, :][mask], l_step_size)
        new_label[:, label2_channel, :, :] = torch.sub(torch.ones_like(cur_label[:, label1_channel, :, :]), new_label[:, label1_channel, :, :])
    return new_label, l_step, z_step, l_done_step, z_done_step

label = cur_label
l_step = 0
z_step = 0
l_done_step = 0
z_done_step = 0

centres = generate_centres(2, 6, 12, 2)
for i in range(10):
    label, l_step, z_step, l_done_step, z_done_step = circular_effects(0, 1, label, 2, 0.2, 6, 12, l_step, z_step,
                                                      l_done_step, z_done_step, 4, centres)
    print(label)