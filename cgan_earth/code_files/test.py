from os import XATTR_REPLACE
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

a = torch.tensor([[[1, 2, 3, 4],
                  [4, 5, 6, 7],
                  [5, 6, 7, 8],
                  [8, 9, 10, 11]],
                 [[11, 21, 31, 41],
                  [41, 51, 61, 71],
                  [51, 61, 71, 81],
                  [81, 91, 101, 111]]])
l = [[0, 0], [1, 1], [2, 2], [3, 3], [0, 1]]
l = torch.from_numpy(np.array(l))
one_hot = torch.zeros((l.shape[0], 4))
for n in range(l.shape[0]):
    if l[n, 0] == l[n, 1]:
        one_hot[n,l[n]] = 1.0
    else:
        one_hot[n,l[n, 0]] = 0.5
        one_hot[n,l[n, 1]] = 0.5

labels = torch.tensor([0, 1, 2])
n_classes = 3
cur_label = F.one_hot(labels, num_classes=n_classes)[:, :, None, None]
cur_label = cur_label[0].repeat(1, 1, 4, 8)

def make_circle(max_len, radius, ratio):
    '''
    Create a single circle around the image centre
    Parameters:
        max_len: maximum diameter of the generated circle
        radius: float between 0 and 1
        ratio: image width / image height
    Return:
        mask: an array of dimension (max_len/ratio, max_len) filled with boolean
    '''
    height = max_len / ratio
    Y, X = np.ogrid[:height, :max_len]
    dist_from_center = np.sqrt((X - (max_len//2))**2 + (Y-(height//2))**2)
    if radius != None:
        mask = dist_from_center < max_len*radius
    else:
        mask = dist_from_center < max_len + 100
    return mask[None, None, :, :]

def circular_transit(label1_channel, label2_channel, cur_label, z_step_size, l_step_size, lf, ratio, l_step, z_step,
                 l_done_step, z_done_step):
    if z_step_size > 1:
        z_step_size = 1
    z_step_radius = z_step_size / 2
    new_label = cur_label.float()
    max_len = lf*ratio
    step_radius = z_step + z_step_radius
    l_done_radius = l_done_step + z_step_radius
    z_done_radius = z_done_step + z_step_radius
    l_stop_step = 1 // l_step_size
    if z_done_radius >= 0.5 or l_done_radius >= 0.5:
        new_label[:, label1_channel, :, :] = torch.zeros_like(new_label[:, label1_channel, :, :])
        new_label[:, label2_channel, :, :] = torch.ones_like(new_label[:, label2_channel, :, :])
    else:
        if step_radius > 0.5:
            step_radius = None
        if l_step < l_stop_step:
            mask = make_circle(max_len, step_radius, ratio)
            l_step += 1
            z_step += z_step_radius
        elif l_step == l_stop_step and z_step < 0.5:
            # prevent label from getting larger than 1
            mask_s = make_circle(max_len, l_done_radius, ratio)
            mask_b = make_circle(max_len, step_radius, ratio)
            mask = np.invert(np.invert(mask_b) + mask_s)
            l_done_step += z_step_radius
            z_step += z_step_radius
        else:
            if l_done_radius != 0:
                z_done_radius = l_done_radius
                l_done_step += z_step_radius
            mask_s = make_circle(max_len, z_done_radius, ratio)
            mask_b = make_circle(max_len, step_radius, ratio)
            mask = np.invert(np.invert(mask_b) + mask_s)
            z_done_step += z_step_radius 
        new_label[:, label1_channel, :, :][mask] = torch.sub(new_label[:, label1_channel, :, :][mask], l_step_size)
        new_label[:, label2_channel, :, :] = torch.sub(torch.ones_like(cur_label[:, label1_channel, :, :]), new_label[:, label1_channel, :, :])
    return new_label, l_step, z_step, l_done_step, z_done_step

new_label = cur_label
l_step = 0
z_step = 0
l_done_step = 0
z_done_step = 0
for i in range(15):
    new_label, l_step, z_step, l_done_step, z_done_step = circular_transit(0, 1, new_label, 0.3, 0.5, 4, 2, l_step, z_step,
                                                                        l_done_step, z_done_step)

    print(new_label)

