import numpy as np
import torch

def roll_pixels(original_img, n_pixels, max_pixels, IntStep=False):
    if IntStep == False:
        n_pixels = 32
    if n_pixels < max_pixels:
        out_img = np.concatenate((original_img[:, :, :, n_pixels:], original_img[:, :, :, n_pixels:n_pixels*2]), -1)
    else:
        out_img = original_img
    return out_img

def roll_noise(original_noise, step, max_step, IntStep=True):
    '''
    roll noise from left to right with specified step size
    Params:
        original_noise: tensor of dimension (1, z_dim, lf, lf*ratio)
        step: float or integer indicating the position in z space compared to original noise
        max_len: maximum lenth of z vector, lf*ratio
    Return:
        new noise of dimension (1, z_dim, lf, lf*ratio)
    '''
    int_step = int(step)
    if IntStep == True:
        repeat_idx = 2
    else:
        repeat_idx = 3

    if int_step < max_step:
        out_noise = torch.cat((original_noise[:, :, :, int_step:], original_noise[:, :, :, repeat_idx:repeat_idx+int_step]), -1)
    else:
        out_noise = original_noise
    return out_noise

def uniform_transit(label1_channel, label2_channel, cur_label, l_step_size):
    '''
    Compute uniform transition form one label to the other
    Params:
        label1_channel: integer, index indicating the original channel
        label2_channel: integer, index indicating the target channel
        cur_label: tensor of shape (1, n_classes, lf, lf*ratio)
        l_step_size: float with value between 0 and 1
    Return:
        new label of shape (1, n_classes, lf, lf*ratio)
    '''
    new_label = cur_label.float()
    if cur_label[:, label1_channel, 0, 0] > 0:
        new_label[:, label1_channel, :, :] = torch.sub(cur_label[:, label1_channel, :, :], l_step_size)
        new_label[:, label2_channel, :, :] = torch.sub(torch.ones_like(cur_label[:, label1_channel, :, :]), new_label[:, label1_channel, :, :])
    return new_label
    
def scroll_transit(label1_channel, label2_channel, cur_label, z_step_size, l_step_size, z_len, l_step, z_step,
                 l_done_step, z_done_step):
    if z_step_size > 1:
        z_step_size = 1
    step_idx = int(z_step + z_len*z_step_size)
    l_done_idx = int(l_done_step + z_len*z_step_size)
    z_done_idx = int(z_done_step + z_len*z_step_size)
    l_stop_step = 1 // l_step_size
    new_label = cur_label.float()
    if z_done_idx >= z_len or l_done_idx >= z_len:
        new_label[:, label1_channel, :, :] = torch.zeros_like(new_label[:, label1_channel, :, :])
        new_label[:, label2_channel, :, :] = torch.ones_like(new_label[:, label2_channel, :, :])
    else:
        if step_idx > z_len:
            step_idx = None
        if l_step < l_stop_step:
            new_label[:, label1_channel, :, :step_idx] = torch.sub(new_label[:, label1_channel, :, :step_idx], l_step_size)
            new_label[:, label2_channel, :, :] = torch.sub(torch.ones_like(cur_label[:, label1_channel, :, :]), new_label[:, label1_channel, :, :])
            l_step += 1
            z_step += int(z_len*z_step_size)
        elif l_step == l_stop_step and z_step < z_len:
            new_label[:, label1_channel, :, l_done_idx:step_idx] = torch.sub(new_label[:, label1_channel, :, l_done_idx:step_idx], l_step_size)
            new_label[:, label2_channel, :, :] = torch.sub(torch.ones_like(cur_label[:, label1_channel, :, :]), new_label[:, label1_channel, :, :])
            l_done_step += int(z_len*z_step_size)
            z_step += int(z_len*z_step_size)
        else:
            if l_done_idx != 0:
                z_done_idx = l_done_idx
                l_done_step += int(z_len*z_step_size)
            new_label[:, label1_channel, :, z_done_idx:step_idx] = torch.sub(new_label[:, label1_channel, :, z_done_idx:step_idx], l_step_size)
            new_label[:, label2_channel, :, :] = torch.sub(torch.ones_like(cur_label[:, label1_channel, :, :]), new_label[:, label1_channel, :, :])
            z_done_step += int(z_len*z_step_size)
    return new_label, l_step, z_step, l_done_step, z_done_step

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

def circular_transit(label1_channel, label2_channel, cur_label, z_step_size, l_step_size, lf, ratio, max_len, l_step, z_step,
                 l_done_step, z_done_step):
    if z_step_size > 1:
        z_step_size = 1
    z_step_radius = z_step_size / 2
    new_label = cur_label.float()
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