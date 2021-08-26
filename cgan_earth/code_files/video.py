import torch

def roll_noise(original_noise, step):
    '''
    roll noise from left to right with specified step size
    Params:
        original_noise: tensor of dimension (1, z_dim, lf, lf*ratio)
        step: float or integer indicating the position in z space compared to original noise
        max_step: integer of the largest possible position in z (lf*ratio) 
    Return:
        new noise of dimension (1, z_dim, lf, lf*ratio)
    '''
    int_step = int(step)
    if int_step == step:
        out_noise = torch.cat((original_noise[:, :, :, int_step:], original_noise[:, :, :, 2:2+int_step]), -1)
    else:
        # do linear interpolation
        prev_noise = torch.cat((original_noise[:, :, :, int_step:], original_noise[:, :, :, 2:2+int_step]), -1)
        int_step += 1
        new_noise = torch.cat((original_noise[:, :, :, int_step:], original_noise[:, :, :, 2:2+int_step]), -1)
        diff = torch.sub(new_noise, prev_noise)
        diff = torch.multiply(diff, step-int_step+1)
        out_noise = torch.add(prev_noise, diff)
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
    if z_step_size > z_len:
        z_step_size = z_len
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
            #print(new_label[:, label1_channel, :, :step_idx])
            new_label[:, label2_channel, :, :] = torch.sub(torch.ones_like(cur_label[:, label1_channel, :, :]), new_label[:, label1_channel, :, :])
            l_step += 1
            z_step += int(z_len*z_step_size)
        elif l_step == l_stop_step and z_step < z_len:
            # prevent label from getting larger than 1
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



