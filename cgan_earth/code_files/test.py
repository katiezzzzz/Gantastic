import numpy as np
import torch
import torch.nn.functional as F

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

def scroll_transit(label1_channel, label2_channel, cur_label, z_step_size, l_step_size, z_len, l_step, z_step,
                 l_done_step, z_done_step):
    if z_step_size > z_len:
        z_step_size = z_len
    step_idx = int(z_step + z_len*z_step_size)
    l_done_idx = int(l_done_step + z_len*z_step_size)
    z_done_idx = int(z_done_step + z_len*z_step_size)
    l_stop_step = 1 // l_step_size
    #print(f'l stop: {l_stop_step}')
    #print(f'z idx: {step_idx}')
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

z_done_step = 0
l_done_step = 0
z_step = 0
l_step = 0
new_label = cur_label
for _ in range(10):
    new_label, l_step, z_step, l_done_step, z_done_step = scroll_transit(0, 1, new_label, 0.5, 0.5, 8, l_step, z_step, l_done_step, z_done_step)
    print(new_label)