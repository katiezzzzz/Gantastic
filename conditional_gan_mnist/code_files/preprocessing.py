import torch
import torch.nn.functional as F

# labelling
def get_one_hot_labels(labels, n_classes):
    return F.one_hot(labels, num_classes=n_classes)

# concatenate one-hot vectors to class vectors
def combine_vectors(x, y):
    combined = torch.cat((x.float(),y.float()),1)
    return combined

