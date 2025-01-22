import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

'''
#cude not working
# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = np.argsort(loss_1.data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2
'''

'''
# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    loss_1 = F.cross_entropy(y_1, t, reduce=False)
    ind_1_sorted = torch.argsort(loss_1.data.cpu())  # Move to CPU before sorting with NumPy
    loss_1_sorted = loss_1[ind_1_sorted]
    ind = ind.cpu().numpy()
    loss_2 = F.cross_entropy(y_2, t, reduce=False)
    ind_2_sorted = torch.argsort(loss_2.data.cpu())  # Move to CPU before sorting with NumPy
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    # Move indices to CPU to ensure compatibility with numpy
    ind_1_sorted_cpu = ind_1_sorted.cpu().numpy()
    ind_1_sorted_cpu = np.array(ind_1_sorted_cpu, dtype=int)
    print(type(ind_1_sorted_cpu))
    ind_2_sorted_cpu = ind_2_sorted.cpu().numpy()

    # Pure ratio calculations (ensure `noise_or_not` is also on the CPU)
    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted_cpu[:num_remember]]]) / float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted_cpu[:num_remember]]]) / float(num_remember)

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update) / num_remember, torch.sum(loss_2_update) / num_remember, pure_ratio_1, pure_ratio_2
'''
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    """
    Compute the Co-Teaching loss.
    Args:
        y_1: Output logits from model 1.
        y_2: Output logits from model 2.
        t: Ground truth labels.
        forget_rate: Fraction of samples to forget (noisy samples).
        ind: Indices of the samples in the batch (as a PyTorch tensor).
        noise_or_not: Boolean array indicating whether each sample is clean or noisy.
    """
    # Ensure ind is a PyTorch tensor
    if isinstance(ind, np.ndarray):
        ind = torch.from_numpy(ind).to(y_1.device)  # Convert to tensor and move to the correct device

    # Ensure noise_or_not is a PyTorch tensor on the correct device
    if isinstance(noise_or_not, np.ndarray):
        noise_or_not = torch.from_numpy(noise_or_not).to(y_1.device)

    # Compute losses
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    loss_2 = F.cross_entropy(y_2, t, reduction='none')

    # Sort losses
    _, ind_1_sorted = torch.sort(loss_1.data)
    _, ind_2_sorted = torch.sort(loss_2.data)

    # Calculate the number of samples to remember
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1))

    # Select the indices of the samples with the smallest losses
    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    # Compute pure ratios
    pure_ratio_1 = torch.sum(noise_or_not[ind[ind_1_update]]).item() / num_remember
    pure_ratio_2 = torch.sum(noise_or_not[ind[ind_2_update]]).item() / num_remember

    # Exchange updates
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return loss_1_update, loss_2_update, pure_ratio_1, pure_ratio_2