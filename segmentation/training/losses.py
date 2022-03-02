import math
import torch
import torch.nn as nn


def get_loss(loss_function):
    """ Get loss function(s) for the training process.

    :param loss_function: Loss function to use.
        :type loss_function: str
    :return: Loss function / dict of loss functions.
    """

    if loss_function == 'l1':
        criterion = nn.L1Loss()
    elif loss_function == 'l2':
        criterion = nn.MSELoss()
    elif loss_function == 'smooth_l1':
        criterion = nn.SmoothL1Loss()
    elif loss_function == 'weighted_smooth_l1':
        criterion = weighted_smooth_l1

    return criterion


def weighted_smooth_l1(y_pred, y_true, weight_map):

    loss_smooth_l1 = nn.SmoothL1Loss(reduction='none')
    loss = torch.mean(weight_map * loss_smooth_l1(y_pred, y_true))

    return loss


def get_weight_map(ytrue, weight):

    weight_map = gaussian_smoothing_2d(ytrue, 1, 3, 0.5)
    weight_map[weight_map > 0] = weight - 1
    weight_map += 1

    return weight_map


def gaussian_smoothing_2d(x, channels, kernel_size, sigma):

    kernel_size = [kernel_size] * 2
    sigma = [sigma] * 2

    # The gaussian kernel is the product of the
    # gaussian function of each dimension.
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing='ij')

    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding=1, bias=False)
    conv = conv.to('cuda')
    conv.weight.data = kernel.to('cuda')
    conv.weight.requires_grad = False

    return conv(x)
