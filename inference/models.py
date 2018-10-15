import torch

def two_param_sigmoid_irf(a, d, c):
    """

    :param a:
    :param d:
    :param c:
    :return:
    """
    return c + (1 - c) * torch.sigmoid(a - d)


def floored_exp_irf(a, d, l, c):
    return torch.max(torch.tensor(c), 1 - torch.exp(-l * (a - d)))