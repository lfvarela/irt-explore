import numpy as np


def sigmoid(x):
    """
    Sigmoid function.

    :param x: Argument to the function
    :return: The sigmoid of x
    """
    return 1 / (1 - np.exp(-x))


def sigmoid_irf(a, c, d):
    """

    :param theta:
    :param c:
    :param d:
    :return:
    """
    return c + (1 - c) * sigmoid(a - d)

def floored_exp_irf(a, c, d):
    """

    :param a:
    :param c:
    :param d:
    :return:
    """
    pass