import numpy as np
import torch
import torch.nn.functional as f
import matplotlib.pyplot as plt


def calc_acc(preds, targets):
    return (targets.eq(preds.float()).sum().float() / (targets.shape[0] * targets.shape[1])).item()


def calc_precision(preds, targets):
    return ((targets.int() & preds).sum().float() / preds.sum().float()).item()


def rmse(A, B):
    if isinstance(A, torch.Tensor):
        return torch.sqrt(torch.sum((A - B)**2).float() / (A.shape[0] * A.shape[1])).item()
    return np.sqrt(np.mean((A - B)**2))


def min_max_normalize(A):
    return (A - A.min()) / (A.max() - A.min() + 1e-7)


def train(model, optim, epochs, Y):
    """
    General train sequence for model.

    :param model: The model. Should inherit from nn.Module
    :param optim: A torch.optim optimizer.
    :param epochs: The number of epochs to train the model for.
    :param Y: The data, a (num_students, num_questions) tensor.

    :return: Three lists of length epochs holding the loss, accuracy and precision values for each epoch (on the train
    set).
    """
    losses = []
    train_acc = []
    precision = []

    model.train()

    for t in range(epochs):
        # Forward pass: compute predicted y

        scores = model(Y)
        print(scores[0].shape)

        # Compute and print loss
        loss = f.binary_cross_entropy(scores[0], Y)

        preds = (scores > 0.5).int()

        losses.append(loss.item())
        train_acc.append(calc_acc(preds, Y))
        precision.append(calc_precision(preds, Y))

        # We zero out gradients before optimization step.
        optim.zero_grad()

        # Backprop to compute gradients of params wrt loss
        loss.backward()

        # Update the parameters using the gradients computed.
        optim.step()

    return losses, train_acc, precision


def sample(r):
    if isinstance(r, tuple):
        return 10**np.random.uniform(r[0], r[1])
    return r


def hyperparam_search(model_class, init_agrs, params, samples, epochs, Y):
    """
    Carries out random hyperparameter search on model.

    :param model_class: The model class.
    :param init_args: A tuple holding the arguments to initialize a new model of model_class.
    :param params: A dict mapping hyperparameter to a (min exponent, max exponent) tuple, determining a range from which
    to sample. Notice that the range is given as a range of exponents, this is so we get a unifrom distribution on the
    exponents of the samples, instead of on the values of the parameters themselves.
    Alternatively, keys can map to a single number if no tuning is needed on that hyperparameter. Valid parameters are
    'lr', 'reg', 'momentum'.
    :param samples: The number of samples to draw for each hyperparameter.
    :param epochs: The number of epochs on which to train each candidate model. These should be small.
    :param Y: The data set.

    :return: The maximum accuracy achieved and a tuple holding the optimal parameters found.
    """
    lr_range = params.get('lr', 1e-3)
    reg_range = params.get('reg', 1e-2)
    m_range = params.get('momentum', 0)

    n = int(isinstance(lr_range, tuple)) + int(isinstance(reg_range, tuple)) + int(isinstance(m_range, tuple))

    param_candidates = [(sample(lr_range), sample(reg_range), sample(m_range)) for _ in range(n * samples)]

    max_acc = 0
    best_params = (0, 0, 0)

    for candidate in param_candidates:
        model = model_class(*init_agrs)
        l, r, m = candidate
        optim = torch.optim.SGD(model.parameters(), lr=l, weight_decay=r, momentum=m)
        losses, train_acc, precision = train(model, optim, epochs, Y)
        if train_acc[-1] > max_acc:
            best_params = (l, r, m)
            max_acc = train_acc[-1]

    return max_acc, best_params



def plot_train_metrics(losses, train_acc, precision):
    """
    Plots training metrics returned by train.

    :param losses: list of losses by epoch.
    :param train_acc: list of train accuracy by epoch.
    :param precision: list of precision by epoch.
    """
    plt.figure(figsize=(10, 12))
    plt.subplot(311)
    plt.title('Loss')
    plt.plot(losses, color='c')
    plt.subplot(312)
    plt.title('Accuracy')
    plt.plot(train_acc, color='c')
    plt.subplot(313)
    plt.title('Precision')
    plt.plot(precision, color='c')
    plt.show()

def min_max_norm(A):
    """
    Min-Max normalization of an input
    :param A: Input to be normalized
    :return: Normalized input
    """
