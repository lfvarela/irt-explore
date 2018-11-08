import torch
import torch.nn as nn
import torch.nn.functional as f

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


class Sigmoid_Model(nn.Module):
    def __init__(self, num_students, num_questions, num_concepts, concepts, guess_prob=0.2):
        """
        Initializes a two parameter sigmoid IRF model.

        :param num_students: The number of students per batch (usually the entire train set)
        :param num_questions: The number of questions.
        :param num_concepts: The number of concepts the questions test.
        :param concepts: A tensor of shape (num_questions, num_concepts), where a 1 in the (i,j) entry implies that
        question i tests concept j.
        :param guess_prob: The probability that a student gets a question right by guessing.
        """

        super(Sigmoid_Model, self).__init__()
        self.A = nn.Parameter(torch.randn(num_students, num_concepts), requires_grad=True)
        _D = torch.zeros(num_questions, num_concepts)
        _D[concepts] = torch.randn(num_questions)
        self.D = nn.Parameter(_D, requires_grad=True)
        self.concepts = concepts
        self.guess_prob = guess_prob

    def forward(self, x):
        """
        Calculates model predictions. Notice x, is not used. That is, our predictions don't depend on any input
        features. This is because this is really an unsupervised learning problem, we only have access to unlabeled data
        (the test results), and we are trying to infer matrices A and D from it, which we then use to
        predict the test results.

        :param x: Not used. Required by torch architecture.
        :return: Predictions for the entire batch.
        """
        return two_param_sigmoid_irf(self.A[:, self.concepts[1]], self.D[self.concepts], self.guess_prob)


class FE_Model(nn.Module):
    def __init__(self, num_students, num_questions, num_concepts, concepts, guess_prob=0.2, l=10):
        """
        Initializes a floored exponential IRF model.

        :param num_students: The number of students per batch (usually the entire train set)
        :param num_questions: The number of questions.
        :param num_concepts: The number of concepts the questions test.
        :param concepts: A tensor of shape (num_questions, num_concepts), where a 1 in the (i,j) entry implies that
        question i tests concept j.
        :param guess_prob: The probability that a student gets a question right by guessing.
        :param l:
        """

        super(FE_Model, self).__init__()
        self.A = nn.Parameter(0.1 * torch.randn(num_students, num_concepts), requires_grad=True)
        _D = torch.zeros(num_questions, num_concepts)
        _D[concepts] = 0.1 * torch.randn(num_questions)
        self.D = nn.Parameter(_D, requires_grad=True)
        self.concepts = concepts
        self.guess_prob = guess_prob
        self.l = l

    def forward(self, x):
        """
        Calculates model predictions. Notice x, is not used. That is, our predictions don't depend on any input
        features. This is because this is really an unsupervised learning problem, we only have access to unlabeled data
        (the test results), and we are trying to infer matrices A and D from it, which we then use to
        predict the test results.

        :param x: Not used. Required by torch architecture.
        :return: Predictions for the entire batch.
        """
        return floored_exp_irf(self.A[:, self.concepts[1]], self.D[self.concepts], self.l, self.guess_prob)