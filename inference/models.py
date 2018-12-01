import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np

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
        print(concepts)
        _D[concepts] = 0.1 * torch.randn(num_questions)
        print(_D[concepts])
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



    
class RNN_Model(nn.Module):

    def __init__(self, hidden_size, num_students=1000, num_layers=8):
        """
        RNN: linear -> LSTM -> linear -> sigmoid
        :param hidden_size: size for hidden layer of linear layer
        :param num_students: number of students
        :param num_layers: number of layers for lstm step
        """
        super(RNN_Model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.inp = nn.Linear(num_students, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=0.05)  # TODO: hyperparam dropout?, other?
        self.lin = nn.Linear(hidden_size, num_students)
        self.sig = nn.Sigmoid()
        self.num_students = num_students


    def step(self, input, hidden=None):
        input = self.inp(input.view(1, -1)).unsqueeze(1)
        output, hidden = self.lstm(input, hidden)
        output = self.lin(output.squeeze(1))
        output = self.sig(output)
        return output, hidden


    def forward(self, inputs, hidden=None, steps=0):
        """
        :param inputs: Q x N matrix (questions x students)
        """
        if steps == 0: steps = len(inputs)
        num_students = inputs.size()[1]
        outputs = Variable(torch.zeros(steps, num_students))
        for i in range(steps):
            if i == 0:
                input = inputs[i]
            else:
                input = output
            output, hidden = self.step(input, hidden)
            outputs[i] = output
        return outputs, hidden


class FE(nn.Module):

    def forward(self, A, D, l, concepts, guess_prob=0.25):
        """
        Calculates model predictions. Notice x, is not used. That is, our predictions don't depend on any input
        features. This is because this is really an unsupervised learning problem, we only have access to unlabeled data
        (the test results), and we are trying to infer matrices A and D from it, which we then use to
        predict the test results.

        :param x: Not used. Required by torch architecture.
        :return: Predictions for the entire batch.
        """
        return floored_exp_irf(A[:, concepts[1]], D[concepts], l, guess_prob)


class RNN_Skills_Model(nn.Module):

    def __init__(self, concepts, num_concepts, num_questions, hidden_size, num_students=1000, num_layers=8):
        """
        RNN: linear -> LSTM -> linear -> sigmoid
        :param concepts: tuple as returned by np.nonzero on questions matrix to get indeces of concepts.
        :param hidden_size: size for hidden layer of linear layer
        :param num_students: number of students
        :param num_layers: number of layers for lstm step
        """
        self.concepts = concepts
        self.num_questions = num_questions
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_students = num_students
        self.num_concepts = num_concepts

        super(RNN_Skills_Model, self).__init__()
        _D = torch.zeros(self.num_questions, self.num_concepts)
        _D[concepts] = 0.1 * torch.randn(self.num_questions)
        self.D = nn.Parameter(_D, requires_grad=True)
        self.l = torch.ones(self.num_students)

        self.inp = nn.Linear(num_students, hidden_size)
        self.lstm = nn.LSTM(hidden_size, self.num_concepts, num_layers, dropout=0.05)  # TODO: hyperparam dropout?, other?
        self.fe = FE()
        self.lin = nn.Linear(hidden_size, num_students)


    def step(self, input, hidden=None):
        input = self.inp(input.view(1, -1)).unsqueeze(1)
        output, hidden = self.lstm(input, hidden)
        skills = output.squeeze(1)
        print(self.D.size())
        print(skills.size())
        output = self.fe(skills, self.D, self.l, self.concepts)
        return output, hidden, skills


    def forward(self, inputs, hidden=None, steps=0):
        """
        :param inputs: Q x N matrix (questions x students)
        """
        if steps == 0: steps = len(inputs)
        num_students = inputs.size()[1]
        outputs = Variable(torch.zeros(steps, num_students))
        for i in range(steps):
            if i == 0:
                input = inputs[i]
            else:
                input = output
            output, hidden, skills = self.step(input, hidden)
            outputs[i] = output
        return outputs, hidden, skills
