import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as f
import utils
from torch.autograd import Variable


def two_param_sigmoid_irf(a, d, c):
    return c + (1 - c) * torch.sigmoid(a - d)


def floored_exp_irf(a, d, l, c):
    return torch.max(torch.tensor(c), 1 - torch.exp(-l  * (a - d)))


def flatten(x):
    return x.view(-1)


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


class RNN_Model(nn.Module):

    def __init__(self, hidden_size, num_students, num_layers=8, dropout = 0.1):
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
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout)  # TODO: hyperparam dropout?, other?
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
        outputs = Variable(torch.zeros(steps, self.num_students))
        for i in range(steps):
            #if i == 0:
             #   input = inputs[i]
            #else:
                #input = output
            input = inputs[i] 
            output, hidden = self.step(input, hidden)
            outputs[i] = output
        return outputs, hidden


class FE(nn.Module):

    def forward(self, A, D, l, i,concepts, guess_prob=0.25):
        """
        Calculates model predictions. Notice x, is not used. That is, our predictions don't depend on any input
        features. This is because this is really an unsupervised learning problem, we only have access to unlabeled data
        (the test results), and we are trying to infer matrices A and D from it, which we then use to
        predict the test results.

        :param x: Not used. Required by torch architecture.
        :return: Predictions for the entire batch.
        """
        
        return floored_exp_irf(A[:, concepts[1][i]], D[concepts][i], l, guess_prob)

class SIG(nn.Module):

    def forward(self, A, D, i,concepts, guess_prob=0.25):
        """
        Calculates model predictions. Notice x, is not used. That is, our predictions don't depend on any input
        features. This is because this is really an unsupervised learning problem, we only have access to unlabeled data
        (the test results), and we are trying to infer matrices A and D from it, which we then use to
        predict the test results.

        :param x: Not used. Required by torch architecture.
        :return: Predictions for the entire batch.
        """
        
        return two_param_sigmoid_irf(A[:, concepts[1][i]], D[concepts][i], guess_prob)


class RNN_Skills_Model(nn.Module):

    def __init__(self, average, concepts, num_concepts, num_questions, hidden_size, num_students=1000, num_layers=8, dropout = 0.05, sigmoid = False):
        """
        RNN: linear -> LSTM -> linear -> FE or IRF Sigmoid
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
        self.average = average
        self.sigmoid = sigmoid

        super(RNN_Skills_Model, self).__init__()
        _D = torch.zeros(self.num_questions, self.num_concepts)
        _D[concepts] = 0.1 * torch.randn(self.num_questions)
        self.D = nn.Parameter(_D, requires_grad=True)
        self.l = 10#*torch.ones(self.num_students)

        self.inp = nn.Linear(num_students, hidden_size)
        self.lstm = nn.LSTM(hidden_size, self.num_students, num_layers, dropout=dropout)  # TODO: hyperparam dropout?, other?
        self.fe = FE()
        self.irf_sig = SIG()
        self.lin = nn.Linear(1, num_concepts)


    def step(self, input, i,hidden=None):
        input = self.inp(input.view(1, -1)).unsqueeze(1)
        output, hidden = self.lstm(input, hidden)
        skills = output.squeeze(1)
        skills = self.lin(skills.transpose(0,1))
        
        #skills = torch.clamp(skills, min = 0, max = 1)
        #self.D = torch.clamp(self.D, 0, 1)
        
        skills = (skills - skills.min())/(skills.max()-skills.min())
        #self.D = (self.D - self.D.min())/(self.D.max()-self.D.min())
        
        if self.sigmoid:
            output = self.irf_sig(skills, self.D, i, self.concepts)
        else:
            output = self.fe(skills, self.D, self.l,i, self.concepts)

        return output, hidden, skills


    def forward(self, inputs, hidden=None, steps=0):
        """
        :param inputs: Q x N matrix (questions x students)
        """
        if steps == 0: steps = len(inputs)
        num_students = inputs.size()[1]
        outputs = Variable(torch.zeros(steps, num_students))
        n_skills = Variable(torch.zeros(steps, num_students,self.num_concepts))
        for i in range(steps):
            #if i == 0:
            #    input = inputs[i]
            #else:
            #    input = output
            input = inputs[i]
            output, hidden, skills = self.step(input,i, hidden)
            outputs[i] = output
            n_skills[i] = skills
        if self.average:
            skills = n_skills.mean(dim=0)
        return outputs, hidden, skills, self.D


def modal_decomp(R):
    """
    Runs the modal decomposition scoring on a results matrix R. Assumes a single concept/skill is tested by the exam.

    :param R: A (num_students, num_questions) numpy array, holding the results of an exam

    :return: A skill vector holding a latent skill measure for every student, and a question vector holding a difficulty
    measure for every question.
    """
    ks0 = np.maximum(np.sum(R, axis=1), 1)
    kq0 = np.maximum(np.sum(R, axis=0), 1)
    Ms = R.dot((R / kq0).T) / ks0

    evals, evecs = np.linalg.eig(Ms)
    i = np.argsort(np.abs(evals))[-2]
    s = utils.min_max_normalize(np.real(evecs[:, i]))

    Mq = R.T.dot((R / np.expand_dims(ks0, axis=1))) / kq0
    evals, evecs = np.linalg.eig(Mq)
    i = np.argsort(np.abs(evals))[-2]
    q = utils.min_max_normalize(np.real(evecs[:, i]))

    return s, q





############################################### EXPERIMENTAL. DISREGARD ################################################

def lucky_questions(s, q, c):
    _s = np.expand_dims(s, axis=0)
    _q = np.expand_dims(q, axis=1)
    indices = _q > _s
    bias_num = c * np.sum(np.where(indices, _q, 0), axis=0)
    bias_denom = c * np.sum(indices, axis=0)
    return bias_num, bias_denom


def lucky_students(s, q, c):
    _s = np.expand_dims(s, axis=1)
    _q = np.expand_dims(q, axis=0)
    indices = _s <= _q
    bias_num = c * np.sum(np.where(indices, _s, 0), axis=0)
    bias_denom = c * np.sum(indices, axis=0)
    return bias_num, bias_denom


def iterative_scoring_1(R, c=0.2, threshold=1e-3):
    # Iterative implementation
    S, Q = R.shape
    diff = threshold + 1
    ks0 = np.maximum(np.sum(R, axis=1), 1)
    kq0 = np.maximum(np.sum(R, axis=0), 1)
    s_prev = ks0 / Q
    q_prev = (S - kq0) / S
    while diff > threshold:
        bias1, bias2 = lucky_questions(s_prev, q_prev, c)
        denom = np.maximum(ks0 - bias2, 1)
        s = utils.min_max_normalize((1 / denom) * (R.dot(q_prev) - bias1))
        bias1, bias2 = lucky_students(s_prev, q_prev, c)
        denom = np.maximum(kq0 - bias2, 1)
        q = utils.min_max_normalize((1 / denom) * (R.T.dot(s_prev) - bias1))
        diff = max(np.linalg.norm(s - s_prev), np.linalg.norm(q - q_prev))
        s_prev = s
        q_prev = q

    return s, q

def iterative_scoring_2(R, c=0.2, threshold=1e-3):
    S, Q = R.shape
    diff = threshold + 1
    ks0 = np.maximum(np.sum(R, axis=1), 1)
    kq0 = np.maximum(np.sum(R, axis=0), 1)
    s_prev = ks0 / Q
    q_prev = (S - kq0) / S
    while diff > threshold:
        s = utils.min_max_normalize((1 / ks0) * (R.dot(q_prev) + 1))
        q = utils.min_max_normalize((1 / kq0) * (R.T.dot(s_prev) + 1))
        diff = max(np.linalg.norm(s - s_prev), np.linalg.norm(q - q_prev))
        s_prev = s
        q_prev = q

    return s, q


def iterative_scoring_3(R, c=0.2, threshold=1e-3):
    S, Q = R.shape
    diff = threshold + 1
    ks0 = np.maximum(np.sum(R, axis=1), 1)
    kq0 = np.maximum(np.sum(R, axis=0), 1)
    s_prev = ks0 / Q
    q_prev = (S - kq0) / S
    while diff > threshold:
        s = utils.min_max_normalize((1 / ks0) * (R.dot(q_prev)))
        q = utils.min_max_normalize((1 / kq0) * (R.T.dot(s_prev)))
        diff = max(np.linalg.norm(s - s_prev), np.linalg.norm(q - q_prev))
        s_prev = s
        q_prev = q

    return s, q


# This class was taken from https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/3
# All credit to Adam Paszke
class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


# TODO: Think about adding bias
class NN_IRF(nn.Module):
    def __init__(self, num_students, num_questions, num_concepts, layers):
        """
        Initializes an NN IRF model.

        :param num_students: The number of students.
        :param num_questions: The number of questions.
        :param num_concepts: The number of concepts.
        :param layers: A list holding the number of hidden units for the ith layer. The list can hold a number p between
        0 and 1, exclusive, to signify a  dropout layer with drop probability p. Note layers[-1] should equal the output
        size.
        """
        super(NN_IRF, self).__init__()
        self.input_size = (num_students + num_questions)*num_concepts
        self.output_shape = (num_students, num_questions)

        self.A = nn.Parameter(0.1 * torch.randn(num_students, num_concepts), requires_grad=True)
        self.D = nn.Parameter(0.1 * torch.randn(num_questions, num_concepts), requires_grad=True)

        self.num_layers = len(layers)
        prev_out = self.input_size
        for i in range(self.num_layers):
            if layers[i] < 1:  # dropout layer
               self.add_module('l_' + str(i), nn.Dropout(layers[i]))
            else:
                self.add_module('l_' + str(i), nn.Linear(prev_out, layers[i]))
                prev_out = layers[i]
        self.layers = AttrProxy(self, 'l_')


    def forward(self, x):
        del x
        x = torch.cat((flatten(self.A), flatten(self.D)))
        for i in range(self.num_layers):
            x = f.relu(self.layers[i](x))
        x = f.sigmoid(x)
        # Now we reshape the output to fit the (num_students, num_questions) shape
        return x.view(self.output_shape)
