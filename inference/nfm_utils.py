import numpy as np
import matplotlib.pyplot as plt

class NMF():

    def __init__(self, R, name, epsilon=1e-3, alpha=0.01, beta=100, verbose=False, max_k = 10):
        '''
        :param R: n x m binary matrix of results (n students, m questions)
        :param name: file name from which R is gotten from, for plotting/printing purposes, s.a. 'Floored Exp Uncorrel';
        :param epsilon: stopping parameter for NMF
        :param alpha: regularization for H
        :param beta: regularization for W
        :param verbose: print useful messages
        :param max_k: number of k iterations, where k is the number of skills we predict
        '''
        self.R = R.T
        self.name = name
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose
        self.max_k = max_k
        self.sols = None

    def run_all(self):
        """
        Run NFM, plot, show relevant graphs
        :return: solutions from run()
        """
        self.sols = self.run()
        self.get_k_with_norm(plot=True, solutions=self.sols)
        return self.sols

    def print_rms_results(self, k, true_students, true_questions, round_to=3):
        """
        Must be run after run()
        :param k: true kk
        :param true_questions: m x k matrix
        :param true_students: n x k matrix
        """
        if self.sols is None:
            print("Must call run() first.")
        else:
            assert(self.sols[k][1].shape == true_questions.shape)
            assert(self.sols[k][2].shape == true_students.T.shape)
            rms_w = round(self.rms(self.sols[k][1] - true_questions), round_to)
            rms_h = round(self.rms(self.sols[k][2] - true_students.T), round_to)
            print("RMS for Students (H): {}, RMS for Questions (W): {}".format(rms_h, rms_w))

    def rms(self, M):
        return np.sqrt(np.mean(np.square(M)))

    def run(self):
        '''
        Runs NMF on R
        :return: dict of length self.max_k, where ret[k] is a tuple of (Rk, Wk, Hk),
        where Wk and Hk are factored out from R.
        Wk: m x k matrix (m questions, k skills)
        Hk: n x k (n students, k skills)
        where Rk the resulting matrix from np.dot(Wk, Hk).
        '''
        m, n = self.R.shape
        solutions = {}
        for k in range(1, self.max_k + 1):
            W = np.random.rand(m, k)
            H = np.random.rand(k, n)
            i = 0
            while True:
                # Solve for H
                LH = np.dot(W.T, W) + np.sqrt(self.alpha) * np.identity(k)
                RH = np.dot(W.T, self.R)
                H_new = np.dot(np.linalg.pinv(LH), RH)
                H_new[H_new < 0] = 0
                H_new[H_new > 1] = 1  # TODO: decide if we want to cap at 1 at all, if so here or right before breaking, same for W

                # Solve for W
                LH = np.dot(H_new, H_new.T) + np.sqrt(self.beta)
                RH = np.dot(self.R, H_new.T)
                W_new = np.dot(RH, np.linalg.pinv(LH))
                W_new[W_new < 0] = 0
                W_new[W_new > 1] = 1  # TODO: decide if we want to cap at 1, see above

                if self.rms(W_new - W) < self.epsilon and self.rms(H_new - H) < self.epsilon:
                    solutions[k] = (np.dot(W_new, H_new), W_new, H_new)
                    if self.verbose: print("Iteration {} done.".format(k))
                    break

                W = W_new
                H = H_new

                if i > 1000:
                    W = np.random.rand(m, k)
                    H = np.random.rand(k, n)
                    i = 0
                    if self.verbose: print("W dif: {}, H dif: {}".format(self.rms(W_new - W), self.rms(H_new - H)))

                i += 1
        self.sols = solutions
        return self.sols


    def get_k_with_norm(self, plot=False, solutions=None):
        """

        :param plot: Plot graph of all k's
        :param solutions:
        :return:
        """
        if solutions is None:
            solutions = self.run()
        best = (None, float('Inf'))
        results = []
        for k, v in solutions.items():
            Rk = v[0]
            dif = np.linalg.norm(self.R - Rk)
            results.append((k, dif))
            if dif < best[1]:
                best = (k, dif)
        if self.verbose: print("Best k with RMS: {}".format(best[0]))
        if plot: self.plot(results, 'Results with Norm for {}'.format(self.name), 'k', 'RMS(Rk - R)')


    def plot(self, tuples, title, xlabel, ylabel):
        """
        Plots list of x, y tuples
        :param tuples: list of x, y tuples
        :param title: title for plot
        :param xlabel: x label for plot
        :param ylabel: y label for plot
        :return:
        """
        plt.plot(*zip(*tuples))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()


