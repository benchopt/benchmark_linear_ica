import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from benchopt import BaseSolver


def logcosh(Y):
    return np.abs(Y) + np.log1p(np.exp(-2. * np.abs(Y)))


class Solver(BaseSolver):
    """L-BFGS"""
    name = 'lbfgs'

    install_cmd = 'conda'
    requirements = ['scipy']

    def set_objective(self, X, A):
        self.X = X
        self.A = A

    def run(self, n_iter):
        # whitening
        u, d, _ = np.linalg.svd(self.X, full_matrices=False)
        del _
        K = (u / d).T
        del u, d
        p, _ = K.shape
        K *= np.sqrt(p)

        def loss_and_grad(w, X):
            p, n = X.shape
            W = w.reshape(p, p)
            Y = np.dot(W, X)
            loss = - np.linalg.slogdet(W)[1] + np.mean(logcosh(Y)) * p
            grad = np.dot(np.tanh(Y), X.T) / n - np.linalg.inv(W).T
            return loss, grad.ravel()

        w, *_ = fmin_l_bfgs_b(loss_and_grad, np.eye(p).ravel(),
                              args=(np.dot(K, self.X),), maxfun=10 * n_iter)

        W = w.reshape(p, p)
        self.W = W @ K

    def get_result(self):
        return {'W': self.W}
