from benchopt import BaseObjective

import numpy as np


def amari_distance(W, A):
    """
    Computes the Amari distance between two matrices W and A.
    It cancels when WA is a permutation and scale matrix.

    Parameters
    ----------
    W : ndarray, shape (n_features, n_features)
        Input matrix

    A : ndarray, shape (n_features, n_features)
        Input matrix

    Returns
    -------
    d : float
        The Amari distance
    """
    P = np.dot(W, A)

    def s(r):
        return np.sum(np.sum(r ** 2, axis=1) / np.max(r ** 2, axis=1) - 1)
    return (s(np.abs(P)) + s(np.abs(P.T))) / (2 * P.shape[0])


class Objective(BaseObjective):
    name = "Amari Distance"
    is_convex = False
    url = "https://github.com/benchopt/benchmark_linear_ica"

    min_benchopt_version = "1.7"  # could probably be relaxed

    def set_data(self, A, X):
        self.A = A
        self.X = X

    def evaluate_result(self, W):
        return {'value': amari_distance(W, self.A)}

    def get_one_result(self):
        return {'W': np.eye(self.A.shape[0])}

    def get_objective(self):
        return {'A': self.A, 'X': self.X}
