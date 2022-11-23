from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
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
    min_benchopt_version = "1.3"
    name = "Amari Distance"
    is_convex = False

    def set_data(self, A, X):
        self.A = A
        self.X = X

    def compute(self, W):
        return amari_distance(W, self.A)

    def get_objective(self):
        return dict(X=self.X, A=self.A)
