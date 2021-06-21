import warnings
from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from mne.preprocessing.infomax_ import infomax


class Solver(BaseSolver):
    """Infomax."""
    name = 'infomax'

    install_cmd = 'pip'
    requirements = ['mne']

    def set_objective(self, X, A):
        self.X = X
        self.A = A

    def run(self, n_iter):
        # whitening
        u, d, _ = np.linalg.svd(self.X, full_matrices=False)
        del _
        K = (u / d).T
        del u, d
        K *= np.sqrt(K.shape[0])
        W = infomax(np.dot(K, self.X).T, max_iter=n_iter + 1, verbose=False,
                    random_state=0)
        self.W = W @ K

    def get_result(self):
        return self.W
