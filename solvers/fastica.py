import warnings
from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.decomposition import fastica
    from sklearn.exceptions import ConvergenceWarning


class Solver(BaseSolver):
    """FastICA."""
    name = 'fastica'

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    def set_objective(self, X, A):
        self.X = X
        self.A = A

    def run(self, n_iter):
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        K, W, _ = fastica(self.X.T, max_iter=n_iter + 1, tol=1e-12,
                          compute_sources=False)
        self.W = W @ K

    def get_result(self):
        return {'W': self.W}
