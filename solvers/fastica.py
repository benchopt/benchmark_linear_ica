import warnings

from sklearn.decomposition import fastica
from sklearn.exceptions import ConvergenceWarning

from benchopt import BaseSolver


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
        # XXX : here we fix the seed of fastica to keep it deterministic
        # but it hides the randomness due to the initialization choice
        K, W, _ = fastica(self.X.T, max_iter=n_iter + 1, tol=1e-12,
                          compute_sources=False, random_state=42)
        self.W = W @ K

    def get_result(self):
        return {'W': self.W}
