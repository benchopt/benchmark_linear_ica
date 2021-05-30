import warnings
from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    from picard import picard


class Solver(BaseSolver):
    """PICARD."""
    name = 'picard'

    install_cmd = 'conda'
    requirements = ['python-picard']

    def set_objective(self, X, A):
        self.X = X
        self.A = A

    def run(self, n_iter):
        warnings.filterwarnings('ignore', category=UserWarning)
        K, W, _ = picard(self.X, max_iter=n_iter + 1, tol=1e-12)
        self.W = W @ K

    def get_result(self):
        return self.W
