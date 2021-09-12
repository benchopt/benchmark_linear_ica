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
        # XXX : here we fix the seed of picard to keep it deterministic
        # but it hides the randomness due to the initialization choice
        K, W, _ = picard(self.X, max_iter=n_iter + 1, tol=1e-12,
                         random_state=42)
        self.W = W @ K

    def get_result(self):
        return self.W
