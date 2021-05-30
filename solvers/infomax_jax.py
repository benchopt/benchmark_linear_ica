from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import jax.numpy as jnp
    from jax import grad, jit, vmap


class Solver(BaseSolver):
    """Infomax with JAX."""
    name = 'infomax'

    install_cmd = 'conda'
    requirements = ['pip:jax', 'pip:jaxlib']

    parameters = {
        'gradient': [
            'euclidian',
            'relative'
        ],
        'step_init': [0.1]
    }
    parameter_template = "{gradient}"
    stop_strategy = 'callback'

    def set_objective(self, X, A):
        self.X = jnp.array(X)
        self.A = jnp.array(A)

    def run(self, callback):

        X = self.X

        @jit
        def nll(W, x):
            return (
                jnp.log(jnp.cosh(W @ x)).sum() -
                jnp.log(abs(jnp.linalg.det(W)))
            )

        batched_nll = vmap(nll, in_axes=(None, 0))

        @jit
        def loss(W, X):
            return jnp.mean(batched_nll(W, X.T))

        grad_loss = jit(grad(loss, 0))

        p = self.X.shape[0]
        W = jnp.eye(p)

        while callback(W):
            if self.gradient == 'relative':
                W -= self.step_size * grad_loss(W, X).dot(W.T).dot(W)
            else:
                W -= self.step_size * grad_loss(W, X)

        self.W = W

    def get_result(self):
        return np.array(self.W)
