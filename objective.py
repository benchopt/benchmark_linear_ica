from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from picard import amari_distance


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
