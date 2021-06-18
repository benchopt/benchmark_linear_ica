from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from picard import amari_distance


class Objective(BaseObjective):
    name = "Amari Distance"

    def set_data(self, A, X):
        self.A = A
        self.X = X

    def compute(self, W):
        return amari_distance(W, self.A)

    def to_dict(self):
        return dict(X=self.X, A=self.A)
