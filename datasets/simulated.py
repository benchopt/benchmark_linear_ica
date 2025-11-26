import numpy as np

from benchopt import BaseDataset


class Dataset(BaseDataset):
    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, n_features, n_gaussians': [
            (10_000, 5, 0),
            (10_000, 100, 50),
            (10_000, 200, 100)
        ],
    }

    def __init__(self, n_samples=1000, n_features=2, n_gaussians=1,
                 random_state=27, whiten=True):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_gaussians = n_gaussians
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        A = rng.randn(self.n_features, self.n_features)
        S = rng.laplace(size=(self.n_features, self.n_samples))
        if self.n_gaussians:
            S[:self.n_gaussians] = rng.randn(self.n_gaussians, self.n_samples)
        X = A @ S

        if self.whiten:


        return dict(A=A, X=X)
