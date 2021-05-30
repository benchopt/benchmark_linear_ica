import numpy as np

from benchopt import BaseDataset


class Dataset(BaseDataset):

    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, n_features': [
            (10_000, 5),
            (100_000, 20)]
    }

    def __init__(self, n_samples=1000, n_features=2, random_state=27):
        # Store the parameters of the dataset
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def get_data(self):

        rng = np.random.RandomState(self.random_state)
        A = rng.randn(self.n_features, self.n_features)
        S = rng.laplace(size=(self.n_features, self.n_samples))
        X = A @ S
        data = dict(A=A, X=X)

        return (self.n_features, self.n_features), data
