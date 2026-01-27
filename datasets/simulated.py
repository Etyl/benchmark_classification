from benchopt import BaseDataset

from sklearn.datasets import make_classification


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'n_samples, n_features': [
            (1000, 100),
            (5000, 10),
        ],
        'random_state': [27],
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    requirements = []

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.
        #
        # Data splitting is handled by the `Objective.get_objective` method
        # and `Objective.cv` property

        # Generate pseudorandom data using `sklearn` for classification.
        # Generating synthetic dataset
        X, y = make_classification(
            n_samples=self.n_samples, n_features=self.n_features,
            n_informative=1, n_redundant=0, n_clusters_per_class=1,
            random_state=self.random_state
        )

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(X=X, y=y)
