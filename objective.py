from benchopt import BaseObjective

from sklearn.model_selection import KFold
from sklearn.dummy import DummyClassifier


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "Template benchmark"

    # URL of the main repo for this benchmark.
    url = "https://github.com/benchopt/#BENCHMARK"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.
    parameters = {
        'random_state': [32],
    }

    # List of packages needed to run the benchmark.
    # They are installed with conda; to use pip, use 'pip:packagename'. To
    # install from a specific conda channel, use 'channelname:packagename'.
    # Packages not necessary to the whole benchmark but only to some
    # solvers or datasets should be declared in Dataset or Solver (see
    # simulated.py and python-gd.py).
    # Example syntax: requirements = ['numpy', 'pip:jax', 'pytorch:pytorch']
    requirements = ['scikit-learn']

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.8"

    # Disable performance curves - each solver runs once to completion
    # See https://benchopt.github.io/stable/user_guide/performance_curves.html
    # for more details.
    sampling_strategy = "run_once"

    def set_data(self, X, y):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.X, self.y = X, y

        # Specify a cross-validation splitter as the `cv` attribute.
        # This will be automatically used in `self.get_split` to split
        # the arrays provided.
        self.cv = KFold(
            n_splits=5, shuffle=True, random_state=self.random_state
        )

        # If the cross-validation requires some metadata, it can be
        # provided in the `cv_metadata` attribute. This will be passed
        # to `self.cv.split` and `self.cv.get_n_splits`.
        self.cv_metadata = {}

    def evaluate_result(self, model):
        # The keyword arguments of this function are the keys of the
        # dictionary returned by `Solver.get_result`. This defines the
        # benchmark's API to pass the solvers' result. This can be
        # customized for each benchmark.
        #
        # Here, the solver returns a trained model,
        # with which we can call ``score`` to get the accurcay.
        accuracy_train = model.score(self.X_train, self.y_train)
        accuracy_test = model.score(self.X_test, self.y_test)

        # This method can return many metrics in a dictionary.
        return dict(
            accuracy_test=accuracy_test,
            accuracy_train=accuracy_train,
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        clf = DummyClassifier()
        clf.fit(self.X_train, self.y_train)
        return dict(model=clf)

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The keys of this method's output dictionary are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # This can be customized in each benchmark.

        self.X_train, self.X_test, self.y_train, self.y_test = self.get_split(
            self.X, self.y
        )

        return dict(
            X_train=self.X_train,
            y_train=self.y_train,
        )
