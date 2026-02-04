from benchopt import BaseSolver
from benchopt.stopping_criterion import (
    NoCriterion
)

import torch
import torch.nn as nn


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'torch-logreg'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'
    # and are set to one value of the list.
    parameters = {
        'lr': [0.001],
        'batch_size': [64],
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py.
    requirements = ['pytorch:pytorch']

    # TODO: change with maxmimizing criterion
    # stopping_criterion = SufficientProgressCriterion(
    #     strategy="iteration", eps=1e-4, patience=5,
    #     key_to_monitor="accuracy_test"
    # )
    stopping_criterion = NoCriterion()

    def set_objective(self, X_train, y_train):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.clf = nn.Sequential(
            nn.Linear(X_train.shape[1], 2),
            nn.Softmax(dim=1)
        )
        self.dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)
            ),
            batch_size=self.batch_size,
            shuffle=True
        )

    def run(self, n_iter):
        # This is the method that is called to fit the model.
        optim = torch.optim.SGD(self.clf.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(n_iter):
            for X_batch, y_batch in self.dataloader:
                optim.zero_grad()
                y_pred = self.clf(X_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optim.step()

    def get_result(self):
        # Returns the model after fitting.
        # The output of this function is a dictionary whose keys define the
        # keyword arguments for `Objective.evaluate_result`.
        # This defines the benchmark's API for solvers' results.
        # It is customizable for each benchmark.
        def predict(X):
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                y_pred = self.clf(X_tensor)
            return torch.argmax(y_pred, axis=1).numpy()

        return dict(model=predict)
