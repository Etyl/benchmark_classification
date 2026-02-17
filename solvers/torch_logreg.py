from benchopt import BaseSolver
from benchopt.stopping_criterion import (
    SufficientProgressCriterion
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
        'lr': [0.01],
        'batch_size': [64],
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py.
    requirements = ['pytorch:pytorch']

    stopping_criterion = SufficientProgressCriterion(
        strategy="callback", eps=1e-3, patience=3,
        key_to_monitor="accuracy_test", minimize=False
    )

    def set_objective(self, X_train, y_train):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.clf = nn.Linear(X_train.shape[1], 2)
        self.dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)
            ),
            batch_size=self.batch_size,
            shuffle=True
        )

    def get_next(self, stop_val):
        # This controls when the solver is evaluated, which is used by
        # the stopping criterion.
        return stop_val + 50

    def run(self, callback):
        # This is the method that is called to fit the model.
        optim = torch.optim.SGD(self.clf.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        max_epochs = 100
        for _ in range(max_epochs):
            for X_batch, y_batch in self.dataloader:
                optim.zero_grad()
                y_pred = self.clf(X_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optim.step()

                if not callback():
                    return

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

        return dict(predict=predict)
