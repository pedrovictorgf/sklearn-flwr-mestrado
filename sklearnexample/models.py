"""sklearnexample: model architectures and the registry that selects them."""

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn
import torch.nn.functional as F

N_FEATURES = 70
N_CLASSES = 2


class CNN(nn.Module):
    """1D CNN over the flow features, treated as a single-channel sequence."""

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        # Two poolings: 70 -> 35 -> 17
        self.fc1 = nn.Linear(32 * (N_FEATURES // 4), 64)
        self.fc2 = nn.Linear(64, N_CLASSES)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, features) -> (batch, 1, features)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MLP(nn.Module):
    """One hidden layer of 100 units, the default of scikit-learn's MLPClassifier."""

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(N_FEATURES, 100)
        self.fc2 = nn.Linear(100, N_CLASSES)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class LogisticRegression(nn.Module):
    """Multinomial logistic regression: one linear layer over the flow features."""

    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(N_FEATURES, N_CLASSES)

    def forward(self, x):
        return self.linear(x)


class LinearSVM(nn.Module):
    """Linear SVM: one linear layer, trained with hinge loss and an L2 penalty."""

    def __init__(self):
        super(LinearSVM, self).__init__()
        self.linear = nn.Linear(N_FEATURES, N_CLASSES)

    def forward(self, x):
        return self.linear(x)


@dataclass(frozen=True)
class ModelSpec:
    """How to build a model and the objective it is trained with."""

    build: Callable[[], nn.Module]
    criterion: Callable[[], nn.Module] = nn.CrossEntropyLoss
    weight_decay: float = 0.0


MODELS = {
    "cnn": ModelSpec(CNN),
    "mlp": ModelSpec(MLP),
    "logreg": ModelSpec(LogisticRegression),
    # The L2 penalty is part of the SVM objective, hence the weight decay
    "svm": ModelSpec(LinearSVM, criterion=nn.MultiMarginLoss, weight_decay=1e-4),
}


def get_model_spec(name: str) -> ModelSpec:
    """Return the spec registered under `name`."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available models: {sorted(MODELS)}")
    return MODELS[name]
