import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from flwr.common import NDArrays
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearnexample.preprocess import (
    prepare_datasets, TARGET_COLUMN
)

# This information is needed to create a correct scikit-learn model
UNIQUE_LABELS = [0, 1]
FEATURES = ["petal_length", "petal_width", "sepal_length", "sepal_width"]


def get_model_params(model: MLPClassifier) -> NDArrays:
    """Return the parameters of a sklearn LogisticRegression model."""
    params = []

    for coef, intercept in zip(model.coefs_, model.intercepts_):
        params.append(coef)
        params.append(intercept)

    return params


def set_model_params(model: MLPClassifier, params: NDArrays) -> MLPClassifier:
    """Set the parameters of a sklean LogisticRegression model."""
    for i in range(len(model.coefs_)):
        model.coefs_[i] = params[2 * i]
        model.intercepts_[i] = params[2 * i + 1]
    return model


def set_initial_params(model: MLPClassifier, n_classes: int, n_features: int):
    """Set initial parameters as zeros.

    Required since model params are uninitialized until model.fit is called but server
    asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    X_mock = np.zeros((n_classes, n_features))
    y_mock = np.array([0, 1])
    model.partial_fit(X_mock, y_mock, classes=np.arange(n_classes))


def create_mlp_and_instantiate_parameters():
    model = MLPClassifier()
    set_initial_params(model, n_features=70, n_classes=2)
    return model

fds = None  # Cache FederatedDataset
partitioner = None

def load_data(partition_id: int, num_partitions: int):
    """Load the data for the given partition."""
    global fds
    global partitioner
    if partitioner is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        # df = pd.read_csv("//sklearnexample/GenIDS-UNSW15.csv")
        datasets = {"GenIDS-UNSW15": pd.read_csv("sklearnexample/GenIDS-UNSW15.csv").drop(columns=['multiclass'])}
        df = prepare_datasets(datasets, TARGET_COLUMN)
        fds = Dataset.from_pandas(df)
        # fds = FederatedDataset(
        #     dataset="hitorilabs/iris", partitioners={"train": partitioner}
        # )
        # fds =load_dataset("/home/pedrovictorgf/PycharmProjects/sklearn-flwr-mestrado-2/sklearnexample/", data_files=["GenIDS-UNSW15.csv"])
        if partitioner is not None:
            partitioner.dataset = fds

    # dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    dataset = partitioner.load_partition(partition_id).with_format("pandas")[:]
    # y = df['binary'].astype(str)  # string para o label encoder
    # X = df.drop(columns=['binary'])
    X = dataset.drop(columns=[TARGET_COLUMN])
    y = dataset[TARGET_COLUMN]
    # Split the on-edge data: 80% train, 20% test
    X_train, X_test = X[: int(0.8 * len(X))], X[int(0.8 * len(X)) :]
    y_train, y_test = y[: int(0.8 * len(y))], y[int(0.8 * len(y)) :]
    return X_train.values, y_train.values, X_test.values, y_test.values
