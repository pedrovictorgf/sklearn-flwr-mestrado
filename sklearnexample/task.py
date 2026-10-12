"""sklearnexample: A Flower / PyTorch app."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Mapeamento direto e leve dos arquivos por cliente
CLIENT_DATASETS = { 0: "sklearnexample/partitions/client_0.parquet",
                    1: "sklearnexample/partitions/client_1.parquet",
                    2: "sklearnexample/partitions/client_2.parquet",
                    3: "sklearnexample/partitions/client_3.parquet",}

TARGET_COLUMN = "binary"
N_FEATURES = 70
N_CLASSES = 2


class Net(nn.Module):
    """1D CNN over the flow features, treated as a single-channel sequence."""

    def __init__(self):
        super(Net, self).__init__()
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


def get_data_path(node_config) -> str:
    """Return the parquet file of the node.

    In deployment each node sets `data-path` in its node config. In simulation it
    is not set, so the file is looked up by `partition-id` in CLIENT_DATASETS.
    """
    file_path = node_config.get("data-path")
    if file_path is not None:
        return file_path

    partition_id = node_config["partition-id"]
    file_path = CLIENT_DATASETS.get(partition_id)
    if file_path is None: raise ValueError(
        f"O partition_id {partition_id} não possui um dataset associado em CLIENT_DATASETS.")
    return file_path


def load_data(file_path: str, batch_size: int):
    """Load the data from the given parquet file."""
    parquet_file = pq.ParquetFile(file_path)
    feature_names = [
        name for name in parquet_file.schema_arrow.names if name != TARGET_COLUMN
    ]

    # Split the on-edge data: 80% train, 20% test (fixed seed so that train and
    # evaluate always see the same split)
    indices = np.random.default_rng(42).permutation(parquet_file.metadata.num_rows)
    split = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]

    y = parquet_file.read(columns=[TARGET_COLUMN]).column(0).to_numpy()
    y_train = y[train_idx].astype(np.int64)
    y_test = y[test_idx].astype(np.int64)

    # Read, scale and store one feature at a time, so that the only large arrays
    # ever allocated are the final float32 ones. Each feature is scaled with the
    # statistics of the local train split, in float64 because the timestamp
    # features (~1e12) do not fit the precision of float32.
    X_train = np.empty((len(train_idx), len(feature_names)), dtype=np.float32)
    X_test = np.empty((len(test_idx), len(feature_names)), dtype=np.float32)
    for i, name in enumerate(feature_names):
        column = np.log1p(parquet_file.read(columns=[name]).column(0).to_numpy())
        train_column = column[train_idx]
        mean = train_column.mean()
        std = train_column.std()
        if std == 0:
            std = 1.0
        X_train[:, i] = (train_column - mean) / std
        X_test[:, i] = (column[test_idx] - mean) / std
    parquet_file.close()
    # Arrow keeps freed decode buffers cached in its memory pool
    pa.default_memory_pool().release_unused()

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    return trainloader, testloader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for features, labels in trainloader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(features), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()
    loss = 0.0
    # Confusion matrix counts, where the positive class is the attack (label 1)
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    with torch.no_grad():
        for features, labels in testloader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            predictions = torch.max(outputs.data, 1)[1]
            counts["tp"] += ((predictions == 1) & (labels == 1)).sum().item()
            counts["fp"] += ((predictions == 1) & (labels == 0)).sum().item()
            counts["tn"] += ((predictions == 0) & (labels == 0)).sum().item()
            counts["fn"] += ((predictions == 0) & (labels == 1)).sum().item()
    loss = loss / len(testloader)
    return loss, counts
