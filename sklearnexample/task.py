"""sklearnexample: A Flower / PyTorch app."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, TensorDataset

# Mapeamento direto e leve dos arquivos por cliente
CLIENT_DATASETS = { 0: "sklearnexample/partitions/client_0.parquet",
                    1: "sklearnexample/partitions/client_1.parquet",
                    2: "sklearnexample/partitions/client_2.parquet",
                    3: "sklearnexample/partitions/client_3.parquet",}

TARGET_COLUMN = "binary"


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


def limit_threads(node_config) -> None:
    """Use the number of threads the node was given, if any.

    PyTorch starts one thread per core of the host, so clients sharing a machine
    slow each other down unless each one is told how many threads it may use.
    """
    num_threads = node_config.get("num-threads")
    if num_threads:
        torch.set_num_threads(num_threads)


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


def train(net, criterion, trainloader, epochs, lr, device, weight_decay=0.0):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = criterion.to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )
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


def test(net, criterion, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()
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
