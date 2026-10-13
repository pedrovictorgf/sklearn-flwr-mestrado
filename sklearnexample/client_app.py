"""sklearnexample: A Flower / PyTorch app."""

import time

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from sklearnexample.models import get_model_spec
from sklearnexample.task import get_data_path, limit_threads, load_data
from sklearnexample.task import test as test_fn
from sklearnexample.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    start_time = time.perf_counter()
    limit_threads(context.node_config)

    # Load the model and initialize it with the received weights
    spec = get_model_spec(context.run_config["model"])
    model = spec.build()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    data_path = get_data_path(context.node_config)
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(data_path, batch_size)

    # Call the training function
    train_loss = train_fn(
        model,
        spec.criterion(),
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
        weight_decay=spec.weight_decay,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "partition-id": context.node_config.get("partition-id", -1),
        "duration_seconds": time.perf_counter() - start_time,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    start_time = time.perf_counter()
    limit_threads(context.node_config)

    # Load the model and initialize it with the received weights
    spec = get_model_spec(context.run_config["model"])
    model = spec.build()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    data_path = get_data_path(context.node_config)
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(data_path, batch_size)

    # Call the evaluation function
    eval_loss, counts = test_fn(
        model,
        spec.criterion(),
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        **counts,
        "num-examples": len(valloader.dataset),
        "partition-id": context.node_config.get("partition-id", -1),
        "duration_seconds": time.perf_counter() - start_time,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
