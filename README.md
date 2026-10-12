---
tags: [quickstart, tabular, fds]
dataset: [GenIDS-UNSW15, GenIDS-CIC17]
framework: [torch]
---

# Federated Learning with PyTorch and Flower (1D CNN on network flow data)

This Flower app trains a 1D convolutional neural network with `PyTorch` to detect attacks in network flows
(binary classification over 70 flow features). Each client reads its own pre-generated partition from
`sklearnexample/partitions/`, created by `sklearnexample/preprocess.py` from the GenIDS-UNSW15 and GenIDS-CIC17
datasets, and standardizes the features with its local statistics.

## Set up the project

### Fetch the app

Install Flower:

```shell
pip install flwr
```

Fetch the app:

```shell
flwr new @flwrlabs/sklearn-flwr-mestrado-2
```

This will create a new directory called `sklearn-flwr-mestrado-2` with the following structure:

```shell
sklearn-flwr-mestrado-2
├── sklearnexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training and data loading
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md
```

### Install dependencies and project

Install the dependencies defined in `pyproject.toml` as well as the `sklearnexample` package.

```bash
pip install -e .
```

## Run the project

You can run your Flower project in both _simulation_ and _deployment_ mode without making changes to the code. If you are starting with Flower, we recommend you using the _simulation_ mode as it requires fewer components to be launched manually. By default, `flwr run` will make use of the Simulation Engine.

### Run with the Simulation Engine

> [!NOTE]
> Check the [Simulation Engine documentation](https://flower.ai/docs/framework/how-to-run-simulations.html) to learn more about Flower simulations and how to optimize them.

```bash
flwr run . --stream
```

You can also override some of the settings for your `ClientApp` and `ServerApp` defined in `pyproject.toml`. For example:

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05" --stream
```

### Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be intersted in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

If you are already familiar with how the Deployment Engine works, you may want to learn how to run it using Docker. Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.
# sklearn-flwr-mestrado
