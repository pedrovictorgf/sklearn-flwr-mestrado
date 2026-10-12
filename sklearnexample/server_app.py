"""sklearnexample: A Flower / PyTorch app."""

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from sklearnexample.metrics import MetricsCollector
from sklearnexample.task import Net

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    collector = MetricsCollector()
    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=fraction_evaluate,
        train_metrics_aggr_fn=collector.aggregate_train,
        evaluate_metrics_aggr_fn=collector.aggregate_evaluate,
    )

    # Start strategy, run FedAvg for `num_rounds`
    started_at = datetime.now(timezone.utc)
    collector.start()
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )
    finished_at = datetime.now(timezone.utc)

    # Every run gets its own output directory
    output_dir = Path(
        context.run_config["output-dir"]
        or f"results/{started_at.astimezone().strftime('%Y%m%d-%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_file = None
    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        model_file = "final_model.pt"
        torch.save(state_dict, output_dir / model_file)

    results = {
        "run_id": context.run_id,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "strategy": type(strategy).__name__,
        "run_config": dict(context.run_config),
        "rounds": collector.rounds,
        "final_model": model_file,
    }
    results_file = output_dir / "results.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {results_file.resolve()}")
