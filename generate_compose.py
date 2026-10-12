"""Generate the Docker Compose deployment of an experiment.

Usage: python generate_compose.py <experiment.yaml> <run directory>

Reads the experiment config and writes into the run directory:
  compose.yaml     one SuperLink, one SuperNode per client and their SuperExecs
  experiment.yaml  copy of the config that was used
  run_info.json    git commit, clients and run config

It prints the value to pass to `flwr run --run-config`.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent

FLWR_VERSION = "1.39.0"
SUPEREXEC_IMAGE = "flwr_superexec:0.0.1"
SUPERLINK_FLEET_PORT = 9092
SUPERLINK_RUNTIME_PORT = 8000  # also published to the host, for `flwr run`
SUPERNODE_PORT = 9094
CONTAINER_DATA_PATH = "/data/client.parquet"
CONTAINER_RESULTS_DIR = "/results"
MIN_CLIENTS = 2  # minimum number of nodes required by the strategy


def load_config(config_file: Path) -> dict:
    config = yaml.safe_load(config_file.read_text())
    clients = config.get("clients") or []
    if len(clients) < MIN_CLIENTS:
        sys.exit(f"{config_file}: at least {MIN_CLIENTS} clients are required.")
    for client in clients:
        data_file = (PROJECT_DIR / client["data"]).resolve()
        if not data_file.is_file():
            sys.exit(f"{config_file}: data file not found: {data_file}")
        client["data"] = str(data_file)
    config["run-config"] = config.get("run-config") or {}
    return config


def build_compose(config: dict, run_dir: Path) -> dict:
    superexec = {"image": SUPEREXEC_IMAGE, "restart": "on-failure"}
    services = {
        "superlink": {
            "image": f"flwr/superlink:{FLWR_VERSION}",
            "command": ["--insecure", "--host", "0.0.0.0", "--isolation", "process"],
            "ports": [f"{SUPERLINK_RUNTIME_PORT}:{SUPERLINK_RUNTIME_PORT}"],
        },
        "superexec-serverapp": {
            **superexec,
            "build": {
                "context": str(PROJECT_DIR),
                "dockerfile": "superexec.Dockerfile",
            },
            "command": [
                "--insecure",
                "--runtime-api-address",
                f"superlink:{SUPERLINK_RUNTIME_PORT}",
            ],
            "volumes": [f"{run_dir}:{CONTAINER_RESULTS_DIR}"],
            "depends_on": ["superlink"],
        },
    }
    for partition_id, client in enumerate(config["clients"]):
        number = partition_id + 1
        services[f"supernode-{number}"] = {
            "image": f"flwr/supernode:{FLWR_VERSION}",
            "restart": "on-failure",
            "command": [
                "--insecure",
                "--superlink",
                f"superlink:{SUPERLINK_FLEET_PORT}",
                "--node-config",
                f"partition-id={partition_id} data-path='{CONTAINER_DATA_PATH}'",
                "--host",
                "0.0.0.0",
                "--port",
                str(SUPERNODE_PORT),
                "--isolation",
                "process",
            ],
            "depends_on": ["superlink"],
        }
        services[f"superexec-clientapp-{number}"] = {
            **superexec,
            "command": [
                "--insecure",
                "--runtime-api-address",
                f"supernode-{number}:{SUPERNODE_PORT}",
            ],
            # Each client only sees its own data
            "volumes": [f"{client['data']}:{CONTAINER_DATA_PATH}:ro"],
            "depends_on": [f"supernode-{number}", "superexec-serverapp"],
        }
    return {"services": services}


def format_run_config(run_config: dict) -> str:
    """Format the run config as the key-value pairs expected by `flwr run`."""
    pairs = []
    for key, value in run_config.items():
        if isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, str):
            value = f"'{value}'"
        pairs.append(f"{key}={value}")
    return " ".join(pairs)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_DIR, capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    config_file, run_dir = Path(sys.argv[1]), Path(sys.argv[2]).resolve()

    config = load_config(config_file)
    run_config = {**config["run-config"], "output-dir": CONTAINER_RESULTS_DIR}

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "compose.yaml").write_text(
        yaml.safe_dump(build_compose(config, run_dir), sort_keys=False)
    )
    shutil.copyfile(config_file, run_dir / "experiment.yaml")
    run_info = {
        "run_name": run_dir.name,
        "experiment": config.get("name"),
        "git_commit": git("rev-parse", "HEAD") or None,
        "git_uncommitted_files": len(git("status", "--porcelain").splitlines()),
        "flwr_version": FLWR_VERSION,
        "superexec_image": SUPEREXEC_IMAGE,
        "clients": [
            {
                "name": f"supernode-{partition_id + 1}",
                "partition_id": partition_id,
                "data_file": client["data"],
            }
            for partition_id, client in enumerate(config["clients"])
        ],
        "run_config": run_config,
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))

    print(format_run_config(run_config))


if __name__ == "__main__":
    main()
