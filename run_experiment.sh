#!/usr/bin/env bash
# Start a local Flower deployment in Docker (one SuperLink, NUM_CLIENTS SuperNodes
# and their SuperExecs), run the app on it and tear everything down afterwards.
#
# Each run writes to results/<timestamp>/:
#   results.json     per-round metrics and run config (written by the ServerApp)
#   final_model.pt   only with --run-config "save-model=true"
#   run_info.json    git commit, clients and settings used by this script
#   logs/            output of every container
#
# Usage:
#   ./run_experiment.sh                                    # 2 clients
#   NUM_CLIENTS=4 ./run_experiment.sh
#   ./run_experiment.sh --run-config "num-server-rounds=5"  # extra args go to `flwr run`
#
# Environment variables:
#   NUM_CLIENTS    number of clients; client i uses partitions/client_<i-1>.parquet
#   DOCKER         docker command (default: "sudo docker")
#   SKIP_BUILD=1   do not rebuild the SuperExec image
#   KEEP_RUNNING=1 leave the containers up after the run
#   DRY_RUN=1      only print the commands
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

NUM_CLIENTS="${NUM_CLIENTS:-2}"
DOCKER="${DOCKER:-sudo docker}"
FLWR_VERSION="1.39.0"
SUPEREXEC_IMAGE="flwr_superexec:0.0.1"
NETWORK="flwr-network"
SUPERLINK_CONNECTION="local-deployment"  # entry in ~/.flwr/config.toml
PARTITIONS_DIR="$PROJECT_DIR/sklearnexample/partitions"
CONTAINER_DATA_PATH="/data/client.parquet"
SUPERNODE_BASE_PORT=9094
RUN_NAME="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$PROJECT_DIR/results/$RUN_NAME"
CONTAINER_RESULTS_DIR="/results"

FLWR="flwr"
if [ -x "$PROJECT_DIR/.venv/bin/flwr" ]; then FLWR="$PROJECT_DIR/.venv/bin/flwr"; fi

# When the whole script is started with sudo, docker needs no extra sudo, and
# `flwr run` must go back to the invoking user, whose ~/.flwr/config.toml holds
# the SuperLink connection.
AS_USER=()
if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
    DOCKER="docker"
    AS_USER=(sudo -u "$SUDO_USER" -H)
fi

containers=(superlink superexec-serverapp)
for i in $(seq 1 "$NUM_CLIENTS"); do
    containers+=("supernode-$i" "superexec-clientapp-$i")
    if [ ! -f "$PARTITIONS_DIR/client_$((i - 1)).parquet" ]; then
        echo "Missing partition for client $i: $PARTITIONS_DIR/client_$((i - 1)).parquet" >&2
        exit 1
    fi
done

run() {
    echo "+ $*"
    if [ -z "${DRY_RUN:-}" ]; then "$@"; fi
}

stop_containers() {
    echo "Stopping containers..."
    # Containers are started with --rm, so stopping also removes them
    run $DOCKER stop "${containers[@]}" >/dev/null 2>&1 || true
}

save_logs() {
    [ -n "${DRY_RUN:-}" ] && return 0
    mkdir -p "$RUN_DIR/logs"
    for container in "${containers[@]}"; do
        $DOCKER logs "$container" >"$RUN_DIR/logs/$container.log" 2>&1 || true
    done
}

finish() {
    save_logs
    if [ -z "${KEEP_RUNNING:-}" ]; then stop_containers; fi
    # Hand the outputs back to the invoking user when started with sudo
    if [ -z "${DRY_RUN:-}" ] && [ -n "${AS_USER[*]}" ]; then
        chown -R "$SUDO_USER" "$RUN_DIR"
    fi
    echo "Outputs: $RUN_DIR"
}

write_run_info() {
    [ -n "${DRY_RUN:-}" ] && return 0
    RUN_NAME="$RUN_NAME" NUM_CLIENTS="$NUM_CLIENTS" FLWR_VERSION="$FLWR_VERSION" \
    SUPEREXEC_IMAGE="$SUPEREXEC_IMAGE" PARTITIONS_DIR="$PARTITIONS_DIR" \
    GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)" \
    GIT_DIRTY="$(git status --porcelain 2>/dev/null | grep -c . || true)" \
    python3 - "$@" >"$RUN_DIR/run_info.json" <<'PY'
import json, os, sys

n = int(os.environ["NUM_CLIENTS"])
print(json.dumps({
    "run_name": os.environ["RUN_NAME"],
    "git_commit": os.environ["GIT_COMMIT"] or None,
    "git_uncommitted_files": int(os.environ["GIT_DIRTY"] or 0),
    "flwr_version": os.environ["FLWR_VERSION"],
    "superexec_image": os.environ["SUPEREXEC_IMAGE"],
    "num_clients": n,
    "clients": [
        {
            "name": f"supernode-{i + 1}",
            "partition_id": i,
            "data_file": os.path.join(os.environ["PARTITIONS_DIR"], f"client_{i}.parquet"),
        }
        for i in range(n)
    ],
    "flwr_run_args": sys.argv[1:],
}, indent=2))
PY
}

wait_for_port() {
    [ -n "${DRY_RUN:-}" ] && return 0
    for _ in $(seq 1 60); do
        if (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; then return 0; fi
        sleep 1
    done
    echo "Timed out waiting for port $1" >&2
    return 1
}

# Remove leftovers from a previous run
stop_containers
trap finish EXIT

# The ServerApp container runs as a different user and must be able to write here
run "${AS_USER[@]}" mkdir -p "$RUN_DIR"
run chmod a+rwx "$RUN_DIR"
write_run_info "$@"

if [ -z "${SKIP_BUILD:-}" ]; then
    run $DOCKER build -f superexec.Dockerfile -t "$SUPEREXEC_IMAGE" .
fi

if [ -n "${DRY_RUN:-}" ] || ! $DOCKER network inspect "$NETWORK" >/dev/null 2>&1; then
    run $DOCKER network create --driver bridge "$NETWORK"
fi

run $DOCKER run --rm \
    -p 8000:8000 -p 9092:9092 \
    --network "$NETWORK" \
    --name superlink \
    --detach \
    "flwr/superlink:$FLWR_VERSION" \
    --insecure \
    --host 0.0.0.0 \
    --isolation process
wait_for_port 8000

run $DOCKER run --rm \
    --network "$NETWORK" \
    --name superexec-serverapp \
    --volume "$RUN_DIR:$CONTAINER_RESULTS_DIR" \
    --detach \
    "$SUPEREXEC_IMAGE" \
    --insecure \
    --runtime-api-address superlink:8000

for i in $(seq 1 "$NUM_CLIENTS"); do
    partition_id=$((i - 1))
    port=$((SUPERNODE_BASE_PORT + partition_id))

    run $DOCKER run --rm \
        -p "$port:$port" \
        --network "$NETWORK" \
        --name "supernode-$i" \
        --detach \
        "flwr/supernode:$FLWR_VERSION" \
        --insecure \
        --superlink superlink:9092 \
        --node-config "partition-id=$partition_id data-path='$CONTAINER_DATA_PATH'" \
        --host 0.0.0.0 \
        --port "$port" \
        --isolation process
    wait_for_port "$port"

    run $DOCKER run --rm \
        --network "$NETWORK" \
        --name "superexec-clientapp-$i" \
        --volume "$PARTITIONS_DIR/client_$partition_id.parquet:$CONTAINER_DATA_PATH:ro" \
        --detach \
        "$SUPEREXEC_IMAGE" \
        --insecure \
        --runtime-api-address "supernode-$i:$port"
done

run "${AS_USER[@]}" "$FLWR" run . "$SUPERLINK_CONNECTION" --stream \
    --run-config "output-dir='$CONTAINER_RESULTS_DIR'" "$@"
