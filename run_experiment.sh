#!/usr/bin/env bash
# Deploy an experiment with Docker Compose (one SuperLink, one SuperNode per client
# and their SuperExecs), run the app on it and tear everything down afterwards.
#
# Each run writes to results/<timestamp>/:
#   results.json     per-round metrics and run config (written by the ServerApp)
#   final_model.pt   only when save-model is true
#   compose.yaml     the deployment that was used (see generate_compose.py)
#   experiment.yaml  copy of the experiment config
#   run_info.json    git commit, clients and run config
#   logs/            output of every container
#
# Usage:
#   ./run_experiment.sh                                    # uses experiment.yaml
#   ./run_experiment.sh other.yaml
#   ./run_experiment.sh --run-config "num-server-rounds=5"  # extra args go to `flwr run`
#
# Environment variables:
#   DOCKER         docker command (default: "sudo docker")
#   SKIP_BUILD=1   do not rebuild the SuperExec image
#   KEEP_RUNNING=1 leave the containers up after the run
#   DRY_RUN=1      generate the run directory and only print the commands
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

CONFIG="experiment.yaml"
case "${1:-}" in
    *.yaml | *.yml) CONFIG="$1"; shift ;;
esac

DOCKER="${DOCKER:-sudo docker}"
COMPOSE_PROJECT="flwr-experiment"
SUPERLINK_CONNECTION="local-deployment"  # entry in ~/.flwr/config.toml
SUPERLINK_PORT=8000
RUN_DIR="$PROJECT_DIR/results/$(date +%Y%m%d-%H%M%S)"

FLWR="flwr"
PYTHON="python3"
if [ -x "$PROJECT_DIR/.venv/bin/flwr" ]; then FLWR="$PROJECT_DIR/.venv/bin/flwr"; fi
if [ -x "$PROJECT_DIR/.venv/bin/python" ]; then PYTHON="$PROJECT_DIR/.venv/bin/python"; fi

# When the whole script is started with sudo, docker needs no extra sudo, and
# everything else must go back to the invoking user, whose ~/.flwr/config.toml
# holds the SuperLink connection.
AS_USER=()
if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
    DOCKER="docker"
    AS_USER=(sudo -u "$SUDO_USER" -H)
fi

run() {
    echo "+ $*"
    if [ -z "${DRY_RUN:-}" ]; then "$@"; fi
}

compose() {
    run $DOCKER compose -p "$COMPOSE_PROJECT" -f "$RUN_DIR/compose.yaml" "$@"
}

save_logs() {
    [ -n "${DRY_RUN:-}" ] && return 0
    mkdir -p "$RUN_DIR/logs"
    for service in $($DOCKER compose -p "$COMPOSE_PROJECT" -f "$RUN_DIR/compose.yaml" config --services); do
        $DOCKER compose -p "$COMPOSE_PROJECT" -f "$RUN_DIR/compose.yaml" \
            logs --no-color --no-log-prefix "$service" >"$RUN_DIR/logs/$service.log" 2>&1 || true
    done
}

finish() {
    save_logs
    if [ -z "${KEEP_RUNNING:-}" ]; then compose down --remove-orphans; fi
    # Hand the outputs back to the invoking user when started with sudo
    if [ -z "${DRY_RUN:-}" ] && [ -n "${AS_USER[*]}" ]; then
        chown -R "$SUDO_USER" "$RUN_DIR"
    fi
    echo "Outputs: $RUN_DIR"
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

# Write compose.yaml, experiment.yaml and run_info.json into the run directory
RUN_CONFIG="$("${AS_USER[@]}" "$PYTHON" generate_compose.py "$CONFIG" "$RUN_DIR")"
# The ServerApp container runs as a different user and must be able to write here
chmod a+rwx "$RUN_DIR"

# Remove leftovers from a previous run
run $DOCKER compose -p "$COMPOSE_PROJECT" down --remove-orphans || true
trap finish EXIT

if [ -z "${SKIP_BUILD:-}" ]; then compose build; fi
compose up --detach
wait_for_port "$SUPERLINK_PORT"

run "${AS_USER[@]}" "$FLWR" run . "$SUPERLINK_CONNECTION" --stream \
    --run-config "$RUN_CONFIG" "$@"
