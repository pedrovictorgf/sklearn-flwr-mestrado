"""sklearnexample: metric aggregation and collection for the ServerApp."""

import time

from flwr.app import MetricRecord, RecordDict


def classification_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    """Compute the classification metrics from confusion matrix counts.

    The positive class is the attack class (label 1).
    """
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    )
    return {
        "accuracy": (tp + tn) / total if total > 0 else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _weighted_average(clients: list[dict], key: str) -> float:
    total = sum(client["num_examples"] for client in clients)
    return sum(client[key] * client["num_examples"] for client in clients) / total


class MetricsCollector:
    """Aggregate the metrics replied by the clients and keep them per round.

    Its `aggregate_train` and `aggregate_evaluate` methods are meant to be passed to
    a strategy as `train_metrics_aggr_fn` and `evaluate_metrics_aggr_fn`. Besides
    returning the aggregated MetricRecord, they store the metrics of each client and
    the wall-clock time of each phase, as seen by the server, in `rounds`.
    """

    def __init__(self) -> None:
        self.rounds: list[dict] = []
        self._last_event = time.perf_counter()

    def start(self) -> None:
        """Mark the moment the strategy starts."""
        self._last_event = time.perf_counter()

    def _phase_duration(self) -> float:
        now = time.perf_counter()
        duration = now - self._last_event
        self._last_event = now
        return duration

    @staticmethod
    def _client_metrics(records: list[RecordDict], weighted_by_key: str) -> list[dict]:
        clients = []
        for record in records:
            metrics = dict(next(iter(record.metric_records.values())))
            client = {
                "partition_id": metrics.pop("partition-id"),
                "num_examples": metrics.pop(weighted_by_key),
            }
            client.update(metrics)
            clients.append(client)
        return sorted(clients, key=lambda client: client["partition_id"])

    def aggregate_train(
        self, records: list[RecordDict], weighted_by_key: str
    ) -> MetricRecord:
        """Aggregate the train metrics and start the record of a new round."""
        duration = self._phase_duration()
        clients = self._client_metrics(records, weighted_by_key)
        aggregate = {"train_loss": _weighted_average(clients, "train_loss")}
        self.rounds.append(
            {
                "round": len(self.rounds) + 1,
                "duration_seconds": duration,
                "train": {
                    "duration_seconds": duration,
                    "aggregate": aggregate,
                    "clients": clients,
                },
                "evaluate": None,
            }
        )
        return MetricRecord(aggregate)

    def aggregate_evaluate(
        self, records: list[RecordDict], weighted_by_key: str
    ) -> MetricRecord:
        """Aggregate the evaluate metrics into the record of the current round."""
        duration = self._phase_duration()
        clients = self._client_metrics(records, weighted_by_key)
        for client in clients:
            client.update(
                classification_metrics(
                    client["tp"], client["fp"], client["tn"], client["fn"]
                )
            )

        # The global metrics come from the summed counts: averaging the metrics of
        # each client would be wrong when the class balance differs between clients
        counts = {
            key: sum(client[key] for client in clients)
            for key in ("tp", "fp", "tn", "fn")
        }
        aggregate = {
            "eval_loss": _weighted_average(clients, "eval_loss"),
            **classification_metrics(**counts),
            **counts,
        }

        current_round = self.rounds[-1]
        current_round["duration_seconds"] += duration
        current_round["evaluate"] = {
            "duration_seconds": duration,
            "aggregate": aggregate,
            "clients": clients,
        }
        return MetricRecord(aggregate)
