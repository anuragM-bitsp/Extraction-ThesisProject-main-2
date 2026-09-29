"""
Experiment tracking via MLflow (LLD section 23's tech table).

Uses MLflow's local sqlite backend (`sqlite:///path/to/mlflow.db`) by
default — no tracking server, no network required. This is genuinely
runnable and tested in this sandboxed environment, unlike GROBID/HF Hub/
PubChem elsewhere in this project, which all depend on real external
services this sandbox can't reach. Point `tracking_uri` at a real MLflow
server later (`http://...`) and nothing else here changes.

A real gotcha found while building this: MLflow's plain filesystem backend
(`file:./mlruns`, the option most tutorials show) is now in maintenance
mode and raises on newer MLflow versions unless
`MLFLOW_ALLOW_FILE_STORE=true` is set. The sqlite backend is the currently
recommended local-only option and is what this module actually uses.
"""

from __future__ import annotations

import json
from contextlib import contextmanager

import mlflow

from experiment.runner import ExperimentResult


class MlflowTracker:
    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db", experiment_name: str = "sci-extract"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.tracking_uri = tracking_uri

    @contextmanager
    def start_run(self, run_name: str):
        with mlflow.start_run(run_name=run_name) as run:
            yield run

    def log_experiment_result(self, result: ExperimentResult, run_name: str | None = None) -> str:
        """
        Logs:
          - every ExperimentConfig field as an MLflow param (reproducibility)
          - paper_count / scored_paper_count as metrics
          - one precision/recall/f1 metric triple per (extractor, level) row
            in the ablation table
          - the full list of per-paper EvaluationReports as a JSON artifact,
            so a run can be inspected down to individual-paper results, not
            just the aggregate numbers

        Returns the MLflow run_id for later lookup (e.g. via
        `mlflow.tracking.MlflowClient(tracking_uri).get_run(run_id)`).
        """
        with self.start_run(run_name or result.config.name) as run:
            mlflow.log_params(result.config.as_params())
            mlflow.log_metric("paper_count", result.paper_count)
            mlflow.log_metric("scored_paper_count", result.scored_paper_count)

            for row in result.ablation:
                prefix = f"{row['extractor']}.{row['level']}"
                mlflow.log_metric(f"{prefix}.precision", row["precision"])
                mlflow.log_metric(f"{prefix}.recall", row["recall"])
                mlflow.log_metric(f"{prefix}.f1", row["f1"])

            mlflow.log_text(
                json.dumps([r.model_dump() for r in result.reports], default=str, indent=2),
                "reports.json",
            )
            return run.info.run_id
