"""Rastreamento no MLflow e exportação de métricas no formato Prometheus.

- **MLflow**: um run para o treino do baseline (parâmetros, métricas, modelo
  registrado) e um run por lote de produção (PSI por feature, performance,
  contagem de alertas, relatório Evidently como artefato). A UI do MLflow vira a
  linha do tempo auditável do modelo.
- **Prometheus**: as mesmas métricas de saúde em ``reports/metrics.prom``
  (formato de exposição em texto), prontas para o *textfile collector* do
  node_exporter ou para um Pushgateway, o que permite ver os mesmos números no Grafana.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from prometheus_client import CollectorRegistry, Gauge, write_to_textfile  # noqa: E402

from credit_monitor import config  # noqa: E402


def setup() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)


def log_training(bundle: Any, ingestion: dict[str, Any], importance: dict[str, float]) -> str:
    """Registra o treino do baseline e o modelo no Model Registry."""
    with mlflow.start_run(run_name="train-baseline") as run:
        mlflow.set_tags(
            {
                "stage": "training",
                "dataset": "GiveMeSomeCredit (OpenML 46929)",
                "contract_version": ingestion["contract_version"],
            }
        )
        mlflow.log_params({f"xgb.{k}": v for k, v in bundle.params.items()})
        mlflow.log_params(
            {
                "threshold": round(bundle.threshold, 5),
                "rows_train": ingestion["rows_reference_train"],
                "rows_holdout": ingestion["rows_reference_holdout"],
                "rows_quarantined": ingestion["rows_quarantined"],
            }
        )
        mlflow.log_metrics({f"holdout.{k}": v for k, v in bundle.metrics.items()})
        mlflow.log_dict(importance, "feature_importance.json")
        mlflow.log_dict(ingestion, "ingestion_summary.json")
        mlflow.xgboost.log_model(
            bundle.model,
            name="model",
            registered_model_name="credit-scoring-xgb",
            input_example=None,
        )
        return run.info.run_id


def log_batch(month: str, step: int, record: dict[str, Any], report_path: Path | None) -> str:
    """Registra o monitoramento de um lote de produção."""
    with mlflow.start_run(run_name=f"monitor-{month}") as run:
        mlflow.set_tags(
            {
                "stage": "monitoring",
                "batch": month,
                "diagnosis": record["diagnosis"],
                "contract_status": record["contract"]["status"],
            }
        )
        metrics: dict[str, float] = {
            "rows": record["rows"],
            "alerts_total": record["alerts"]["total"],
            "alerts_critical": record["alerts"]["critical"],
            "contract_blocked_deliveries": record["contract"]["blocked_deliveries"],
            "features_drift_critical": record["drift"]["n_critical"],
            "features_drift_warning": record["drift"]["n_warning"],
            "latency_ms_per_1k": record["latency_ms_per_1k"],
        }
        for row in record["drift"]["table"]:
            metrics[f"psi.{row['column']}"] = row["psi"]
            metrics[f"ks_d.{row['column']}"] = row["ks_statistic"]
        for key, value in (record.get("performance") or {}).items():
            metrics[f"perf.{key}"] = value
        mlflow.log_metrics(metrics, step=step)
        mlflow.log_dict(record, "batch_summary.json")
        if report_path is not None and report_path.exists():
            mlflow.log_artifact(str(report_path), artifact_path="evidently")
        return run.info.run_id


def export_prometheus(summary: dict[str, Any], path: Path) -> None:
    """Escreve as métricas de saúde no formato de exposição do Prometheus."""
    registry = CollectorRegistry()
    psi = Gauge(
        "credit_feature_psi", "PSI da coluna vs. referência", ["batch", "column"], registry=registry
    )
    auc = Gauge(
        "credit_model_auc", "AUC do modelo no lote (rótulo tardio)", ["batch"], registry=registry
    )
    approval = Gauge(
        "credit_approval_rate", "Taxa de aprovação no lote", ["batch"], registry=registry
    )
    alerts = Gauge(
        "credit_alerts", "Alertas gerados no lote", ["batch", "severity"], registry=registry
    )
    rows = Gauge("credit_batch_rows", "Linhas processadas no lote", ["batch"], registry=registry)
    latency = Gauge(
        "credit_inference_latency_ms_per_1k",
        "Latência de inferência por 1.000 linhas",
        ["batch"],
        registry=registry,
    )
    blocked = Gauge(
        "credit_contract_blocked_deliveries",
        "Entregas bloqueadas pelo contrato",
        ["batch"],
        registry=registry,
    )
    tasks = Gauge(
        "credit_pipeline_tasks", "Tarefas do pipeline por status", ["status"], registry=registry
    )

    for record in summary["batches"]:
        month = record["month"]
        for row in record["drift"]["table"]:
            psi.labels(month, row["column"]).set(row["psi"])
        if record.get("performance"):
            auc.labels(month).set(record["performance"]["auc"])
            approval.labels(month).set(record["performance"]["approval_rate"])
        alerts.labels(month, "critical").set(record["alerts"]["critical"])
        alerts.labels(month, "warning").set(record["alerts"]["warning"])
        rows.labels(month).set(record["rows"])
        latency.labels(month).set(record["latency_ms_per_1k"])
        blocked.labels(month).set(record["contract"]["blocked_deliveries"])
    for status, count in summary["pipeline"]["tasks_by_status"].items():
        tasks.labels(status).set(count)

    path.parent.mkdir(parents=True, exist_ok=True)
    write_to_textfile(str(path), registry)
