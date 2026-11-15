"""Rastreamento no MLflow e exportação de métricas no formato Prometheus.

- **MLflow**: um run para o treino do baseline (parâmetros, métricas, modelo
  registrado) e um run por lote de produção (PSI por feature, performance,
  contagem de alertas, relatório Evidently como artefato). A UI do MLflow vira a
  linha do tempo auditável do modelo.
- **MLflow Tracing**: um trace por execução do pipeline. A raiz é a execução
  inteira; cada lote mensal e cada tarefa do ``RunTracker`` viram spans filhos, e
  cada alerta vira um evento no span da tarefa que o gerou. Só agregados entram
  nos spans (contagens, PSI, AUC): nenhuma linha de cliente.
- **Prometheus**: as mesmas métricas de saúde em ``reports/metrics.prom``
  (formato de exposição em texto), prontas para o *textfile collector* do
  node_exporter ou para um Pushgateway, o que permite ver os mesmos números no Grafana.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from mlflow.entities import SpanEvent, SpanType  # noqa: E402
from mlflow.entities.trace_location import MlflowExperimentLocation  # noqa: E402
from prometheus_client import CollectorRegistry, Gauge, write_to_textfile  # noqa: E402

from credit_monitor import config  # noqa: E402


def setup() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    # Os runs ficam no experimento do projeto; os traces vão para o experimento
    # indicado por MLFLOW_EXPERIMENT_ID.
    mlflow.tracing.set_destination(MlflowExperimentLocation(str(config.MLFLOW_EXPERIMENT_ID)))


# --- Tracing -------------------------------------------------------------------


def _clean(values: dict[str, Any]) -> dict[str, Any]:
    """Atributos de span/evento não aceitam ``None``."""
    return {k: v for k, v in values.items() if v is not None}


@contextmanager
def pipeline_trace(run_id: str, inputs: dict[str, Any], metadata: dict[str, str]) -> Iterator[Any]:
    """Abre o span raiz (um trace por execução do pipeline) e descarrega no fim."""
    try:
        with mlflow.start_span(name="credit_scoring_pipeline", span_type=SpanType.WORKFLOW) as span:
            span.set_inputs(inputs)
            months = inputs["production_months"]
            mode = "reaproveitando modelo" if inputs["skip_ingestion"] else "com ingestão e treino"
            mlflow.update_current_trace(
                metadata={"credit_monitor.run_id": run_id, **metadata},
                request_preview=(
                    f"Monitorar {len(months)} lotes ({months[0]} a {months[-1]}), {mode}"
                ),
            )
            yield span
    finally:
        mlflow.flush_trace_async_logging()


def finish_pipeline_trace(span: Any, summary: dict[str, Any]) -> None:
    """Grava na raiz um resumo compacto: o resumo completo fica em ``reports/``."""
    batches = [
        {
            "month": b["month"],
            "diagnosis": b["diagnosis"],
            "auc": b["performance"]["auc"],
            "auc_drop": b["auc_drop"],
            "approval_rate": b["performance"]["approval_rate"],
            "features_drift_critical": b["drift"]["n_critical"],
            "contract_blocked_deliveries": b["contract"]["blocked_deliveries"],
            "alerts": b["alerts"]["total"],
        }
        for b in summary["batches"]
    ]
    critical = sum(a["severity"] == "critical" for a in summary["alerts"])
    span.set_outputs(
        {
            "run_id": summary["run_id"],
            "reference_auc": summary["reference"]["metrics"]["auc"],
            "batches": batches,
            "alerts_total": len(summary["alerts"]),
            "alerts_critical": critical,
            "tasks_by_status": summary["pipeline"]["tasks_by_status"],
            "duration_s": summary["pipeline"]["duration_s"],
        }
    )
    last = batches[-1]
    mlflow.update_current_trace(
        response_preview=(
            f"{len(summary['alerts'])} alertas ({critical} críticos); "
            f"{last['month']}: {last['diagnosis']}, AUC {last['auc']:.3f}, "
            f"{last['features_drift_critical']} features em drift crítico"
        )
    )


@contextmanager
def batch_span(month: str, step: int, scenario: dict[str, Any]) -> Iterator[Any]:
    """Agrupa as tarefas de um lote mensal sob um span próprio."""
    with mlflow.start_span(name=f"monitor_batch {month}", span_type=SpanType.CHAIN) as span:
        span.set_inputs({"month": month, "step": step, "scenario": scenario})
        yield span


def finish_batch_span(span: Any, record: dict[str, Any]) -> None:
    span.set_outputs(
        {
            "diagnosis": record["diagnosis"],
            "rows": record["rows"],
            "contract": {
                "status": record["contract"]["status"],
                "blocked_deliveries": record["contract"]["blocked_deliveries"],
            },
            "drift": {
                "n_critical": record["drift"]["n_critical"],
                "n_warning": record["drift"]["n_warning"],
                "psi": {row["column"]: round(row["psi"], 4) for row in record["drift"]["table"]},
            },
            "performance": record["performance"],
            "auc_drop": record["auc_drop"],
            "alerts": record["alerts"],
        }
    )


@contextmanager
def task_span(name: str, batch: str, stage: str) -> Iterator[Any]:
    """Span de uma tarefa do ``RunTracker``.

    Só existe dentro de um trace do pipeline: usado fora dele (testes, scripts
    avulsos), o tracker continua funcionando e nenhum trace é criado.
    """
    if mlflow.get_current_active_span() is None:
        yield None
        return
    with mlflow.start_span(name=name, span_type=SpanType.TASK) as span:
        span.set_inputs({"stage": stage, "batch": batch})
        yield span


def record_alert(alert: dict[str, Any]) -> None:
    """Anexa o alerta, como evento, ao span da tarefa em execução."""
    span = mlflow.get_current_active_span()
    if span is not None:
        span.add_event(SpanEvent(name=alert["kind"], attributes=_clean(alert)))


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
