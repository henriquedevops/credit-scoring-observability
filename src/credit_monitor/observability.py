"""Logs estruturados, eventos de pipeline e alertas — a "telemetria" do projeto.

Três arquivos JSON Lines (uma linha = um evento) ficam em ``logs/``:

- ``pipeline.jsonl``  — log de aplicação (mensagens INFO/WARNING/ERROR);
- ``tasks.jsonl``     — um registro por tarefa executada (status, duração, linhas, memória);
- ``alerts.jsonl``    — alertas de negócio/qualidade (contrato violado, drift, degradação).

JSON Lines é o formato que agregadores (Datadog, Loki, CloudWatch) ingerem sem
parser customizado: cada linha é um JSON completo, com chaves consistentes.
"""

from __future__ import annotations

import json
import logging
import resource
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from credit_monitor import config

LOGGER_NAME = "credit_monitor"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class JsonFormatter(logging.Formatter):
    """Formata cada registro de log como um JSON de uma linha."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": _now(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "ctx", {}))
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger() -> logging.Logger:
    """Logger com saída legível no console e JSON em ``logs/pipeline.jsonl``."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    logger.addHandler(console)

    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(config.LOGS_DIR / "pipeline.jsonl", encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)
    return logger


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@dataclass
class Alert:
    """Um alerta acionável: o que aconteceu, onde, quão grave e o que fazer."""

    kind: str  # DATA_CONTRACT_VIOLATION | FEATURE_DRIFT | DATASET_DRIFT | ...
    severity: str  # "warning" | "critical"
    batch: str
    message: str
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None
    action: str | None = None
    ts: str = field(default_factory=_now)


class RunTracker:
    """Acompanha uma execução do pipeline: tarefas, alertas e saúde operacional."""

    def __init__(self, run_id: str | None = None, logs_dir: Path | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.logs_dir = logs_dir or config.LOGS_DIR
        self.logger = get_logger()
        self.tasks: list[dict[str, Any]] = []
        self.alerts: list[Alert] = []

    @property
    def tasks_path(self) -> Path:
        return self.logs_dir / "tasks.jsonl"

    @property
    def alerts_path(self) -> Path:
        return self.logs_dir / "alerts.jsonl"

    def reset_files(self) -> None:
        """Começa uma execução limpa (o histórico vive no MLflow, não nos arquivos)."""
        for path in (self.tasks_path, self.alerts_path, self.logs_dir / "pipeline.jsonl"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    @contextmanager
    def task(self, name: str, batch: str = "-", stage: str = "pipeline"):
        """Mede uma tarefa e registra sucesso/falha.

        Uso::

            with tracker.task("validate_contract", batch="2026-03") as info:
                ...
                info["rows_in"] = len(df)

        Se o bloco levantar exceção, a tarefa é registrada como ``failed`` e a
        exceção segue adiante — quem chamou decide se o pipeline continua.
        """
        info: dict[str, Any] = {}
        start = time.perf_counter()
        cpu_start = time.process_time()
        status = "success"
        error = None
        try:
            yield info
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
            raise
        finally:
            record = {
                "ts": _now(),
                "run_id": self.run_id,
                "stage": stage,
                "task": name,
                "batch": batch,
                "status": info.pop("status", status),
                "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                "cpu_ms": round((time.process_time() - cpu_start) * 1000, 2),
                "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                "error": error,
                **info,
            }
            self.tasks.append(record)
            _append_jsonl(self.tasks_path, record)
            level = logging.ERROR if record["status"] == "failed" else logging.INFO
            self.logger.log(
                level,
                "task=%s batch=%s status=%s duration_ms=%.0f",
                name,
                batch,
                record["status"],
                record["duration_ms"],
                extra={"ctx": {k: v for k, v in record.items() if k != "ts"}},
            )

    def alert(self, alert: Alert) -> None:
        self.alerts.append(alert)
        _append_jsonl(self.alerts_path, {"run_id": self.run_id, **asdict(alert)})
        level = logging.ERROR if alert.severity == "critical" else logging.WARNING
        self.logger.log(
            level,
            "ALERTA [%s/%s] lote=%s — %s",
            alert.severity,
            alert.kind,
            alert.batch,
            alert.message,
            extra={"ctx": {"run_id": self.run_id, "alert": asdict(alert)}},
        )
