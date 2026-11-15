"""Detecção de drift: relatório HTML do Evidently + tabela própria de PSI/KS.

Para cada lote de produção comparamos com a **Referência** (o holdout da safra de
treino: mesma distribuição do treino, mas nunca vista pelo modelo, então serve
de régua justa tanto para drift quanto para performance).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset, DataDriftPreset

from credit_monitor import config
from credit_monitor.drift import stats
from credit_monitor.models.train import ModelBundle

MONITORED = [*config.FEATURES, config.SCORE_COL]


def with_predictions(df: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """Acrescenta score e decisão do modelo ao lote."""
    out = df.copy()
    out[config.SCORE_COL] = bundle.score(out)
    out[config.PRED_COL] = bundle.decide(out[config.SCORE_COL].to_numpy())
    return out


def _data_definition(has_target: bool) -> DataDefinition:
    classification = None
    if has_target:
        classification = [
            BinaryClassification(target=config.TARGET, prediction_labels=config.PRED_COL)
        ]
    return DataDefinition(numerical_columns=MONITORED, classification=classification)


def evidently_report(
    reference: pd.DataFrame, current: pd.DataFrame, month: str, output: Path
) -> dict[str, Any]:
    """Gera o relatório HTML do Evidently e devolve os números principais.

    O relatório tem três blocos:

    1. Drift por **PSI** (limiar 0,2): features + score do modelo;
    2. Drift por **Kolmogorov-Smirnov** (p-valor < 0,05): as mesmas colunas;
    3. Qualidade de classificação, só quando o rótulo do lote já chegou.
    """
    has_target = config.TARGET in current.columns
    cols = [*MONITORED, config.PRED_COL] + ([config.TARGET] if has_target else [])
    definition = _data_definition(has_target)

    metrics: list[Any] = [
        DataDriftPreset(columns=MONITORED, method="psi", threshold=config.PSI_CRITICAL),
        DataDriftPreset(columns=MONITORED, method="ks", threshold=config.KS_ALPHA),
    ]
    if has_target:
        metrics.append(ClassificationPreset())

    report = Report(
        metrics,
        include_tests=True,
        metadata={"batch": month, "reference": config.REFERENCE_PERIOD},
        tags=["credit-scoring", month],
    )
    snapshot = report.run(
        Dataset.from_pandas(current[cols], data_definition=definition),
        Dataset.from_pandas(reference[cols], data_definition=definition),
        name=f"Drift e performance: lote {month}",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output))

    summary: dict[str, Any] = {"psi": {}, "ks_pvalue": {}, "tests_failed": 0, "tests_total": 0}
    payload = snapshot.dict()
    for metric in payload["metrics"]:
        params = metric.get("config", {}) or {}
        name = str(metric.get("metric_name", ""))
        if not name.startswith("ValueDrift"):
            continue
        column = params.get("column") or name.split("column=")[1].split(",")[0]
        method = params.get("method") or name.split("method=")[1].split(",")[0]
        target = summary["psi"] if method == "psi" else summary["ks_pvalue"]
        target[column] = float(metric["value"])
    for test in payload.get("tests", []):
        summary["tests_total"] += 1
        summary["tests_failed"] += int(test.get("status") != "SUCCESS")
    return summary


def analyze(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Tabela própria (PSI + KS por coluna) usada em alertas, MLflow e dashboard."""
    return stats.drift_table(reference, current, MONITORED)


def histogram(reference: pd.Series, current: pd.Series, bins: int = 20) -> dict[str, list[float]]:
    """Histogramas normalizados em bins comuns (para os gráficos do dashboard)."""
    lo = float(min(reference.quantile(0.005), current.quantile(0.005)))
    hi = float(max(reference.quantile(0.995), current.quantile(0.995)))
    if hi <= lo:
        hi = lo + 1.0
    edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
    ref_hist = pd.cut(reference.clip(lo, hi), edges, include_lowest=True).value_counts(
        sort=False, normalize=True
    )
    cur_hist = pd.cut(current.clip(lo, hi), edges, include_lowest=True).value_counts(
        sort=False, normalize=True
    )
    return {
        "edges": [round(e, 4) for e in edges],
        "reference": [round(float(v), 5) for v in ref_hist],
        "current": [round(float(v), 5) for v in cur_hist],
    }
