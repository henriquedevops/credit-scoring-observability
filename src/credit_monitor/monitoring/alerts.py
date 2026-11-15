"""Regras de alerta: transformam números (PSI, AUC, falhas de contrato) em ação.

Um alerta bom responde a três perguntas: o que aconteceu, quão grave é, e o que
fazer agora. Por isso cada regra carrega uma ``action``.
"""

from __future__ import annotations

import pandas as pd

from credit_monitor import config
from credit_monitor.contracts.credit_contract import ValidationResult
from credit_monitor.observability import Alert

ACTION_CONTRACT = (
    "Lote bloqueado e enviado à quarentena. Acionar o time dono da origem "
    "e reprocessar após a correção."
)
ACTION_FEATURE_WARNING = "Investigar a causa na origem e aumentar a frequência de monitoramento."
ACTION_FEATURE_CRITICAL = (
    "Abrir incidente: confirmar se a mudança é real (economia/público) ou defeito de pipeline."
)
ACTION_DATASET = "Mudança ampla de população: revisar política de crédito e planejar retreino."
ACTION_PREDICTION = (
    "A distribuição de scores mudou: revisar o limiar de aprovação e priorizar a "
    "coleta de rótulos recentes."
)
ACTION_PERFORMANCE = (
    "Degradação confirmada com rótulos: retreinar com janela recente e reavaliar "
    "antes de promover (ver README, plano de mitigação)."
)


def contract_alert(batch: str, result: ValidationResult) -> Alert:
    return Alert(
        kind="DATA_CONTRACT_VIOLATION",
        severity="critical",
        batch=batch,
        message=result.summary(),
        metric="contract_failures",
        value=float(result.n_failures),
        threshold=0.0,
        action=ACTION_CONTRACT,
    )


def drift_alerts(batch: str, table: pd.DataFrame) -> list[Alert]:
    """Alertas de drift por feature, de dataset e de predição a partir da tabela PSI/KS."""
    alerts: list[Alert] = []
    features = table[table["column"] != config.SCORE_COL]

    for row in features.itertuples():
        if row.severity == "ok":
            continue
        critical = row.severity == "critical"
        label = config.FEATURE_LABELS.get(row.column, row.column)
        alerts.append(
            Alert(
                kind="FEATURE_DRIFT",
                severity=row.severity,
                batch=batch,
                message=(
                    f"{label} ({row.column}): PSI={row.psi:.3f}, KS D={row.ks_statistic:.3f}, "
                    f"média {row.mean_shift_pct:+.1f}% vs. referência"
                ),
                metric=f"psi.{row.column}",
                value=float(row.psi),
                threshold=config.PSI_CRITICAL if critical else config.PSI_WARNING,
                action=ACTION_FEATURE_CRITICAL if critical else ACTION_FEATURE_WARNING,
            )
        )

    share = float((features["severity"] == "critical").mean())
    if share >= config.DATASET_DRIFT_SHARE:
        alerts.append(
            Alert(
                kind="DATASET_DRIFT",
                severity="critical",
                batch=batch,
                message=f"{share:.0%} das features em drift crítico (PSI >= {config.PSI_CRITICAL})",
                metric="drifted_feature_share",
                value=share,
                threshold=config.DATASET_DRIFT_SHARE,
                action=ACTION_DATASET,
            )
        )

    score = table[table["column"] == config.SCORE_COL]
    if not score.empty and score.iloc[0]["severity"] != "ok":
        row = score.iloc[0]
        critical = row["severity"] == "critical"
        alerts.append(
            Alert(
                kind="PREDICTION_DRIFT",
                severity=row["severity"],
                batch=batch,
                message=(
                    f"Score do modelo: PSI={row['psi']:.3f}, score médio "
                    f"{row['mean_shift_pct']:+.1f}% vs. referência"
                ),
                metric="psi.score",
                value=float(row["psi"]),
                threshold=config.PSI_CRITICAL if critical else config.PSI_WARNING,
                action=ACTION_PREDICTION,
            )
        )
    return alerts


def performance_alert(batch: str, auc: float, reference_auc: float) -> Alert | None:
    """Alerta de degradação — só possível quando o rótulo do lote já chegou."""
    drop = reference_auc - auc
    if drop < config.AUC_DROP_WARNING:
        return None
    critical = drop >= config.AUC_DROP_CRITICAL
    return Alert(
        kind="PERFORMANCE_DEGRADATION",
        severity="critical" if critical else "warning",
        batch=batch,
        message=(
            f"AUC caiu de {reference_auc:.3f} (referência) para {auc:.3f} (queda de {drop:.3f})"
        ),
        metric="auc_drop",
        value=round(drop, 4),
        threshold=config.AUC_DROP_CRITICAL if critical else config.AUC_DROP_WARNING,
        action=ACTION_PERFORMANCE,
    )


def diagnose(table: pd.DataFrame, auc_drop: float | None) -> str:
    """Classifica o tipo de drift cruzando 'X mudou?' com 'a performance caiu?'.

    =================  ====================  ==============================
    P(X) mudou?        Performance caiu?     Diagnóstico
    =================  ====================  ==============================
    não                não                   estável
    sim                não                   data drift (covariate shift)
    não                sim                   concept drift puro
    sim                sim                   data drift + concept drift
    =================  ====================  ==============================
    """
    features = table[table["column"] != config.SCORE_COL]
    x_changed = bool((features["severity"] == "critical").any())
    if auc_drop is None:
        return "data_drift_sem_rotulo" if x_changed else "estavel_sem_rotulo"
    degraded = auc_drop >= config.AUC_DROP_CRITICAL
    if x_changed and degraded:
        return "data_drift+concept_drift"
    if degraded:
        return "concept_drift"
    if x_changed:
        return "data_drift"
    return "estavel"
