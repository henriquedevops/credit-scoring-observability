"""Pipeline de ponta a ponta: ingestão -> treino -> 6 meses de produção monitorados.

Execução::

    python -m credit_monitor.pipeline

Cada etapa roda dentro de ``tracker.task(...)``, que registra status, duração,
volume e memória. No fim, tudo é consolidado em ``reports/monitoring_summary.json``
(fonte única do dashboard), no MLflow e em ``reports/metrics.prom``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from credit_monitor import config
from credit_monitor.analysis import causal, fairness
from credit_monitor.contracts import credit_contract
from credit_monitor.contracts.credit_contract import DataContractError
from credit_monitor.data import prepare
from credit_monitor.drift import detect, simulate
from credit_monitor.models import train
from credit_monitor.monitoring import alerts as alert_rules
from credit_monitor.monitoring import dashboard, tracking
from credit_monitor.observability import RunTracker

# Em abril o sistema de origem entrega primeiro um arquivo defeituoso; o contrato
# bloqueia, e a segunda entrega (corrigida) segue o fluxo normal.
CORRUPTED_FIRST_DELIVERY = {"2026-04"}
HISTOGRAM_COLUMNS = ["revolving_utilization", "monthly_income", "age", "debt_ratio", "score"]


def _ingest_and_train(tracker: RunTracker) -> tuple[dict[str, Any], train.ModelBundle]:
    with tracker.task("ingest_raw", stage="ingestion") as info:
        ingestion = prepare.prepare()
        info.update(
            rows_in=ingestion["rows_raw"],
            rows_out=ingestion["rows_valid"],
            rows_quarantined=ingestion["rows_quarantined"],
        )

    with tracker.task("train_baseline", stage="training") as info:
        train_df, holdout_df = prepare.load_reference()
        bundle = train.train(train_df, holdout_df)
        train.save(bundle)
        info.update(rows_in=len(train_df), auc_holdout=round(bundle.metrics["auc"], 4))

    with tracker.task("mlflow_log_training", stage="tracking"):
        tracking.log_training(bundle, ingestion, train.feature_importance(bundle))
    return ingestion, bundle


def _receive_batch(
    tracker: RunTracker, month: str, pool: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Recebe o lote do mês, passando cada entrega pelo portão do contrato."""
    scenario = simulate.SCHEDULE[month]
    with tracker.task("extract_batch", batch=month, stage="ingestion") as info:
        batch = simulate.make_batch(pool, month, scenario)
        features, labels = simulate.split_features_labels(batch)
        folder = config.PRODUCTION_DIR / month
        folder.mkdir(parents=True, exist_ok=True)
        features.to_parquet(folder / "features.parquet", index=False)
        labels.to_parquet(folder / "labels.parquet", index=False)  # "chega depois"
        info["rows_out"] = len(features)

    deliveries = [features]
    if month in CORRUPTED_FIRST_DELIVERY:
        deliveries.insert(0, simulate.corrupt_batch(features))

    contract: dict[str, Any] = {"status": "passed", "blocked_deliveries": 0, "deliveries": []}
    accepted = None
    for attempt, delivery in enumerate(deliveries, start=1):
        try:
            with tracker.task("validate_contract", batch=month, stage="quality") as info:
                info.update(rows_in=len(delivery), delivery=attempt)
                credit_contract.enforce(delivery, month)
        except DataContractError as exc:
            contract["blocked_deliveries"] += 1
            contract["deliveries"].append(
                {
                    "delivery": attempt,
                    "status": "blocked",
                    "rows": exc.result.rows,
                    "n_failures": exc.result.n_failures,
                    "failures_by_rule": exc.result.failures_by_rule,
                    "examples": exc.result.examples,
                }
            )
            config.QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
            delivery.to_parquet(
                config.QUARANTINE_DIR / f"{month}_delivery{attempt}.parquet", index=False
            )
            tracker.alert(alert_rules.contract_alert(month, exc.result))
            continue
        contract["deliveries"].append(
            {"delivery": attempt, "status": "passed", "rows": len(delivery)}
        )
        accepted = delivery
        break

    if accepted is None:
        contract["status"] = "blocked"
        raise RuntimeError(f"lote {month}: nenhuma entrega passou no contrato")
    return accepted, labels, contract


def _monitor_batch(
    tracker: RunTracker,
    month: str,
    step: int,
    pool: pd.DataFrame,
    reference: pd.DataFrame,
    bundle: train.ModelBundle,
) -> tuple[dict[str, Any], pd.DataFrame]:
    alerts_before = len(tracker.alerts)
    features, labels, contract = _receive_batch(tracker, month, pool)

    with tracker.task("score_batch", batch=month, stage="inference") as info:
        start = time.perf_counter()
        scored = detect.with_predictions(features, bundle)
        latency = (time.perf_counter() - start) * 1000 / (len(features) / 1000)
        info.update(rows_in=len(features), latency_ms_per_1k=round(latency, 2))

    with tracker.task("detect_drift", batch=month, stage="monitoring") as info:
        table = detect.analyze(reference, scored)
        for alert in alert_rules.drift_alerts(month, table):
            tracker.alert(alert)
        info.update(
            n_critical=int((table["severity"] == "critical").sum()),
            n_warning=int((table["severity"] == "warning").sum()),
        )

    # O rótulo (pagou ou não) só existe meses depois da decisão. Na simulação ele
    # já está em disco, mas é lido numa tarefa separada, como seria na vida real.
    with tracker.task("evaluate_with_labels", batch=month, stage="monitoring") as info:
        labeled = scored.merge(labels, on=config.ID_COL, validate="one_to_one")
        performance = train.evaluate(
            labeled[config.TARGET].to_numpy(),
            labeled[config.SCORE_COL].to_numpy(),
            bundle.threshold,
        )
        auc_drop = bundle.metrics["auc"] - performance["auc"]
        perf_alert = alert_rules.performance_alert(month, performance["auc"], bundle.metrics["auc"])
        if perf_alert:
            tracker.alert(perf_alert)
        info.update(auc=round(performance["auc"], 4), auc_drop=round(auc_drop, 4))

    report_path = config.REPORTS_DIR / "evidently" / f"drift_{month}.html"
    with tracker.task("evidently_report", batch=month, stage="monitoring") as info:
        evidently_summary = detect.evidently_report(reference, labeled, month, report_path)
        info.update(tests_failed=evidently_summary["tests_failed"])

    batch_alerts = tracker.alerts[alerts_before:]
    features_table = table[table["column"] != config.SCORE_COL]
    record: dict[str, Any] = {
        "month": month,
        "scenario": simulate.SCHEDULE[month].__dict__,
        "rows": int(len(features)),
        "latency_ms_per_1k": round(latency, 2),
        "contract": contract,
        "drift": {
            "table": table.to_dict(orient="records"),
            "n_critical": int((features_table["severity"] == "critical").sum()),
            "n_warning": int((features_table["severity"] == "warning").sum()),
            "evidently": evidently_summary,
            "histograms": {
                col: detect.histogram(reference[col], scored[col]) for col in HISTOGRAM_COLUMNS
            },
        },
        "performance": {k: round(v, 4) for k, v in performance.items()},
        "auc_drop": round(auc_drop, 4),
        "diagnosis": alert_rules.diagnose(table, auc_drop),
        "alerts": {
            "total": len(batch_alerts),
            "critical": sum(a.severity == "critical" for a in batch_alerts),
            "warning": sum(a.severity == "warning" for a in batch_alerts),
        },
        "report": f"evidently/drift_{month}.html",
    }
    with tracker.task("mlflow_log_batch", batch=month, stage="tracking"):
        tracking.log_batch(month, step, record, report_path)
    return record, labeled


def run(skip_ingestion: bool = False) -> dict[str, Any]:
    tracker = RunTracker()
    tracker.reset_files()
    tracking.setup()
    started = time.perf_counter()
    tracker.logger.info("pipeline iniciado run_id=%s", tracker.run_id)

    if skip_ingestion and config.MODEL_PATH.exists():
        ingestion = json.loads((config.REPORTS_DIR / "ingestion_summary.json").read_text())
        bundle = train.load()
    else:
        ingestion, bundle = _ingest_and_train(tracker)

    train_df, holdout_df = prepare.load_reference()
    pool = prepare.load_future_pool()
    reference = detect.with_predictions(holdout_df, bundle)

    batches: list[dict[str, Any]] = []
    labeled_batches: dict[str, pd.DataFrame] = {}
    for step, month in enumerate(config.PRODUCTION_MONTHS, start=1):
        record, labeled = _monitor_batch(tracker, month, step, pool, reference, bundle)
        batches.append(record)
        labeled_batches[month] = labeled

    with tracker.task("causal_analysis", stage="analysis"):
        interventions = causal.run_interventions(pool, reference, bundle)
        retrain = causal.retrain_experiment(train_df, labeled_batches, bundle)
        causal.save(interventions, retrain)

    last_month = config.PRODUCTION_MONTHS[-1]
    with tracker.task("fairness_check", stage="analysis"):
        fairness_table = fairness.compare(reference, labeled_batches[last_month], last_month)

    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": tracker.run_id,
        "reference": {
            "period": config.REFERENCE_PERIOD,
            "rows_train": len(train_df),
            "rows_holdout": len(holdout_df),
            "metrics": {k: round(v, 4) for k, v in bundle.metrics.items()},
            "threshold": round(bundle.threshold, 5),
            "feature_importance": train.feature_importance(bundle),
        },
        "thresholds": {
            "psi_warning": config.PSI_WARNING,
            "psi_critical": config.PSI_CRITICAL,
            "ks_alpha": config.KS_ALPHA,
            "auc_drop_warning": config.AUC_DROP_WARNING,
            "auc_drop_critical": config.AUC_DROP_CRITICAL,
            "dataset_drift_share": config.DATASET_DRIFT_SHARE,
            "slo_latency_ms_per_1k": config.SLO_LATENCY_MS_PER_1K,
            "slo_min_rows": config.SLO_MIN_ROWS,
        },
        "ingestion": ingestion,
        "batches": batches,
        "alerts": [a.__dict__ for a in tracker.alerts],
        "causal": interventions.to_dict(orient="records"),
        "retrain": retrain,
        "fairness": fairness_table.to_dict(orient="records"),
    }

    def _pipeline_health() -> dict[str, Any]:
        return {
            "tasks": tracker.tasks,
            "tasks_by_status": dict(Counter(t["status"] for t in tracker.tasks)),
            "duration_s": round(time.perf_counter() - started, 2),
            "max_rss_mb": max(t["max_rss_mb"] for t in tracker.tasks),
        }

    with tracker.task("export_prometheus", stage="tracking"):
        summary["pipeline"] = _pipeline_health()
        tracking.export_prometheus(summary, config.REPORTS_DIR / "metrics.prom")

    with tracker.task("build_dashboard", stage="reporting"):
        summary["pipeline"] = _pipeline_health()
        dashboard.build(summary, config.REPORTS_DIR / "dashboard.html")

    summary["pipeline"] = _pipeline_health()
    (config.REPORTS_DIR / "monitoring_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    dashboard.build(summary, config.REPORTS_DIR / "dashboard.html")
    # Cópia dos logs como evidência versionada (a pasta logs/ fica fora do git).
    evidence = config.REPORTS_DIR / "logs"
    evidence.mkdir(parents=True, exist_ok=True)
    for name in ("tasks.jsonl", "alerts.jsonl", "pipeline.jsonl"):
        (evidence / name).write_text(
            (config.LOGS_DIR / name).read_text(encoding="utf-8"), encoding="utf-8"
        )

    tracker.logger.info(
        "pipeline concluído em %.1fs: %d alertas (%d críticos)",
        summary["pipeline"]["duration_s"],
        len(tracker.alerts),
        sum(a.severity == "critical" for a in tracker.alerts),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="reaproveita Referência e modelo já gerados (pula download e treino)",
    )
    args = parser.parse_args()
    run(skip_ingestion=args.skip_ingestion)


if __name__ == "__main__":
    main()
