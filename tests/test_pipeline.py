"""Teste de ponta a ponta: o pipeline inteiro roda sobre dados sintéticos."""

import json

import pandas as pd
import pytest

from credit_monitor import config

from .conftest import make_clean_frame


@pytest.fixture()
def isolated_project(tmp_path, monkeypatch):
    """Aponta todos os caminhos do projeto para uma pasta temporária."""
    paths = {
        "ROOT": tmp_path,
        "DATA_DIR": tmp_path / "data",
        "RAW_PATH": tmp_path / "data" / "raw" / "raw.csv",
        "REFERENCE_DIR": tmp_path / "data" / "reference",
        "PRODUCTION_DIR": tmp_path / "data" / "production",
        "QUARANTINE_DIR": tmp_path / "data" / "quarantine",
        "MODELS_DIR": tmp_path / "models",
        "MODEL_PATH": tmp_path / "models" / "credit_model.joblib",
        "REPORTS_DIR": tmp_path / "reports",
        "LOGS_DIR": tmp_path / "logs",
        "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_path / 'mlflow.db'}",
    }
    for name, value in paths.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def synthetic_raw(n: int = 20_000) -> pd.DataFrame:
    """Dados sintéticos no formato bruto do OpenML, com alguns defeitos reais."""
    clean = make_clean_frame(n, seed=7).drop(columns=[config.ID_COL])
    inverse = {short: original for original, short in config.COLUMN_MAP.items()}
    raw = clean.rename(columns=inverse)
    target = inverse[config.TARGET]
    raw[target] = raw[target].map({1: "Yes", 0: "No"})
    raw["NumberOfDependents"] = raw["NumberOfDependents"].astype(float)
    raw.loc[raw.index[:400], "MonthlyIncome"] = None  # renda nula
    raw.loc[raw.index[400:420], "age"] = 0  # idade inválida
    raw.loc[raw.index[420:440], "NumberOfTimes90DaysLate"] = 98  # sentinela
    return raw


def test_pipeline_end_to_end(isolated_project, monkeypatch):
    from credit_monitor import pipeline
    from credit_monitor.data import prepare

    monkeypatch.setattr(prepare, "download_raw", lambda force=False: synthetic_raw())
    summary = pipeline.run()
    reports = isolated_project / "reports"

    # Ingestão: os defeitos injetados foram para quarentena, com a regra registrada.
    assert summary["ingestion"]["rows_quarantined"] == 440
    assert summary["ingestion"]["failures_by_rule"]["monthly_income: not_nullable"] == 400

    # Seis lotes monitorados; o de abril teve a primeira entrega bloqueada.
    assert [b["month"] for b in summary["batches"]] == config.PRODUCTION_MONTHS
    by_month = {b["month"]: b for b in summary["batches"]}
    assert by_month["2026-04"]["contract"]["blocked_deliveries"] == 1
    assert by_month["2026-04"]["contract"]["status"] == "passed"
    assert by_month["2026-01"]["contract"]["blocked_deliveries"] == 0

    # Controle sem alarmes; drift crítico e degradação no fim.
    assert by_month["2026-01"]["alerts"]["total"] == 0
    assert by_month["2026-01"]["diagnosis"] == "estavel"
    assert by_month["2026-06"]["drift"]["n_critical"] >= 2
    assert by_month["2026-06"]["performance"]["auc"] < by_month["2026-01"]["performance"]["auc"]
    kinds = {a["kind"] for a in summary["alerts"]}
    assert {"DATA_CONTRACT_VIOLATION", "FEATURE_DRIFT", "PERFORMANCE_DEGRADATION"} <= kinds

    # Saúde do pipeline: exatamente uma tarefa falhou (a validação bloqueada).
    assert summary["pipeline"]["tasks_by_status"]["failed"] == 1

    # Análise causal: o choque de desemprego derruba o AUC sem mexer nas features.
    causal = {row["scenario"]: row for row in summary["causal"]}
    shock = causal["Só choque de desemprego (conceito)"]
    assert shock["n_features_critical"] == 0
    assert shock["delta_auc"] < -0.02

    # Artefatos de evidência.
    for name in (
        "dashboard.html",
        "monitoring_summary.json",
        "metrics.prom",
        "causal_interventions.csv",
        "fairness_by_age.csv",
        "retrain_experiment.json",
        "logs/tasks.jsonl",
        "logs/alerts.jsonl",
    ):
        assert (reports / name).exists(), name
    assert len(list((reports / "evidently").glob("drift_*.html"))) == 6
    assert "credit_feature_psi" in (reports / "metrics.prom").read_text()
    assert "Observabilidade do modelo" in (reports / "dashboard.html").read_text()
    json.loads((reports / "monitoring_summary.json").read_text())

    # MLflow: um run de treino + um por lote.
    import mlflow

    runs = mlflow.search_runs(experiment_names=[config.MLFLOW_EXPERIMENT])
    assert len(runs) == 7
