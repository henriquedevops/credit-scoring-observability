"""Modelo, alertas, diagnóstico, logs, viés e dashboard."""

import json

import pandas as pd

from credit_monitor import config
from credit_monitor.analysis import fairness
from credit_monitor.contracts import credit_contract
from credit_monitor.data.prepare import pseudonymize
from credit_monitor.drift import detect, simulate, stats
from credit_monitor.drift.simulate import DriftScenario
from credit_monitor.models import train
from credit_monitor.monitoring import alerts
from credit_monitor.observability import Alert, RunTracker, read_jsonl


def test_model_learns_and_threshold_approves_target_share(bundle, clean_frame):
    assert bundle.metrics["auc"] > 0.65
    scores = bundle.score(clean_frame.iloc[:2200])
    approval = (bundle.decide(scores) == 0).mean()
    assert abs(approval - config.TARGET_APPROVAL_RATE) < 0.02


def test_evaluate_reports_credit_metrics(bundle, clean_frame):
    holdout = clean_frame.iloc[2200:]
    metrics = train.evaluate(
        holdout[config.TARGET].to_numpy(), bundle.score(holdout), bundle.threshold
    )
    assert {"auc", "ks", "approval_rate", "bad_rate_approved", "recall_bad"} <= set(metrics)
    # Quem é aprovado deve ser, em média, menos arriscado que a carteira toda.
    assert metrics["bad_rate_approved"] < metrics["default_rate"]


def test_pseudonymize_is_deterministic_and_salted():
    ids = pd.Series(["1", "2", "1"])
    hashed = pseudonymize(ids, salt="a")
    assert hashed[0] == hashed[2] != hashed[1]
    assert hashed.str.fullmatch(r"[0-9a-f]{16}").all()
    assert (pseudonymize(ids, salt="b") != hashed).all()


def _table(bundle, reference, scenario):
    batch = simulate.make_batch(reference, "2026-06", scenario, size=1500)
    ref_scored = detect.with_predictions(reference, bundle)
    return detect.analyze(ref_scored, detect.with_predictions(batch, bundle))


def test_no_alerts_when_stable(bundle, clean_frame):
    table = _table(bundle, clean_frame, DriftScenario())
    assert alerts.drift_alerts("2026-01", table) == []
    assert alerts.diagnose(table, auc_drop=0.0) == "estavel"


def test_feature_and_prediction_alerts_under_stress(bundle, clean_frame):
    table = _table(bundle, clean_frame, DriftScenario(credit_stress=1.0))
    fired = alerts.drift_alerts("2026-06", table)
    kinds = {a.kind for a in fired}
    assert "FEATURE_DRIFT" in kinds
    assert any(a.metric == "psi.revolving_utilization" and a.severity == "critical" for a in fired)
    assert all(a.action for a in fired)
    assert alerts.diagnose(table, auc_drop=0.0) == "data_drift"


def test_diagnosis_separates_concept_from_data_drift(bundle, clean_frame):
    stable = _table(bundle, clean_frame, DriftScenario())
    assert alerts.diagnose(stable, auc_drop=0.10) == "concept_drift"
    drifted = _table(bundle, clean_frame, DriftScenario(credit_stress=1.0))
    assert alerts.diagnose(drifted, auc_drop=0.10) == "data_drift+concept_drift"
    assert alerts.diagnose(drifted, auc_drop=None) == "data_drift_sem_rotulo"


def test_performance_alert_thresholds():
    assert alerts.performance_alert("m", auc=0.85, reference_auc=0.855) is None
    assert alerts.performance_alert("m", auc=0.83, reference_auc=0.855).severity == "warning"
    assert alerts.performance_alert("m", auc=0.70, reference_auc=0.855).severity == "critical"


def test_contract_alert_is_critical(features):
    result = credit_contract.validate(simulate.corrupt_batch(features))
    alert = alerts.contract_alert("2026-04", result)
    assert alert.kind == "DATA_CONTRACT_VIOLATION"
    assert alert.severity == "critical"
    assert alert.value == result.n_failures


def test_tracker_records_success_failure_and_alerts(tmp_path):
    tracker = RunTracker(run_id="test", logs_dir=tmp_path)
    with tracker.task("ok_task", batch="2026-01") as info:
        info["rows_in"] = 10
    try:
        with tracker.task("bad_task", batch="2026-01"):
            raise ValueError("boom")
    except ValueError:
        pass
    tracker.alert(Alert(kind="FEATURE_DRIFT", severity="warning", batch="2026-01", message="x"))

    tasks = read_jsonl(tmp_path / "tasks.jsonl")
    assert [t["status"] for t in tasks] == ["success", "failed"]
    assert tasks[0]["rows_in"] == 10
    assert "ValueError: boom" in tasks[1]["error"]
    assert tasks[0]["duration_ms"] >= 0
    logged = read_jsonl(tmp_path / "alerts.jsonl")
    assert logged[0]["kind"] == "FEATURE_DRIFT"
    assert logged[0]["run_id"] == "test"


def test_fairness_table_flags_disparate_impact(bundle, clean_frame):
    scored = detect.with_predictions(clean_frame, bundle)
    table = fairness.by_age_group(scored, "referencia")
    assert set(table["age_group"]) <= set(fairness.AGE_LABELS)
    assert table["disparate_impact"].max() == 1.0
    assert (table["below_80pct_rule"] == (table["disparate_impact"] < 0.8)).all()
    assert table["share"].sum() == 1.0 or abs(table["share"].sum() - 1.0) < 1e-3


def test_evidently_report_is_generated(bundle, clean_frame, tmp_path):
    reference = detect.with_predictions(clean_frame.iloc[:1500], bundle)
    batch = simulate.make_batch(clean_frame, "2026-06", DriftScenario(credit_stress=1.0), size=1000)
    current = detect.with_predictions(batch, bundle)
    output = tmp_path / "drift.html"
    summary = detect.evidently_report(reference, current, "2026-06", output)
    assert output.stat().st_size > 100_000
    assert set(summary["psi"]) == set(detect.MONITORED)
    assert set(summary["ks_pvalue"]) == set(detect.MONITORED)
    # O PSI do Evidently e o nosso concordam sobre a variável mais afetada.
    ours = stats.drift_table(reference, current, detect.MONITORED).iloc[0]["column"]
    assert max(summary["psi"], key=summary["psi"].get) == ours


def test_histogram_is_normalized(clean_frame):
    hist = detect.histogram(clean_frame["age"], clean_frame["age"] + 5, bins=10)
    assert len(hist["edges"]) == 11
    assert abs(sum(hist["reference"]) - 1.0) < 1e-3
    assert abs(sum(hist["current"]) - 1.0) < 1e-3


def test_summary_json_is_serializable(bundle, clean_frame):
    table = _table(bundle, clean_frame, DriftScenario(credit_stress=1.0))
    json.dumps(table.to_dict(orient="records"))
