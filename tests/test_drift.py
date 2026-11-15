"""PSI/KS com respostas conhecidas e simulação de drift com efeito esperado."""

import numpy as np
import pandas as pd
import pytest

from credit_monitor import config
from credit_monitor.drift import simulate, stats
from credit_monitor.drift.simulate import DriftScenario


def test_psi_is_zero_for_identical_samples():
    x = np.random.default_rng(0).normal(size=5000)
    assert stats.psi(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_is_small_for_same_distribution():
    rng = np.random.default_rng(1)
    assert stats.psi(rng.normal(size=20000), rng.normal(size=20000)) < 0.01


def test_psi_grows_with_the_shift():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=20000)
    small = stats.psi(ref, rng.normal(0.2, 1, 20000))
    large = stats.psi(ref, rng.normal(1.0, 1, 20000))
    assert small < config.PSI_WARNING < config.PSI_CRITICAL < large


def test_psi_matches_closed_form_on_two_bins():
    # Referência 50/50; atual 80/20 => PSI = 0,3·ln(0,8/0,5)·... calculado à mão.
    ref = np.array([0] * 500 + [1] * 500)
    cur = np.array([0] * 800 + [1] * 200)
    expected = (0.8 - 0.5) * np.log(0.8 / 0.5) + (0.2 - 0.5) * np.log(0.2 / 0.5)
    assert stats.psi(ref, cur) == pytest.approx(expected, rel=1e-6)


def test_psi_handles_discrete_mostly_zero_feature():
    rng = np.random.default_rng(3)
    ref = rng.poisson(0.1, 10000)
    assert stats.psi(ref, rng.poisson(0.1, 10000)) < 0.01
    assert stats.psi(ref, rng.poisson(1.0, 10000)) > config.PSI_CRITICAL


def test_severity_bands():
    assert stats.severity(0.05) == "ok"
    assert stats.severity(0.15) == "warning"
    assert stats.severity(0.30) == "critical"


def test_no_drift_scenario_raises_no_flags(clean_frame):
    batch = simulate.make_batch(clean_frame, "2026-01", DriftScenario(), size=1500)
    table = stats.drift_table(clean_frame, batch, config.FEATURES)
    assert (table["severity"] == "ok").all()


def test_income_inflation_drifts_only_income(clean_frame):
    batch = simulate.make_batch(
        clean_frame, "2026-06", DriftScenario(income_inflation=1.0), size=1500
    )
    table = stats.drift_table(clean_frame, batch, config.FEATURES).set_index("column")
    assert table.loc["monthly_income", "severity"] == "critical"
    assert table.loc["monthly_income", "ks_rejects_h0"]
    others = table.drop(index="monthly_income")
    assert (others["severity"] == "ok").all()


def test_credit_stress_drifts_utilization_and_debt(clean_frame):
    batch = simulate.make_batch(clean_frame, "2026-06", DriftScenario(credit_stress=1.0), size=1500)
    table = stats.drift_table(clean_frame, batch, config.FEATURES).set_index("column")
    assert table.loc["revolving_utilization", "severity"] == "critical"
    assert table.loc["debt_ratio", "severity"] != "ok"
    assert batch["revolving_utilization"].between(0, 2).all()
    assert batch["debt_ratio"].between(0, 10).all()


def test_unemployment_shock_changes_labels_but_not_features(clean_frame):
    """Concept drift puro: X idêntico, só P(Y|X) muda — invisível a PSI/KS."""
    control = simulate.make_batch(clean_frame, "2026-06", DriftScenario(), size=1500)
    shocked = simulate.make_batch(
        clean_frame, "2026-06", DriftScenario(unemployment_shock=1.0), size=1500
    )
    pd.testing.assert_frame_equal(control[config.FEATURES], shocked[config.FEATURES])
    assert shocked[config.TARGET].mean() > control[config.TARGET].mean()


def test_batches_are_reproducible(clean_frame):
    scenario = simulate.SCHEDULE["2026-05"]
    first = simulate.make_batch(clean_frame, "2026-05", scenario, size=1000)
    second = simulate.make_batch(clean_frame, "2026-05", scenario, size=1000)
    pd.testing.assert_frame_equal(first, second)


def test_drifted_batches_still_respect_the_contract(clean_frame):
    """Drift não é dado inválido: o lote de junho muda de distribuição mas passa no contrato."""
    from credit_monitor.contracts import credit_contract

    batch = simulate.make_batch(clean_frame, "2026-06", simulate.SCHEDULE["2026-06"], size=1500)
    features, labels = simulate.split_features_labels(batch)
    assert credit_contract.validate(features).passed
    assert list(labels.columns) == [config.ID_COL, config.TARGET]
