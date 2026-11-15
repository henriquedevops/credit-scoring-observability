"""O contrato deve aceitar dado bom e barrar cada tipo de dado ruim."""

import numpy as np
import pytest

from credit_monitor import config
from credit_monitor.contracts import credit_contract
from credit_monitor.contracts.credit_contract import DataContractError
from credit_monitor.drift import simulate


def test_clean_batch_passes(features):
    result = credit_contract.validate(features)
    assert result.passed
    assert result.n_failures == 0


def test_training_data_requires_target(clean_frame):
    assert credit_contract.validate(clean_frame, with_target=True).passed
    # O mesmo dado com rótulo é barrado no contrato de inferência (coluna inesperada).
    assert not credit_contract.validate(clean_frame, with_target=False).passed


@pytest.mark.parametrize(
    ("column", "bad_value", "expected_rule"),
    [
        ("age", 17, "age: in_range(18, 100)"),
        ("age", 130, "age: in_range(18, 100)"),
        ("monthly_income", np.nan, "monthly_income: not_nullable"),
        ("monthly_income", -100.0, "monthly_income: greater_than(0)"),
        ("monthly_income", 250_000.0, "monthly_income: less_than_or_equal_to(100000)"),
        ("debt_ratio", 4500.0, "debt_ratio: in_range(0, 10)"),
        ("revolving_utilization", 50.0, "revolving_utilization: in_range(0, 2)"),
        ("late_90_days", 98, "late_90_days: in_range(0, 20)"),
    ],
)
def test_each_hard_rule_blocks(features, column, bad_value, expected_rule):
    features.loc[features.index[3], column] = bad_value
    result = credit_contract.validate(features)
    assert not result.passed
    assert expected_rule in result.failures_by_rule


def test_duplicated_client_is_blocked(features):
    import pandas as pd

    duplicated = pd.concat([features, features.iloc[[0]]], ignore_index=True)
    result = credit_contract.validate(duplicated)
    assert not result.passed
    assert "client_id: field_uniqueness" in result.failures_by_rule
    assert "linha inteiramente duplicada" in result.failures_by_rule


def test_pii_column_is_blocked(features):
    features["cpf"] = "12345678901"
    result = credit_contract.validate(features)
    assert not result.passed
    assert "coluna fora do contrato: cpf" in result.failures_by_rule


def test_missing_required_column_is_blocked(features):
    result = credit_contract.validate(features.drop(columns=["monthly_income"]))
    assert not result.passed
    assert "coluna obrigatória ausente: monthly_income" in result.failures_by_rule


def test_wrong_type_is_not_silently_coerced(features):
    features["age"] = features["age"].astype(str)
    assert not credit_contract.validate(features).passed


def test_enforce_raises_and_reports_all_failures(features):
    bad = simulate.corrupt_batch(features)
    with pytest.raises(DataContractError) as excinfo:
        credit_contract.enforce(bad, "2026-04")
    result = excinfo.value.result
    # Modo lazy: todas as regras violadas aparecem de uma vez, não só a primeira.
    assert len(result.failures_by_rule) >= 6
    assert "2026-04" in str(excinfo.value)


def test_split_valid_invalid_keeps_only_good_rows(clean_frame):
    frame = clean_frame.copy()
    frame.loc[frame.index[:5], "age"] = 0
    valid, invalid, result = credit_contract.split_valid_invalid(frame, with_target=True)
    assert len(invalid) == 5
    assert len(valid) == len(frame) - 5
    assert credit_contract.validate(valid, with_target=True).passed
    assert not result.passed
    assert config.TARGET in valid.columns
