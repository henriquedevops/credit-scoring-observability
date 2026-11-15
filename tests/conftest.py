"""Fixtures: dados sintéticos pequenos, para os testes não dependerem de download."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_monitor import config
from credit_monitor.data.prepare import pseudonymize


def make_clean_frame(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Gera clientes sintéticos que cumprem o contrato, com rótulo ligado aos atrasos."""
    rng = np.random.default_rng(seed)
    late_90 = rng.poisson(0.15, n).clip(0, 10)
    late_30 = rng.poisson(0.3, n).clip(0, 10)
    util = rng.beta(0.6, 2.0, n)
    logit = -3.2 + 1.4 * late_90 + 0.7 * late_30 + 2.2 * util
    df = pd.DataFrame(
        {
            config.ID_COL: pseudonymize(pd.Series(range(n)), salt=f"test-{seed}"),
            "revolving_utilization": util,
            "age": rng.integers(21, 80, n),
            "late_30_59_days": late_30,
            "debt_ratio": rng.gamma(2.0, 0.18, n).clip(0, 9),
            "monthly_income": rng.lognormal(8.6, 0.5, n).clip(500, 90_000).round(0),
            "open_credit_lines": rng.integers(0, 20, n),
            "late_90_days": late_90,
            "real_estate_loans": rng.integers(0, 4, n),
            "late_60_89_days": rng.poisson(0.1, n).clip(0, 10),
            "dependents": rng.integers(0, 5, n),
            config.TARGET: (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int),
        }
    )
    int_cols = [
        c
        for c in df.columns
        if c not in (config.ID_COL, "revolving_utilization", "debt_ratio", "monthly_income")
    ]
    df[int_cols] = df[int_cols].astype(int)
    return df


@pytest.fixture(scope="session")
def clean_frame() -> pd.DataFrame:
    return make_clean_frame()


@pytest.fixture()
def features(clean_frame: pd.DataFrame) -> pd.DataFrame:
    return clean_frame.drop(columns=[config.TARGET]).copy()


@pytest.fixture(scope="session")
def bundle(clean_frame: pd.DataFrame):
    from credit_monitor.models import train

    return train.train(clean_frame.iloc[:2200], clean_frame.iloc[2200:])
