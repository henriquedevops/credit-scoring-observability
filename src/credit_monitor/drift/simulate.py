"""Simulação da passagem do tempo: seis lotes mensais de produção com drift crescente.

A história econômica simulada (jan-jun/2026):

- **jan-fev**: mundo estável. Os lotes vêm da mesma distribuição da Referência.
  Servem de *controle*: o monitoramento não pode alarmar aqui (falso positivo).
- **mar-abr**: inflação e juros altos. A renda nominal sobe (reajustes), as famílias
  usam mais o rotativo e comprometem mais renda, e a fintech passa a captar um
  público mais jovem. Isso é **data drift** (muda P(X)).
- **mai-jun**: choque de desemprego. Clientes de histórico limpo (os que o modelo
  aprova de olhos fechados) passam a inadimplir. Isso é **concept drift** (muda
  P(Y|X)): as mesmas features agora significam outro risco.

Cada mecanismo é uma função pura com ``intensity`` em [0, 1], o que permite ligar
um de cada vez na análise causal (``credit_monitor.analysis.causal``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from credit_monitor import config
from credit_monitor.data.prepare import pseudonymize

LATE_COLS = ["late_30_59_days", "late_60_89_days", "late_90_days"]


@dataclass(frozen=True)
class DriftScenario:
    """Intensidade (0 = desligado, 1 = máximo) de cada mecanismo de drift."""

    income_inflation: float = 0.0
    credit_stress: float = 0.0
    younger_profile: float = 0.0
    unemployment_shock: float = 0.0

    def any_drift(self) -> bool:
        return any(v > 0 for v in asdict(self).values())


# Cronograma: o drift de dados começa em março; o de conceito, em maio.
SCHEDULE: dict[str, DriftScenario] = {
    "2026-01": DriftScenario(),
    "2026-02": DriftScenario(),
    "2026-03": DriftScenario(0.35, 0.30, 0.30, 0.0),
    "2026-04": DriftScenario(0.65, 0.60, 0.60, 0.0),
    "2026-05": DriftScenario(0.85, 0.80, 0.80, 0.5),
    "2026-06": DriftScenario(1.00, 1.00, 1.00, 1.0),
}

# Os valores máximos são um cenário de estresse deliberadamente forte: o objetivo é
# exercitar o monitoramento, não prever a economia.
MAX_INCOME_INFLATION = 0.70  # renda nominal até +70% no pior mês
MAX_STRESSED_SHARE = 0.55  # até 55% dos clientes passam a depender mais do rotativo
MAX_UTILIZATION_PUSH = 0.70  # um cliente estressado fecha até 70% da distância até o limite
MAX_DEBT_RATIO_GROWTH = 0.60  # comprometimento de renda até +60%
MAX_YOUNG_WEIGHT = 6.0  # clientes < 35 anos até 7x mais prováveis na amostra
MAX_SHOCK_FLIP = 0.045  # prob. base de um bom pagador de histórico limpo virar inadimplente


def sample_batch(
    pool: pd.DataFrame, size: int, younger_profile: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Sorteia um lote do pool. Com ``younger_profile`` > 0, favorece clientes jovens.

    É um *covariate shift* puro: muda quem aparece (P(X)), mas cada cliente
    mantém seu rótulo verdadeiro (P(Y|X) intacta).
    """
    weights = np.ones(len(pool))
    if younger_profile > 0:
        weights = weights + MAX_YOUNG_WEIGHT * younger_profile * (pool["age"].to_numpy() < 35)
    idx = rng.choice(len(pool), size=size, replace=False, p=weights / weights.sum())
    return pool.iloc[idx].reset_index(drop=True)


def inflate_income(df: pd.DataFrame, intensity: float, rng: np.random.Generator) -> pd.DataFrame:
    """Inflação: renda nominal sobe (reajuste médio + ruído individual)."""
    if intensity <= 0:
        return df
    out = df.copy()
    factor = 1 + MAX_INCOME_INFLATION * intensity * rng.normal(1.0, 0.15, len(out)).clip(0.5, 1.5)
    out["monthly_income"] = (out["monthly_income"] * factor).clip(upper=100_000).round(0)
    return out


def apply_credit_stress(
    df: pd.DataFrame, intensity: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Juros altos: parte dos clientes usa mais o rotativo; todos comprometem mais renda.

    Só uma fração dos clientes (crescente com a intensidade) é afetada no rotativo,
    cada um em grau diferente: o aperto não atinge todo mundo por igual.
    """
    if intensity <= 0:
        return df
    out = df.copy()
    util = out["revolving_utilization"].to_numpy()
    stressed = rng.random(len(out)) < MAX_STRESSED_SHARE * intensity
    push = rng.uniform(0.1, MAX_UTILIZATION_PUSH, len(out))
    pushed = util + push * (1 - util)
    out["revolving_utilization"] = np.where(stressed & (util < 1), pushed, util).clip(0, 2)
    out["debt_ratio"] = (out["debt_ratio"] * (1 + MAX_DEBT_RATIO_GROWTH * intensity)).clip(0, 10)
    return out


def apply_unemployment_shock(
    df: pd.DataFrame, intensity: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Concept drift: bons pagadores de histórico limpo passam a inadimplir.

    Nenhuma feature muda, só o rótulo. Por isso nenhum teste de distribuição
    em X enxerga este drift; ele só aparece quando o rótulo chega.
    A probabilidade de virar inadimplente é maior para quem tem dependentes e
    renda abaixo da mediana (mais vulnerável à perda do emprego).
    """
    if intensity <= 0:
        return df
    out = df.copy()
    clean_history = (out[LATE_COLS].sum(axis=1) == 0) & (out[config.TARGET] == 0)
    vulnerability = (
        1.0
        + 0.8 * (out["dependents"] >= 2)
        + 0.8 * (out["monthly_income"] < out["monthly_income"].median())
    )
    flip_prob = MAX_SHOCK_FLIP * intensity * vulnerability
    flips = clean_history & (rng.random(len(out)) < flip_prob)
    out.loc[flips, config.TARGET] = 1
    return out


def make_batch(
    pool: pd.DataFrame,
    month: str,
    scenario: DriftScenario,
    size: int = config.BATCH_SIZE,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Gera o lote de um mês (features + rótulo verdadeiro) aplicando o cenário."""
    rng = np.random.default_rng([seed, int(month.replace("-", ""))])
    batch = sample_batch(pool, size, scenario.younger_profile, rng)
    # O rótulo é decidido sobre o cliente "real" (antes da inflação nominal).
    batch = apply_unemployment_shock(batch, scenario.unemployment_shock, rng)
    batch = inflate_income(batch, scenario.income_inflation, rng)
    batch = apply_credit_stress(batch, scenario.credit_stress, rng)
    # Cada pedido mensal é um novo registro: novo id pseudonimizado.
    batch[config.ID_COL] = pseudonymize(month + ":" + batch[config.ID_COL])
    return batch


def corrupt_batch(df: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Simula uma entrega defeituosa do sistema de origem (para testar o contrato).

    Defeitos injetados, todos comuns em produção:

    - renda nula (campo opcional novo no formulário) e renda negativa (estorno);
    - idade 0 (default do cadastro) e menor de idade;
    - código-sentinela 98 nos contadores de atraso;
    - comprometimento de renda absurdo (dívida em reais no lugar da razão);
    - linhas duplicadas (job de ingestão reprocessado);
    - coluna ``cpf``: PII que nunca deveria chegar à camada analítica.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)

    def pick(frac: float) -> np.ndarray:
        return rng.choice(n, size=max(1, int(n * frac)), replace=False)

    out.loc[pick(0.05), "monthly_income"] = np.nan
    out.loc[pick(0.005), "monthly_income"] = -1500.0
    out.loc[pick(0.01), "age"] = 0
    out.loc[pick(0.005), "age"] = 16
    out.loc[pick(0.01), "late_90_days"] = 98
    out.loc[pick(0.01), "debt_ratio"] = 4500.0
    out["cpf"] = [f"{x:011d}" for x in rng.integers(0, 10**11, n)]
    duplicated = out.iloc[pick(0.02)]
    return pd.concat([out, duplicated], ignore_index=True)


def split_features_labels(batch: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa o que chega na hora (features) do que só chega depois (rótulo)."""
    features = batch[[config.ID_COL, *config.FEATURES]].copy()
    labels = batch[[config.ID_COL, config.TARGET]].copy()
    return features, labels
