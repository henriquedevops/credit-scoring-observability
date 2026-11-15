"""Testes estatísticos de drift implementados "na mão": PSI e Kolmogorov-Smirnov.

O Evidently calcula as mesmas métricas para o relatório HTML. Mantemos esta
implementação própria por três motivos: (1) é ela que alimenta alertas, MLflow e
dashboard, sem depender do formato interno do Evidently; (2) é testável com
casos de resposta conhecida; (3) deixa a matemática à vista.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from credit_monitor import config

EPS = 1e-4  # piso para proporções: evita log(0) quando um bin fica vazio


MAX_DISCRETE_VALUES = 20


def _bin_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    """Bordas dos bins, definidas só pela referência.

    - Variável **discreta** (até 20 valores distintos, ex.: nº de atrasos): cada
      valor vira um bin. Quantis não servem aqui — com 90% de zeros todos os
      quantis valem 0, sobra um único bin e o PSI daria sempre 0 (ponto cego).
    - Variável **contínua**: quantis da referência (cada bin com ~mesma massa),
      o que evita bins vazios e torna o PSI comparável entre variáveis.
    """
    distinct = np.unique(reference)
    if len(distinct) <= MAX_DISCRETE_VALUES:
        inner = (distinct[:-1] + distinct[1:]) / 2  # pontos médios entre valores vizinhos
    else:
        inner = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))[1:-1]
    return np.concatenate(([-np.inf], inner.astype(float), [np.inf]))


def psi(reference: np.ndarray, current: np.ndarray, bins: int = config.PSI_BINS) -> float:
    """Population Stability Index.

    PSI = Σ (p_atual − p_ref) · ln(p_atual / p_ref), somado sobre os bins.

    Leitura (aula 02.02): < 0,10 estável · 0,10-0,20 mudança moderada · >= 0,20
    mudança significativa. Não é um teste de hipótese: é um *tamanho de efeito*,
    então não "explode" com amostras grandes como o p-valor do KS.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = _bin_edges(reference, bins)
    # side="left": cada bin é (a, b], então um valor discreto cai no bin do próprio quantil.
    ref_counts = np.bincount(
        np.searchsorted(edges, reference, side="left") - 1, minlength=len(edges) - 1
    )[: len(edges) - 1]
    cur_counts = np.bincount(
        np.searchsorted(edges, current, side="left") - 1, minlength=len(edges) - 1
    )[: len(edges) - 1]
    p_ref = np.clip(ref_counts / ref_counts.sum(), EPS, None)
    p_cur = np.clip(cur_counts / cur_counts.sum(), EPS, None)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def severity(psi_value: float) -> str:
    if psi_value >= config.PSI_CRITICAL:
        return "critical"
    if psi_value >= config.PSI_WARNING:
        return "warning"
    return "ok"


@dataclass
class ColumnDrift:
    column: str
    psi: float
    severity: str  # ok | warning | critical (pelo PSI)
    ks_statistic: float  # D: maior distância entre as CDFs empíricas (0-1)
    ks_pvalue: float
    ks_rejects_h0: bool  # p < alfa => "as amostras vêm de distribuições diferentes"
    ref_mean: float
    cur_mean: float
    mean_shift_pct: float


def column_drift(reference: pd.Series, current: pd.Series, column: str) -> ColumnDrift:
    ref = reference.to_numpy(dtype=float)
    cur = current.to_numpy(dtype=float)
    value = psi(ref, cur)
    ks = ks_2samp(ref, cur)
    ref_mean, cur_mean = float(ref.mean()), float(cur.mean())
    return ColumnDrift(
        column=column,
        psi=round(value, 4),
        severity=severity(value),
        ks_statistic=round(float(ks.statistic), 4),
        ks_pvalue=float(ks.pvalue),
        ks_rejects_h0=bool(ks.pvalue < config.KS_ALPHA),
        ref_mean=round(ref_mean, 4),
        cur_mean=round(cur_mean, 4),
        mean_shift_pct=round(100 * (cur_mean - ref_mean) / abs(ref_mean), 2) if ref_mean else 0.0,
    )


def drift_table(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Uma linha por coluna com PSI, KS e deslocamento da média."""
    rows = [asdict(column_drift(reference[c], current[c], c)) for c in columns]
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
