"""Monitoramento de viés: o modelo trata faixas etárias de forma desproporcional?

O dataset não traz gênero, raça ou outros dados sensíveis (LGPD art. 5º, II), mas
traz **idade**, que o modelo usa como feature. Idade não é "dado sensível" na
LGPD, porém decisões que prejudicam sistematicamente uma faixa etária ferem o
princípio da não discriminação (art. 6º, IX). Por isso medimos.

Métricas por faixa etária:

- ``approval_rate``: fração aprovada;
- ``disparate_impact``: aprovação da faixa / aprovação da faixa mais aprovada.
  A "regra dos 80%" (four-fifths rule) usa < 0,80 como sinal de alerta;
- ``fpr``: bons pagadores negados (erro que prejudica o cliente);
- ``recall_bad``: inadimplentes barrados (erro que prejudica a fintech quando baixo).
"""

from __future__ import annotations

import pandas as pd

from credit_monitor import config

AGE_BINS = [17, 29, 44, 59, 120]
AGE_LABELS = ["18-29", "30-44", "45-59", "60+"]
DISPARATE_IMPACT_MIN = 0.80


def by_age_group(scored: pd.DataFrame, period: str) -> pd.DataFrame:
    """Métricas de decisão por faixa etária para um conjunto já pontuado e rotulado."""
    df = scored.copy()
    df["age_group"] = pd.cut(df["age"], AGE_BINS, labels=AGE_LABELS)
    rows = []
    for group, part in df.groupby("age_group", observed=True):
        approved = part[config.PRED_COL] == 0
        good = part[config.TARGET] == 0
        rows.append(
            {
                "period": period,
                "age_group": str(group),
                "n": int(len(part)),
                "share": round(len(part) / len(df), 4),
                "approval_rate": round(float(approved.mean()), 4),
                "default_rate": round(float((~good).mean()), 4),
                "fpr": round(float((~approved & good).sum() / max(good.sum(), 1)), 4),
                "recall_bad": round(float((~approved & ~good).sum() / max((~good).sum(), 1)), 4),
                "bad_rate_approved": round(float((~good)[approved].mean()), 4)
                if approved.any()
                else 0.0,
            }
        )
    out = pd.DataFrame(rows)
    out["disparate_impact"] = (out["approval_rate"] / out["approval_rate"].max()).round(3)
    out["below_80pct_rule"] = out["disparate_impact"] < DISPARATE_IMPACT_MIN
    return out


def compare(
    reference_scored: pd.DataFrame, current_scored: pd.DataFrame, month: str
) -> pd.DataFrame:
    table = pd.concat(
        [by_age_group(reference_scored, "referencia"), by_age_group(current_scored, month)],
        ignore_index=True,
    )
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.REPORTS_DIR / "fairness_by_age.csv", index=False)
    return table
