"""Análise causal da degradação: qual mecanismo causou qual sintoma?

Correlação não basta: em junho *tudo* mudou ao mesmo tempo (renda, rotativo,
idade, rótulos). Para atribuir causa usamos **intervenção** — a ideia do
operador ``do()`` de Pearl: num simulador podemos ligar um mecanismo de cada
vez, mantendo todo o resto fixo (mesmo pool, mesma semente), e medir o efeito.

A pergunta contrafactual é: "como estaria o modelo em junho se *apenas* este
fator tivesse mudado?". A diferença para o cenário de controle é o efeito causal
daquele fator (dentro do mundo simulado).

Em produção real não dá para intervir na economia; lá a mesma lógica vira
(a) reponderar o lote para a distribuição de referência (importance weighting)
e ver se a performance volta, e (b) comparar safras antes/depois de um evento.
"""

from __future__ import annotations

import json

import pandas as pd

from credit_monitor import config
from credit_monitor.drift import detect, simulate
from credit_monitor.drift.simulate import DriftScenario
from credit_monitor.models import train
from credit_monitor.models.train import ModelBundle

SCENARIOS: dict[str, DriftScenario] = {
    "Controle (nada muda)": DriftScenario(),
    "Só inflação da renda": DriftScenario(income_inflation=1.0),
    "Só estresse de crédito (rotativo + dívida)": DriftScenario(credit_stress=1.0),
    "Só novo perfil (mais jovens)": DriftScenario(younger_profile=1.0),
    "Só choque de desemprego (conceito)": DriftScenario(unemployment_shock=1.0),
    "Todos os drifts de dados, sem choque": DriftScenario(1.0, 1.0, 1.0, 0.0),
    "Tudo junto (junho/2026)": DriftScenario(1.0, 1.0, 1.0, 1.0),
}

INTERVENTION_MONTH = "2026-06"


def run_interventions(
    pool: pd.DataFrame, reference: pd.DataFrame, bundle: ModelBundle
) -> pd.DataFrame:
    """Roda cada cenário isolado e mede sintomas de drift e de performance."""
    rows = []
    for name, scenario in SCENARIOS.items():
        batch = simulate.make_batch(pool, INTERVENTION_MONTH, scenario)
        scored = detect.with_predictions(batch, bundle)
        table = detect.analyze(reference, scored)
        features = table[table["column"] != config.SCORE_COL]
        metrics = train.evaluate(
            scored[config.TARGET].to_numpy(), scored[config.SCORE_COL].to_numpy(), bundle.threshold
        )
        rows.append(
            {
                "scenario": name,
                "features_critical": ", ".join(
                    features.loc[features["severity"] == "critical", "column"]
                )
                or "—",
                "n_features_critical": int((features["severity"] == "critical").sum()),
                "psi_score": float(table.loc[table["column"] == config.SCORE_COL, "psi"].iloc[0]),
                "auc": round(metrics["auc"], 4),
                "approval_rate": round(metrics["approval_rate"], 4),
                "bad_rate_approved": round(metrics["bad_rate_approved"], 4),
                "default_rate": round(metrics["default_rate"], 4),
            }
        )
    result = pd.DataFrame(rows)
    control = result.iloc[0]
    result["delta_auc"] = (result["auc"] - control["auc"]).round(4)
    result["delta_approval_pp"] = (
        100 * (result["approval_rate"] - control["approval_rate"])
    ).round(2)
    result["delta_bad_rate_pp"] = (
        100 * (result["bad_rate_approved"] - control["bad_rate_approved"])
    ).round(2)
    return result


def retrain_experiment(
    train_df: pd.DataFrame, labeled_batches: dict[str, pd.DataFrame], bundle: ModelBundle
) -> dict:
    """Prescrição: retreinar com dados recentes recupera o modelo?

    Compara, no lote mais recente (junho), o modelo em produção com um modelo
    retreinado em Referência + lotes rotulados de março a maio. Junho fica de
    fora do treino — é o teste "fora do tempo" (out-of-time).
    """
    months = sorted(labeled_batches)
    test_month = months[-1]
    recent = pd.concat([labeled_batches[m] for m in months[2:-1]], ignore_index=True)
    test = labeled_batches[test_month]
    cols = [*config.FEATURES, config.TARGET]

    # Dados recentes pesam mais: são poucos, mas descrevem o mundo atual.
    combined = pd.concat([train_df[cols], recent[cols]], ignore_index=True)
    weights = [1.0] * len(train_df) + [5.0] * len(recent)
    from xgboost import XGBClassifier

    model = XGBClassifier(**train.XGB_PARAMS)
    model.fit(combined[config.FEATURES], combined[config.TARGET], sample_weight=weights)

    y = test[config.TARGET].to_numpy()
    old_scores = bundle.score(test)
    new_scores = model.predict_proba(test[config.FEATURES])[:, 1]
    # Mesma política de aprovação (aprovar ~80%) recalculada sobre o lote atual.
    new_threshold = float(pd.Series(new_scores).quantile(config.TARGET_APPROVAL_RATE))
    old_same_rate = float(pd.Series(old_scores).quantile(config.TARGET_APPROVAL_RATE))

    return {
        "test_month": test_month,
        "retrain_months": months[2:-1],
        "rows_recent": int(len(recent)),
        "production_model": train.evaluate(y, old_scores, bundle.threshold),
        "production_model_threshold_recalibrated": train.evaluate(y, old_scores, old_same_rate),
        "retrained_model": train.evaluate(y, new_scores, new_threshold),
    }


def save(interventions: pd.DataFrame, retrain: dict) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    interventions.to_csv(config.REPORTS_DIR / "causal_interventions.csv", index=False)
    (config.REPORTS_DIR / "retrain_experiment.json").write_text(
        json.dumps(retrain, indent=2, ensure_ascii=False), encoding="utf-8"
    )
