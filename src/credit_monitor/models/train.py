"""Treino do modelo baseline de risco de crédito (XGBoost) no dataset de Referência.

O modelo é propositalmente simples: o projeto avalia a *sustentação* do modelo
(contratos, drift, observabilidade), não a busca pelo melhor AUC.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from credit_monitor import config
from credit_monitor.data import prepare

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": config.SEED,
    "n_jobs": 4,
}


@dataclass
class ModelBundle:
    """Tudo que a inferência precisa: modelo, limiar de decisão e metadados."""

    model: Any
    threshold: float
    features: list[str]
    metrics: dict[str, float]
    params: dict[str, Any]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilidade de inadimplência (0-1) para cada linha."""
        return self.model.predict_proba(df[self.features])[:, 1]

    def decide(self, scores: np.ndarray) -> np.ndarray:
        """1 = negar/encaminhar para análise manual; 0 = aprovar."""
        return (scores >= self.threshold).astype(int)


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Métricas de risco de crédito para um conjunto rotulado.

    - ``auc``: capacidade de ordenar bons e maus pagadores (0,5 = aleatório).
    - ``ks``: maior distância entre as curvas acumuladas de score de bons e maus —
      métrica clássica de crédito (não confundir com o teste KS de drift, que
      compara a mesma variável em dois períodos).
    - ``approval_rate``: fração aprovada (score abaixo do limiar).
    - ``bad_rate_approved``: inadimplência dentro dos aprovados — é aqui que a
      degradação vira prejuízo.
    - ``recall_bad``: fração dos inadimplentes que o modelo barrou.
    """
    y_true = np.asarray(y_true)
    approved = scores < threshold
    bad = y_true == 1
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "ks": float(ks_2samp(scores[bad], scores[~bad]).statistic),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "brier": float(brier_score_loss(y_true, scores)),
        "approval_rate": float(approved.mean()),
        "bad_rate_approved": float(y_true[approved].mean()) if approved.any() else 0.0,
        "recall_bad": float((~approved & bad).sum() / max(bad.sum(), 1)),
        "default_rate": float(bad.mean()),
        "mean_score": float(scores.mean()),
    }


def train(train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> ModelBundle:
    x_train, y_train = train_df[config.FEATURES], train_df[config.TARGET]

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(x_train, y_train)

    # Limiar: aprova os ~80% de menor risco da carteira de treino.
    train_scores = model.predict_proba(x_train)[:, 1]
    threshold = float(np.quantile(train_scores, config.TARGET_APPROVAL_RATE))

    holdout_scores = model.predict_proba(holdout_df[config.FEATURES])[:, 1]
    metrics = evaluate(holdout_df[config.TARGET].to_numpy(), holdout_scores, threshold)

    # Baseline linear só para comparação (um modelo simples já resolve boa parte?).
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    logreg.fit(x_train, y_train)
    metrics["auc_logreg_baseline"] = float(
        roc_auc_score(
            holdout_df[config.TARGET], logreg.predict_proba(holdout_df[config.FEATURES])[:, 1]
        )
    )

    return ModelBundle(
        model=model,
        threshold=threshold,
        features=list(config.FEATURES),
        metrics=metrics,
        params=XGB_PARAMS,
    )


def feature_importance(bundle: ModelBundle) -> dict[str, float]:
    gains = bundle.model.get_booster().get_score(importance_type="gain")
    total = sum(gains.values()) or 1.0
    ranked = sorted(gains.items(), key=lambda kv: kv[1], reverse=True)
    return {name: round(gain / total, 4) for name, gain in ranked}


def save(bundle: ModelBundle) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.MODEL_PATH)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "XGBClassifier",
        "threshold": bundle.threshold,
        "metrics_holdout": bundle.metrics,
        "feature_importance_gain": feature_importance(bundle),
        "params": bundle.params,
    }
    (config.REPORTS_DIR / "baseline_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load() -> ModelBundle:
    return joblib.load(config.MODEL_PATH)


def main() -> ModelBundle:
    train_df, holdout_df = prepare.load_reference()
    bundle = train(train_df, holdout_df)
    save(bundle)
    return bundle


if __name__ == "__main__":
    # Importa pelo nome do pacote para o pickle gravar ``credit_monitor.models.train.ModelBundle``
    # (rodando com ``-m`` a classe seria ``__main__.ModelBundle`` e não recarregaria depois).
    from credit_monitor.models import train as _train

    result = _train.main()
    print(json.dumps(result.metrics, indent=2))
    print(json.dumps(_train.feature_importance(result), indent=2))
