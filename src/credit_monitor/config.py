"""Configuração central: caminhos, nomes de colunas e limiares de alerta.

Tudo que é "número mágico" do projeto mora aqui, para que o README, o dashboard
e os testes falem dos mesmos valores.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("CREDIT_MONITOR_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "give_me_some_credit.csv"
REFERENCE_DIR = DATA_DIR / "reference"
PRODUCTION_DIR = DATA_DIR / "production"
QUARANTINE_DIR = DATA_DIR / "quarantine"

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "credit_model.joblib"

REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = "credit-scoring-observability"  # runs de treino e de monitoramento
MLFLOW_EXPERIMENT_ID = os.environ.get("MLFLOW_EXPERIMENT_ID", "0")  # destino dos traces

SEED = 42

# --- Dataset -----------------------------------------------------------------
OPENML_DATA_ID = 46929  # "GiveMeSomeCredit" (Kaggle 2011), 150.000 linhas

# Nomes originais -> nomes curtos usados em todo o projeto.
COLUMN_MAP = {
    "FinancialDistressNextTwoYears": "default_2y",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "late_30_59_days",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "late_90_days",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "late_60_89_days",
    "NumberOfDependents": "dependents",
}

ID_COL = "client_id"
TARGET = "default_2y"
SCORE_COL = "score"  # probabilidade de inadimplência prevista pelo modelo
PRED_COL = "prediction"  # 1 = crédito negado (score >= limiar)

FEATURES = [
    "revolving_utilization",
    "age",
    "late_30_59_days",
    "debt_ratio",
    "monthly_income",
    "open_credit_lines",
    "late_90_days",
    "real_estate_loans",
    "late_60_89_days",
    "dependents",
]

FEATURE_LABELS = {
    "revolving_utilization": "Utilização do rotativo",
    "age": "Idade",
    "late_30_59_days": "Atrasos 30-59 dias",
    "debt_ratio": "Comprometimento de renda",
    "monthly_income": "Renda mensal",
    "open_credit_lines": "Linhas de crédito abertas",
    "late_90_days": "Atrasos 90+ dias",
    "real_estate_loans": "Financiamentos imobiliários",
    "late_60_89_days": "Atrasos 60-89 dias",
    "dependents": "Dependentes",
    "score": "Score do modelo",
}

# --- Separação Referência x Produção -----------------------------------------
REFERENCE_FRACTION = 0.60  # 60% vira Referência (treino + holdout); 40% vira "futuro"
HOLDOUT_FRACTION = 0.25  # fatia da Referência reservada para medir o modelo
REFERENCE_PERIOD = "safra de treino 2025"
PRODUCTION_MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
BATCH_SIZE = 6000

# --- Decisão de crédito ------------------------------------------------------
# Score >= limiar => pedido vai para análise manual/negativa. O valor é escolhido
# no treino para aprovar ~80% da carteira de referência.
TARGET_APPROVAL_RATE = 0.80

# --- Limiares de drift (PSI < 0,1 estável; 0,1-0,2 moderado; >= 0,2 crítico)
PSI_WARNING = 0.10
PSI_CRITICAL = 0.20
PSI_BINS = 10
KS_ALPHA = 0.05  # nível de significância do teste KS
KS_D_WARNING = 0.10  # em crédito, D > 0,1 já preocupa (regra prática)
DATASET_DRIFT_SHARE = 0.30  # fração de features em drift crítico que dispara alerta de dataset

# --- Limiares de performance (só avaliáveis quando o rótulo chega) -------------
AUC_DROP_WARNING = 0.02
AUC_DROP_CRITICAL = 0.05

# --- SLOs operacionais ---------------------------------------------------------
SLO_LATENCY_MS_PER_1K = 50.0  # latência de inferência por 1.000 linhas
SLO_MIN_ROWS = 5000  # volume mínimo esperado por lote mensal
