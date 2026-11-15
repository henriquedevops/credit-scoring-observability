"""Ingestão do dataset bruto e separação Referência x Produção.

Fluxo::

    OpenML (CSV bruto) -> padroniza colunas -> pseudonimiza id -> contrato de dados
        -> linhas válidas  -> 60% REFERÊNCIA (treino + holdout)
                           -> 40% "FUTURO"   (matéria-prima dos lotes de produção)
        -> linhas inválidas -> quarentena (com a regra violada registrada)
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_monitor import config
from credit_monitor.contracts import credit_contract

INT_COLUMNS = [
    "age",
    "late_30_59_days",
    "open_credit_lines",
    "late_90_days",
    "real_estate_loans",
    "late_60_89_days",
]


def download_raw(force: bool = False) -> pd.DataFrame:
    """Baixa o *Give Me Some Credit* do OpenML (sem credenciais) e guarda em cache."""
    if config.RAW_PATH.exists() and not force:
        return pd.read_csv(config.RAW_PATH)
    from sklearn.datasets import fetch_openml

    frame = fetch_openml(data_id=config.OPENML_DATA_ID, as_frame=True, parser="auto").frame
    config.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.RAW_PATH, index=False)
    return pd.read_csv(config.RAW_PATH)


def pseudonymize(values: pd.Series, salt: str | None = None) -> pd.Series:
    """Troca um identificador interno por um pseudônimo (SHA-256 com sal, 16 hex).

    O sal vem da variável de ambiente ``CREDIT_MONITOR_ID_SALT`` — fica fora do
    código e do git. Sem o sal não dá para refazer o caminho pseudônimo -> cliente;
    com ele, o controlador consegue atender pedidos do titular (acesso, exclusão).
    Isso é *pseudonimização* (LGPD art. 13, §4º), não anonimização.
    """
    salt = salt if salt is not None else os.environ.get("CREDIT_MONITOR_ID_SALT", "dev-only-salt")
    return values.astype(str).map(
        lambda v: hashlib.sha256(f"{salt}:{v}".encode()).hexdigest()[:16]
    )


def standardize(raw: pd.DataFrame) -> pd.DataFrame:
    """Renomeia colunas, converte o alvo para 0/1 e cria o id pseudonimizado.

    Não remove nem corrige nada: quem decide o que é dado ruim é o contrato.
    """
    df = raw.rename(columns=config.COLUMN_MAP).copy()
    if df[config.TARGET].dtype == object:
        df[config.TARGET] = df[config.TARGET].map({"Yes": 1, "No": 0})
    df[config.TARGET] = df[config.TARGET].astype(int)
    # O dataset público não traz identificador; simulamos o id interno do sistema
    # de origem com o número da linha e o pseudonimizamos na entrada.
    df.insert(0, config.ID_COL, pseudonymize(pd.Series(df.index, index=df.index)))
    return df


def cast_valid_types(df: pd.DataFrame) -> pd.DataFrame:
    """Converte para os tipos do contrato as colunas que já estão sem nulos."""
    out = df.copy()
    for col in INT_COLUMNS:
        out[col] = out[col].astype(int)
    if out["dependents"].notna().all():
        out["dependents"] = out["dependents"].astype(int)
    for col in ("revolving_utilization", "debt_ratio", "monthly_income"):
        out[col] = out[col].astype(float)
    return out


def prepare(raw: pd.DataFrame | None = None, seed: int = config.SEED) -> dict:
    """Roda a ingestão completa e grava Referência, pool de produção e quarentena."""
    raw = download_raw() if raw is None else raw
    df = standardize(raw)

    # Nulos impedem o cast para inteiro; o contrato precisa ver o dado como veio.
    # Por isso validamos em duas passadas: nulos primeiro, depois tipos e faixas.
    null_mask = df[config.FEATURES].isna().any(axis=1)
    null_counts = df[config.FEATURES].isna().sum()
    with_nulls = df[null_mask]
    complete = cast_valid_types(df[~null_mask])

    valid, invalid, result = credit_contract.split_valid_invalid(complete, with_target=True)

    failures = {f"{col}: not_nullable": int(n) for col, n in null_counts.items() if n > 0}
    failures.update(result.failures_by_rule)

    reference, future = train_test_split(
        valid,
        train_size=config.REFERENCE_FRACTION,
        stratify=valid[config.TARGET],
        random_state=seed,
    )
    ref_train, ref_holdout = train_test_split(
        reference,
        test_size=config.HOLDOUT_FRACTION,
        stratify=reference[config.TARGET],
        random_state=seed,
    )

    for folder in (config.REFERENCE_DIR, config.PRODUCTION_DIR, config.QUARANTINE_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    ref_train.to_parquet(config.REFERENCE_DIR / "train.parquet", index=False)
    ref_holdout.to_parquet(config.REFERENCE_DIR / "holdout.parquet", index=False)
    future.to_parquet(config.PRODUCTION_DIR / "_future_pool.parquet", index=False)
    quarantined = pd.concat([with_nulls, invalid.astype(with_nulls.dtypes.to_dict())])
    quarantined.to_parquet(config.QUARANTINE_DIR / "raw_rejected.parquet", index=False)

    summary = {
        "contract_version": credit_contract.CONTRACT_VERSION,
        "rows_raw": int(len(df)),
        "rows_valid": int(len(valid)),
        "rows_quarantined": int(len(quarantined)),
        "pct_quarantined": round(100 * len(quarantined) / len(df), 2),
        "failures_by_rule": failures,
        "rows_reference_train": int(len(ref_train)),
        "rows_reference_holdout": int(len(ref_holdout)),
        "rows_future_pool": int(len(future)),
        "default_rate_raw": round(float(df[config.TARGET].mean()), 4),
        "default_rate_valid": round(float(valid[config.TARGET].mean()), 4),
        "default_rate_quarantined": round(float(quarantined[config.TARGET].mean()), 4),
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "ingestion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def load_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet(config.REFERENCE_DIR / "train.parquet"),
        pd.read_parquet(config.REFERENCE_DIR / "holdout.parquet"),
    )


def load_future_pool() -> pd.DataFrame:
    return pd.read_parquet(config.PRODUCTION_DIR / "_future_pool.parquet")


def rng(seed: int = config.SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2, ensure_ascii=False))
