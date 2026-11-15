"""Contrato de dados do lote de crédito (Pandera).

Um *data contract* é um acordo explícito e executável entre quem produz o dado
(sistema de originação de crédito) e quem consome (o modelo). Aqui ele é código:
se o lote não cumpre o contrato, a ingestão é bloqueada **antes** do modelo.

Regras rígidas (qualquer violação bloqueia o lote):

1.  ``client_id`` obrigatório, pseudonimizado (16 hex) e **único** — sem duplicatas.
2.  ``age`` entre 18 e 100 — crédito só para maiores de idade; 0 ou 109 é erro de cadastro.
3.  ``monthly_income`` **não nula**, > 0 e <= 100.000 — renda é insumo central do score.
4.  ``debt_ratio`` entre 0 e 10 — acima disso é sintoma de renda ausente ou zerada na origem.
5.  ``revolving_utilization`` entre 0 e 2 — o bruto chega a 50.708 (erro de unidade).
6.  Contadores de atraso entre 0 e 20 — 96/98 são códigos-sentinela, não contagens.
7.  ``dependents`` não nulo, entre 0 e 20.
8.  Nenhuma linha inteiramente duplicada.
9.  ``strict=True``: coluna fora do contrato (ex.: ``cpf``, ``nome``) bloqueia o lote
    — é a *minimização de dados* da LGPD virando regra executável.
10. ``coerce=False``: tipo errado é erro, não é "consertado" em silêncio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pandera.pandas as pa

from credit_monitor import config

CONTRACT_VERSION = "1.0.0"

_LATE = {"ge": 0, "le": 20}


def _late_column(title: str) -> pa.Column:
    return pa.Column(
        int,
        pa.Check.in_range(_LATE["ge"], _LATE["le"]),
        nullable=False,
        title=title,
        description="Contagem de atrasos; 96/98 são códigos-sentinela da origem e são barrados.",
    )


def build_schema(*, with_target: bool = False) -> pa.DataFrameSchema:
    """Monta o contrato. ``with_target=True`` exige também o rótulo (dados de treino)."""
    columns: dict[str, pa.Column] = {
        config.ID_COL: pa.Column(
            str,
            pa.Check.str_matches(r"^[0-9a-f]{16}$"),
            nullable=False,
            unique=True,
            title="Identificador pseudonimizado",
            description="Hash com sal do id interno. Nunca CPF, nome ou e-mail.",
        ),
        "revolving_utilization": pa.Column(
            float,
            pa.Check.in_range(0, 2),
            nullable=False,
            title="Utilização do limite rotativo (saldo / limite)",
        ),
        "age": pa.Column(
            int,
            pa.Check.in_range(18, 100),
            nullable=False,
            title="Idade em anos",
        ),
        "late_30_59_days": _late_column("Atrasos de 30-59 dias (últimos 2 anos)"),
        "debt_ratio": pa.Column(
            float,
            pa.Check.in_range(0, 10),
            nullable=False,
            title="Comprometimento de renda (dívida mensal / renda mensal)",
        ),
        "monthly_income": pa.Column(
            float,
            [pa.Check.gt(0), pa.Check.le(100_000)],
            nullable=False,
            title="Renda mensal declarada",
        ),
        "open_credit_lines": pa.Column(
            int, pa.Check.in_range(0, 60), nullable=False, title="Linhas de crédito abertas"
        ),
        "late_90_days": _late_column("Atrasos de 90+ dias"),
        "real_estate_loans": pa.Column(
            int, pa.Check.in_range(0, 60), nullable=False, title="Financiamentos imobiliários"
        ),
        "late_60_89_days": _late_column("Atrasos de 60-89 dias (últimos 2 anos)"),
        "dependents": pa.Column(
            int, pa.Check.in_range(0, 20), nullable=False, title="Número de dependentes"
        ),
    }
    if with_target:
        columns[config.TARGET] = pa.Column(
            int, pa.Check.isin([0, 1]), nullable=False, title="Inadimplência em 2 anos (rótulo)"
        )

    return pa.DataFrameSchema(
        columns=columns,
        checks=[
            pa.Check(
                lambda df: ~df.duplicated(keep="first"),
                name="no_duplicate_rows",
                error="linha inteiramente duplicada",
            ),
        ],
        strict=True,
        coerce=False,
        ordered=False,
        name=f"credit_batch_contract_v{CONTRACT_VERSION}",
        description="Contrato de dados dos lotes de pedidos de crédito.",
    )


@dataclass
class ValidationResult:
    """Resultado de validar um lote contra o contrato."""

    passed: bool
    rows: int
    n_failures: int = 0
    failures_by_rule: dict[str, int] = field(default_factory=dict)
    examples: list[dict[str, Any]] = field(default_factory=list)
    invalid_index: list[Any] = field(default_factory=list)

    def summary(self) -> str:
        if self.passed:
            return f"contrato OK ({self.rows} linhas)"
        rules = "; ".join(f"{rule}: {n}" for rule, n in self.failures_by_rule.items())
        return f"contrato VIOLADO — {self.n_failures} falhas em {self.rows} linhas ({rules})"


class DataContractError(Exception):
    """Levantada quando um lote viola o contrato e a ingestão é bloqueada."""

    def __init__(self, batch: str, result: ValidationResult):
        self.batch = batch
        self.result = result
        super().__init__(f"lote {batch}: {result.summary()}")


def _rule_name(row: pd.Series) -> str:
    column = row["column"]
    check = str(row["check"])
    if row["schema_context"] == "DataFrameSchema":
        if check == "column_in_schema":
            return f"coluna fora do contrato: {row['failure_case']}"
        if check == "column_in_dataframe":
            return f"coluna obrigatória ausente: {row['failure_case']}"
        return check
    return f"{column}: {check}"


def validate(df: pd.DataFrame, *, with_target: bool = False) -> ValidationResult:
    """Valida ``df`` em modo *lazy* (coleta todas as falhas, não só a primeira)."""
    schema = build_schema(with_target=with_target)
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases.copy()
        cases["rule"] = cases.apply(_rule_name, axis=1)
        # Conta linhas distintas por regra (uma checagem de DataFrame reporta a mesma
        # linha uma vez por coluna); falhas sem índice (ex.: coluna extra) contam 1.
        by_rule = (
            cases.groupby("rule")["index"]
            .agg(lambda idx: max(idx.nunique(), 1))
            .sort_values(ascending=False)
            .to_dict()
        )
        examples = (
            cases.groupby("rule")
            .head(1)[["rule", "failure_case", "index"]]
            .astype(str)
            .to_dict(orient="records")
        )
        invalid = cases["index"].dropna().unique().tolist()
        return ValidationResult(
            passed=False,
            rows=len(df),
            n_failures=int(sum(by_rule.values())),
            failures_by_rule={str(k): int(v) for k, v in by_rule.items()},
            examples=examples,
            invalid_index=invalid,
        )
    return ValidationResult(passed=True, rows=len(df))


def enforce(df: pd.DataFrame, batch: str, *, with_target: bool = False) -> ValidationResult:
    """Portão de ingestão: devolve o resultado se passou; levanta erro se violou."""
    result = validate(df, with_target=with_target)
    if not result.passed:
        raise DataContractError(batch, result)
    return result


def split_valid_invalid(
    df: pd.DataFrame, *, with_target: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, ValidationResult]:
    """Separa linhas válidas das inválidas (usado só na curadoria da Referência).

    Em produção o lote inteiro é bloqueado; na construção do dataset de
    referência queremos o contrário: aproveitar as linhas boas e mandar as ruins
    para quarentena, com a contagem por regra registrada.
    """
    result = validate(df, with_target=with_target)
    if result.passed:
        return df, df.iloc[0:0], result
    bad_mask = df.index.isin(result.invalid_index)
    return df[~bad_mask], df[bad_mask], result
