"""Demonstração do contrato: um lote limpo passa; um lote defeituoso é bloqueado.

Execução::

    python -m credit_monitor.contracts.demo

Sai com código 0 se o contrato se comportou como esperado (aceitou o lote bom e
bloqueou o ruim) — o que permite usar o script como teste de fumaça no CI.
"""

from __future__ import annotations

import json
import sys

from credit_monitor import config
from credit_monitor.contracts import credit_contract
from credit_monitor.contracts.credit_contract import DataContractError
from credit_monitor.data import prepare
from credit_monitor.drift import simulate


def main() -> int:
    pool = prepare.load_future_pool()
    batch = simulate.make_batch(pool, "2026-01", simulate.DriftScenario(), size=2000)
    good, _ = simulate.split_features_labels(batch)
    bad = simulate.corrupt_batch(good)

    print("=" * 78)
    print("1) Lote LIMPO  ->", credit_contract.enforce(good, "demo-limpo").summary())

    print("=" * 78)
    print("2) Lote COM ERROS (entrega defeituosa do sistema de origem)")
    try:
        credit_contract.enforce(bad, "demo-defeituoso")
    except DataContractError as exc:
        print("   INGESTÃO BLOQUEADA:", exc)
        print("\n   Violações por regra:")
        for rule, count in exc.result.failures_by_rule.items():
            print(f"     - {rule:<45} {count:>5}")
        print("\n   Um exemplo de cada violação:")
        for example in exc.result.examples:
            print(f"     - {example['rule']:<45} valor={example['failure_case']}")
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        evidence = {
            "batch": exc.batch,
            "contract_version": credit_contract.CONTRACT_VERSION,
            "rows": exc.result.rows,
            "n_failures": exc.result.n_failures,
            "failures_by_rule": exc.result.failures_by_rule,
            "examples": exc.result.examples,
        }
        path = config.REPORTS_DIR / "contract_violation_demo.json"
        path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n   Evidência gravada em {path.relative_to(config.ROOT)}")
        print("=" * 78)
        return 0
    print("   ERRO: o lote defeituoso passou pelo contrato!")
    return 1


if __name__ == "__main__":
    sys.exit(main())
