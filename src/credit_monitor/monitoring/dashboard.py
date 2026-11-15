"""Dashboard HTML estático de observabilidade (sem servidor, sem dependência externa).

Lê o ``monitoring_summary.json`` produzido pelo pipeline e gera um único arquivo
``reports/dashboard.html`` com SVG embutido: abre com dois cliques, funciona
offline e pode ser publicado no GitHub Pages.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from credit_monitor import config

TEMPLATE_DIR = Path(__file__).parent / "templates"

STATUS_LABEL = {"ok": "Estável", "warning": "Atenção", "critical": "Crítico"}
STATUS_ICON = {"ok": "✓", "warning": "▲", "critical": "✕"}
DIAGNOSIS_LABEL = {
    "estavel": ("ok", "Estável"),
    "data_drift": ("warning", "Data drift (população mudou, modelo ainda ordena bem)"),
    "concept_drift": ("critical", "Concept drift (performance caiu sem mudança em X)"),
    "data_drift+concept_drift": ("critical", "Data drift + degradação de performance"),
    "data_drift_sem_rotulo": ("warning", "Data drift (rótulo ainda não disponível)"),
    "estavel_sem_rotulo": ("ok", "Estável (rótulo ainda não disponível)"),
}
ALERT_KIND_LABEL = {
    "DATA_CONTRACT_VIOLATION": "Contrato de dados violado",
    "FEATURE_DRIFT": "Drift de feature",
    "DATASET_DRIFT": "Drift de dataset",
    "PREDICTION_DRIFT": "Drift de predição",
    "PERFORMANCE_DEGRADATION": "Degradação de performance",
}
TASK_LABEL = {
    "ingest_raw": "Ingestão do bruto + contrato",
    "train_baseline": "Treino do baseline",
    "mlflow_log_training": "Registro do treino no MLflow",
    "extract_batch": "Extração do lote mensal",
    "validate_contract": "Validação do contrato",
    "score_batch": "Inferência (score)",
    "detect_drift": "Detecção de drift (PSI/KS)",
    "evaluate_with_labels": "Avaliação com rótulos",
    "evidently_report": "Relatório Evidently",
    "mlflow_log_batch": "Registro do lote no MLflow",
    "causal_analysis": "Análise causal + retreino",
    "fairness_check": "Checagem de viés",
    "export_prometheus": "Exportação Prometheus",
    "build_dashboard": "Geração do dashboard",
}


def _fmt(value: float, kind: str = "num") -> str:
    """Formata números no padrão brasileiro (vírgula decimal)."""
    if kind == "pct":
        text = f"{100 * value:.1f}%"
    elif kind == "int":
        return f"{int(value):,}".replace(",", ".")
    elif kind == "num2":
        text = f"{value:.2f}"
    else:
        text = f"{value:.3f}"
    return text.replace(".", ",")


def _month_label(month: str) -> str:
    names = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
    year, number = month.split("-")
    return f"{names[int(number) - 1]}/{year[2:]}"


def line_chart(
    labels: list[str],
    values: list[float],
    *,
    reference: float | None,
    kind: str,
    name: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> str:
    """Gráfico de linha de uma série (um eixo, linha de referência, rótulos nas pontas)."""
    width, height = 380, 210
    left, right, top, bottom = 46, 54, 16, 30
    plot_w, plot_h = width - left - right, height - top - bottom
    candidates = values + ([reference] if reference is not None else [])
    lo = min(candidates) if y_min is None else y_min
    hi = max(candidates) if y_max is None else y_max
    pad = (hi - lo) * 0.15 or 0.01
    lo, hi = (lo - pad if y_min is None else lo), (hi + pad if y_max is None else hi)

    def x(i: int) -> float:
        return left + plot_w * i / max(len(values) - 1, 1)

    def y(v: float) -> float:
        return top + plot_h * (1 - (v - lo) / (hi - lo))

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(name)} por lote mensal">'
    ]
    for step in range(5):
        tick = lo + (hi - lo) * step / 4
        parts.append(
            f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{y(tick):.1f}" '
            f'y2="{y(tick):.1f}"/>'
            f'<text class="tick" x="{left - 6}" y="{y(tick) + 3.5:.1f}" text-anchor="end">'
            f"{_fmt(tick, kind)}</text>"
        )
    for i, label in enumerate(labels):
        parts.append(
            f'<text class="tick" x="{x(i):.1f}" y="{height - 10}" text-anchor="middle">'
            f"{label}</text>"
        )
    if reference is not None:
        parts.append(
            f'<line class="refline" x1="{left}" x2="{left + plot_w}" y1="{y(reference):.1f}" '
            f'y2="{y(reference):.1f}"/>'
            f'<text class="tick" x="{left + plot_w + 6}" y="{y(reference) - 4:.1f}">ref.</text>'
            f'<text class="tick" x="{left + plot_w + 6}" y="{y(reference) + 9:.1f}">'
            f"{_fmt(reference, kind)}</text>"
        )
    path = " ".join(f"{'M' if i == 0 else 'L'}{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    parts.append(f'<path class="series" d="{path}"/>')
    for i, v in enumerate(values):
        tip = f"{labels[i]} · {name}: {_fmt(v, kind)}"
        parts.append(
            f'<circle class="dot" cx="{x(i):.1f}" cy="{y(v):.1f}" r="4"/>'
            f'<circle class="hit" cx="{x(i):.1f}" cy="{y(v):.1f}" r="16" tabindex="0" '
            f'data-tip="{html.escape(tip)}"/>'
        )
    last = len(values) - 1
    # Rótulo direto só no último ponto (o que conta a história); o resto fica no eixo/tooltip.
    ref_y = y(reference) if reference is not None else -100
    label_y = y(values[last]) + 4
    if abs(label_y - ref_y) < 22:
        label_y = ref_y + (26 if y(values[last]) >= ref_y else -18)
    parts.append(
        f'<text class="endlabel" x="{x(last) + 8:.1f}" y="{label_y:.1f}">'
        f"{_fmt(values[last], kind)}</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def histogram_chart(hist: dict[str, list[float]], name: str) -> str:
    """Duas distribuições sobrepostas: referência (área) x produção (linha em degraus)."""
    width, height = 380, 190
    left, right, top, bottom = 10, 10, 10, 26
    plot_w, plot_h = width - left - right, height - top - bottom
    edges, ref, cur = hist["edges"], hist["reference"], hist["current"]
    n = len(ref)
    peak = max(max(ref), max(cur)) * 1.08 or 1.0

    def x(i: float) -> float:
        return left + plot_w * i / n

    def y(v: float) -> float:
        return top + plot_h * (1 - v / peak)

    def steps(values: list[float]) -> str:
        points = [f"M{x(0):.1f},{y(values[0]):.1f}"]
        for i, v in enumerate(values):
            points.append(f"L{x(i):.1f},{y(v):.1f} L{x(i + 1):.1f},{y(v):.1f}")
        return " ".join(points)

    base = y(0)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Distribuição de {html.escape(name)}: referência e produção">',
        f'<path class="area-ref" d="{steps(ref)} L{x(n):.1f},{base:.1f} '
        f'L{x(0):.1f},{base:.1f} Z"/>',
        f'<path class="series" d="{steps(ref)}"/>',
        f'<path class="series s2" d="{steps(cur)}"/>',
        f'<line class="axis" x1="{left}" x2="{left + plot_w}" y1="{base:.1f}" y2="{base:.1f}"/>',
    ]
    for i in (0, n // 2, n):
        anchor = "start" if i == 0 else "end" if i == n else "middle"
        value = edges[i]
        text = _fmt(value, "int") if abs(value) >= 100 else _fmt(value, "num2")
        parts.append(
            f'<text class="tick" x="{x(i):.1f}" y="{height - 8}" text-anchor="{anchor}">'
            f"{text}</text>"
        )
    for i in range(n):
        lo_edge = _fmt(edges[i], "int") if abs(edges[i + 1]) >= 100 else _fmt(edges[i], "num2")
        hi_edge = (
            _fmt(edges[i + 1], "int") if abs(edges[i + 1]) >= 100 else _fmt(edges[i + 1], "num2")
        )
        tip = (
            f"{name} de {lo_edge} a {hi_edge} · referência {_fmt(ref[i], 'pct')} · "
            f"produção {_fmt(cur[i], 'pct')}"
        )
        parts.append(
            f'<rect class="hit" x="{x(i):.1f}" y="{top}" width="{plot_w / n:.1f}" '
            f'height="{plot_h}" tabindex="0" data-tip="{html.escape(tip)}"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _task_health(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for task in tasks:
        entry = grouped.setdefault(
            task["task"],
            {
                "task": task["task"],
                "stage": task["stage"],
                "runs": 0,
                "failed": 0,
                "total_ms": 0.0,
                "max_ms": 0.0,
                "rows": 0,
            },
        )
        entry["runs"] += 1
        entry["failed"] += int(task["status"] == "failed")
        entry["total_ms"] += task["duration_ms"]
        entry["max_ms"] = max(entry["max_ms"], task["duration_ms"])
        entry["rows"] += int(task.get("rows_in") or task.get("rows_out") or 0)
    rows = []
    for entry in grouped.values():
        entry["label"] = TASK_LABEL.get(entry["task"], entry["task"])
        entry["avg_ms"] = entry["total_ms"] / entry["runs"]
        entry["status"] = "critical" if entry["failed"] else "ok"
        rows.append(entry)
    return rows


def build_context(summary: dict[str, Any]) -> dict[str, Any]:
    batches = summary["batches"]
    labels = [_month_label(b["month"]) for b in batches]
    ref = summary["reference"]["metrics"]
    last = batches[-1]
    alerts = summary["alerts"]
    n_critical = sum(a["severity"] == "critical" for a in alerts)

    columns = [row["column"] for row in sorted(last["drift"]["table"], key=lambda r: -r["psi"])]
    heat_rows = []
    for column in columns:
        cells = []
        for batch in batches:
            row = next(r for r in batch["drift"]["table"] if r["column"] == column)
            cells.append(
                {
                    "psi": _fmt(row["psi"]),
                    "status": row["severity"],
                    "icon": STATUS_ICON[row["severity"]],
                    "tip": (
                        f"{config.FEATURE_LABELS.get(column, column)} em "
                        f"{_month_label(batch['month'])}: PSI {_fmt(row['psi'])} "
                        f"({STATUS_LABEL[row['severity']]}) · KS D {_fmt(row['ks_statistic'])} · "
                        f"média {row['mean_shift_pct']:+.1f}%".replace(".", ",")
                    ),
                }
            )
        heat_rows.append(
            {
                "label": config.FEATURE_LABELS.get(column, column),
                "column": column,
                "is_score": column == config.SCORE_COL,
                "cells": cells,
            }
        )

    overall = "critical" if n_critical else ("warning" if alerts else "ok")
    blocked = sum(b["contract"]["blocked_deliveries"] for b in batches)
    tasks = summary["pipeline"]["tasks"]
    failed_tasks = sum(t["status"] == "failed" for t in tasks)
    max_latency = max(b["latency_ms_per_1k"] for b in batches)
    thresholds = summary["thresholds"]

    charts = {
        "auc": line_chart(
            labels,
            [b["performance"]["auc"] for b in batches],
            reference=ref["auc"],
            kind="num",
            name="AUC",
        ),
        "approval": line_chart(
            labels,
            [b["performance"]["approval_rate"] for b in batches],
            reference=ref["approval_rate"],
            kind="pct",
            name="Taxa de aprovação",
        ),
        "bad_rate": line_chart(
            labels,
            [b["performance"]["bad_rate_approved"] for b in batches],
            reference=ref["bad_rate_approved"],
            kind="pct",
            name="Inadimplência nos aprovados",
        ),
        "psi_score": line_chart(
            labels,
            [
                next(r["psi"] for r in b["drift"]["table"] if r["column"] == "score")
                for b in batches
            ],
            reference=thresholds["psi_critical"],
            kind="num",
            name="PSI do score",
            y_min=0,
        ),
    }
    histograms = [
        {
            "label": config.FEATURE_LABELS.get(column, column),
            "svg": histogram_chart(hist, config.FEATURE_LABELS.get(column, column)),
        }
        for column, hist in last["drift"]["histograms"].items()
    ]

    ingestion = summary["ingestion"]
    max_failures = max(ingestion["failures_by_rule"].values())
    ingestion_rules = [
        {"rule": rule, "count": _fmt(count, "int"), "width": max(1.0, 100 * count / max_failures)}
        for rule, count in sorted(ingestion["failures_by_rule"].items(), key=lambda kv: -kv[1])
    ]
    blocked_deliveries = [
        {"month": _month_label(b["month"]), **d}
        for b in batches
        for d in b["contract"]["deliveries"]
        if d["status"] == "blocked"
    ]

    slo = [
        {
            "name": "Latência de inferência por 1.000 linhas",
            "target": f"≤ {_fmt(thresholds['slo_latency_ms_per_1k'], 'num2')} ms",
            "actual": f"pior lote: {_fmt(max_latency, 'num2')} ms",
            "status": "ok" if max_latency <= thresholds["slo_latency_ms_per_1k"] else "critical",
        },
        {
            "name": "Volume mínimo por lote",
            "target": f"≥ {_fmt(thresholds['slo_min_rows'], 'int')} linhas",
            "actual": f"menor lote: {_fmt(min(b['rows'] for b in batches), 'int')} linhas",
            "status": "ok"
            if min(b["rows"] for b in batches) >= thresholds["slo_min_rows"]
            else "critical",
        },
        {
            "name": "Lotes entregues ao modelo",
            "target": f"{len(batches)} de {len(batches)}",
            "actual": f"{sum(b['contract']['status'] == 'passed' for b in batches)} de "
            f"{len(batches)}",
            "status": "ok",
        },
        {
            "name": "Dado inválido que chegou ao modelo",
            "target": "0 lotes",
            "actual": f"0 lotes ({blocked} entrega(s) bloqueada(s) antes da inferência)",
            "status": "ok",
        },
    ]

    return {
        "summary": summary,
        "fmt": _fmt,
        "month_label": _month_label,
        "labels": labels,
        "batches": batches,
        "last": last,
        "ref": ref,
        "overall": overall,
        "status_label": STATUS_LABEL,
        "status_icon": STATUS_ICON,
        "diagnosis_label": DIAGNOSIS_LABEL,
        "alert_kind_label": ALERT_KIND_LABEL,
        "feature_labels": config.FEATURE_LABELS,
        "n_alerts": len(alerts),
        "n_critical": n_critical,
        "n_warning": len(alerts) - n_critical,
        "blocked": blocked,
        "failed_tasks": failed_tasks,
        "n_tasks": len(tasks),
        "heat_rows": heat_rows,
        "charts": charts,
        "histograms": histograms,
        "ingestion_rules": ingestion_rules,
        "blocked_deliveries": blocked_deliveries,
        "task_health": _task_health(tasks),
        "slo": slo,
        "thresholds": thresholds,
    }


def build(summary: dict[str, Any], output: Path) -> Path:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    rendered = env.get_template("dashboard.html.j2").render(**build_context(summary))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    return output


if __name__ == "__main__":
    data = json.loads((config.REPORTS_DIR / "monitoring_summary.json").read_text(encoding="utf-8"))
    print(build(data, config.REPORTS_DIR / "dashboard.html"))
