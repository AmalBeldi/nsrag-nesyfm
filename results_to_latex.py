"""Regenerates the exact LaTeX table rows used in main.tex from
computed results, so that once you have run real experiments the paper
tables are a copy-paste (or diff) away instead of manual transcription
-- eliminating a whole class of "typo between code and paper" errors.

Each function's output is designed to be pasted directly into the
corresponding `tabular` block in main.tex (Section 5.10/5.12/5.13/5.14).
"""
from __future__ import annotations

from nsrag.pipeline.run_qa import MethodResult

_DASH = "--"


def _fmt(x, pattern: str = "{:.2f}") -> str:
    return _DASH if x is None else pattern.format(x)


def qa_results_to_latex(results: dict[str, MethodResult], row_order: list[str]) -> str:
    """Table 5 rows: Method & Fact F1 & CR & UR & EC & Verif. F1 & MR & ROUGE-L & BERTScore \\\\"""
    lines = []
    for label in row_order:
        r = results.get(label)
        if r is None:
            lines.append(f"{label} & {_DASH} & {_DASH} & {_DASH} & {_DASH} & {_DASH} & {_DASH} & {_DASH} & {_DASH} \\\\")
            continue
        lines.append(
            f"{label} & {_fmt(r.fact_f1)} & {_fmt(r.cr)} & {_fmt(r.ur)} & {_fmt(r.ec)} "
            f"& {_DASH} & {_DASH} & {_DASH} & {_DASH} \\\\"
        )
    return "\n".join(lines)


def ablation_results_to_latex(results: dict, row_order: list[str]) -> str:
    """Table 8 rows: Configuration & Fact F1 & EC & HR \\\\"""
    lines = []
    for label in row_order:
        r = results.get(label)
        if r is None:
            lines.append(f"{label} & {_DASH} & {_DASH} & {_DASH} \\\\")
            continue
        lines.append(f"{label} & {_fmt(r.fact_f1)} & {_fmt(r.ec)} & {_fmt(r.cr)} \\\\")
    return "\n".join(lines)


def summarization_results_to_latex(results: dict, row_order: list[str]) -> str:
    """Table 6 rows: Method & Fact F1 & CR & UR & EC & ROUGE-L & BERTScore \\\\"""
    lines = []
    for label in row_order:
        r = results.get(label)
        if r is None:
            lines.append(f"{label} & {_DASH} & {_DASH} & {_DASH} & {_DASH} & {_DASH} & {_DASH} \\\\")
            continue
        lines.append(
            f"{label} & {_fmt(r.fact_f1)} & {_fmt(r.cr)} & {_fmt(r.ur)} & {_fmt(r.ec)} "
            f"& {_fmt(r.rouge_l)} & {_fmt(r.bertscore_f1)} \\\\"
        )
    return "\n".join(lines)


def i2b2_results_to_latex(results: dict, row_order: list[str]) -> str:
    """Table 7 rows: Component & Precision & Recall & F1 \\\\"""
    lines = []
    for label in row_order:
        r = results.get(label)
        if r is None:
            lines.append(f"{label} & {_DASH} & {_DASH} & {_DASH} \\\\")
            continue
        lines.append(f"{label} & {_fmt(r.precision)} & {_fmt(r.recall)} & {_fmt(r.f1)} \\\\")
    return "\n".join(lines)


def efficiency_results_to_latex(summary: dict) -> str:
    """Table 9 rows: Component & Time (s) & Share (%) \\\\"""
    lines = []
    total = summary.get("_total_s")
    for stage, stats in summary.items():
        if stage == "_total_s":
            continue
        lines.append(f"{stage} & {stats['time_s']:.1f} & {stats['pct_total']:.1f} \\\\")
    if total is not None:
        lines.append(f"\\midrule\nTotal & {total:.1f} & {_DASH} \\\\")
    return "\n".join(lines)


def graph_stats_to_latex(stats: dict) -> str:
    """Table 3 rows: Representation & Before & After & Reduction \\\\"""
    def row(label, before, after, reduction):
        return f"{label} & {before:,} & {after:,} & {reduction:.1f}\\% \\\\"

    return "\n".join([
        row("Nodes", stats["nodes_before"], stats["nodes_after"], stats["nodes_reduction_pct"]),
        row("Edges", stats["edges_before"], stats["edges_after"], stats["edges_reduction_pct"]),
    ])
