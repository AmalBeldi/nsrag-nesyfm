"""CLI: python -m nsrag.scripts.results_to_paper [--results-dir results/]

Reads every results/*.json produced by the other CLIs and prints the
corresponding LaTeX table rows, ready to paste into main.tex's
tabular blocks (Section 5.10/5.12/5.13/5.14/5.17). Prints "(missing --
run <script>)" for any table whose results file isn't present yet,
instead of guessing or leaving a silent gap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    d = Path(args.results_dir)

    print("=" * 70)
    print("Table 5 -- Clinical QA results")
    print("=" * 70)
    table5 = _load(d / "table5_qa.json")
    if table5 is None:
        print("(missing -- run scripts/run_all.sh step 1, or run_qa_cli.py)")
    else:
        for name, r in table5.items():
            if name == "_efficiency":
                continue
            f = lambda x: "--" if x is None else f"{x:.2f}"
            print(f"{name} & {f(r['fact_f1'])} & {f(r['cr'])} & {f(r['ur'])} & {f(r['ec'])} "
                  f"& -- & -- & -- & -- \\\\")

    print()
    print("=" * 70)
    print("Table 6 -- Clinical Summarization results")
    print("=" * 70)
    table6 = _load(d / "table6_summarization.json")
    if table6 is None:
        print("(missing -- run scripts/run_all.sh step 2, or run_summarization_cli.py)")
    else:
        for name, r in table6.items():
            f = lambda x: "--" if x is None else f"{x:.2f}"
            print(f"{name} & {f(r['fact_f1'])} & {f(r['cr'])} & {f(r['ur'])} & {f(r['ec'])} "
                  f"& {f(r['rouge_l'])} & {f(r['bertscore_f1'])} \\\\")

    print()
    print("=" * 70)
    print("Table 7 -- i2b2 component-level evaluation")
    print("=" * 70)
    table7 = _load(d / "table7_i2b2.json")
    if table7 is None:
        print("(missing -- run scripts/run_all.sh step 3, or run_i2b2_cli.py)")
    else:
        for name, r in table7.items():
            print(f"{name} & {r['precision']:.2f} & {r['recall']:.2f} & {r['f1']:.2f} \\\\")

    print()
    print("=" * 70)
    print("Table 8 -- Ablation study")
    print("=" * 70)
    table8 = _load(d / "table8_ablation.json")
    if table8 is None:
        print("(missing -- run scripts/run_all.sh step 4, or run_ablation_cli.py)")
    else:
        for name, r in table8.items():
            if name == "_efficiency":
                continue
            f = lambda x: "--" if x is None else f"{x:.2f}"
            print(f"{name} & {f(r['fact_f1'])} & {f(r['ec'])} & {f(r['cr'])} \\\\")

    print()
    print("=" * 70)
    print("Table 9 -- Efficiency")
    print("=" * 70)
    table9 = _load(d / "table9_efficiency.json")
    if table9 is None:
        print("(missing -- run scripts/run_all.sh step 6, or run_efficiency_cli.py)")
    else:
        total = table9.get("_total_s")
        for stage, stats in table9.items():
            if stage == "_total_s":
                continue
            print(f"{stage} & {stats['time_s']:.1f} & {stats['pct_total']:.1f} \\\\")
        if total is not None:
            print(f"\\midrule\nTotal & {total:.1f} & -- \\\\")


if __name__ == "__main__":
    main()
