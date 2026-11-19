#!/usr/bin/env bash
# Runs the full experimental protocol described in Section 5 of the
# paper and writes one JSON file per table under results/.
#
# Prerequisites (see README.md):
#   1. pip install -r requirements.txt
#   2. Credentialed MIMIC-III + i2b2 2010 data, paths set in
#      configs/default.yaml (data.mimic_dir / data.i2b2_dir)
#   3. A CUDA GPU for HFGenerator at a reasonable speed (Section 5.16
#      reports numbers on an A100 40GB; a smaller GPU or CPU will run
#      but far more slowly)
#
# Each step is independent -- comment out any you don't need, or run
# `python -m nsrag.pipeline.<module>` directly for finer control.

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p results

echo "[1/6] Clinical QA (Table 5, RQ1/RQ2/RQ4)"
python -m nsrag.scripts.run_qa_cli        --config configs/default.yaml --out results/table5_qa.json

echo "[2/6] Clinical Summarization (Table 6)"
python -m nsrag.scripts.run_summarization_cli --config configs/default.yaml --out results/table6_summarization.json

echo "[3/6] i2b2 component evaluation (Table 7, RQ6)"
python -m nsrag.scripts.run_i2b2_cli      --config configs/default.yaml --out results/table7_i2b2.json

echo "[4/6] Ablation study (Table 8 / Figure 6, RQ5)"
python -m nsrag.scripts.run_ablation_cli  --config configs/default.yaml --out results/table8_ablation.json

echo "[5/6] Context-length robustness (Figure 5)"
python -m nsrag.scripts.run_robustness_cli --config configs/default.yaml --out results/figure5_robustness.json

echo "[6/6] Efficiency (Table 9)"
python -m nsrag.scripts.run_efficiency_cli --config configs/default.yaml --out results/table9_efficiency.json

echo "Done. Run 'python -m nsrag.scripts.results_to_paper' to render LaTeX rows from results/*.json."
