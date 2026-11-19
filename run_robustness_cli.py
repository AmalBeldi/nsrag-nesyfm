"""CLI: python -m nsrag.scripts.run_robustness_cli --instances path.jsonl --out results/figure5_robustness.json"""
from __future__ import annotations

from dataclasses import asdict

from nsrag.pipeline.run_context_robustness import run_context_robustness_experiment
from nsrag.scripts.common import base_arg_parser, build_generator, load_config, load_qa_instances, write_json


def main():
    parser = base_arg_parser("Run the context-length robustness sweep (Figure 5).")
    args = parser.parse_args()
    if not args.instances:
        parser.error("--instances is required (see README.md 'Preparing instances')")

    config = load_config(args.config)
    generator = build_generator(config, args.mock)
    instances = load_qa_instances(args.instances)
    budgets = config.get("context_robustness_budgets", [128, 256, 512, 1024, 2048])

    points = run_context_robustness_experiment(generator, instances, budgets=budgets)
    write_json([asdict(p) for p in points], args.out)


if __name__ == "__main__":
    main()
