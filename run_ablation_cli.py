"""CLI: python -m nsrag.scripts.run_ablation_cli --instances path.jsonl --out results/table8_ablation.json"""
from __future__ import annotations

from nsrag.pipeline.run_ablation import run_ablation_experiment
from nsrag.scripts.common import base_arg_parser, build_generator, load_config, load_qa_instances, write_json


def main():
    parser = base_arg_parser("Run the ablation study (Table 8 / Figure 6).")
    args = parser.parse_args()
    if not args.instances:
        parser.error("--instances is required (see README.md 'Preparing instances')")

    config = load_config(args.config)
    generator = build_generator(config, args.mock)
    instances = load_qa_instances(args.instances)

    results = run_ablation_experiment(generator, instances)
    serializable = {}
    for name, r in results.items():
        if name == "_efficiency":
            serializable[name] = r
            continue
        serializable[name] = {
            "fact_f1": r.fact_f1, "cr": r.cr, "ur": r.ur, "ec": r.ec,
        }
    write_json(serializable, args.out)


if __name__ == "__main__":
    main()
