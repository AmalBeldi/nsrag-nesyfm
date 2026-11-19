"""CLI: python -m nsrag.scripts.run_efficiency_cli --instances path.jsonl --out results/table9_efficiency.json"""
from __future__ import annotations

from nsrag.pipeline.run_efficiency import run_efficiency_experiment
from nsrag.scripts.common import base_arg_parser, build_generator, load_config, load_qa_instances, write_json


def main():
    parser = base_arg_parser("Run stage-level efficiency measurement (Table 9).")
    args = parser.parse_args()
    if not args.instances:
        parser.error("--instances is required (see README.md 'Preparing instances')")

    config = load_config(args.config)
    generator = build_generator(config, args.mock)
    instances = load_qa_instances(args.instances)

    summary = run_efficiency_experiment(generator, instances)
    write_json(summary, args.out)


if __name__ == "__main__":
    main()
