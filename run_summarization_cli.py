"""CLI: python -m nsrag.scripts.run_summarization_cli --instances path.jsonl --out results/table6_summarization.json"""
from __future__ import annotations

from nsrag.baselines.graph_linearization import GraphLinearization
from nsrag.baselines.llm_only import LLMOnly
from nsrag.baselines.ns_rag import NSRAG, NSRAGWithoutVerification
from nsrag.baselines.textual_rag import TextualRAG
from nsrag.pipeline.run_summarization import run_summarization_experiment
from nsrag.scripts.common import (
    base_arg_parser, build_generator, load_config, load_summarization_instances, write_json,
)


def main():
    parser = base_arg_parser("Run the Clinical Summarization experiment (Table 6).")
    args = parser.parse_args()
    if not args.instances:
        parser.error("--instances is required (see README.md 'Preparing instances')")

    config = load_config(args.config)
    generator = build_generator(config, args.mock)
    instances = load_summarization_instances(args.instances)

    methods = [
        LLMOnly(generator), TextualRAG(generator), GraphLinearization(generator),
        NSRAGWithoutVerification(generator), NSRAG(generator),
    ]

    results = run_summarization_experiment(methods, instances)
    serializable = {
        name: {
            "fact_f1": r.fact_f1, "cr": r.cr, "ur": r.ur, "ec": r.ec,
            "rouge_l": r.rouge_l, "bertscore_f1": r.bertscore_f1,
        }
        for name, r in results.items()
    }
    write_json(serializable, args.out)


if __name__ == "__main__":
    main()
