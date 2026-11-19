"""CLI: python -m nsrag.scripts.run_qa_cli --instances path.jsonl --out results/table5_qa.json"""
from __future__ import annotations

from nsrag.baselines.graph_linearization import GraphLinearization
from nsrag.baselines.llm_only import LLMOnly
from nsrag.baselines.ns_rag import NSRAG, NSRAGWithoutVerification
from nsrag.baselines.textual_rag import TextualRAG
from nsrag.pipeline.run_qa import run_qa_experiment
from nsrag.scripts.common import base_arg_parser, build_generator, load_config, load_qa_instances, write_json


def main():
    parser = base_arg_parser("Run the Clinical QA experiment (Table 5).")
    args = parser.parse_args()
    if not args.instances:
        parser.error("--instances is required (see README.md 'Preparing instances')")

    config = load_config(args.config)
    generator = build_generator(config, args.mock)
    instances = load_qa_instances(args.instances)

    methods = [
        LLMOnly(generator),
        TextualRAG(generator),
        GraphLinearization(generator),
        NSRAGWithoutVerification(generator),
        NSRAG(generator),
        # B4/B5 (GraphRAG/LightRAG) and B6 (External KG+LLM) need extra
        # setup (a pre-built index / a KG client) -- add them here once
        # configured, see nsrag/baselines/graph_rag_family.py and
        # nsrag/baselines/external_kg_llm.py.
    ]

    results = run_qa_experiment(methods, instances)
    serializable = {
        name: (r if name == "_efficiency" else {
            "fact_precision": r.fact_precision, "fact_recall": r.fact_recall,
            "fact_f1": r.fact_f1, "cr": r.cr, "ur": r.ur, "ec": r.ec,
            "n_instances": r.n_instances,
        })
        for name, r in results.items()
    }
    write_json(serializable, args.out)


if __name__ == "__main__":
    main()
